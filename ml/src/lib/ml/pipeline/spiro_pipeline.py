"""
SPIRO ML — SPIROPipeline
Production end-to-end inference pipeline orchestrator.

Pipeline stages
---------------
1.  Image Validation + Preprocessing
2.  YOLOv11 Detection (ONNX Runtime)
3.  NMS + Bounding Box Decoding
4.  Object Cropping
5.  EfficientNetV2 Verification (ONNX Runtime, batched)
6.  Confidence Fusion (5 methods)
7.  Waste Category Assignment
8.  Contamination Analysis
9.  Guidance Generation
10. Explainability (reasoning chain + optional Grad-CAM)
11. Structured JSON Response

Usage
-----
>>> pipeline = SPIROPipeline.from_config("configs/pipeline/inference_pipeline.yaml")
>>> result = pipeline.infer("photo.jpg")
>>> print(result["detections"])
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from lib.ml.core.logger import get_logger
from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader
from lib.ml.pipeline.pipeline_config import PipelineConfig
from lib.ml.pipeline.preprocessing.preprocessor import ImagePreprocessor
from lib.ml.pipeline.detection.yolo_engine import YOLODetectionEngine, Detection
from lib.ml.pipeline.cropping.cropper import ObjectCropper, Crop
from lib.ml.pipeline.contamination.contamination_engine import ContaminationEngine
from lib.ml.pipeline.guidance.guidance_engine import GuidanceEngine, _STREAM_LOOKUP
from lib.ml.pipeline.explainability.explainer import PipelineExplainer
from lib.ml.pipeline.output.output_builder import OutputBuilder, PipelineTimings
from lib.ml.verification.fusion.confidence_fusion import ConfidenceFusion

log = get_logger(__name__)


class SPIROPipeline:
    """
    Full SPIRO inference pipeline.

    Parameters
    ----------
    cfg : PipelineConfig
    yolo_engine : YOLODetectionEngine
    effnet_session : ort.InferenceSession (EfficientNetV2 ONNX)
    preprocessor : ImagePreprocessor
    cropper : ObjectCropper
    fusion : ConfidenceFusion
    contamination : ContaminationEngine
    guidance : GuidanceEngine
    explainer : PipelineExplainer
    output_builder : OutputBuilder

    Preferred construction: use `SPIROPipeline.from_config()`
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        yolo_engine: YOLODetectionEngine,
        effnet_session: Any,                 # ort.InferenceSession
        preprocessor: ImagePreprocessor,
        cropper: ObjectCropper,
        fusion: ConfidenceFusion,
        contamination: ContaminationEngine,
        guidance: GuidanceEngine,
        explainer: PipelineExplainer,
        output_builder: OutputBuilder,
    ) -> None:
        self.cfg = cfg
        self.yolo = yolo_engine
        self.effnet = effnet_session
        self.preprocessor = preprocessor
        self.cropper = cropper
        self.fusion = fusion
        self.contamination = contamination
        self.guidance = guidance
        self.explainer = explainer
        self.output = output_builder
        self.taxonomy = TaxonomyLoader()

        # Resolve EfficientNet input details
        self._effnet_input_name = effnet_session.get_inputs()[0].name
        self._effnet_output_names = [o.name for o in effnet_session.get_outputs()]
        self._effnet_input_size = cfg.models.effnet_input_size

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path] = "configs/pipeline/inference_pipeline.yaml",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "SPIROPipeline":
        """
        Build the complete pipeline from a YAML config.

        Parameters
        ----------
        config_path : str | Path
        overrides : dict of dot-notation config overrides

        Returns
        -------
        SPIROPipeline
        """
        import onnxruntime as ort

        cfg = PipelineConfig.load(config_path, overrides=overrides)
        log.info(f"Building SPIROPipeline from {config_path}")

        # Provider selection
        available = ort.get_available_providers()
        providers = [p for p in list(cfg.runtime.providers) if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        log.info(f"ORT providers (available): {providers}")

        # ── YOLO engine ───────────────────────────────────────────────
        yolo_path = Path(cfg.models.yolo_onnx)
        if not yolo_path.exists():
            raise FileNotFoundError(
                f"YOLO ONNX not found: {yolo_path}. "
                "Train and export the YOLOv11 model first."
            )
        yolo_engine = YOLODetectionEngine(
            onnx_path=yolo_path,
            providers=providers,
            input_size=cfg.models.yolo_input_size,
            conf_threshold=cfg.detection.conf_threshold,
            iou_threshold=cfg.detection.iou_threshold,
            max_detections=cfg.detection.max_detections,
            min_bbox_area=cfg.detection.min_bbox_area,
            warmup_runs=cfg.runtime.warmup_runs,
            num_threads=cfg.runtime.yolo_threads,
        )

        # ── EfficientNetV2 ORT session ────────────────────────────────
        effnet_path = Path(cfg.models.effnet_onnx)
        if not effnet_path.exists():
            raise FileNotFoundError(
                f"EfficientNetV2 ONNX not found: {effnet_path}. "
                "Train and export the EffNet verifier first."
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = cfg.runtime.effnet_threads
        opts.graph_optimization_level = getattr(
            ort.GraphOptimizationLevel,
            cfg.runtime.graph_optimization,
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        )
        effnet_session = ort.InferenceSession(
            str(effnet_path), sess_options=opts, providers=providers
        )
        log.info(f"EfficientNetV2 loaded: {effnet_path.name}")

        # Warmup EffNet
        if cfg.runtime.warmup_runs > 0:
            sz = cfg.models.effnet_input_size
            dummy = np.zeros((1, 3, sz, sz), dtype=np.float32)
            inp_name = effnet_session.get_inputs()[0].name
            for _ in range(cfg.runtime.warmup_runs):
                effnet_session.run(None, {inp_name: dummy})
            log.info(f"EfficientNetV2 warmup complete ({cfg.runtime.warmup_runs} runs)")

        # ── Sub-components ────────────────────────────────────────────
        pp_cfg = cfg.preprocessing
        preprocessor = ImagePreprocessor(
            target_size=pp_cfg.target_size,
            blur_threshold=pp_cfg.blur_threshold,
            brightness_min=pp_cfg.brightness_min,
            brightness_max=pp_cfg.brightness_max,
            quality_threshold=pp_cfg.quality_score_threshold,
            max_file_size_mb=pp_cfg.max_file_size_mb,
            auto_orient=pp_cfg.auto_orient,
        )

        cropper = ObjectCropper(
            padding_fraction=cfg.cropping.padding_fraction,
            min_crop_pixels=cfg.cropping.min_crop_pixels,
        )

        fusion_cfg = cfg.fusion
        fusion = ConfidenceFusion(
            method=fusion_cfg.method,
            yolo_weight=fusion_cfg.yolo_weight,
            effnet_weight=fusion_cfg.effnet_weight,
            temperature=1.0,
            min_confidence=fusion_cfg.min_final_confidence,
            num_classes=cfg.models.num_classes,
        )

        contamination = ContaminationEngine()
        guidance = GuidanceEngine()

        exp_cfg = cfg.explainability
        explainer = PipelineExplainer(
            save_visualizations=exp_cfg.save_visualizations,
            visualization_dir=Path(exp_cfg.visualization_dir),
            include_reasoning_chain=exp_cfg.include_reasoning_chain,
        )

        output_builder = OutputBuilder(
            include_probabilities=cfg.output.include_probabilities,
            include_timings=cfg.output.include_timings,
            include_model_versions=cfg.output.include_model_versions,
            model_versions={
                "yolo": yolo_path.name,
                "effnet": effnet_path.name,
            },
        )

        log.info("SPIROPipeline ready ✓")
        return cls(
            cfg=cfg,
            yolo_engine=yolo_engine,
            effnet_session=effnet_session,
            preprocessor=preprocessor,
            cropper=cropper,
            fusion=fusion,
            contamination=contamination,
            guidance=guidance,
            explainer=explainer,
            output_builder=output_builder,
        )

    # ------------------------------------------------------------------
    # Main inference entry point
    # ------------------------------------------------------------------

    def infer(
        self,
        source: Union[str, Path, bytes, np.ndarray],
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete SPIRO inference pipeline on one image.

        Parameters
        ----------
        source : file path, raw bytes, or BGR ndarray
        request_id : optional trace ID for logging

        Returns
        -------
        dict — fully structured JSON response
        """
        t_pipeline_start = time.perf_counter()
        timings = PipelineTimings()
        warnings: List[str] = []
        rid = request_id or ""

        # ── Stage 1: Preprocessing ────────────────────────────────────
        t0 = time.perf_counter()
        try:
            prep = self.preprocessor.process(source)
        except Exception as e:
            log.error(f"[{rid}] Preprocessing crashed: {e}")
            return self.output.error_response(
                f"Preprocessing error: {e}", "PREPROCESS_ERROR"
            )
        timings.preprocess_ms = (time.perf_counter() - t0) * 1000

        if not prep.quality.passed:
            log.info(f"[{rid}] Image rejected: {prep.quality.rejection_reason}")
            return self.output.rejected_response(
                prep.quality.rejection_reason,
                prep.quality.to_dict(),
            )

        # ── Stage 2: YOLOv11 Detection ────────────────────────────────
        t0 = time.perf_counter()
        try:
            detections, det_latency = self.yolo.detect(
                blob=prep.blob,
                original_shape=prep.original_shape,
                scale=prep.scale,
                pad_w=prep.pad_w,
                pad_h=prep.pad_h,
            )
        except Exception as e:
            log.error(f"[{rid}] Detection failed: {e}")
            return self.output.error_response(f"Detection error: {e}", "DETECTION_ERROR")
        timings.detection_ms = (time.perf_counter() - t0) * 1000
        log.info(f"[{rid}] Detected {len(detections)} objects in {timings.detection_ms:.1f}ms")

        if not detections:
            warnings.append("No objects detected.")

        # ── Stage 3: Object Cropping ──────────────────────────────────
        t0 = time.perf_counter()
        crops: List[Crop] = []
        if detections:
            crops = self.cropper.crop(prep.image_bgr, detections)
        timings.cropping_ms = (time.perf_counter() - t0) * 1000

        # ── Stage 4: EfficientNetV2 Verification (batched) ────────────
        t0 = time.perf_counter()
        verifications: List[Optional[Dict[str, Any]]] = [None] * len(detections)
        if crops:
            verifications = self._run_effnet_batch(crops)
        timings.verification_ms = (time.perf_counter() - t0) * 1000

        # ── Stage 5: Confidence Fusion ────────────────────────────────
        t0 = time.perf_counter()
        fused_results: List[Dict[str, Any]] = []
        for det, verif in zip(detections, verifications):
            yolo_dict = det.to_dict()
            # Build full probability vector for YOLO
            nc = self.taxonomy.num_classes
            yolo_probs = np.zeros(nc)
            yolo_probs[det.class_id] = det.confidence
            yolo_dict["probabilities"] = yolo_probs.tolist()

            if verif is not None:
                fr = self.fusion.fuse(yolo_dict, verif)
                fd = fr.to_dict()
                # Enrich with stream
                fd["waste_stream"] = _STREAM_LOOKUP.get(fr.class_id, "reject_waste")
            else:
                fd = {
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "fused_confidence": det.confidence,
                    "confidence": det.confidence,
                    "method": "yolo_only",
                    "agreement": True,
                    "waste_stream": _STREAM_LOOKUP.get(det.class_id, "reject_waste"),
                }
            fused_results.append(fd)
        timings.fusion_ms = (time.perf_counter() - t0) * 1000

        # ── Stage 6: Contamination Analysis ──────────────────────────
        t0 = time.perf_counter()
        contamination_result = None
        if self.cfg.contamination.enabled:
            contamination_result = self.contamination.analyse(fused_results)
        timings.contamination_ms = (time.perf_counter() - t0) * 1000

        # ── Stage 7: Guidance Generation ─────────────────────────────
        t0 = time.perf_counter()
        guidance_items = []
        if self.cfg.guidance.enabled and fused_results:
            guidance_items = self.guidance.generate(
                fused_results,
                include_preparation=self.cfg.guidance.include_preparation_steps,
                include_warnings=self.cfg.guidance.include_warnings,
            )
        timings.guidance_ms = (time.perf_counter() - t0) * 1000

        # ── Stage 8: Explainability ───────────────────────────────────
        t0 = time.perf_counter()
        evidence = []
        annotated_path = None
        if self.cfg.explainability.enabled and fused_results:
            evidence = self.explainer.build_evidence(
                detections=detections,
                verifications=verifications,
                fused_results=fused_results,
                stream_lookup=_STREAM_LOOKUP,
            )
            if self.cfg.explainability.save_visualizations and detections:
                annotated = self.explainer.annotate_image(
                    prep.image_bgr, evidence, detections
                )
                annotated_path = self.explainer.save_annotated(annotated)

            if self.cfg.explainability.gradcam and crops:
                self.explainer.generate_gradcam(crops, evidence)
        timings.explainability_ms = (time.perf_counter() - t0) * 1000

        # ── Stage 9: Assemble Output ──────────────────────────────────
        timings.total_ms = (time.perf_counter() - t_pipeline_start) * 1000

        if timings.total_ms > self.cfg.performance.log_slow_threshold_ms:
            log.warning(
                f"[{rid}] Slow inference: {timings.total_ms:.0f}ms "
                f"(threshold {self.cfg.performance.log_slow_threshold_ms}ms)"
            )

        image_meta = {
            "original_size": {
                "width": prep.original_shape[1],
                "height": prep.original_shape[0],
            },
            "quality": prep.quality.to_dict(),
            **prep.metadata,
        }

        status = "success" if detections else "partial"
        response = self.output.build(
            status=status,
            image_meta=image_meta,
            detections=detections,
            fused_results=fused_results,
            guidance_items=guidance_items,
            contamination=contamination_result,
            evidence=evidence,
            timings=timings,
            warnings=warnings,
            annotated_image_path=annotated_path,
        )
        log.info(
            f"[{rid}] Pipeline complete: {len(detections)} detections "
            f"in {timings.total_ms:.1f}ms"
        )
        return response

    def infer_batch(
        self,
        sources: List[Union[str, Path, bytes, np.ndarray]],
    ) -> List[Dict[str, Any]]:
        """Run inference on a list of images sequentially."""
        return [self.infer(s) for s in sources]

    def benchmark(self, n_runs: int = 50) -> Dict[str, Any]:
        """
        Benchmark full pipeline + individual stages on a synthetic image.

        Returns
        -------
        dict with mean/std/min/p95/fps for each stage.
        """
        import time as _time

        sz = self.cfg.preprocessing.target_size
        dummy_img = np.full((sz, sz, 3), 128, dtype=np.uint8)

        stage_times: Dict[str, List[float]] = {
            "preprocess": [], "detect": [], "verify": [], "total": []
        }

        for _ in range(n_runs):
            t_total = _time.perf_counter()

            t0 = _time.perf_counter()
            prep = self.preprocessor.process(dummy_img)
            stage_times["preprocess"].append((_time.perf_counter() - t0) * 1000)

            t0 = _time.perf_counter()
            dets, _ = self.yolo.detect(prep.blob, prep.original_shape,
                                        prep.scale, prep.pad_w, prep.pad_h)
            stage_times["detect"].append((_time.perf_counter() - t0) * 1000)

            t0 = _time.perf_counter()
            crops = self.cropper.crop(dummy_img, dets) if dets else []
            if crops:
                self._run_effnet_batch(crops)
            stage_times["verify"].append((_time.perf_counter() - t0) * 1000)

            stage_times["total"].append((_time.perf_counter() - t_total) * 1000)

        def _stats(arr):
            a = np.array(arr)
            return {
                "mean_ms": round(float(a.mean()), 2),
                "std_ms": round(float(a.std()), 2),
                "min_ms": round(float(a.min()), 2),
                "p95_ms": round(float(np.percentile(a, 95)), 2),
                "fps": round(float(1000 / a.mean()), 1),
            }

        return {
            "n_runs": n_runs,
            "providers": self.yolo.active_providers,
            **{stage: _stats(times) for stage, times in stage_times.items()},
        }

    # ------------------------------------------------------------------
    # Internal: EfficientNetV2 batch inference
    # ------------------------------------------------------------------

    def _run_effnet_batch(
        self,
        crops: List[Crop],
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Run EfficientNetV2 on all valid crops in mini-batches.

        Returns a list parallel to crops; invalid crops get None.
        """
        nc = self.taxonomy.num_classes
        results: List[Optional[Dict[str, Any]]] = [None] * len(crops)
        batch_size = self.cfg.runtime.effnet_batch
        top_k = self.cfg.verification.top_k
        temp = self.cfg.verification.temperature

        # Collect valid crop indices
        valid_indices = [i for i, c in enumerate(crops) if c.is_valid]
        if not valid_indices:
            return results

        # Process in mini-batches
        for batch_start in range(0, len(valid_indices), batch_size):
            batch_idx = valid_indices[batch_start: batch_start + batch_size]
            batch_crops = [crops[i] for i in batch_idx]

            # Build NCHW blob
            blob = self.cropper.preprocess_batch(batch_crops, self._effnet_input_size)
            if blob.shape[0] == 0:
                continue

            # ORT inference
            logits_batch = self.effnet.run(
                self._effnet_output_names,
                {self._effnet_input_name: blob}
            )[0]   # [B, nc]

            for local_i, global_i in enumerate(batch_idx):
                if local_i >= logits_batch.shape[0]:
                    break
                logits = logits_batch[local_i]
                if temp != 1.0:
                    logits = logits / temp

                # Softmax
                e = np.exp(logits - logits.max())
                probs = e / e.sum()

                top_k_actual = min(top_k, nc)
                top_idx = np.argsort(probs)[::-1][:top_k_actual].tolist()
                class_names = self.taxonomy.all_names()

                results[global_i] = {
                    "class_id": int(top_idx[0]),
                    "class_name": class_names[top_idx[0]],
                    "confidence": float(probs[top_idx[0]]),
                    "top_k": [
                        {
                            "rank": r + 1,
                            "class_id": int(idx),
                            "class_name": class_names[idx],
                            "confidence": float(probs[idx]),
                        }
                        for r, idx in enumerate(top_idx)
                    ],
                    "probabilities": probs.tolist() if self.cfg.output.include_probabilities else [],
                }

        return results
