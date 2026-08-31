"""
SPIRO ML — YOLODetectionEngine
Production ORT-based YOLOv11 detector with:
  - Auto provider selection (TensorRT → CUDA → CPU)
  - Pure-NumPy NMS
  - Bounding box rescaling back to original image coordinates
  - Label decoding against SPIRO taxonomy
  - Batch detection support
  - Graph-optimised ORT session
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from lib.ml.core.logger import get_logger
from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader

log = get_logger(__name__)


@dataclass
class Detection:
    """A single raw detection from YOLOv11."""
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    probabilities: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox_xyxy": [round(v, 2) for v in self.bbox_xyxy],
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


class YOLODetectionEngine:
    """
    ORT-based YOLOv11 detection engine for SPIRO.

    Parameters
    ----------
    onnx_path : str | Path
    providers : list[str], optional — auto-detected if None
    input_size : int — must match export size
    conf_threshold : float
    iou_threshold : float
    max_detections : int
    warmup_runs : int
    num_threads : int — ORT intra-op threads

    Example
    -------
    >>> engine = YOLODetectionEngine("models/exports/spiro_yolov11s_best.onnx")
    >>> detections = engine.detect(preprocess_result)
    """

    def __init__(
        self,
        onnx_path: str | Path,
        providers: Optional[List[str]] = None,
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        min_bbox_area: float = 0.001,
        warmup_runs: int = 3,
        num_threads: int = 4,
        graph_optimization: str = "ORT_ENABLE_ALL",
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.min_bbox_area = min_bbox_area

        self.taxonomy = TaxonomyLoader()

        # Build ORT session
        self.providers = providers or self._auto_providers()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = getattr(
            ort.GraphOptimizationLevel,
            graph_optimization,
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        )
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True

        log.info(f"Loading YOLO ONNX: {self.onnx_path}")
        log.info(f"Providers: {self.providers}")
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=opts,
            providers=self.providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self._active_providers = self.session.get_providers()
        log.info(f"Active providers: {self._active_providers}")

        if warmup_runs > 0:
            self._warmup(warmup_runs)

    # ------------------------------------------------------------------
    # Detect
    # ------------------------------------------------------------------

    def detect(
        self,
        blob: np.ndarray,
        original_shape: Tuple[int, int],
        scale: float,
        pad_w: int,
        pad_h: int,
    ) -> Tuple[List[Detection], float]:
        """
        Run detection on a pre-processed blob.

        Parameters
        ----------
        blob : ndarray [1, 3, H, W] float32
        original_shape : (orig_H, orig_W)
        scale : letterbox scale
        pad_w, pad_h : letterbox padding

        Returns
        -------
        (detections, latency_ms)
        """
        t0 = time.perf_counter()
        raw_outputs = self.session.run(self.output_names, {self.input_name: blob})
        latency_ms = (time.perf_counter() - t0) * 1000

        detections = self._postprocess(
            raw_outputs[0], original_shape, scale, pad_w, pad_h
        )
        return detections, latency_ms

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        raw: np.ndarray,
        orig_shape: Tuple[int, int],
        scale: float,
        pad_w: int,
        pad_h: int,
    ) -> List[Detection]:
        """
        Decode YOLOv11 ONNX output into Detection objects.

        YOLOv11 exports in [1, 4+nc, N] format (cx, cy, w, h, cls_probs...).
        """
        orig_h, orig_w = orig_shape

        # Squeeze batch dim
        if raw.ndim == 3:
            raw = raw[0]  # [4+nc, N] or [N, 4+nc]

        # Ensure [N, 4+nc]
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T

        nc = self.taxonomy.num_classes
        boxes_xywh = raw[:, :4]   # cx, cy, w, h in input_size coords
        cls_scores = raw[:, 4:]   # [N, nc] or [N, 1+nc]

        # Handle objectness × cls format (nc+1 columns)
        if cls_scores.shape[1] == nc + 1:
            obj = cls_scores[:, 0:1]
            cls_scores = cls_scores[:, 1:] * obj

        # Get best class and score per anchor
        cls_ids = np.argmax(cls_scores, axis=1)
        confs = cls_scores[np.arange(len(cls_ids)), cls_ids]

        # Filter by confidence
        mask = confs >= self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confs = confs[mask]
        cls_ids = cls_ids[mask]

        if len(confs) == 0:
            return []

        # cx,cy,w,h → x1,y1,x2,y2 in letterboxed input space
        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        # Remove padding and rescale to original coords
        x1 = np.clip((x1 - pad_w) / scale, 0, orig_w)
        y1 = np.clip((y1 - pad_h) / scale, 0, orig_h)
        x2 = np.clip((x2 - pad_w) / scale, 0, orig_w)
        y2 = np.clip((y2 - pad_h) / scale, 0, orig_h)

        # Filter tiny boxes
        img_area = orig_w * orig_h
        box_areas = (x2 - x1) * (y2 - y1)
        area_mask = (box_areas / max(img_area, 1)) >= self.min_bbox_area
        x1, y1, x2, y2 = x1[area_mask], y1[area_mask], x2[area_mask], y2[area_mask]
        confs, cls_ids = confs[area_mask], cls_ids[area_mask]

        if len(confs) == 0:
            return []

        # NMS
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        keep = self._nms(boxes_xyxy, confs, self.iou_threshold)
        keep = keep[: self.max_detections]

        detections: List[Detection] = []
        for i in keep:
            cls_id = int(cls_ids[i])
            if cls_id >= self.taxonomy.num_classes:
                continue
            detections.append(Detection(
                bbox_xyxy=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                confidence=float(confs[i]),
                class_id=cls_id,
                class_name=self.taxonomy.id_to_name(cls_id),
            ))

        return detections

    # ------------------------------------------------------------------
    # NMS
    # ------------------------------------------------------------------

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
        """Pure NumPy NMS. Returns sorted keep indices."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= iou_thr]
        return np.array(keep, dtype=np.int64)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_providers() -> List[str]:
        available = ort.get_available_providers()
        priority = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        return [p for p in priority if p in available] or ["CPUExecutionProvider"]

    def _warmup(self, n: int) -> None:
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        for _ in range(n):
            self.session.run(self.output_names, {self.input_name: dummy})
        log.info(f"YOLO warmup complete ({n} runs)")

    def benchmark(self, n_runs: int = 100) -> Dict[str, float]:
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.session.run(self.output_names, {self.input_name: dummy})
            times.append((time.perf_counter() - t0) * 1000)
        arr = np.array(times)
        return {
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "min_ms": float(arr.min()),
            "p95_ms": float(np.percentile(arr, 95)),
            "fps": float(1000 / arr.mean()),
        }

    @property
    def active_providers(self) -> List[str]:
        return self._active_providers
