"""
SPIRO ML — ONNXInferenceEngine
Production inference using ONNXRuntime.
  - Auto-selects CUDA / CPU execution provider
  - Letterbox pre-processing matching training pipeline
  - Warmup runs for latency stabilisation
  - Batch inference support
  - Per-call timing metrics
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort

from lib.ml.core.logger import get_logger

log = get_logger(__name__)


class ONNXInferenceEngine:
    """
    Unified ONNX inference engine for SPIRO detection and classification.

    Parameters
    ----------
    onnx_path : str | Path
        Path to the .onnx model file.
    providers : list, optional
        ONNXRuntime execution providers. Auto-detected if None.
    input_size : tuple
        (width, height) — must match export-time size.
    conf_threshold : float
        Detection confidence threshold (detection models only).
    iou_threshold : float
        NMS IoU threshold (detection models only).
    warmup_runs : int
        Number of warmup inference calls after load.

    Example
    -------
    >>> engine = ONNXInferenceEngine("models/exports/yolo.onnx")
    >>> result = engine.infer("image.jpg")
    """

    def __init__(
        self,
        onnx_path: Union[str, Path],
        providers: Optional[List[str]] = None,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        warmup_runs: int = 3,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.input_size = input_size  # (W, H)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or []

        # Providers
        available = ort.get_available_providers()
        if providers is None:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available
                else ["CPUExecutionProvider"]
            )
        self.providers = providers

        log.info(f"Loading ONNX model: {self.onnx_path}")
        log.info(f"Providers: {self.providers}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            str(self.onnx_path), sess_options=opts, providers=self.providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Metadata
        self._meta = self._load_metadata()
        if not self.class_names and "class_names" in self._meta:
            self.class_names = json.loads(self._meta["class_names"])

        # Warmup
        if warmup_runs > 0:
            self._warmup(warmup_runs)

        log.info(
            f"Engine ready — input: {self.input_name}{self.input_shape} "
            f"| outputs: {self.output_names}"
        )

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def infer(
        self,
        source: Union[str, Path, np.ndarray],
        return_timing: bool = False,
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], float]]:
        """
        Run inference on a single image.

        Parameters
        ----------
        source : str | Path | np.ndarray
            Image path or BGR numpy array.
        return_timing : bool
            If True, also return elapsed ms.

        Returns
        -------
        dict with "detections" (list of dicts) or "probabilities" (list of floats).
        """
        img_bgr = self._load(source)
        blob, scale, padding = self._preprocess(img_bgr)

        t0 = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = self._postprocess(outputs, img_bgr.shape[:2], scale, padding)

        if return_timing:
            return result, elapsed_ms
        return result

    def infer_batch(
        self,
        sources: List[Union[str, Path, np.ndarray]],
    ) -> List[Dict[str, Any]]:
        """Run inference on a list of images (sequentially; for true batching export with batch>1)."""
        return [self.infer(s) for s in sources]

    def benchmark(
        self,
        n_runs: int = 100,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, float]:
        """
        Measure latency statistics over n_runs dummy inferences.

        Returns
        -------
        dict: mean_ms, std_ms, min_ms, max_ms, fps
        """
        shape = input_shape or (1, 3, *self.input_size[::-1])
        dummy = np.random.rand(*shape).astype(np.float32)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.session.run(self.output_names, {self.input_name: dummy})
            times.append((time.perf_counter() - t0) * 1000)
        times = np.array(times)
        stats = {
            "mean_ms": float(times.mean()),
            "std_ms": float(times.std()),
            "min_ms": float(times.min()),
            "max_ms": float(times.max()),
            "fps": float(1000 / times.mean()),
        }
        log.info(
            f"Benchmark ({n_runs} runs) — mean={stats['mean_ms']:.2f}ms "
            f"fps={stats['fps']:.1f}"
        )
        return stats

    # ------------------------------------------------------------------
    # Pre/post processing
    # ------------------------------------------------------------------

    def _preprocess(
        self, img_bgr: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Letterbox resize → BGR→RGB → HWC→NCHW → normalise [0,1].
        Returns (blob, scale, (pad_w, pad_h)).
        """
        w, h = self.input_size
        ih, iw = img_bgr.shape[:2]
        scale = min(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

        pad_w = (w - nw) // 2
        pad_h = (h - nh) // 2
        canvas = np.full((h, w, 3), 114, dtype=np.uint8)
        canvas[pad_h : pad_h + nh, pad_w : pad_w + nw] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis]  # NCHW
        return blob, scale, (pad_w, pad_h)

    def _postprocess(
        self,
        outputs: List[np.ndarray],
        orig_shape: Tuple[int, int],
        scale: float,
        padding: Tuple[int, int],
    ) -> Dict[str, Any]:
        """
        Decode raw ONNX outputs.
        Handles YOLO (shape [1, 5+nc, anchors]) and classifier (shape [1, nc]).
        """
        raw = outputs[0]

        # Classification branch — single vector
        if raw.ndim == 2 and raw.shape[0] == 1 and raw.shape[1] == len(self.class_names):
            probs = _softmax(raw[0])
            cls_id = int(np.argmax(probs))
            return {
                "type": "classification",
                "class_id": cls_id,
                "class_name": self.class_names[cls_id] if self.class_names else str(cls_id),
                "confidence": float(probs[cls_id]),
                "probabilities": probs.tolist(),
            }

        # Detection branch — YOLO output [1, 4+nc, N] or [1, N, 5+nc]
        detections = self._decode_yolo(raw, orig_shape, scale, padding)
        return {"type": "detection", "detections": detections}

    def _decode_yolo(
        self,
        raw: np.ndarray,
        orig_shape: Tuple[int, int],
        scale: float,
        padding: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """Decode YOLOv8/v11 ONNX output [1, 4+nc, N] → list of dets."""
        if raw.ndim == 3:
            raw = raw[0]  # [4+nc, N] or [N, 5+nc]
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T  # → [N, 4+nc]

        nc = len(self.class_names)
        pad_w, pad_h = padding
        orig_h, orig_w = orig_shape

        boxes_raw = raw[:, :4]   # cx, cy, w, h (in input_size coords)
        scores = raw[:, 4:]      # [N, nc] or [N, 1+nc]
        if scores.shape[1] == nc + 1:
            # objectness × class_prob
            obj_conf = scores[:, 0:1]
            cls_probs = scores[:, 1:] * obj_conf
        else:
            cls_probs = scores

        cls_ids = np.argmax(cls_probs, axis=1)
        confs = cls_probs[np.arange(len(cls_ids)), cls_ids]
        mask = confs >= self.conf_threshold

        boxes_raw, confs, cls_ids = boxes_raw[mask], confs[mask], cls_ids[mask]

        # cx,cy,w,h → x1,y1,x2,y2 (in input_size space)
        x1 = boxes_raw[:, 0] - boxes_raw[:, 2] / 2
        y1 = boxes_raw[:, 1] - boxes_raw[:, 3] / 2
        x2 = boxes_raw[:, 0] + boxes_raw[:, 2] / 2
        y2 = boxes_raw[:, 1] + boxes_raw[:, 3] / 2

        # Remove padding and rescale to original image
        x1 = np.clip((x1 - pad_w) / scale, 0, orig_w)
        y1 = np.clip((y1 - pad_h) / scale, 0, orig_h)
        x2 = np.clip((x2 - pad_w) / scale, 0, orig_w)
        y2 = np.clip((y2 - pad_h) / scale, 0, orig_h)

        # NMS
        keep = _nms(np.stack([x1, y1, x2, y2], axis=1), confs, self.iou_threshold)

        results = []
        for i in keep:
            name = self.class_names[cls_ids[i]] if self.class_names else str(cls_ids[i])
            results.append({
                "bbox_xyxy": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                "confidence": float(confs[i]),
                "class_id": int(cls_ids[i]),
                "class_name": name,
            })
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _warmup(self, n: int) -> None:
        shape = (1, 3, self.input_size[1], self.input_size[0])
        dummy = np.random.rand(*shape).astype(np.float32)
        for _ in range(n):
            self.session.run(self.output_names, {self.input_name: dummy})
        log.info(f"Warmup complete ({n} runs)")

    def _load_metadata(self) -> Dict[str, str]:
        model = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        return {k: v for k, v in model.get_modelmeta().custom_metadata_map.items()}

    @staticmethod
    def _load(source: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(source, (str, Path)):
            img = cv2.imread(str(source))
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
            return img
        return source


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> List[int]:
    """Pure NumPy NMS implementation."""
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
        order = order[1:][iou <= iou_threshold]
    return keep
