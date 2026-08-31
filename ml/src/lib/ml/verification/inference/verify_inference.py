"""
SPIRO ML — VerifyInference
Production inference engine for the two-stage SPIRO pipeline:
  1. Single image / batch image verification (PyTorch)
  2. ONNX-based inference (production)
  3. Crop-based verification from YOLOv11 detections
  4. Temperature scaling for confidence calibration
  5. Top-5 predictions
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image

from lib.ml.core.logger import get_logger
from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader

log = get_logger(__name__)


class VerifyInferenceEngine:
    """
    ONNX-based inference engine for the EfficientNetV2 verifier.
    Mirrors the interface of ONNXInferenceEngine (detection) for consistency.

    Parameters
    ----------
    onnx_path : str | Path
        Path to the exported .onnx model.
    input_size : int
        Image input size (square).
    class_names : list[str], optional
        If None, loads from SPIRO taxonomy.
    providers : list[str], optional
        ORT execution providers.
    warmup_runs : int
        Number of warmup forward passes.

    Example
    -------
    >>> engine = VerifyInferenceEngine("models/exports/spiro_effnetv2_s_best.onnx")
    >>> result = engine.verify("crop.jpg")
    >>> results = engine.verify_batch(["crop1.jpg", "crop2.jpg"])
    """

    def __init__(
        self,
        onnx_path: Union[str, Path],
        input_size: int = 300,
        class_names: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        warmup_runs: int = 3,
        temperature: float = 1.0,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.input_size = input_size
        self.temperature = temperature

        self.taxonomy = TaxonomyLoader()
        self.class_names = class_names or self.taxonomy.all_names()

        # Providers
        available = ort.get_available_providers()
        if providers is None:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available
                else ["CPUExecutionProvider"]
            )
        self.providers = providers

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            str(self.onnx_path), sess_options=opts, providers=self.providers
        )
        self.input_name = self.session.get_inputs()[0].name
        log.info(f"VerifyInferenceEngine loaded: {self.onnx_path.name}")
        log.info(f"  Providers: {self.providers}")
        log.info(f"  Input: {self.input_name} | size: {self.input_size}×{self.input_size}")

        if warmup_runs > 0:
            self._warmup(warmup_runs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        source: Union[str, Path, np.ndarray, Image.Image],
        top_k: int = 5,
        return_timing: bool = False,
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], float]]:
        """
        Verify a single image crop.

        Parameters
        ----------
        source : image path, BGR ndarray, or PIL Image
        top_k : int — number of top predictions
        return_timing : bool

        Returns
        -------
        dict with class_id, class_name, confidence, top_k predictions
        """
        blob = self._preprocess(self._load(source))
        t0 = time.perf_counter()
        logits = self.session.run(None, {self.input_name: blob})[0][0]
        elapsed = (time.perf_counter() - t0) * 1000

        if self.temperature != 1.0:
            logits = logits / self.temperature

        probs = self._softmax(logits)
        top_k = min(top_k, len(probs))
        top_indices = np.argsort(probs)[::-1][:top_k].tolist()

        result = {
            "class_id": int(top_indices[0]),
            "class_name": self.class_names[top_indices[0]],
            "confidence": float(probs[top_indices[0]]),
            "top_k": [
                {
                    "rank": i + 1,
                    "class_id": int(idx),
                    "class_name": self.class_names[idx],
                    "confidence": float(probs[idx]),
                }
                for i, idx in enumerate(top_indices)
            ],
            "probabilities": probs.tolist(),
        }

        if return_timing:
            return result, elapsed
        return result

    def verify_batch(
        self,
        sources: List[Union[str, Path, np.ndarray, Image.Image]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Run verification on a list of image crops."""
        return [self.verify(s, top_k=top_k) for s in sources]

    def verify_detections(
        self,
        image_bgr: np.ndarray,
        detections: List[Dict[str, Any]],
        top_k: int = 5,
        min_crop_size: int = 16,
        padding: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Verify every detection from YOLOv11 by cropping and classifying.

        Parameters
        ----------
        image_bgr : np.ndarray
            Full-size BGR image.
        detections : list
            Output from ONNXInferenceEngine (bbox_xyxy, confidence, class_id, class_name).
        top_k : int
        min_crop_size : int
            Minimum crop edge size (pixels) to skip tiny boxes.
        padding : float
            Fractional padding around each bbox.

        Returns
        -------
        List of dicts — each detection enriched with effnet verification.
        """
        ih, iw = image_bgr.shape[:2]
        enriched = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            pw = (x2 - x1) * padding
            ph = (y2 - y1) * padding
            x1c = max(0, int(x1 - pw))
            y1c = max(0, int(y1 - ph))
            x2c = min(iw, int(x2 + pw))
            y2c = min(ih, int(y2 + ph))

            crop = image_bgr[y1c:y2c, x1c:x2c]
            if crop.shape[0] < min_crop_size or crop.shape[1] < min_crop_size:
                enriched.append({**det, "verification": None})
                continue

            verification, elapsed = self.verify(crop, top_k=top_k, return_timing=True)
            enriched.append({
                **det,
                "verification": verification,
                "verification_latency_ms": round(elapsed, 2),
            })

        return enriched

    def calibrate_temperature(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Find optimal temperature T via NLL minimisation on calibration data.

        Parameters
        ----------
        logits : ndarray [N, C]
        labels : ndarray [N]
        lr : float
        max_iter : int

        Returns
        -------
        float — optimal temperature
        """
        import torch
        from torch import optim

        T = torch.nn.Parameter(torch.ones(1))
        optimizer = optim.LBFGS([T], lr=lr, max_iter=max_iter)
        logits_t = torch.from_numpy(logits).float()
        labels_t = torch.from_numpy(labels).long()

        def eval_fn():
            optimizer.zero_grad()
            scaled = logits_t / T
            loss = F.cross_entropy(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        optimal_T = float(T.item())
        log.info(f"Calibrated temperature: {optimal_T:.4f}")
        self.temperature = optimal_T
        return optimal_T

    def benchmark(self, n_runs: int = 100) -> Dict[str, float]:
        """Measure inference latency statistics."""
        dummy = np.random.rand(1, 3, self.input_size, self.input_size).astype(np.float32)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.session.run(None, {self.input_name: dummy})
            times.append((time.perf_counter() - t0) * 1000)
        arr = np.array(times)
        return {
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "min_ms": float(arr.min()),
            "p95_ms": float(np.percentile(arr, 95)),
            "fps": float(1000 / arr.mean()),
        }

    # ------------------------------------------------------------------
    # Pre / post processing
    # ------------------------------------------------------------------

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """Resize, RGB convert, normalise, add batch dim."""
        resized = cv2.resize(img_bgr, (self.input_size, self.input_size),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb  = (rgb - mean) / std
        return rgb.transpose(2, 0, 1)[np.newaxis]  # NCHW

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    @staticmethod
    def _load(source: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(source, (str, Path)):
            img = cv2.imread(str(source))
            if img is None:
                raise FileNotFoundError(f"Cannot read: {source}")
            return img
        if isinstance(source, np.ndarray):
            return source
        if isinstance(source, Image.Image):
            return cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2BGR)
        raise TypeError(f"Unsupported type: {type(source)}")

    def _warmup(self, n: int) -> None:
        dummy = np.random.rand(1, 3, self.input_size, self.input_size).astype(np.float32)
        for _ in range(n):
            self.session.run(None, {self.input_name: dummy})
        log.info(f"Warmup complete ({n} runs)")
