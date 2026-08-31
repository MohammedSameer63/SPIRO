"""
SPIRO ML — ConfidenceFusion
Fuses YOLOv11 detection scores with EfficientNetV2 verification probabilities
to produce a unified SPIRO waste category assignment.

Supported fusion methods:
  - weighted_average   : α·P(YOLO) + β·P(EffNet)
  - geometric_mean     : P(YOLO)^α × P(EffNet)^β  (normalised)
  - harmonic_mean      : 2·P(YOLO)·P(EffNet) / (P(YOLO)+P(EffNet))
  - bayesian           : prior × P(EffNet | class) (Bayesian update)
  - temperature        : Temperature-scaled softmax of concatenated logits
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from lib.ml.core.logger import get_logger
from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader

log = get_logger(__name__)


class FusionResult:
    """Container for a fused SPIRO classification result."""

    __slots__ = (
        "class_id", "class_name", "fused_confidence",
        "yolo_class_id", "yolo_confidence",
        "effnet_class_id", "effnet_confidence",
        "method", "agreement",
    )

    def __init__(
        self,
        class_id: int,
        class_name: str,
        fused_confidence: float,
        yolo_class_id: int,
        yolo_confidence: float,
        effnet_class_id: int,
        effnet_confidence: float,
        method: str,
    ) -> None:
        self.class_id = class_id
        self.class_name = class_name
        self.fused_confidence = fused_confidence
        self.yolo_class_id = yolo_class_id
        self.yolo_confidence = yolo_confidence
        self.effnet_class_id = effnet_class_id
        self.effnet_confidence = effnet_confidence
        self.method = method
        self.agreement = (yolo_class_id == effnet_class_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "fused_confidence": round(self.fused_confidence, 4),
            "yolo_class_id": self.yolo_class_id,
            "yolo_confidence": round(self.yolo_confidence, 4),
            "effnet_class_id": self.effnet_class_id,
            "effnet_confidence": round(self.effnet_confidence, 4),
            "method": self.method,
            "agreement": self.agreement,
        }

    def __repr__(self) -> str:
        agree = "✓" if self.agreement else "✗"
        return (
            f"FusionResult({self.class_name!r}, "
            f"conf={self.fused_confidence:.3f}, {agree} agree)"
        )


class ConfidenceFusion:
    """
    Fuses YOLOv11 and EfficientNetV2 outputs for SPIRO waste categorisation.

    Parameters
    ----------
    method : str
        One of: weighted_average | geometric_mean | harmonic_mean | bayesian | temperature
    yolo_weight : float
        Weight for YOLO in weighted / geometric fusion (0–1).
    effnet_weight : float
        Weight for EffNet in weighted / geometric fusion.
    temperature : float
        Temperature for temperature-scaling fusion mode.
    prior_path : str, optional
        Path to JSON file with class prior probabilities [num_classes].
    min_confidence : float
        Minimum fused confidence; below this → "uncertain" label.
    num_classes : int

    Example
    -------
    >>> fusion = ConfidenceFusion(method="weighted_average")
    >>> result = fusion.fuse(yolo_det, effnet_result)
    >>> print(result.class_name, result.fused_confidence)
    """

    _VALID_METHODS = {
        "weighted_average", "geometric_mean",
        "harmonic_mean", "bayesian", "temperature",
    }

    def __init__(
        self,
        method: str = "weighted_average",
        yolo_weight: float = 0.4,
        effnet_weight: float = 0.6,
        temperature: float = 1.5,
        prior_path: Optional[str] = None,
        min_confidence: float = 0.2,
        num_classes: int = 109,
    ) -> None:
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"Unknown fusion method {method!r}. Valid: {self._VALID_METHODS}"
            )
        self.method = method
        self.yolo_weight = yolo_weight
        self.effnet_weight = effnet_weight
        self.temperature = temperature
        self.min_confidence = min_confidence
        self.num_classes = num_classes
        self.taxonomy = TaxonomyLoader()

        # Normalise weights
        total = yolo_weight + effnet_weight
        self.yolo_weight /= total
        self.effnet_weight /= total

        # Load class priors
        self._prior: Optional[np.ndarray] = None
        if prior_path:
            with open(prior_path) as f:
                p = json.load(f)
            arr = np.array(p, dtype=np.float64)
            self._prior = arr / arr.sum()
        elif method == "bayesian":
            # Uniform prior
            self._prior = np.ones(num_classes, dtype=np.float64) / num_classes

    # ------------------------------------------------------------------
    # Core fusion
    # ------------------------------------------------------------------

    def fuse(
        self,
        yolo_detection: Dict[str, Any],
        effnet_result: Dict[str, Any],
    ) -> FusionResult:
        """
        Fuse a single YOLOv11 detection with an EfficientNetV2 result.

        Parameters
        ----------
        yolo_detection : dict
            Must have: class_id (int), confidence (float),
            optionally probabilities (list[float]).
        effnet_result : dict
            Must have: class_id (int), confidence (float),
            probabilities (list[float]).

        Returns
        -------
        FusionResult
        """
        nc = self.num_classes
        yolo_cls = int(yolo_detection["class_id"])
        yolo_conf = float(yolo_detection["confidence"])
        effnet_cls = int(effnet_result["class_id"])
        effnet_conf = float(effnet_result["confidence"])

        # Build full probability vectors
        effnet_probs = np.array(effnet_result.get("probabilities", []), dtype=np.float64)
        if len(effnet_probs) != nc:
            effnet_probs = np.zeros(nc)
            if effnet_cls < nc:
                effnet_probs[effnet_cls] = effnet_conf

        yolo_probs = np.array(yolo_detection.get("probabilities", []), dtype=np.float64)
        if len(yolo_probs) != nc:
            yolo_probs = np.zeros(nc)
            if yolo_cls < nc:
                yolo_probs[yolo_cls] = yolo_conf

        # Apply chosen fusion method
        fused_probs = self._apply_fusion(yolo_probs, effnet_probs, yolo_conf, effnet_conf)

        best_cls = int(np.argmax(fused_probs))
        best_conf = float(fused_probs[best_cls])

        # Fall back to "uncertain" if below threshold
        class_name = self.taxonomy.id_to_name(best_cls)
        if best_conf < self.min_confidence:
            class_name = "uncertain"

        return FusionResult(
            class_id=best_cls,
            class_name=class_name,
            fused_confidence=best_conf,
            yolo_class_id=yolo_cls,
            yolo_confidence=yolo_conf,
            effnet_class_id=effnet_cls,
            effnet_confidence=effnet_conf,
            method=self.method,
        )

    def fuse_batch(
        self,
        yolo_detections: List[Dict[str, Any]],
        effnet_results: List[Dict[str, Any]],
    ) -> List[FusionResult]:
        """Fuse a batch of detections with their verification results."""
        assert len(yolo_detections) == len(effnet_results), (
            "yolo_detections and effnet_results must have same length"
        )
        return [self.fuse(y, e) for y, e in zip(yolo_detections, effnet_results)]

    # ------------------------------------------------------------------
    # Fusion method implementations
    # ------------------------------------------------------------------

    def _apply_fusion(
        self,
        yolo_probs: np.ndarray,
        effnet_probs: np.ndarray,
        yolo_conf: float,
        effnet_conf: float,
    ) -> np.ndarray:
        method = self.method

        if method == "weighted_average":
            fused = self.yolo_weight * yolo_probs + self.effnet_weight * effnet_probs
            return self._safe_normalise(fused)

        elif method == "geometric_mean":
            # Element-wise geometric mean with weights as exponents
            eps = 1e-9
            fused = (np.maximum(yolo_probs, eps) ** self.yolo_weight *
                     np.maximum(effnet_probs, eps) ** self.effnet_weight)
            return self._safe_normalise(fused)

        elif method == "harmonic_mean":
            eps = 1e-9
            fused = (2 * np.maximum(yolo_probs, eps) * np.maximum(effnet_probs, eps) /
                     (np.maximum(yolo_probs, eps) + np.maximum(effnet_probs, eps)))
            return self._safe_normalise(fused)

        elif method == "bayesian":
            # Prior × P(EffNet) × P(YOLO) (conditional independence assumption)
            prior = self._prior if self._prior is not None else (
                np.ones(self.num_classes) / self.num_classes
            )
            eps = 1e-12
            posterior = prior * np.maximum(effnet_probs, eps) * np.maximum(yolo_probs, eps)
            return self._safe_normalise(posterior)

        elif method == "temperature":
            # Average logits then apply temperature-scaled softmax
            yolo_logits = np.log(np.maximum(yolo_probs, 1e-12))
            effnet_logits = np.log(np.maximum(effnet_probs, 1e-12))
            avg_logits = (self.yolo_weight * yolo_logits +
                          self.effnet_weight * effnet_logits)
            scaled = avg_logits / self.temperature
            return self._softmax(scaled)

        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def _safe_normalise(arr: np.ndarray) -> np.ndarray:
        s = arr.sum()
        if s < 1e-12:
            return np.ones_like(arr) / len(arr)
        return arr / s

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()
