"""
SPIRO ML — SPIRODetector (YOLOv11 wrapper)
Wraps Ultralytics YOLO with SPIRO-specific initialisation,
fine-tuning helpers, and a clean predict() API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from lib.ml.core.config import ConfigManager
from lib.ml.core.device import resolve_device
from lib.ml.core.logger import get_logger

log = get_logger(__name__)


class Detection:
    """Lightweight container for a single detection result."""

    __slots__ = ("bbox_xyxy", "confidence", "class_id", "class_name")

    def __init__(
        self,
        bbox_xyxy: Tuple[float, float, float, float],
        confidence: float,
        class_id: int,
        class_name: str,
    ) -> None:
        self.bbox_xyxy = bbox_xyxy
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

    def __repr__(self) -> str:
        x1, y1, x2, y2 = [f"{v:.1f}" for v in self.bbox_xyxy]
        return (
            f"Detection(class={self.class_name!r}, conf={self.confidence:.3f}, "
            f"box=[{x1},{y1},{x2},{y2}])"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
        }


class SPIRODetector:
    """
    YOLOv11 object detector for SPIRO.

    Parameters
    ----------
    cfg : ConfigManager
        Loaded project config.
    weights : str | Path, optional
        Path to .pt weights file. If None, uses cfg.model.pretrained_weights.

    Example
    -------
    >>> detector = SPIRODetector(cfg)
    >>> results = detector.predict("image.jpg")
    >>> for det in results:
    ...     print(det)
    """

    def __init__(
        self,
        cfg: ConfigManager,
        weights: Optional[Union[str, Path]] = None,
    ) -> None:
        self.cfg = cfg
        self.class_names: List[str] = list(cfg.dataset.class_names)
        self.device = resolve_device(cfg.training.device)
        self.conf_threshold = cfg.inference.conf_threshold
        self.iou_threshold = cfg.inference.iou_threshold
        self.imgsz = tuple(cfg.model.input_size)

        weights_path = weights or cfg.model.pretrained_weights
        log.info(f"Loading YOLOv11 weights: {weights_path}")
        self.model = YOLO(str(weights_path))
        self.model.to(self.device)
        log.info(f"SPIRODetector ready on {self.device}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        verbose: bool = False,
    ) -> List[Detection]:
        """
        Run detection on a single image.

        Parameters
        ----------
        source : str | Path | np.ndarray
            File path or BGR numpy array.
        conf : float, optional
            Override confidence threshold.
        iou : float, optional
            Override NMS IoU threshold.
        verbose : bool
            Print Ultralytics inference log.

        Returns
        -------
        List[Detection]
        """
        results = self.model.predict(
            source=source,
            conf=conf or self.conf_threshold,
            iou=iou or self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=verbose,
        )

        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            for box, conf_val, cls_id in zip(boxes, confs, cls_ids):
                name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                detections.append(
                    Detection(
                        bbox_xyxy=tuple(box.tolist()),
                        confidence=float(conf_val),
                        class_id=int(cls_id),
                        class_name=name,
                    )
                )
        return detections

    def predict_batch(
        self,
        sources: List[Union[str, Path, np.ndarray]],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[List[Detection]]:
        """Run detection on a list of images."""
        return [self.predict(src, conf=conf, iou=iou) for src in sources]

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save PyTorch weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        log.info(f"Model saved to {path}")

    @classmethod
    def from_checkpoint(
        cls, cfg: ConfigManager, checkpoint_path: Union[str, Path]
    ) -> "SPIRODetector":
        """Load detector from a specific checkpoint."""
        return cls(cfg, weights=checkpoint_path)

    def info(self) -> Dict[str, Any]:
        """Return model architecture summary dict."""
        return {
            "architecture": "YOLOv11",
            "variant": self.cfg.model.variant,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "input_size": self.imgsz,
            "device": str(self.device),
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def draw(
        self,
        image: np.ndarray,
        detections: List[Detection],
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes on a BGR image (in-place copy)."""
        canvas = image.copy()
        cmap = self._class_colors()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            color = cmap[det.class_id % len(cmap)]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(
                canvas, label, (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
        return canvas

    @staticmethod
    def _class_colors() -> List[Tuple[int, int, int]]:
        return [
            (0, 114, 189), (217, 83, 25), (237, 177, 32),
            (126, 47, 142), (119, 172, 48), (77, 190, 238),
            (162, 20, 47), (76, 153, 0), (255, 128, 0),
        ]
