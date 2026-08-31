"""
SPIRO ML — Utility helpers.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np


def compute_file_hash(path: Union[str, Path], algo: str = "md5") -> str:
    """Return hex digest of a file (for deduplication / integrity checks)."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    color: tuple = (0, 200, 80),
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw detection boxes on a BGR image copy.

    Parameters
    ----------
    image : np.ndarray
        BGR image.
    detections : list
        List of detection dicts from ONNXInferenceEngine.

    Returns
    -------
    np.ndarray
        Annotated image copy.
    """
    canvas = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        label = f"{det.get('class_name', '')} {det.get('confidence', 0):.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            canvas, label, (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
        )
    return canvas


def load_class_names(path: Union[str, Path]) -> List[str]:
    """Load class names from a plain text file (one class per line)."""
    path = Path(path)
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def load_json(path: Union[str, Path]) -> Any:
    with open(path) as f:
        return json.load(f)


class Timer:
    """Context-manager timer for profiling."""

    def __init__(self, name: str = "block") -> None:
        self.name = name
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000

    def __str__(self) -> str:
        return f"{self.name}: {self.elapsed_ms:.2f}ms"
