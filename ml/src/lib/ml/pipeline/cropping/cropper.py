"""
SPIRO ML — ObjectCropper
Extracts padded, aspect-ratio-preserving crop patches from detections.
Validates crops before returning them to the verification stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from lib.ml.core.logger import get_logger
from lib.ml.pipeline.detection.yolo_engine import Detection

log = get_logger(__name__)


@dataclass
class Crop:
    """A validated object crop ready for EfficientNetV2 verification."""
    image_bgr: np.ndarray           # BGR crop patch
    detection_idx: int              # index back into detections list
    detection: Detection
    bbox_padded: Tuple[float, float, float, float]  # padded bbox in original coords
    is_valid: bool = True
    rejection_reason: Optional[str] = None


class ObjectCropper:
    """
    Extracts padded bounding-box crops from the original image.

    Parameters
    ----------
    padding_fraction : float
        Fractional padding around each bbox (0.08 = 8% each side).
    min_crop_pixels : int
        Minimum edge length in pixels; smaller crops are rejected.
    target_size : int, optional
        If set, resize every crop to this square size before returning.
        Leave None to return variable-size crops (EfficientNetV2 handles resize).

    Example
    -------
    >>> cropper = ObjectCropper(padding_fraction=0.08, min_crop_pixels=16)
    >>> crops = cropper.crop(image_bgr, detections)
    """

    def __init__(
        self,
        padding_fraction: float = 0.08,
        min_crop_pixels: int = 16,
        target_size: Optional[int] = None,
    ) -> None:
        self.padding_fraction = padding_fraction
        self.min_crop_pixels = min_crop_pixels
        self.target_size = target_size

    def crop(
        self,
        image_bgr: np.ndarray,
        detections: List[Detection],
    ) -> List[Crop]:
        """
        Crop every detection from the image.

        Parameters
        ----------
        image_bgr : np.ndarray — full-size original image
        detections : list of Detection objects

        Returns
        -------
        List[Crop] — one per detection; check `.is_valid`
        """
        ih, iw = image_bgr.shape[:2]
        crops: List[Crop] = []

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox_xyxy
            bw = x2 - x1
            bh = y2 - y1

            # Add padding
            pw = bw * self.padding_fraction
            ph = bh * self.padding_fraction
            x1p = max(0, int(x1 - pw))
            y1p = max(0, int(y1 - ph))
            x2p = min(iw, int(x2 + pw))
            y2p = min(ih, int(y2 + ph))

            crop_w = x2p - x1p
            crop_h = y2p - y1p

            # Validate
            if crop_w < self.min_crop_pixels or crop_h < self.min_crop_pixels:
                crops.append(Crop(
                    image_bgr=np.zeros((self.min_crop_pixels, self.min_crop_pixels, 3), dtype=np.uint8),
                    detection_idx=idx,
                    detection=det,
                    bbox_padded=(x1p, y1p, x2p, y2p),
                    is_valid=False,
                    rejection_reason=f"crop_too_small_{crop_w}x{crop_h}",
                ))
                continue

            patch = image_bgr[y1p:y2p, x1p:x2p].copy()

            if patch.size == 0:
                crops.append(Crop(
                    image_bgr=np.zeros((self.min_crop_pixels, self.min_crop_pixels, 3), dtype=np.uint8),
                    detection_idx=idx,
                    detection=det,
                    bbox_padded=(x1p, y1p, x2p, y2p),
                    is_valid=False,
                    rejection_reason="empty_crop",
                ))
                continue

            if self.target_size:
                patch = cv2.resize(patch, (self.target_size, self.target_size),
                                   interpolation=cv2.INTER_LINEAR)

            crops.append(Crop(
                image_bgr=patch,
                detection_idx=idx,
                detection=det,
                bbox_padded=(float(x1p), float(y1p), float(x2p), float(y2p)),
                is_valid=True,
            ))

        valid = sum(1 for c in crops if c.is_valid)
        log.debug(f"Cropped {valid}/{len(detections)} valid patches")
        return crops

    def preprocess_for_effnet(
        self,
        crop: Crop,
        target_size: int,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> np.ndarray:
        """
        Resize, RGB-convert, and normalise a crop for EfficientNetV2.

        Returns
        -------
        np.ndarray — [1, 3, H, W] float32 NCHW blob
        """
        resized = cv2.resize(crop.image_bgr, (target_size, target_size),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean_arr = np.array(mean, dtype=np.float32)
        std_arr  = np.array(std, dtype=np.float32)
        rgb = (rgb - mean_arr) / std_arr
        return rgb.transpose(2, 0, 1)[np.newaxis]

    def preprocess_batch(
        self,
        crops: List[Crop],
        target_size: int,
    ) -> np.ndarray:
        """
        Build a batched NCHW blob from a list of valid crops.

        Returns
        -------
        np.ndarray — [N, 3, target_size, target_size] float32
        """
        blobs = [
            self.preprocess_for_effnet(c, target_size)
            for c in crops if c.is_valid
        ]
        if not blobs:
            return np.zeros((0, 3, target_size, target_size), dtype=np.float32)
        return np.concatenate(blobs, axis=0)
