"""
SPIRO ML — ImagePreprocessor
Full image validation and preprocessing pipeline:
  1. Format & size validation
  2. EXIF auto-orientation
  3. Blur detection (Laplacian variance)
  4. Brightness check
  5. Quality scoring
  6. Letterbox resize with aspect-ratio preservation
  7. BGR→RGB normalisation for YOLO input
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from lib.ml.core.logger import get_logger

log = get_logger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


@dataclass
class ImageQuality:
    """Result of image quality checks."""
    blur_score: float = 0.0          # Laplacian variance (higher = sharper)
    brightness_mean: float = 0.0     # mean pixel value 0–255
    quality_score: float = 0.0       # composite 0–1
    is_blurry: bool = False
    is_too_dark: bool = False
    is_overexposed: bool = False
    rejection_reason: Optional[str] = None
    passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blur_score": round(self.blur_score, 2),
            "brightness_mean": round(self.brightness_mean, 2),
            "quality_score": round(self.quality_score, 3),
            "is_blurry": self.is_blurry,
            "is_too_dark": self.is_too_dark,
            "is_overexposed": self.is_overexposed,
            "rejection_reason": self.rejection_reason,
            "passed": self.passed,
        }


@dataclass
class PreprocessResult:
    """Output of the preprocessing stage."""
    blob: np.ndarray              # NCHW float32 [0,1] for YOLO
    image_bgr: np.ndarray         # Original BGR (for cropping later)
    image_rgb: np.ndarray         # RGB, letterboxed to target size
    original_shape: Tuple[int, int]  # (H, W) before resize
    padded_shape: Tuple[int, int]    # (H, W) after letterbox
    scale: float                  # resize scale factor
    pad_w: int                    # horizontal padding (pixels each side)
    pad_h: int                    # vertical padding (pixels each side)
    quality: ImageQuality
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImagePreprocessor:
    """
    Validates and prepares images for the SPIRO detection pipeline.

    Parameters
    ----------
    target_size : int
        Letterbox target (square). Default 640.
    blur_threshold : float
        Laplacian variance below this → blurry rejection.
    brightness_min / max : float
        Mean brightness bounds.
    quality_threshold : float
        Composite quality score below this → rejection.
    max_file_size_mb : float
        Reject files larger than this.
    auto_orient : bool
        Apply EXIF orientation correction.

    Example
    -------
    >>> prep = ImagePreprocessor(target_size=640)
    >>> result = prep.process("photo.jpg")
    >>> if result.quality.passed:
    ...     blob = result.blob  # feed to YOLO
    """

    def __init__(
        self,
        target_size: int = 640,
        letterbox_color: Tuple[int, int, int] = (114, 114, 114),
        blur_threshold: float = 80.0,
        brightness_min: float = 20.0,
        brightness_max: float = 245.0,
        quality_threshold: float = 0.3,
        max_file_size_mb: float = 50.0,
        min_resolution: Tuple[int, int] = (64, 64),
        max_resolution: Tuple[int, int] = (8192, 8192),
        auto_orient: bool = True,
    ) -> None:
        self.target_size = target_size
        self.letterbox_color = letterbox_color
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.quality_threshold = quality_threshold
        self.max_file_size_mb = max_file_size_mb
        self.min_res = min_resolution
        self.max_res = max_resolution
        self.auto_orient = auto_orient

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(
        self,
        source: Union[str, Path, bytes, np.ndarray],
    ) -> PreprocessResult:
        """
        Full preprocessing pipeline.

        Parameters
        ----------
        source : file path, raw bytes, or BGR ndarray.

        Returns
        -------
        PreprocessResult — always returned; check `.quality.passed`.
        """
        # Load image
        img_bgr, metadata = self._load(source)
        quality = ImageQuality()

        if img_bgr is None:
            quality.rejection_reason = "unreadable_image"
            quality.passed = False
            dummy = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            return self._make_failed(dummy, quality, metadata)

        h, w = img_bgr.shape[:2]
        metadata["original_wh"] = (w, h)

        # Resolution check
        if w < self.min_res[0] or h < self.min_res[1]:
            quality.rejection_reason = f"resolution_too_small_{w}x{h}"
            quality.passed = False
            return self._make_failed(img_bgr, quality, metadata)

        if w > self.max_res[0] or h > self.max_res[1]:
            # Resize down and continue (warn but don't reject)
            log.warning(f"Image {w}x{h} exceeds max resolution — resizing")
            scale = min(self.max_res[0] / w, self.max_res[1] / h)
            img_bgr = cv2.resize(
                img_bgr, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )
            h, w = img_bgr.shape[:2]

        # Quality checks
        quality = self._assess_quality(img_bgr)
        if not quality.passed:
            return self._make_failed(img_bgr, quality, metadata)

        # Letterbox
        letterboxed, scale, pad_w, pad_h = self._letterbox(img_bgr)

        # Build YOLO blob: RGB, NCHW, float32 [0,1]
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis]

        return PreprocessResult(
            blob=blob,
            image_bgr=img_bgr,
            image_rgb=rgb,
            original_shape=(h, w),
            padded_shape=(self.target_size, self.target_size),
            scale=scale,
            pad_w=pad_w,
            pad_h=pad_h,
            quality=quality,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Quality assessment
    # ------------------------------------------------------------------

    def _assess_quality(self, img_bgr: np.ndarray) -> ImageQuality:
        quality = ImageQuality()
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Blur detection (Laplacian variance)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        quality.blur_score = blur_score
        quality.is_blurry = blur_score < self.blur_threshold

        # Brightness
        brightness = float(gray.mean())
        quality.brightness_mean = brightness
        quality.is_too_dark = brightness < self.brightness_min
        quality.is_overexposed = brightness > self.brightness_max

        # Composite quality score (0–1)
        blur_norm = min(1.0, blur_score / (self.blur_threshold * 5))
        bright_norm = 1.0 - abs(brightness - 128) / 128
        quality.quality_score = float(0.6 * blur_norm + 0.4 * bright_norm)

        # Determine rejection
        if quality.is_blurry:
            quality.rejection_reason = f"blur_score_{blur_score:.1f}_below_{self.blur_threshold}"
            quality.passed = False
        elif quality.is_too_dark:
            quality.rejection_reason = f"brightness_{brightness:.1f}_below_{self.brightness_min}"
            quality.passed = False
        elif quality.is_overexposed:
            quality.rejection_reason = f"brightness_{brightness:.1f}_above_{self.brightness_max}"
            quality.passed = False
        elif quality.quality_score < self.quality_threshold:
            quality.rejection_reason = f"quality_score_{quality.quality_score:.3f}_below_{self.quality_threshold}"
            quality.passed = False

        return quality

    # ------------------------------------------------------------------
    # Letterbox
    # ------------------------------------------------------------------

    def _letterbox(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, float, int, int]:
        """
        Letterbox-resize to target_size × target_size.
        Returns (padded_img, scale, pad_w, pad_h).
        """
        h, w = img.shape[:2]
        t = self.target_size
        scale = min(t / w, t / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((t, t, 3), self.letterbox_color, dtype=np.uint8)
        pad_w = (t - nw) // 2
        pad_h = (t - nh) // 2
        canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
        return canvas, scale, pad_w, pad_h

    # ------------------------------------------------------------------
    # Loading with EXIF
    # ------------------------------------------------------------------

    def _load(
        self, source: Union[str, Path, bytes, np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Dict]:
        metadata: Dict[str, Any] = {}

        if isinstance(source, np.ndarray):
            return source, metadata

        if isinstance(source, bytes):
            arr = np.frombuffer(source, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if self.auto_orient:
                img = self._apply_exif_orientation_from_bytes(source, img)
            return img, metadata

        path = Path(source)
        if not path.exists():
            log.error(f"File not found: {path}")
            return None, {"error": "file_not_found"}

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTS:
            log.warning(f"Unsupported format: {suffix}")
            return None, {"error": f"unsupported_format_{suffix}"}

        size_mb = path.stat().st_size / 1e6
        metadata["file_size_mb"] = round(size_mb, 2)
        if size_mb > self.max_file_size_mb:
            log.warning(f"File too large: {size_mb:.1f}MB > {self.max_file_size_mb}MB")
            return None, {"error": "file_too_large"}

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return None, {"error": "imread_failed"}

        if self.auto_orient:
            img = self._apply_exif_orientation(path, img)

        metadata["path"] = str(path)
        metadata["format"] = suffix.lstrip(".")
        return img, metadata

    @staticmethod
    def _apply_exif_orientation(path: Path, img: np.ndarray) -> np.ndarray:
        """Apply EXIF orientation correction using PIL if available."""
        try:
            from PIL import Image, ExifTags
            pil = Image.open(str(path))
            exif = pil._getexif()
            if exif is None:
                return img
            orientation_key = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
            )
            if orientation_key is None:
                return img
            orientation = exif.get(orientation_key, 1)
            rot_map = {3: cv2.ROTATE_180, 6: cv2.ROTATE_90_COUNTERCLOCKWISE,
                       8: cv2.ROTATE_90_CLOCKWISE}
            if orientation in rot_map:
                img = cv2.rotate(img, rot_map[orientation])
        except Exception:
            pass
        return img

    @staticmethod
    def _apply_exif_orientation_from_bytes(data: bytes, img: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if img is None:
            return None
        try:
            from PIL import Image
            pil = Image.open(io.BytesIO(data))
            from PIL import ExifTags
            exif = pil._getexif()
            if exif is None:
                return img
            orientation_key = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
            )
            if orientation_key is None:
                return img
            orientation = exif.get(orientation_key, 1)
            rot_map = {3: cv2.ROTATE_180, 6: cv2.ROTATE_90_COUNTERCLOCKWISE,
                       8: cv2.ROTATE_90_CLOCKWISE}
            if orientation in rot_map:
                img = cv2.rotate(img, rot_map[orientation])
        except Exception:
            pass
        return img

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_failed(
        self,
        img: np.ndarray,
        quality: ImageQuality,
        metadata: Dict,
    ) -> PreprocessResult:
        h, w = img.shape[:2] if img is not None else (0, 0)
        dummy_blob = np.zeros((1, 3, self.target_size, self.target_size), dtype=np.float32)
        dummy_rgb = np.zeros((self.target_size, self.target_size, 3), dtype=np.float32)
        return PreprocessResult(
            blob=dummy_blob,
            image_bgr=img if img is not None else np.zeros((64, 64, 3), dtype=np.uint8),
            image_rgb=dummy_rgb,
            original_shape=(h, w),
            padded_shape=(self.target_size, self.target_size),
            scale=1.0,
            pad_w=0,
            pad_h=0,
            quality=quality,
            metadata=metadata,
        )
