"""
SPIRO ML — AugmentationPipeline
Production Albumentations pipeline for SPIRO with:
  - Geometric: flip, rotate, scale, perspective, crop
  - Color: brightness, contrast, hue, saturation
  - Filtering: blur, CLAHE, noise
  - Weather: rain, fog, shadow
  - Compression artifacts
  - Bounding-box synchronisation throughout
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

from lib.ml.core.logger import get_logger

log = get_logger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_heavy_augmentation(image_size: Tuple[int, int] = (640, 640)) -> A.Compose:
    """
    Full training augmentation pipeline.

    Returns
    -------
    A.Compose with bbox_params set for YOLO format.
    """
    return A.Compose(
        [
            # ── Geometric ──────────────────────────────────────────────────
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.05),
            A.RandomRotate90(p=0.2),
            A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.4),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.3, rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101, p=0.4,
            ),
            A.Perspective(scale=(0.02, 0.08), p=0.3),
            A.RandomResizedCrop(
                height=image_size[1], width=image_size[0],
                scale=(0.5, 1.0), ratio=(0.75, 1.33), p=0.3,
            ),
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.85, 1.15),
                shear=(-8, 8),
                p=0.3,
            ),

            # ── Color / Exposure ───────────────────────────────────────────
            A.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.2, hue=0.1, p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=25, sat_shift_limit=35, val_shift_limit=30, p=0.4,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.4,
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
            A.ToGray(p=0.05),

            # ── Blur / Noise ───────────────────────────────────────────────
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ], p=0.2),
            A.GaussNoise(var_limit=(10.0, 60.0), p=0.2),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.15),

            # ── Weather effects ────────────────────────────────────────────
            A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.25, alpha_coef=0.1, p=0.1),
            A.RandomRain(
                slant_lower=-10, slant_upper=10,
                drop_length=12, drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=3,
                brightness_coefficient=0.9,
                rain_type="drizzle",
                p=0.1,
            ),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.15,
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0, angle_upper=1,
                num_flare_circles_lower=3,
                num_flare_circles_upper=6,
                src_radius=200,
                src_color=(255, 255, 200),
                p=0.05,
            ),

            # ── Compression / Degradation ─────────────────────────────────
            A.ImageCompression(quality_lower=40, quality_upper=95, p=0.15),
            A.Downscale(scale_min=0.5, scale_max=0.9, p=0.1),

            # ── Dropout / Occlusion ───────────────────────────────────────
            A.CoarseDropout(
                max_holes=12, max_height=48, max_width=48,
                min_holes=1, fill_value=114, p=0.3,
            ),
            A.GridDropout(ratio=0.3, p=0.1),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.25,
            clip=True,
        ),
    )


def build_val_augmentation(image_size: Tuple[int, int] = (640, 640)) -> A.Compose:
    """Validation/inference — no augmentation, just resize."""
    return A.Compose(
        [A.LongestMaxSize(max_size=max(image_size))],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], clip=True,
        ),
    )


# =============================================================================
# Offline batch augmentor
# =============================================================================

class AugmentationPipeline:
    """
    Runs offline augmentation on a split directory.

    Reads images + YOLO labels, applies random augmentations,
    writes N copies per image back to the same directory.

    Parameters
    ----------
    image_size : tuple
        Target (W, H) for resize.
    factor : int
        Number of augmented copies per original image.
    seed : int
        Random seed for reproducibility.

    Example
    -------
    >>> aug = AugmentationPipeline(factor=3)
    >>> aug.augment_split("datasets/processed/train")
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (640, 640),
        factor: int = 3,
        seed: int = 42,
    ) -> None:
        self.image_size = image_size
        self.factor = factor
        self.seed = seed
        self._transform = build_heavy_augmentation(image_size)

    def augment_split(self, split_dir: Path) -> int:
        """
        Augment all images in split_dir.

        Returns number of new images written.
        """
        split_dir = Path(split_dir)
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        image_paths = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        np.random.seed(self.seed)

        written = 0
        for img_path in tqdm(image_paths, desc=f"Augmenting {split_dir.name}"):
            img, bboxes, cls_ids = self._load(img_path, labels_dir)
            if img is None:
                continue

            for i in range(self.factor):
                try:
                    result = self._transform(
                        image=img, bboxes=bboxes, class_labels=cls_ids
                    )
                except Exception as e:
                    log.debug(f"Aug failed for {img_path.name}: {e}")
                    continue

                aug_img = result["image"]
                aug_bboxes = result["bboxes"]
                aug_cls = result["class_labels"]

                out_stem = f"{img_path.stem}_aug{i:03d}"
                out_img = images_dir / f"{out_stem}.jpg"
                out_lbl = labels_dir / f"{out_stem}.txt"

                cv2.imwrite(str(out_img), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                with open(out_lbl, "w") as f:
                    for cls_id, (cx, cy, w, h) in zip(aug_cls, aug_bboxes):
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                written += 1

        log.info(f"Augmentation complete — {written} new images in {split_dir}")
        return written

    def preview(
        self,
        image_path: Path,
        labels_path: Optional[Path] = None,
        n_samples: int = 4,
        output_dir: Optional[Path] = None,
    ) -> List[np.ndarray]:
        """
        Generate N augmented previews of a single image.
        Draws bounding boxes for visual inspection.

        Returns list of BGR images.
        """
        from lib.ml.dataset_engineering.visualization.charts import draw_yolo_boxes

        img, bboxes, cls_ids = self._load(image_path, image_path.parent.parent / "labels")
        if labels_path and labels_path.exists():
            img, bboxes, cls_ids = self._load(image_path, labels_path.parent)

        previews = []
        for i in range(n_samples):
            try:
                result = self._transform(image=img, bboxes=bboxes, class_labels=cls_ids)
                annotated = draw_yolo_boxes(
                    result["image"], result["bboxes"], result["class_labels"]
                )
                previews.append(annotated)
                if output_dir:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(Path(output_dir) / f"preview_{i:02d}.jpg"), annotated
                    )
            except Exception as e:
                log.debug(f"Preview error: {e}")

        return previews

    @staticmethod
    def _load(
        img_path: Path, labels_dir: Path
    ) -> Tuple[Optional[np.ndarray], List[Tuple], List[int]]:
        img = cv2.imread(str(img_path))
        if img is None:
            return None, [], []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes: List[Tuple] = []
        cls_ids: List[int] = []
        lbl = labels_dir / f"{img_path.stem}.txt"
        if lbl.exists():
            with open(lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_ids.append(int(parts[0]))
                        bboxes.append(tuple(map(float, parts[1:])))
        return img, bboxes, cls_ids
