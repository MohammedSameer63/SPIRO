"""
SPIRO ML — Augmentation Pipeline
Builds Albumentations pipelines for:
  - Training (heavy augmentation)
  - Validation / Inference (normalisation only)
  - Offline batch augmentation (multiply dataset N×)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from lib.ml.core.config import ConfigManager
from lib.ml.core.logger import get_logger

log = get_logger(__name__)


def build_train_transform(cfg: ConfigManager) -> A.Compose:
    """
    Return a heavy Albumentations pipeline for training.
    Compatible with YOLO bounding box format (pascal_voc after conversion).
    """
    aug = cfg.augmentation
    return A.Compose(
        [
            A.HorizontalFlip(p=aug.horizontal_flip),
            A.VerticalFlip(p=aug.vertical_flip),
            A.Rotate(
                limit=aug.rotate_limit,
                border_mode=cv2.BORDER_CONSTANT,
                p=aug.rotate_prob,
            ),
            A.ShiftScaleRotate(
                shift_limit=aug.shift_limit,
                scale_limit=aug.scale_limit,
                rotate_limit=0,
                border_mode=cv2.BORDER_CONSTANT,
                p=aug.shift_prob,
            ),
            A.Perspective(scale=(0.0, aug.perspective_distortion), p=aug.perspective_prob),
            A.ColorJitter(
                brightness=aug.brightness_limit,
                contrast=aug.contrast_limit,
                p=aug.color_prob,
            ),
            A.HueSaturationValue(
                hue_shift_limit=aug.hue_shift,
                sat_shift_limit=aug.sat_shift,
                val_shift_limit=aug.val_shift,
                p=aug.hsv_prob,
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=aug.blur_prob),
            A.GaussNoise(var_limit=(10, 50), p=aug.gaussian_noise_prob),
            A.CoarseDropout(
                max_holes=aug.cutout_num_holes,
                max_height=aug.cutout_max_h,
                max_width=aug.cutout_max_w,
                fill_value=0,
                p=aug.cutout_prob,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
            clip=True,
        ),
    )


def build_val_transform(cfg: ConfigManager) -> A.Compose:
    """Return a no-augmentation pipeline for validation/inference."""
    size = tuple(cfg.dataset.image_size)
    return A.Compose(
        [A.LongestMaxSize(max_size=max(size))],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
        ),
    )


# ---------------------------------------------------------------------------
# Offline batch augmentor
# ---------------------------------------------------------------------------


class OfflineAugmentor:
    """
    Multiply a split's images N× by applying random augmentations offline.
    Writes augmented images + labels back to the same split directory.

    Usage
    -----
    aug = OfflineAugmentor(cfg)
    aug.augment_split("datasets/processed/train", factor=3)
    """

    def __init__(self, cfg: ConfigManager) -> None:
        self.cfg = cfg
        self.transform = build_train_transform(cfg)
        self.image_size = tuple(cfg.dataset.image_size)

    def augment_split(self, split_dir: str | Path, factor: int = 3) -> int:
        """
        Augment all images in split_dir/images  and write results.

        Parameters
        ----------
        split_dir : Path
            Root of split (contains images/ and labels/ subdirectories).
        factor : int
            How many augmented copies per original image.

        Returns
        -------
        int — number of new images written.
        """
        split_dir = Path(split_dir)
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        image_paths = sorted(
            p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        written = 0
        for img_path in tqdm(image_paths, desc=f"Augmenting {split_dir.name}"):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            img, bboxes, class_ids = self._load(img_path, lbl_path)

            for i in range(factor):
                augmented = self.transform(
                    image=img, bboxes=bboxes, class_labels=class_ids
                )
                aug_img = augmented["image"]
                aug_bboxes = augmented["bboxes"]
                aug_cls = augmented["class_labels"]

                out_stem = f"{img_path.stem}_aug{i:03d}"
                out_img = img_dir / f"{out_stem}.jpg"
                out_lbl = lbl_dir / f"{out_stem}.txt"

                cv2.imwrite(
                    str(out_img), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 90]
                )
                with open(out_lbl, "w") as f:
                    for cls_id, (cx, cy, w, h) in zip(aug_cls, aug_bboxes):
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                written += 1

        log.info(f"Augmentation complete — {written} new images written to {split_dir}")
        return written

    @staticmethod
    def _load(
        img_path: Path, lbl_path: Path
    ) -> Tuple[np.ndarray, List[Tuple], List[int]]:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes: List[Tuple] = []
        class_ids: List[int] = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    class_ids.append(int(parts[0]))
                    bboxes.append(tuple(map(float, parts[1:])))
        return img, bboxes, class_ids
