"""
SPIRO ML — VerifyDataset
PyTorch Dataset for the EfficientNetV2 verification pipeline.

Two modes:
  1. ImageFolder mode — loads from train/ val/ test/ class subdirectories
  2. Crop mode — loads from datasets/crops/<class_name>/ (patches from detection)

Also provides:
  - CropExtractor — extracts object crops from images using YOLO detections
  - WeightedSamplerBuilder — builds sampler to handle class imbalance
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from lib.ml.core.logger import get_logger
from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader
from lib.ml.verification.verify_config import VerifyConfig

log = get_logger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =============================================================================
# Albumentations augmentation pipelines
# =============================================================================

def build_train_transforms(cfg: VerifyConfig) -> Callable:
    """Return a callable augmentation pipeline for training."""
    aug = cfg.augmentation
    size = cfg.input_size

    alb_pipeline = A.Compose([
        A.RandomResizedCrop(
            height=size, width=size,
            scale=tuple(aug.crop_scale),
            ratio=tuple(aug.crop_ratio),
            p=1.0 if aug.random_crop else 0.0,
        ),
        A.HorizontalFlip(p=aug.horizontal_flip),
        A.VerticalFlip(p=aug.vertical_flip),
        A.Rotate(limit=aug.rotation_degrees, border_mode=cv2.BORDER_REFLECT_101, p=0.4),
        A.Perspective(scale=(0.0, aug.perspective_distortion), p=0.2),
        A.ColorJitter(
            brightness=aug.brightness,
            contrast=aug.contrast,
            saturation=aug.saturation,
            hue=aug.hue,
            p=aug.color_jitter_prob,
        ),
        A.ToGray(p=aug.grayscale_prob),
        A.CLAHE(clip_limit=aug.clahe_clip_limit, p=aug.clahe_prob),
        A.GaussianBlur(blur_limit=tuple(aug.blur_kernel), p=aug.blur_prob),
        A.GaussNoise(var_limit=tuple(aug.noise_var), p=aug.noise_prob),
        A.CoarseDropout(
            max_holes=aug.cutout_holes,
            max_height=aug.cutout_max_size,
            max_width=aug.cutout_max_size,
            p=aug.cutout_prob,
        ),
        A.Resize(height=size, width=size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    random_erase = transforms.RandomErasing(
        p=aug.random_erasing_prob,
        scale=tuple(aug.random_erasing_scale),
    )

    def transform(image: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        out = alb_pipeline(image=rgb)["image"]
        tensor = torch.from_numpy(out.transpose(2, 0, 1)).float()
        tensor = random_erase(tensor)
        return tensor

    return transform


def build_val_transforms(cfg: VerifyConfig) -> Callable:
    """Return validation/inference transforms (no augmentation)."""
    size = cfg.input_size
    pipeline = A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def transform(image: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        out = pipeline(image=rgb)["image"]
        return torch.from_numpy(out.transpose(2, 0, 1)).float()

    return transform


# =============================================================================
# Dataset
# =============================================================================

class VerifyDataset(Dataset):
    """
    Classification dataset for the EfficientNetV2 verifier.

    Expects structure:
        root/
          train/images/  + train/labels/  (YOLO format — class from label)
          val/images/    + val/labels/
          test/images/   + test/labels/

    Or crop mode:
        crops/
          plastic_bottle/
            img_001.jpg
          glass_bottle/
            img_002.jpg

    Parameters
    ----------
    root : Path
        Dataset root (with split subdirs) or crops root.
    split : str
        "train" | "val" | "test"
    transform : callable
        Image transform function.
    mode : str
        "yolo" (from images+labels) | "crop" (from class folders)
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        mode: str = "yolo",
        min_samples_per_class: int = 1,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.mode = mode
        self.taxonomy = TaxonomyLoader()
        self.class_names = self.taxonomy.all_names()
        self.num_classes = self.taxonomy.num_classes

        self.samples: List[Tuple[Path, int]] = []
        self._class_counts: Dict[int, int] = {}

        if mode == "yolo":
            self._load_yolo_split(min_samples_per_class)
        elif mode == "crop":
            self._load_crop_folders(min_samples_per_class)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'yolo' or 'crop'.")

        log.info(
            f"VerifyDataset [{split}] — {len(self.samples)} samples | "
            f"mode={mode} | "
            f"{len(self._class_counts)} classes populated"
        )

    def _load_yolo_split(self, min_samples: int) -> None:
        """Load from YOLO split — use dominant class per image as label."""
        split_dir = self.root / self.split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists():
            log.warning(f"images/ not found at {images_dir}")
            return

        from collections import Counter
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = labels_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                continue
            cls_counts: Counter = Counter()
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_counts[int(parts[0])] += 1
            if not cls_counts:
                continue
            primary = cls_counts.most_common(1)[0][0]
            if 0 <= primary < self.num_classes:
                self.samples.append((img_path, primary))
                self._class_counts[primary] = self._class_counts.get(primary, 0) + 1

    def _load_crop_folders(self, min_samples: int) -> None:
        """Load from ImageFolder-style crop directories."""
        split_dir = self.root / self.split if (self.root / self.split).exists() else self.root
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            try:
                cls_id = self.taxonomy.name_to_id(class_dir.name)
            except KeyError:
                log.debug(f"Unknown class dir: {class_dir.name}")
                continue
            images = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
            if len(images) < min_samples:
                continue
            for img_path in images:
                self.samples.append((img_path, cls_id))
                self._class_counts[cls_id] = self._class_counts.get(cls_id, 0) + 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((self.cfg_size, self.cfg_size, 3), dtype=np.uint8) if hasattr(self, "cfg_size") else np.zeros((224, 224, 3), dtype=np.uint8)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float() / 255.0
        return img, label

    def class_weights(self) -> torch.Tensor:
        """Compute per-class weights inversely proportional to frequency."""
        counts = torch.zeros(self.num_classes)
        for cls_id, cnt in self._class_counts.items():
            counts[cls_id] = cnt
        weights = torch.where(counts > 0, 1.0 / counts, torch.zeros_like(counts))
        weights = weights / weights.sum()
        return weights

    def sample_weights(self) -> List[float]:
        """Return per-sample weight for WeightedRandomSampler."""
        cw = self.class_weights()
        return [float(cw[label]) for _, label in self.samples]


# =============================================================================
# DataLoader factory
# =============================================================================

def build_dataloaders(
    cfg: VerifyConfig,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, test DataLoaders.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    t_tf = train_transform or build_train_transforms(cfg)
    v_tf = val_transform or build_val_transforms(cfg)

    root = Path(cfg.dataset.root)
    mode = "crop" if (Path(cfg.dataset.crop_dir).exists()) else "yolo"

    train_ds = VerifyDataset(root, "train", t_tf, mode=mode,
                              min_samples_per_class=cfg.dataset.min_samples_per_class)
    val_ds   = VerifyDataset(root, "val",   v_tf, mode=mode)
    test_ds  = VerifyDataset(root, "test",  v_tf, mode=mode)

    tr_cfg = cfg.training

    # Weighted sampler for class imbalance
    sampler = None
    if cfg.dataset.use_weighted_sampler and len(train_ds) > 0:
        sw = train_ds.sample_weights()
        sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=tr_cfg.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=tr_cfg.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tr_cfg.batch_size,
        shuffle=False,
        num_workers=tr_cfg.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=tr_cfg.batch_size,
        shuffle=False,
        num_workers=tr_cfg.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )

    log.info(
        f"DataLoaders — train:{len(train_ds)} val:{len(val_ds)} test:{len(test_ds)} "
        f"| sampler={'weighted' if sampler else 'sequential'}"
    )
    return train_loader, val_loader, test_loader


# =============================================================================
# Crop extractor (creates crop dataset from YOLO detections)
# =============================================================================

class CropExtractor:
    """
    Extracts object crop patches from images using YOLOv11 detections.
    Writes crops into datasets/crops/<class_name>/ for classifier training.

    Example
    -------
    >>> extractor = CropExtractor(output_dir=Path("datasets/crops"))
    >>> extractor.extract_from_split(
    ...     images_dir=Path("datasets/processed/train/images"),
    ...     labels_dir=Path("datasets/processed/train/labels"),
    ... )
    """

    def __init__(
        self,
        output_dir: Path,
        min_crop_size: int = 32,
        padding_fraction: float = 0.1,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_crop_size = min_crop_size
        self.padding = padding_fraction
        self.taxonomy = TaxonomyLoader()

    def extract_from_split(
        self,
        images_dir: Path,
        labels_dir: Path,
        split: str = "train",
    ) -> Dict[str, int]:
        """
        Extract crops from a YOLO-format split.

        Returns
        -------
        dict mapping class_name → number of crops extracted
        """
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        counts: Dict[str, int] = {}

        for img_path in sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        ):
            lbl_path = labels_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]

            with open(lbl_path) as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])

                    # Convert to pixel coords with padding
                    pad = self.padding
                    x1 = max(0, int((cx - w / 2 * (1 + pad)) * iw))
                    y1 = max(0, int((cy - h / 2 * (1 + pad)) * ih))
                    x2 = min(iw, int((cx + w / 2 * (1 + pad)) * iw))
                    y2 = min(ih, int((cy + h / 2 * (1 + pad)) * ih))

                    crop = img[y1:y2, x1:x2]
                    if crop.shape[0] < self.min_crop_size or crop.shape[1] < self.min_crop_size:
                        continue

                    class_name = self.taxonomy.id_to_name(cls_id)
                    out_dir = self.output_dir / split / class_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{img_path.stem}_box{i:03d}.jpg"
                    cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    counts[class_name] = counts.get(class_name, 0) + 1

        total = sum(counts.values())
        log.info(f"Extracted {total} crops from {split} split into {self.output_dir}")
        return counts
