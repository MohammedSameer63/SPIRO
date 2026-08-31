"""
SPIRO ML — TrainingConfig
Loads training YAML configs with _base_ inheritance, validates all fields,
and provides a clean typed interface for the trainer.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from lib.ml.core.logger import get_logger

log = get_logger(__name__)

_DEFAULT_TRAINING_CONFIG = Path("configs/training/yolov11_training.yaml")


class TrainingConfig:
    """
    Loads and validates a SPIRO training YAML config.

    Supports _base_ inheritance (same pattern as Prompt 1 ConfigManager).
    All fields are accessible via dot notation.

    Example
    -------
    >>> tc = TrainingConfig.load("configs/training/yolo11s.yaml")
    >>> tc.model.weights        # "yolo11s.pt"
    >>> tc.training.epochs      # 300
    >>> tc.optimizer.lr0        # 0.01
    """

    _REQUIRED_FIELDS = [
        "experiment.name",
        "model.weights",
        "model.num_classes",
        "training.epochs",
        "training.batch_size",
        "dataset.yaml",
    ]

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        config_path: Union[str, Path] = _DEFAULT_TRAINING_CONFIG,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "TrainingConfig":
        """
        Load a training YAML, apply _base_ inheritance, validate,
        and apply any CLI overrides.

        Parameters
        ----------
        config_path : str | Path
        overrides : dict, optional
            Flat dot-notation overrides, e.g. {"training.epochs": 50}
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Training config not found: {config_path}")

        raw = cls._load_with_base(config_path)

        if overrides:
            for key, value in overrides.items():
                OmegaConf.update(raw, key, value, merge=True)

        instance = cls(raw)
        instance._validate()
        cls._seed_everything(OmegaConf.select(raw, "experiment.seed", default=42))
        return instance

    @staticmethod
    def _load_with_base(path: Path) -> DictConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        base_key = "_base_"
        if base_key in data:
            base_rel = data.pop(base_key)
            base_path = (path.parent / base_rel).resolve()
            base_cfg = TrainingConfig._load_with_base(base_path)
            override_cfg = OmegaConf.create(data)
            return OmegaConf.merge(base_cfg, override_cfg)
        return OmegaConf.create(data)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Raise ValueError for any missing required field."""
        missing = []
        for field in self._REQUIRED_FIELDS:
            val = OmegaConf.select(self._cfg, field)
            if val is None:
                missing.append(field)
        if missing:
            raise ValueError(f"Training config missing required fields: {missing}")

        # Validate variant
        valid_weights = {"yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"}
        w = self._cfg.model.weights
        if w not in valid_weights and not Path(w).exists():
            log.warning(
                f"model.weights={w!r} is not a standard variant and does not exist as a file. "
                "Ultralytics will attempt to download it."
            )

        # Validate num_classes
        nc = self._cfg.model.num_classes
        if nc < 1 or nc > 1000:
            raise ValueError(f"model.num_classes={nc} out of range [1, 1000]")

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return getattr(self._cfg, name)
        except AttributeError:
            raise AttributeError(f"TrainingConfig has no key '{name}'")

    def get(self, key: str, default: Any = None) -> Any:
        return OmegaConf.select(self._cfg, key, default=default)

    def as_dict(self) -> Dict:
        return OmegaConf.to_container(self._cfg, resolve=True)

    def to_yaml(self) -> str:
        return OmegaConf.to_yaml(self._cfg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def build_ultralytics_args(self) -> Dict[str, Any]:
        """
        Convert this config into the kwargs dict accepted by YOLO.train().
        Every supported Ultralytics train() parameter is mapped here.
        """
        aug = self._cfg.augmentation
        opt = self._cfg.optimizer
        sched = self._cfg.scheduler
        tr = self._cfg.training
        val = self._cfg.validation
        ckpt = self._cfg.checkpoint
        exp = self._cfg.experiment

        args: Dict[str, Any] = {
            # Dataset
            "data": str(Path(self._cfg.dataset.yaml).resolve()),
            "imgsz": self._cfg.model.input_size,
            "workers": self._cfg.dataset.workers,
            "cache": self._cfg.dataset.cache,
            "rect": self._cfg.dataset.rect,
            "single_cls": self._cfg.dataset.single_cls,

            # Training loop
            "epochs": tr.epochs,
            "batch": tr.batch_size,
            "amp": tr.amp,
            "patience": tr.patience,
            "save_period": tr.save_period,
            "val": True,
            "plots": val.plots,

            # Device
            "device": self._resolve_device(tr.device),

            # Optimizer
            "optimizer": opt.name,
            "lr0": opt.lr0,
            "lrf": opt.lrf,
            "momentum": opt.momentum,
            "weight_decay": opt.weight_decay,
            "warmup_epochs": opt.warmup_epochs,
            "warmup_momentum": opt.warmup_momentum,
            "warmup_bias_lr": opt.warmup_bias_lr,
            "nbs": opt.nbs,

            # Loss
            "box": self._cfg.loss.box,
            "cls": self._cfg.loss.cls,
            "dfl": self._cfg.loss.dfl,

            # Augmentation
            "hsv_h": aug.hsv_h,
            "hsv_s": aug.hsv_s,
            "hsv_v": aug.hsv_v,
            "degrees": aug.degrees,
            "translate": aug.translate,
            "scale": aug.scale,
            "shear": aug.shear,
            "perspective": aug.perspective,
            "flipud": aug.flipud,
            "fliplr": aug.fliplr,
            "mosaic": aug.mosaic if aug.enabled else 0.0,
            "mixup": aug.mixup if aug.enabled else 0.0,
            "copy_paste": aug.copy_paste if aug.enabled else 0.0,
            "label_smoothing": aug.label_smoothing,
            "erasing": aug.erasing,
            "crop_fraction": aug.crop_fraction,

            # Validation
            "conf": val.conf_threshold,
            "iou": val.iou_threshold,
            "max_det": val.max_det,
            "save_json": val.save_json,

            # Checkpoint output
            "project": ckpt.dir,
            "name": exp.name,
            "exist_ok": True,
            "save": True,
            "verbose": self._cfg.logging.verbose,
            "seed": exp.seed,
            "deterministic": tr.deterministic,
            "benchmark": tr.benchmark,

            # EMA
            "close_mosaic": 10,  # disable mosaic last N epochs
        }

        # Resume
        if tr.resume:
            args["resume"] = True

        return args

    @staticmethod
    def _resolve_device(device_str: str) -> Union[str, int, List[int]]:
        """Resolve 'auto' to the best available device string."""
        if device_str != "auto":
            return device_str
        if torch.cuda.is_available():
            return 0  # first GPU
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
