"""
SPIRO ML — VerifyConfig
Loads EfficientNetV2 verification pipeline config with _base_ inheritance.
Validates all required fields and resolves variant-specific defaults.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from lib.ml.core.logger import get_logger

log = get_logger(__name__)

_DEFAULT_CONFIG = Path("configs/verification/effnetv2_verify.yaml")

# timm model strings per variant
_VARIANT_TIMM_MAP = {
    "efficientnetv2_b0": "tf_efficientnetv2_b0",
    "efficientnetv2_b1": "tf_efficientnetv2_b1",
    "efficientnetv2_b2": "tf_efficientnetv2_b2",
    "efficientnetv2_b3": "tf_efficientnetv2_b3",
    "efficientnetv2_s":  "tf_efficientnetv2_s",
    "efficientnetv2_m":  "tf_efficientnetv2_m",
    "efficientnetv2_l":  "tf_efficientnetv2_l",
}

_VARIANT_INPUT_SIZE = {
    "efficientnetv2_b0": 192,
    "efficientnetv2_b1": 240,
    "efficientnetv2_b2": 260,
    "efficientnetv2_b3": 300,
    "efficientnetv2_s":  300,
    "efficientnetv2_m":  384,
    "efficientnetv2_l":  480,
}

_REQUIRED_FIELDS = [
    "experiment.name",
    "model.variant",
    "model.num_classes",
    "training.epochs",
    "training.batch_size",
]


class VerifyConfig:
    """
    Loads and validates EfficientNetV2 verification config.

    Example
    -------
    >>> cfg = VerifyConfig.load("configs/verification/effnetv2_s.yaml")
    >>> cfg.model.variant      # "efficientnetv2_s"
    >>> cfg.model.timm_name    # "tf_efficientnetv2_s"
    >>> cfg.model.input_size   # 300
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    @classmethod
    def load(
        cls,
        config_path: Union[str, Path] = _DEFAULT_CONFIG,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "VerifyConfig":
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        raw = cls._load_with_base(config_path)

        if overrides:
            for key, value in overrides.items():
                OmegaConf.update(raw, key, value, merge=True)

        # Auto-resolve variant defaults
        variant = OmegaConf.select(raw, "model.variant", default="efficientnetv2_s")
        if OmegaConf.select(raw, "model.timm_name") is None:
            timm_name = _VARIANT_TIMM_MAP.get(variant, f"tf_{variant}")
            OmegaConf.update(raw, "model.timm_name", timm_name)
        if OmegaConf.select(raw, "model.input_size") is None:
            size = _VARIANT_INPUT_SIZE.get(variant, 224)
            OmegaConf.update(raw, "model.input_size", size)

        instance = cls(raw)
        instance._validate()
        cls._seed_everything(OmegaConf.select(raw, "experiment.seed", default=42))
        return instance

    @staticmethod
    def _load_with_base(path: Path) -> DictConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if "_base_" in data:
            base_rel = data.pop("_base_")
            base_path = (path.parent / base_rel).resolve()
            base = VerifyConfig._load_with_base(base_path)
            return OmegaConf.merge(base, OmegaConf.create(data))
        return OmegaConf.create(data)

    def _validate(self) -> None:
        missing = [f for f in _REQUIRED_FIELDS
                   if OmegaConf.select(self._cfg, f) is None]
        if missing:
            raise ValueError(f"VerifyConfig missing required fields: {missing}")
        variant = self._cfg.model.variant
        if variant not in _VARIANT_TIMM_MAP:
            raise ValueError(
                f"Unknown variant {variant!r}. "
                f"Valid: {list(_VARIANT_TIMM_MAP)}"
            )

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return getattr(self._cfg, name)
        except AttributeError:
            raise AttributeError(f"VerifyConfig has no key '{name}'")

    def get(self, key: str, default: Any = None) -> Any:
        return OmegaConf.select(self._cfg, key, default=default)

    def as_dict(self) -> Dict:
        return OmegaConf.to_container(self._cfg, resolve=True)

    def to_yaml(self) -> str:
        return OmegaConf.to_yaml(self._cfg)

    @property
    def timm_name(self) -> str:
        return self._cfg.model.timm_name

    @property
    def input_size(self) -> int:
        return int(self._cfg.model.input_size)
