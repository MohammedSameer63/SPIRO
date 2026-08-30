"""
SPIRO ML — PipelineConfig
Loads and validates the production inference pipeline configuration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf

from lib.ml.core.logger import get_logger

log = get_logger(__name__)

_DEFAULT = Path("configs/pipeline/inference_pipeline.yaml")


class PipelineConfig:
    """
    Loads the inference pipeline YAML with optional key overrides.

    Example
    -------
    >>> cfg = PipelineConfig.load()
    >>> cfg.models.yolo_onnx
    >>> cfg.detection.conf_threshold
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    @classmethod
    def load(
        cls,
        path: Union[str, Path] = _DEFAULT,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "PipelineConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        cfg = OmegaConf.create(raw)
        if overrides:
            for k, v in overrides.items():
                OmegaConf.update(cfg, k, v, merge=True)
        instance = cls(cfg)
        instance._validate()
        return instance

    def _validate(self) -> None:
        required = ["models.num_classes", "detection.conf_threshold",
                    "detection.iou_threshold", "fusion.method"]
        missing = [f for f in required if OmegaConf.select(self._cfg, f) is None]
        if missing:
            raise ValueError(f"PipelineConfig missing: {missing}")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return getattr(self._cfg, name)
        except AttributeError:
            raise AttributeError(f"PipelineConfig has no key '{name}'")

    def get(self, key: str, default: Any = None) -> Any:
        return OmegaConf.select(self._cfg, key, default=default)

    def as_dict(self) -> Dict:
        return OmegaConf.to_container(self._cfg, resolve=True)
