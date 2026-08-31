"""
SPIRO ML — ConfigManager
Loads base_config.yaml, merges experiment overrides, resolves paths,
and exposes a dot-access namespace throughout the codebase.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf


class ConfigManager:
    """
    Singleton-style config loader.

    Usage
    -----
    cfg = ConfigManager.load("configs/base_config.yaml")
    cfg = ConfigManager.load("configs/experiments/yolov11_baseline.yaml")

    Access values:
        cfg.training.epochs
        cfg.paths.checkpoints_dir
    """

    _instance: Optional["ConfigManager"] = None
    _cfg: Optional[DictConfig] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Class-level factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        config_path: str | Path,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "ConfigManager":
        """
        Load a YAML config (with optional _base_ inheritance) and apply
        any runtime overrides supplied as a flat dict, e.g.:
            {"training.epochs": 200, "training.batch_size": 32}
        """
        config_path = Path(config_path)
        raw = cls._load_yaml_with_base(config_path)

        if overrides:
            for key, value in overrides.items():
                OmegaConf.update(raw, key, value, merge=True)

        instance = cls(raw)
        cls._instance = instance
        cls._seed_everything(raw.get("project", {}).get("seed", 42))
        return instance

    @classmethod
    def get(cls) -> "ConfigManager":
        """Return the current singleton instance (must call load first)."""
        if cls._instance is None:
            raise RuntimeError(
                "ConfigManager not initialised — call ConfigManager.load() first."
            )
        return cls._instance

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return getattr(self._cfg, name)
        except AttributeError:
            raise AttributeError(f"Config has no key '{name}'")

    def __getitem__(self, key: str) -> Any:
        return OmegaConf.select(self._cfg, key)

    def as_dict(self) -> dict:
        return OmegaConf.to_container(self._cfg, resolve=True)

    def to_yaml(self) -> str:
        return OmegaConf.to_yaml(self._cfg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml_with_base(path: Path) -> DictConfig:
        with open(path, "r") as f:
            raw_dict = yaml.safe_load(f) or {}

        base_key = "_base_"
        if base_key in raw_dict:
            base_rel = raw_dict.pop(base_key)
            base_path = (path.parent / base_rel).resolve()
            base_cfg = ConfigManager._load_yaml_with_base(base_path)
            override_cfg = OmegaConf.create(raw_dict)
            merged = OmegaConf.merge(base_cfg, override_cfg)
            return merged

        return OmegaConf.create(raw_dict)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    # ------------------------------------------------------------------
    # Path resolution helpers
    # ------------------------------------------------------------------

    def resolve_path(self, key: str, root: Optional[Path] = None) -> Path:
        """Return an absolute Path for a paths.* key, creating dir if needed."""
        rel = OmegaConf.select(self._cfg, f"paths.{key}")
        if rel is None:
            raise KeyError(f"paths.{key} not found in config")
        base = root or Path.cwd()
        p = (base / rel).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
