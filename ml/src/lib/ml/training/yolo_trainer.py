"""
SPIRO ML — YOLOTrainer
Full training loop for YOLOv11 using Ultralytics' train() API,
extended with SPIRO-specific callbacks, TensorBoard, and MLflow logging.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import mlflow
import torch
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO
from ultralytics.utils.callbacks.base import add_integration_callbacks

from lib.ml.core.config import ConfigManager
from lib.ml.core.device import resolve_device
from lib.ml.core.logger import get_logger
from lib.ml.models.registry import ModelRegistry

log = get_logger(__name__)


class YOLOTrainer:
    """
    Manages the full YOLOv11 training lifecycle.

    Parameters
    ----------
    cfg : ConfigManager
        Loaded project config.
    experiment_name : str, optional
        MLflow experiment name override.

    Example
    -------
    >>> trainer = YOLOTrainer(cfg)
    >>> results = trainer.train()
    >>> trainer.save_best()
    """

    def __init__(
        self,
        cfg: ConfigManager,
        experiment_name: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.training.device)
        self.tb_dir = Path(cfg.paths.tensorboard_dir)
        self.ckpt_dir = Path(cfg.paths.checkpoints_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self._exp_name = (
            experiment_name
            or getattr(getattr(cfg, "experiment", None), "name", "spiro_yolo")
        )

        self.writer: Optional[SummaryWriter] = None
        self._results: Optional[Any] = None
        self._best_weights: Optional[Path] = None

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train(self) -> Any:
        """
        Run the full training loop.

        Returns
        -------
        Ultralytics Results object.
        """
        cfg = self.cfg
        t_cfg = cfg.training
        m_cfg = cfg.model

        log.info(f"Starting YOLOv11 training — experiment: {self._exp_name}")

        # TensorBoard
        if cfg.logging.tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.tb_dir / self._exp_name))
            log.info(f"TensorBoard at: {self.tb_dir / self._exp_name}")

        # MLflow
        mlflow_active = cfg.logging.mlflow
        if mlflow_active:
            mlflow.set_tracking_uri(str(Path(cfg.paths.mlflow_dir).resolve()))
            mlflow.set_experiment(self._exp_name)
            mlflow.start_run(run_name=self._exp_name)
            mlflow.log_params(self._flatten_cfg(cfg))

        try:
            model = YOLO(m_cfg.pretrained_weights if m_cfg.pretrained else m_cfg.variant)

            train_args = dict(
                data=str(Path("configs/dataset.yaml").resolve()),
                epochs=t_cfg.epochs,
                batch=t_cfg.batch_size,
                imgsz=m_cfg.input_size[0],
                lr0=t_cfg.learning_rate,
                lrf=0.01,
                momentum=t_cfg.momentum,
                weight_decay=t_cfg.weight_decay,
                warmup_epochs=t_cfg.warmup_epochs,
                warmup_momentum=t_cfg.warmup_momentum,
                optimizer=t_cfg.optimizer,
                amp=t_cfg.amp,
                patience=t_cfg.patience,
                save_period=t_cfg.save_period,
                val=True,
                plots=True,
                device=str(self.device) if self.device.type != "cpu" else "cpu",
                project=str(self.ckpt_dir),
                name=self._exp_name,
                exist_ok=True,
                # Loss weights
                box=cfg.loss.box,
                cls=cfg.loss.cls,
                dfl=cfg.loss.dfl,
                # Augmentation
                hsv_h=cfg.augmentation.hue_shift / 360.0,
                hsv_s=cfg.augmentation.sat_shift / 100.0,
                hsv_v=cfg.augmentation.val_shift / 100.0,
                degrees=cfg.augmentation.rotate_limit,
                scale=cfg.augmentation.scale_limit,
                mosaic=cfg.augmentation.mosaic,
                mixup=cfg.augmentation.mixup,
                copy_paste=cfg.augmentation.copy_paste,
                flipud=cfg.augmentation.vertical_flip,
                fliplr=cfg.augmentation.horizontal_flip,
                workers=t_cfg.workers,
                seed=cfg.project.seed,
                verbose=True,
            )

            if t_cfg.resume and t_cfg.resume_checkpoint:
                train_args["resume"] = True
                model = YOLO(t_cfg.resume_checkpoint)
                log.info(f"Resuming from {t_cfg.resume_checkpoint}")

            self._results = model.train(**train_args)

            # Best weights location
            run_dir = self.ckpt_dir / self._exp_name
            best = run_dir / "weights" / "best.pt"
            if best.exists():
                self._best_weights = best

            # Log final metrics to MLflow
            if mlflow_active and self._results:
                self._log_results_to_mlflow(self._results)

        finally:
            if mlflow_active:
                mlflow.end_run()
            if self.writer:
                self.writer.close()

        log.info("Training complete")
        return self._results

    # ------------------------------------------------------------------
    # Post-training helpers
    # ------------------------------------------------------------------

    def validate(self) -> Dict[str, float]:
        """
        Run validation on the best checkpoint.
        Returns a flat dict of metrics.
        """
        if not self._best_weights:
            raise RuntimeError("No best weights found — run train() first.")
        model = YOLO(str(self._best_weights))
        metrics = model.val(
            data=str(Path("configs/dataset.yaml").resolve()),
            imgsz=self.cfg.model.input_size[0],
            conf=self.cfg.evaluation.conf_threshold,
            iou=self.cfg.evaluation.iou_threshold,
            device=str(self.device) if self.device.type != "cpu" else "cpu",
            plots=True,
        )
        result = {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
        log.info(f"Validation metrics: {result}")
        return result

    def save_best(self, dst: Optional[Union[str, Path]] = None) -> Path:
        """Copy the best.pt to a custom destination."""
        if not self._best_weights:
            raise RuntimeError("No best weights available.")
        dst = Path(dst) if dst else self.ckpt_dir / "best_final.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(self._best_weights, dst)
        log.info(f"Best weights saved to {dst}")
        return dst

    @property
    def best_weights(self) -> Optional[Path]:
        return self._best_weights

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_cfg(cfg: ConfigManager) -> Dict[str, Any]:
        """Flatten nested config to a single-level dict for MLflow."""
        flat = {}
        raw = cfg.as_dict()

        def _recurse(d: dict, prefix: str = "") -> None:
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _recurse(v, key)
                elif isinstance(v, (list, tuple)):
                    flat[key] = str(v)
                else:
                    flat[key] = v

        _recurse(raw)
        return flat

    @staticmethod
    def _log_results_to_mlflow(results: Any) -> None:
        try:
            mlflow.log_metric("mAP50", float(results.results_dict.get("metrics/mAP50(B)", 0)))
            mlflow.log_metric("mAP50-95", float(results.results_dict.get("metrics/mAP50-95(B)", 0)))
            mlflow.log_metric("precision", float(results.results_dict.get("metrics/precision(B)", 0)))
            mlflow.log_metric("recall", float(results.results_dict.get("metrics/recall(B)", 0)))
        except Exception as e:
            log.warning(f"MLflow metric logging failed: {e}")
