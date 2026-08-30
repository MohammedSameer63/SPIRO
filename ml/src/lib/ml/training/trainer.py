"""
SPIRO ML — SPIROYOLOTrainer
Production YOLOv11 trainer with:
  - All 5 model variants (n/s/m/l/x)
  - Multi-GPU support via Ultralytics DDP
  - Automatic Mixed Precision (AMP)
  - Gradient accumulation
  - EMA
  - Custom callbacks (TensorBoard, CSV, early stopping)
  - Automatic resume
  - Checkpoint management
  - ONNX export after training
  - MLflow experiment tracking
  - Full metrics validation per epoch
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from ultralytics import YOLO

from lib.ml.core.logger import get_logger
from lib.ml.training.callbacks.callbacks import build_callbacks
from lib.ml.training.checkpoints.checkpoint_manager import CheckpointManager
from lib.ml.training.experiment.tracker import ExperimentTracker
from lib.ml.training.experiment.plotter import ResultsPlotter
from lib.ml.training.training_config import TrainingConfig

log = get_logger(__name__)


class SPIROYOLOTrainer:
    """
    Complete YOLOv11 training pipeline for SPIRO.

    Parameters
    ----------
    config : TrainingConfig
        Loaded training configuration.

    Example
    -------
    >>> config = TrainingConfig.load("configs/training/yolo11s.yaml")
    >>> trainer = SPIROYOLOTrainer(config)
    >>> results = trainer.train()
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.cfg = config
        self.exp_name = config.experiment.name

        # Paths
        self.run_dir = (
            Path(config.checkpoint.dir) / self.exp_name
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.checkpoint_mgr = CheckpointManager(
            run_dir=self.run_dir,
            metric=config.checkpoint.metric,
            maximize=True,
        )
        self.tracker = ExperimentTracker(
            experiment_name=config.experiment.project,
            run_name=self.exp_name,
            tracking_uri=config.logging.log_dir + "/mlflow",
            tags=list(config.experiment.tags),
        )

        # State
        self._model: Optional[YOLO] = None
        self._results: Optional[Any] = None
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """
        Run the full training loop.

        Returns
        -------
        dict with final metrics, best checkpoint path, onnx path.
        """
        log.info("═" * 60)
        log.info(f"  SPIRO YOLOv11 Training — {self.exp_name}")
        log.info("═" * 60)
        self._log_system_info()

        self._start_time = time.perf_counter()

        # Resolve weights (resume or fresh)
        weights = self._resolve_weights()

        # Build Ultralytics training args
        train_args = self.cfg.build_ultralytics_args()
        train_args["resume"] = self.cfg.training.resume and (
            self.checkpoint_mgr.last_path() is not None
        )

        log.info(f"Weights:      {weights}")
        log.info(f"Dataset:      {train_args['data']}")
        log.info(f"Epochs:       {train_args['epochs']}")
        log.info(f"Batch:        {train_args['batch']}")
        log.info(f"Device:       {train_args['device']}")
        log.info(f"AMP:          {train_args['amp']}")
        log.info(f"Resume:       {train_args['resume']}")
        log.info(f"Output dir:   {self.run_dir}")

        # Start MLflow tracking
        flat_params = self._flatten_dict(self.cfg.as_dict())
        self.tracker.start(params=flat_params)

        # Register custom callbacks with Ultralytics
        self._model = YOLO(weights)
        self._register_callbacks()

        try:
            self._results = self._model.train(**train_args)
        except KeyboardInterrupt:
            log.warning("Training interrupted by user")
            self.tracker.set_tag("status", "INTERRUPTED")
        except Exception as e:
            log.error(f"Training failed: {e}")
            self.tracker.end(status="FAILED")
            raise

        # Post-training
        summary = self._post_training()
        self.tracker.end(status="FINISHED")
        return summary

    def validate(
        self,
        weights: Optional[Union[str, Path]] = None,
        split: str = "val",
    ) -> Dict[str, float]:
        """
        Run validation on best.pt (or specified weights).

        Parameters
        ----------
        weights : str | Path, optional
            Defaults to best.pt from this run.
        split : str
            "val" | "test"

        Returns
        -------
        dict with mAP50, mAP50-95, precision, recall, F1.
        """
        weights = weights or self.checkpoint_mgr.best_path()
        if weights is None:
            raise RuntimeError("No weights available for validation. Run train() first.")

        log.info(f"Validating {weights} on {split} split")
        model = YOLO(str(weights))
        metrics = model.val(
            data=str(Path(self.cfg.dataset.yaml).resolve()),
            split=split,
            imgsz=self.cfg.model.input_size,
            conf=self.cfg.validation.conf_threshold,
            iou=self.cfg.validation.iou_threshold,
            device=self.cfg.training.device,
            plots=self.cfg.validation.plots,
            save_json=self.cfg.validation.save_json,
            verbose=False,
        )

        result = {
            "mAP50":     float(metrics.box.map50),
            "mAP50-95":  float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall":    float(metrics.box.mr),
            "f1":        self._compute_f1(
                float(metrics.box.mp), float(metrics.box.mr)
            ),
        }

        # Per-class AP
        if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
            class_names = list(self.cfg.dataset.get("class_names", []) or [])
            per_class: Dict[str, float] = {}
            for i, cls_idx in enumerate(metrics.box.ap_class_index):
                name = (class_names[cls_idx]
                        if cls_idx < len(class_names)
                        else str(cls_idx))
                if i < len(metrics.box.ap50):
                    per_class[name] = float(metrics.box.ap50[i])
            result["per_class_AP50"] = per_class

        log.info(
            f"Validation: mAP50={result['mAP50']:.4f} "
            f"mAP50-95={result['mAP50-95']:.4f} "
            f"P={result['precision']:.4f} "
            f"R={result['recall']:.4f} "
            f"F1={result['f1']:.4f}"
        )
        return result

    # ------------------------------------------------------------------
    # Post-training
    # ------------------------------------------------------------------

    def _post_training(self) -> Dict[str, Any]:
        """Run post-training tasks: plot, export, register."""
        elapsed = time.perf_counter() - self._start_time
        log.info(f"Training complete in {elapsed/3600:.2f}h")

        summary: Dict[str, Any] = {
            "experiment": self.exp_name,
            "elapsed_seconds": round(elapsed, 1),
            "run_dir": str(self.run_dir),
        }

        # Locate best.pt produced by Ultralytics
        ult_best = self.run_dir / "weights" / "best.pt"
        ult_last = self.run_dir / "weights" / "last.pt"

        best_path = ult_best if ult_best.exists() else None
        last_path = ult_last if ult_last.exists() else None

        if best_path:
            summary["best_weights"] = str(best_path)
            log.info(f"Best weights: {best_path}")

            # Final validation metrics
            try:
                val_metrics = self.validate(weights=best_path, split="val")
                summary["val_metrics"] = val_metrics
                self.tracker.log_metrics(val_metrics, step=self.cfg.training.epochs)
            except Exception as e:
                log.warning(f"Post-training validation failed: {e}")

            # ONNX export
            if self.cfg.export.auto_export_onnx:
                onnx_path = self._export_onnx(best_path)
                if onnx_path:
                    summary["onnx_path"] = str(onnx_path)
                    self.tracker.log_artifact(onnx_path)

        # Generate plots
        csv_path = Path(self.cfg.logging.csv_path)
        if csv_path.exists():
            try:
                plotter = ResultsPlotter(
                    csv_path=csv_path,
                    run_dir=self.run_dir,
                    output_dir=Path("reports/training") / self.exp_name,
                )
                plotter.generate_all()
                self.tracker.log_artifacts_dir(
                    Path("reports/training") / self.exp_name
                )
            except Exception as e:
                log.warning(f"Plot generation failed: {e}")

        # Save summary JSON
        summary_path = self.run_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        self.tracker.log_artifact(summary_path)

        log.info(f"Training summary: {summary_path}")
        return summary

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def _export_onnx(self, weights_path: Path) -> Optional[Path]:
        """Export best.pt to ONNX and validate."""
        log.info("Exporting best model to ONNX...")
        try:
            from lib.ml.export.onnx_exporter import ONNXExporter
            from lib.ml.core.config import ConfigManager

            # Build a minimal ConfigManager-compatible namespace
            # ONNXExporter only needs cfg.paths.exports_dir and cfg.export.*
            exports_dir = Path("models/exports")
            exports_dir.mkdir(parents=True, exist_ok=True)

            model = YOLO(str(weights_path))
            onnx_result = model.export(
                format="onnx",
                imgsz=self.cfg.model.input_size,
                opset=self.cfg.export.opset,
                dynamic=self.cfg.export.dynamic,
                simplify=self.cfg.export.simplify,
                half=self.cfg.export.half,
            )

            if onnx_result is None:
                log.warning("ONNX export returned None")
                return None

            onnx_src = Path(str(onnx_result))
            if not onnx_src.exists():
                log.warning(f"ONNX file not found at {onnx_src}")
                return None

            # Move to exports dir
            out_name = f"{self.exp_name}_best.onnx"
            onnx_dst = exports_dir / out_name
            shutil.copy2(onnx_src, onnx_dst)
            log.info(f"ONNX exported: {onnx_dst}")

            # Verify
            if self.cfg.export.verify:
                self._verify_onnx(onnx_dst, weights_path)

            return onnx_dst
        except Exception as e:
            log.error(f"ONNX export failed: {e}")
            return None

    def _verify_onnx(self, onnx_path: Path, pt_path: Path) -> bool:
        """
        Verify ONNX inference matches PyTorch on a random input.
        Logs mismatch if detected but does not raise.
        """
        try:
            import numpy as np
            import onnxruntime as ort

            imgsz = self.cfg.model.input_size
            if isinstance(imgsz, int):
                imgsz = (imgsz, imgsz)

            dummy = np.random.rand(1, 3, imgsz[1], imgsz[0]).astype(np.float32)

            # ORT
            sess = ort.InferenceSession(
                str(onnx_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            ort_out = sess.run(None, {sess.get_inputs()[0].name: dummy})[0]

            # PyTorch
            model = YOLO(str(pt_path))
            import torch
            with torch.no_grad():
                pt_out = model.model(
                    torch.from_numpy(dummy).to("cpu")
                )[0].cpu().numpy()

            max_diff = float(np.abs(ort_out - pt_out).max()) if ort_out.shape == pt_out.shape else -1
            if max_diff < 0:
                log.warning(
                    f"ONNX/PyTorch output shapes differ: "
                    f"ORT={ort_out.shape} PT={pt_out.shape}"
                )
                return False
            if max_diff > 1e-2:
                log.warning(f"ONNX/PyTorch max diff = {max_diff:.6f} (threshold 0.01)")
            else:
                log.info(f"ONNX verification passed (max_diff={max_diff:.6f}) ✓")
            return max_diff <= 1e-2
        except Exception as e:
            log.warning(f"ONNX verification failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _register_callbacks(self) -> None:
        """Register custom callbacks with the Ultralytics YOLO model."""
        cfg = self.cfg
        callbacks = build_callbacks(
            tb_log_dir=(
                Path(cfg.logging.tensorboard_dir) / self.exp_name
                if cfg.logging.tensorboard else None
            ),
            csv_path=(
                Path(cfg.logging.csv_path)
                if cfg.logging.csv else None
            ),
            patience=cfg.training.patience,
            total_epochs=cfg.training.epochs,
            metric="metrics/mAP50-95(B)",
        )

        for event, handlers in callbacks.items():
            for handler in handlers:
                self._model.add_callback(event, handler)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_weights(self) -> str:
        """Determine starting weights (resume or pretrained)."""
        cfg = self.cfg.training

        if cfg.resume:
            # Explicit checkpoint
            if cfg.resume_checkpoint:
                p = Path(cfg.resume_checkpoint)
                if p.exists():
                    log.info(f"Resuming from explicit checkpoint: {p}")
                    return str(p)
                log.warning(f"Explicit checkpoint not found: {p}")

            # Auto-detect last.pt
            last = self.checkpoint_mgr.auto_resume()
            if last:
                return str(last)

        # Fresh start
        return str(self.cfg.model.weights)

    def _log_system_info(self) -> None:
        log.info(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                log.info(
                    f"GPU {i}: {props.name} "
                    f"({props.total_memory / 1e9:.1f} GB)"
                )
        else:
            log.info("No GPU detected — training on CPU")

    @staticmethod
    def _compute_f1(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)

    @staticmethod
    def _flatten_dict(d: Dict, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dict for MLflow param logging."""
        result: Dict[str, Any] = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(SPIROYOLOTrainer._flatten_dict(v, key))
            else:
                result[key] = str(v)[:490]
        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_weights(self) -> Optional[Path]:
        return self.checkpoint_mgr.best_path()

    @property
    def last_weights(self) -> Optional[Path]:
        return self.checkpoint_mgr.last_path()
