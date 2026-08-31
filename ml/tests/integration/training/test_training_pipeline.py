"""
Integration tests for the YOLOv11 training pipeline.
Ultralytics YOLO.train() is mocked to avoid actual training.
Tests verify the full orchestration: config → trainer → checkpoint → export → report.
"""
import csv
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


def _make_mock_results(tmp_path: Path) -> MagicMock:
    """Build a mock Ultralytics Results object with realistic structure."""
    m = MagicMock()
    m.results_dict = {
        "metrics/mAP50(B)": 0.55,
        "metrics/mAP50-95(B)": 0.35,
        "metrics/precision(B)": 0.70,
        "metrics/recall(B)": 0.65,
    }
    m.box = MagicMock()
    m.box.map50 = 0.55
    m.box.map = 0.35
    m.box.mp = 0.70
    m.box.mr = 0.65
    m.box.ap_class_index = [0, 1, 2]
    m.box.ap50 = [0.60, 0.50, 0.55]
    return m


def _write_fake_best_pt(run_dir: Path) -> Path:
    """Write a fake best.pt to the expected Ultralytics location."""
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    torch.save({"model": None, "epoch": 2}, str(best))
    torch.save({"model": None, "epoch": 2}, str(last))
    return best


@pytest.mark.integration
class TestSPIROYOLOTrainerIntegration:
    def test_trainer_init(self, training_config_path, tmp_path):
        """Trainer initialises correctly with a valid config."""
        from lib.ml.training.training_config import TrainingConfig
        from lib.ml.training.trainer import SPIROYOLOTrainer
        from omegaconf import OmegaConf

        cfg = TrainingConfig.load(
            training_config_path,
            overrides={"checkpoint.dir": str(tmp_path / "ckpt")}
        )
        trainer = SPIROYOLOTrainer(cfg)
        assert trainer.exp_name == "test_run"
        assert trainer.checkpoint_mgr is not None
        assert trainer.tracker is not None

    def test_trainer_train_mock(self, training_config_path, tmp_path):
        """Full train() call with mocked YOLO — verifies orchestration."""
        from lib.ml.training.training_config import TrainingConfig
        from lib.ml.training.trainer import SPIROYOLOTrainer
        from omegaconf import OmegaConf

        ckpt_dir = tmp_path / "ckpt"
        cfg = TrainingConfig.load(
            training_config_path,
            overrides={
                "checkpoint.dir": str(ckpt_dir),
                "logging.csv": False,
                "logging.tensorboard": False,
                "export.auto_export_onnx": False,
            }
        )
        trainer = SPIROYOLOTrainer(cfg)
        mock_results = _make_mock_results(tmp_path)

        with patch("lib.ml.training.trainer.YOLO") as MockYOLO:
            mock_model = MagicMock()
            MockYOLO.return_value = mock_model
            mock_model.train.return_value = mock_results

            # Write fake best.pt so post-training steps find it
            run_dir = ckpt_dir / "test_run"
            _write_fake_best_pt(run_dir)

            with patch.object(trainer, "validate", return_value={
                "mAP50": 0.55, "mAP50-95": 0.35,
                "precision": 0.70, "recall": 0.65, "f1": 0.67,
            }):
                with patch.object(trainer.tracker, "start"):
                    with patch.object(trainer.tracker, "end"):
                        summary = trainer.train()

        assert isinstance(summary, dict)
        assert "experiment" in summary
        assert "elapsed_seconds" in summary

    def test_resume_resolves_last_pt(self, training_config_path, tmp_path):
        """When resume=True and last.pt exists, it is used as weights."""
        from lib.ml.training.training_config import TrainingConfig
        from lib.ml.training.trainer import SPIROYOLOTrainer

        ckpt_dir = tmp_path / "ckpt"
        cfg = TrainingConfig.load(
            training_config_path,
            overrides={
                "checkpoint.dir": str(ckpt_dir),
                "training.resume": True,
            }
        )
        trainer = SPIROYOLOTrainer(cfg)

        # Pre-create last.pt
        run_dir = ckpt_dir / "test_run"
        _write_fake_best_pt(run_dir)
        trainer.checkpoint_mgr.run_dir = run_dir
        trainer.checkpoint_mgr.weights_dir = run_dir / "weights"
        # Simulate metadata for auto_resume
        trainer.checkpoint_mgr._meta["last_epoch"] = 10

        weights = trainer._resolve_weights()
        # Should prefer last.pt
        assert "last.pt" in str(weights)

    def test_validate_after_training(self, training_config_path, tmp_path):
        """validate() returns correctly structured metrics dict."""
        from lib.ml.training.training_config import TrainingConfig
        from lib.ml.training.trainer import SPIROYOLOTrainer

        ckpt_dir = tmp_path / "ckpt"
        cfg = TrainingConfig.load(
            training_config_path,
            overrides={"checkpoint.dir": str(ckpt_dir)}
        )
        trainer = SPIROYOLOTrainer(cfg)
        run_dir = ckpt_dir / "test_run"
        _write_fake_best_pt(run_dir)

        mock_metrics = MagicMock()
        mock_metrics.box.map50 = 0.55
        mock_metrics.box.map = 0.35
        mock_metrics.box.mp = 0.70
        mock_metrics.box.mr = 0.65
        mock_metrics.box.ap_class_index = [0, 1]
        mock_metrics.box.ap50 = [0.60, 0.50]

        with patch("lib.ml.training.trainer.YOLO") as MockYOLO:
            mock_yolo = MagicMock()
            MockYOLO.return_value = mock_yolo
            mock_yolo.val.return_value = mock_metrics

            result = trainer.validate(
                weights=run_dir / "weights" / "best.pt",
                split="val",
            )

        assert "mAP50" in result
        assert "mAP50-95" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert abs(result["mAP50"] - 0.55) < 1e-4
        assert abs(result["f1"] - 0.6741) < 0.01

    def test_checkpoint_saved_by_manager(self, training_config_path, tmp_path):
        """CheckpointManager tracks best metric correctly during simulated training."""
        from lib.ml.training.training_config import TrainingConfig
        from lib.ml.training.checkpoints.checkpoint_manager import CheckpointManager

        run_dir = tmp_path / "test_run"
        cm = CheckpointManager(run_dir, metric="mAP50-95")

        # Simulate training epochs
        epochs_metrics = [
            {"mAP50-95": 0.20},
            {"mAP50-95": 0.32},
            {"mAP50-95": 0.45},  # best
            {"mAP50-95": 0.40},
            {"mAP50-95": 0.38},
        ]
        fake_pt = tmp_path / "model.pt"
        torch.save({}, str(fake_pt))

        for ep, metrics in enumerate(epochs_metrics, 1):
            cm.save_last(fake_pt, epoch=ep, metrics=metrics)
            cm.save_best(fake_pt, epoch=ep, metrics=metrics)

        assert abs(cm._best_value - 0.45) < 1e-6
        meta = cm.get_meta()
        assert meta["best_epoch"] == 3

    def test_csv_logger_records_all_epochs(self, training_config_path, tmp_path):
        """CSVLoggerCallback records one row per epoch."""
        from lib.ml.training.callbacks.callbacks import CSVLoggerCallback
        import pandas as pd

        csv_path = tmp_path / "history.csv"
        cb = CSVLoggerCallback(csv_path)

        class MockTrainer:
            def __init__(self, ep):
                self.epoch = ep
                self.metrics = {
                    "metrics/mAP50(B)": 0.4 + ep * 0.01,
                    "metrics/mAP50-95(B)": 0.25 + ep * 0.005,
                    "metrics/precision(B)": 0.65,
                    "metrics/recall(B)": 0.60,
                    "val/box_loss": 1.5 - ep * 0.02,
                    "val/cls_loss": 0.8 - ep * 0.01,
                    "val/dfl_loss": 0.5 - ep * 0.01,
                }
                self.tloss = None
                self.optimizer = MagicMock()
                self.optimizer.param_groups = [{"lr": 0.01}]

            def label_loss_items(self, loss, prefix="train"):
                return {"train/box_loss": 1.5, "train/cls_loss": 0.8, "train/dfl_loss": 0.5}

        for ep in range(10):
            t = MockTrainer(ep)
            cb.on_train_epoch_start(t)
            cb.on_fit_epoch_end(t)

        df = pd.read_csv(csv_path)
        assert len(df) == 10
        assert list(df["epoch"]) == list(range(1, 11))

    def test_post_training_summary_json(self, training_config_path, tmp_path):
        """Training summary JSON is written correctly."""
        from lib.ml.training.training_config import TrainingConfig
        from lib.ml.training.trainer import SPIROYOLOTrainer

        ckpt_dir = tmp_path / "ckpt"
        cfg = TrainingConfig.load(
            training_config_path,
            overrides={
                "checkpoint.dir": str(ckpt_dir),
                "export.auto_export_onnx": False,
                "logging.tensorboard": False,
                "logging.csv": False,
            }
        )
        trainer = SPIROYOLOTrainer(cfg)
        run_dir = ckpt_dir / "test_run"
        _write_fake_best_pt(run_dir)

        with patch.object(trainer, "validate", return_value={
            "mAP50": 0.55, "mAP50-95": 0.35,
            "precision": 0.70, "recall": 0.65, "f1": 0.67,
        }):
            summary = trainer._post_training()

        assert isinstance(summary, dict)
        assert "experiment" in summary

        # Check summary JSON file
        summary_path = run_dir / "training_summary.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert data["experiment"] == "test_run"
