"""Tests for training callbacks."""
import csv
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


def _make_trainer(epoch: int = 1) -> MagicMock:
    """Build a minimal mock trainer."""
    t = MagicMock()
    t.epoch = epoch
    t.metrics = {
        "metrics/mAP50(B)": 0.50,
        "metrics/mAP50-95(B)": 0.32,
        "metrics/precision(B)": 0.68,
        "metrics/recall(B)": 0.62,
        "val/box_loss": 1.2,
        "val/cls_loss": 0.8,
        "val/dfl_loss": 0.4,
    }
    t.tloss = None
    t.label_loss_items.return_value = {
        "train/box_loss": 1.5,
        "train/cls_loss": 0.9,
        "train/dfl_loss": 0.5,
    }
    t.optimizer = MagicMock()
    t.optimizer.param_groups = [{"lr": 0.01}]
    return t


class TestCSVLoggerCallback:
    def test_creates_csv_with_header(self, tmp_path):
        from lib.ml.training.callbacks.callbacks import CSVLoggerCallback
        csv_path = tmp_path / "history.csv"
        cb = CSVLoggerCallback(csv_path)
        t = _make_trainer(0)
        cb.on_train_epoch_start(t)
        cb.on_fit_epoch_end(t)
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "epoch" in header
        assert "mAP50" in header
        assert "mAP50-95" in header

    def test_appends_rows(self, tmp_path):
        from lib.ml.training.callbacks.callbacks import CSVLoggerCallback
        csv_path = tmp_path / "history.csv"
        cb = CSVLoggerCallback(csv_path)
        for ep in range(5):
            t = _make_trainer(ep)
            cb.on_train_epoch_start(t)
            cb.on_fit_epoch_end(t)

        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) == 5

    def test_epoch_column_correct(self, tmp_path):
        from lib.ml.training.callbacks.callbacks import CSVLoggerCallback
        import pandas as pd
        csv_path = tmp_path / "history.csv"
        cb = CSVLoggerCallback(csv_path)
        for ep in range(3):
            t = _make_trainer(ep)
            cb.on_train_epoch_start(t)
            cb.on_fit_epoch_end(t)
        df = pd.read_csv(csv_path)
        assert list(df["epoch"]) == [1, 2, 3]


class TestEarlyStoppingCallback:
    def test_no_stop_improving(self):
        from lib.ml.training.callbacks.callbacks import EarlyStoppingCallback
        cb = EarlyStoppingCallback(patience=5, min_delta=0.001)
        for i in range(10):
            t = _make_trainer(i)
            t.metrics["metrics/mAP50-95(B)"] = 0.3 + i * 0.01
            cb.on_val_end(t)
        assert not cb.stopped_early

    def test_stops_on_plateau(self):
        from lib.ml.training.callbacks.callbacks import EarlyStoppingCallback
        cb = EarlyStoppingCallback(patience=3, min_delta=0.001)
        # Initial improvement
        t0 = _make_trainer(0)
        t0.metrics["metrics/mAP50-95(B)"] = 0.5
        cb.on_val_end(t0)

        # No improvement for 3 epochs
        for i in range(4):
            t = _make_trainer(i + 1)
            t.metrics["metrics/mAP50-95(B)"] = 0.49
            cb.on_val_end(t)

        assert cb.stopped_early

    def test_disabled_when_patience_zero(self):
        from lib.ml.training.callbacks.callbacks import EarlyStoppingCallback
        cb = EarlyStoppingCallback(patience=0)
        for i in range(10):
            t = _make_trainer(i)
            t.metrics["metrics/mAP50-95(B)"] = 0.3
            cb.on_val_end(t)
        assert not cb.stopped_early

    def test_minimize_mode(self):
        from lib.ml.training.callbacks.callbacks import EarlyStoppingCallback
        cb = EarlyStoppingCallback(
            metric="val/box_loss", patience=3, min_delta=0.001, maximize=False
        )
        t0 = _make_trainer(0)
        t0.metrics["val/box_loss"] = 1.0
        cb.on_val_end(t0)
        for i in range(4):
            t = _make_trainer(i + 1)
            t.metrics["val/box_loss"] = 1.0  # no improvement
            cb.on_val_end(t)
        assert cb.stopped_early


class TestEpochSummaryCallback:
    def test_summary_does_not_raise(self):
        from lib.ml.training.callbacks.callbacks import EpochSummaryCallback
        cb = EpochSummaryCallback(total_epochs=100)
        t = _make_trainer(0)
        cb.on_train_epoch_start(t)
        cb.on_fit_epoch_end(t)  # Should not raise


class TestBuildCallbacks:
    def test_returns_dict_with_events(self, tmp_path):
        from lib.ml.training.callbacks.callbacks import build_callbacks
        cbs = build_callbacks(
            tb_log_dir=None,
            csv_path=tmp_path / "hist.csv",
            patience=5,
            total_epochs=10,
        )
        assert "on_fit_epoch_end" in cbs
        assert "on_val_end" in cbs
        assert "on_train_epoch_start" in cbs
        assert len(cbs["on_fit_epoch_end"]) > 0

    def test_no_tb_when_dir_is_none(self):
        from lib.ml.training.callbacks.callbacks import build_callbacks
        cbs = build_callbacks(tb_log_dir=None, csv_path=None)
        # Should still build without errors
        assert isinstance(cbs, dict)

    def test_all_handlers_callable(self, tmp_path):
        from lib.ml.training.callbacks.callbacks import build_callbacks
        cbs = build_callbacks(csv_path=tmp_path / "hist.csv", patience=5)
        for event, handlers in cbs.items():
            for h in handlers:
                assert callable(h)
