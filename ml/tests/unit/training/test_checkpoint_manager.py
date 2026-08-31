"""Tests for CheckpointManager."""
import json
import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


def _write_fake_pt(path: Path) -> Path:
    """Write a minimal fake .pt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": 0, "state_dict": {}}, str(path))
    return path


class TestCheckpointManager:
    def test_init_creates_weights_dir(self, tmp_path):
        from lib.ml.training.checkpoints.checkpoint_manager import CheckpointManager
        cm = CheckpointManager(tmp_path / "run")
        assert (tmp_path / "run" / "weights").exists()

    def test_save_last(self, tmp_path):
        from lib.ml.training.checkpoints.checkpoint_manager import CheckpointManager
        run = tmp_path / "run"
        src = tmp_path / "epoch.pt"
        _write_fake_pt(src)

        cm = CheckpointManager(run)
        dst = cm.save_last(src, epoch=5, metrics={"mAP50-95": 0.5})
        assert dst.exists()
        assert dst.name == "last.pt"
        meta = json.loads((run / "checkpoint_meta.json").read_text())
        assert meta["last_epoch"] == 5

    def test_save_best_improves(self, tmp_path):
        from lib.ml.training.checkpoints.checkpoint_manager import CheckpointManager
        run = tmp_path / "run"
        cm = CheckpointManager(run, metric="mAP50-95", maximize=True)
        src = tmp_path / "model.pt"
        _write_fake_pt(src)

        _, improved = cm.save_best(src, epoch=1, metrics={"mAP50-95": 0.4})
        assert improved is True
        assert (run / "weights" / "best.pt").exists()

    def test_save_best_no_improvement(self, tmp_path):
        from lib.ml.training.checkpoints.checkpoint_manager import CheckpointManager
        run = tmp_path / "run"
        cm = CheckpointManager(run, metric="mAP50-95")
        src = tmp_path / "m.pt"
        _write_fake_pt(src)

        cm.save_best(src, epoch=1, metrics={"mAP50-95": 0.5})
        _, improved = cm.save_best(src, epoch=2, metrics={"mAP50-95": 0.3})
        assert improved is False

    def test_save_best_tracks_best_value(self, tmp_path):
        from lib.ml.training.checkpoints.checkpoint_manager import CheckpointManager
        run = tmp_path / "run"
        cm = CheckpointManager(run)
        src = tmp_path / "m.pt"
        _write_fake_pt(src)

        cm.save_best(src, 1, {"mAP50-95": 0.3})
        cm.save_best(src, 2, {"mAP50-95": 0.6})
        cm.save_best(src, 3, {"mAP50-95": 0.5})
        assert abs(cm._best_value - 0.6) < 1e-9

    def test_periodic_checkpoint_saved(self, tmp_path):
        from lib.ml.training.checkpoints.checkpoint_manager import CheckpointManager
        run = tmp_path / "run"
        cm = CheckpointManager(run)
        src = tmp_path / "m.pt"
        _write_fake_pt(src)

        cm.save_periodic(src, epoch=50)
        assert (run / "weights" / "epoch_00050.pt").exists()

    def test_periodic_pruning(self, tmp_path):
        from lib.ml.training.checkpoints.checkpoint_manager import CheckpointManager
        run = tmp_path / "run"
        cm = CheckpointManager(run, keep_last_n=2)
        src = tmp_path / "m.pt"
        _write_fake_pt(src)

        for ep in [10, 20, 30]:
            cm.save_periodic(src, epoch=ep)

        # Only last 2 should remain
        remaining = list((run / "weights").glob("epoch_*.pt"))
        assert len(remaining) == 2
        names = {p.name for p in remaining}
        assert "epoch_00020.pt" in names
        assert "epoch_00030.pt" in names

    def test_auto_resume_finds_last(self, tmp_path):
        from lib.ml.training.checkpoints.checkpoint_manager import CheckpointManager
        run = tmp_path / "run"
        cm = CheckpointManager(run)
        src = tmp_path / "m.pt"
        _write_fake_pt(src)
        cm.save_last(src, epoch=10, metrics={})

        resumed = cm.auto_resume()
        assert resumed is not None
        assert resumed.name == "last.pt"

    def test_auto_resume_no_last_returns_none(self, tmp_path):
        from lib.ml.training.checkpoints.checkpoint_manager import CheckpointManager
        cm = CheckpointManager(tmp_path / "fresh")
        assert cm.auto_resume() is None

    def test_best_path_none_when_empty(self, tmp_path):
        from lib.ml.training.checkpoints.checkpoint_manager import CheckpointManager
        cm = CheckpointManager(tmp_path / "empty")
        assert cm.best_path() is None

    def test_meta_persists_across_instances(self, tmp_path):
        from lib.ml.training.checkpoints.checkpoint_manager import CheckpointManager
        run = tmp_path / "run"
        src = tmp_path / "m.pt"
        _write_fake_pt(src)

        cm1 = CheckpointManager(run)
        cm1.save_best(src, epoch=5, metrics={"mAP50-95": 0.7})

        cm2 = CheckpointManager(run)
        assert abs(cm2._best_value - 0.7) < 1e-9

    def test_summary_string(self, tmp_path):
        from lib.ml.training.checkpoints.checkpoint_manager import CheckpointManager
        cm = CheckpointManager(tmp_path / "run")
        s = cm.summary()
        assert isinstance(s, str)
        assert "best" in s.lower()
