"""Tests for ModelRegistry."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lib.ml.models.registry import ModelRegistry


@pytest.mark.unit
class TestModelRegistry:
    def test_register_and_get(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry")
        # Create a dummy weights file
        weights = tmp_path / "model.pt"
        weights.write_bytes(b"\x00" * 16)

        entry = reg.register(
            version="v1",
            architecture="yolov11",
            weights_path=weights,
            metrics={"mAP50": 0.85, "mAP50-95": 0.62},
            copy_weights=True,
        )
        assert entry["version"] == "v1"
        assert entry["metrics"]["mAP50"] == 0.85

        got = reg.get("v1")
        assert got["architecture"] == "yolov11"

    def test_list_versions(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry")
        for i in range(3):
            w = tmp_path / f"m{i}.pt"
            w.write_bytes(b"\x00")
            reg.register(f"v{i}", "yolov11", w, {"mAP50": 0.5 + i * 0.1}, copy_weights=False)
        versions = reg.list_versions()
        assert len(versions) == 3

    def test_promote(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry")
        for v in ["v1", "v2"]:
            w = tmp_path / f"{v}.pt"
            w.write_bytes(b"\x00")
            reg.register(v, "yolov11", w, {"mAP50": 0.8}, copy_weights=False)
        reg.promote("v2")
        prod = reg.get_production()
        assert prod["version"] == "v2"

    def test_promote_demotes_previous(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry")
        for v in ["v1", "v2"]:
            w = tmp_path / f"{v}.pt"
            w.write_bytes(b"\x00")
            reg.register(v, "yolov11", w, {"mAP50": 0.8}, copy_weights=False)
        reg.promote("v1")
        reg.promote("v2")
        assert reg.get("v1")["production"] is False
        assert reg.get("v2")["production"] is True

    def test_best_version(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry")
        for i, score in enumerate([0.7, 0.9, 0.8]):
            w = tmp_path / f"m{i}.pt"
            w.write_bytes(b"\x00")
            reg.register(f"v{i}", "yolov11", w, {"mAP50": score}, copy_weights=False)
        best = reg.best_version("mAP50")
        assert best["version"] == "v1"  # 0.9

    def test_delete_version(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry")
        w = tmp_path / "m.pt"
        w.write_bytes(b"\x00")
        reg.register("v1", "yolov11", w, {"mAP50": 0.8}, copy_weights=False)
        reg.delete("v1")
        with pytest.raises(KeyError):
            reg.get("v1")

    def test_overwrite_version(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry")
        w = tmp_path / "m.pt"
        w.write_bytes(b"\x00")
        reg.register("v1", "yolov11", w, {"mAP50": 0.5}, copy_weights=False)
        reg.register("v1", "yolov11", w, {"mAP50": 0.9}, copy_weights=False)
        assert reg.get("v1")["metrics"]["mAP50"] == 0.9

    def test_get_nonexistent_raises(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry")
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_no_production_returns_none(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry")
        assert reg.get_production() is None
