"""
Integration test: EfficientNet → ONNX export → ONNXRuntime verification.
Uses untrained model (pretrained=False) so no weights download required.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


@pytest.mark.integration
class TestONNXExportPipeline:
    def test_efficientnet_export_and_verify(self, base_config, tmp_path):
        from lib.ml.core.config import ConfigManager
        from lib.ml.models.efficientnet_model import SPIROClassifier
        from lib.ml.export.onnx_exporter import ONNXExporter

        cfg = ConfigManager.load(base_config)

        # Build tiny export dir
        import yaml
        from omegaconf import OmegaConf
        # Override exports dir to tmp_path
        cfg._cfg = OmegaConf.merge(cfg._cfg, {"paths": {"exports_dir": str(tmp_path)}})

        model = SPIROClassifier(cfg)
        model.eval()

        exporter = ONNXExporter(cfg)
        # Patch simplify to off (onnxsim may not be installed in CI)
        exporter.exp_cfg = OmegaConf.merge(
            exporter.exp_cfg, {"simplify": False, "dynamic_axes": False}
        )

        input_shape = (1, 3, 128, 128)
        onnx_path = exporter.export_efficientnet(
            model,
            output_name="test_classifier.onnx",
            input_shape=input_shape,
            half=False,
        )

        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

        # Verify with ORT
        ok = exporter.verify(onnx_path, input_shape=input_shape, torch_model=model)
        assert ok is True

    def test_exported_onnx_is_valid(self, base_config, tmp_path):
        import onnx
        from lib.ml.core.config import ConfigManager
        from lib.ml.models.efficientnet_model import SPIROClassifier
        from lib.ml.export.onnx_exporter import ONNXExporter
        from omegaconf import OmegaConf

        cfg = ConfigManager.load(base_config)
        cfg._cfg = OmegaConf.merge(cfg._cfg, {"paths": {"exports_dir": str(tmp_path)}})

        model = SPIROClassifier(cfg)
        exporter = ONNXExporter(cfg)
        exporter.exp_cfg = OmegaConf.merge(
            exporter.exp_cfg, {"simplify": False, "dynamic_axes": False}
        )

        path = exporter.export_efficientnet(
            model, output_name="valid_test.onnx", input_shape=(1, 3, 128, 128)
        )

        onnx_model = onnx.load(str(path))
        onnx.checker.check_model(onnx_model)  # raises if invalid

    def test_metadata_embedded(self, base_config, tmp_path):
        import onnx
        from lib.ml.core.config import ConfigManager
        from lib.ml.models.efficientnet_model import SPIROClassifier
        from lib.ml.export.onnx_exporter import ONNXExporter
        from omegaconf import OmegaConf

        cfg = ConfigManager.load(base_config)
        cfg._cfg = OmegaConf.merge(cfg._cfg, {"paths": {"exports_dir": str(tmp_path)}})
        model = SPIROClassifier(cfg)
        exporter = ONNXExporter(cfg)
        exporter.exp_cfg = OmegaConf.merge(
            exporter.exp_cfg, {"simplify": False, "dynamic_axes": False}
        )

        path = exporter.export_efficientnet(
            model, output_name="meta_test.onnx", input_shape=(1, 3, 128, 128)
        )
        onnx_model = onnx.load(str(path))
        meta_keys = {p.key for p in onnx_model.metadata_props}
        assert "architecture" in meta_keys
        assert "num_classes" in meta_keys
        assert "exported_at" in meta_keys
