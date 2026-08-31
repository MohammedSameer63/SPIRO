"""Tests for ClassificationEvaluator (no GPU/model loading required)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lib.ml.evaluation.evaluator import ClassificationEvaluator


@pytest.mark.unit
class TestClassificationEvaluator:
    def test_perfect_predictions(self, base_config, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from lib.ml.core.config import ConfigManager
        cfg = ConfigManager.load(base_config)

        ev = ClassificationEvaluator(cfg)
        ev.reports_dir = tmp_path  # redirect output

        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        result = ev.evaluate(y_true, y_pred, save_report=True)
        assert result["accuracy"] == 1.0

    def test_random_predictions_accuracy(self, base_config, tmp_path):
        from lib.ml.core.config import ConfigManager
        cfg = ConfigManager.load(base_config)
        ev = ClassificationEvaluator(cfg)
        ev.reports_dir = tmp_path

        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 3, n).tolist()
        y_pred = np.random.randint(0, 3, n).tolist()
        result = ev.evaluate(y_true, y_pred, save_report=False)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_with_probabilities(self, base_config, tmp_path):
        from lib.ml.core.config import ConfigManager
        cfg = ConfigManager.load(base_config)
        ev = ClassificationEvaluator(cfg)
        ev.reports_dir = tmp_path

        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        y_proba = [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.85, 0.1, 0.05],
            [0.05, 0.9, 0.05],
        ]
        result = ev.evaluate(y_true, y_pred, y_proba=y_proba, save_report=True)
        assert "roc_auc_macro" in result
        assert 0.0 <= result["roc_auc_macro"] <= 1.0

    def test_report_json_written(self, base_config, tmp_path):
        from lib.ml.core.config import ConfigManager
        import json
        cfg = ConfigManager.load(base_config)
        ev = ClassificationEvaluator(cfg)
        ev.reports_dir = tmp_path

        ev.evaluate([0, 1, 2], [0, 1, 2], save_report=True)
        out = tmp_path / "classification_report.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert "accuracy" in data
        assert "classification_report" in data

    def test_confusion_matrix_plot_created(self, base_config, tmp_path):
        from lib.ml.core.config import ConfigManager
        cfg = ConfigManager.load(base_config)
        ev = ClassificationEvaluator(cfg)
        ev.reports_dir = tmp_path

        ev.evaluate([0, 1, 2, 0], [0, 1, 0, 0], save_report=True)
        assert (tmp_path / "classification_confusion_matrix.png").exists()
