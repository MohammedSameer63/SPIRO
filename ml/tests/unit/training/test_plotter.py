"""Tests for ResultsPlotter."""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


def _write_fake_csv(path: Path, n_epochs: int = 20) -> Path:
    """Write a realistic training_history.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ep in range(1, n_epochs + 1):
        rows.append({
            "epoch": ep,
            "train/box_loss": max(0.1, 2.0 - ep * 0.08),
            "train/cls_loss": max(0.05, 1.0 - ep * 0.04),
            "train/dfl_loss": max(0.05, 0.8 - ep * 0.03),
            "val/box_loss":   max(0.15, 2.1 - ep * 0.08),
            "val/cls_loss":   max(0.06, 1.1 - ep * 0.04),
            "val/dfl_loss":   max(0.06, 0.9 - ep * 0.03),
            "precision": min(0.95, 0.5 + ep * 0.02),
            "recall":    min(0.90, 0.45 + ep * 0.02),
            "mAP50":     min(0.92, 0.3 + ep * 0.03),
            "mAP50-95":  min(0.65, 0.2 + ep * 0.02),
            "lr": 0.01 * (0.99 ** ep),
            "time_s": 45.0,
            "gpu_mem_GB": 4.2,
        })

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


class TestResultsPlotter:
    def test_load_csv(self, tmp_path):
        from lib.ml.training.experiment.plotter import ResultsPlotter
        csv_path = _write_fake_csv(tmp_path / "hist.csv")
        plotter = ResultsPlotter(csv_path, output_dir=tmp_path / "plots")
        plotter._df = plotter._load_csv()
        assert plotter._df is not None
        assert len(plotter._df) == 20
        assert "mAP50" in plotter._df.columns

    def test_load_missing_csv_returns_none(self, tmp_path):
        from lib.ml.training.experiment.plotter import ResultsPlotter
        plotter = ResultsPlotter(
            tmp_path / "nonexistent.csv",
            output_dir=tmp_path / "plots",
        )
        assert plotter._load_csv() is None

    def test_plot_loss_curves(self, tmp_path):
        from lib.ml.training.experiment.plotter import ResultsPlotter
        csv_path = _write_fake_csv(tmp_path / "hist.csv")
        plotter = ResultsPlotter(csv_path, output_dir=tmp_path / "plots")
        plotter._df = plotter._load_csv()
        out = plotter.plot_loss_curves()
        assert out.exists()
        assert out.suffix == ".png"

    def test_plot_map_curve(self, tmp_path):
        from lib.ml.training.experiment.plotter import ResultsPlotter
        csv_path = _write_fake_csv(tmp_path / "hist.csv")
        plotter = ResultsPlotter(csv_path, output_dir=tmp_path / "plots")
        plotter._df = plotter._load_csv()
        out = plotter.plot_map_curve()
        assert out.exists()

    def test_plot_precision_recall(self, tmp_path):
        from lib.ml.training.experiment.plotter import ResultsPlotter
        csv_path = _write_fake_csv(tmp_path / "hist.csv")
        plotter = ResultsPlotter(csv_path, output_dir=tmp_path / "plots")
        plotter._df = plotter._load_csv()
        out = plotter.plot_precision_recall()
        assert out.exists()

    def test_plot_results_dashboard(self, tmp_path):
        from lib.ml.training.experiment.plotter import ResultsPlotter
        csv_path = _write_fake_csv(tmp_path / "hist.csv")
        plotter = ResultsPlotter(csv_path, output_dir=tmp_path / "plots")
        plotter._df = plotter._load_csv()
        out = plotter.plot_results_dashboard()
        assert out.exists()

    def test_generate_all_creates_multiple_plots(self, tmp_path):
        from lib.ml.training.experiment.plotter import ResultsPlotter
        csv_path = _write_fake_csv(tmp_path / "hist.csv")
        plotter = ResultsPlotter(csv_path, output_dir=tmp_path / "plots")
        outputs = plotter.generate_all()
        assert len(outputs) >= 4
        for p in outputs:
            assert Path(p).exists()

    def test_output_dir_created(self, tmp_path):
        from lib.ml.training.experiment.plotter import ResultsPlotter
        csv_path = _write_fake_csv(tmp_path / "hist.csv")
        out_dir = tmp_path / "deep" / "nested" / "plots"
        plotter = ResultsPlotter(csv_path, output_dir=out_dir)
        assert out_dir.exists()

    def test_handles_empty_csv_gracefully(self, tmp_path):
        from lib.ml.training.experiment.plotter import ResultsPlotter
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("epoch,mAP50\n")  # header only, no data
        plotter = ResultsPlotter(csv_path, output_dir=tmp_path / "plots")
        # Should not raise, but output list may be empty
        outputs = plotter.generate_all()
        assert isinstance(outputs, list)
