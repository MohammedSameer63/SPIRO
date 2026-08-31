#!/usr/bin/env python3
"""
SPIRO ML — training/experiment.py
Manage and compare training experiments.

Usage
-----
    # List all experiments in a checkpoint directory
    python training/experiment.py list

    # Show details of a specific experiment
    python training/experiment.py show --name spiro_yolo11s

    # Compare two experiments
    python training/experiment.py compare --names spiro_yolo11n spiro_yolo11s

    # Generate plots for a completed experiment
    python training/experiment.py plot --name spiro_yolo11s

    # Export best model for an experiment
    python training/experiment.py export --name spiro_yolo11s
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPIRO Experiment Manager")
    sub = p.add_subparsers(dest="command", required=True)

    # list
    ls = sub.add_parser("list", help="List all experiments")
    ls.add_argument("--checkpoint-dir", default="models/checkpoints")

    # show
    show = sub.add_parser("show", help="Show experiment details")
    show.add_argument("--name", required=True)
    show.add_argument("--checkpoint-dir", default="models/checkpoints")

    # compare
    cmp = sub.add_parser("compare", help="Compare experiments side by side")
    cmp.add_argument("--names", nargs="+", required=True)
    cmp.add_argument("--checkpoint-dir", default="models/checkpoints")

    # plot
    plot = sub.add_parser("plot", help="Generate plots for an experiment")
    plot.add_argument("--name", required=True)
    plot.add_argument("--checkpoint-dir", default="models/checkpoints")
    plot.add_argument("--csv", default=None, help="Path to training_history.csv")

    # export
    exp_cmd = sub.add_parser("export", help="Export best model to ONNX")
    exp_cmd.add_argument("--name", required=True)
    exp_cmd.add_argument("--checkpoint-dir", default="models/checkpoints")
    exp_cmd.add_argument("--config", default="configs/training/yolov11_training.yaml")

    return p.parse_args()


def _load_summary(run_dir: Path) -> dict:
    s = run_dir / "training_summary.json"
    if s.exists():
        with open(s) as f:
            return json.load(f)
    meta = run_dir / "checkpoint_meta.json"
    if meta.exists():
        with open(meta) as f:
            return json.load(f)
    return {}


def main() -> None:
    args = parse_args()
    from lib.ml.core.logger import get_logger
    log = get_logger("experiment")

    ckpt_dir = Path(args.checkpoint_dir)

    if args.command == "list":
        experiments = [d for d in sorted(ckpt_dir.iterdir()) if d.is_dir()]
        if not experiments:
            log.info(f"No experiments found in {ckpt_dir}")
            return
        print(f"\n{'Name':<35} {'Best mAP50-95':>14} {'Epochs':>8} {'Status'}")
        print("-" * 70)
        for exp_dir in experiments:
            summary = _load_summary(exp_dir)
            val_m = summary.get("val_metrics", {})
            best = val_m.get("mAP50-95", summary.get("best_value", "N/A"))
            epochs = summary.get("elapsed_seconds", "?")
            has_best = (exp_dir / "weights" / "best.pt").exists()
            status = "✓ best.pt" if has_best else "in-progress"
            best_str = f"{best:.4f}" if isinstance(best, float) else str(best)
            print(f"  {exp_dir.name:<33} {best_str:>14} {'?':>8}  {status}")
        print()

    elif args.command == "show":
        run_dir = ckpt_dir / args.name
        if not run_dir.exists():
            log.error(f"Experiment not found: {run_dir}")
            sys.exit(1)
        summary = _load_summary(run_dir)
        print(f"\n{'='*55}")
        print(f"  Experiment: {args.name}")
        print(f"{'='*55}")
        for k, v in summary.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for k2, v2 in v.items():
                    if not isinstance(v2, dict):
                        print(f"    {k2}: {v2}")
            else:
                print(f"  {k}: {v}")
        # Check artefacts
        print("\n  Artefacts:")
        for fname in ["weights/best.pt", "weights/last.pt"]:
            p = run_dir / fname
            size = f"{p.stat().st_size/1e6:.1f}MB" if p.exists() else "missing"
            print(f"    {fname}: {size}")
        onnx_dir = Path("models/exports")
        for onnx in sorted(onnx_dir.glob(f"{args.name}*.onnx")):
            print(f"    {onnx}: {onnx.stat().st_size/1e6:.1f}MB")
        print()

    elif args.command == "compare":
        rows = []
        for name in args.names:
            run_dir = ckpt_dir / name
            summary = _load_summary(run_dir)
            val_m = summary.get("val_metrics", {})
            rows.append({
                "name": name,
                "mAP50":     val_m.get("mAP50",     "N/A"),
                "mAP50-95":  val_m.get("mAP50-95",  "N/A"),
                "precision": val_m.get("precision",  "N/A"),
                "recall":    val_m.get("recall",     "N/A"),
                "f1":        val_m.get("f1",         "N/A"),
                "time_h":    round(summary.get("elapsed_seconds", 0) / 3600, 2),
            })

        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)

        print(f"\n{'Metric':<15}", end="")
        for r in rows:
            print(f"  {r['name']:<22}", end="")
        print()
        print("-" * (15 + 24 * len(rows)))
        for metric in ["mAP50", "mAP50-95", "precision", "recall", "f1", "time_h"]:
            print(f"  {metric:<13}", end="")
            for r in rows:
                print(f"  {_fmt(r[metric]):<22}", end="")
            print()
        print()

    elif args.command == "plot":
        csv_path = Path(args.csv) if args.csv else Path("logs/training_history.csv")
        run_dir = ckpt_dir / args.name

        from lib.ml.training.experiment.plotter import ResultsPlotter
        plotter = ResultsPlotter(
            csv_path=csv_path,
            run_dir=run_dir if run_dir.exists() else None,
            output_dir=Path("reports/training") / args.name,
        )
        outputs = plotter.generate_all()
        log.info(f"Generated {len(outputs)} plots for {args.name}")

    elif args.command == "export":
        run_dir = ckpt_dir / args.name
        best = run_dir / "weights" / "best.pt"
        if not best.exists():
            log.error(f"best.pt not found: {best}")
            sys.exit(1)

        from lib.ml.training.training_config import TrainingConfig
        from lib.ml.training.trainer import SPIROYOLOTrainer

        config = TrainingConfig.load(args.config)
        trainer = SPIROYOLOTrainer(config)
        onnx_path = trainer._export_onnx(best)
        if onnx_path:
            log.info(f"Exported: {onnx_path}")
        else:
            log.error("Export failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
