#!/usr/bin/env python3
"""
SPIRO ML — training/callbacks.py
Utility for listing, testing, and validating training callbacks.

Usage
-----
    # List all available callbacks
    python training/callbacks.py --list

    # Test CSV logger (writes to /tmp/test_training.csv)
    python training/callbacks.py --test-csv

    # Test TensorBoard logger
    python training/callbacks.py --test-tb --tb-dir /tmp/spiro_tb_test

    # Replay a training CSV to verify callback output
    python training/callbacks.py --replay logs/training_history.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPIRO Training Callback Utilities")
    p.add_argument("--list", action="store_true", help="List available callbacks")
    p.add_argument("--test-csv", action="store_true", help="Test CSV logger")
    p.add_argument("--csv-path", default="/tmp/spiro_test_training.csv")
    p.add_argument("--test-tb", action="store_true", help="Test TensorBoard logger")
    p.add_argument("--tb-dir", default="/tmp/spiro_tb_test")
    p.add_argument("--replay", type=str, default=None,
                   help="Replay metrics from a training CSV")
    return p.parse_args()


class MockTrainer:
    """Minimal trainer mock for testing callbacks."""
    def __init__(self, epoch: int = 0):
        self.epoch = epoch
        self.metrics = {
            "metrics/mAP50(B)": 0.45 + epoch * 0.002,
            "metrics/mAP50-95(B)": 0.28 + epoch * 0.001,
            "metrics/precision(B)": 0.65 + epoch * 0.001,
            "metrics/recall(B)": 0.60 + epoch * 0.001,
            "val/box_loss": 1.2 - epoch * 0.005,
            "val/cls_loss": 0.8 - epoch * 0.003,
            "val/dfl_loss": 0.5 - epoch * 0.002,
        }
        self.tloss = None

        class FakeOptimizer:
            param_groups = [{"lr": 0.01 * (0.99 ** epoch)}]
        self.optimizer = FakeOptimizer()

    def label_loss_items(self, loss, prefix="train"):
        return {
            f"{prefix}/box_loss": 1.5 - self.epoch * 0.005,
            f"{prefix}/cls_loss": 0.9 - self.epoch * 0.003,
            f"{prefix}/dfl_loss": 0.6 - self.epoch * 0.002,
        }


def main() -> None:
    args = parse_args()
    from lib.ml.core.logger import get_logger
    from lib.ml.training.callbacks.callbacks import (
        TensorBoardCallback, CSVLoggerCallback, EarlyStoppingCallback,
        EpochSummaryCallback, build_callbacks,
    )

    log = get_logger("callbacks")

    if args.list:
        print("\nAvailable SPIRO Training Callbacks:")
        print("-" * 50)
        callbacks_info = [
            ("TensorBoardCallback",  "on_fit_epoch_end", "Logs metrics to TensorBoard"),
            ("CSVLoggerCallback",    "on_fit_epoch_end", "Appends rows to training_history.csv"),
            ("EarlyStoppingCallback","on_val_end",       "Stops training on plateau"),
            ("EpochSummaryCallback", "on_fit_epoch_end", "Prints concise epoch summary"),
        ]
        for name, event, desc in callbacks_info:
            print(f"  {name:<28} [{event}] — {desc}")
        print()
        return

    if args.test_csv:
        log.info(f"Testing CSVLoggerCallback → {args.csv_path}")
        cb = CSVLoggerCallback(Path(args.csv_path))
        for epoch in range(5):
            trainer = MockTrainer(epoch)
            cb.on_train_epoch_start(trainer)
            cb.on_fit_epoch_end(trainer)
        log.info(f"CSV test complete. Check: {args.csv_path}")
        import pandas as pd
        df = pd.read_csv(args.csv_path)
        print(df.to_string())

    if args.test_tb:
        log.info(f"Testing TensorBoardCallback → {args.tb_dir}")
        cb = TensorBoardCallback(Path(args.tb_dir))
        for epoch in range(10):
            trainer = MockTrainer(epoch)
            cb.on_fit_epoch_end(trainer)
        cb.close()
        log.info(f"TensorBoard test complete. Run: tensorboard --logdir {args.tb_dir}")

    if args.replay:
        log.info(f"Replaying training metrics from {args.replay}")
        import pandas as pd
        df = pd.read_csv(args.replay)
        df.columns = [c.strip() for c in df.columns]
        print(f"\nLoaded {len(df)} epochs from {args.replay}")
        print("\nBest results:")
        for col in ["mAP50", "mAP50-95", "precision", "recall"]:
            if col in df.columns:
                best_val = df[col].max()
                best_epoch = df.loc[df[col].idxmax(), "epoch"]
                print(f"  {col:<15}: {best_val:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
