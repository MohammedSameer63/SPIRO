"""
SPIRO ML — HyperparameterSweep
Lightweight grid/random search over training hyperparameters.
Runs multiple training experiments with different settings and
produces a comparison report.

Example
-------
>>> sweep = HyperparameterSweep(base_config="configs/training/yolo11n.yaml")
>>> sweep.add_param("optimizer.lr0", [0.005, 0.01, 0.02])
>>> sweep.add_param("training.batch_size", [16, 32])
>>> results = sweep.run(n_trials=4, mode="random")
"""
from __future__ import annotations

import itertools
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lib.ml.core.logger import get_logger
from lib.ml.training.training_config import TrainingConfig
from lib.ml.training.trainer import SPIROYOLOTrainer

log = get_logger(__name__)


class HyperparameterSweep:
    """
    Grid or random hyperparameter sweep over a base training config.

    Parameters
    ----------
    base_config : str | Path
        Path to base training YAML config.
    output_dir : Path
        Where to write sweep results.
    """

    def __init__(
        self,
        base_config: str | Path,
        output_dir: Path = Path("reports/hparam_sweep"),
    ) -> None:
        self.base_config = Path(base_config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._params: Dict[str, List[Any]] = {}
        self._results: List[Dict[str, Any]] = []

    def add_param(self, key: str, values: List[Any]) -> "HyperparameterSweep":
        """Add a hyperparameter to sweep."""
        self._params[key] = values
        return self

    def _generate_configs_grid(self) -> List[Dict[str, Any]]:
        """Generate all combinations (grid search)."""
        keys = list(self._params.keys())
        value_lists = [self._params[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]

    def _generate_configs_random(self, n: int, seed: int = 42) -> List[Dict[str, Any]]:
        """Generate N random combinations."""
        random.seed(seed)
        configs = []
        for _ in range(n):
            trial = {k: random.choice(v) for k, v in self._params.items()}
            configs.append(trial)
        return configs

    def run(
        self,
        mode: str = "grid",
        n_trials: int = 8,
        seed: int = 42,
        dry_run: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Execute the sweep.

        Parameters
        ----------
        mode : str
            "grid" (all combinations) or "random" (N random samples).
        n_trials : int
            Number of trials for random mode.
        seed : int
            Random seed for reproducibility.
        dry_run : bool
            If True, print configs without running training.

        Returns
        -------
        list of result dicts per trial.
        """
        if mode == "grid":
            trial_configs = self._generate_configs_grid()
        else:
            trial_configs = self._generate_configs_random(n_trials, seed)

        log.info(
            f"Hyperparameter sweep: {len(trial_configs)} trials | mode={mode}"
        )

        for i, overrides in enumerate(trial_configs, 1):
            trial_name = f"sweep_trial_{i:03d}"
            overrides["experiment.name"] = trial_name

            log.info(f"Trial {i}/{len(trial_configs)}: {overrides}")

            if dry_run:
                self._results.append({"trial": trial_name, "overrides": overrides})
                continue

            try:
                cfg = TrainingConfig.load(self.base_config, overrides=overrides)
                trainer = SPIROYOLOTrainer(cfg)
                summary = trainer.train()
                metrics = summary.get("val_metrics", {})
                result = {
                    "trial": trial_name,
                    "overrides": overrides,
                    "metrics": metrics,
                    "status": "completed",
                }
            except Exception as e:
                log.error(f"Trial {trial_name} failed: {e}")
                result = {
                    "trial": trial_name,
                    "overrides": overrides,
                    "metrics": {},
                    "status": f"failed: {e}",
                }

            self._results.append(result)
            self._save_results()

        self._save_results()
        self._print_summary()
        return self._results

    def _save_results(self) -> None:
        out = self.output_dir / "sweep_results.json"
        with open(out, "w") as f:
            json.dump(self._results, f, indent=2)

    def _print_summary(self) -> None:
        if not self._results:
            return
        log.info("\n" + "=" * 55)
        log.info("  Hyperparameter Sweep Summary")
        log.info("=" * 55)
        completed = [r for r in self._results if r["status"] == "completed"]
        if not completed:
            log.info("  No completed trials")
            return

        # Sort by mAP50-95
        completed.sort(key=lambda r: r["metrics"].get("mAP50-95", 0), reverse=True)
        log.info(f"{'Trial':<22} {'mAP50-95':>10} {'mAP50':>8} {'Config'}")
        log.info("-" * 70)
        for r in completed:
            m = r["metrics"]
            cfg_str = " | ".join(f"{k.split('.')[-1]}={v}" for k, v in r["overrides"].items()
                                  if k != "experiment.name")
            log.info(
                f"  {r['trial']:<20} {m.get('mAP50-95', 0):>10.4f} "
                f"{m.get('mAP50', 0):>8.4f}  {cfg_str}"
            )
        best = completed[0]
        log.info(f"\n  Best trial: {best['trial']} (mAP50-95={best['metrics'].get('mAP50-95', 0):.4f})")
        log.info("=" * 55)
