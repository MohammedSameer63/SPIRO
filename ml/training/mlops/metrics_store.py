#!/usr/bin/env python3
"""SPIRO ML — training/mlops/metrics_store.py
Query and export time-series metrics.

Usage
-----
    python training/mlops/metrics_store.py trend --model-id yolov11s --metric mAP50_95
    python training/mlops/metrics_store.py health --model-id yolov11s
    python training/mlops/metrics_store.py export
"""
import argparse, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

def main():
    from lib.ml.mlops.monitoring.metrics_store import MetricsStore
    from lib.ml.core.logger import get_logger
    log = get_logger("metrics_store")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("trend")
    tr.add_argument("--model-id", required=True)
    tr.add_argument("--metric", required=True)
    tr.add_argument("--last-n", type=int, default=20)
    hl = sub.add_parser("health")
    hl.add_argument("--model-id", required=True)
    sub.add_parser("export")
    args = p.parse_args()

    store = MetricsStore()
    if args.command == "trend":
        t = store.trend(args.model_id, args.metric, last_n=args.last_n)
        print(json.dumps(t, indent=2))
    elif args.command == "health":
        r = store.health_report(args.model_id)
        print(json.dumps(r, indent=2))
    elif args.command == "export":
        out = store.export_performance_report()
        log.info(f"Exported: {out}")

if __name__ == "__main__":
    main()
