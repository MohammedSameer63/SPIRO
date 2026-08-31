#!/usr/bin/env python3
"""SPIRO ML — training/mlops/performance_monitor.py
Monitor inference performance in real time.

Usage
-----
    python training/mlops/performance_monitor.py report
    python training/mlops/performance_monitor.py export
"""
import argparse, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

def main():
    from lib.ml.mlops.monitoring.performance_monitor import PerformanceMonitor
    from lib.ml.core.logger import get_logger
    log = get_logger("perf_monitor")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("report")
    sub.add_parser("export")
    args = p.parse_args()

    monitor = PerformanceMonitor(enable_system_monitoring=False)
    if args.command == "report":
        print(json.dumps(monitor.report(), indent=2))
    elif args.command == "export":
        out = monitor.export_performance_report()
        log.info(f"Exported: {out}")

if __name__ == "__main__":
    main()
