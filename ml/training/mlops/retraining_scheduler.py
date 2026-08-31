#!/usr/bin/env python3
"""SPIRO ML — training/mlops/retraining_scheduler.py
Manage the retraining job queue.

Usage
-----
    python training/mlops/retraining_scheduler.py schedule --model-id yolov11s --schedule weekly
    python training/mlops/retraining_scheduler.py list
    python training/mlops/retraining_scheduler.py run-pending
    python training/mlops/retraining_scheduler.py cancel --job-id <id>
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

def main():
    from lib.ml.mlops.continuous_learning.retraining_scheduler import RetrainingScheduler
    from lib.ml.core.logger import get_logger
    log = get_logger("retraining_scheduler")

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    sched = sub.add_parser("schedule")
    sched.add_argument("--model-id", required=True)
    sched.add_argument("--schedule", default="weekly",
                       choices=["daily","weekly","monthly","manual"])
    sched.add_argument("--trigger", default="scheduled")
    sched.add_argument("--dataset-version", default="")
    sched.add_argument("--priority", type=int, default=5)
    sub.add_parser("list")
    sub.add_parser("run-pending")
    cancel = sub.add_parser("cancel")
    cancel.add_argument("--job-id", required=True)
    sub.add_parser("export")
    args = p.parse_args()

    scheduler = RetrainingScheduler()
    if args.command == "schedule":
        job_id = scheduler.schedule(
            model_id=args.model_id,
            schedule=args.schedule,
            trigger=args.trigger,
            dataset_version_id=args.dataset_version,
            priority=args.priority,
        )
        log.info(f"Scheduled job: {job_id}")
    elif args.command == "list":
        jobs = scheduler.pending_jobs()
        for j in jobs:
            log.info(f"  {j.job_id} [{j.model_id}|{j.schedule}|{j.trigger}] → {j.scheduled_at}")
    elif args.command == "run-pending":
        ran = scheduler.run_pending()
        log.info(f"Ran {len(ran)} jobs: {ran}")
    elif args.command == "cancel":
        ok = scheduler.cancel(args.job_id)
        log.info(f"Cancelled: {ok}")
    elif args.command == "export":
        scheduler.export_queue_json()

if __name__ == "__main__":
    main()
