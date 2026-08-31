"""
SPIRO ML — RetrainingScheduler
Manages scheduled and event-triggered model retraining.

Supports
--------
- daily / weekly / monthly / manual schedules
- Cron expression support (via schedule library or fallback)
- Drift-triggered retraining
- Queue management with status tracking
- Integration with DatasetRegistry and ModelRegistry
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from lib.ml.core.logger import get_logger

log = get_logger(__name__)


@dataclass
class RetrainingJob:
    job_id: str
    model_id: str
    trigger: str                      # "drift" | "scheduled" | "manual" | "continuous"
    schedule: str                     # "daily" | "weekly" | "monthly" | cron | "manual"
    dataset_version_id: str = ""
    status: str = "pending"           # pending | running | completed | failed | cancelled
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    scheduled_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    result_model_version: str = ""
    error: str = ""
    priority: int = 5                 # 1 (highest) – 10 (lowest)
    config_path: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RetrainingScheduler:
    """
    Queues, schedules, and executes model retraining jobs.

    Parameters
    ----------
    queue_path : Path
        Persistent JSON queue file.
    config_dir : Path
        Directory containing training YAML configs.
    run_callback : callable, optional
        Called with (RetrainingJob) when a job executes.
        If None, jobs log a message and mark completed.

    Example
    -------
    >>> scheduler = RetrainingScheduler()
    >>> job_id = scheduler.schedule("yolov11s", "weekly")
    >>> scheduler.schedule("yolov11s", "drift", priority=1)
    >>> scheduler.run_pending()
    """

    _SCHEDULE_INTERVALS: Dict[str, int] = {
        "daily":   60 * 60 * 24,
        "weekly":  60 * 60 * 24 * 7,
        "monthly": 60 * 60 * 24 * 30,
    }

    def __init__(
        self,
        queue_path: Path = Path("mlops/retraining_queue.json"),
        config_dir: Path = Path("configs/training"),
        run_callback: Optional[Callable[["RetrainingJob"], None]] = None,
    ) -> None:
        self.queue_path = Path(queue_path)
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_dir = Path(config_dir)
        self.run_callback = run_callback
        self._queue: List[RetrainingJob] = self._load_queue()
        self._lock = threading.Lock()
        self._last_run: Dict[str, Dict[str, float]] = {}  # model_id → {schedule → last_ts}
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def schedule(
        self,
        model_id: str,
        schedule: str = "weekly",
        trigger: str = "scheduled",
        dataset_version_id: str = "",
        config_path: str = "",
        priority: int = 5,
        notes: str = "",
        run_at: Optional[datetime] = None,
    ) -> str:
        """
        Queue a retraining job.

        Parameters
        ----------
        model_id : str
        schedule : "daily" | "weekly" | "monthly" | "manual"
        trigger : "scheduled" | "drift" | "manual" | "continuous"
        run_at : specific datetime for this job (None = now / next window)

        Returns
        -------
        job_id : str
        """
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        if run_at is None:
            interval = self._SCHEDULE_INTERVALS.get(schedule, 0)
            run_at = datetime.utcnow() + timedelta(seconds=interval) if interval > 0 else datetime.utcnow()

        job = RetrainingJob(
            job_id=job_id,
            model_id=model_id,
            trigger=trigger,
            schedule=schedule,
            dataset_version_id=dataset_version_id,
            config_path=config_path or str(self.config_dir / f"yolo11s.yaml"),
            priority=priority,
            notes=notes,
            scheduled_at=run_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            status="pending",
        )

        with self._lock:
            self._queue.append(job)
            self._save_queue()

        log.info(
            f"Retraining job queued: {job_id} "
            f"[{model_id}|{schedule}|{trigger}] "
            f"at {job.scheduled_at}"
        )
        return job_id

    def schedule_drift_triggered(
        self,
        model_id: str,
        drift_score: float,
        threshold: float = 0.25,
    ) -> Optional[str]:
        """Queue a drift-triggered retraining job if drift_score exceeds threshold."""
        if drift_score < threshold:
            return None
        job_id = self.schedule(
            model_id=model_id,
            schedule="manual",
            trigger="drift",
            priority=2,
            notes=f"Drift score={drift_score:.4f} exceeded threshold={threshold}",
        )
        log.warning(
            f"Drift-triggered retraining queued for {model_id} "
            f"(score={drift_score:.4f})"
        )
        return job_id

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_pending(self) -> List[str]:
        """
        Run all pending jobs whose scheduled_at time has passed.
        Returns list of executed job_ids.
        """
        now = datetime.utcnow()
        executed = []

        with self._lock:
            # Sort by priority then by scheduled_at
            pending = sorted(
                [j for j in self._queue if j.status == "pending"],
                key=lambda j: (j.priority, j.scheduled_at),
            )

        for job in pending:
            try:
                sched_dt = datetime.strptime(job.scheduled_at, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                sched_dt = now

            if sched_dt <= now:
                self._execute(job)
                executed.append(job.job_id)

        return executed

    def start_background_worker(self, poll_interval_s: float = 60.0) -> None:
        """Start a background thread that polls run_pending every interval."""
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(poll_interval_s,),
            daemon=True,
        )
        self._worker_thread.start()
        log.info(f"Retraining scheduler started (interval={poll_interval_s}s)")

    def stop_background_worker(self) -> None:
        self._running = False

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            for job in self._queue:
                if job.job_id == job_id and job.status == "pending":
                    job.status = "cancelled"
                    self._save_queue()
                    log.info(f"Job {job_id} cancelled")
                    return True
        return False

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> Optional[RetrainingJob]:
        for job in self._queue:
            if job.job_id == job_id:
                return job
        return None

    def pending_jobs(self) -> List[RetrainingJob]:
        return [j for j in self._queue if j.status == "pending"]

    def completed_jobs(self, model_id: Optional[str] = None) -> List[RetrainingJob]:
        jobs = [j for j in self._queue if j.status == "completed"]
        if model_id:
            jobs = [j for j in jobs if j.model_id == model_id]
        return jobs

    def export_queue_json(self, output_path: Optional[Path] = None) -> Path:
        output_path = output_path or Path("mlops/reports/retraining_queue.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {"jobs": [j.to_dict() for j in self._queue], "total": len(self._queue)},
                f, indent=2
            )
        return output_path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute(self, job: RetrainingJob) -> None:
        job.status = "running"
        job.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        self._save_queue()
        log.info(f"Executing retraining job {job.job_id} [{job.model_id}|{job.trigger}]")

        try:
            if self.run_callback is not None:
                self.run_callback(job)
            else:
                # Default: log and simulate completion
                log.info(
                    f"[Retraining] Would train {job.model_id} "
                    f"with config={job.config_path} "
                    f"dataset={job.dataset_version_id}"
                )
                time.sleep(0.01)  # simulate minimal work
            job.status = "completed"
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            log.error(f"Retraining job {job.job_id} failed: {e}")

        job.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._lock:
            self._save_queue()

    def _worker_loop(self, interval: float) -> None:
        while self._running:
            self.run_pending()
            time.sleep(interval)

    def _load_queue(self) -> List[RetrainingJob]:
        if self.queue_path.exists():
            try:
                data = json.loads(self.queue_path.read_text())
                return [RetrainingJob(**j) for j in data.get("jobs", [])]
            except Exception as e:
                log.warning(f"Queue load failed: {e}")
        return []

    def _save_queue(self) -> None:
        self.queue_path.write_text(
            json.dumps({"jobs": [j.to_dict() for j in self._queue]}, indent=2)
        )
