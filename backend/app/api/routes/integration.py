from datetime import datetime, time
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies import require_role
from app.core.database import get_db
from app.enums.audit_event import AuditEvent
from app.enums.report_status import ReportStatus
from app.enums.user_role import UserRole
from app.models import (
    Assignment,
    CollectionSchedule,
    Household,
    User,
    Ward,
    WasteCategory,
    WasteReport,
    WorkerWard,
)
from app.services.audit_log_service import AuditLogService
from app.repositories.audit_log_repository import AuditLogRepository


router = APIRouter(tags=["Frontend Integration"])


class WorkerReportStatusUpdate(BaseModel):
    status: ReportStatus


class AdminScheduleCreate(BaseModel):
    ward_id: UUID
    waste_category_id: UUID
    day_of_week: str
    start_time: time
    end_time: time


def success(data, message: str | None = None):
    response = {"success": True, "data": data}
    if message is not None:
        response["message"] = message
    return response


def report_payload(report: WasteReport) -> dict:
    return {
        "id": str(report.id),
        "user_id": str(report.user_id),
        "schedule_id": str(report.schedule_id) if report.schedule_id else None,
        "description": report.description,
        "image_url": report.image_url,
        "latitude": float(report.latitude) if report.latitude is not None else None,
        "longitude": float(report.longitude) if report.longitude is not None else None,
        "address": report.address,
        "status": report.status.value,
        "created_at": report.created_at.isoformat(),
        "updated_at": report.updated_at.isoformat(),
    }


def worker_ward_ids(db: Session, worker_id: UUID) -> list[UUID]:
    return list(
        db.scalars(
            select(WorkerWard.ward_id).where(WorkerWard.worker_id == worker_id)
        ).all()
    )


def report_ward_id(db: Session, report_id: UUID) -> UUID | None:
    return db.scalar(
        select(Household.ward_id)
        .join(User, User.household_id == Household.id)
        .join(WasteReport, WasteReport.user_id == User.id)
        .where(WasteReport.id == report_id)
    )


@router.get("/admin/dashboard")
def get_admin_dashboard(
    current_user: User = Depends(require_role("ADMIN")),
    db: Session = Depends(get_db),
):
    today = datetime.now().date()
    users = list(db.scalars(select(User)).all())
    reports = list(db.scalars(select(WasteReport)).all())
    assignments = list(db.scalars(select(Assignment)).all())
    wards = list(db.scalars(select(Ward).order_by(Ward.name)).all())

    report_wards = dict(
        db.execute(
            select(WasteReport.id, Household.ward_id)
            .join(User, WasteReport.user_id == User.id)
            .join(Household, User.household_id == Household.id)
        ).all()
    )
    active_by_worker: dict[UUID, int] = {}
    completed_today_by_worker: dict[UUID, int] = {}
    reports_by_id = {report.id: report for report in reports}
    for assignment in assignments:
        report = reports_by_id.get(assignment.report_id)
        if report and report.status in {ReportStatus.ACCEPTED, ReportStatus.IN_PROGRESS}:
            active_by_worker[assignment.worker_id] = active_by_worker.get(assignment.worker_id, 0) + 1
        if assignment.completed_at and assignment.completed_at.date() == today:
            completed_today_by_worker[assignment.worker_id] = completed_today_by_worker.get(assignment.worker_id, 0) + 1

    return success({
        "stats": {
            "total_citizens": sum(user.role == UserRole.CITIZEN for user in users),
            "total_workers": sum(user.role == UserRole.WORKER for user in users),
            "total_reports": len(reports),
            "pending_reports": sum(report.status == ReportStatus.PENDING for report in reports),
            "completed_reports": sum(report.status == ReportStatus.COMPLETED for report in reports),
        },
        "workers": [
            {
                "id": str(user.id),
                "name": user.name,
                "email": user.email,
                "active_reports": active_by_worker.get(user.id, 0),
                "completed_today": completed_today_by_worker.get(user.id, 0),
            }
            for user in users if user.role == UserRole.WORKER
        ],
        "wards": [
            {
                "ward": ward.name,
                "reports": sum(report_wards.get(report.id) == ward.id for report in reports),
                "pending": sum(
                    report_wards.get(report.id) == ward.id and report.status == ReportStatus.PENDING
                    for report in reports
                ),
            }
            for ward in wards
        ],
    })


@router.get("/admin/workers")
def get_admin_workers(
    current_user: User = Depends(require_role("ADMIN")),
    db: Session = Depends(get_db),
):
    workers = list(db.scalars(select(User).where(User.role == UserRole.WORKER).order_by(User.name)).all())
    mappings = list(db.scalars(select(WorkerWard)).all())
    wards = {ward.id: ward for ward in db.scalars(select(Ward)).all()}
    return success([
        {
            "id": str(worker.id), "name": worker.name, "email": worker.email,
            "status": worker.status.value,
            "wards": [
                {"id": str(mapping.ward_id), "name": wards[mapping.ward_id].name}
                for mapping in mappings
                if mapping.worker_id == worker.id and mapping.ward_id in wards
            ],
        }
        for worker in workers
    ])


@router.get("/admin/wards")
def get_admin_wards(
    current_user: User = Depends(require_role("ADMIN")),
    db: Session = Depends(get_db),
):
    return success([
        {"id": str(ward.id), "name": ward.name, "zone": ward.zone, "description": ward.description}
        for ward in db.scalars(select(Ward).order_by(Ward.name)).all()
    ])


@router.post("/admin/workers/{worker_id}/wards/{ward_id}", status_code=status.HTTP_201_CREATED)
def assign_worker_to_ward(
    worker_id: UUID,
    ward_id: UUID,
    current_user: User = Depends(require_role("ADMIN")),
    db: Session = Depends(get_db),
):
    worker = db.get(User, worker_id)
    ward = db.get(Ward, ward_id)
    if worker is None or worker.role != UserRole.WORKER:
        raise HTTPException(status_code=400, detail="The selected user is not a worker.")
    if ward is None:
        raise HTTPException(status_code=404, detail="Ward not found.")
    if db.scalar(select(WorkerWard).where(WorkerWard.worker_id == worker_id, WorkerWard.ward_id == ward_id)):
        raise HTTPException(status_code=409, detail="Worker is already assigned to this ward.")
    if db.scalar(select(func.count()).select_from(WorkerWard).where(WorkerWard.worker_id == worker_id)) >= 5:
        raise HTTPException(status_code=400, detail="A worker can be assigned to a maximum of 5 wards.")
    db.add(WorkerWard(worker_id=worker_id, ward_id=ward_id, assigned_by=current_user.id))
    db.commit()
    return success({"worker_id": str(worker_id), "ward_id": str(ward_id), "ward_name": ward.name}, "Worker assigned to ward.")


@router.delete("/admin/workers/{worker_id}/wards/{ward_id}")
def remove_worker_from_ward(
    worker_id: UUID,
    ward_id: UUID,
    current_user: User = Depends(require_role("ADMIN")),
    db: Session = Depends(get_db),
):
    assignment = db.scalar(
        select(WorkerWard).where(
            WorkerWard.worker_id == worker_id,
            WorkerWard.ward_id == ward_id,
        )
    )
    if assignment is None:
        raise HTTPException(status_code=404, detail="Worker is not assigned to this ward.")
    db.delete(assignment)
    db.commit()
    return success(
        {"worker_id": str(worker_id), "ward_id": str(ward_id)},
        "Worker removed from ward.",
    )


def schedule_payload(schedule: CollectionSchedule, wards: dict, categories: dict) -> dict:
    return {
        "id": str(schedule.id), "ward_id": str(schedule.ward_id), "ward": wards[schedule.ward_id].name,
        "category_id": str(schedule.waste_category_id), "category": categories[schedule.waste_category_id].name,
        "day": schedule.day_of_week.title(),
        "startTime": schedule.start_time.isoformat() if schedule.start_time else "",
        "endTime": schedule.end_time.isoformat() if schedule.end_time else "",
    }


@router.get("/admin/schedules")
def get_admin_schedules(current_user: User = Depends(require_role("ADMIN")), db: Session = Depends(get_db)):
    wards = {ward.id: ward for ward in db.scalars(select(Ward)).all()}
    categories = {category.id: category for category in db.scalars(select(WasteCategory)).all()}
    schedules = db.scalars(select(CollectionSchedule).order_by(CollectionSchedule.day_of_week, CollectionSchedule.start_time)).all()
    return success([schedule_payload(schedule, wards, categories) for schedule in schedules])


@router.post("/admin/schedules", status_code=status.HTTP_201_CREATED)
def create_admin_schedule(data: AdminScheduleCreate, current_user: User = Depends(require_role("ADMIN")), db: Session = Depends(get_db)):
    day = data.day_of_week.strip().upper()
    if day not in {"MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"}:
        raise HTTPException(status_code=422, detail="day_of_week must be a valid weekday.")
    if data.start_time >= data.end_time:
        raise HTTPException(status_code=400, detail="start_time must be earlier than end_time.")
    ward, category = db.get(Ward, data.ward_id), db.get(WasteCategory, data.waste_category_id)
    if ward is None:
        raise HTTPException(status_code=404, detail="Ward not found.")
    if category is None:
        raise HTTPException(status_code=404, detail="Waste category not found.")
    schedule = CollectionSchedule(ward_id=data.ward_id, waste_category_id=data.waste_category_id, day_of_week=day, start_time=data.start_time, end_time=data.end_time)
    db.add(schedule)
    db.commit()
    db.refresh(schedule)
    return success(schedule_payload(schedule, {ward.id: ward}, {category.id: category}), "Schedule created.")


@router.delete("/admin/schedules/{schedule_id}")
def delete_admin_schedule(schedule_id: UUID, current_user: User = Depends(require_role("ADMIN")), db: Session = Depends(get_db)):
    schedule = db.get(CollectionSchedule, schedule_id)
    if schedule is None:
        raise HTTPException(status_code=404, detail="Collection schedule not found.")
    db.delete(schedule)
    db.commit()
    return success(None, "Schedule deleted.")


@router.get("/admin/waste-categories")
def get_admin_waste_categories(current_user: User = Depends(require_role("ADMIN")), db: Session = Depends(get_db)):
    return success([
        {"id": str(category.id), "name": category.name, "description": category.description}
        for category in db.scalars(select(WasteCategory).order_by(WasteCategory.name)).all()
    ])


@router.get("/worker/wards")
def get_worker_wards(current_user: User = Depends(require_role("WORKER")), db: Session = Depends(get_db)):
    wards = db.execute(
        select(WorkerWard, Ward).join(Ward, WorkerWard.ward_id == Ward.id)
        .where(WorkerWard.worker_id == current_user.id).order_by(Ward.name)
    ).all()
    return success([
        {"id": str(mapping.id), "worker_id": str(mapping.worker_id), "ward_id": str(mapping.ward_id),
         "assigned_by": str(mapping.assigned_by), "assigned_at": mapping.assigned_at.isoformat(),
         "name": ward.name, "zone": ward.zone, "description": ward.description}
        for mapping, ward in wards
    ])


@router.get("/worker/dashboard")
def get_worker_dashboard(current_user: User = Depends(require_role("WORKER")), db: Session = Depends(get_db)):
    reports = list(db.scalars(
        select(WasteReport).join(Assignment).where(Assignment.worker_id == current_user.id)
    ).all())
    today = datetime.now().date()
    return success({
        "pending": len(get_worker_queue_reports(db, current_user.id)),
        "accepted": sum(report.status == ReportStatus.ACCEPTED for report in reports),
        "in_progress": sum(report.status == ReportStatus.IN_PROGRESS for report in reports),
        "completed_today": sum(report.status == ReportStatus.COMPLETED and report.updated_at.date() == today for report in reports),
    })


def get_worker_queue_reports(db: Session, worker_id: UUID) -> list[WasteReport]:
    return list(db.scalars(
        select(WasteReport)
        .join(User, WasteReport.user_id == User.id)
        .join(Household, User.household_id == Household.id)
        .join(WorkerWard, WorkerWard.ward_id == Household.ward_id)
        .where(
            WorkerWard.worker_id == worker_id,
            WasteReport.status == ReportStatus.PENDING,
        )
        .distinct()
        .order_by(WasteReport.created_at.desc())
    ).all())


@router.get("/worker/reports/queue")
def get_worker_queue(current_user: User = Depends(require_role("WORKER")), db: Session = Depends(get_db)):
    return success([report_payload(report) for report in get_worker_queue_reports(db, current_user.id)])


@router.get("/worker/reports/active")
def get_worker_active_reports(current_user: User = Depends(require_role("WORKER")), db: Session = Depends(get_db)):
    reports = db.scalars(
        select(WasteReport).join(Assignment).where(
            Assignment.worker_id == current_user.id,
            WasteReport.status.in_([ReportStatus.ACCEPTED, ReportStatus.IN_PROGRESS]),
        ).order_by(WasteReport.updated_at.desc())
    ).all()
    return success([report_payload(report) for report in reports])


@router.patch("/worker/reports/{report_id}/status")
def update_worker_report_status(report_id: UUID, data: WorkerReportStatusUpdate, current_user: User = Depends(require_role("WORKER")), db: Session = Depends(get_db)):
    report = db.get(WasteReport, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Waste report not found.")
    assignment = db.scalar(select(Assignment).where(Assignment.report_id == report_id))
    if data.status == ReportStatus.ACCEPTED:
        if report.status != ReportStatus.PENDING or assignment is not None:
            raise HTTPException(status_code=409, detail="This report is no longer available to accept.")
        if report_ward_id(db, report_id) not in worker_ward_ids(db, current_user.id):
            raise HTTPException(status_code=403, detail="This report is not in one of your assigned wards.")
        assignment = Assignment(report_id=report_id, worker_id=current_user.id, assigned_by=current_user.id)
        db.add(assignment)
        audit_event = AuditEvent.REPORT_ASSIGNED
    else:
        if assignment is None or assignment.worker_id != current_user.id:
            raise HTTPException(status_code=403, detail="This report is not assigned to you.")
        allowed = {ReportStatus.ACCEPTED: ReportStatus.IN_PROGRESS, ReportStatus.IN_PROGRESS: ReportStatus.COMPLETED}
        if allowed.get(report.status) != data.status:
            raise HTTPException(status_code=409, detail="Invalid report status transition.")
        audit_event = AuditEvent.REPORT_COMPLETED if data.status == ReportStatus.COMPLETED else AuditEvent.REPORT_UPDATED
        if data.status == ReportStatus.COMPLETED:
            assignment.completed_at = datetime.utcnow()
    report.status = data.status
    AuditLogService(AuditLogRepository(db)).create_log(
        event_type=audit_event, entity_type="waste_report", entity_id=report.id, performed_by=current_user.id,
        metadata={"worker_id": str(current_user.id), "status": data.status.value},
    )
    db.commit()
    db.refresh(report)
    return success(report_payload(report), "Report status updated.")


@router.get("/worker/schedules")
def get_worker_schedules(current_user: User = Depends(require_role("WORKER")), db: Session = Depends(get_db)):
    ward_ids = worker_ward_ids(db, current_user.id)
    if not ward_ids:
        return success([])
    wards = {ward.id: ward for ward in db.scalars(select(Ward).where(Ward.id.in_(ward_ids))).all()}
    categories = {category.id: category for category in db.scalars(select(WasteCategory)).all()}
    schedules = db.scalars(select(CollectionSchedule).where(CollectionSchedule.ward_id.in_(ward_ids)).order_by(CollectionSchedule.day_of_week, CollectionSchedule.start_time)).all()
    return success([schedule_payload(schedule, wards, categories) for schedule in schedules])
