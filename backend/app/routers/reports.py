import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError
from sqlalchemy import text
from sqlalchemy.orm import Session


from app.core.config import settings
from app.core.database import get_db
from app.core.security import decode_token
from app.model.user import User
from app.model.report import WasteReport


router = APIRouter(
    prefix="/reports",
    tags=["Reports"],
)

security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    token = credentials.credentials

    try:
        payload = decode_token(token)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user_id = payload.get("sub")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return user


# ============================================================
# POST /reports
# ============================================================

@router.post("")
async def create_report(
    description: str = Form(...),
    address: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role.value != "CITIZEN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only citizens can create reports",
        )

    if not image.content_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image",
        )

    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only image files are allowed",
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    extension = Path(
        image.filename or ""
    ).suffix.lower()

    if not extension:
        extension = ".jpg"

    filename = f"{uuid.uuid4()}{extension}"
    file_path = upload_dir / filename

    contents = await image.read()

    max_size = (
        settings.max_image_size_mb
        * 1024
        * 1024
    )

    if len(contents) > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Image size cannot exceed "
                f"{settings.max_image_size_mb} MB"
            ),
        )

    with open(file_path, "wb") as file:
        file.write(contents)

    now = datetime.now(timezone.utc)

    report = WasteReport(
        user_id=current_user.id,
        description=description,
        image_url=str(file_path),
        latitude=latitude,
        longitude=longitude,
        address=address,
        status="PENDING",
        created_at=now,
        updated_at=now,
    )

    db.add(report)
    db.commit()
    db.refresh(report)

    return {
        "success": True,
        "message": "Report submitted successfully.",
        "data": {
            "id": str(report.id),
            "status": report.status,
            "description": report.description,
            "address": report.address,
            "latitude": float(report.latitude),
            "longitude": float(report.longitude),
            "created_at": report.created_at,
        },
    }
# ============================================================
# GET /reports/my
# ============================================================

@router.get("/my")
def get_my_reports(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    reports = (
        db.query(WasteReport)
        .filter(
            WasteReport.user_id
            == current_user.id
        )
        .order_by(
            WasteReport.created_at.desc()
        )
        .all()
    )

    return {
        "success": True,
        "data": [
            {
                "id": str(report.id),
                "description": report.description,
                "address": report.address,
                "latitude": float(report.latitude),
                "longitude": float(report.longitude),
                "status": report.status,
                "image_url": report.image_url,
                "created_at": report.created_at,
            }
            for report in reports
        ],
    }


# ============================================================
# GET /reports/stats
# ============================================================

@router.get("/stats")
def get_report_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    reports = (
        db.query(WasteReport)
        .filter(
            WasteReport.user_id
            == current_user.id
        )
        .all()
    )

    total = len(reports)

    pending = sum(
        1
        for report in reports
        if report.status == "PENDING"
    )

    completed = sum(
        1
        for report in reports
        if report.status == "COMPLETED"
    )

    return {
        "success": True,
        "data": {
            "total": total,
            "pending": pending,
            "completed": completed,
        },
    }

# ============================================================
# GET /reports/admin/pending
# ============================================================

@router.get("/admin/pending")
def get_admin_pending_reports(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role.value != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can access pending reports",
        )

    reports = (
        db.query(WasteReport)
        .filter(
            WasteReport.status == "PENDING"
        )
        .order_by(
            WasteReport.created_at.asc()
        )
        .all()
    )

    return {
        "success": True,
        "data": [
            {
                "id": str(report.id),
                "description": report.description,
                "address": report.address,
                "latitude": float(report.latitude),
                "longitude": float(report.longitude),
                "status": report.status,
                "image_url": report.image_url,
                "created_at": report.created_at,
                "updated_at": report.updated_at,
            }
            for report in reports
        ],
    }


# ============================================================
# POST /reports/admin/{report_id}/assign/{worker_id}
# ============================================================

@router.post("/admin/{report_id}/assign/{worker_id}")
def assign_report_to_worker(
    report_id: uuid.UUID,
    worker_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role.value != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can assign reports",
        )

    # --------------------------------------------------------
    # Check report
    # --------------------------------------------------------

    report = (
        db.query(WasteReport)
        .filter(
            WasteReport.id == report_id
        )
        .first()
    )

    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found",
        )

    if report.status != "PENDING":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only pending reports can be assigned",
        )

    # --------------------------------------------------------
    # Check worker
    # --------------------------------------------------------

    worker = (
        db.query(User)
        .filter(
            User.id == worker_id,
            User.role == "WORKER",
            User.status == "ACTIVE",
        )
        .first()
    )

    if not worker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Active worker not found",
        )

    # --------------------------------------------------------
    # Check existing assignment
    # --------------------------------------------------------

    existing = db.execute(
        text(
            """
            SELECT id
            FROM assignments
            WHERE report_id = :report_id
            """
        ),
        {
            "report_id": report_id,
        },
    ).first()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Report is already assigned",
        )

    # --------------------------------------------------------
    # Create assignment
    # --------------------------------------------------------

    db.execute(
        text(
            """
            INSERT INTO assignments
                (
                    report_id,
                    worker_id,
                    assigned_by
                )
            VALUES
                (
                    :report_id,
                    :worker_id,
                    :assigned_by
                )
            """
        ),
        {
            "report_id": report_id,
            "worker_id": worker_id,
            "assigned_by": current_user.id,
        },
    )

    db.commit()

    return {
        "success": True,
        "message": "Report assigned to worker successfully",
        "data": {
            "report_id": str(report_id),
            "worker_id": str(worker_id),
            "worker_name": worker.name,
        },
    }

# ============================================================
# GET /reports/worker/queue
# ============================================================

@router.get("/worker/queue")
def get_worker_queue(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role.value != "WORKER":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only workers can access the report queue",
        )

    result = db.execute(
        text(
            """
            SELECT
                r.id,
                r.description,
                r.address,
                r.latitude,
                r.longitude,
                r.status,
                r.image_url,
                r.created_at,
                r.updated_at
            FROM waste_reports r
            INNER JOIN assignments a
                ON a.report_id = r.id
            WHERE a.worker_id = :worker_id
              AND r.status = 'PENDING'
            ORDER BY r.created_at ASC
            """
        ),
        {
            "worker_id": current_user.id,
        },
    )

    reports = []

    for row in result.mappings():
        reports.append(
            {
                "id": str(row["id"]),
                "description": row["description"],
                "address": row["address"],
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "status": row["status"],
                "image_url": row["image_url"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )

    return {
        "success": True,
        "data": reports,
    }

# ============================================================
# GET /reports/worker/active
# ============================================================

@router.get("/worker/active")
def get_worker_active_reports(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role.value != "WORKER":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only workers can access active reports",
        )

    result = db.execute(
        text(
            """
            SELECT
                r.id,
                r.description,
                r.address,
                r.latitude,
                r.longitude,
                r.status,
                r.image_url,
                r.created_at,
                r.updated_at
            FROM waste_reports r
            INNER JOIN assignments a
                ON a.report_id = r.id
            WHERE a.worker_id = :worker_id
              AND r.status IN ('ACCEPTED', 'IN_PROGRESS')
            ORDER BY r.updated_at DESC
            """
        ),
        {
            "worker_id": current_user.id,
        },
    )

    reports = []

    for row in result.mappings():
        reports.append(
            {
                "id": str(row["id"]),
                "description": row["description"],
                "address": row["address"],
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "status": row["status"],
                "image_url": row["image_url"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )

    return {
        "success": True,
        "data": reports,
    }


# ============================================================
# PATCH /reports/worker/{report_id}/status
# ============================================================

@router.patch("/worker/{report_id}/status")
def update_worker_report_status(
    report_id: uuid.UUID,
    new_status: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role.value != "WORKER":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only workers can update reports",
        )

    allowed_statuses = [
        "ACCEPTED",
        "IN_PROGRESS",
        "COMPLETED",
    ]

    if new_status not in allowed_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid report status",
        )

    report = (
        db.query(WasteReport)
        .filter(
            WasteReport.id == report_id
        )
        .first()
    )

    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found",
        )

    if new_status == "ACCEPTED":
        if report.status != "PENDING":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only pending reports can be accepted",
            )

    elif new_status == "IN_PROGRESS":
        if report.status != "ACCEPTED":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only accepted reports can be started",
            )

    elif new_status == "COMPLETED":
        if report.status != "IN_PROGRESS":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only reports in progress can be completed",
            )

    report.status = new_status
    report.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(report)

    return {
        "success": True,
        "message": f"Report status updated to {new_status}.",
        "data": {
            "id": str(report.id),
            "status": report.status,
            "description": report.description,
            "address": report.address,
            "latitude": float(report.latitude),
            "longitude": float(report.longitude),
            "image_url": report.image_url,
            "created_at": report.created_at,
            "updated_at": report.updated_at,
        },
    }
# ============================================================
# GET /reports/worker/stats
# ============================================================

@router.get("/worker/stats")
def get_worker_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role.value != "WORKER":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only workers can access worker statistics",
        )

    reports = (
        db.query(WasteReport)
        .all()
    )

    pending = sum(
        1
        for report in reports
        if report.status == "PENDING"
    )

    accepted = sum(
        1
        for report in reports
        if report.status == "ACCEPTED"
    )

    in_progress = sum(
        1
        for report in reports
        if report.status == "IN_PROGRESS"
    )

    today = datetime.now(timezone.utc).date()

    completed_today = sum(
        1
        for report in reports
        if (
            report.status == "COMPLETED"
            and report.updated_at
            and report.updated_at.date() == today
        )
    )

    return {
        "success": True,
        "data": {
            "pending": pending,
            "accepted": accepted,
            "in_progress": in_progress,
            "completed_today": completed_today,
        },
    }

# ============================================================
# GET /admin/workers
# ============================================================

@router.get("/admin/workers")
def get_admin_workers(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role.value != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can access worker management",
        )

    result = db.execute(
        text(
            """
            SELECT
                u.id,
                u.name,
                u.email,
                u.status,
                COALESCE(
                    json_agg(
                        json_build_object(
                            'id', w.id,
                            'name', w.name
                        )
                    ) FILTER (WHERE w.id IS NOT NULL),
                    '[]'
                ) AS wards
            FROM users u
            LEFT JOIN worker_wards ww
                ON ww.worker_id = u.id
            LEFT JOIN wards w
                ON w.id = ww.ward_id
            WHERE u.role = 'WORKER'
            GROUP BY u.id, u.name, u.email, u.status
            ORDER BY u.name
            """
        )
    )

    workers = []

    for row in result.mappings():
        workers.append(
            {
                "id": str(row["id"]),
                "name": row["name"],
                "email": row["email"],
                "status": row["status"],
                "wards": row["wards"],
            }
        )

    return {
        "success": True,
        "data": workers,
    }


# ============================================================
# GET /admin/wards
# ============================================================

@router.get("/admin/wards")
def get_admin_wards(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role.value != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can access ward management",
        )

    result = db.execute(
        text(
            """
            SELECT
                id,
                name,
                zone,
                description
            FROM wards
            ORDER BY name
            """
        )
    )

    wards = []

    for row in result.mappings():
        wards.append(
            {
                "id": str(row["id"]),
                "name": row["name"],
                "zone": row["zone"],
                "description": row["description"],
            }
        )

    return {
        "success": True,
        "data": wards,
    }


# ============================================================
# POST /admin/workers/{worker_id}/wards/{ward_id}
# ============================================================

@router.post("/admin/workers/{worker_id}/wards/{ward_id}")
def assign_worker_to_ward(
    worker_id: uuid.UUID,
    ward_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role.value != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can assign workers to wards",
        )

    # Check worker
    worker = (
        db.query(User)
        .filter(
            User.id == worker_id,
            User.role == "WORKER",
            User.status == "ACTIVE",
        )
        .first()
    )

    if not worker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Active worker not found",
        )

    # Check ward
    ward = db.execute(
        text(
            """
            SELECT id, name
            FROM wards
            WHERE id = :ward_id
            """
        ),
        {"ward_id": ward_id},
    ).mappings().first()

    if not ward:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ward not found",
        )

    # Check duplicate assignment
    existing = db.execute(
        text(
            """
            SELECT id
            FROM worker_wards
            WHERE worker_id = :worker_id
              AND ward_id = :ward_id
            """
        ),
        {
            "worker_id": worker_id,
            "ward_id": ward_id,
        },
    ).first()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Worker is already assigned to this ward",
        )

    # Maximum 5 wards
    ward_count = db.execute(
        text(
            """
            SELECT COUNT(*)
            FROM worker_wards
            WHERE worker_id = :worker_id
            """
        ),
        {"worker_id": worker_id},
    ).scalar()

    if ward_count >= 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A worker can be assigned to a maximum of 5 wards",
        )

    # Create assignment
    db.execute(
        text(
            """
            INSERT INTO worker_wards
                (worker_id, ward_id, assigned_by)
            VALUES
                (:worker_id, :ward_id, :assigned_by)
            """
        ),
        {
            "worker_id": worker_id,
            "ward_id": ward_id,
            "assigned_by": current_user.id,
        },
    )

    db.commit()

    return {
        "success": True,
        "message": "Worker assigned to ward successfully",
        "data": {
            "worker_id": str(worker_id),
            "ward_id": str(ward_id),
            "ward_name": ward["name"],
        },
    }

# ============================================================
# GET /admin/dashboard
# ============================================================

@router.get("/admin/dashboard")
def get_admin_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # --------------------------------------------------------
    # Admin authorization
    # --------------------------------------------------------

    if current_user.role.value != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can access the admin dashboard",
        )

    # --------------------------------------------------------
    # User statistics
    # --------------------------------------------------------

    total_citizens = (
        db.query(User)
        .filter(
            User.role == "CITIZEN",
            User.status == "ACTIVE",
        )
        .count()
    )

    total_workers = (
        db.query(User)
        .filter(
            User.role == "WORKER",
            User.status == "ACTIVE",
        )
        .count()
    )

    # --------------------------------------------------------
    # Report statistics
    # --------------------------------------------------------

    total_reports = (
        db.query(WasteReport)
        .count()
    )

    pending_reports = (
        db.query(WasteReport)
        .filter(
            WasteReport.status == "PENDING"
        )
        .count()
    )

    completed_reports = (
        db.query(WasteReport)
        .filter(
            WasteReport.status == "COMPLETED"
        )
        .count()
    )

    # --------------------------------------------------------
    # Worker overview
    # --------------------------------------------------------

    workers = (
        db.query(User)
        .filter(
            User.role == "WORKER",
            User.status == "ACTIVE",
        )
        .all()
    )

    today = datetime.now(timezone.utc).date()

    worker_overview = []

    for worker in workers:

        active_reports = (
            db.query(WasteReport)
            .filter(
                WasteReport.status.in_(
                    ["ACCEPTED", "IN_PROGRESS"]
                )
            )
            .count()
        )

        completed_today = (
            db.query(WasteReport)
            .filter(
                WasteReport.status == "COMPLETED",
                WasteReport.updated_at.isnot(None),
            )
            .all()
        )

        completed_today_count = sum(
            1
            for report in completed_today
            if report.updated_at.date() == today
        )

        worker_overview.append(
            {
                "id": str(worker.id),
                "name": worker.name,
                "email": worker.email,
                "active_reports": active_reports,
                "completed_today": completed_today_count,
            }
        )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "success": True,
        "data": {
            "stats": {
                "total_citizens": total_citizens,
                "total_workers": total_workers,
                "total_reports": total_reports,
                "pending_reports": pending_reports,
                "completed_reports": completed_reports,
            },
            "workers": worker_overview,
            "wards": [],
        },
    }

# ============================================================
# GET /reports/worker/wards
# ============================================================

@router.get("/worker/wards")
def get_worker_wards(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role.value != "WORKER":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only workers can access assigned wards",
        )

    result = db.execute(
        text(
            """
            SELECT
                w.id,
                w.name,
                w.zone,
                w.description
            FROM worker_wards ww
            JOIN wards w
                ON w.id = ww.ward_id
            WHERE ww.worker_id = :worker_id
            ORDER BY w.name
            """
        ),
        {
            "worker_id": current_user.id,
        },
    )

    wards = []

    for row in result.mappings():
        wards.append(
            {
                "id": str(row["id"]),
                "name": row["name"],
                "zone": row["zone"],
                "description": row["description"],
            }
        )

    return {
        "success": True,
        "data": wards,
    }

# ============================================================
# GET /reports/{report_id}
# ============================================================

@router.get("/{report_id}")
def get_report(
    report_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    report = (
        db.query(WasteReport)
        .filter(
            WasteReport.id == report_id,
            WasteReport.user_id == current_user.id,
        )
        .first()
    )

    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found",
        )

    return {
        "success": True,
        "data": {
            "id": str(report.id),
            "description": report.description,
            "address": report.address,
            "latitude": float(report.latitude),
            "longitude": float(report.longitude),
            "status": report.status,
            "image_url": report.image_url,
            "created_at": report.created_at,
            "updated_at": report.updated_at,
        },
    }
# ============================================================
# ADMIN SCHEDULE MANAGEMENT
# ============================================================

# GET /reports/admin/schedules
@router.get("/admin/schedules")
def get_admin_schedules(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role.value != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can access schedules",
        )

    result = db.execute(
        text(
            """
            SELECT
                cs.id,
                w.id AS ward_id,
                w.name AS ward,
                wc.id AS category_id,
                wc.name AS category,
                cs.day_of_week,
                cs.start_time,
                cs.end_time
            FROM collection_schedules cs
            JOIN wards w
                ON w.id = cs.ward_id
            JOIN waste_categories wc
                ON wc.id = cs.waste_category_id
            ORDER BY w.name, cs.day_of_week, cs.start_time
            """
        )
    )

    schedules = []

    for row in result.mappings():
        schedules.append(
            {
                "id": str(row["id"]),
                "ward_id": str(row["ward_id"]),
                "ward": row["ward"],
                "category_id": str(row["category_id"]),
                "category": row["category"],
                "day": row["day_of_week"],
                "startTime": (
                    row["start_time"].strftime("%H:%M")
                    if row["start_time"]
                    else ""
                ),
                "endTime": (
                    row["end_time"].strftime("%H:%M")
                    if row["end_time"]
                    else ""
                ),
            }
        )

    return {
        "success": True,
        "data": schedules,
    }


# POST /reports/admin/schedules
@router.post("/admin/schedules")
def create_admin_schedule(
    ward_id: uuid.UUID,
    waste_category_id: uuid.UUID,
    day_of_week: str,
    start_time: str,
    end_time: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role.value != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can create schedules",
        )

    # Check ward
    ward = db.execute(
        text(
            """
            SELECT id, name
            FROM wards
            WHERE id = :ward_id
            """
        ),
        {"ward_id": ward_id},
    ).mappings().first()

    if not ward:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ward not found",
        )

    # Check waste category
    category = db.execute(
        text(
            """
            SELECT id, name
            FROM waste_categories
            WHERE id = :category_id
            """
        ),
        {"category_id": waste_category_id},
    ).mappings().first()

    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Waste category not found",
        )

    # Check duplicate
    duplicate = db.execute(
        text(
            """
            SELECT id
            FROM collection_schedules
            WHERE ward_id = :ward_id
              AND waste_category_id = :category_id
              AND day_of_week = :day_of_week
            """
        ),
        {
            "ward_id": ward_id,
            "category_id": waste_category_id,
            "day_of_week": day_of_week,
        },
    ).first()

    if duplicate:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Duplicate schedule is not allowed",
        )

    # Insert schedule
    schedule_id = uuid.uuid4()

    db.execute(
        text(
            """
            INSERT INTO collection_schedules (
                id,
                ward_id,
                waste_category_id,
                day_of_week,
                start_time,
                end_time
            )
            VALUES (
                :id,
                :ward_id,
                :category_id,
                :day_of_week,
                :start_time,
                :end_time
            )
            """
        ),
        {
            "id": schedule_id,
            "ward_id": ward_id,
            "category_id": waste_category_id,
            "day_of_week": day_of_week,
            "start_time": start_time,
            "end_time": end_time,
        },
    )

    db.commit()

    return {
        "success": True,
        "message": "Schedule created successfully",
        "data": {
            "id": str(schedule_id),
            "ward_id": str(ward_id),
            "ward": ward["name"],
            "category_id": str(waste_category_id),
            "category": category["name"],
            "day": day_of_week,
            "startTime": start_time,
            "endTime": end_time,
        },
    }


# DELETE /reports/admin/schedules/{schedule_id}
@router.delete("/admin/schedules/{schedule_id}")
def delete_admin_schedule(
    schedule_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role.value != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can delete schedules",
        )

    result = db.execute(
        text(
            """
            DELETE FROM collection_schedules
            WHERE id = :schedule_id
            RETURNING id
            """
        ),
        {"schedule_id": schedule_id},
    )

    deleted = result.first()

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schedule not found",
        )

    db.commit()

    return {
        "success": True,
        "message": "Schedule deleted successfully",
    }
# ============================================================
# GET /reports/admin/waste-categories
# ============================================================

@router.get("/admin/waste-categories")
def get_admin_waste_categories(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role.value != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can access waste categories",
        )

    result = db.execute(
        text(
            """
            SELECT
                id,
                name,
                recyclable,
                description
            FROM waste_categories
            ORDER BY name
            """
        )
    )

    categories = []

    for row in result.mappings():
        categories.append(
            {
                "id": str(row["id"]),
                "name": row["name"],
                "recyclable": row["recyclable"],
                "description": row["description"],
            }
        )

    return {
        "success": True,
        "data": categories,
    }# ============================================================
# GET /reports/worker/schedules
# ============================================================

@router.get("/worker/schedules")
def get_worker_schedules(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role.value != "WORKER":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only workers can access schedules",
        )

    result = db.execute(
        text(
            """
            SELECT
                cs.id,
                w.id AS ward_id,
                w.name AS ward,
                wc.id AS category_id,
                wc.name AS category,
                cs.day_of_week,
                cs.start_time,
                cs.end_time
            FROM collection_schedules cs
            INNER JOIN worker_wards ww
                ON ww.ward_id = cs.ward_id
            INNER JOIN wards w
                ON w.id = cs.ward_id
            INNER JOIN waste_categories wc
                ON wc.id = cs.waste_category_id
            WHERE ww.worker_id = :worker_id
            ORDER BY
                w.name,
                CASE cs.day_of_week
                    WHEN 'Monday' THEN 1
                    WHEN 'Tuesday' THEN 2
                    WHEN 'Wednesday' THEN 3
                    WHEN 'Thursday' THEN 4
                    WHEN 'Friday' THEN 5
                    WHEN 'Saturday' THEN 6
                    WHEN 'Sunday' THEN 7
                    ELSE 8
                END,
                cs.start_time
            """
        ),
        {
            "worker_id": current_user.id,
        },
    )

    schedules = []

    for row in result.mappings():
        schedules.append(
            {
                "id": str(row["id"]),
                "ward_id": str(row["ward_id"]),
                "ward": row["ward"],
                "category_id": str(row["category_id"]),
                "category": row["category"],
                "day": row["day_of_week"],
                "startTime": (
                    row["start_time"].strftime("%H:%M")
                    if row["start_time"]
                    else ""
                ),
                "endTime": (
                    row["end_time"].strftime("%H:%M")
                    if row["end_time"]
                    else ""
                ),
            }
        )

    return {
        "success": True,
        "data": schedules,
    }