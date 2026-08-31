from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, require_role
from app.core.database import get_db
from app.models.user import User
from app.schemas.assignment import (
    AssignmentCreateRequest,
    AssignmentResponse,
)
from app.repositories.assignment_repository import AssignmentRepository
from app.repositories.user_repository import UserRepository
from app.repositories.waste_report_repository import WasteReportRepository
from app.repositories.audit_log_repository import AuditLogRepository
from app.services.audit_log_service import AuditLogService
from app.services.assignment_service import AssignmentService


router = APIRouter(
    prefix="/assignments",
    tags=["Assignments"],
)


def get_assignment_service(
    db: Session = Depends(get_db),
) -> AssignmentService:
    assignment_repository = AssignmentRepository(db)
    user_repository = UserRepository(db)
    waste_report_repository = WasteReportRepository(db)

    audit_log_repository = AuditLogRepository(db)
    audit_log_service = AuditLogService(
        audit_log_repository
    )

    return AssignmentService(
        db=db,
        assignment_repository=assignment_repository,
        user_repository=user_repository,
        waste_report_repository=waste_report_repository,
        audit_log_service=audit_log_service,
    )


@router.post(
    "",
    response_model=AssignmentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_assignment(
    request: AssignmentCreateRequest,
    current_user: User = Depends(require_role("ADMIN")),
    service: AssignmentService = Depends(get_assignment_service),
):
    return service.create_assignment(
        report_id=request.report_id,
        worker_id=request.worker_id,
        assigned_by=current_user.id,
    )


@router.get(
    "/{assignment_id}",
    response_model=AssignmentResponse,
)
async def get_assignment(
    assignment_id: UUID,
    current_user: User = Depends(require_role("ADMIN", "WORKER")),
    service: AssignmentService = Depends(get_assignment_service),
):
    assignment = service.get_assignment(assignment_id)
    if current_user.role.value != "ADMIN" and assignment.worker_id != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have permission to view this assignment.")
    return assignment


@router.get(
    "/report/{report_id}",
    response_model=AssignmentResponse,
)
async def get_assignment_by_report(
    report_id: UUID,
    current_user: User = Depends(require_role("ADMIN", "WORKER")),
    service: AssignmentService = Depends(get_assignment_service),
):
    assignment = service.get_assignment_by_report(report_id)
    if current_user.role.value != "ADMIN" and assignment.worker_id != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have permission to view this assignment.")
    return assignment


@router.patch(
    "/{assignment_id}/complete",
    response_model=AssignmentResponse,
)
async def complete_assignment(
    assignment_id: UUID,
    current_user: User = Depends(require_role("WORKER")),
    service: AssignmentService = Depends(get_assignment_service),
):
    assignment = service.get_assignment(assignment_id)
    if assignment.worker_id != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have permission to complete this assignment.")
    return service.complete_assignment(
        assignment_id=assignment_id,
        performed_by=current_user.id,
    )
