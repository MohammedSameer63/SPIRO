from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.dependencies import (
    get_current_user,
    require_role,
)
from app.core.database import get_db
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.repositories.ward_repository import WardRepository
from app.repositories.worker_ward_repository import (
    WorkerWardRepository,
)
from app.schemas.worker_ward import WorkerWardResponse
from app.services.worker_ward_service import WorkerWardService


router = APIRouter(
    prefix="/worker-wards",
    tags=["Worker-Ward Mapping"],
)


def get_worker_ward_service(
    db: Session = Depends(get_db),
) -> WorkerWardService:

    return WorkerWardService(
        db=db,
        worker_ward_repository=WorkerWardRepository(db),
        user_repository=UserRepository(db),
        ward_repository=WardRepository(db),
    )


@router.post(
    "/{worker_id}/{ward_id}",
    response_model=WorkerWardResponse,
    status_code=status.HTTP_201_CREATED,
)
def assign_worker_to_ward(
    worker_id: UUID,
    ward_id: UUID,
    current_user: Annotated[
        User,
        Depends(require_role("ADMIN")),
    ],
    service: WorkerWardService = Depends(
        get_worker_ward_service
    ),
):
    return service.assign_worker_to_ward(
        worker_id=worker_id,
        ward_id=ward_id,
        assigned_by=current_user.id,
    )


@router.get(
    "/worker/{worker_id}",
    response_model=list[WorkerWardResponse],
)
def get_worker_wards(
    worker_id: UUID,
    current_user: User = Depends(get_current_user),
    service: WorkerWardService = Depends(
        get_worker_ward_service
    ),
):
    # Workers can view their own mappings.
    # Admins can view any worker's mappings.
    if (
        current_user.role.value != "ADMIN"
        and current_user.id != worker_id
    ):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view these assignments.",
        )

    return service.get_worker_wards(
        worker_id
    )


@router.get(
    "/ward/{ward_id}",
    response_model=list[WorkerWardResponse],
)
def get_ward_workers(
    ward_id: UUID,
    current_user: Annotated[
        User,
        Depends(require_role("ADMIN")),
    ],
    service: WorkerWardService = Depends(
        get_worker_ward_service
    ),
):
    return service.get_ward_workers(
        ward_id
    )


@router.delete(
    "/{worker_id}/{ward_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def remove_worker_from_ward(
    worker_id: UUID,
    ward_id: UUID,
    current_user: Annotated[
        User,
        Depends(require_role("ADMIN")),
    ],
    service: WorkerWardService = Depends(
        get_worker_ward_service
    ),
):
    service.remove_worker_from_ward(
        worker_id=worker_id,
        ward_id=ward_id,
    )