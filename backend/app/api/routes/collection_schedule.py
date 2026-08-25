from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, require_role, get_db
from app.models.user import User
from app.schemas.collection_schedule import (
    CollectionScheduleCreate,
    CollectionScheduleResponse,
)
from app.services.collection_schedule_service import (
    CollectionScheduleService,
)
from app.repositories.collection_schedule_repository import (
    CollectionScheduleRepository,
)
from app.repositories.ward_repository import WardRepository
from app.repositories.waste_category_repository import (
    WasteCategoryRepository,
)


router = APIRouter(
    prefix="/collection-schedules",
    tags=["Collection Schedules"],
)


def get_collection_schedule_service(
    db: Session = Depends(get_db),
) -> CollectionScheduleService:
    return CollectionScheduleService(
        collection_schedule_repository=CollectionScheduleRepository(db),
        ward_repository=WardRepository(db),
        waste_category_repository=WasteCategoryRepository(db),
    )


@router.get(
    "/{schedule_id}",
    response_model=CollectionScheduleResponse,
)
def get_schedule(
    schedule_id: UUID,
    service: CollectionScheduleService = Depends(
        get_collection_schedule_service
    ),
    current_user: User = Depends(get_current_user),
):
    schedule = service.get_schedule(schedule_id)

    if schedule is None:
        # We don't currently have a ScheduleNotFoundError.
        # Return a normal 404 for now.
        from fastapi import HTTPException

        raise HTTPException(
            status_code=404,
            detail=f"Collection schedule with id '{schedule_id}' not found.",
        )

    return schedule


@router.get(
    "/ward/{ward_id}",
    response_model=list[CollectionScheduleResponse],
)
def get_schedules_by_ward(
    ward_id: UUID,
    service: CollectionScheduleService = Depends(
        get_collection_schedule_service
    ),
    current_user: User = Depends(get_current_user),
):
    return service.get_schedules_by_ward(ward_id)


@router.get(
    "/category/{category_id}",
    response_model=list[CollectionScheduleResponse],
)
def get_schedules_by_category(
    category_id: UUID,
    service: CollectionScheduleService = Depends(
        get_collection_schedule_service
    ),
    current_user: User = Depends(get_current_user),
):
    return service.get_schedules_by_category(category_id)


@router.post(
    "",
    response_model=CollectionScheduleResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_schedule(
    data: CollectionScheduleCreate,
    service: CollectionScheduleService = Depends(
        get_collection_schedule_service
    ),
    current_user: User = Depends(
        require_role("ADMIN")
    ),
):
    return service.create_schedule(
        ward_id=data.ward_id,
        category_id=data.waste_category_id,
        day_of_week=data.day_of_week,
        start_time=data.start_time,
        end_time=data.end_time,
    )