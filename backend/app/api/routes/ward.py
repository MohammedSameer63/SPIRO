from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.repositories.ward_repository import WardRepository
from app.schemas.ward import WardResponse
from app.services.ward_service import WardService
from app.api.dependencies import get_current_user
from app.models import User

router = APIRouter(
    prefix="/wards",
    tags=["Wards"],
)


def get_ward_service(
    db: Session = Depends(get_db),
) -> WardService:
    ward_repository = WardRepository(db)

    return WardService(
        ward_repository=ward_repository,
    )


@router.get(
    "",
    response_model=list[WardResponse],
    status_code=status.HTTP_200_OK,
)
def get_wards(
    current_user: User = Depends(get_current_user),
    service: WardService = Depends(get_ward_service),
):
    return service.get_all_wards()


@router.get(
    "/{ward_id}",
    response_model=WardResponse,
    status_code=status.HTTP_200_OK,
)
def get_ward(
    ward_id: UUID,
    current_user: User = Depends(get_current_user),
    service: WardService = Depends(get_ward_service),
):
    ward = service.get_ward_by_id(ward_id)

    if ward is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ward not found.",
        )

    return ward