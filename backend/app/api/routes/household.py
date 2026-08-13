from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.database import get_db
from app.models import User
from app.repositories.household_repository import HouseholdRepository
from app.schemas.household import HouseholdResponse
from app.services.household_service import HouseholdService


router = APIRouter(
    prefix="/households",
    tags=["Households"],
)


def get_household_service(
    db: Session = Depends(get_db),
) -> HouseholdService:
    household_repository = HouseholdRepository(db)

    return HouseholdService(
        household_repository=household_repository,
    )


@router.get(
    "",
    response_model=list[HouseholdResponse],
    status_code=200,
)
def get_households(
    ward_id: UUID | None = None,
    current_user: User = Depends(get_current_user),
    service: HouseholdService = Depends(
        get_household_service
    ),
):
    return service.get_all_households(
        current_user=current_user,
        ward_id=ward_id,
    )


@router.get(
    "/{household_id}",
    response_model=HouseholdResponse,
    status_code=200,
)
def get_household(
    household_id: UUID,
    current_user: User = Depends(get_current_user),
    service: HouseholdService = Depends(
        get_household_service
    ),
):
    return service.get_household_by_id(
        household_id=household_id,
        current_user=current_user,
    )