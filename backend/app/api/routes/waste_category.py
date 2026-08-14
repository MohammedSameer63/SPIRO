from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.database import get_db
from app.models import User
from app.repositories.waste_category_repository import (
    WasteCategoryRepository,
)
from app.schemas.waste_category import WasteCategoryResponse
from app.services.waste_category_service import (
    WasteCategoryService,
)


router = APIRouter(
    prefix="/categories",
    tags=["Waste Categories"],
)


def get_waste_category_service(
    db: Session = Depends(get_db),
) -> WasteCategoryService:
    repository = WasteCategoryRepository(db)

    return WasteCategoryService(
        waste_category_repository=repository,
    )


@router.get(
    "",
    response_model=list[WasteCategoryResponse],
    status_code=200,
)
def get_categories(
    current_user: User = Depends(get_current_user),
    service: WasteCategoryService = Depends(
        get_waste_category_service
    ),
):
    return service.get_all_categories()