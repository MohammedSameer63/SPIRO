from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import verify_ml_service
from app.core.database import get_db
from app.repositories.prediction_repository import (
    PredictionRepository,
)
from app.repositories.waste_category_repository import (
    WasteCategoryRepository,
)
from app.repositories.waste_report_repository import (
    WasteReportRepository,
)
from app.schemas.prediction import (
    PredictionResponse,
    PredictionResult,
)
from app.services.prediction_service import PredictionService


router = APIRouter(
    prefix="/internal/predictions",
    tags=["Internal - Predictions"],
)


def get_prediction_service(
    db: Session = Depends(get_db),
) -> PredictionService:
    return PredictionService(
        db=db,
        prediction_repository=PredictionRepository(db),
        waste_report_repository=WasteReportRepository(db),
        waste_category_repository=WasteCategoryRepository(db),
    )


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=201,
    dependencies=[Depends(verify_ml_service)],
)
def create_prediction(
    result: PredictionResult,
    service: PredictionService = Depends(
        get_prediction_service
    ),
):
    return service.create_prediction(result)