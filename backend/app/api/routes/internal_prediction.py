from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from uuid import UUID

from app.api.dependencies import verify_ml_service, get_current_user
from app.core.database import get_db
from app.models import User
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
from app.services.ml_inference_service import ml_inference_service
from app.services.prediction_service import PredictionService
from app.core.config import settings
from pathlib import Path


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


@router.post(
    "/infer/{report_id}",
    response_model=PredictionResponse,
    status_code=201,
    dependencies=[Depends(verify_ml_service)],
)
def run_inference_for_report(
    report_id: UUID,
    service: PredictionService = Depends(
        get_prediction_service
    ),
    db: Session = Depends(get_db),
):
    """
    Run ML inference for a specific report.
    Used by external ML services or for re-running inference.
    """
    from app.repositories.waste_report_repository import WasteReportRepository

    waste_report_repo = WasteReportRepository(db)
    report = waste_report_repo.find_by_id(report_id)

    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Waste report not found."
        )

    # Resolve image path (image_url is "/uploads/reports/<file>")
    relative = report.image_url.lstrip("/")
    backend_dir = Path(__file__).parent.parent.parent.parent
    candidates = [
        Path.cwd() / relative,
        backend_dir / relative,
        backend_dir / "uploads" / "reports" / Path(relative).name,
    ]
    if Path(settings.upload_dir).is_absolute():
        candidates.insert(0, Path(settings.upload_dir) / Path(relative).name)
    image_path = next((p for p in candidates if p.exists()), None)

    if image_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report image not found."
        )

    # Run inference
    top_prediction = ml_inference_service.get_top_prediction(str(image_path))

    if not top_prediction:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ML inference returned no results."
        )

    # Map waste_stream -> waste_category UUID (categories store streams,
    # not fine-grained classes)
    from app.repositories.waste_category_repository import WasteCategoryRepository
    category_repo = WasteCategoryRepository(db)
    all_categories = category_repo.find_all()

    if not all_categories:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No waste categories configured."
        )

    # Stream-name matching (handles "e_waste" vs "E-Waste", etc.)
    waste_stream = (top_prediction.get("waste_stream") or "").strip().lower().replace("_", "-")
    category_id = None
    for category in all_categories:
        if category.name.strip().lower().replace("_", "-") == waste_stream:
            category_id = category.id
            break
    if category_id is None:
        compact = waste_stream.replace("-", "").replace(" ", "")
        for category in all_categories:
            cat_compact = (
                category.name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
            )
            if cat_compact == compact:
                category_id = category.id
                break

    if not category_id:
        category_id = all_categories[0].id

    # DB expects confidence in 0-100; ML returns 0-1
    raw_conf = top_prediction.get("confidence") or 0
    confidence_100 = max(0.0, min(100.0, round(float(raw_conf) * 100, 2)))

    # Save prediction
    prediction_result = PredictionResult(
        report_id=report_id,
        category_id=category_id,
        confidence=confidence_100,
        model_used=f"{top_prediction.get('method', 'spiro_pipeline')}",
    )

    return service.create_prediction(prediction_result)


@router.get(
    "/report/{report_id}",
    response_model=PredictionResponse,
)
def get_prediction_for_report(
    report_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get the prediction for a specific report.
    """
    prediction_repo = PredictionRepository(db)
    prediction = prediction_repo.find_by_report_id(report_id)

    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found for this report."
        )

    # Check authorization - user can only see their own report's prediction
    waste_report_repo = WasteReportRepository(db)
    report = waste_report_repo.find_by_id(report_id)

    if not report or report.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this prediction."
        )

    return prediction


def _get_class_name(class_id: int) -> str:
    """Get class name from the loaded ML model label map."""
    try:
        names = ml_inference_service._class_names
        if 0 <= class_id < len(names):
            return names[class_id]
    except Exception:
        pass
    return f"class_{class_id}"