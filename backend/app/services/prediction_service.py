from sqlalchemy.orm import Session

from app.core.exceptions import (
    PredictionAlreadyExistsError,
    ReportNotFoundError,
    WasteCategoryNotFoundError,
)
from app.models import Prediction
from app.repositories.prediction_repository import (
    PredictionRepository,
)
from app.repositories.waste_category_repository import (
    WasteCategoryRepository,
)
from app.repositories.waste_report_repository import (
    WasteReportRepository,
)
from app.schemas.prediction import PredictionResult


class PredictionService:
    """
    Handles prediction business logic.
    """

    def __init__(
        self,
        db: Session,
        prediction_repository: PredictionRepository,
        waste_report_repository: WasteReportRepository,
        waste_category_repository: WasteCategoryRepository,
    ):
        self.db = db
        self.prediction_repository = prediction_repository
        self.waste_report_repository = (
            waste_report_repository
        )
        self.waste_category_repository = (
            waste_category_repository
        )

    def create_prediction(
        self,
        request: PredictionResult,
    ) -> Prediction:

        report = self.waste_report_repository.find_by_id(
            request.report_id
        )

        if report is None:
            raise ReportNotFoundError(
                "Waste report not found."
            )

        category = (
            self.waste_category_repository.find_by_id(
                request.category_id
            )
        )

        if category is None:
            raise WasteCategoryNotFoundError(
                "Waste category not found."
            )

        existing_prediction = (
            self.prediction_repository.find_by_report_id(
                request.report_id
            )
        )

        if existing_prediction is not None:
            raise PredictionAlreadyExistsError(
                "A prediction already exists for this report."
            )

        prediction = Prediction(
            report_id=request.report_id,
            category_id=request.category_id,
            confidence=request.confidence,
            model_used=request.model_used,
        )

        try:
            prediction = self.prediction_repository.create(
                prediction
            )

            self.db.commit()

            return prediction

        except Exception:
            self.db.rollback()
            raise