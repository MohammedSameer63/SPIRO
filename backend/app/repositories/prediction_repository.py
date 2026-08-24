from uuid import UUID

from sqlalchemy import select

from app.models import Prediction
from app.repositories.base_repository import BaseRepository


class PredictionRepository(BaseRepository):
    """
    Handles database operations related to predictions.
    """

    def create(
        self,
        prediction: Prediction,
    ) -> Prediction:
        """
        Persist a new prediction.
        """

        return self._persist(prediction)

    def find_by_id(
        self,
        prediction_id: UUID,
    ) -> Prediction | None:
        """
        Retrieve a prediction by ID.
        """

        stmt = select(Prediction).where(
            Prediction.id == prediction_id
        )

        result = self.db.execute(stmt)

        return result.scalar_one_or_none()

    def find_by_report_id(
        self,
        report_id: UUID,
    ) -> Prediction | None:
        """
        Retrieve the prediction associated with a report.
        """

        stmt = select(Prediction).where(
            Prediction.report_id == report_id
        )

        result = self.db.execute(stmt)

        return result.scalar_one_or_none()