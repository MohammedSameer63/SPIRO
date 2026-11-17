from uuid import UUID

from sqlalchemy.orm import Session

from app.models.pickup_validation import PickupValidation
from app.repositories.base_repository import BaseRepository


class PickupValidationRepository(BaseRepository):
    """
    Repository for pickup-validation persistence.
    """

    def __init__(self, db: Session):
        super().__init__(db)

    def create(
        self,
        validation: PickupValidation,
    ) -> PickupValidation:
        return self._persist(validation)

    def find_by_report_id(
        self,
        report_id: UUID,
    ) -> PickupValidation | None:
        return (
            self.db.query(PickupValidation)
            .filter(PickupValidation.report_id == report_id)
            .first()
        )
