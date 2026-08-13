from uuid import UUID

from app.models import Ward
from app.repositories.ward_repository import WardRepository


class WardService:
    """
    Handles ward-related business logic.
    """

    def __init__(
        self,
        ward_repository: WardRepository,
    ):
        self.ward_repository = ward_repository

    def get_all_wards(self) -> list[Ward]:
        """
        Retrieve all wards.
        """

        return self.ward_repository.find_all()

    def get_ward_by_id(
        self,
        ward_id: UUID,
    ) -> Ward | None:
        """
        Retrieve a ward by ID.
        """

        return self.ward_repository.find_by_id(ward_id)