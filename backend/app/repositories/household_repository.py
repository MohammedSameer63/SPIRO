from uuid import UUID

from sqlalchemy import select

from app.models import Household
from app.repositories.base_repository import BaseRepository


class HouseholdRepository(BaseRepository):
    """
    Handles all database operations related to the Household model.
    """

    def find_by_id(
        self,
        household_id: UUID,
    ) -> Household | None:

        stmt = select(Household).where(
            Household.id == household_id
        )

        result = self.db.execute(stmt)

        return result.scalar_one_or_none()

    def find_by_location(
        self,
        ward_id: UUID,
        house_number: str,
        street_name: str,
    ) -> Household | None:
        """
        Retrieve a household using its unique location.
        """

        stmt = select(Household).where(
            Household.ward_id == ward_id,
            Household.house_number == house_number,
            Household.street_name == street_name,
        )

        result = self.db.execute(stmt)

        return result.scalar_one_or_none()

    def create(
        self,
        household: Household,
    ) -> Household:

        return self._persist(household)
