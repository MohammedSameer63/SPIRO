from uuid import UUID

from sqlalchemy import select

from app.models import Ward
from app.repositories.base_repository import BaseRepository


class WardRepository(BaseRepository):
    """
    Handles all database operations related to the Ward model.
    """

    def find_by_id(
        self,
        ward_id: UUID,
    ) -> Ward | None:
        """
        Retrieve a ward by ID.
        """

        stmt = select(Ward).where(Ward.id == ward_id)

        result = self.db.execute(stmt)

        return result.scalar_one_or_none()

    def find_all(self) -> list[Ward]:
        """
        Retrieve all wards.
        """

        stmt = select(Ward).order_by(Ward.name)

        result = self.db.execute(stmt)

        return list(result.scalars().all())
