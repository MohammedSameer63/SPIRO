from uuid import UUID

from sqlalchemy import select

from app.models import WasteCategory
from app.repositories.base_repository import BaseRepository


class WasteCategoryRepository(BaseRepository):
    """
    Handles database operations related to waste categories.
    """

    def find_all(self) -> list[WasteCategory]:
        """
        Retrieve all waste categories.
        """

        stmt = select(WasteCategory).order_by(
            WasteCategory.name
        )

        result = self.db.execute(stmt)

        return list(result.scalars().all())

    def find_by_id(
        self,
        category_id: UUID,
    ) -> WasteCategory | None:
        """
        Retrieve a waste category by ID.
        """

        stmt = select(WasteCategory).where(
            WasteCategory.id == category_id
        )

        result = self.db.execute(stmt)

        return result.scalar_one_or_none()