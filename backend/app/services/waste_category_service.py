from uuid import UUID

from app.models import WasteCategory
from app.repositories.waste_category_repository import (
    WasteCategoryRepository,
)


class WasteCategoryService:
    """
    Handles waste category business logic.
    """

    def __init__(
        self,
        waste_category_repository: WasteCategoryRepository,
    ):
        self.waste_category_repository = (
            waste_category_repository
        )

    def get_all_categories(self) -> list[WasteCategory]:
        """
        Retrieve all waste categories.
        """

        return self.waste_category_repository.find_all()

    def get_category_by_id(
        self,
        category_id: UUID,
    ) -> WasteCategory | None:
        """
        Retrieve a waste category by ID.
        """

        return self.waste_category_repository.find_by_id(
            category_id
        )