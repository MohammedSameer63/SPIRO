from datetime import time
from uuid import UUID

from app.core.exceptions import (
    InvalidCollectionScheduleError,
    WardNotFoundError,
    WasteCategoryNotFoundError,
)
from app.models import CollectionSchedule
from app.repositories.collection_schedule_repository import (
    CollectionScheduleRepository,
)
from app.repositories.waste_category_repository import (
    WasteCategoryRepository,
)
from app.repositories.ward_repository import WardRepository


class CollectionScheduleService:
    """
    Handles business logic for collection schedules.
    """

    VALID_DAYS = {
        "MONDAY",
        "TUESDAY",
        "WEDNESDAY",
        "THURSDAY",
        "FRIDAY",
        "SATURDAY",
        "SUNDAY",
    }

    def __init__(
        self,
        collection_schedule_repository: CollectionScheduleRepository,
        ward_repository: WardRepository,
        waste_category_repository: WasteCategoryRepository,
    ):
        self.collection_schedule_repository = (
            collection_schedule_repository
        )
        self.ward_repository = ward_repository
        self.waste_category_repository = waste_category_repository

    def get_schedule(
        self,
        schedule_id: UUID,
    ) -> CollectionSchedule | None:

        return self.collection_schedule_repository.find_by_id(
            schedule_id
        )

    def get_schedules_by_ward(
        self,
        ward_id: UUID,
    ) -> list[CollectionSchedule]:

        if not self.ward_repository.find_by_id(ward_id):
            raise WardNotFoundError(
                f"Ward with id '{ward_id}' not found."
            )

        return self.collection_schedule_repository.find_by_ward(
            ward_id
        )

    def get_schedules_by_category(
        self,
        category_id: UUID,
    ) -> list[CollectionSchedule]:

        if not self.waste_category_repository.find_by_id(category_id):
            raise ValueError(
                f"Waste category with id '{category_id}' not found."
            )

        return self.collection_schedule_repository.find_by_category(
            category_id
        )

    def create_schedule(
        self,
        ward_id: UUID,
        category_id: UUID,
        day_of_week: str,
        start_time: time | None,
        end_time: time | None,
    ) -> CollectionSchedule:

        if not self.ward_repository.find_by_id(ward_id):
            raise WardNotFoundError(
                f"Ward with id '{ward_id}' not found."
            )

        if not self.waste_category_repository.find_by_id(category_id):
            raise WasteCategoryNotFoundError(
                f"Waste category with id '{category_id}' not found."
            )

        normalized_day = day_of_week.strip().upper()

        if normalized_day not in self.VALID_DAYS:
            raise InvalidCollectionScheduleError(
                "day_of_week must be a valid weekday."
            )

        if (
            start_time is not None
            and end_time is not None
            and start_time >= end_time
        ):
            raise InvalidCollectionScheduleError(
                "start_time must be earlier than end_time."
            )

        schedule = CollectionSchedule(
            ward_id=ward_id,
            waste_category_id=category_id,
            day_of_week=normalized_day,
            start_time=start_time,
            end_time=end_time,
        )

        return self.collection_schedule_repository.create(
            schedule
        )