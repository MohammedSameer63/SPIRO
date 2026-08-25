from uuid import UUID

from sqlalchemy import select

from app.models import CollectionSchedule
from app.repositories.base_repository import BaseRepository


class CollectionScheduleRepository(BaseRepository):
    """
    Handles all database operations related to collection schedules.
    """

    def find_by_id(
        self,
        schedule_id: UUID,
    ) -> CollectionSchedule | None:

        stmt = select(CollectionSchedule).where(
            CollectionSchedule.id == schedule_id
        )

        result = self.db.execute(stmt)

        return result.scalar_one_or_none()

    def find_by_ward(
        self,
        ward_id: UUID,
    ) -> list[CollectionSchedule]:

        stmt = (
            select(CollectionSchedule)
            .where(
                CollectionSchedule.ward_id == ward_id
            )
            .order_by(
                CollectionSchedule.day_of_week,
                CollectionSchedule.start_time,
            )
        )

        result = self.db.execute(stmt)

        return list(result.scalars().all())

    def find_by_category(
        self,
        category_id: UUID,
    ) -> list[CollectionSchedule]:

        stmt = (
            select(CollectionSchedule)
            .where(
                CollectionSchedule.waste_category_id == category_id
            )
            .order_by(
                CollectionSchedule.day_of_week,
                CollectionSchedule.start_time,
            )
        )

        result = self.db.execute(stmt)

        return list(result.scalars().all())

    def find_by_ward_and_category(
        self,
        ward_id: UUID,
        category_id: UUID,
    ) -> list[CollectionSchedule]:

        stmt = (
            select(CollectionSchedule)
            .where(
                CollectionSchedule.ward_id == ward_id,
                CollectionSchedule.waste_category_id == category_id,
            )
            .order_by(
                CollectionSchedule.day_of_week,
                CollectionSchedule.start_time,
            )
        )

        result = self.db.execute(stmt)

        return list(result.scalars().all())

    def create(
        self,
        schedule: CollectionSchedule,
    ) -> CollectionSchedule:

        return self._persist(schedule)