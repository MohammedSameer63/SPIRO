from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.worker_ward import WorkerWard
from app.repositories.base_repository import BaseRepository


class WorkerWardRepository(BaseRepository):
    """
    Handles persistence operations for worker-ward mappings.
    """

    def __init__(self, db: Session):
        super().__init__(db)

    def create(
        self,
        worker_ward: WorkerWard,
    ) -> WorkerWard:
        return self._persist(worker_ward)

    def find_by_id(
        self,
        worker_ward_id: UUID,
    ) -> WorkerWard | None:
        return self.db.get(
            WorkerWard,
            worker_ward_id,
        )

    def find_by_worker_and_ward(
        self,
        worker_id: UUID,
        ward_id: UUID,
    ) -> WorkerWard | None:
        statement = select(WorkerWard).where(
            WorkerWard.worker_id == worker_id,
            WorkerWard.ward_id == ward_id,
        )

        return self.db.scalar(statement)

    def find_by_worker(
        self,
        worker_id: UUID,
    ) -> list[WorkerWard]:
        statement = (
            select(WorkerWard)
            .where(WorkerWard.worker_id == worker_id)
            .order_by(WorkerWard.assigned_at)
        )

        return list(
            self.db.scalars(statement).all()
        )

    def find_by_ward(
        self,
        ward_id: UUID,
    ) -> list[WorkerWard]:
        statement = (
            select(WorkerWard)
            .where(WorkerWard.ward_id == ward_id)
            .order_by(WorkerWard.assigned_at)
        )

        return list(
            self.db.scalars(statement).all()
        )

    def delete(
        self,
        worker_ward: WorkerWard,
    ) -> None:
        self._remove(worker_ward)