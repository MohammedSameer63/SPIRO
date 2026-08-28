from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.assignment import Assignment
from app.repositories.base_repository import BaseRepository


class AssignmentRepository(BaseRepository):
    """
    Repository for assignment persistence and queries.
    """

    def __init__(self, db: Session):
        super().__init__(db)

    def create(
        self,
        assignment: Assignment,
    ) -> Assignment:
        return self._persist(assignment)

    def find_by_id(
        self,
        assignment_id: UUID,
    ) -> Assignment | None:
        statement = select(Assignment).where(
            Assignment.id == assignment_id
        )

        return self.db.scalar(statement)

    def find_by_report_id(
        self,
        report_id: UUID,
    ) -> Assignment | None:
        statement = select(Assignment).where(
            Assignment.report_id == report_id
        )

        return self.db.scalar(statement)

    def update(
        self,
        assignment: Assignment,
    ) -> Assignment:
        self.db.flush()
        self.db.refresh(assignment)

        return assignment

    def delete(
        self,
        assignment: Assignment,
    ) -> None:
        self._remove(assignment)