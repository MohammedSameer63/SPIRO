from uuid import UUID

from sqlalchemy.orm import Session

from app.core.exceptions import (
    InvalidWorkerError,
    WardNotFoundError,
    WorkerNotFoundError,
    WorkerWardAlreadyExistsError,
    WorkerWardAssignmentNotFoundError,
)
from app.enums.user_role import UserRole
from app.models.worker_ward import WorkerWard
from app.repositories.user_repository import UserRepository
from app.repositories.ward_repository import WardRepository
from app.repositories.worker_ward_repository import (
    WorkerWardRepository,
)


class WorkerWardService:
    """
    Handles worker-ward mapping business logic.
    """

    def __init__(
        self,
        db: Session,
        worker_ward_repository: WorkerWardRepository,
        user_repository: UserRepository,
        ward_repository: WardRepository,
    ):
        self.db = db
        self.worker_ward_repository = (
            worker_ward_repository
        )
        self.user_repository = user_repository
        self.ward_repository = ward_repository

    def assign_worker_to_ward(
        self,
        worker_id: UUID,
        ward_id: UUID,
        assigned_by: UUID,
    ) -> WorkerWard:

        # Verify worker exists.
        worker = self.user_repository.find_by_id(
            worker_id
        )

        if worker is None:
            raise WorkerNotFoundError(
                f"Worker with id '{worker_id}' not found."
            )

        # Verify user is actually a worker.
        if worker.role != UserRole.WORKER:
            raise InvalidWorkerError(
                f"User with id '{worker_id}' is not a worker."
            )

        # Verify ward exists.
        ward = self.ward_repository.find_by_id(
            ward_id
        )

        if ward is None:
            raise WardNotFoundError(
                f"Ward with id '{ward_id}' not found."
            )

        # Prevent duplicate mapping.
        existing = (
            self.worker_ward_repository
            .find_by_worker_and_ward(
                worker_id=worker_id,
                ward_id=ward_id,
            )
        )

        if existing is not None:
            raise WorkerWardAlreadyExistsError(
                f"Worker '{worker_id}' is already assigned "
                f"to ward '{ward_id}'."
            )

        assignment = WorkerWard(
            worker_id=worker_id,
            ward_id=ward_id,
            assigned_by=assigned_by,
        )

        try:
            assignment = (
                self.worker_ward_repository.create(
                    assignment
                )
            )

            self.db.commit()

            return assignment

        except Exception:
            self.db.rollback()
            raise

    def get_worker_wards(
        self,
        worker_id: UUID,
    ) -> list[WorkerWard]:

        worker = self.user_repository.find_by_id(
            worker_id
        )

        if worker is None:
            raise WorkerNotFoundError(
                f"Worker with id '{worker_id}' not found."
            )

        return self.worker_ward_repository.find_by_worker(
            worker_id
        )

    def get_ward_workers(
        self,
        ward_id: UUID,
    ) -> list[WorkerWard]:

        ward = self.ward_repository.find_by_id(
            ward_id
        )

        if ward is None:
            raise WardNotFoundError(
                f"Ward with id '{ward_id}' not found."
            )

        return self.worker_ward_repository.find_by_ward(
            ward_id
        )

    def remove_worker_from_ward(
        self,
        worker_id: UUID,
        ward_id: UUID,
    ) -> None:

        assignment = (
            self.worker_ward_repository
            .find_by_worker_and_ward(
                worker_id=worker_id,
                ward_id=ward_id,
            )
        )

        if assignment is None:
            raise WorkerWardAssignmentNotFoundError(
                f"Worker '{worker_id}' is not assigned "
                f"to ward '{ward_id}'."
            )

        try:
            self.worker_ward_repository.delete(
                assignment
            )

            self.db.commit()

        except Exception:
            self.db.rollback()
            raise