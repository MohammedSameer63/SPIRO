from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.exceptions import (
    AssignmentAlreadyExistsError,
    AssignmentNotFoundError,
    InvalidWorkerError,
    ReportNotFoundError,
)
from app.enums.user_role import UserRole
from app.models.assignment import Assignment
from app.models.waste_report import ReportStatus
from app.repositories.assignment_repository import AssignmentRepository
from app.repositories.user_repository import UserRepository
from app.repositories.waste_report_repository import WasteReportRepository


class AssignmentService:
    """
    Business logic for waste-report assignments.
    """

    def __init__(
        self,
        db: Session,
        assignment_repository: AssignmentRepository,
        user_repository: UserRepository,
        waste_report_repository: WasteReportRepository,
    ):
        self.db = db
        self.assignment_repository = assignment_repository
        self.user_repository = user_repository
        self.waste_report_repository = waste_report_repository

    def create_assignment(
        self,
        report_id: UUID,
        worker_id: UUID,
        assigned_by: UUID,
    ) -> Assignment:
        # 1. Verify report exists.
        report = self.waste_report_repository.find_by_id(report_id)

        if report is None:
            raise ReportNotFoundError(
                f"Waste report with id '{report_id}' not found."
            )

        # 2. Verify worker exists.
        worker = self.user_repository.find_by_id(worker_id)

        if worker is None:
            raise InvalidWorkerError(
                f"User with id '{worker_id}' not found."
            )

        # 3. Verify the selected user is actually a worker.
        if worker.role != UserRole.WORKER:
            raise InvalidWorkerError(
                f"User with id '{worker_id}' is not a worker."
            )

        # 4. Prevent duplicate assignment.
        existing_assignment = (
            self.assignment_repository.find_by_report_id(report_id)
        )

        if existing_assignment is not None:
            raise AssignmentAlreadyExistsError(
                f"Waste report with id '{report_id}' is already assigned."
            )

        # 5. Create assignment.
        assignment = Assignment(
            report_id=report_id,
            worker_id=worker_id,
            assigned_by=assigned_by,
        )

        return self.assignment_repository.create(assignment)

    def get_assignment(
        self,
        assignment_id: UUID,
    ) -> Assignment:
        assignment = self.assignment_repository.find_by_id(
            assignment_id
        )

        if assignment is None:
            raise AssignmentNotFoundError(
                f"Assignment with id '{assignment_id}' not found."
            )

        return assignment

    def get_assignment_by_report(
        self,
        report_id: UUID,
    ) -> Assignment:
        assignment = self.assignment_repository.find_by_report_id(
            report_id
        )

        if assignment is None:
            raise AssignmentNotFoundError(
                f"No assignment found for report '{report_id}'."
            )

        return assignment

    def complete_assignment(
        self,
        assignment_id: UUID,
    ) -> Assignment:
        assignment = self.get_assignment(assignment_id)

        try:
            # Mark assignment as completed.
            if assignment.completed_at is None:
                assignment.completed_at = datetime.utcnow()

            # Find the associated waste report.
            report = self.waste_report_repository.find_by_id(
                assignment.report_id
            )

            if report is None:
                raise ReportNotFoundError(
                    f"Report with id '{assignment.report_id}' not found."
                )

            # Mark the waste report as completed.
            report.status = ReportStatus.COMPLETED

            # Both repositories use the same SQLAlchemy session.
            self.db.commit()

            # Refresh assignment so the latest DB state is returned.
            self.db.refresh(assignment)

            return assignment

        except Exception:
            self.db.rollback()
            raise