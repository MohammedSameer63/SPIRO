from uuid import UUID

from app.core.exceptions import (
    HouseholdAccessDeniedError,
    InvalidQrPayloadError,
    ReportNotFoundError,
)
from app.models import PickupValidation, User, WasteReport
from app.enums.report_status import ReportStatus
from app.repositories.pickup_validation_repository import (
    PickupValidationRepository,
)
from app.repositories.waste_report_repository import WasteReportRepository
from app.services.qr_service import verify_household_qr_payload


class PickupService:
    """
    Validates a waste pickup by scanning the citizen's household QR code.

    The worker scans the QR at collection time. Validation succeeds only if:
    - the QR signature is genuine,
    - the household in the QR owns the report, and
    - the report is assigned to the scanning worker and still actionable.
    """

    def __init__(
        self,
        db,
        validation_repository: PickupValidationRepository,
        waste_report_repository: WasteReportRepository,
    ):
        self.db = db
        self.validation_repository = validation_repository
        self.waste_report_repository = waste_report_repository

    def validate_pickup(
        self,
        report_id: UUID,
        qr_payload: str,
        worker: User,
    ) -> PickupValidation:
        from app.models import Assignment
        from sqlalchemy import select

        report: WasteReport | None = (
            self.waste_report_repository.find_by_id(report_id)
        )
        if report is None:
            raise ReportNotFoundError("Waste report not found.")

        household_id = verify_household_qr_payload(qr_payload)

        # The QR must belong to the household that filed the report.
        if report.user.household_id is None or (
            report.user.household_id != household_id
        ):
            raise InvalidQrPayloadError(
                "This QR code does not belong to the report's household."
            )

        # Only the assigned worker can validate, while work is in progress.
        assignment = self.db.scalar(
            select(Assignment).where(Assignment.report_id == report_id)
        )
        if assignment is None or assignment.worker_id != worker.id:
            raise HouseholdAccessDeniedError(
                "This report is not assigned to you."
            )
        if report.status not in (
            ReportStatus.ACCEPTED,
            ReportStatus.IN_PROGRESS,
        ):
            raise InvalidQrPayloadError(
                "Pickup can only be validated for accepted reports."
            )

        # Idempotent: a previous scan for the same report is returned as-is.
        existing = self.validation_repository.find_by_report_id(report_id)
        if existing is not None:
            return existing

        return self.validation_repository.create(
            PickupValidation(
                report_id=report_id,
                household_id=household_id,
                worker_id=worker.id,
                method="QR_SCAN",
            )
        )
