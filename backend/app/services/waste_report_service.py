from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.enums.audit_event import AuditEvent
from app.models import User, WasteReport
from app.repositories.waste_report_repository import (
    WasteReportRepository,
)
from app.schemas.waste_report import WasteReportCreateRequest
from app.services.audit_log_service import AuditLogService
from app.services.file_storage_service import FileStorageService
from app.core.exceptions import ReportNotFoundError

class WasteReportService:
    """
    Handles waste report business logic.
    """

    def __init__(
        self,
        db: Session,
        waste_report_repository: WasteReportRepository,
        file_storage_service: FileStorageService,
        audit_log_service: AuditLogService,
    ):
        self.db = db
        self.waste_report_repository = (
            waste_report_repository
        )
        self.file_storage_service = (
            file_storage_service
        )
        self.audit_log_service = audit_log_service

    async def create_report(
        self,
        request: WasteReportCreateRequest,
        image: UploadFile,
        current_user: User,
    ) -> WasteReport:
        """
        Create a new waste report for the authenticated user.
        """

        image_url = (
            await self.file_storage_service.save_report_image(
                image
            )
        )

        report = WasteReport(
            user_id=current_user.id,
            description=request.description,
            image_url=image_url,
            latitude=request.latitude,
            longitude=request.longitude,
            address=request.address,
        )

        try:
            report = self.waste_report_repository.create(
                report
            )

            self.audit_log_service.create_log(
                event_type=AuditEvent.REPORT_CREATED,
                entity_type="waste_report",
                entity_id=report.id,
                performed_by=current_user.id,
            )

            self.db.commit()

            return report

        except Exception:
            self.db.rollback()
            raise

    def get_my_reports(
        self,
        current_user: User,
    ) -> list[WasteReport]:
        """
        Retrieve all waste reports created by the authenticated user.
        """

        return self.waste_report_repository.find_by_user(
            current_user.id
        )

    def get_report(
        self,
        report_id,
        current_user: User,
    ) -> WasteReport:
        """
        Retrieve a specific waste report belonging
        to the authenticated user.
        """

        report = self.waste_report_repository.find_by_id(
            report_id
        )

        if report is None:
            raise ReportNotFoundError("Waste report not found.")

        if report.user_id != current_user.id:
            raise ReportNotFoundError("Waste report not found.")

        return report