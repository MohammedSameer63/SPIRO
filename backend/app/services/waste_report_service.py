from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.models import User, WasteReport
from app.repositories.waste_report_repository import (
    WasteReportRepository,
)
from app.schemas.waste_report import WasteReportCreateRequest
from app.services.file_storage_service import FileStorageService


class WasteReportService:
    """
    Handles waste report business logic.
    """

    def __init__(
        self,
        db: Session,
        waste_report_repository: WasteReportRepository,
        file_storage_service: FileStorageService,
    ):
        self.db = db
        self.waste_report_repository = (
            waste_report_repository
        )
        self.file_storage_service = (
            file_storage_service
        )

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

            self.db.commit()

            return report

        except Exception:
            self.db.rollback()
            raise