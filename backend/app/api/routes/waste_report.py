from decimal import Decimal
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.database import get_db
from app.core.config import settings
from app.models import User
from app.repositories.waste_report_repository import (
    WasteReportRepository,
)
from app.schemas.waste_report import (
    WasteReportCreateRequest,
    WasteReportResponse,
)
from app.services.file_storage_service import FileStorageService
from app.services.waste_report_service import WasteReportService


router = APIRouter(
    prefix="/reports",
    tags=["Waste Reports"],
)

def get_waste_report_service(
    db: Session = Depends(get_db),
) -> WasteReportService:
    waste_report_repository = WasteReportRepository(db)

    file_storage_service = FileStorageService(
        upload_directory=(
            Path(settings.upload_dir) / "reports"
        ),
        max_image_size_mb=settings.max_image_size_mb,
    )

    return WasteReportService(
        db=db,
        waste_report_repository=waste_report_repository,
        file_storage_service=file_storage_service,
    )


@router.post(
    "",
    response_model=WasteReportResponse,
    status_code=201,
)
async def create_report(
    image: UploadFile = File(...),
    description: str | None = Form(default=None),
    latitude: Annotated[
        Decimal | None,
        Form(ge=-90, le=90),
    ] = None,
    longitude: Annotated[
        Decimal | None,
        Form(ge=-180, le=180),
    ] = None,
    address: str | None = Form(default=None),
    current_user: User = Depends(get_current_user),
    service: WasteReportService = Depends(
        get_waste_report_service
    ),
):
    request = WasteReportCreateRequest(
        description=description,
        latitude=latitude,
        longitude=longitude,
        address=address,
    )

    return await service.create_report(
        request=request,
        image=image,
        current_user=current_user,
    )