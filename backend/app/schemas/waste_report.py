from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.enums.report_status import ReportStatus


class WasteReportCreateRequest(BaseModel):
    description: str | None = None

    latitude: Decimal | None = Field(
        default=None,
        ge=-90,
        le=90,
    )

    longitude: Decimal | None = Field(
        default=None,
        ge=-180,
        le=180,
    )

    address: str | None = None


class WasteReportResponse(BaseModel):
    id: UUID
    user_id: UUID
    schedule_id: UUID | None = None

    description: str | None = None
    image_url: str

    latitude: Decimal | None = None
    longitude: Decimal | None = None
    address: str | None = None

    status: ReportStatus

    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )