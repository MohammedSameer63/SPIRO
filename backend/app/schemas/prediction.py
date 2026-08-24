from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PredictionResult(BaseModel):
    """
    Prediction result produced by the ML service.
    """
    report_id: UUID
    category_id: UUID

    confidence: Decimal | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    model_used: str | None = None


class PredictionResponse(BaseModel):
    id: UUID
    report_id: UUID
    category_id: UUID

    confidence: Decimal | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    model_used: str | None = None
    prediction_time: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )