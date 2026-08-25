from datetime import time
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator


class CollectionScheduleCreate(BaseModel):
    ward_id: UUID
    waste_category_id: UUID
    day_of_week: str
    start_time: time | None = None
    end_time: time | None = None

    @field_validator("day_of_week")
    @classmethod
    def normalize_day(cls, value: str) -> str:
        value = value.strip().upper()

        valid_days = {
            "MONDAY",
            "TUESDAY",
            "WEDNESDAY",
            "THURSDAY",
            "FRIDAY",
            "SATURDAY",
            "SUNDAY",
        }

        if value not in valid_days:
            raise ValueError(
                "day_of_week must be a valid weekday."
            )

        return value


class CollectionScheduleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    ward_id: UUID
    waste_category_id: UUID
    day_of_week: str
    start_time: time | None
    end_time: time | None