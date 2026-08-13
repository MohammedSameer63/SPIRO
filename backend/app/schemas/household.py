from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class HouseholdResponse(BaseModel):
    id: UUID
    ward_id: UUID
    house_number: str
    street_name: str | None = None
    address: str
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )