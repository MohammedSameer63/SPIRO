from uuid import UUID

from pydantic import BaseModel, ConfigDict


class WardResponse(BaseModel):
    id: UUID
    name: str
    zone: str | None = None

    model_config = ConfigDict(
        from_attributes=True,
    )