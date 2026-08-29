from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class WorkerWardCreateRequest(BaseModel):
    worker_id: UUID
    ward_id: UUID


class WorkerWardResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
    )

    id: UUID
    worker_id: UUID
    ward_id: UUID
    assigned_by: UUID
    assigned_at: datetime