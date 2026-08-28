from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AssignmentCreateRequest(BaseModel):
    report_id: UUID
    worker_id: UUID

class AssignmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    report_id: UUID
    worker_id: UUID
    assigned_by: UUID
    assigned_at: datetime
    completed_at: datetime | None