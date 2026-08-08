from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.enums.user_role import UserRole
from app.enums.user_status import UserStatus


class UserResponse(BaseModel):
    id: UUID
    household_id: UUID | None

    name: str
    email: str
    phone: str

    role: UserRole
    status: UserStatus

    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )
