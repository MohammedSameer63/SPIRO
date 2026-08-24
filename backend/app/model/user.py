import uuid
from enum import Enum

from sqlalchemy import Column, String, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class UserRole(str, Enum):
    CITIZEN = "CITIZEN"
    WORKER = "WORKER"
    ADMIN = "ADMIN"


class UserStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    SUSPENDED = "SUSPENDED"


class User(Base):
    __tablename__ = "users"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    household_id = Column(
        UUID(as_uuid=True),
        nullable=True,
    )

    name = Column(
        String(100),
        nullable=False,
    )

    email = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )

    phone = Column(
        String(15),
        nullable=True,
    )

    password_hash = Column(
        String(255),
        nullable=False,
    )

    role = Column(
        SQLEnum(
            UserRole,
            name="user_role",
            create_type=False,
        ),
        nullable=False,
    )

    status = Column(
        SQLEnum(
            UserStatus,
            name="user_status",
            create_type=False,
        ),
        nullable=False,
        default=UserStatus.ACTIVE,
    )