from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text
from sqlalchemy import Enum, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.enums.user_role import UserRole
from app.enums.user_status import UserStatus

if TYPE_CHECKING:
    from app.models.household import Household
    from app.models.waste_report import WasteReport


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(
    	primary_key=True,
    	server_default=text("gen_random_uuid()"),
    )

    household_id: Mapped[UUID | None] = mapped_column(
        ForeignKey(
            "households.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
    )

    phone: Mapped[str | None] = mapped_column(
        String(15),
        nullable=True,
    )

    password_hash: Mapped[str] =  mapped_column(
    	String(255),
    	nullable=False,
    )

    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role"),
        nullable=False,
    )

    status: Mapped[UserStatus] = mapped_column(
        Enum(UserStatus, name="user_status"),
        server_default="ACTIVE",
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        server_default=func.current_timestamp(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=False,
    )

    household: Mapped["Household | None"] = relationship(
        back_populates="users",
    )

    waste_reports: Mapped[list["WasteReport"]] = relationship(
        back_populates="user",
    )