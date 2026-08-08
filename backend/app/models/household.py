from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text
from sqlalchemy import ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.ward import Ward


class Household(Base):
    __tablename__ = "households"

    id: Mapped[UUID] = mapped_column(
    	primary_key=True,
    	server_default=text("gen_random_uuid()"),
    )

    ward_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "wards.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    house_number: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    street_name: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    address: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        server_default=func.current_timestamp(),
        nullable=False,
    )

    ward: Mapped["Ward"] = relationship(
        back_populates="households",
    )

    users: Mapped[list["User"]] = relationship(
        back_populates="household",
    )
