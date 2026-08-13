from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text
from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.household import Household


class Ward(Base):
    __tablename__ = "wards"

    id: Mapped[UUID] = mapped_column(
    	primary_key=True,
    	server_default=text("gen_random_uuid()"),
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    zone: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    households: Mapped[list["Household"]] = relationship(
        back_populates="ward",
    )
