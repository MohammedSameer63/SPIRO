from __future__ import annotations

from uuid import UUID
from typing import TYPE_CHECKING

from sqlalchemy import String, Text, text
from sqlalchemy.orm import Mapped, mapped_column,relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.collection_schedule import CollectionSchedule

class WasteCategory(Base):
    __tablename__ = "waste_categories"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )

    name: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        unique=True,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    collection_schedules: Mapped[list["CollectionSchedule"]] = relationship(
        back_populates="waste_category",
    )