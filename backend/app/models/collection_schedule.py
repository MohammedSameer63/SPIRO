from __future__ import annotations

from datetime import time
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey, String, Time, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.ward import Ward
    from app.models.waste_category import WasteCategory


class CollectionSchedule(Base):
    __tablename__ = "collection_schedules"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )

    ward_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "wards.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    waste_category_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "waste_categories.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    day_of_week: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    start_time: Mapped[time | None] = mapped_column(
        Time,
        nullable=True,
    )

    end_time: Mapped[time | None] = mapped_column(
        Time,
        nullable=True,
    )

    ward: Mapped["Ward"] = relationship(
        back_populates="collection_schedules",
    )

    waste_category: Mapped["WasteCategory"] = relationship(
        back_populates="collection_schedules",
    )