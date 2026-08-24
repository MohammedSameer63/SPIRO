from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey, Numeric, String, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.waste_category import WasteCategory
    from app.models.waste_report import WasteReport


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )

    report_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "waste_reports.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        unique=True,
    )

    category_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "waste_categories.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    confidence: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 2),
        nullable=True,
    )

    model_used: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    prediction_time: Mapped[datetime] = mapped_column(
        server_default=func.current_timestamp(),
        nullable=False,
    )

    report: Mapped["WasteReport"] = relationship(
        back_populates="prediction",
    )

    category: Mapped["WasteCategory"] = relationship()