from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey, String, func, text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

if TYPE_CHECKING:
    pass


class PickupValidation(Base):
    """
    Records that a worker scanned the household QR code at pickup time.
    One row per report (a report is collected at most once).
    """

    __tablename__ = "pickup_validations"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )

    report_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "waste_reports.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        unique=True,
    )

    household_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "households.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    worker_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "users.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    method: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default=text("'QR_SCAN'"),
    )

    validated_at: Mapped[datetime] = mapped_column(
        server_default=func.current_timestamp(),
        nullable=False,
    )
