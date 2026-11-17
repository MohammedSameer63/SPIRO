from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, String, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class CoinTransaction(Base):
    """
    Append-only ledger of green-coin movements.
    `report_id` is set for completion awards; the partial unique index in
    schema.sql guarantees at most one completion award per report.
    """

    __tablename__ = "coin_transactions"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    amount: Mapped[int] = mapped_column(
        nullable=False,
    )

    reason: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    report_id: Mapped[UUID | None] = mapped_column(
        ForeignKey(
            "waste_reports.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    note: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        server_default=func.current_timestamp(),
        nullable=False,
    )
