from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.waste_report import WasteReport


class Assignment(Base):
    __tablename__ = "assignments"

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

    worker_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "users.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    assigned_by: Mapped[UUID] = mapped_column(
        ForeignKey(
            "users.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    assigned_at: Mapped[datetime] = mapped_column(
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        nullable=True,
    )

    report: Mapped["WasteReport"] = relationship(
        back_populates="assignment",
    )

    worker: Mapped["User"] = relationship(
        foreign_keys=[worker_id],
        back_populates="assignments",
    )

    assigned_by_user: Mapped["User"] = relationship(
        foreign_keys=[assigned_by],
        back_populates="created_assignments",
    )