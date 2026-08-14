from __future__ import annotations

from uuid import UUID

from sqlalchemy import Boolean, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


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