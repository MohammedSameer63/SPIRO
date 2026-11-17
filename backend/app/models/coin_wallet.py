from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, func, text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class CoinWallet(Base):
    """
    Green-coin balance per citizen. Rows are created lazily on first award.
    The authoritative history lives in coin_transactions; balance is a
    cached sum kept in the same DB transaction.
    """

    __tablename__ = "coin_wallets"

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )

    balance: Mapped[int] = mapped_column(
        nullable=False,
        server_default=text("0"),
    )

    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=False,
    )
