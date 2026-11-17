from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.coin_transaction import CoinTransaction
from app.repositories.base_repository import BaseRepository


class CoinTransactionRepository(BaseRepository):
    """
    Repository for the green-coin ledger.
    """

    def __init__(self, db: Session):
        super().__init__(db)

    def create(
        self,
        transaction: CoinTransaction,
    ) -> CoinTransaction:
        return self._persist(transaction)

    def find_completion_for_report(
        self,
        report_id: UUID,
    ) -> CoinTransaction | None:
        """
        Return the completion award for a report, if one was already granted.
        Used to keep awards idempotent across retries / double clicks.
        """
        return (
            self.db.query(CoinTransaction)
            .filter(
                CoinTransaction.report_id == report_id,
                CoinTransaction.reason == "completion",
            )
            .first()
        )

    def total_earned_by_user(
        self,
        user_id: UUID,
    ) -> int:
        total = (
            self.db.query(func.coalesce(func.sum(CoinTransaction.amount), 0))
            .filter(
                CoinTransaction.user_id == user_id,
                CoinTransaction.amount > 0,
            )
            .scalar()
        )
        return int(total or 0)

    def recent_by_user(
        self,
        user_id: UUID,
        limit: int = 20,
    ) -> list[CoinTransaction]:
        return (
            self.db.query(CoinTransaction)
            .filter(CoinTransaction.user_id == user_id)
            .order_by(CoinTransaction.created_at.desc())
            .limit(limit)
            .all()
        )

    def sum_for_report(
        self,
        report_id: UUID,
    ) -> int:
        total = (
            self.db.query(func.coalesce(func.sum(CoinTransaction.amount), 0))
            .filter(CoinTransaction.report_id == report_id)
            .scalar()
        )
        return int(total or 0)
