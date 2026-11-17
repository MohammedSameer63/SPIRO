from uuid import UUID

from sqlalchemy.exc import IntegrityError

from app.core.config import settings
from app.models import CoinTransaction
from app.repositories.coin_transaction_repository import (
    CoinTransactionRepository,
)
from app.repositories.coin_wallet_repository import CoinWalletRepository
from app.models import CoinWallet


# Ledger reason for awards granted when a report is completed.
REASON_COMPLETION = "completion"


class CoinService:
    """
    Green-coin rewards for citizens.

    Coins are awarded when a worker marks a report COMPLETED. A report can
    only ever earn one completion award (enforced by lookup + unique index),
    so retries and double clicks are safe.
    """

    def __init__(
        self,
        db,
        wallet_repository: CoinWalletRepository,
        transaction_repository: CoinTransactionRepository,
    ):
        self.db = db
        self.wallet_repository = wallet_repository
        self.transaction_repository = transaction_repository

    def award_completion(
        self,
        user_id: UUID,
        report_id: UUID,
        qr_validated: bool,
    ) -> CoinTransaction | None:
        """
        Grant the completion award (plus QR bonus when the pickup was
        validated by scan). Returns the transaction, or the existing one
        if this report was already rewarded.
        """
        existing = self.transaction_repository.find_completion_for_report(
            report_id
        )
        if existing is not None:
            return existing

        amount = settings.coins_per_completion + (
            settings.coins_qr_bonus if qr_validated else 0
        )
        if amount <= 0:
            return None

        note = (
            "Waste report completed (pickup QR-validated)."
            if qr_validated
            else "Waste report completed."
        )

        try:
            transaction = self.transaction_repository.create(
                CoinTransaction(
                    user_id=user_id,
                    amount=amount,
                    reason=REASON_COMPLETION,
                    report_id=report_id,
                    note=note,
                )
            )
        except IntegrityError:
            # Lost a race with a concurrent award for the same report.
            self.db.rollback()
            return self.transaction_repository.find_completion_for_report(
                report_id
            )

        wallet = self.wallet_repository.find_by_user_id(user_id)
        if wallet is None:
            wallet = self.wallet_repository.create(
                CoinWallet(user_id=user_id, balance=0)
            )
        wallet.balance = (wallet.balance or 0) + amount
        self.db.flush()

        return transaction

    def get_balance(self, user_id: UUID) -> int:
        wallet = self.wallet_repository.find_by_user_id(user_id)
        return wallet.balance if wallet is not None else 0

    def get_wallet_view(self, user_id: UUID, limit: int = 20) -> dict:
        transactions = self.transaction_repository.recent_by_user(
            user_id, limit=limit
        )
        return {
            "balance": self.get_balance(user_id),
            "total_earned": self.transaction_repository.total_earned_by_user(
                user_id
            ),
            "transactions": [
                {
                    "id": str(tx.id),
                    "amount": tx.amount,
                    "reason": tx.reason,
                    "report_id": str(tx.report_id)
                    if tx.report_id
                    else None,
                    "note": tx.note,
                    "created_at": tx.created_at.isoformat()
                    if tx.created_at
                    else None,
                }
                for tx in transactions
            ],
        }
