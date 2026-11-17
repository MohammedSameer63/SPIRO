from uuid import UUID

from sqlalchemy.orm import Session

from app.models.coin_wallet import CoinWallet
from app.repositories.base_repository import BaseRepository


class CoinWalletRepository(BaseRepository):
    """
    Repository for green-coin wallet persistence.
    """

    def __init__(self, db: Session):
        super().__init__(db)

    def find_by_user_id(
        self,
        user_id: UUID,
    ) -> CoinWallet | None:
        return (
            self.db.query(CoinWallet)
            .filter(CoinWallet.user_id == user_id)
            .first()
        )

    def create(
        self,
        wallet: CoinWallet,
    ) -> CoinWallet:
        return self._persist(wallet)
