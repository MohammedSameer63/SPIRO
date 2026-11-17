from pydantic import BaseModel, ConfigDict


class CoinTransactionView(BaseModel):
    id: str
    amount: int
    reason: str
    report_id: str | None = None
    note: str | None = None
    created_at: str | None = None

    model_config = ConfigDict(
        from_attributes=True,
    )


class CoinWalletView(BaseModel):
    balance: int
    total_earned: int
    transactions: list[CoinTransactionView]

    model_config = ConfigDict(
        from_attributes=True,
    )
