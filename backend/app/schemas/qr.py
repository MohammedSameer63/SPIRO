from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class HouseholdQrResponse(BaseModel):
    """
    Signed household QR code, per the API contract
    (GET /households/{id}/qr returns camelCase keys).
    """

    household_id: UUID = Field(serialization_alias="householdId")
    qr_payload: str = Field(serialization_alias="qrPayload")
    qr_code: str = Field(serialization_alias="qrCode")

    model_config = ConfigDict(
        from_attributes=True,
    )


class PickupValidationRequest(BaseModel):
    """
    Body for POST /worker/reports/{report_id}/validate-pickup.
    `qr_payload` is the raw string from scanning the household QR code.
    """

    qr_payload: str


class PickupValidationResponse(BaseModel):
    report_id: UUID
    household_id: UUID
    worker_id: UUID
    method: str
    validated_at: str

    model_config = ConfigDict(
        from_attributes=True,
    )
