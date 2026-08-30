from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.schemas.user import UserResponse


class RegisterRequest(BaseModel):
    ward_id: UUID

    house_number: str = Field(
        min_length=1,
        max_length=20,
    )

    street_name: str = Field(
        min_length=1,
        max_length=255,
    )

    address: str = Field(
        min_length=5,
        max_length=1000,
    )

    name: str = Field(
        min_length=2,
        max_length=255,
    )

    email: EmailStr

    phone: str

    password: str = Field(
        min_length=8,
    )


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str

    token_type: str = "bearer"

    user: UserResponse

    model_config = ConfigDict(
        from_attributes=True
    )
