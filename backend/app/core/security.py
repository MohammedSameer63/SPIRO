from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings


# Password hashing configuration
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


def hash_password(password: str) -> str:
    """
    Hash a plain-text password.
    """
    return pwd_context.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    """
    Verify a plain-text password against its hashed version.
    """
    return pwd_context.verify(
        plain_password,
        hashed_password,
    )


def create_access_token(
    user_id: UUID,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Generate a JWT access token for the given user.
    """

    expire = datetime.now(timezone.utc) + (
        expires_delta
        if expires_delta is not None
        else timedelta(
            minutes=settings.access_token_expire_minutes
        )
    )

    payload = {
        "sub": str(user_id),
        "exp": expire,
    }

    encoded_jwt = jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    return encoded_jwt


def decode_token(
    token: str,
) -> dict[str, Any]:
    """
    Decode and verify a JWT token.

    Raises:
        JWTError:
            If the token is invalid or expired.
    """

    payload = jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
    )

    return payload
