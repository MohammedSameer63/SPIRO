from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_token,
)
from app.model.user import User, UserRole, UserStatus


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)

security = HTTPBearer()


# ============================================================
# Request Models
# ============================================================

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    phone: str | None = None

   # ============================================================
# POST /auth/register
# ============================================================

@router.post("/register")
def register(
    data: RegisterRequest,
    db: Session = Depends(get_db),
):
    # Check whether email already exists
    existing_user = (
        db.query(User)
        .filter(User.email == data.email)
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create new citizen
    user = User(
        name=data.name,
        email=data.email,
        phone=data.phone,
        password_hash=hash_password(data.password),
        role=UserRole.CITIZEN,
        status=UserStatus.ACTIVE,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "success": True,
        "message": "Registration successful.",
        "data": {
            "user": {
                "id": str(user.id),
                "name": user.name,
                "email": user.email,
                "role": user.role.value,
            }
        },
    }


# ============================================================
# POST /auth/login
# ============================================================

@router.post("/login")
def login(
    data: LoginRequest,
    db: Session = Depends(get_db),
):
    user = (
        db.query(User)
        .filter(User.email == data.email)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    if not verify_password(
        data.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account disabled",
        )

    token = create_access_token(
        {
            "sub": str(user.id),
            "role": user.role.value,
        }
    )

    return {
        "success": True,
        "message": "Login successful.",
        "data": {
            "token": token,
            "user": {
                "id": str(user.id),
                "name": user.name,
                "role": user.role.value,
            },
        },
    }


# ============================================================
# GET /auth/me
# ============================================================

@router.get("/me")
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    token = credentials.credentials

    try:
        payload = decode_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user_id = payload.get("sub")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return {
        "success": True,
        "data": {
            "id": str(user.id),
            "name": user.name,
            "email": user.email,
            "role": user.role.value,
        },
    }


# ============================================================
# POST /auth/logout
# ============================================================

@router.post("/logout")
def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    return {
        "success": True,
        "message": "Logged out successfully.",
    }