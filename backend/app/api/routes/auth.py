from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.repositories.household_repository import HouseholdRepository
from app.repositories.user_repository import UserRepository
from app.repositories.ward_repository import WardRepository
from app.schemas.auth import (
    AuthResponse,
    RegisterRequest,
)
from app.services.auth_service import AuthService

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)

@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=201,
)

def register(
    request: RegisterRequest,
    db: Session = Depends(get_db),
):
	user_repository = UserRepository(db)

	household_repository = HouseholdRepository(db)

	ward_repository = WardRepository(db)

	service = AuthService(
	    db=db,
	    user_repository=user_repository,
	    household_repository=household_repository,
	    ward_repository=ward_repository,
	)

	return service.register(request)
