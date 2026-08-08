from sqlalchemy.orm import Session

from app.core.exceptions import (
    EmailAlreadyRegisteredError,
    WardNotFoundError,
)
from app.core.security import (
    create_access_token,
    hash_password,
)
from app.models import Household, User
from app.repositories.household_repository import HouseholdRepository
from app.repositories.user_repository import UserRepository
from app.repositories.ward_repository import WardRepository
from app.schemas.auth import (
    AuthResponse,
    RegisterRequest,
)
from app.schemas.user import UserResponse
from app.enums.user_role import UserRole


class AuthService:
    """
    Handles authentication business logic.
    """

    def __init__(
        self,
        db: Session,
        user_repository: UserRepository,
        household_repository: HouseholdRepository,
        ward_repository: WardRepository,
    ):
        self.db = db
        self.user_repository = user_repository
        self.household_repository = household_repository
        self.ward_repository = ward_repository

    def register(
        self,
        request: RegisterRequest,
    ) -> AuthResponse:
        """
        Register a new citizen account.
        """

        try:

            # Verify ward exists
            ward = self.ward_repository.find_by_id(
                request.ward_id
            )

            if ward is None:
                raise WardNotFoundError("Ward does not exist.")

            # Check email availability
            existing_user = self.user_repository.find_by_email(
                request.email
            )

            if existing_user is not None:
                raise EmailAlreadyRegisteredError(
                    "Email is already registered."
                )

            # Find existing household
            household = (
                self.household_repository.find_by_location(
                    ward_id=request.ward_id,
                    house_number=request.house_number,
                    street_name=request.street_name,
                )
            )

            # Create household if needed
            if household is None:
                household = Household(
                    ward_id=request.ward_id,
                    house_number=request.house_number,
                    street_name=request.street_name,
                    address=request.address,
                )

                household = (
                    self.household_repository.create(
                        household
                    )
                )

            # Hash password
            password_hash = hash_password(
                request.password
            )

            # Create user
            user = User(
                household_id=household.id,
                name=request.name,
                email=request.email,
                phone=request.phone,
                password_hash=password_hash,
                role=UserRole.CITIZEN,
            )

            user = self.user_repository.create(user)

            # Commit transaction
            self.db.commit()

            # Generate JWT
            access_token = create_access_token(
                user_id=user.id,
            )

            # Return response
            return AuthResponse(
                access_token=access_token,
                token_type="bearer",
                user=UserResponse.model_validate(user),
            )

        except Exception:

            self.db.rollback()

            raise
