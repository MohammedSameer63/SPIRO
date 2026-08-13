from uuid import UUID

from app.core.exceptions import (
    HouseholdAccessDeniedError,
    HouseholdNotFoundError,
)
from app.enums.user_role import UserRole
from app.models import Household, User
from app.repositories.household_repository import HouseholdRepository


class HouseholdService:
    """
    Handles household-related business logic.
    """

    def __init__(
        self,
        household_repository: HouseholdRepository,
    ):
        self.household_repository = household_repository

    def get_all_households(
        self,
        current_user: User,
        ward_id: UUID | None = None,
    ) -> list[Household]:
        """
        Retrieve households according to the user's role.
        """

        if current_user.role == UserRole.CITIZEN:
            household = self.household_repository.find_by_id(
                current_user.household_id
            )

            if household is None:
                raise HouseholdNotFoundError(
                    "Household not found."
                )

            return [household]

        if current_user.role in (
            UserRole.WORKER,
            UserRole.ADMIN,
        ):
            return self.household_repository.find_all(
                ward_id=ward_id
            )

        raise HouseholdAccessDeniedError(
            "You do not have permission to access households."
        )

    def get_household_by_id(
        self,
        household_id: UUID,
        current_user: User,
    ) -> Household:
        """
        Retrieve a household according to the user's role.
        """

        household = self.household_repository.find_by_id(
            household_id
        )

        if household is None:
            raise HouseholdNotFoundError(
                "Household not found."
            )

        if current_user.role == UserRole.CITIZEN:
            if current_user.household_id != household.id:
                raise HouseholdAccessDeniedError(
                    "You do not have permission to access this household."
                )

        elif current_user.role not in (
            UserRole.WORKER,
            UserRole.ADMIN,
        ):
            raise HouseholdAccessDeniedError(
                "You do not have permission to access households."
            )

        return household