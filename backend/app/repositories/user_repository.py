from uuid import UUID

from sqlalchemy import select

from app.models import User
from app.repositories.base_repository import BaseRepository


class UserRepository(BaseRepository):
    """
    Handles all database operations related to the User model.
    """

    def find_by_id(
        self,
        user_id: UUID,
    ) -> User | None:
        """
        Retrieve a user by ID.
        """

        stmt = select(User).where(User.id == user_id)

        result = self.db.execute(stmt)

        return result.scalar_one_or_none()


    def find_by_email(
        self,
        email: str,
    ) -> User | None:
        """
        Retrieve a user by email.
        """

        stmt = select(User).where(User.email == email)

        result = self.db.execute(stmt)

        return result.scalar_one_or_none()


    def create(self, user: User) -> User:
    	return self._persist(user)


    def update(
        self,
        user: User,
    ) -> User:
        """
        Persist changes to an existing user.
        """

        self.db.flush()

        self.db.refresh(user)

        return user


    def delete(self, user: User) -> None:
    	self._remove(user)
