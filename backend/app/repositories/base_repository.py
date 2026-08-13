from abc import ABC
from typing import TypeVar

from sqlalchemy.orm import Session

T = TypeVar("T")


class BaseRepository(ABC):
    """
    Base class for all repositories.

    Provides common persistence operations and access
    to the shared SQLAlchemy session.
    """

    def __init__(self, db: Session):
        self.db = db

    def _persist(self, entity: T) -> T:
        """
        Add an entity to the current transaction.
        """

        self.db.add(entity)
        self.db.flush()
        self.db.refresh(entity)

        return entity

    def _remove(self, entity: T) -> None:
        """
        Remove an entity from the current transaction.
        """

        self.db.delete(entity)
        self.db.flush()

    def commit(self) -> None:
        """
        Commit the current transaction.
        """

        self.db.commit()

    def rollback(self) -> None:
        """
        Roll back the current transaction.
        """

        self.db.rollback()
