class WardNotFoundError(Exception):
    """Raised when the requested ward does not exist."""


class EmailAlreadyRegisteredError(Exception):
    """Raised when an email is already registered."""

class InvalidCredentialsError(Exception):
    """Raised when login credentials are invalid."""

class AccountNotActiveError(Exception):
    """Raised when a non-active user attempts to log in."""

class HouseholdNotFoundError(Exception):
    """Raised when the requested household does not exist."""

class HouseholdAccessDeniedError(Exception):
    """Raised when a user is not allowed to access a household."""

class InvalidReportImageError(Exception):
    """Raised when an uploaded report image is invalid."""