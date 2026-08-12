class WardNotFoundError(Exception):
    """Raised when the requested ward does not exist."""


class EmailAlreadyRegisteredError(Exception):
    """Raised when an email is already registered."""

class InvalidCredentialsError(Exception):
    """Raised when login credentials are invalid."""

class AccountNotActiveError(Exception):
    """Raised when a non-active user attempts to log in."""