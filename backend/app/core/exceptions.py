class WardNotFoundError(Exception):
    """Raised when the requested ward does not exist."""


class EmailAlreadyRegisteredError(Exception):
    """Raised when an email is already registered."""
