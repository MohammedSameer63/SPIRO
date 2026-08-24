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

class ReportNotFoundError(Exception):
    """Raised when the requested waste report does not exist."""


class WasteCategoryNotFoundError(Exception):
    """Raised when the requested waste category does not exist."""


class PredictionAlreadyExistsError(Exception):
    """Raised when a report already has a prediction."""