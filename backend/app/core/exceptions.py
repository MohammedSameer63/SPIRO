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

class InvalidCollectionScheduleError(Exception):
    """Raised when a collection schedule contains invalid data."""

class AssignmentNotFoundError(Exception):
    """Raised when an assignment cannot be found."""
    
    pass

class AssignmentAlreadyExistsError(Exception):
    """Raised when a report is already assigned."""
    
    pass

class InvalidWorkerError(Exception):
    """Raised when a user cannot be assigned as a worker."""

    pass

class WorkerWardAssignmentNotFoundError(Exception):
    """
    Raised when a worker-ward assignment does not exist.
    """
    pass


class WorkerWardAlreadyExistsError(Exception):
    """
    Raised when a worker is already assigned to a ward.
    """
    pass


class WorkerNotFoundError(Exception):
    """
    Raised when the specified worker does not exist.
    """
    pass