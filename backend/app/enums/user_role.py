from enum import Enum


class UserRole(str, Enum):
    CITIZEN = "CITIZEN"
    WORKER = "WORKER"
    ADMIN = "ADMIN"
