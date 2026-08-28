from typing import Any
from uuid import UUID

from app.enums.audit_event import AuditEvent
from app.models.audit_log import AuditLog
from app.repositories.audit_log_repository import AuditLogRepository


class AuditLogService:
    """
    Handles creation of audit log entries.
    """

    def __init__(
        self,
        audit_log_repository: AuditLogRepository,
    ):
        self.audit_log_repository = audit_log_repository

    def create_log(
        self,
        event_type: AuditEvent,
        entity_type: str | None = None,
        entity_id: UUID | None = None,
        performed_by: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditLog:

        audit_log = AuditLog(
            event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            performed_by=performed_by,
            event_metadata=metadata or {},
        )

        return self.audit_log_repository.create(audit_log)