from uuid import UUID

from sqlalchemy.orm import Session

from app.models.audit_log import AuditLog
from app.repositories.base_repository import BaseRepository


class AuditLogRepository(BaseRepository):
    """
    Repository for audit log persistence.
    """

    def __init__(self, db: Session):
        super().__init__(db)

    def create(
        self,
        audit_log: AuditLog,
    ) -> AuditLog:
        return self._persist(audit_log)

    def find_by_id(
        self,
        audit_log_id: UUID,
    ) -> AuditLog | None:
        return (
            self.db.query(AuditLog)
            .filter(AuditLog.id == audit_log_id)
            .first()
        )