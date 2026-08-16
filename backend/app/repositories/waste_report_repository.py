from uuid import UUID

from sqlalchemy import select

from app.models import WasteReport
from app.repositories.base_repository import BaseRepository


class WasteReportRepository(BaseRepository):
    """
    Handles database operations related to waste reports.
    """

    def create(
        self,
        report: WasteReport,
    ) -> WasteReport:
        """
        Persist a new waste report.
        """

        return self._persist(report)

    def find_by_id(
        self,
        report_id: UUID,
    ) -> WasteReport | None:
        """
        Retrieve a waste report by ID.
        """

        stmt = select(WasteReport).where(
            WasteReport.id == report_id
        )

        result = self.db.execute(stmt)

        return result.scalar_one_or_none()

    def find_by_user(
        self,
        user_id: UUID,
    ) -> list[WasteReport]:
        """
        Retrieve waste reports created by a user.
        """

        stmt = (
            select(WasteReport)
            .where(WasteReport.user_id == user_id)
            .order_by(WasteReport.created_at.desc())
        )

        result = self.db.execute(stmt)

        return list(result.scalars().all())