import uuid

from sqlalchemy import Column, Text, Numeric, Enum as SQLEnum, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class WasteReport(Base):
    __tablename__ = "waste_reports"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    user_id = Column(
        UUID(as_uuid=True),
        nullable=False,
    )

    schedule_id = Column(
        UUID(as_uuid=True),
        nullable=True,
    )

    description = Column(
        Text,
        nullable=True,
    )

    image_url = Column(
        Text,
        nullable=False,
    )

    latitude = Column(
        Numeric(9, 6),
        nullable=False,
    )

    longitude = Column(
        Numeric(9, 6),
        nullable=False,
    )

    address = Column(
        Text,
        nullable=True,
    )

    status = Column(
        SQLEnum(
            "PENDING",
            "ACCEPTED",
            "IN_PROGRESS",
            "COMPLETED",
            name="report_status",
            create_type=False,
        ),
        nullable=False,
        default="PENDING",
    )

    created_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
    )

    updated_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
    )