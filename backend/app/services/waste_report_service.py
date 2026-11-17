from fastapi import UploadFile
from pathlib import Path
from sqlalchemy.orm import Session

from app.core.config import settings
from app.enums.audit_event import AuditEvent
from app.models import User, WasteReport
from app.repositories.prediction_repository import PredictionRepository
from app.repositories.waste_category_repository import WasteCategoryRepository
from app.repositories.waste_report_repository import (
    WasteReportRepository,
)
from app.schemas.prediction import PredictionResult
from app.schemas.waste_report import WasteReportCreateRequest
from app.services.audit_log_service import AuditLogService
from app.services.file_storage_service import FileStorageService
from app.services.ml_inference_service import ml_inference_service
from app.services.prediction_service import PredictionService
from app.core.exceptions import (
    PredictionAlreadyExistsError,
    ReportNotFoundError,
)

class WasteReportService:
    """
    Handles waste report business logic.
    """

    def __init__(
        self,
        db: Session,
        waste_report_repository: WasteReportRepository,
        file_storage_service: FileStorageService,
        audit_log_service: AuditLogService,
    ):
        self.db = db
        self.waste_report_repository = (
            waste_report_repository
        )
        self.file_storage_service = (
            file_storage_service
        )
        self.audit_log_service = audit_log_service

    async def create_report(
        self,
        request: WasteReportCreateRequest,
        image: UploadFile,
        current_user: User,
    ) -> WasteReport:
        """
        Create a new waste report for the authenticated user.
        Triggers ML inference after successful creation.
        """

        image_url = (
            await self.file_storage_service.save_report_image(
                image
            )
        )

        address = request.address or (
            self._build_household_address(
                current_user
            )
        )

        report = WasteReport(
            user_id=current_user.id,
            description=request.description,
            image_url=image_url,
            latitude=request.latitude,
            longitude=request.longitude,
            address=address,
        )

        try:
            report = self.waste_report_repository.create(
                report
            )

            self.audit_log_service.create_log(
                event_type=AuditEvent.REPORT_CREATED,
                entity_type="waste_report",
                entity_id=report.id,
                performed_by=current_user.id,
            )

            self.db.commit()

            # Trigger ML inference asynchronously (fire and forget)
            self._trigger_ml_inference(report)

            return report

        except Exception:
            self.db.rollback()
            raise

    def _build_household_address(
        self,
        current_user: User,
    ) -> str | None:
        """
        Build the full address from the citizen's
        linked household when available.
        """

        household = current_user.household

        if household is None:
            return None

        address_parts = [
            household.house_number,
            household.street_name,
            household.address,
        ]

        return ", ".join(
            part
            for part in address_parts
            if part
        )

    def _resolve_image_path(self, image_url: str) -> Path | None:
        """Resolve a stored image_url to a filesystem path.

        image_url is stored as "/uploads/reports/<file>". Depending on how
        the server was started, settings.upload_dir may be relative
        ("uploads") or absolute. Try candidates in order.
        """
        relative = image_url.lstrip("/")
        backend_dir = Path(__file__).parent.parent.parent
        candidates = [
            Path.cwd() / relative,
            backend_dir / relative,
            Path(settings.upload_dir) / Path(relative).name
            if Path(relative).name
            else None,
            backend_dir / "uploads" / "reports" / Path(relative).name,
        ]
        # Absolute upload_dir case
        if Path(settings.upload_dir).is_absolute():
            candidates.insert(
                0, Path(settings.upload_dir) / Path(relative).name
            )
        for candidate in candidates:
            if candidate is not None and candidate.exists():
                return candidate
        return None

    def _trigger_ml_inference(self, report: WasteReport) -> None:
        """
        Trigger ML inference for a waste report.
        Runs synchronously for now; can be made async with a task queue.
        """
        if not settings.ml_enabled:
            return

        try:
            image_path = self._resolve_image_path(report.image_url)

            if image_path is None:
                print(f"Image not found for ML inference: {report.image_url}")
                return

            # Run inference
            top_prediction = ml_inference_service.get_top_prediction(str(image_path))

            if not top_prediction:
                # The model found nothing above threshold. Record an
                # explicit "unclassified" result (Reject stream) so the
                # citizen still sees an AI outcome instead of a report
                # stuck forever in "processing".
                print(f"ML found no waste for report {report.id}; saving unclassified result")
                self._save_unclassified_prediction(report)
                return

            # Map waste_stream -> waste_category UUID
            category_id = self._map_stream_to_category(
                top_prediction.get("waste_stream")
            )

            if not category_id:
                print(f"No category mapping for stream {top_prediction.get('waste_stream')}")
                return

            # Save prediction to database.
            # DB CHECK constraint requires confidence in 0-100; the ML
            # service returns 0-1, so scale it here.
            raw_confidence = top_prediction["confidence"] or 0
            confidence_100 = round(float(raw_confidence) * 100, 2)
            confidence_100 = max(0.0, min(100.0, confidence_100))

            self._save_prediction_result(
                report=report,
                category_id=category_id,
                confidence=confidence_100,
                model_used=f"{top_prediction.get('method', 'spiro_pipeline')}",
            )
            print(f"ML prediction saved for report {report.id}: class={top_prediction['class_name']}, stream={top_prediction.get('waste_stream')}, confidence={confidence_100:.2f}")

        except Exception as e:
            # Log but don't fail the report creation
            print(f"ML inference failed for report {report.id}: {e}")

    def _save_prediction_result(
        self,
        report: WasteReport,
        category_id,
        confidence: float,
        model_used: str,
    ) -> None:
        """Persist a prediction; silently skip if one already exists."""
        prediction_service = PredictionService(
            db=self.db,
            prediction_repository=PredictionRepository(self.db),
            waste_report_repository=WasteReportRepository(self.db),
            waste_category_repository=WasteCategoryRepository(self.db),
        )

        prediction_result = PredictionResult(
            report_id=report.id,
            category_id=category_id,
            confidence=confidence,
            model_used=model_used,
        )

        try:
            prediction_service.create_prediction(prediction_result)
        except PredictionAlreadyExistsError:
            pass

    def _save_unclassified_prediction(self, report: WasteReport) -> None:
        """Record an explicit 'could not classify' result (Reject stream)."""
        category_id = self._map_stream_to_category("reject")
        if not category_id:
            print(f"No Reject category configured; skipping unclassified result for {report.id}")
            return
        self._save_prediction_result(
            report=report,
            category_id=category_id,
            confidence=0,
            model_used="yolo_no_detection",
        )

    def analyze_report(self, report_id, current_user: User) -> WasteReport:
        """
        (Re-)run AI analysis for a citizen's report right now.

        Returns the report with its prediction (existing or newly
        created). Used by the citizen "Retry AI analysis" action for
        reports that show no AI result.
        """
        report = self.get_report(
            report_id=report_id,
            current_user=current_user,
        )

        if not settings.ml_enabled:
            raise ValueError("ML inference is disabled.")

        existing = PredictionRepository(self.db).find_by_report_id(
            report.id
        )
        if existing is not None:
            return report

        image_path = self._resolve_image_path(report.image_url)
        if image_path is None:
            raise ValueError("Report image not found.")

        top_prediction = ml_inference_service.get_top_prediction(
            str(image_path)
        )

        if not top_prediction:
            self._save_unclassified_prediction(report)
        else:
            category_id = self._map_stream_to_category(
                top_prediction.get("waste_stream")
            )
            if not category_id:
                raise ValueError("Could not map AI result to a category.")
            raw_confidence = top_prediction["confidence"] or 0
            confidence_100 = max(
                0.0, min(100.0, round(float(raw_confidence) * 100, 2))
            )
            self._save_prediction_result(
                report=report,
                category_id=category_id,
                confidence=confidence_100,
                model_used=f"{top_prediction.get('method', 'spiro_pipeline')}",
            )

        self.db.expire_all()
        return self.get_report(
            report_id=report_id,
            current_user=current_user,
        )

    # waste_categories stores disposal streams ("Recoverable", "Organic",
    # ...), NOT 109 fine-grained classes. Map via waste_stream so a
    # plastic_bottle prediction lands on "Recoverable" instead of the
    # first alphabetical category ("E-Waste").
    STREAM_TO_CATEGORY = {
        "organic": "organic",
        "recoverable": "recoverable",
        "non_recoverable": "non-recoverable",
        "hazardous": "hazardous",
        "sanitary": "sanitary",
        "e_waste": "e-waste",
        "reject": "reject",
    }

    def _map_stream_to_category(self, waste_stream: str | None):
        """Map an ML waste_stream to a waste_category UUID."""
        from app.repositories.waste_category_repository import WasteCategoryRepository

        repo = WasteCategoryRepository(self.db)
        all_categories = repo.find_all()
        if not all_categories:
            return None
        if not waste_stream:
            return all_categories[0].id

        normalized = waste_stream.strip().lower().replace("_", "-")
        # Direct stream-name match (handles "e_waste" vs "E-Waste",
        # "non_recoverable" vs "Non-Recoverable").
        for category in all_categories:
            if category.name.strip().lower().replace("_", "-") == normalized:
                return category.id
        # Alias without separators ("nonrecoverable", "ewaste")
        compact = normalized.replace("-", "").replace(" ", "")
        for category in all_categories:
            cat_compact = (
                category.name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
            )
            if cat_compact == compact:
                return category.id
        return all_categories[0].id

    def _map_class_to_category(self, class_id: int):
        """
        Legacy entry point — kept for backwards compatibility.
        Resolves the class name via the ML service (which reads the real
        ONNX label map) and then maps via waste stream.
        """
        try:
            names = ml_inference_service._class_names
            class_name = (
                names[class_id]
                if 0 <= class_id < len(names)
                else f"class_{class_id}"
            )
        except Exception:
            class_name = f"class_{class_id}"
        waste_stream = ml_inference_service._get_waste_stream_for_name(class_name)
        return self._map_stream_to_category(waste_stream)

    def _get_class_name(self, class_id: int) -> str:
        """Get class name from the loaded ML model label map."""
        try:
            names = ml_inference_service._class_names
            if 0 <= class_id < len(names):
                return names[class_id]
        except Exception:
            pass
        return f"class_{class_id}"

    def get_my_reports(
        self,
        current_user: User,
    ) -> list[WasteReport]:
        """
        Retrieve all waste reports created by the authenticated user.
        """

        return self.waste_report_repository.find_by_user(
            current_user.id
        )

    def get_report(
        self,
        report_id,
        current_user: User,
    ) -> WasteReport:
        """
        Retrieve a specific waste report belonging
        to the authenticated user.
        """

        report = self.waste_report_repository.find_by_id(
            report_id
        )

        if report is None:
            raise ReportNotFoundError("Waste report not found.")

        if report.user_id != current_user.id:
            raise ReportNotFoundError("Waste report not found.")

        return report