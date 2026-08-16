from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.exceptions import InvalidReportImageError

class FileStorageService:
    """
    Handles local filesystem storage for uploaded files.
    """

    ALLOWED_IMAGE_TYPES = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    def __init__(
        self,
        upload_directory: Path,
        max_image_size_mb: int,
    ):
        self.upload_directory = upload_directory
        self.max_image_size_bytes = (
            max_image_size_mb * 1024 * 1024
        )

    async def save_report_image(
        self,
        image: UploadFile,
    ) -> str:
        """
        Validate and save a report image.
        """

        if image.content_type not in self.ALLOWED_IMAGE_TYPES:
            raise InvalidReportImageError(
                "Only JPEG, PNG, and WEBP images are allowed."
            )

        extension = Path(
            image.filename or ""
        ).suffix.lower()

        if not extension:
            raise InvalidReportImageError(
                "Uploaded image must have a valid file extension."
            )

        filename = f"{uuid4()}{extension}"

        self.upload_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_path = self.upload_directory / filename

        contents = await image.read()

        if len(contents) > self.max_image_size_bytes:
            raise InvalidReportImageError(
                "Image size must not exceed "
                f"{self.max_image_size_bytes // (1024 * 1024)} MB."
            )

        file_path.write_bytes(contents)

        return f"/uploads/reports/{filename}"