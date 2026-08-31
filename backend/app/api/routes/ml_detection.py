from io import BytesIO

from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image

from app.services.yolo_service import YOLOService


router = APIRouter(
    prefix="/ml",
    tags=["ML Detection"],
)

yolo_service = YOLOService()


@router.post("/detect")
async def detect_waste(
    image: UploadFile = File(...),
):
    if image.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG, PNG and WEBP images are supported.",
        )

    contents = await image.read()

    try:
        pil_image = Image.open(
            BytesIO(contents)
        ).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        )

    return yolo_service.predict(pil_image)