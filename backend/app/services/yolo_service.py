from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from app.core.config import settings


class YOLOService:
    def __init__(self):
        model_path = Path(settings.yolo_model_path)

        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parents[3] / model_path

        self.model = YOLO(str(model_path))

    def predict(self, image: Image.Image):
        results = self.model.predict(
            source=image,
            imgsz=512,
            conf=0.25,
            device=0,
            verbose=False,
        )

        result = results[0]

        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": round(confidence * 100, 2),
                    "box": {
                        "x1": round(float(box.xyxy[0][0]), 2),
                        "y1": round(float(box.xyxy[0][1]), 2),
                        "x2": round(float(box.xyxy[0][2]), 2),
                        "y2": round(float(box.xyxy[0][3]), 2),
                    },
                }
            )

        return {
            "image_width": image.width,
            "image_height": image.height,
            "detections": detections,
            "count": len(detections),
            "model": "spiro_yolo11n",
        }