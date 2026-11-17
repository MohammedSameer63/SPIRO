"""
ML Inference Service for SPIRO Backend.
Supports YOLO-only inference via ONNX Runtime when EfficientNet not available.
Falls back to mock mode if no models.
"""

import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import numpy as np
import cv2
from loguru import logger

from app.core.config import settings


class MLInferenceService:
    """
    Service for running ML inference on waste images.
    - Uses ONNX Runtime for YOLO detection when model available
    - Falls back to mock mode if models fail
    """

    # Mock predictions for testing (class_id, class_name, waste_stream)
    # Matches the 34-class production YOLO model IDs
    MOCK_PREDICTIONS = [
        (0, "plastic_bottle", "recoverable"),
        (12, "glass_bottle", "recoverable"),
        (16, "metal_can_beverage", "recoverable"),
        (22, "cardboard_fragment", "recoverable"),
        (28, "food_waste_generic", "organic"),
        (31, "battery", "e_waste"),
        (21, "aerosol_can", "hazardous"),
        (33, "unknown_litter", "reject"),
        (29, "cigarette_butt", "non_recoverable"),
        (30, "shoe", "non_recoverable"),
    ]

    # Canonical SPIRO waste-stream lookup by class NAME.
    # Works for both the 34-class production YOLO model and the full
    # 109-class taxonomy — names are stable, numeric ids are not.
    ORGANIC_NAMES = {
        "food_waste_generic", "fruit_peel", "vegetable_scraps",
        "eggshell", "coffee_grounds", "tea_bag", "leaf_litter",
        "twig_branch", "soil_bag",
    }
    RECOVERABLE_NAMES = {
        "plastic_bottle", "plastic_bottle_cap", "plastic_bag",
        "plastic_film", "plastic_straw", "plastic_cup", "plastic_lid",
        "plastic_cutlery_fork", "plastic_cutlery_knife",
        "plastic_cutlery_spoon", "plastic_container", "plastic_tray",
        "plastic_packaging", "plastic_wrap", "plastic_tube",
        "plastic_bucket", "glass_bottle", "glass_jar", "glass_cup",
        "metal_can_beverage", "metal_can_food", "metal_bottle_cap",
        "metal_lid", "metal_foil", "metal_tin", "cardboard_box",
        "cardboard_fragment", "paper_bag", "newspaper", "magazine",
        "paper_packaging", "paper_straw", "receipt", "carton_drink",
        "coffee_pod", "wood_scrap", "pallet", "pizza_box",
    }
    HAZARDOUS_NAMES = {
        "aerosol_can", "paint_can", "chemical_container",
        "oil_container", "propane_tank_small",
    }
    SANITARY_NAMES = {
        "mask_disposable", "glove_disposable", "syringe",
        "medicine_blister", "bandage", "sanitary_pad", "diaper",
    }
    E_WASTE_NAMES = {
        "battery", "cable_wire", "circuit_board",
        "mobile_phone", "earphone",
    }
    REJECT_NAMES = {"unknown_litter"}

    # Fallback class names matching the production 34-class YOLO model
    # (read from models/exports/spiro_yolov11n_best.onnx metadata).
    # Preferred source is always the ONNX metadata itself (see initialize()).
    MODEL_34_CLASS_NAMES = [
        "plastic_bottle", "plastic_bottle_cap", "plastic_bag",
        "plastic_film", "plastic_straw", "plastic_cup", "plastic_lid",
        "plastic_container", "plastic_packaging", "plastic_tube",
        "polystyrene_cup", "polystyrene_container",
        "glass_bottle", "glass_jar", "glass_fragment", "glass_cup",
        "metal_can_beverage", "metal_can_food", "metal_bottle_cap",
        "metal_lid", "metal_foil", "aerosol_can",
        "cardboard_fragment", "paper_bag", "paper_cup", "magazine",
        "paper_packaging", "carton_drink", "food_waste_generic",
        "cigarette_butt", "shoe", "battery", "pizza_box",
        "unknown_litter",
    ]

    def __init__(self):
        self._ort_session = None
        self._initialized = False
        self._mock_mode = settings.ml_mock_mode
        self._class_names: List[str] = list(self.MODEL_34_CLASS_NAMES)
        self._max_detections = 100
        self._min_bbox_area_fraction = 0.001

    def _load_class_names(self) -> List[str]:
        """Return fallback class names (34-class production model)."""
        return list(self.MODEL_34_CLASS_NAMES)

    def _load_class_names_from_onnx(self) -> Optional[List[str]]:
        """Read class names embedded in the ONNX model metadata."""
        try:
            meta = self._ort_session.get_modelmeta()
            custom = dict(meta.custom_metadata_map)
            names_raw = custom.get("names")
            if not names_raw:
                return None
            import ast
            import json as _json

            try:
                parsed = ast.literal_eval(names_raw)
            except Exception:
                parsed = _json.loads(names_raw)
            if isinstance(parsed, dict):
                return [parsed[k] for k in sorted(parsed.keys())]
            if isinstance(parsed, list):
                return list(parsed)
        except Exception as e:
            logger.warning(f"Could not read class names from ONNX metadata: {e}")
        return None

    def _get_waste_stream(self, class_id: int) -> str:
        """Map class_id to waste stream via stable class NAME lookup."""
        name = (
            self._class_names[class_id]
            if 0 <= class_id < len(self._class_names)
            else f"class_{class_id}"
        )
        return self._get_waste_stream_for_name(name)

    @classmethod
    def _get_waste_stream_for_name(cls, class_name: str) -> str:
        """Map a taxonomy class NAME to one of the 7 disposal streams."""
        if class_name in cls.ORGANIC_NAMES:
            return "organic"
        if class_name in cls.RECOVERABLE_NAMES:
            return "recoverable"
        if class_name in cls.HAZARDOUS_NAMES:
            return "hazardous"
        if class_name in cls.SANITARY_NAMES:
            return "sanitary"
        if class_name in cls.E_WASTE_NAMES:
            return "e_waste"
        if class_name in cls.REJECT_NAMES:
            return "reject"
        return "non_recoverable"

    def initialize(self) -> bool:
        """Initialize the ML inference engine."""
        if self._initialized:
            return True

        if not settings.ml_enabled:
            logger.info("ML inference is disabled via configuration")
            return False

        # If mock mode forced, use it
        if self._mock_mode:
            logger.info("ML inference running in MOCK mode (forced via config)")
            self._initialized = True
            return True

        # Try to load YOLO ONNX model
        yolo_path = Path(settings.yolo_onnx_path)
        if not yolo_path.is_absolute():
            # Try relative to backend directory
            yolo_path = Path(__file__).parent.parent.parent / settings.yolo_onnx_path

        if not yolo_path.exists():
            logger.warning(f"YOLO ONNX model not found at: {yolo_path}")
            logger.warning("Falling back to MOCK mode")
            self._mock_mode = True
            self._initialized = True
            return True

        try:
            import onnxruntime as ort

            logger.info(f"Loading YOLO ONNX model: {yolo_path}")

            # Configure ONNX Runtime session
            providers = ["CPUExecutionProvider"]
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4

            self._ort_session = ort.InferenceSession(
                str(yolo_path), sess_options=opts, providers=providers
            )

            self._input_name = self._ort_session.get_inputs()[0].name
            self._output_names = [o.name for o in self._ort_session.get_outputs()]
            self._input_shape = self._ort_session.get_inputs()[0].shape

            # Warmup with realistic 0-1 normalized input
            in_shape = [
                d if isinstance(d, int) and d > 0 else 1
                for d in self._input_shape
            ]
            in_shape[0] = 1
            dummy = np.random.rand(*in_shape).astype(np.float32)
            self._ort_session.run(self._output_names, {self._input_name: dummy})

            # Use class names embedded in the ONNX model when available.
            onnx_names = self._load_class_names_from_onnx()
            if onnx_names:
                self._class_names = onnx_names
                logger.info(f"Loaded {len(self._class_names)} class names from ONNX metadata")
            else:
                logger.info(f"Using fallback {len(self._class_names)} class names")

            self._initialized = True
            logger.info(f"YOLO ONNX model loaded successfully")
            logger.info(f"  Input: {self._input_name} {self._input_shape}")
            logger.info(f"  Outputs: {self._output_names}")
            return True

        except Exception as e:
            logger.error(f"Failed to load YOLO ONNX model: {e}")
            logger.warning("Falling back to MOCK mode")
            self._mock_mode = True
            self._initialized = True
            return True

    def infer(self, image_source: Union[str, Path, bytes, np.ndarray]) -> Optional[Dict[str, Any]]:
        """Run inference on an image."""
        if not self._initialized:
            if not self.initialize():
                return None

        if self._mock_mode:
            return self._mock_infer()

        try:
            return self._yolo_infer(image_source)
        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            return self._mock_infer()

    def _yolo_infer(self, image_source: Union[str, Path, bytes, np.ndarray]) -> Dict[str, Any]:
        """Run YOLO inference via ONNX Runtime."""
        # Load image (BGR)
        if isinstance(image_source, (str, Path)):
            img = cv2.imread(str(image_source))
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image_source}")
        elif isinstance(image_source, bytes):
            img = cv2.imdecode(np.frombuffer(image_source, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Cannot decode image bytes")
        else:
            img = image_source

        orig_h, orig_w = img.shape[:2]
        input_size = self._input_shape[2]  # H (assuming NCHW)

        # Preprocess: letterbox resize to input_size (must match training:
        # BGR -> RGB, normalized to 0-1). Feeding 0-255 BGR saturates the
        # model and yields thousands of confidence-1.0 false detections.
        scale = min(input_size / orig_w, input_size / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        pad_w = (input_size - new_w) // 2
        pad_h = (input_size - new_h) // 2
        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # BGR -> RGB, 0-1 normalize, HWC -> NCHW
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]  # [1, 3, H, W]

        # Inference
        outputs = self._ort_session.run(self._output_names, {self._input_name: blob})
        raw = outputs[0]  # Shape: [1, num_boxes, 5+num_classes] or [1, 5+num_classes, num_boxes]

        # Post-process
        detections = self._postprocess_yolo(raw, (orig_h, orig_w), scale, (pad_w, pad_h))

        # Build fused_results (YOLO-only mode)
        fused_results = []
        for det in detections:
            class_id = det["class_id"]
            confidence = det["confidence"]
            class_name = det["class_name"]
            waste_stream = self._get_waste_stream(class_id)

            fused_results.append({
                "class_id": class_id,
                "class_name": class_name,
                "fused_confidence": confidence,
                "confidence": confidence,
                "method": "yolo_only",
                "agreement": True,
                "waste_stream": waste_stream,
            })

        return {
            "type": "detection",
            "detections": detections,
            "fused_results": fused_results,
            "timings": {"total_ms": 0},
            "status": "success",
        }

    def _postprocess_yolo(
        self,
        raw: np.ndarray,
        orig_shape: tuple,
        scale: float,
        padding: tuple,
    ) -> List[Dict[str, Any]]:
        """Decode YOLO ONNX output to detections.

        The production ONNX model already outputs class probabilities in
        [0, 1] (Ultralytics export includes the sigmoid), so no extra
        activation is applied here.
        """
        orig_h, orig_w = orig_shape
        pad_w, pad_h = padding

        # Handle different output formats
        if raw.ndim == 3:
            raw = raw[0]  # [num_boxes, 4+nc] or [4+nc, num_boxes]

        # Ensure [num_boxes, 4+nc]
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T

        num_classes = len(self._class_names)
        boxes = raw[:, :4]  # cx, cy, w, h
        scores = raw[:, 4:4 + num_classes]  # [num_boxes, nc]

        # Guard against model/taxonomy class-count mismatch
        if scores.shape[1] == 0:
            return []

        # Get class predictions
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(class_ids)), class_ids]

        # Filter by confidence threshold
        conf_threshold = settings.ml_confidence_threshold
        mask = confidences >= conf_threshold
        boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

        if len(boxes) == 0:
            return []

        # Convert cx,cy,w,h to x1,y1,x2,y2 in input space
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        # Remove padding and rescale to original image
        x1 = np.clip((x1 - pad_w) / scale, 0, orig_w)
        y1 = np.clip((y1 - pad_h) / scale, 0, orig_h)
        x2 = np.clip((x2 - pad_w) / scale, 0, orig_w)
        y2 = np.clip((y2 - pad_h) / scale, 0, orig_h)

        # Filter tiny boxes (< 0.1% of image area)
        img_area = max(orig_w * orig_h, 1)
        box_areas = (x2 - x1) * (y2 - y1)
        area_mask = (box_areas / img_area) >= self._min_bbox_area_fraction
        x1, y1, x2, y2 = x1[area_mask], y1[area_mask], x2[area_mask], y2[area_mask]
        confidences, class_ids = confidences[area_mask], class_ids[area_mask]

        if len(confidences) == 0:
            return []

        # NMS (keep at most max_detections, sorted by confidence)
        keep = self._nms(np.stack([x1, y1, x2, y2], axis=1), confidences, 0.45)
        keep = keep[: self._max_detections]

        results = []
        for i in keep:
            class_id = int(class_ids[i])
            if class_id >= len(self._class_names):
                continue
            class_name = self._class_names[class_id]
            results.append({
                "bbox_xyxy": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                "confidence": float(confidences[i]),
                "class_id": class_id,
                "class_name": class_name,
            })
        # Sort by confidence descending so [0] is the top detection
        results.sort(key=lambda d: d["confidence"], reverse=True)
        return results

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Pure NumPy NMS."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= iou_threshold]
        return keep

    def _mock_infer(self) -> Dict[str, Any]:
        """Generate a mock prediction for testing."""
        class_id, class_name, waste_stream = random.choice(self.MOCK_PREDICTIONS)
        confidence = round(random.uniform(0.65, 0.98), 2)

        fused_results = [{
            "class_id": class_id,
            "class_name": class_name,
            "fused_confidence": confidence,
            "confidence": confidence,
            "method": "mock_spiro_pipeline",
            "agreement": True,
            "waste_stream": waste_stream,
        }]

        return {
            "type": "detection",
            "detections": [{
                "bbox_xyxy": [100, 100, 300, 300],
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name,
            }],
            "fused_results": fused_results,
            "timings": {"total_ms": random.randint(50, 150)},
            "status": "success",
        }

    def get_top_prediction(self, image_source: Union[str, Path, bytes, np.ndarray]) -> Optional[Dict[str, Any]]:
        """Get the top prediction from inference results."""
        result = self.infer(image_source)
        if not result:
            return None

        fused_results = result.get("fused_results", [])
        if not fused_results:
            return None

        top = max(fused_results, key=lambda x: x.get("fused_confidence", x.get("confidence", 0)))

        return {
            "class_id": top.get("class_id"),
            "class_name": top.get("class_name"),
            "confidence": top.get("fused_confidence", top.get("confidence", 0)),
            "waste_stream": top.get("waste_stream"),
            "method": top.get("method"),
            "agreement": top.get("agreement", False),
        }

    def is_ready(self) -> bool:
        """Check if the ML service is ready for inference."""
        return self._initialized


# Singleton instance
ml_inference_service = MLInferenceService()