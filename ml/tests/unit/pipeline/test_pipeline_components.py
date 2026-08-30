"""Unit tests for pipeline components — no real models required."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import lib.ml.dataset_engineering.taxonomy as _tax_module


@pytest.fixture(autouse=True)
def reset_taxonomy():
    _tax_module.TaxonomyLoader._instance = None
    yield
    _tax_module.TaxonomyLoader._instance = None


def _make_bgr(h=256, w=256, value=128):
    return np.full((h, w, 3), value, dtype=np.uint8)


# =============================================================================
# ImagePreprocessor
# =============================================================================
class TestImagePreprocessor:
    def test_sharp_image_passes(self):
        from lib.ml.pipeline.preprocessing.preprocessor import ImagePreprocessor
        prep = ImagePreprocessor(target_size=64, blur_threshold=10.0)
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = prep.process(img)
        assert result.quality.passed

    def test_blurry_image_rejected(self):
        from lib.ml.pipeline.preprocessing.preprocessor import ImagePreprocessor
        prep = ImagePreprocessor(target_size=64, blur_threshold=9999.0)
        img = _make_bgr()  # solid colour → 0 Laplacian variance
        result = prep.process(img)
        assert not result.quality.passed
        assert "blur" in result.quality.rejection_reason

    def test_too_dark_rejected(self):
        from lib.ml.pipeline.preprocessing.preprocessor import ImagePreprocessor
        prep = ImagePreprocessor(target_size=64, brightness_min=100.0, blur_threshold=0.0)
        img = _make_bgr(value=10)  # very dark
        result = prep.process(img)
        assert not result.quality.passed
        assert "brightness" in result.quality.rejection_reason

    def test_overexposed_rejected(self):
        from lib.ml.pipeline.preprocessing.preprocessor import ImagePreprocessor
        prep = ImagePreprocessor(target_size=64, brightness_max=50.0, blur_threshold=0.0)
        img = _make_bgr(value=255)
        result = prep.process(img)
        assert not result.quality.passed

    def test_letterbox_output_shape(self):
        from lib.ml.pipeline.preprocessing.preprocessor import ImagePreprocessor
        prep = ImagePreprocessor(target_size=128, blur_threshold=0.0)
        img = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        result = prep.process(img)
        if result.quality.passed:
            assert result.blob.shape == (1, 3, 128, 128)
            assert result.blob.dtype == np.float32
            assert result.blob.min() >= 0.0
            assert result.blob.max() <= 1.0

    def test_scale_and_padding_correct(self):
        from lib.ml.pipeline.preprocessing.preprocessor import ImagePreprocessor
        prep = ImagePreprocessor(target_size=64, blur_threshold=0.0)
        img = np.random.randint(30, 200, (128, 64, 3), dtype=np.uint8)
        _, scale, pad_w, pad_h = prep._letterbox(img)
        assert scale == pytest.approx(0.5, abs=0.01)

    def test_missing_file_returns_failed(self):
        from lib.ml.pipeline.preprocessing.preprocessor import ImagePreprocessor
        prep = ImagePreprocessor(target_size=64)
        result = prep.process("/nonexistent/path/img.jpg")
        assert not result.quality.passed

    def test_small_resolution_rejected(self):
        from lib.ml.pipeline.preprocessing.preprocessor import ImagePreprocessor
        prep = ImagePreprocessor(target_size=64, min_resolution=(128, 128), blur_threshold=0.0)
        tiny = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = prep.process(tiny)
        assert not result.quality.passed
        assert "resolution_too_small" in result.quality.rejection_reason

    def test_quality_score_computed(self):
        from lib.ml.pipeline.preprocessing.preprocessor import ImagePreprocessor
        prep = ImagePreprocessor(target_size=64, blur_threshold=0.0)
        img = np.random.randint(30, 200, (128, 128, 3), dtype=np.uint8)
        result = prep.process(img)
        assert 0.0 <= result.quality.quality_score <= 1.0


# =============================================================================
# YOLODetectionEngine — NMS only (no model loading)
# =============================================================================
class TestYOLONMS:
    def test_nms_removes_overlapping(self):
        from lib.ml.pipeline.detection.yolo_engine import YOLODetectionEngine
        boxes = np.array([[0,0,10,10],[1,1,11,11],[50,50,60,60]], dtype=float)
        scores = np.array([0.9, 0.8, 0.7])
        keep = YOLODetectionEngine._nms(boxes, scores, 0.5)
        assert 0 in keep
        assert 2 in keep
        assert 1 not in keep

    def test_nms_empty_input(self):
        from lib.ml.pipeline.detection.yolo_engine import YOLODetectionEngine
        keep = YOLODetectionEngine._nms(
            np.empty((0,4), dtype=float), np.array([]), 0.5
        )
        assert len(keep) == 0

    def test_nms_no_overlap_keeps_all(self):
        from lib.ml.pipeline.detection.yolo_engine import YOLODetectionEngine
        boxes = np.array([[0,0,5,5],[20,20,25,25],[40,40,45,45]], dtype=float)
        scores = np.array([0.9, 0.8, 0.7])
        keep = YOLODetectionEngine._nms(boxes, scores, 0.5)
        assert set(keep.tolist()) == {0, 1, 2}


# =============================================================================
# ObjectCropper
# =============================================================================
class TestObjectCropper:
    def _make_det(self, x1=10, y1=10, x2=100, y2=100, cls_id=0):
        from lib.ml.pipeline.detection.yolo_engine import Detection
        return Detection(
            bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
            confidence=0.85,
            class_id=cls_id,
            class_name="plastic_bottle",
        )

    def test_crop_valid(self):
        from lib.ml.pipeline.cropping.cropper import ObjectCropper
        cropper = ObjectCropper(padding_fraction=0.1, min_crop_pixels=8)
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        dets = [self._make_det(20, 20, 100, 100)]
        crops = cropper.crop(img, dets)
        assert len(crops) == 1
        assert crops[0].is_valid
        assert crops[0].image_bgr.shape[0] > 0

    def test_tiny_bbox_rejected(self):
        from lib.ml.pipeline.cropping.cropper import ObjectCropper
        cropper = ObjectCropper(min_crop_pixels=64)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        dets = [self._make_det(10, 10, 12, 12)]  # 2×2 pixel box
        crops = cropper.crop(img, dets)
        assert len(crops) == 1
        assert not crops[0].is_valid

    def test_preprocess_for_effnet_shape(self):
        from lib.ml.pipeline.cropping.cropper import ObjectCropper, Crop
        from lib.ml.pipeline.detection.yolo_engine import Detection
        cropper = ObjectCropper()
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        det = self._make_det(5, 5, 60, 60)
        crop = Crop(image_bgr=img, detection_idx=0, detection=det,
                    bbox_padded=(0,0,64,64), is_valid=True)
        blob = cropper.preprocess_for_effnet(crop, target_size=32)
        assert blob.shape == (1, 3, 32, 32)
        assert blob.dtype == np.float32

    def test_preprocess_batch_shape(self):
        from lib.ml.pipeline.cropping.cropper import ObjectCropper, Crop
        from lib.ml.pipeline.detection.yolo_engine import Detection
        cropper = ObjectCropper()
        crops = []
        for i in range(4):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            det = self._make_det()
            crops.append(Crop(img, i, det, (0,0,64,64), is_valid=True))
        batch = cropper.preprocess_batch(crops, target_size=32)
        assert batch.shape == (4, 3, 32, 32)


# =============================================================================
# ContaminationEngine
# =============================================================================
class TestContaminationEngine:
    def test_clean_image_no_flags(self):
        from lib.ml.pipeline.contamination.contamination_engine import ContaminationEngine
        engine = ContaminationEngine()
        # Only recoverable items — no contamination
        results = [
            {"class_id": 0, "class_name": "plastic_bottle", "fused_confidence": 0.85},
            {"class_id": 36, "class_name": "cardboard_box", "fused_confidence": 0.80},
        ]
        result = engine.analyse(results)
        # No food or hazardous — clean
        food_flag = any(f.code == "FOOD_RESIDUE_ON_RECYCLABLE" for f in result.flags)
        assert not food_flag

    def test_food_residue_detected(self):
        from lib.ml.pipeline.contamination.contamination_engine import ContaminationEngine
        engine = ContaminationEngine()
        results = [
            {"class_id": 0, "class_name": "plastic_bottle", "fused_confidence": 0.85},
            {"class_id": 47, "class_name": "food_waste_generic", "fused_confidence": 0.75},
        ]
        result = engine.analyse(results)
        assert any(f.code == "FOOD_RESIDUE_ON_RECYCLABLE" for f in result.flags)
        assert result.contamination_score > 0

    def test_organic_hazardous_mix_critical(self):
        from lib.ml.pipeline.contamination.contamination_engine import ContaminationEngine
        engine = ContaminationEngine()
        results = [
            {"class_id": 47, "class_name": "food_waste_generic", "fused_confidence": 0.9},
            {"class_id": 61, "class_name": "battery", "fused_confidence": 0.85},
        ]
        result = engine.analyse(results)
        assert any(f.code == "ORGANIC_HAZARDOUS_MIX" for f in result.flags)
        crit = [f for f in result.flags if f.severity == "critical"]
        assert len(crit) > 0

    def test_score_capped_at_one(self):
        from lib.ml.pipeline.contamination.contamination_engine import ContaminationEngine
        engine = ContaminationEngine()
        results = [
            {"class_id": 47, "class_name": "food_waste_generic", "fused_confidence": 0.9},
            {"class_id": 61, "class_name": "battery", "fused_confidence": 0.9},
            {"class_id": 59, "class_name": "mask_disposable", "fused_confidence": 0.9},
            {"class_id": 0, "class_name": "plastic_bottle", "fused_confidence": 0.9},
        ]
        result = engine.analyse(results)
        assert result.contamination_score <= 1.0

    def test_empty_results(self):
        from lib.ml.pipeline.contamination.contamination_engine import ContaminationEngine
        engine = ContaminationEngine()
        result = engine.analyse([])
        assert result.contamination_score == 0.0
        assert result.flags == []

    def test_to_dict_structure(self):
        from lib.ml.pipeline.contamination.contamination_engine import ContaminationEngine
        engine = ContaminationEngine()
        result = engine.analyse([{"class_id": 0, "class_name": "plastic_bottle", "fused_confidence": 0.8}])
        d = result.to_dict()
        assert "contamination_score" in d
        assert "is_contaminated" in d
        assert "flags" in d
        assert "explanation" in d
        assert "action_required" in d


# =============================================================================
# GuidanceEngine
# =============================================================================
class TestGuidanceEngine:
    def test_plastic_bottle_recoverable(self):
        from lib.ml.pipeline.guidance.guidance_engine import GuidanceEngine
        engine = GuidanceEngine()
        results = [{"class_id": 0, "class_name": "plastic_bottle", "fused_confidence": 0.85}]
        items = engine.generate(results)
        assert len(items) == 1
        assert "Recoverable" in items[0].stream or "recoverable" in items[0].stream.lower()

    def test_battery_hazardous_stream(self):
        from lib.ml.pipeline.guidance.guidance_engine import GuidanceEngine
        engine = GuidanceEngine()
        results = [{"class_id": 61, "class_name": "battery", "fused_confidence": 0.9}]
        items = engine.generate(results)
        assert len(items) == 1
        assert "e_waste" in engine.stream_for_class(61) or "hazardous" in engine.stream_for_class(61)

    def test_food_waste_organic_stream(self):
        from lib.ml.pipeline.guidance.guidance_engine import GuidanceEngine
        engine = GuidanceEngine()
        results = [{"class_id": 47, "class_name": "food_waste_generic", "fused_confidence": 0.9}]
        items = engine.generate(results)
        assert "Organic" in items[0].stream

    def test_unique_class_deduplication(self):
        from lib.ml.pipeline.guidance.guidance_engine import GuidanceEngine
        engine = GuidanceEngine()
        results = [
            {"class_id": 0, "class_name": "plastic_bottle", "fused_confidence": 0.85},
            {"class_id": 0, "class_name": "plastic_bottle", "fused_confidence": 0.80},
        ]
        items = engine.generate(results)
        assert len(items) == 1  # deduped

    def test_to_dict_has_required_keys(self):
        from lib.ml.pipeline.guidance.guidance_engine import GuidanceEngine
        engine = GuidanceEngine()
        results = [{"class_id": 0, "class_name": "plastic_bottle", "fused_confidence": 0.8}]
        items = engine.generate(results)
        d = items[0].to_dict()
        for key in ["class_id", "class_name", "waste_stream", "bin_colour",
                    "collection_point", "preparation_steps", "warnings", "regulatory_note"]:
            assert key in d

    def test_class_with_override_note(self):
        from lib.ml.pipeline.guidance.guidance_engine import GuidanceEngine
        engine = GuidanceEngine()
        results = [{"class_id": 61, "class_name": "battery", "fused_confidence": 0.9}]
        items = engine.generate(results)
        assert items[0].extra_note is not None and "battery" in items[0].extra_note.lower()

    def test_summary_guidance_single_stream(self):
        from lib.ml.pipeline.guidance.guidance_engine import GuidanceEngine, GuidanceItem
        engine = GuidanceEngine()
        items = [GuidanceItem(0, "plastic_bottle", "Recoverable", "Blue", "Recycle bin",
                               [], [], "", None)]
        summary = engine.summary_guidance(items)
        assert "Recoverable" in summary

    def test_summary_guidance_mixed(self):
        from lib.ml.pipeline.guidance.guidance_engine import GuidanceEngine, GuidanceItem
        engine = GuidanceEngine()
        items = [
            GuidanceItem(0, "plastic_bottle", "Recoverable", "Blue", "Recycle", [], [], "", None),
            GuidanceItem(47, "food_waste", "Organic", "Green", "Green bin", [], [], "", None),
        ]
        summary = engine.summary_guidance(items)
        assert "Mixed" in summary or "separate" in summary.lower()


# =============================================================================
# OutputBuilder
# =============================================================================
class TestOutputBuilder:
    def test_success_response_structure(self):
        from lib.ml.pipeline.output.output_builder import OutputBuilder, PipelineTimings
        builder = OutputBuilder()
        timings = PipelineTimings(preprocess_ms=10, detection_ms=50, total_ms=80)
        response = builder.build(
            status="success",
            image_meta={"quality": {"passed": True}},
            detections=[],
            fused_results=[],
            guidance_items=[],
            contamination=None,
            evidence=[],
            timings=timings,
        )
        assert response["status"] == "success"
        assert "detections" in response
        assert "guidance" in response
        assert "timings" in response

    def test_error_response(self):
        from lib.ml.pipeline.output.output_builder import OutputBuilder
        r = OutputBuilder.error_response("model missing", "MODEL_ERROR")
        assert r["status"] == "error"
        assert r["error"]["code"] == "MODEL_ERROR"

    def test_rejected_response(self):
        from lib.ml.pipeline.output.output_builder import OutputBuilder
        r = OutputBuilder.rejected_response("blur_score_5.0_below_80.0", {"blur_score": 5.0})
        assert r["status"] == "rejected"
        assert r["detection_count"] == 0

    def test_timings_in_output(self):
        from lib.ml.pipeline.output.output_builder import OutputBuilder, PipelineTimings
        builder = OutputBuilder(include_timings=True)
        t = PipelineTimings(total_ms=99.5)
        r = builder.build("success", {}, [], [], [], None, [], t)
        assert "timings" in r
        assert r["timings"]["total_ms"] == 99.5

    def test_pipeline_version_present(self):
        from lib.ml.pipeline.output.output_builder import OutputBuilder, PipelineTimings
        r = OutputBuilder().build("success", {}, [], [], [], None, [], PipelineTimings())
        assert "pipeline_version" in r
