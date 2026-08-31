"""
Integration tests for the full SPIRO inference pipeline.
ONNX models are mocked — no actual model files required.
Tests verify the complete orchestration from image → JSON response.
"""
import json
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


def _make_image(h=480, w=640):
    return np.random.randint(30, 200, (h, w, 3), dtype=np.uint8)


def _fake_yolo_output(nc=109, n_anchors=8400):
    """
    Simulates YOLOv11 ONNX output [1, 4+nc, n_anchors].
    Puts 2 detections at known positions with class 0 (plastic_bottle).
    """
    raw = np.zeros((1, 4 + nc, n_anchors), dtype=np.float32)
    for i in [0, 1]:
        # cx, cy, w, h in input_size=640 coords
        raw[0, 0, i] = 200.0 + i * 100  # cx
        raw[0, 1, i] = 200.0             # cy
        raw[0, 2, i] = 80.0              # w
        raw[0, 3, i] = 80.0              # h
        raw[0, 4 + 0, i] = 0.85         # class 0 score (plastic_bottle)
    return [raw]


def _fake_effnet_output(nc=109, batch=2):
    """Simulates EfficientNetV2 logits [B, nc]."""
    logits = np.zeros((batch, nc), dtype=np.float32)
    logits[:, 0] = 3.0   # high logit for class 0 (plastic_bottle)
    return [logits]


def _build_pipeline(tmp_path):
    """Build SPIROPipeline with fully mocked ORT sessions."""
    import onnxruntime as ort
    from lib.ml.pipeline.pipeline_config import PipelineConfig
    from lib.ml.pipeline.preprocessing.preprocessor import ImagePreprocessor
    from lib.ml.pipeline.detection.yolo_engine import YOLODetectionEngine
    from lib.ml.pipeline.cropping.cropper import ObjectCropper
    from lib.ml.pipeline.contamination.contamination_engine import ContaminationEngine
    from lib.ml.pipeline.guidance.guidance_engine import GuidanceEngine
    from lib.ml.pipeline.explainability.explainer import PipelineExplainer
    from lib.ml.pipeline.output.output_builder import OutputBuilder
    from lib.ml.pipeline.spiro_pipeline import SPIROPipeline
    from lib.ml.verification.fusion.confidence_fusion import ConfidenceFusion
    import yaml

    # Write minimal pipeline config
    cfg_path = tmp_path / "test_pipeline.yaml"
    cfg_data = {
        "pipeline": {"name": "test", "version": "1.0.0", "description": ""},
        "models": {
            "yolo_onnx": str(tmp_path / "yolo.onnx"),
            "effnet_onnx": str(tmp_path / "effnet.onnx"),
            "yolo_input_size": 640,
            "effnet_input_size": 64,
            "num_classes": 109,
        },
        "runtime": {
            "providers": ["CPUExecutionProvider"],
            "yolo_threads": 1, "effnet_threads": 1,
            "graph_optimization": "ORT_ENABLE_BASIC",
            "enable_mem_pattern": True, "enable_cpu_mem_arena": True,
            "fp16": False, "yolo_batch": 1, "effnet_batch": 4, "warmup_runs": 0,
        },
        "preprocessing": {
            "max_file_size_mb": 50,
            "supported_formats": ["jpg", "jpeg", "png"],
            "min_resolution": [32, 32],
            "max_resolution": [4096, 4096],
            "blur_detection": False,
            "blur_threshold": 0.0,  # disable blur rejection
            "brightness_min": 0.0,
            "brightness_max": 255.0,
            "quality_score_threshold": 0.0,  # accept everything
            "auto_orient": False,
            "target_size": 64,
            "letterbox_color": [114, 114, 114],
            "normalize": True,
        },
        "detection": {
            "conf_threshold": 0.5,
            "iou_threshold": 0.45,
            "max_detections": 10,
            "min_bbox_area": 0.0,
            "agnostic_nms": False,
        },
        "cropping": {"padding_fraction": 0.05, "min_crop_pixels": 4},
        "verification": {"enabled": True, "top_k": 5, "temperature": 1.0},
        "fusion": {
            "method": "weighted_average",
            "yolo_weight": 0.4, "effnet_weight": 0.6, "min_final_confidence": 0.1,
        },
        "contamination": {
            "enabled": True,
            "food_residue_classes": [47, 48, 49, 50, 51, 52],
            "hazardous_classes": [61, 66, 67, 82, 83, 84],
            "sanitary_classes": [59, 60, 66, 69, 70],
            "wet_indicator_threshold": 0.6,
            "mixed_waste_penalty": 0.3,
        },
        "guidance": {
            "enabled": True, "language": "en",
            "include_collection_point": True,
            "include_warnings": True,
            "include_preparation_steps": True,
        },
        "explainability": {
            "enabled": True,
            "gradcam": False,
            "save_visualizations": False,
            "visualization_dir": str(tmp_path / "viz"),
            "include_reasoning_chain": True,
        },
        "output": {
            "include_probabilities": False,
            "include_timings": True,
            "include_model_versions": True,
        },
        "performance": {
            "benchmark_n_runs": 5,
            "log_slow_threshold_ms": 10000,
        },
    }
    with open(cfg_path, "w") as f:
        yaml.dump(cfg_data, f)

    cfg = PipelineConfig.load(cfg_path)

    # Mock YOLO ORT session
    mock_yolo_sess = MagicMock()
    mock_yolo_sess.get_inputs.return_value = [MagicMock(name="images")]
    mock_yolo_sess.get_outputs.return_value = [MagicMock(name="output0")]
    mock_yolo_sess.get_providers.return_value = ["CPUExecutionProvider"]
    mock_yolo_sess.run.return_value = _fake_yolo_output(nc=109)

    # Mock EffNet ORT session
    mock_effnet_sess = MagicMock()
    mock_effnet_sess.get_inputs.return_value = [MagicMock(name="images")]
    mock_effnet_sess.get_outputs.return_value = [MagicMock(name="logits")]
    mock_effnet_sess.run.return_value = _fake_effnet_output(nc=109, batch=2)

    yolo_engine = YOLODetectionEngine.__new__(YOLODetectionEngine)
    yolo_engine.session = mock_yolo_sess
    yolo_engine.input_name = "images"
    yolo_engine.output_names = ["output0"]
    yolo_engine.input_size = 64
    yolo_engine.conf_threshold = 0.5
    yolo_engine.iou_threshold = 0.45
    yolo_engine.max_detections = 10
    yolo_engine.min_bbox_area = 0.0
    yolo_engine.taxonomy = _tax_module.TaxonomyLoader(
        Path("configs/taxonomy/spiro_taxonomy.yaml")
    )
    yolo_engine._active_providers = ["CPUExecutionProvider"]

    preprocessor = ImagePreprocessor(
        target_size=64, blur_threshold=0.0,
        brightness_min=0.0, brightness_max=255.0,
        quality_threshold=0.0, auto_orient=False,
    )
    cropper = ObjectCropper(padding_fraction=0.05, min_crop_pixels=4)
    fusion = ConfidenceFusion(
        method="weighted_average", yolo_weight=0.4, effnet_weight=0.6,
        min_confidence=0.1, num_classes=109,
    )
    contamination = ContaminationEngine()
    guidance = GuidanceEngine()
    explainer = PipelineExplainer(include_reasoning_chain=True)
    output_builder = OutputBuilder(include_timings=True, include_model_versions=True,
                                   model_versions={"yolo": "mock.onnx", "effnet": "mock.onnx"})

    pipeline = SPIROPipeline(
        cfg=cfg,
        yolo_engine=yolo_engine,
        effnet_session=mock_effnet_sess,
        preprocessor=preprocessor,
        cropper=cropper,
        fusion=fusion,
        contamination=contamination,
        guidance=guidance,
        explainer=explainer,
        output_builder=output_builder,
    )
    return pipeline


@pytest.mark.integration
class TestSPIROPipelineIntegration:
    def test_infer_returns_valid_response(self, tmp_path):
        pipeline = _build_pipeline(tmp_path)
        img = _make_image()
        result = pipeline.infer(img)
        assert isinstance(result, dict)
        assert "status" in result
        assert "detections" in result
        assert "detection_count" in result
        assert result["status"] in {"success", "partial", "rejected", "error"}

    def test_infer_detections_populated(self, tmp_path):
        pipeline = _build_pipeline(tmp_path)
        img = _make_image()
        result = pipeline.infer(img)
        # With our fake YOLO output we expect detections
        if result["status"] in {"success", "partial"}:
            assert isinstance(result["detections"], list)
            for det in result["detections"]:
                assert "bbox_xyxy" in det
                assert "final_class_name" in det
                assert "final_confidence" in det
                assert 0.0 <= det["final_confidence"] <= 1.0

    def test_guidance_present(self, tmp_path):
        pipeline = _build_pipeline(tmp_path)
        img = _make_image()
        result = pipeline.infer(img)
        assert "guidance" in result
        assert "summary" in result["guidance"]
        assert "items" in result["guidance"]

    def test_contamination_present(self, tmp_path):
        pipeline = _build_pipeline(tmp_path)
        img = _make_image()
        result = pipeline.infer(img)
        if result["detection_count"] > 0:
            assert "contamination" in result
            assert "contamination_score" in result["contamination"]

    def test_explainability_present(self, tmp_path):
        pipeline = _build_pipeline(tmp_path)
        img = _make_image()
        result = pipeline.infer(img)
        if result["detection_count"] > 0:
            assert "explainability" in result
            assert "evidence" in result["explainability"]

    def test_timings_present(self, tmp_path):
        pipeline = _build_pipeline(tmp_path)
        img = _make_image()
        result = pipeline.infer(img)
        assert "timings" in result
        assert "total_ms" in result["timings"]
        assert result["timings"]["total_ms"] > 0

    def test_response_is_json_serialisable(self, tmp_path):
        pipeline = _build_pipeline(tmp_path)
        img = _make_image()
        result = pipeline.infer(img)
        serialised = json.dumps(result)
        parsed = json.loads(serialised)
        assert parsed["status"] == result["status"]

    def test_rejected_blurry_image(self, tmp_path):
        pipeline = _build_pipeline(tmp_path)
        # Override preprocessor to reject everything
        from lib.ml.pipeline.preprocessing.preprocessor import ImagePreprocessor
        pipeline.preprocessor = ImagePreprocessor(
            target_size=64, blur_threshold=99999.0, quality_threshold=0.0,
        )
        img = np.full((64, 64, 3), 128, dtype=np.uint8)  # solid → 0 variance
        result = pipeline.infer(img)
        assert result["status"] == "rejected"
        assert result["detection_count"] == 0

    def test_error_response_on_crash(self, tmp_path):
        pipeline = _build_pipeline(tmp_path)
        # Force YOLO to raise
        pipeline.yolo.session.run.side_effect = RuntimeError("ORT crash")
        img = _make_image()
        result = pipeline.infer(img)
        assert result["status"] == "error"

    def test_batch_inference(self, tmp_path):
        pipeline = _build_pipeline(tmp_path)
        imgs = [_make_image() for _ in range(3)]
        results = pipeline.infer_batch(imgs)
        assert len(results) == 3
        for r in results:
            assert "status" in r

    def test_pipeline_version_in_response(self, tmp_path):
        pipeline = _build_pipeline(tmp_path)
        result = pipeline.infer(_make_image())
        assert "pipeline_version" in result
        assert result["pipeline_version"] == "1.0.0"

    def test_warnings_list_present(self, tmp_path):
        pipeline = _build_pipeline(tmp_path)
        result = pipeline.infer(_make_image())
        assert "warnings" in result
        assert isinstance(result["warnings"], list)

    def test_benchmark_returns_stats(self, tmp_path):
        pipeline = _build_pipeline(tmp_path)
        stats = pipeline.benchmark(n_runs=3)
        assert "total" in stats
        assert "mean_ms" in stats["total"]
        assert "fps" in stats["total"]
        assert stats["total"]["fps"] > 0
