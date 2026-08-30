"""Unit tests for PipelineExplainer."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


def _make_det(cls_id=0, name="plastic_bottle", conf=0.85):
    from lib.ml.pipeline.detection.yolo_engine import Detection
    return Detection(
        bbox_xyxy=(10.0, 20.0, 100.0, 120.0),
        confidence=conf,
        class_id=cls_id,
        class_name=name,
    )


def _make_verif(cls_id=0, name="plastic_bottle", conf=0.78):
    probs = np.zeros(109)
    probs[cls_id] = conf
    return {
        "class_id": cls_id,
        "class_name": name,
        "confidence": conf,
        "top_k": [{"rank": 1, "class_id": cls_id, "class_name": name, "confidence": conf}],
        "probabilities": probs.tolist(),
    }


def _make_fused(cls_id=0, name="plastic_bottle", conf=0.81):
    return {
        "class_id": cls_id,
        "class_name": name,
        "fused_confidence": conf,
        "method": "weighted_average",
        "agreement": True,
        "waste_stream": "recoverable",
        "yolo_confidence": 0.85,
        "effnet_confidence": 0.78,
    }


class TestPipelineExplainer:
    def test_build_evidence_structure(self):
        from lib.ml.pipeline.explainability.explainer import PipelineExplainer, DetectionEvidence
        explainer = PipelineExplainer()
        dets   = [_make_det(), _make_det(1, "glass_bottle", 0.72)]
        verifs = [_make_verif(), _make_verif(1, "glass_bottle", 0.68)]
        fused  = [_make_fused(), _make_fused(1, "glass_bottle", 0.70)]
        evidence = explainer.build_evidence(dets, verifs, fused)
        assert len(evidence) == 2
        for ev in evidence:
            assert isinstance(ev, DetectionEvidence)
            assert ev.fused_confidence > 0

    def test_reasoning_chain_populated(self):
        from lib.ml.pipeline.explainability.explainer import PipelineExplainer
        explainer = PipelineExplainer(include_reasoning_chain=True)
        evidence = explainer.build_evidence(
            [_make_det()], [_make_verif()], [_make_fused()]
        )
        assert len(evidence[0].reasoning_chain) >= 4

    def test_reasoning_chain_disabled(self):
        from lib.ml.pipeline.explainability.explainer import PipelineExplainer
        explainer = PipelineExplainer(include_reasoning_chain=False)
        evidence = explainer.build_evidence(
            [_make_det()], [_make_verif()], [_make_fused()]
        )
        assert evidence[0].reasoning_chain == []

    def test_disagreement_in_chain(self):
        from lib.ml.pipeline.explainability.explainer import PipelineExplainer
        explainer = PipelineExplainer(include_reasoning_chain=True)
        fused = _make_fused()
        fused["agreement"] = False
        evidence = explainer.build_evidence(
            [_make_det(0, "plastic_bottle")],
            [_make_verif(5, "plastic_straw")],
            [fused],
        )
        chain = evidence[0].reasoning_chain
        assert any("Disagreement" in step or "disagree" in step.lower() for step in chain)

    def test_annotate_image_returns_array(self):
        from lib.ml.pipeline.explainability.explainer import PipelineExplainer
        explainer = PipelineExplainer()
        dets   = [_make_det()]
        verifs = [_make_verif()]
        fused  = [_make_fused()]
        evidence = explainer.build_evidence(dets, verifs, fused)
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        annotated = explainer.annotate_image(img, evidence, dets)
        assert annotated.shape == img.shape
        assert annotated.dtype == np.uint8

    def test_save_annotated(self, tmp_path):
        from lib.ml.pipeline.explainability.explainer import PipelineExplainer
        explainer = PipelineExplainer(
            save_visualizations=True,
            visualization_dir=tmp_path / "viz",
        )
        annotated = np.zeros((128, 128, 3), dtype=np.uint8)
        out_path = explainer.save_annotated(annotated, stem="test_img")
        assert out_path.exists()

    def test_to_json_serialisable(self):
        import json
        from lib.ml.pipeline.explainability.explainer import PipelineExplainer
        explainer = PipelineExplainer()
        evidence = explainer.build_evidence(
            [_make_det()], [_make_verif()], [_make_fused()]
        )
        data = explainer.to_json(evidence)
        # Must be JSON-serialisable
        serialised = json.dumps(data)
        assert len(serialised) > 0
        parsed = json.loads(serialised)
        assert isinstance(parsed, list)
        assert "fusion" in parsed[0]
        assert "reasoning_chain" in parsed[0]

    def test_evidence_to_dict_keys(self):
        from lib.ml.pipeline.explainability.explainer import PipelineExplainer
        explainer = PipelineExplainer()
        evidence = explainer.build_evidence(
            [_make_det()], [_make_verif()], [_make_fused()]
        )
        d = evidence[0].to_dict()
        for key in ["detection_idx", "yolo", "effnet", "fusion",
                    "waste_stream", "reasoning_chain"]:
            assert key in d

    def test_none_verif_handled(self):
        from lib.ml.pipeline.explainability.explainer import PipelineExplainer
        explainer = PipelineExplainer()
        evidence = explainer.build_evidence(
            [_make_det()], [None], [_make_fused()]
        )
        assert len(evidence) == 1
        # When verif is None, should fall back to YOLO values
        assert evidence[0].effnet_class_id == evidence[0].yolo_class_id

    def test_reasoning_chain_static_method(self):
        from lib.ml.pipeline.explainability.explainer import PipelineExplainer
        chain = PipelineExplainer._build_reasoning_chain(
            "plastic_bottle", 0.85,
            "plastic_bottle", 0.78,
            "plastic_bottle", 0.81,
            "weighted_average", True,
        )
        assert len(chain) >= 4
        assert any("YOLOv11" in step for step in chain)
        assert any("EfficientNetV2" in step for step in chain)
        assert any("Fusion" in step for step in chain)
