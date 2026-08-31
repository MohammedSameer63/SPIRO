"""Tests for ConfidenceFusion — all 5 fusion methods."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


def _make_det(class_id=0, confidence=0.8, nc=109):
    probs = np.zeros(nc)
    probs[class_id] = confidence
    return {
        "class_id": class_id,
        "confidence": confidence,
        "class_name": "plastic_bottle",
        "probabilities": probs.tolist(),
    }


def _make_effnet(class_id=0, confidence=0.7, nc=109):
    probs = np.zeros(nc)
    probs[class_id] = confidence
    return {
        "class_id": class_id,
        "confidence": confidence,
        "class_name": "plastic_bottle",
        "probabilities": probs.tolist(),
    }


class TestConfidenceFusion:
    def test_weighted_average(self, dummy_detections, effnet_result):
        from lib.ml.verification.fusion.confidence_fusion import ConfidenceFusion
        fusion = ConfidenceFusion(method="weighted_average", yolo_weight=0.4, effnet_weight=0.6)
        result = fusion.fuse(dummy_detections[0], effnet_result)
        assert 0.0 <= result.fused_confidence <= 1.0
        assert result.method == "weighted_average"

    def test_geometric_mean(self, dummy_detections, effnet_result):
        from lib.ml.verification.fusion.confidence_fusion import ConfidenceFusion
        fusion = ConfidenceFusion(method="geometric_mean")
        result = fusion.fuse(dummy_detections[0], effnet_result)
        assert 0.0 <= result.fused_confidence <= 1.0

    def test_harmonic_mean(self, dummy_detections, effnet_result):
        from lib.ml.verification.fusion.confidence_fusion import ConfidenceFusion
        fusion = ConfidenceFusion(method="harmonic_mean")
        result = fusion.fuse(dummy_detections[0], effnet_result)
        assert 0.0 <= result.fused_confidence <= 1.0

    def test_bayesian(self, dummy_detections, effnet_result):
        from lib.ml.verification.fusion.confidence_fusion import ConfidenceFusion
        fusion = ConfidenceFusion(method="bayesian")
        result = fusion.fuse(dummy_detections[0], effnet_result)
        assert 0.0 <= result.fused_confidence <= 1.0

    def test_temperature(self, dummy_detections, effnet_result):
        from lib.ml.verification.fusion.confidence_fusion import ConfidenceFusion
        fusion = ConfidenceFusion(method="temperature", temperature=1.5)
        result = fusion.fuse(dummy_detections[0], effnet_result)
        assert 0.0 <= result.fused_confidence <= 1.0

    def test_agreement_flag_true(self):
        from lib.ml.verification.fusion.confidence_fusion import ConfidenceFusion
        fusion = ConfidenceFusion(method="weighted_average")
        det = _make_det(class_id=0, confidence=0.8)
        eff = _make_effnet(class_id=0, confidence=0.75)
        result = fusion.fuse(det, eff)
        assert result.agreement is True

    def test_agreement_flag_false(self):
        from lib.ml.verification.fusion.confidence_fusion import ConfidenceFusion
        fusion = ConfidenceFusion(method="weighted_average")
        det = _make_det(class_id=0, confidence=0.8)
        eff = _make_effnet(class_id=5, confidence=0.9)
        result = fusion.fuse(det, eff)
        assert result.agreement is False

    def test_low_confidence_returns_uncertain(self):
        from lib.ml.verification.fusion.confidence_fusion import ConfidenceFusion
        # Very low probabilities on all classes → fused conf < min_confidence
        fusion = ConfidenceFusion(method="weighted_average", min_confidence=0.99)
        det = _make_det(class_id=0, confidence=0.1)
        eff = _make_effnet(class_id=0, confidence=0.1)
        result = fusion.fuse(det, eff)
        assert result.class_name == "uncertain"

    def test_fuse_batch(self):
        from lib.ml.verification.fusion.confidence_fusion import ConfidenceFusion
        fusion = ConfidenceFusion(method="weighted_average")
        dets = [_make_det(i % 5) for i in range(4)]
        effs = [_make_effnet(i % 5) for i in range(4)]
        results = fusion.fuse_batch(dets, effs)
        assert len(results) == 4

    def test_invalid_method_raises(self):
        from lib.ml.verification.fusion.confidence_fusion import ConfidenceFusion
        with pytest.raises(ValueError, match="Unknown fusion method"):
            ConfidenceFusion(method="invalid_xyz")

    def test_result_to_dict(self, dummy_detections, effnet_result):
        from lib.ml.verification.fusion.confidence_fusion import ConfidenceFusion
        fusion = ConfidenceFusion()
        result = fusion.fuse(dummy_detections[0], effnet_result)
        d = result.to_dict()
        assert "class_id" in d
        assert "fused_confidence" in d
        assert "method" in d
        assert "agreement" in d

    def test_weights_normalised(self):
        from lib.ml.verification.fusion.confidence_fusion import ConfidenceFusion
        # Weights don't need to sum to 1 — they get normalised
        fusion = ConfidenceFusion(method="weighted_average", yolo_weight=2.0, effnet_weight=3.0)
        assert abs(fusion.yolo_weight + fusion.effnet_weight - 1.0) < 1e-9

    def test_probabilities_sum_to_one(self, dummy_detections, effnet_result):
        from lib.ml.verification.fusion.confidence_fusion import ConfidenceFusion
        import numpy as np
        for method in ["weighted_average", "geometric_mean", "harmonic_mean", "bayesian", "temperature"]:
            fusion = ConfidenceFusion(method=method)
            probs = fusion._apply_fusion(
                np.array(dummy_detections[0]["probabilities"]),
                np.array(effnet_result["probabilities"]),
                dummy_detections[0]["confidence"],
                effnet_result["confidence"],
            )
            assert abs(probs.sum() - 1.0) < 1e-6, f"Failed for method={method}"
