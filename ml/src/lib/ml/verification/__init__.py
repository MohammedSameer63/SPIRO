"""
SPIRO ML — Verification Package
EfficientNetV2 second-stage verification pipeline.

Usage
-----
from lib.ml.verification import VerifierModel, VerifyConfig, ConfidenceFusion
from lib.ml.verification.inference import VerifyInferenceEngine
from lib.ml.verification.explainability import GradCAM
"""
from lib.ml.verification.verify_config import VerifyConfig
from lib.ml.verification.model.verify_model import VerifierModel, VerificationResult
from lib.ml.verification.fusion.confidence_fusion import ConfidenceFusion, FusionResult
from lib.ml.verification.inference.verify_inference import VerifyInferenceEngine
from lib.ml.verification.explainability.gradcam import GradCAM

__all__ = [
    "VerifyConfig",
    "VerifierModel",
    "VerificationResult",
    "ConfidenceFusion",
    "FusionResult",
    "VerifyInferenceEngine",
    "GradCAM",
]
