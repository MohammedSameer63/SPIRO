"""
SPIRO ML — Public API surface.

Import paths for integration teams:
    from lib.ml import SPIRODetector, SPIROClassifier
    from lib.ml.inference import ONNXInferenceEngine
    from lib.ml.training import YOLOTrainer, EfficientNetTrainer
    from lib.ml.evaluation import Evaluator
    from lib.ml.export import ONNXExporter
    from lib.ml.continuous_learning import ContinuousLearningManager
    from lib.ml.data import DatasetManager
"""

from lib.ml.core.config import ConfigManager
from lib.ml.core.logger import get_logger
from lib.ml.models.yolo_model import SPIRODetector
from lib.ml.models.efficientnet_model import SPIROClassifier
from lib.ml.inference.onnx_engine import ONNXInferenceEngine
from lib.ml.data.dataset_manager import DatasetManager

__version__ = "1.0.0"
__all__ = [
    "ConfigManager",
    "get_logger",
    "SPIRODetector",
    "SPIROClassifier",
    "ONNXInferenceEngine",
    "DatasetManager",
]
