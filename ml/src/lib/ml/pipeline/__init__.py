"""
SPIRO ML — Production Inference Pipeline

Usage
-----
from lib.ml.pipeline import SPIROPipeline

pipeline = SPIROPipeline.from_config("configs/pipeline/inference_pipeline.yaml")
result = pipeline.infer("photo.jpg")
"""
from lib.ml.pipeline.spiro_pipeline import SPIROPipeline
from lib.ml.pipeline.pipeline_config import PipelineConfig

__all__ = ["SPIROPipeline", "PipelineConfig"]
