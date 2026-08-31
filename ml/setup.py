"""
SPIRO ML Package - Production-ready AI/ML subsystem for SPIRO.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="spiro-ml",
    version="1.0.0",
    author="SPIRO AI/ML Team",
    author_email="ml@spiro.ai",
    description="Production-ready ML subsystem for SPIRO — dataset management, training, evaluation, ONNX export, inference, and continuous learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/spiro-ai/spiro-ml",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "ruff>=0.0.287",
        ],
        "gpu": [
            "torch>=2.1.0+cu118",
            "torchvision>=0.16.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "spiro-train=lib.ml.scripts.train:main",
            "spiro-eval=lib.ml.scripts.evaluate:main",
            "spiro-export=lib.ml.scripts.export_onnx:main",
            "spiro-infer=lib.ml.scripts.infer:main",
            "spiro-dataset=lib.ml.scripts.dataset_manager:main",
            # MLOps platform CLI
            "spiro-mlops-registry=training.mlops.model_registry:main",
            "spiro-mlops-validate=training.mlops.model_validator:main",
            "spiro-mlops-deploy=training.mlops.model_deployer:main",
            "spiro-mlops-promote=training.mlops.model_promoter:main",
            "spiro-mlops-rollback=training.mlops.rollback:main",
            "spiro-mlops-drift=training.mlops.drift_detector:main",
            "spiro-mlops-cl=training.mlops.continuous_learning:main",
            "spiro-mlops-retrain=training.mlops.retraining_scheduler:main",
            "spiro-trainer=training.trainer:main",
            "spiro-experiment=training.experiment:main",
            "spiro-callbacks=training.callbacks:main",
            "spiro-map=training.dataset_mapper:main",
            "spiro-clean=training.dataset_cleaner:main",
            "spiro-merge=training.dataset_merger:main",
            "spiro-split=training.dataset_splitter:main",
            "spiro-stats=training.dataset_statistics:main",
            "spiro-version=training.dataset_versioning:main",
            "spiro-dedup=training.duplicate_detector:main",
            "spiro-quality=training.quality_checker:main",
            "spiro-aug-preview=training.augment_preview:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
