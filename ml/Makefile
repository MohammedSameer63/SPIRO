# SPIRO ML — Makefile
.PHONY: install install-dev test test-unit test-integration coverage lint format clean dataset train eval export infer

# ─── Install ───────────────────────────────────────────────────────────────────
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-gpu:
	pip install -e ".[gpu]"

# ─── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -m unit -v

test-integration:
	pytest tests/integration/ -m integration -v

coverage:
	pytest tests/unit/ --cov=src/lib/ml --cov-report=html --cov-report=term-missing
	@echo "HTML report: htmlcov/index.html"

# ─── Code Quality ──────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	mypy src/lib/ml --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

# ─── Data Pipeline ─────────────────────────────────────────────────────────────
dataset:
	python -m lib.ml.scripts.dataset_manager \
		--config configs/base_config.yaml \
		--action all

dataset-validate:
	python -m lib.ml.scripts.dataset_manager \
		--config configs/base_config.yaml \
		--action validate

dataset-augment:
	python -m lib.ml.scripts.dataset_manager \
		--config configs/base_config.yaml \
		--action augment --factor 3

# ─── Training ──────────────────────────────────────────────────────────────────
train-yolo:
	python -m lib.ml.scripts.train \
		--config configs/experiments/yolov11_baseline.yaml \
		--version v1

train-efficientnet:
	python -m lib.ml.scripts.train \
		--config configs/experiments/efficientnetv2_classifier.yaml \
		--architecture efficientnetv2 \
		--version v1

# ─── Evaluation ────────────────────────────────────────────────────────────────
eval:
	python -m lib.ml.scripts.evaluate \
		--config configs/base_config.yaml \
		--production

# ─── Export ────────────────────────────────────────────────────────────────────
export:
	python -m lib.ml.scripts.export_onnx \
		--config configs/base_config.yaml \
		--production \
		--verify

# ─── Inference ─────────────────────────────────────────────────────────────────
benchmark:
	python -m lib.ml.scripts.infer \
		--onnx models/exports/spiro_yolov11.onnx \
		--benchmark --n-runs 200

# ─── Monitoring ────────────────────────────────────────────────────────────────
tensorboard:
	tensorboard --logdir logs/tensorboard --port 6006

mlflow:
	mlflow ui --backend-store-uri logs/mlflow --port 5000

# ─── Production Optimization (Prompt 7) ───────────────────────────────────────
optimize:
	@echo "Usage: make optimize ONNX=path/to/model.onnx MODEL=yolov11s SIZE=640"
	python scripts/optimize.py --onnx $(ONNX) --model-id $(MODEL) --input-size $(SIZE)

benchmark:
	@echo "Usage: make benchmark ONNX=path/to/model.onnx MODEL=yolov11s"
	python scripts/benchmark.py --onnx $(ONNX) --model-id $(MODEL) --n-runs 100 --load-test

robustness:
	@echo "Usage: make robustness MODEL=yolov11s"
	python -c " \
import sys; sys.path.insert(0, 'src'); \
from lib.ml.robustness.robustness_tester import RobustnessTester; \
t = RobustnessTester(n_per_category=20, report_dir=__import__('pathlib').Path('reports')); \
r = t.run_all(model_id='$(MODEL)'); t.save_report(r); \
print(f'Stability: {r.overall_stability:.1%}') \
"

healthcheck:
	python scripts/healthcheck.py

docker-build-cpu:
	docker build -f docker/Dockerfile.cpu -t spiro-ml-cpu:latest .

docker-build-gpu:
	docker build -f docker/Dockerfile.gpu -t spiro-ml-gpu:latest .

docker-build-training:
	docker build -f docker/Dockerfile.training -t spiro-ml-training:latest .

docker-run-cpu:
	docker run --rm \
		-v $(PWD)/models:/app/models:ro \
		-v $(PWD)/configs:/app/configs:ro \
		spiro-ml-cpu:latest python scripts/healthcheck.py

checklist:
	@cat deployment/deployment_checklist.md

# ─── MLOps Platform ────────────────────────────────────────────────────────────
mlops-registry:
	python training/mlops/model_registry.py export

mlops-validate:
	@echo "Usage: make mlops-validate ONNX=path/to/model.onnx MODEL=yolov11s VER=v1.0.0"
	python training/mlops/model_validator.py \
		--onnx $(ONNX) --model-id $(MODEL) --version $(VER) \
		--task detection --input-size 640

mlops-promote:
	@echo "Usage: make mlops-promote MODEL=yolov11s VER=v2.0.0"
	python training/mlops/model_promoter.py \
		--model-id $(MODEL) --version $(VER) --auto-deploy

mlops-deploy:
	@echo "Usage: make mlops-deploy ONNX=path MODEL=yolov11s VER=v1.0.0"
	python training/mlops/model_deployer.py deploy \
		--model-id $(MODEL) --version $(VER) \
		--onnx $(ONNX) --strategy blue_green

mlops-rollback:
	python training/mlops/rollback.py --task detection

mlops-drift:
	@echo "Usage: make mlops-drift MODEL=yolov11s BASELINE=b.npy CURRENT=c.npy"
	python training/mlops/drift_detector.py \
		--model-id $(MODEL) \
		--baseline-confidences $(BASELINE) \
		--current-confidences $(CURRENT)

mlops-performance:
	python training/mlops/performance_monitor.py export

mlops-cl-cycle:
	python training/mlops/continuous_learning.py run-cycle --force

mlops-retrain-weekly:
	python training/mlops/retraining_scheduler.py schedule \
		--model-id yolov11s --schedule weekly

mlops-reports:
	python training/mlops/model_registry.py export
	python training/mlops/dataset_registry.py export
	python training/mlops/experiment_tracker.py export
	python training/mlops/metrics_store.py export
	python training/mlops/performance_monitor.py export
	python training/mlops/continuous_learning.py report

# ─── Production Inference Pipeline ────────────────────────────────────────────
pipeline-infer:
	@echo "Usage: make pipeline-infer IMG=path/to/image.jpg"
	python training/infer_pipeline.py --image $(IMG) --pretty

pipeline-batch:
	@echo "Usage: make pipeline-batch SRC=path/to/images/ OUT=results/"
	python training/infer_pipeline.py --source $(SRC) --output-dir $(OUT)

pipeline-benchmark:
	python training/infer_pipeline.py --benchmark --n-runs 100

# ─── EfficientNetV2 Verification ──────────────────────────────────────────────
train-effnet:
	python training/train_effnet.py --variant s

train-effnet-b0:
	python training/train_effnet.py --variant b0

train-effnet-m:
	python training/train_effnet.py --variant m

export-effnet:
	python training/verify_export.py \
		--config configs/verification/effnetv2_s.yaml \
		--weights models/verification_checkpoints/spiro_effnetv2_s/best_effnet.pt

eval-effnet:
	python training/verify_metrics.py \
		--onnx models/exports/spiro_effnetv2_s_best_effnet.onnx \
		--config configs/verification/effnetv2_s.yaml \
		--split test --plots

explain:
	@echo "Usage: make explain IMAGE=path/to/crop.jpg"
	python training/verify_model.py explain \
		--weights models/verification_checkpoints/spiro_effnetv2_s/best_effnet.pt \
		--config configs/verification/effnetv2_s.yaml \
		--image $(IMAGE) --output-dir reports/gradcam

configs-list:
	python training/verify_config.py list

# ─── YOLOv11 Training ─────────────────────────────────────────────────────────
train-nano:
	python training/trainer.py --variant n

train-small:
	python training/trainer.py --variant s

train-medium:
	python training/trainer.py --variant m

train-large:
	python training/trainer.py --variant l

train-xlarge:
	python training/trainer.py --variant x

resume-small:
	python training/trainer.py --variant s --resume

validate:
	python training/train.py \
		--config configs/training/yolo11s.yaml \
		--validate-only \
		--weights models/checkpoints/spiro_yolo11s/weights/best.pt

experiments:
	python training/experiment.py list

compare-variants:
	python training/experiment.py compare \
		--names spiro_yolo11n spiro_yolo11s spiro_yolo11m

export-best:
	python training/experiment.py export --name spiro_yolo11s

test-callbacks:
	python training/callbacks.py --list
	python training/callbacks.py --test-csv

# ─── Dataset Engineering ───────────────────────────────────────────────────────
download:
	python training/dataset_download.py

map:
	python training/dataset_mapper.py --raw-dir datasets/raw

clean-data:
	python training/dataset_cleaner.py --mapped-dir datasets/mapped

merge:
	python training/dataset_merger.py

split:
	python training/dataset_splitter.py

stats:
	python training/dataset_statistics.py --dir datasets/merged --charts

quality:
	python training/quality_checker.py --dir datasets/merged --charts

dedup:
	python training/duplicate_detector.py --dir datasets/merged --report reports/duplicates.json

version:
	python training/dataset_versioning.py create --processed-dir datasets/processed

validate-data:
	python training/dataset_validator.py --dir datasets/merged --strict

pipeline: download map clean-data merge split stats version
	@echo "Full dataset pipeline complete ✓"

aug-preview:
	@echo "Usage: make aug-preview IMG=path/to/image.jpg"
	python training/augment_preview.py --image $(IMG) --n 6 --show-grid

# ─── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ dist/ build/ *.egg-info/
