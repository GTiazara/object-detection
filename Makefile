.PHONY: help install install-dev clean run run-example test format lint setup

# Variables
DEF_FILE=object-detection.def
IMAGE_NAME=object-detection-image
SIF_FILE=object-detection.sif
#
PYTHON := python
POETRY := poetry
MODULE := airplane_detection
INPUT_DIR := data/input
OUTPUT_DIR := data/output
MODEL := yolov8x
IMGSZ := 960
CONF := 0.25
IOU := 0.45

# Default target
help:
	@echo "Available targets:"
	@echo "  make install          - Install project dependencies"
	@echo "  make install-dev      - Install project with dev dependencies"
	@echo "  make setup            - Setup project (install + create directories)"
	@echo "  make run INPUT=path   - Run detection on image or directory"
	@echo "  make run-example      - Run example script"
	@echo "  make clean            - Clean output files and caches"
	@echo "  make test             - Run tests"
	@echo "  make format           - Format code with black"
	@echo "  make lint             - Lint code with ruff/flake8"
	@echo ""
	@echo "Examples:"
	@echo "  make run INPUT=data/input/image.jpg"
	@echo "  make run INPUT=data/input MODEL=yolov8n IMGSZ=640"
	@echo "  make run INPUT=data/input OUTPUT=results/"

build-docker:
	docker build . -t ${IMAGE_NAME}

build-app:
	rm -f ${SIF_FILE}
	apptainer build ${SIF_FILE} ${DEF_FILE}

build: build-docker build-app

# Install dependencies
install:
	$(POETRY) install --no-dev

# Install with dev dependencies
install-dev:
	$(POETRY) install

# Setup project (install + create directories)
setup: install
	@echo "Creating data directories..."
	@mkdir -p $(INPUT_DIR)
	@mkdir -p $(OUTPUT_DIR)
	@echo "Setup complete!"

# Run detection on input image or directory
run:
	@if [ -z "$(INPUT)" ]; then \
		echo "Error: INPUT is required. Usage: make run INPUT=path/to/image.jpg"; \
		exit 1; \
	fi
	$(POETRY) run $(PYTHON) -m $(MODULE).main $(INPUT) \
		--model $(MODEL) \
		--imgsz $(IMGSZ) \
		--conf $(CONF) \
		--iou $(IOU) \
		$(if $(OUTPUT),--output $(OUTPUT),)

# Run with custom model
run-model:
	@if [ -z "$(INPUT)" ] || [ -z "$(MODEL)" ]; then \
		echo "Error: INPUT and MODEL are required. Usage: make run-model INPUT=path MODEL=yolov8x"; \
		exit 1; \
	fi
	$(POETRY) run $(PYTHON) -m $(MODULE).main $(INPUT) \
		--model $(MODEL) \
		--imgsz $(IMGSZ) \
		--conf $(CONF) \
		--iou $(IOU) \
		$(if $(OUTPUT),--output $(OUTPUT),)

# Run with local model
run-local:
	@if [ -z "$(INPUT)" ] || [ -z "$(MODEL_PATH)" ]; then \
		echo "Error: INPUT and MODEL_PATH are required. Usage: make run-local INPUT=path MODEL_PATH=path/to/model.pt"; \
		exit 1; \
	fi
	$(POETRY) run $(PYTHON) -m $(MODULE).main $(INPUT) \
		--local-model $(MODEL_PATH) \
		--imgsz $(IMGSZ) \
		--conf $(CONF) \
		--iou $(IOU) \
		$(if $(OUTPUT),--output $(OUTPUT),)

# Run example script
run-example:
	$(POETRY) run $(PYTHON) example.py

run-local-train-example:
	$(POETRY) run $(PYTHON) src/examples/train_example.py

# Force download model
download-model:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL is required. Usage: make download-model MODEL=yolov8x"; \
		exit 1; \
	fi
	$(POETRY) run $(PYTHON) -m $(MODULE).main $(INPUT_DIR) --model $(MODEL) --force-download

# Clean output files and caches
clean:
	@echo "Cleaning output files..."
	@rm -rf $(OUTPUT_DIR)/*
	@rm -rf .pytest_cache
	@rm -rf __pycache__
	@rm -rf src/**/__pycache__
	@rm -rf src/**/*.pyc
	@rm -rf .mypy_cache
	@rm -rf .ruff_cache
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@echo "Clean complete!"

# Run tests
test:
	@if [ -d "tests" ] && [ -n "$$(ls -A tests 2>/dev/null)" ]; then \
		$(POETRY) run pytest tests/ -v; \
	else \
		echo "No tests found or tests directory is empty"; \
	fi

# Format code (requires black)
format:
	@if $(POETRY) run which black > /dev/null 2>&1; then \
		$(POETRY) run black src/ tests/ example.py; \
	else \
		echo "black not installed. Install with: poetry add --group dev black"; \
	fi

# Lint code (requires ruff or flake8)
lint:
	@if $(POETRY) run which ruff > /dev/null 2>&1; then \
		$(POETRY) run ruff check src/ tests/ example.py; \
	elif $(POETRY) run which flake8 > /dev/null 2>&1; then \
		$(POETRY) run flake8 src/ tests/ example.py; \
	else \
		echo "No linter found. Install with: poetry add --group dev ruff"; \
	fi

# Show project info
info:
	@echo "Project: airplane-detection"
	@echo "Python: $$($(POETRY) run $(PYTHON) --version)"
	@echo "Poetry: $$($(POETRY) --version)"
	@echo "Input directory: $(INPUT_DIR)"
	@echo "Output directory: $(OUTPUT_DIR)"
	@echo "Default model: $(MODEL)"
	@echo "Default image size: $(IMGSZ)"

# Quick detection with default settings
detect:
	@if [ -z "$(INPUT)" ]; then \
		echo "Error: INPUT is required. Usage: make detect INPUT=path/to/image.jpg"; \
		exit 1; \
	fi
	@make run INPUT=$(INPUT) MODEL=$(MODEL) IMGSZ=$(IMGSZ) CONF=$(CONF) IOU=$(IOU)

run-apptainer-shell:
	apptainer shell --pwd /app --nv \
	--bind ./src:/app/src \
	--bind /home/GTiazara/Documents/workspace/get_experience_project/object-detection/data/local_data:/app/input_data \
	--bind /home/GTiazara/Documents/workspace/get_experience_project/object-detection/training:/app/output_data \
	--bind ./model:/app/model \
	${SIF_FILE} \