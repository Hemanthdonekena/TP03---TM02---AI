SHELL := /bin/bash
MODEL_DIR ?= data

.PHONY: setup data train api test lint

setup:
	pip install -r requirements.txt

data:
	python scripts/generate_synthetic_data.py --out_dir data

train:
	bash scripts/run_training.sh

api:
	bash scripts/start_api.sh

test:
	pytest -q

lint:
	ruff check .
