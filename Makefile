PYTHON ?= python3

.PHONY: help smoke test lint data features validate train baseline compare all clean

help:
	@echo "make smoke     end to end run on a tiny subset (about 20 seconds)"
	@echo "make test      unit tests"
	@echo "make all       full pipeline (needs data/raw from the NS-3 runs)"
	@echo ""
	@echo "individual stages:"
	@echo "  make data      1  flatten NS-3 output"
	@echo "  make features  2  temporal features and paper factors"
	@echo "  make validate  3  data quality gate"
	@echo "  make train     4  our RF + XGBoost predictor"
	@echo "  make baseline  5  SFRNNR paper baseline"
	@echo "  make compare   6  three way comparison"
	@echo ""
	@echo "make clean     remove generated datasets, models and results"

smoke:
	$(PYTHON) pipeline/smoke_test.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m compileall -q config methods pipeline tests

data:
	$(PYTHON) pipeline/generate_data.py

features:
	$(PYTHON) pipeline/engineer_features.py

validate:
	$(PYTHON) pipeline/validate_dataset.py

train:
	$(PYTHON) pipeline/train_predictor.py

baseline:
	$(PYTHON) pipeline/train_models.py --retrain

compare:
	$(PYTHON) pipeline/compare_methods.py

all: data features validate train baseline compare

clean:
	rm -f data/processed/*.csv data/processed/dataset_manifest.json
	rm -f results/*.csv results/*.json results/comparison_summary.md
	rm -f results/models/*.pkl results/models/*.keras results/models/*.json
	rm -rf .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
