PYTHON ?= python3

.PHONY: help smoke test data features validate train baseline compare all clean

help:
	@echo "make smoke     end to end run on a tiny subset, finishes in well under a minute"
	@echo "make test      unit tests"
	@echo "make all       the full pipeline (needs data/raw from the NS-3 runs)"
	@echo "make data      stage 1: flatten NS-3 output"
	@echo "make features  stage 2: temporal features and paper factors"
	@echo "make validate  stage 3: data quality gate"
	@echo "make train     stage 4: our RF + XGBoost predictor"
	@echo "make baseline  stage 5: SFRNNR paper baseline"
	@echo "make compare   stage 6: three way comparison"
	@echo "make clean     remove generated datasets, models and results"

smoke:
	$(PYTHON) pipeline/smoke_test.py

test:
	$(PYTHON) -m pytest

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
	rm -f results/models/* results/*.csv results/*.json results/comparison_summary.md
