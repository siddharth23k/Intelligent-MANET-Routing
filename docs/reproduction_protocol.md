# Reproduction Protocol: Paper Baseline vs Our Model

This repository contains:

1. A **paper-inspired FRLFP baseline** re-implementation.
2. The existing **ensemble + reliability-weighted routing** method.
3. A **classic shortest-path baseline**.

## Scope and claims

- The paper provides simulation settings and equations, but not full released code.
- Therefore, this implementation is a **paper-faithful approximation**, not an exact binary reproduction.
- All comparisons should be run on the **same generated dataset and split protocol** for fairness.

## Fairness contract

- Same run seeds for all methods.
- Same train/test run split (by `run_id`).
- Same source-destination sampling per snapshot.
- Same communication radius per evaluation job.

## Recommended execution order

1. Generate simulation data (optional): `bash scripts/run_paper_simulations.sh`
2. Build paper dataset: `python scripts/paper_build_dataset.py`
3. Compute paper features: `python scripts/paper_feature_engineering.py`
4. Compute paper LFP and threshold columns: `python scripts/paper_compute_lfp.py`
5. Train the paper SFRNNR (or let `paper_compute_lfp.py` train it on first run): `python experiments/train_sfrnnr_paper.py`
6. Train our model (if missing): `python experiments/training.py --dataset dataset/paper/processed/paper_lfp_dataset.csv`
7. Evaluate all methods: `python experiments/compare_all_models.py`

Outputs are written under `results/` and `assets/comparison/`.