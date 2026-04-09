# Intelligent MANET Routing

This repo compares:

1. A **paper FRLFP baseline** trained as an **SFRNNR** (fuzzification → fuzzy RNN → normalization → consequent → defuzzification + adaptive threshold head) plus **FRLFP-style routing**
2. **Our model** (RF + XGBoost failure predictor + reliability-weighted routing)
3. A **classic shortest-path baseline**

All three are now evaluated on the **same NS-3 generated dataset** and on the **same sampled source-destination decisions**.

---

## What changed (important)

- There is now a strict shared-data comparison pipeline.
- `experiments/compare_all_models.py` performs a unified evaluation loop, so both methods get identical `(run_id, time, src, dst)` cases.
- Our training now uses the same paper dataset during comparison:
  - `python experiments/training.py --dataset dataset/paper/processed/paper_lfp_dataset.csv`
- `scripts/paper_build_dataset.py` no longer silently falls back to old dataset files.

---

## One-command style run order

From repo root:

```bash
# 1) Generate NS-3 raw logs for paper baseline
bash scripts/run_paper_simulations.sh

# 2) Build + feature + LFP datasets
python scripts/paper_build_dataset.py
python scripts/paper_feature_engineering.py
python scripts/paper_compute_lfp.py

# 3) Train OUR model on same dataset
python experiments/training.py --dataset dataset/paper/processed/paper_lfp_dataset.csv

# 4) Compare all methods on same sampled decisions
python experiments/compare_all_models.py
```

Outputs:

- `results/ours_results.csv`
- `results/paper_baseline_results.csv`
- `results/classic_baseline_results.csv`
- `results/comparison_metrics.csv`
- `results/stat_tests.csv`
- `results/comparison_summary.md`

---

## File-by-file usage map

## Core comparison flow (active)

- `simulations/paper_frlfp_simulation.cc`
  - Standalone paper-style NS-3 simulation logic.
- `scripts/run_paper_simulations.sh`
  - Runs paper simulation seeds and writes raw logs to `dataset/paper/raw/`.
- `scripts/paper_build_dataset.py`
  - Builds `dataset/paper/processed/paper_raw_dataset.csv` from NS-3 CSV/XML outputs.
- `scripts/paper_feature_engineering.py`
  - Computes paper-inspired factors and our model features in one shared table.
- `scripts/paper_compute_lfp.py`
  - Adds supervised labels, runs **SFRNNR** (trains `models/sfrnnr_paper.keras` if missing), writes `lfp`, `lfp_threshold`, and `paper_predicted_failure`.
- `experiments/train_sfrnnr_paper.py`
  - Standalone trainer for the paper SFRNNR (same outputs as above).
- `baseline_paper/sfrnnr_model.py`
  - Layered SFRNNR Keras model (fuzzification, fuzzy RNN, normalization, consequent, summation, threshold head).
- `baseline_paper/sfrnnr_infer.py`
  - Batch inference over node--time sequences.
- `baseline_paper/threshold_model.py`
  - Teacher signal for the threshold head during SFRNNR training only.
- `baseline_paper/frlfp_router.py`
  - Paper baseline routing behavior.
- `experiments/training.py`
  - Trains RF/XGB model (our model). Use `--dataset` for shared paper dataset.
- `experiments/compare_all_models.py`
  - Unified evaluator that:
    - rebuilds paper datasets,
    - trains our model on shared dataset,
    - evaluates both methods on the same sampled pairs,
    - writes result tables.
- `src/predict.py`
  - Loads trained RF/XGB artifacts.
- `src/routing_from_dataset.py`
  - Graph creation + route metrics for our method.

## Legacy/original pipeline (not used in strict paper comparison)

- `simulations/manet_simulation.cc`
- `scripts/run_simulations.sh`
- `scripts/build_dataset.py`
- `scripts/add_failure_label.py`
- `scripts/feature_engineering.py`
- `experiments/evaluate_routing.py`
- `experiments/evaluate_ours.py`
- `experiments/evaluate_paper_baseline.py`
- `experiments/evaluate_classic_baseline.py`

These are kept for backward compatibility and debugging, but `compare_all_models.py` is now the authoritative comparison path.

## Visualization/utility scripts (manual)

- `src/routing_animation.py`
  - Optional animation; not called by unified comparison script.
- `src/routing_dijkstra.py`
  - Standalone demo; not required for comparison.

---

## Folder structure (current expected)

```text
Intelligent-MANET-Routing/
├── baseline_paper/
├── configs/
├── dataset/
│   └── paper/
│       ├── raw/
│       └── processed/
├── docs/
├── experiments/
├── models/
├── results/
├── scripts/
├── simulations/
└── src/
```

This layout is correct for the current workflow.

---

## Setup

### Prerequisites

- Python 3.11 recommended
- NS-3 (for simulation generation)

### Install

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Notes and limitations

- The SFRNNR is implemented in our codebase to match the *described* stack (fuzzy inputs, recurrent encoder, defuzzified LFP, learned threshold). It is **not** byte-identical to the original authors’ unreleased code; hyperparameters and membership counts can be tuned in `baseline_paper/sfrnnr_model.py`.
- First run of `paper_compute_lfp.py` **trains** the SFRNNR using **fast defaults** (small net, 2 epochs, batch 512, at most 600 training sequences) so it usually finishes in a few minutes; use `python experiments/train_sfrnnr_paper.py --max-train-sequences 0 --epochs 12` (and larger `--gru-units`, `--rule-units`, `--n-mfs`) for a heavier model. Later runs load `models/sfrnnr_paper.keras` if present.
- Results are simulation-based.
- Routing evaluation is centralized (graph known at evaluation time).

---

## License

MIT License — see `LICENSE`.