# Intelligent MANET Routing

A clean comparison of MANET routing strategies: **Paper Baseline** vs **Our RF+XGBoost Enhancement** vs **Classic Shortest-Path**.

## What This Project Does

1. **Paper Baseline**: Implements FRLFP (Fuzzy Reliability Link Failure Prediction) using SFRNNR (fuzzification → fuzzy RNN → normalization → consequent → defuzzification + adaptive threshold head)
2. **Our Enhancement**: Uses RandomForest + XGBoost failure predictor with reliability-weighted routing 
3. **Classic Baseline**: Traditional shortest-path routing

All methods are evaluated on **identical NS-3 simulation data** with **fair comparison protocols**.

---

## Project Structure (Clean & Intuitive)

```
Intelligent-MANET-Routing/
|
| methods/                   # CORE METHODS COMPARISON
|   baseline/               # Paper baseline (SFRNNR + FRLFP)
|   ours/                   # Our RF+XGBoost enhancement
|
| data/                     # UNIFIED DATASET LOCATION
|   raw/                    # Raw NS-3 simulation outputs
|   processed/              # Processed datasets for training
|
| simulation/              # NS-3 SIMULATION SETUP
| pipeline/                # DATA PROCESSING PIPELINE
| results/                 # COMPARISON RESULTS & MODELS
| config/                  # CONFIGURATION FILES
| legacy/                  # OLD IMPLEMENTATION (archived)
```

---

## Quick Start (4 Commands)

```bash
# 1) Generate NS-3 simulation data
bash simulation/run_simulation.sh

# 2) Process data and compute features
python pipeline/generate_data.py
python pipeline/engineer_features.py

# 3) Train both models on the same data
python pipeline/train_models.py
python pipeline/train_sfrnnr_paper.py

# 4) Compare all methods and get results
python pipeline/compare_methods.py
```

**Results**: Check `results/` for detailed comparison metrics.

---

## Method Details

### **Paper Baseline** (`methods/baseline/`)
- **SFRNNR Model**: Fuzzy neural network with adaptive threshold
- **FRLFP Routing**: Excludes risky nodes based on predicted failures
- **9 Factors**: Distance, LET, ND, RSSI, LS, LA, LQ_mean, LL_d, T_hello

### **Our Enhancement** (`methods/ours/`)
- **RF + XGBoost**: Ensemble failure prediction
- **Reliability-Weighted Routing**: Uses predicted link reliability
- **14 Features**: Network topology + temporal patterns

### **Classic Baseline**
- **Shortest-Path**: Traditional hop-count based routing

---

## Fair Comparison Protocol

- Same NS-3 simulation seeds for all methods
- Identical train/test splits by run_id
- Same source-destination pairs per evaluation
- Unified communication radius settings

---

## Setup

### Prerequisites
- Python 3.11+ 
- NS-3 (for simulation generation)

### Installation
```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Folder Structure (Clean & Organized)

```text
Intelligent-MANET-Routing/
|-- methods/           # Core method implementations
|   |-- baseline/       # Paper SFRNNR + FRLFP
|   `-- ours/          # Our RF+XGBoost enhancement
|
|-- data/              # All datasets in one place
|   |-- raw/           # NS-3 simulation outputs
|   `-- processed/     # Ready-to-use datasets
|
|-- simulation/        # NS-3 simulation setup
|-- pipeline/          # Data processing & training
|-- results/           # Models, comparisons, figures
|-- config/            # Configuration files
|-- legacy/            # Old implementation (archived)
```

---

## Key Differences from Original Project

**Before**: Confusing structure with files scattered across multiple folders
**After**: Clean separation by purpose - methods, data, pipeline, results

**Before**: Legacy files mixed with active code
**After**: Legacy isolated, active code clearly organized

**Before**: Unclear what belongs to baseline vs our method
**After**: Explicit `methods/baseline/` vs `methods/ours/` folders

---

## Notes and limitations

- The SFRNNR is implemented in our codebase to match the *described* stack (fuzzy inputs, recurrent encoder, defuzzified LFP, learned threshold). It is **not** byte-identical to the original authors’ unreleased code; hyperparameters and membership counts can be tuned in `baseline_paper/sfrnnr_model.py`.
- First run of `paper_compute_lfp.py` **trains** the SFRNNR using **fast defaults** (small net, 2 epochs, batch 512, at most 600 training sequences) so it usually finishes in a few minutes; use `python experiments/train_sfrnnr_paper.py --max-train-sequences 0 --epochs 12` (and larger `--gru-units`, `--rule-units`, `--n-mfs`) for a heavier model. Later runs load `models/sfrnnr_paper.keras` if present.
- Results are simulation-based.
- Routing evaluation is centralized (graph known at evaluation time).

---

## License

MIT License — see `LICENSE`.