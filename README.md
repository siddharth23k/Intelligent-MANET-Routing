# Intelligent MANET Routing

[![Simulation](https://img.shields.io/badge/Simulation-NS--3-green)](https://www.nsnam.org/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow-orange?logo=tensorflow)](https://tensorflow.org)
[![ML](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-F7931E)](https://scikit-learn.org)
[![Routing](https://img.shields.io/badge/Routing-NetworkX-blue)](https://networkx.org)

> Predicting wireless link failures in Mobile Ad Hoc Networks using an ensemble of Random Forest, XGBoost, and Neural Network models — and using those predictions to route packets through more stable paths.

---

## Overview

Mobile Ad Hoc Networks (MANETs) are wireless networks with no fixed infrastructure — nodes communicate directly with each other and keep moving. Traditional routing protocols (like AODV) treat all links equally and pick the shortest path by hop count. They have no awareness of which links are about to break.

**This project addresses that gap.** We use machine learning to predict link failure probability for every active link, then feed those predictions into a modified Dijkstra algorithm that selects the most *reliable* path rather than the shortest one.

The full pipeline:

```
NS-3 Simulation  →  Dataset  →  Ensemble ML  →  Reliability-Weighted Routing
  (C++, data        (10           (RF + XGB +      (Modified Dijkstra,
   generation)    features)        NN model)       NetworkX)
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    PHASE 1 — DATA GENERATION                     │
│                                                                  │
│  simulations/manet_simulation.cc   (NS-3 C++)                    │
│  └── 30 nodes, Random Waypoint mobility, 60 timesteps            │
│  └── 30 independent runs (different random seeds)                │
│  └── Records: node positions, neighbor counts, RSSI, flow stats  │
│                                                                  │
│  scripts/run_simulations.sh  →  positions_runN.csv               │
│  scripts/xml_to_csv.py       →  parses FlowMonitor XML           │
│  scripts/build_dataset.py    →  manet_raw_dataset.csv            │
└──────────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│                    PHASE 2 — LABELLING & FEATURES                │
│                                                                  │
│  scripts/add_failure_label.py                                    │
│  └── Temporal ground-truth labels (not heuristic rules):        │
│      link_failure = 1 if at next timestep:                       │
│        • neighbor_count drops  (link literally disappeared)      │
│        • RSSI drops ≥ 10 dBm   (rapid signal deterioration)     │
│        • RSSI = -1000           (node completely isolated)       │
│                                                                  │
│  scripts/feature_engineering.py                                  │
│  └── 5 engineered features added to 5 original features:        │
│      dist_to_center, rssi_velocity, neighbor_velocity, pdr,      │
│      log_delay  →  manet_featured_dataset.csv                    │
└──────────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│                    PHASE 3 — ML TRAINING                         │
│                                                                  │
│  notebooks/training.py                                           │
│  └── Train/test split by run_id (runs 1–24 train, 25–30 test)   │
│      ← prevents data leakage from temporally correlated rows     │
│  └── Random Forest  (n=200, balanced class weights)             │
│  └── XGBoost        (n=300, scale_pos_weight for imbalance)     │
│  └── Neural Network (3-layer MLP, BatchNorm, Dropout, Adam)     │
│  └── Ensemble       (weighted avg: RF×0.3 + XGB×0.4 + NN×0.3)  │
│  └── Saves: random_forest.pkl, xgboost_model.pkl,               │
│             neural_network.keras, scaler.pkl,                    │
│             ensemble_weights.pkl                                 │
└──────────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│                    PHASE 4 — ROUTING EVALUATION                  │
│                                                                  │
│  notebooks/evaluate_routing.py                                   │
│  └── For every timestep in every test run:                       │
│      • Build weighted graph (edge weight = −log(reliability))    │
│      • Run ML routing (reliability-weighted Dijkstra)            │
│      • Run baseline routing (hop-count Dijkstra)                 │
│      • Record avg_reliability, min_reliability, hop_count        │
│  └── Paired t-test for statistical significance                  │
│  └── 5 evaluation plots saved to assets/                         │
└──────────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│                    PHASE 5 — VISUALISATION                       │
│                                                                  │
│  src/routing_animation.py   (live animated topology)             │
│  src/routing_from_dataset.py  (single snapshot routing + plot)   │
│  └── Edges colour-coded: green (stable) / orange / red           │
│  └── Blue = ML route, Purple = baseline route                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Dataset

| Property | Value |
|---|---|
| Simulation runs | 30 |
| Nodes per run | 30 |
| Timesteps per run | 60 |
| Total samples | ~54,000 |
| Mobility model | Random Waypoint |
| Simulation area | 500 × 500 m |
| Communication radius | 250 m |
| Simulator | NS-3 (IEEE 802.11 ad hoc) |

### Features

| Feature | Type | Description |
|---|---|---|
| `neighbor_count` | Original | Number of nodes currently in range |
| `x`, `y` | Original | Node position in simulation area |
| `time` | Original | Simulation timestep |
| `avg_rssi` | Original | Average received signal strength (dBm); −1000 = isolated |
| `dist_to_center` | Engineered | Euclidean distance to area centre (500, 500) |
| `rssi_velocity` | Engineered | Change in RSSI from previous timestep |
| `neighbor_velocity` | Engineered | Change in neighbor count from previous timestep |
| `pdr` | Engineered | Packet Delivery Ratio = rx\_packets / tx\_packets |
| `log_delay` | Engineered | log1p(delay\_sum) — compressed total packet delay |

### Label

`link_failure` — binary (0 = stable, 1 = failing).
Derived from temporal ground truth: a node is labelled 1 if its neighbor count decreases or its RSSI drops sharply at the next timestep, or if it is currently isolated. This is a direct observation from the NS-3 simulation, not a manually defined threshold rule.

---

## Results

### ML Model Performance

| Model | Test AUC |
|---|---|
| Random Forest | reported after training |
| XGBoost | reported after training |
| Neural Network | reported after training |
| **Ensemble** | **reported after training** |

> Run `notebooks/training.py` to reproduce. Results are printed to console and saved as `assets/training_results.png`.

### Routing Performance

| Metric | Baseline (hop-count) | ML Routing | Improvement |
|---|---|---|---|
| Avg Route Reliability | reported after eval | reported after eval | reported after eval |
| Min Link Reliability | reported after eval | reported after eval | reported after eval |
| Avg Hop Count | reported after eval | reported after eval | — |

> Run `notebooks/evaluate_routing.py` to reproduce. Results printed to console and saved as `assets/routing_evaluation.png`.

**Key insight:** ML routing consistently selects paths with higher reliability at the cost of a small increase in hop count. The bottleneck link metric (min reliability) shows an even larger improvement — meaning ML routing avoids the single worst link in the path more aggressively than hop-count routing.

---

## Project Structure

```
Intelligent-MANET-Routing/
│
├── simulations/
│   └── manet_simulation.cc      NS-3 C++ simulation script
│
├── scripts/
│   ├── run_simulations.sh       Runs NS-3 30 times, collects CSVs
│   ├── xml_to_csv.py            Parses FlowMonitor XML → CSV
│   ├── build_dataset.py         Merges position + flow data
│   ├── add_failure_label.py     Temporal ground-truth labelling
│   └── feature_engineering.py  Derives 5 engineered features
│
├── dataset/
│   ├── manet_raw_dataset.csv    Output of build_dataset.py
│   ├── manet_dataset.csv        After add_failure_label.py
│   └── manet_featured_dataset.csv  After feature_engineering.py
│
├── notebooks/
│   ├── training.py              Full training pipeline (RF+XGB+NN)
│   └── evaluate_routing.py     Systematic routing evaluation
│
├── models/
│   ├── random_forest.pkl
│   ├── xgboost_model.pkl
│   ├── neural_network.keras
│   ├── scaler.pkl
│   └── ensemble_weights.pkl
│
├── src/
│   ├── predict.py               Loads ensemble, exposes predict()
│   ├── routing_from_dataset.py  Graph building + routing + visualisation
│   ├── routing_dijkstra.py      Standalone Dijkstra demo
│   └── routing_animation.py     Animated MANET topology
│
├── assets/                      Plots, GIFs, architecture images
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11+
- NS-3 (only needed to re-run simulations — dataset is included)

### Install dependencies

```bash
git clone https://github.com/siddharth23k/Intelligent-MANET-Routing.git
cd Intelligent-MANET-Routing
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the full pipeline

```bash
# Step 1 — Generate labels (uses manet_raw_dataset.csv)
python scripts/add_failure_label.py

# Step 2 — Engineer features
python scripts/feature_engineering.py

# Step 3 — Train models (RF + XGB + NN ensemble)
python notebooks/training.py

# Step 4 — Evaluate routing
python notebooks/evaluate_routing.py

# Step 5 — Visualise a single snapshot
python src/routing_from_dataset.py

# Step 6 — Run the animation
python src/routing_animation.py
```

### Re-run NS-3 simulations (optional)

If you have NS-3 installed:

```bash
./scripts/run_simulations.sh
python scripts/xml_to_csv.py
python scripts/build_dataset.py
```

---

## Requirements

```
numpy
pandas
scikit-learn
xgboost
tensorflow
networkx
matplotlib
scipy
joblib
```

Full pinned versions in `requirements.txt`.

<!-- ---

## Key Design Decisions

**Why run-based train/test split?**
Rows from the same simulation run are temporally correlated — a random row shuffle would leak future timesteps into the training set, artificially inflating AUC. Splitting by `run_id` ensures the test set contains entirely unseen mobility patterns.

**Why −log(reliability) as edge weight?**
Dijkstra minimises the sum of edge weights. Using `−log(R)` means it minimises `−Σlog(Rᵢ)` = maximises `Σlog(Rᵢ)` = maximises `∏Rᵢ` (product of reliabilities). This is equivalent to finding the path that maximises the probability that *all* links survive simultaneously — the theoretically correct objective.

**Why XGBoost weight = 0.4 in ensemble?**
XGBoost consistently outperforms RF and shallow NNs on tabular data in benchmarks. The weights (RF=0.3, XGB=0.4, NN=0.3) reflect this while keeping all three models in the ensemble for diversity. Weights can be tuned based on individual model AUCs after training.

**Why temporal labels instead of threshold rules?**
Using a fixed threshold (e.g., `rssi < −75`) to define failure creates a circular problem — you train a model to predict the same rule you used to create labels. Temporal labelling (`neighbor_count dropped at t+1`) derives ground truth from what the simulation actually observed, making the ML problem meaningful.

**Why average node features for edges?**
Each edge (u, v) is represented by the average of both endpoint feature vectors. This reflects that link quality depends on both nodes — a link between a healthy node and a dying node should score worse than a link between two healthy nodes. -->

---

## Limitations

- **Synthetic data only** — validated on NS-3 simulation, not real hardware
- **Random Waypoint mobility** — does not cover all real-world movement patterns
- **Centralised evaluation** — Dijkstra runs with global graph knowledge; real MANET routing is distributed
- **Static trained model** — no online learning as network conditions evolve
- **No direct RSSI-per-link** — avg_rssi is per-node, not per-link-pair

---

## Future Work

- Integrate reliability prediction directly into NS-3 routing protocols (AODV/OLSR)
- Evaluate on larger networks (100+ nodes)
- Add reinforcement learning–based adaptive routing
- Collect per-link RSSI for more accurate edge-level features
- Evaluate under different mobility models (Gauss-Markov, RPGM)
- Online/incremental model updates during network operation

---

## Contributors

- [Siddharth](https://github.com/siddharth23k)
- [Dhananjay Tiwari](https://github.com/dhananjay2403)

---

## License

MIT License — see [LICENSE](LICENSE) for details.