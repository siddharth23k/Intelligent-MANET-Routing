# Intelligent MANET Routing

[Simulation](https://www.nsnam.org/)
[Python](https://python.org)
[Framework](https://tensorflow.org)
[ML](https://scikit-learn.org)
[Routing](https://networkx.org)

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
│  scripts/build_dataset.py    →  manet_raw_dataset.csv            │
└──────────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│                    PHASE 2 — LABELLING & FEATURES                │
│                                                                  │
│  scripts/add_failure_label.py                                    │
│  └── Temporal labels with a prediction horizon (PREDICT_HORIZON): │
│      link_failure = 1 if within next 5 timesteps:                │
│        • neighbor_count drops by ≥ 2                              │
│        • avg_rssi drops by ≥ 15 dBm (when future RSSI exists)     │
│        • future avg_rssi == -1000 (becomes isolated)              │
│        • current avg_rssi == -1000 (already isolated)             │
│                                                                  │
│  scripts/feature_engineering.py                                  │
│  └── Adds engineered + rolling history features →                │
│      manet_featured_dataset.csv                                   │
└──────────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│                    PHASE 3 — ML TRAINING                         │
│                                                                  │
│  experiments/training.py                                         │
│  └── Train/test split by run_id (25–30 are held-out test runs)    │
│      ← reduces leakage from temporally correlated rows            │
│  └── StandardScaler fit on train features only                    │
│  └── Random Forest + XGBoost trained on scaled features           │
│  └── Ensemble = weighted average of model probabilities           │
│  └── Saves: models/random_forest.pkl, models/xgboost_model.pkl,   │
│           models/scaler.pkl, models/ensemble_weights.pkl          │
└──────────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│                    PHASE 4 — ROUTING EVALUATION                  │
│                                                                  │
│  experiments/evaluate_routing.py                                 │
│  └── For every timestep in every test run:                       │
│      • Build connectivity graph from positions within radius      │
│      • Predict per-edge reliability with the ML ensemble          │
│      • ML routing: Dijkstra on weight = −log(reliability)         │
│      • Baseline: shortest path by hop count                       │
│      • Record avg/min link reliability and hop counts             │
│  └── Paired t-test for statistical significance                  │
│  └── Saves plot: assets/routing_evaluation.png                    │
│  └── Saves raw results: dataset/routing_results.csv               │
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

## End-to-End Workflow (Detailed)

This section describes the **full runtime/data flow**, including **dataset handling**, **model internals**, and **how outputs are produced**.

### 1) Data generation (NS-3 → CSV)

- **Goal**: produce per-node, per-timestep observations + per-run flow aggregates.
- **Where it comes from**:
  - **Positions/topology**: `dataset/positions_run*.csv` (one CSV per run).
  - **Flow aggregates**: `dataset/manet_flowmon_run*.xml` (one XML per run).
- **How it’s merged**:
  - `scripts/build_dataset.py` reads `positions_run*.csv` and `manet_flowmon_run*.xml` and produces:
    - `dataset/manet_raw_dataset.csv`

**Output schema** (high-level):

- **Keys**: `run_id`, `time`, `node_id`
- **Node state**: `x`, `y`, `neighbor_count`, `avg_rssi`
- **Run-level traffic**: `tx_packets`, `rx_packets`, `lost_packets`, `delay_sum`

### 2) Label generation (temporal horizon)

- **Input**: `dataset/manet_raw_dataset.csv`
- **Script**: `scripts/add_failure_label.py`
- **Idea**: for each (runid, nodeid, time) row, look **5 timesteps ahead** (`PREDICT_HORIZON = 5`) and mark whether a “failure-like event” is about to happen.

**Label definition (as implemented)**:

- `link_failure = 1` if any of the following holds:
  - **Neighbor drop**: `neighbor_count - future_neighbor_count >= 2`
  - **RSSI drop**: `avg_rssi - future_avg_rssi >= 15` (and future RSSI is not the sentinel)
  - **Becomes isolated**: `future_avg_rssi == -1000`
  - **Already isolated**: `avg_rssi == -1000`
- **Output**: `dataset/manet_dataset.csv` (same rows + `link_failure`)

### 3) Feature engineering (static + temporal history)

- **Input**: `dataset/manet_dataset.csv`
- **Script**: `scripts/feature_engineering.py`
- **Output**: `dataset/manet_featured_dataset.csv`

Features are built in three layers:

- **Core geometric/velocity features**
  - `dist_to_center = sqrt((x-250)^2 + (y-250)^2)`
  - `rssi_velocity = diff(avg_rssi)` per `(run_id, node_id)`
  - `neighbor_velocity = diff(neighbor_count)` per `(run_id, node_id)`
- **Network/flow-derived features**
  - `pdr = rx_packets / tx_packets` (falls back to 1.0 if `tx_packets==0`)
  - `log_delay = log1p(delay_sum)`
- **Rolling history (using past-only windows)**
  - `rssi_trend_3`: mean of last 3 `rssi_velocity` values (shifted by 1)
  - `neighbor_trend_3`: mean of last 3 `neighbor_velocity` values (shifted by 1)
  - `rssi_std_5`: std of last 5 `avg_rssi` values (shifted by 1)
  - `neighbor_std_5`: std of last 5 `neighbor_count` values (shifted by 1)

### 4) Model training (feature scaling + 2-model ensemble)

- **Script**: `experiments/training.py`
- **Dataset**: `dataset/manet_featured_dataset.csv`
- **Split**: test runs are `run_id ∈ {25,26,27,28,29,30}`.

**Internal model working**

- **Input vector X**: the feature columns listed in `experiments/training.py` (`FEATURES` array).
- **Scaling**:
  - `StandardScaler` is fit on **train** rows only.
  - Then both train/test are transformed using the same scaler.
  - The scaler is saved as `models/scaler.pkl` and is required at inference time.
- **Learners**:
  - Random Forest classifier → `models/random_forest.pkl`
  - XGBoost classifier → `models/xgboost_model.pkl`
- **Ensemble logic**:
  - Each model outputs `P(link_failure=1 | X)`.
  - A weighted sum produces the ensemble failure probability:

  p_{fail} = w_{rf}p_{rf} + w_{xgb}p_{xgb}

  - Then **reliability** is:

  r = 1 - p_{fail}

  - Weights are stored in `models/ensemble_weights.pkl`.

### 5) Inference: node snapshot → link reliabilities

Routing needs **edge** reliabilities, but the dataset is naturally **per-node** per time step. The project converts node features into link features as follows:

- **Snapshot**: filter dataset to a single `(run_id, time)` slice.
- **Connectivity**: connect nodes u,v if their Euclidean distance is within a radius threshold (e.g. `radius=65` or `radius=250` depending on the script).
- **Edge feature vector**: compute
  - `edge_features(u,v) = (node_features(u) + node_features(v)) / 2`
  - (implemented in `src/routing_from_dataset.py` inside `DatasetRouter.build_graph`)
- **Predict** `r(u,v)` using `src/predict.py`:
  - Loads `random_forest.pkl`, `xgboost_model.pkl`, `scaler.pkl`, `ensemble_weights.pkl`
  - Returns both `reliability` and `failure_prob`

### 6) Routing: reliability-weighted Dijkstra

To make Dijkstra prefer reliable links, per-edge reliability is converted to an additive cost:

- **Reliability**: r(u,v)\in(0,1)
- **Weight**:

w(u,v) = -\log(r(u,v))

- **ML route**: shortest path minimizing \sum w(u,v) (prefers higher reliability).
- **Baseline route**: shortest path by hop count (unweighted).

Implementation:

- Graph building + pathfinding: `src/routing_from_dataset.py`
- A standalone runnable demo: `src/routing_dijkstra.py`

### 7) Evaluation + outputs (what gets saved where)

- **Routing evaluation**: `experiments/evaluate_routing.py`
  - **Saves**:
    - `assets/routing_evaluation.png` (summary plot)
    - `dataset/routing_results.csv` (raw per-decision results)
- **Topology animation**: `src/routing_animation.py`
  - **Shows** a live animated figure
  - **Saves**: `assets/reliability_vs_time.jpg` (reliability vs time plot)
- **Training artifacts**: `experiments/training.py`
  - **Saves**:
    - `models/random_forest.pkl`
    - `models/xgboost_model.pkl`
    - `models/scaler.pkl`
    - `models/ensemble_weights.pkl`

---

## Dataset


| Property             | Value                     |
| -------------------- | ------------------------- |
| Simulation runs      | 30                        |
| Nodes per run        | 30                        |
| Timesteps per run    | 60                        |
| Total samples        | ~54,000                   |
| Mobility model       | Random Waypoint           |
| Simulation area      | 500 × 500 m               |
| Communication radius | 250 m                     |
| Simulator            | NS-3 (IEEE 802.11 ad hoc) |


### Features


| Feature             | Type       | Description                                              |
| ------------------- | ---------- | -------------------------------------------------------- |
| `neighbor_count`    | Original   | Number of nodes currently in range                       |
| `x`, `y`            | Original   | Node position in simulation area                         |
| `time`              | Original   | Simulation timestep                                      |
| `avg_rssi`          | Original   | Average received signal strength (dBm); −1000 = isolated |
| `tx_packets`        | Original   | Total transmitted packets in the run (FlowMonitor)       |
| `rx_packets`        | Original   | Total received packets in the run (FlowMonitor)          |
| `lost_packets`      | Original   | Total lost packets in the run (FlowMonitor)              |
| `delay_sum`         | Original   | Total delay sum in ns (FlowMonitor)                      |
| `dist_to_center`    | Engineered | Euclidean distance to area centre (500, 500)             |
| `rssi_velocity`     | Engineered | Change in RSSI from previous timestep                    |
| `neighbor_velocity` | Engineered | Change in neighbor count from previous timestep          |
| `pdr`               | Engineered | Packet Delivery Ratio = rxpackets / txpackets            |
| `log_delay`         | Engineered | log1p(delaysum) — compressed total packet delay          |
| `rssi_trend_3`      | Engineered | Rolling mean of past 3 `rssi_velocity` (shifted)         |
| `neighbor_trend_3`  | Engineered | Rolling mean of past 3 `neighbor_velocity` (shifted)     |
| `rssi_std_5`        | Engineered | Rolling std of past 5 `avg_rssi` (shifted)               |
| `neighbor_std_5`    | Engineered | Rolling std of past 5 `neighbor_count` (shifted)         |


### Label

`link_failure` — binary (0 = stable, 1 = failing).
Derived from a temporal look-ahead: a node is labelled 1 if, within the next 5 timesteps, its neighbor count drops by ≥ 2, its RSSI drops by ≥ 15 dBm (when future RSSI exists), it becomes isolated (future RSSI sentinel −1000), or it is currently isolated.

---

## Results

### ML Model Performance


| Model         | Test AUC                                    |
| ------------- | ------------------------------------------- |
| Random Forest | printed by `experiments/training.py`        |
| XGBoost       | printed by `experiments/training.py`        |
| **Ensemble**  | weighted combination used at inference time |


> Run `experiments/training.py` to reproduce. AUCs are printed to console. Models and scaler are saved into `models/`.

### Routing Performance


| Metric                | Baseline (hop-count) | ML Routing | Improvement |
| --------------------- | -------------------- | ---------- | ----------- |
| Avg Route Reliability | 0.6089               | 0.7170     | +17.8%      |
| Min Link Reliability  | 0.5967               | 0.6912     | +15.8%      |
| Avg Hop Count         | 1.2156               | 1.6344     | +0.42       |


> Run `experiments/evaluate_routing.py` to reproduce. Results printed to console and saved as `assets/routing_evaluation.png` (and raw results in `dataset/routing_results.csv`).

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
│   ├── build_dataset.py         Merges position + flow data
│   ├── add_failure_label.py     Temporal ground-truth labelling
│   └── feature_engineering.py  Derives 5 engineered features
│
├── dataset/
│   ├── manet_raw_dataset.csv    Output of build_dataset.py
│   ├── manet_dataset.csv        After add_failure_label.py
│   └── manet_featured_dataset.csv  After feature_engineering.py
│   └── routing_results.csv       Raw routing eval records (written by experiments)
│
├── experiments/
│   ├── training.py              Training pipeline (RF + XGB ensemble)
│   └── evaluate_routing.py      Systematic routing evaluation
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

# Step 3 — Train models (RF + XGB ensemble)
python experiments/training.py

# Step 4 — Evaluate routing
python experiments/evaluate_routing.py

# Step 5 — Visualise a single snapshot
python src/routing_from_dataset.py

# Step 6 — Run the animation
python src/routing_animation.py
```

### Re-run NS-3 simulations (optional)

If you have NS-3 installed:

```bash
./scripts/run_simulations.sh
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