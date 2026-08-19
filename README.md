# Intelligent MANET Routing

Machine learning for link failure prediction in mobile ad hoc networks, and what
that prediction is worth when you plug it into a router.

Three methods are compared on one shared NS-3 dataset, one shared label
definition and one shared run level split:

1. **Paper baseline (FRLFP / SFRNNR)** — a reimplementation of the base paper's
   fuzzy recurrent link failure predictor with an adaptive threshold head, which
   excludes flagged nodes before routing by hop count.
2. **Ours (RF + XGBoost)** — a tree ensemble link failure predictor over 14
   features, feeding a reliability weighted Dijkstra router.
3. **Classic baseline** — plain hop count shortest path.

---

## Quick start

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

make smoke     # end to end on a tiny subset, finishes in seconds
make test      # unit tests
```

`make smoke` runs every stage of the pipeline on a couple of simulation runs
with a handful of trees and epochs, and asserts that each stage produced what the
next one needs. **Numbers from a smoke run are a liveness signal, not results.**

For real numbers you need the NS-3 output in `data/raw`, then:

```bash
bash simulation/run_simulation.sh    # needs ns-3; writes data/raw
make all                             # stages 1 to 6
```

---

## Pipeline

| Stage | Command | Produces |
|---|---|---|
| 1 | `pipeline/generate_data.py` | `paper_raw_dataset.csv`, `dataset_manifest.json` |
| 2 | `pipeline/engineer_features.py` | `paper_featured_dataset.csv`, `feature_norm_stats.json`, `run_split.json` |
| 3 | `pipeline/validate_dataset.py` | `results/data_quality_report.json` (hard gate) |
| 4 | `pipeline/train_predictor.py` | RF + XGBoost artifacts, `predictor_schema.json`, `predictor_metrics.json` |
| 5 | `pipeline/train_sfrnnr_paper.py` then `pipeline/train_models.py` | `sfrnnr_paper.keras`, `paper_lfp_dataset.csv` |
| 6 | `pipeline/compare_methods.py` | `path_survival.csv`, `comparison_metrics.csv`, `routing_decisions.csv`, `comparison_summary.md` |

Every stage takes `--smoke` for the fast path, and every stage that splits data
takes `--test-run-count` and `--val-run-count`.

---

## Layout

```
config/          paper_scenarios.yaml is the single source of shared parameters
simulation/      NS-3 scenario and the runner script
pipeline/        the six numbered stages, plus smoke_test.py
methods/
  common/        schema, run level splitting, normalisation statistics
  ours/          RF + XGBoost predictor, graph construction, Dijkstra routers
  baseline/      SFRNNR model, inference, adaptive threshold, FRLFP router
  eval/          path survival ground truth metric, route metrics, statistics
tests/           pytest suite
results/         metrics, reports, model artifacts
```

---

## Scenario

Transcribed from the base paper's table into `config/paper_scenarios.yaml`.

| | |
|---|---|
| Simulator | NS-3, 802.11b ad hoc, AODV |
| Nodes | 100 |
| Area | 1000 x 1000 m |
| Mobility | Random waypoint, 0 to 60 m/s, 2 s pause |
| Communication radius | 150 m |
| Traffic | 10 CBR UDP flows, 512 byte packets, 1 s interval |
| Duration | 300 s, sampled every 1 s |
| Runs | 30 independent NS-3 RNG substreams |

That gives 30 x 100 x 300 = 900,000 node time rows.

---

## Method

**Label.** At time `t`, a node is a link failure if, five seconds later, it has
lost two or more neighbours, or its mean neighbour RSSI has fallen by 15 dB, or
it has become isolated. Roughly 28 percent positive. Both methods train on this
exact definition.

**Features.** Position and distance from the area centre, neighbour count, mean
neighbour RSSI, traffic derived delivery ratio and delay, and lagged temporal
derivatives: velocities, three step trends and five step rolling standard
deviations. Every rolling window is `.shift(1)` first, so a row at time `t` is
built only from `t-1` and earlier. Features look backwards, the label looks
forwards, and the two windows meet at `t` without crossing it.

**Split.** By simulation run, never by row. Consecutive rows of one node track
share four of five rolling window elements and overlap in label horizon, so a
random row split would put near duplicates on both sides. Test runs are drawn
first, then validation runs from what remains, so changing the validation size
cannot move a run into or out of the test set. `methods/common/splits.py` is
imported by both the training and the evaluation code, so the held out runs are
provably the same set rather than two scripts that happen to share a seed.

**Routing.** Node reliability is `1 - P(failure)`. Edge reliability is the weaker
endpoint. Edge weight is `-log(reliability)`, so summing weights along a path is
`-log` of the product of link reliabilities, and Dijkstra's minimum weight path
is exactly the maximum reliability path. Weights are strictly positive by
construction, which Dijkstra requires.

---

## Evaluation

Two metrics, and the difference between them matters more than either.

**Path survival (ground truth).** Replay the mobility trace forward. A route
chosen at time `t` survives if every one of its hops is still within radius at
`t + 5`. No model output enters this number, so all three methods are scored on
something none of them can influence. Reported in `results/path_survival.csv`.

**Model derived route quality.** Mean and minimum edge reliability along the
chosen path, which is what the base paper reports. Read this one carefully: our
router minimises the sum of `-log` of the same reliabilities that are averaged
here, so the optimiser and the scorer are the same function. A win rate near
1.0 with a loss rate of 0.0 is a property of that algebra, not evidence of
predictive skill. `compare_methods.py` prints the win, tie and loss rates
alongside the deltas precisely so that this is visible.

Significance is tested at the **run** level. Routing decisions inside one run
share almost all of their topology, so thousands of decisions from a handful of
simulations carry roughly a handful of independent units of evidence. Wilcoxon
is reported next to the paired t test because at that sample size the t test's
normality assumption cannot be checked.

`compare_methods.py` also reports how often FRLFP fell back to the unfiltered
graph. **If that fallback rate is high, the paper baseline has degenerated into
hop count shortest path and any comparison against it is meaningless.** That
situation used to be invisible because the fallback lived inside a bare
`except`.

---

## Known limitations

Stated here rather than discovered by the next reader.

- **Traffic features are not causal unless the simulation is rerun.** FlowMonitor
  serialises end of run aggregates, so using them at time `t` is lookahead. The
  simulation now writes per second FlowMonitor deltas (`--logFlowStats`), which
  makes them causal. Datasets built from older raw output are rejected by stage 3
  unless you pass `--allow-run-level-traffic`.
- **The label is coupled to a feature.** `neighbor_count(t) - neighbor_count(t+5)`
  means a high degree node is mechanically more likely to be labelled a failure,
  and an isolated node can never satisfy that condition. No split fixes this;
  only a link level persistence label does. `label_diagnostics` in
  `methods/baseline/label_utils.py` quantifies the coupling and it is written into
  `results/predictor_metrics.json`.
- **Connectivity features are geometric, not measured.** `neighbor_count` is a
  distance threshold and `avg_rssi` is a closed form path loss plus Gaussian
  shadowing, both computed in a scheduled callback rather than read from the PHY.
  The wireless stack runs; it does not reach these two columns.
- **Node level prediction, edge level need.** There is one row per node per
  second, not one row per link, so edge reliability is approximated from node
  reliabilities. A true per link model needs the simulation to log neighbour
  lists rather than neighbour counts.
- **Centralised evaluation.** The router sees a global snapshot, which no node in
  a real MANET has. This measures the value of the prediction, not a deployable
  protocol.
- **Six held out runs.** Enough for a paired test, not enough for a confidence
  interval worth quoting. Grouped cross validation over runs would be better at
  the same cost.

---

## What the tests cover

`make test` checks the contracts that previously failed silently:

- train, validation and test runs are disjoint, and the split is stable
- the feature list and column order match between the code and the saved artifacts
- the two ensemble members are different model classes **and** actually disagree
  on real inputs, so a duplicated artifact cannot pass
- predicted probabilities vary across the scaler's own input distribution, which
  catches a scaler fitted on different data than the models
- edge weights are strictly positive, and `sum(-log r) == -log(prod r)`
- the grid based neighbour search agrees with brute force
- every declared input to the adaptive threshold actually moves it, and the old
  `_norm` suffixed keys are rejected rather than silently defaulted
- path survival is correct on hand built traces, is inclusive at the radius
  boundary, and excludes decisions with no future snapshot
- normalisation statistics come only from the rows they were fitted on

---

## License

MIT. See `LICENSE`.
