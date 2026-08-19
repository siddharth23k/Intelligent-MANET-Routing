# Intelligent MANET Routing

Machine learning for link failure prediction in mobile ad hoc networks, and what
that prediction is actually worth once it drives a router.

Three methods, one shared NS-3 dataset, one shared label definition, one shared
run level split:

| Method | Predictor | Router |
|---|---|---|
| **Paper baseline (FRLFP)** | SFRNNR, a fuzzy recurrent network with an adaptive threshold head, reimplemented from the published description | exclude flagged nodes, then route by hop count |
| **Ours** | Random Forest + XGBoost ensemble over 14 features | reliability weighted Dijkstra |
| **Classic baseline** | none | hop count shortest path |

---

## Quick start

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

make smoke     # end to end on a tiny subset, about 20 seconds
make test      # unit tests, about 2 seconds
```

`make smoke` runs all six stages on two simulation runs with a dozen trees and a
couple of epochs, and asserts each stage produced what the next one needs.
**Numbers from a smoke run are a liveness signal, not results.**

For real numbers you need NS-3 output in `data/raw`:

```bash
NS3_DIR=~/ns-3.47 bash simulation/run_simulation.sh
make all
```

---

## Pipeline

| Stage | Command | Produces |
|---|---|---|
| 1 | `pipeline/generate_data.py` | `paper_raw_dataset.csv`, `dataset_manifest.json` |
| 2 | `pipeline/engineer_features.py` | `paper_featured_dataset.csv`, `feature_norm_stats.json`, `run_split.json` |
| 3 | `pipeline/validate_dataset.py` | `results/data_quality_report.json` (hard gate) |
| 4 | `pipeline/train_predictor.py` | RF and XGBoost artifacts, `predictor_schema.json`, `predictor_metrics.json` |
| 5 | `pipeline/train_models.py` | `sfrnnr_paper.keras`, `sfrnnr_meta.json`, `paper_lfp_dataset.csv` |
| 6 | `pipeline/compare_methods.py` | `path_survival.csv`, `comparison_metrics.csv`, `routing_decisions.csv`, `comparison_summary.md` |

Every stage accepts `--smoke` for the fast path and `--test-run-count` /
`--val-run-count` to control the split. `pipeline/smoke_test.py --stages 4 5`
runs a subset, which is the quickest way to isolate a failing stage.

---

## Layout

```
config/          paper_scenarios.yaml, the single source of shared parameters
simulation/      NS-3 scenario and the run script
pipeline/        the six numbered stages, plus smoke_test.py
methods/
  common/        schema, run level splitting, normalisation statistics
  ours/          RF + XGBoost predictor, graph construction, Dijkstra routers
  baseline/      SFRNNR model and inference, adaptive threshold, FRLFP router
  eval/          path survival metric, route metrics, statistics
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
| Duration | 300 s, sampled every second |
| Runs | 30, each on its own NS-3 RNG substream |

30 x 100 x 300 = 900,000 node time rows.

---

## Method

**Label.** At time `t`, a node is a link failure if five seconds later it has
lost two or more neighbours, or its mean neighbour RSSI has fallen 15 dB, or it
has become isolated. Roughly 28 percent positive. Both methods train on exactly
this definition.

**Features.** Position and distance from the area centre, neighbour count, mean
neighbour RSSI, traffic derived delivery ratio and delay, and lagged temporal
derivatives: velocities, three step trends, five step rolling standard
deviations. Every rolling window is `.shift(1)` first, so a row at `t` is built
only from `t-1` and earlier. Features look backwards, the label looks forwards,
and the two windows meet at `t` without crossing it.

**Split.** By simulation run, never by row. Consecutive rows of one node track
share four of five rolling window elements and overlap in label horizon, so a
random row split would put near duplicates on both sides. Test runs are drawn
first and validation runs from what remains, so changing the validation size
cannot move a run into or out of the test set. `methods/common/splits.py` is
imported by both the training and the evaluation code, so the held out runs are
provably the same set rather than two scripts that happen to share a seed.

**Routing.** Node reliability is `1 - P(failure)`. Edge reliability is the weaker
endpoint. Edge weight is `-log(reliability)`, so summing weights along a path is
`-log` of the product of link reliabilities and Dijkstra's minimum weight path is
the maximum reliability path. Weights are strictly positive by construction,
which Dijkstra requires.

---

## Evaluation

Two metrics, and the difference between them matters more than either.

**Path survival, the ground truth.** Replay the mobility trace forward: a route
chosen at `t` survives if every hop is still within radius at `t + 5`. No model
output enters this number, so all three methods are scored on something none of
them can influence. Written to `results/path_survival.csv` and reported first.

**Model derived route quality.** Mean and minimum edge reliability along the
chosen path, which is what the base paper reports. Read it carefully: our router
minimises the sum of `-log` of the same reliabilities averaged here, so the
optimiser and the scorer are the same function. A win rate near 1.0 with a loss
rate of 0.0 is a property of that algebra, not evidence of predictive skill.
`compare_methods.py` prints win, tie and loss rates next to the deltas so this
stays visible.

Significance is tested at the **run** level. Decisions inside one run share
almost all of their topology, so thousands of decisions from a handful of
simulations carry roughly a handful of independent units of evidence. Wilcoxon
is reported next to the paired t test because normality cannot be checked at
that sample size.

`compare_methods.py` also reports how often FRLFP fell back to the unfiltered
graph. **A high fallback rate means the paper baseline has degenerated into hop
count shortest path and any comparison against it is meaningless.** That used to
be invisible because the fallback lived inside a bare `except`.

---

## Known limitations

Stated here rather than left for the next reader to find.

- **Traffic features are not causal unless the simulation is rerun.** FlowMonitor
  serialises end of run aggregates, so using them at `t` is lookahead. The
  simulation now writes per second deltas under `--logFlowStats`, which makes
  them causal. Datasets built from older raw output are rejected by stage 3
  unless `--allow-run-level-traffic` is passed.
- **The label is coupled to a feature.** `neighbor_count(t) - neighbor_count(t+5)`
  means a dense node is mechanically more likely to be labelled a failure, and an
  isolated node can never satisfy that condition. No split fixes this; only a
  link level persistence label does. `label_diagnostics` quantifies the coupling
  into `results/predictor_metrics.json`.
- **Connectivity features are geometric, not measured.** `neighbor_count` is a
  distance threshold and `avg_rssi` is log distance path loss plus Gaussian
  shadowing, both computed in a scheduled callback rather than read from the PHY.
  The wireless stack runs; it does not reach those two columns.
- **Node level prediction, edge level need.** One row per node per second, not
  per link, so edge reliability is approximated from node reliabilities. A true
  per link model needs the simulation to log neighbour lists, not counts.
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
- the feature list and column order match between the code and the artifacts
- the two ensemble members are different classes **and** actually disagree on
  real inputs, so a duplicated artifact cannot pass
- predicted probabilities vary across the scaler's own input distribution, which
  catches a scaler fitted on different data than the models
- edge weights are strictly positive, and `sum(-log r) == -log(prod r)`
- the grid based neighbour search agrees with brute force
- every declared input to the adaptive threshold moves it, and the old `_norm`
  suffixed keys are rejected rather than silently defaulted
- path survival is correct on hand built traces, inclusive at the radius
  boundary, and excludes decisions with no future snapshot
- normalisation statistics come only from the rows they were fitted on

The predictor artifact tests skip when nothing has been trained yet, so run
`make smoke` before `make test` if you want the full suite.

---

## Troubleshooting

**Stage 5 stalls.** Nothing in the SFRNNR path calls `model.fit` or
`model.predict`. Both build a `tf.data` pipeline even for a plain numpy array,
and that adapter is the component that deadlocks on some TensorFlow builds.
Training goes through `train_on_batch` and every forward pass calls the model
directly, so there is no adapter and no predict function anywhere. The loop also
feeds one fixed batch shape, so graph mode traces the train step once instead of
retracing on every ragged batch. `model.fit` is still reachable with
`--fit-backend keras` for comparison, and `--run-eagerly` / `--no-run-eagerly`
force the execution mode.

**Any stage hangs.** Output streams live and stage 5 prints one line per epoch,
so the last printed line tells you where. Run a single stage with
`python pipeline/smoke_test.py --stages 5`, raise the limit with
`MANET_SMOKE_STAGE_TIMEOUT=1200`, and run `make diagnose` to time each
TensorFlow component individually in both eager and graph mode. The probe that
never prints its time is the one that hangs. `make diagnose` also probes the
`fit` paths the pipeline no longer uses, so a stall there is informative rather
than blocking.

**Stage 3 fails on traffic features.** Either rerun the simulation with
`--logFlowStats` (the default) so the per second counters exist, or pass
`--allow-run-level-traffic` to acknowledge the limitation.

**`ArtifactError` on load.** The saved models and the code have drifted. Retrain
with `python pipeline/train_predictor.py`.

---

## License

MIT. See `LICENSE`.
