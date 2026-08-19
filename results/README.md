# results/

Everything in this directory is generated. Nothing here is edited by hand.

| File | Written by | What it is |
|---|---|---|
| `data_quality_report.json` | `pipeline/validate_dataset.py` | per column statistics, constant column check, label rate, split |
| `predictor_metrics.json` | `pipeline/train_predictor.py` | AUC, average precision, precision, recall, F1, Brier, confusion matrix, feature importance, ensemble integrity check |
| `models/` | stages 4 and 5 | serialised models plus the provenance files that record which split and which normalisation statistics produced them |
| `routing_decisions.csv` | `pipeline/compare_methods.py` | one row per routing decision, all three methods, including path survival |
| `path_survival.csv` | `pipeline/compare_methods.py` | **the ground truth metric.** Fraction of chosen routes still intact `survival_horizon` steps later, replayed from the mobility trace |
| `path_survival_tests.csv` | `pipeline/compare_methods.py` | significance tests on the survival rates |
| `comparison_metrics.csv` | `pipeline/compare_methods.py` | model derived route quality plus win / tie / loss rates |
| `comparison_diagnostics.json` | `pipeline/compare_methods.py` | split, FRLFP fallback rate, skipped decisions, routing failures |
| `comparison_summary.md` | `pipeline/compare_methods.py` | human readable summary of the above |

## smoke/

Output of `make smoke`: two simulation runs, a dozen trees, three epochs, a
handful of routing decisions. It is committed so the output **format** is
visible without running anything.

**These are not results.** Every file in there carries `"smoke": true`. For
numbers worth quoting, run `make all` on the full thirty run dataset.

## Reading order

Start with `path_survival.csv`. That is the only routing metric no model can
influence. `comparison_metrics.csv` reports the metric the base paper uses, but
our router optimises the same reliabilities that metric averages, so read its
win / tie / loss columns before drawing any conclusion from its deltas.

Then check `comparison_diagnostics.json` for the FRLFP fallback rate. A high
fallback rate means the paper baseline degenerated into hop count shortest path
and the comparison against it says nothing.
