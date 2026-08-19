# results/

Everything here is generated. Nothing is edited by hand.

| File | Written by | Contents |
|---|---|---|
| `data_quality_report.json` | stage 3 | per column statistics, constant column check, label rate, split |
| `predictor_metrics.json` | stage 4 | AUC, average precision, precision, recall, F1, Brier, confusion matrix, feature importance, ensemble integrity check |
| `models/` | stages 4 and 5 | serialised models plus the provenance files recording which split and which normalisation statistics produced them |
| `routing_decisions.csv` | stage 6 | one row per routing decision, all three methods, including path survival |
| `path_survival.csv` | stage 6 | **the ground truth metric**: fraction of routes still intact `survival_horizon` steps later, replayed from the mobility trace |
| `path_survival_tests.csv` | stage 6 | significance tests on the survival rates |
| `comparison_metrics.csv` | stage 6 | model derived route quality plus win / tie / loss rates |
| `comparison_diagnostics.json` | stage 6 | split, FRLFP fallback rate, skipped decisions, routing failures |
| `comparison_summary.md` | stage 6 | human readable summary of the above |

## smoke/

Output of `make smoke`: two simulation runs, a dozen trees, a couple of epochs,
a handful of routing decisions. Committed so the output **format** is visible
without running anything.

**These are not results.** Every file carries `"smoke": true`. For numbers worth
quoting, run `make all` on the full thirty run dataset.

## Reading order

1. `path_survival.csv`. The only routing metric no model can influence.
2. `comparison_diagnostics.json`, specifically the FRLFP fallback rate. A high
   rate means the paper baseline degenerated into hop count shortest path and
   the comparison against it says nothing.
3. `comparison_metrics.csv`. This is the metric the base paper reports, but our
   router optimises the same reliabilities it averages, so check the win, tie
   and loss columns before drawing any conclusion from the deltas.
4. `predictor_metrics.json` for classification quality, and its
   `label_diagnostics` block for how much of the label is implied by a feature
   the model already sees.
