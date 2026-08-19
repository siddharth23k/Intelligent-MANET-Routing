# Comparison Summary

Held out runs: [1, 6]. Decisions: 16. Radius: 150.0 m. Survival horizon: 5 steps.

## 1. Ground truth: path survival

Fraction of chosen routes whose every hop is still within radius `survival_horizon` steps later, replayed from the mobility trace. No model output enters this number.

| method | n_evaluable | survival_rate | mean_surviving_fraction | mean_broken_hops |
| --- | --- | --- | --- | --- |
| ours | 16 | 0.0625 | 0.310565 | 3.125 |
| paper_baseline | 16 | 0.0625 | 0.403274 | 2.8125 |
| classic_baseline | 16 | 0.0625 | 0.403274 | 2.8125 |


## 2. Model derived route quality (report with care)

This is the metric the base paper reports. Our router minimises the sum of -log of the same reliabilities averaged here, so the optimiser and the scorer are the same function. A win rate near 1.0 with a loss rate of 0.0 is a property of that algebra, not evidence of predictive skill. Read section 1 first.

| method | n_decisions | avg_reliability | min_reliability | avg_hops | baseline_avg_reliability | baseline_min_reliability | baseline_avg_hops | delta_avg_rel_pct | delta_min_rel_pct | delta_hops | wlt_n_decisions | wlt_win_rate | wlt_tie_rate | wlt_loss_rate | stat_n_runs | stat_mean_delta | stat_t_p_value | stat_wilcoxon_p_value | stat_cohens_d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ours | 16 | 0.466593 | 0.375103 | 4.5625 | 0.432908 | 0.359098 | 4.5 | 7.78095 | 4.45721 | 0.0625 | 16 | 0.875 | 0.125 | 0 | 2 | 0.0336844 | 0.0228763 | 0.5 | 19.6695 |
| paper_baseline | 16 | 0.619154 | 0.593766 | 4.5 | 0.432908 | 0.359098 | 4.5 | 43.0218 | 65.3494 | 0 | 16 | 0.875 | 0 | 0.125 | 2 | 0.186245 | 0.0536642 | 0.5 | 8.36854 |


## 3. Baseline health

- FRLFP flagged on average 0.0% of nodes as risky.
- FRLFP fell back to the unfiltered graph on 0.0% of routing attempts. At a high fallback rate it is indistinguishable from hop count shortest path and any comparison against it is meaningless.
- Our router failed to find a path: {'hop:no_path': 5, 'ml:no_path': 5}.
- Skipped snapshots or pairs: {'no_edges': 0, 'no_common_path': 5}.

## 4. Notes

- All three methods route over an identical edge set and identical sampled pairs.
- Significance is tested at the run level: decisions inside one run share almost all of their topology.
- Per decision records are in `results/routing_decisions.csv`.
