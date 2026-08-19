# Comparison Summary

Held out runs: [1, 6]. Decisions: 16. Radius: 150.0 m. Survival horizon: 5 steps.

## 1. Ground truth: path survival

Fraction of chosen routes whose every hop is still within radius `survival_horizon` steps later, replayed from the mobility trace. No model output enters this number.

| method | n_evaluable | survival_rate | mean_surviving_fraction | mean_broken_hops |
| --- | --- | --- | --- | --- |
| ours | 16 | 0.0625 | 0.313975 | 3.1875 |
| paper_baseline | 16 | 0.0625 | 0.371503 | 2.9375 |
| classic_baseline | 16 | 0.0625 | 0.371503 | 2.9375 |


## 2. Model derived route quality (report with care)

This is the metric the base paper reports. Note that our router minimises the sum of -log of the same reliabilities that are averaged here, so the optimiser and the scorer are the same function. A win rate near 1.0 with a loss rate of 0.0 is a property of that algebra, not evidence of predictive skill. Read section 1 first.

| method | n_decisions | avg_reliability | min_reliability | avg_hops | baseline_avg_reliability | baseline_min_reliability | baseline_avg_hops | delta_avg_rel_pct | delta_min_rel_pct | delta_hops | wlt_n_decisions | wlt_win_rate | wlt_tie_rate | wlt_loss_rate | stat_n_runs | stat_mean_delta | stat_t_p_value | stat_wilcoxon_p_value | stat_cohens_d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ours | 16 | 0.448159 | 0.381723 | 4.625 | 0.423516 | 0.361039 | 4.4375 | 5.81873 | 5.72916 | 0.1875 | 16 | 0.75 | 0.25 | 0 | 2 | 0.0246432 | 0.139442 | 0.5 | 3.1765 |
| paper_baseline | 16 | 0.489417 | 0.455555 | 4.4375 | 0.423516 | 0.361039 | 4.4375 | 15.5605 | 26.1791 | 0 | 16 | 0.6875 | 0 | 0.3125 | 2 | 0.0659013 | 0.203021 | 0.5 | 2.14161 |


## 3. Baseline health

- FRLFP flagged on average 0.1% of nodes as risky.
- FRLFP fell back to the unfiltered graph on 0.0% of routing attempts. At a high fallback rate the paper baseline is indistinguishable from hop count shortest path and any comparison against it is meaningless.
- Our router failed to find a path: {'hop:no_path': 3, 'ml:no_path': 3}.
- Skipped snapshots or pairs: {'no_edges': 0, 'no_common_path': 3}.

## 4. Notes

- All three methods route over an identical geometric edge set and an identical set of sampled source destination pairs.
- Significance is tested at the run level, not the decision level: decisions inside one run share almost all of their topology.
- Per decision records are in `results/routing_decisions.csv`.
