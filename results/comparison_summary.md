# Comparison Summary

| method | n_samples | avg_reliability | min_reliability | avg_hops | baseline_avg_reliability | baseline_min_reliability | baseline_avg_hops | delta_avg_rel_pct | delta_min_rel_pct | delta_hops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ours | 9000 | 0.685323 | 0.606874 | 4.57689 | 0.626951 | 0.550344 | 4.20778 | 9.31047 | 10.2718 | 0.369111 |
| paper_baseline | 9000 | 0.339259 | 0.325557 | 4.20778 | 0.339259 | 0.325557 | 4.20778 | 0 | 0 | 0 |


## Notes
- `paper_baseline` is a paper-inspired FRLFP re-implementation.
- `ours` is the current ensemble + reliability-weighted routing method.
- `classic_baseline` output is available in `results/classic_baseline_results.csv`.
