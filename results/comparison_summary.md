# Comparison Summary

| method | n_samples | avg_reliability | min_reliability | avg_hops | baseline_avg_reliability | baseline_min_reliability | baseline_avg_hops | delta_avg_rel_pct | delta_min_rel_pct | delta_hops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ours | 9000 | 0.821605 | 0.803122 | 4.24689 | 0.809236 | 0.789691 | 4.20778 | 1.52859 | 1.70088 | 0.0391111 |
| paper_baseline | 9000 | 0.339259 | 0.325557 | 4.20778 | 0.339259 | 0.325557 | 4.20778 | 0 | 0 | 0 |


## Notes
- `paper_baseline` is a paper-inspired FRLFP re-implementation.
- `ours` is the current ensemble + reliability-weighted routing method.
- `classic_baseline` output is available in `results/classic_baseline_results.csv`.
