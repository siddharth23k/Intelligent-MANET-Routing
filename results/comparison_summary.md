# Comparison Summary


| method         | n_samples | avg_reliability | min_reliability | avg_hops | baseline_avg_reliability | baseline_min_reliability | baseline_avg_hops | delta_avg_rel_pct | delta_min_rel_pct | delta_hops |
| -------------- | --------- | --------------- | --------------- | -------- | ------------------------ | ------------------------ | ----------------- | ----------------- | ----------------- | ---------- |
| ours           | 1800      | 0.67924         | 0.640158        | 2.20889  | 0.633893                 | 0.607199                 | 2.06778           | 7.15367           | 5.42807           | 0.141111   |
| paper_baseline | 9000      | 0.339259        | 0.325557        | 4.20778  | 0.339259                 | 0.325557                 | 4.20778           | 0                 | 0                 | 0          |


## Notes

- `paper_baseline` is a paper-inspired FRLFP re-implementation.
- `ours` is the current ensemble + reliability-weighted routing method.
- `classic_baseline` output is available in `results/classic_baseline_results.csv`.

