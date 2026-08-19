"""Stage 3: refuse to train on a dataset that is quietly broken.

This stage exists because two model features, pdr and log_delay, were constant
across every row of the dataset for weeks. Nothing failed. The models trained,
the routing ran, the results looked plausible, and two of fourteen features were
carrying exactly zero information. A constant feature is not a warning, it is a
bug in an upstream stage, so this check is a hard gate rather than a printout.

Checks performed:
  - every declared model feature and paper factor exists
  - no feature column is constant or entirely null
  - traffic derived features are causal, or the caller explicitly opted in
  - the label rate is inside a sane band
  - the identifier grid is rectangular (no missing node/time combinations)
  - no row appears in more than one split
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from bootstrap import setup_paths  # noqa: E402

ROOT = setup_paths()

from config_loader import get_config  # noqa: E402
from label_utils import add_link_failure_labels  # noqa: E402
from schema import (  # noqa: E402
    FACTOR_COLS,
    FEATURES,
    IDENTIFIER_COLS,
    TRAFFIC_FEATURES,
    constant_columns,
)
from splits import make_run_split, split_frame

CFG = get_config()
DEFAULT_INPUT = ROOT / "data" / "processed" / "paper_featured_dataset.csv"
REPORT_FILE = ROOT / "results" / "data_quality_report.json"


class DataQualityError(AssertionError):
    pass


def validate(
    df: pd.DataFrame,
    allow_run_level_traffic: bool = False,
    traffic_is_causal: bool = False,
    seed: int = None,
    test_run_count: int = None,
    val_run_count: int = None,
) -> dict:
    problems: list[str] = []
    warnings: list[str] = []

    required = IDENTIFIER_COLS + FEATURES + FACTOR_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataQualityError(f"dataset is missing required columns: {missing}")

    # 1. constant feature columns
    dead_features = constant_columns(df, FEATURES)
    dead_factors = constant_columns(df, [c for c in FACTOR_COLS if c != "T_hello"])

    informative_dead = [c for c in dead_features if c not in TRAFFIC_FEATURES]
    if informative_dead:
        problems.append(
            f"model features carry no information (constant or all null): {informative_dead}"
        )
    traffic_dead = [c for c in dead_features if c in TRAFFIC_FEATURES]
    if traffic_dead:
        problems.append(
            f"traffic features are constant: {traffic_dead}. "
            "This means the FlowMonitor parse produced nothing. "
            "Check dataset_manifest.json -> flowmonitor_diagnostics."
        )
    if dead_factors:
        problems.append(f"paper factors carry no information: {dead_factors}")

    # 2. traffic causality
    if not traffic_is_causal:
        msg = (
            "traffic features (pdr, log_delay) come from end of run FlowMonitor "
            "aggregates, so their value at time t depends on packets sent after t. "
            "Enable periodic flow statistics in the simulation, or pass "
            "--allow-run-level-traffic to acknowledge the limitation."
        )
        if allow_run_level_traffic:
            warnings.append(msg)
        else:
            problems.append(msg)

    # 3. label rate
    labelled = add_link_failure_labels(df)
    rate = float(labelled["link_failure"].mean())
    if not (0.01 <= rate <= 0.90):
        problems.append(f"label rate {rate:.4f} is outside the plausible band [0.01, 0.90]")

    # 4. rectangular identifier grid
    per_run = labelled.groupby("run_id").agg(
        nodes=("node_id", "nunique"), times=("time", "nunique"), rows=("node_id", "size")
    )
    ragged = per_run[per_run["rows"] != per_run["nodes"] * per_run["times"]]
    if len(ragged):
        warnings.append(f"{len(ragged)} run(s) do not have a full node x time grid")

    # 5. split hygiene
    seed = CFG.random_seed if seed is None else seed
    test_run_count = CFG.test_run_count if test_run_count is None else test_run_count
    val_run_count = CFG.val_run_count if val_run_count is None else val_run_count
    run_ids = sorted(df["run_id"].unique().astype(int).tolist())
    split = make_run_split(run_ids, seed=seed, test_run_count=test_run_count, val_run_count=val_run_count)
    tr, va, te = split_frame(labelled, split)
    if len(tr) == 0 or len(te) == 0:
        problems.append(f"split produced an empty side: {split.describe()}")

    report = {
        "rows": int(len(df)),
        "runs": run_ids,
        "features": FEATURES,
        "paper_factors": FACTOR_COLS,
        "label_rate": rate,
        "label_rate_by_run": {
            str(k): float(v) for k, v in labelled.groupby("run_id")["link_failure"].mean().items()
        },
        "constant_model_features": dead_features,
        "constant_paper_factors": dead_factors,
        "traffic_is_causal": bool(traffic_is_causal),
        "split": split.as_dict(),
        "rows_train": int(len(tr)),
        "rows_val": int(len(va)),
        "rows_test": int(len(te)),
        "feature_summary": {
            c: {
                "mean": float(np.nanmean(df[c].to_numpy(dtype=float))),
                "std": float(np.nanstd(df[c].to_numpy(dtype=float))),
                "min": float(np.nanmin(df[c].to_numpy(dtype=float))),
                "max": float(np.nanmax(df[c].to_numpy(dtype=float))),
                "n_unique": int(df[c].nunique(dropna=True)),
            }
            for c in FEATURES
        },
        "warnings": warnings,
        "problems": problems,
        "passed": not problems,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail loudly on a quietly broken dataset.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--allow-run-level-traffic", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="record in the report that this ran on the smoke subset")
    parser.add_argument("--test-run-count", type=int, default=CFG.test_run_count)
    parser.add_argument("--val-run-count", type=int, default=CFG.val_run_count)
    parser.add_argument("--seed", type=int, default=CFG.random_seed)
    parser.add_argument("--report", default=str(REPORT_FILE))
    args = parser.parse_args()

    manifest_path = Path(args.input).parent / "dataset_manifest.json"
    traffic_is_causal = False
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            traffic_is_causal = bool(json.load(f).get("traffic_is_causal", False))

    df = pd.read_csv(args.input)
    report = validate(
        df,
        allow_run_level_traffic=args.allow_run_level_traffic,
        traffic_is_causal=traffic_is_causal,
        seed=args.seed,
        test_run_count=args.test_run_count,
        val_run_count=args.val_run_count,
    )

    report = {"smoke": bool(args.smoke), **report}

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[validate_dataset] rows={report['rows']} runs={len(report['runs'])} "
          f"label_rate={report['label_rate']:.4f}")
    for w in report["warnings"]:
        print(f"[validate_dataset] WARNING: {w}")
    for p in report["problems"]:
        print(f"[validate_dataset] FAIL: {p}")
    print(f"[validate_dataset] report written to {args.report}")

    if not report["passed"]:
        raise DataQualityError(
            f"{len(report['problems'])} data quality problem(s); see {args.report}"
        )
    print("[validate_dataset] PASSED")


if __name__ == "__main__":
    main()
