"""Stage 3: refuse to train on a dataset that is quietly broken.

Two model features were constant across every row for weeks and nothing failed.
A constant feature is a bug in an upstream stage, so this is a hard gate rather
than a printout.

Checks: declared columns exist, no feature is constant or all null, traffic
features are causal (or the caller opts in), the label rate is plausible, the
identifier grid is rectangular, and the split is non empty.
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
from splits import make_run_split, split_frame  # noqa: E402

CFG = get_config()
DEFAULT_INPUT = ROOT / "data" / "processed" / "paper_featured_dataset.csv"
REPORT_FILE = ROOT / "results" / "data_quality_report.json"

MIN_LABEL_RATE = 0.01
MAX_LABEL_RATE = 0.90


class DataQualityError(AssertionError):
    """The dataset failed a hard check."""


def validate(
    df: pd.DataFrame,
    allow_run_level_traffic: bool = False,
    traffic_is_causal: bool = False,
    seed: int | None = None,
    test_run_count: int | None = None,
    val_run_count: int | None = None,
) -> dict:
    problems: list[str] = []
    warnings: list[str] = []

    required = IDENTIFIER_COLS + FEATURES + FACTOR_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataQualityError(f"dataset is missing required columns: {missing}")

    dead_features = constant_columns(df, FEATURES)
    dead_factors = constant_columns(df, [c for c in FACTOR_COLS if c != "T_hello"])

    informative_dead = [c for c in dead_features if c not in TRAFFIC_FEATURES]
    if informative_dead:
        problems.append(f"model features carry no information: {informative_dead}")
    traffic_dead = [c for c in dead_features if c in TRAFFIC_FEATURES]
    if traffic_dead:
        problems.append(
            f"traffic features are constant: {traffic_dead}. The FlowMonitor parse "
            "produced nothing; see dataset_manifest.json -> flowmonitor_diagnostics."
        )
    if dead_factors:
        problems.append(f"paper factors carry no information: {dead_factors}")

    if not traffic_is_causal:
        message = (
            "traffic features (pdr, log_delay) come from end of run FlowMonitor "
            "aggregates, so their value at time t depends on packets sent after t. "
            "Enable per second flow statistics in the simulation, or pass "
            "--allow-run-level-traffic to acknowledge the limitation."
        )
        (warnings if allow_run_level_traffic else problems).append(message)

    labelled = add_link_failure_labels(df)
    label_rate = float(labelled["link_failure"].mean())
    if not (MIN_LABEL_RATE <= label_rate <= MAX_LABEL_RATE):
        problems.append(
            f"label rate {label_rate:.4f} outside [{MIN_LABEL_RATE}, {MAX_LABEL_RATE}]"
        )

    per_run = labelled.groupby("run_id").agg(
        nodes=("node_id", "nunique"), times=("time", "nunique"), rows=("node_id", "size")
    )
    ragged = per_run[per_run["rows"] != per_run["nodes"] * per_run["times"]]
    if len(ragged):
        warnings.append(f"{len(ragged)} run(s) do not have a full node x time grid")

    seed = CFG.random_seed if seed is None else seed
    test_run_count = CFG.test_run_count if test_run_count is None else test_run_count
    val_run_count = CFG.val_run_count if val_run_count is None else val_run_count
    run_ids = sorted(df["run_id"].unique().astype(int).tolist())
    split = make_run_split(
        run_ids, seed=seed, test_run_count=test_run_count, val_run_count=val_run_count
    )
    train, val, test = split_frame(labelled, split)
    if len(train) == 0 or len(test) == 0:
        problems.append(f"split produced an empty side: {split.describe()}")

    return {
        "rows": int(len(df)),
        "runs": run_ids,
        "features": FEATURES,
        "paper_factors": FACTOR_COLS,
        "label_rate": label_rate,
        "label_rate_by_run": {
            str(k): float(v)
            for k, v in labelled.groupby("run_id")["link_failure"].mean().items()
        },
        "constant_model_features": dead_features,
        "constant_paper_factors": dead_factors,
        "traffic_is_causal": bool(traffic_is_causal),
        "split": split.as_dict(),
        "rows_train": int(len(train)),
        "rows_val": int(len(val)),
        "rows_test": int(len(test)),
        "feature_summary": {
            column: {
                "mean": float(np.nanmean(df[column].to_numpy(dtype=float))),
                "std": float(np.nanstd(df[column].to_numpy(dtype=float))),
                "min": float(np.nanmin(df[column].to_numpy(dtype=float))),
                "max": float(np.nanmax(df[column].to_numpy(dtype=float))),
                "n_unique": int(df[column].nunique(dropna=True)),
            }
            for column in FEATURES
        },
        "warnings": warnings,
        "problems": problems,
        "passed": not problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail loudly on a quietly broken dataset.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--allow-run-level-traffic", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="record that this ran on the smoke subset")
    parser.add_argument("--test-run-count", type=int, default=CFG.test_run_count)
    parser.add_argument("--val-run-count", type=int, default=CFG.val_run_count)
    parser.add_argument("--seed", type=int, default=CFG.random_seed)
    parser.add_argument("--report", default=str(REPORT_FILE))
    args = parser.parse_args()

    manifest_path = Path(args.input).parent / "dataset_manifest.json"
    traffic_is_causal = False
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as handle:
            traffic_is_causal = bool(json.load(handle).get("traffic_is_causal", False))

    report = {
        "smoke": bool(args.smoke),
        **validate(
            pd.read_csv(args.input),
            allow_run_level_traffic=args.allow_run_level_traffic,
            traffic_is_causal=traffic_is_causal,
            seed=args.seed,
            test_run_count=args.test_run_count,
            val_run_count=args.val_run_count,
        ),
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"[validate_dataset] rows={report['rows']} runs={len(report['runs'])} "
          f"label_rate={report['label_rate']:.4f}")
    for warning in report["warnings"]:
        print(f"[validate_dataset] WARNING: {warning}")
    for problem in report["problems"]:
        print(f"[validate_dataset] FAIL: {problem}")
    print(f"[validate_dataset] report written to {args.report}")

    if not report["passed"]:
        raise DataQualityError(f"{len(report['problems'])} problem(s); see {args.report}")
    print("[validate_dataset] PASSED")


if __name__ == "__main__":
    main()
