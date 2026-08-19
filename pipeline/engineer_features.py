"""Stage 2: per node temporal features and the paper's nine link factors.

Two rules govern this file.

1. Features look backwards, labels look forwards. Every rolling or differencing
   operation is preceded by .shift(1) inside the (run_id, node_id) group, so a
   row at time t is built only from t-1 and earlier. The label looks at
   t + horizon. The two windows meet at t and never cross it.
2. Normalisation statistics are fitted on the training runs only. The split is
   deterministic, so this stage can compute it without seeing any labels.
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
from normalization import MinMaxStats  # noqa: E402
from schema import FACTOR_COLS, FEATURES, IDENTIFIER_COLS  # noqa: E402
from splits import make_run_split  # noqa: E402

CFG = get_config()

INPUT_FILE = ROOT / "data" / "processed" / "paper_raw_dataset.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "paper_featured_dataset.csv"
NORM_STATS_FILE = ROOT / "results" / "models" / "feature_norm_stats.json"
SPLIT_FILE = ROOT / "results" / "models" / "run_split.json"

# Factors whose min and max are fitted on training runs and reused everywhere.
NORMALISED_COLUMNS = ["RSSI", "LS", "LA", "LET", "LQ_mean", "LL_d", "ND", "d_res"]

OUT_COLS = IDENTIFIER_COLS + [
    "x", "y", "neighbor_count", "avg_rssi", "is_isolated",
    "tx_packets", "rx_packets", "lost_packets", "delay_sum",
    "dist_to_center", "rssi_velocity", "neighbor_velocity",
    "pdr", "log_delay",
    "rssi_trend_3", "neighbor_trend_3", "rssi_std_5", "neighbor_std_5",
] + FACTOR_COLS


def _lagged(series, window: int, how: str):
    """Backward looking rolling aggregate covering t-window .. t-1."""
    return getattr(series.shift(1).rolling(window, min_periods=1), how)()


def engineer(df: pd.DataFrame, traffic_is_causal: bool) -> pd.DataFrame:
    df = df.sort_values(["run_id", "node_id", "time"]).reset_index(drop=True)

    sentinel, floor = CFG.rssi_sentinel, CFG.rssi_floor
    radius, centre = CFG.communication_radius_default, CFG.area_center

    # Replace the isolation sentinel with a physical floor before any arithmetic;
    # leaving -1000 in made every rolling statistic meaningless for isolated nodes.
    df["is_isolated"] = (df["avg_rssi"] <= sentinel).astype(int)
    df["avg_rssi"] = df["avg_rssi"].where(df["avg_rssi"] > sentinel, floor)

    df["dist_to_center"] = np.sqrt((df["x"] - centre) ** 2 + (df["y"] - centre) ** 2)
    df["d_res"] = np.clip(radius - df["dist_to_center"], 0.0, radius)
    df["ND"] = df["neighbor_count"].astype(float)

    grouped = df.groupby(["run_id", "node_id"], sort=False)
    df["rssi_velocity"] = grouped["avg_rssi"].transform(lambda s: s.diff().shift(1)).fillna(0.0)
    df["neighbor_velocity"] = grouped["neighbor_count"].transform(lambda s: s.diff().shift(1)).fillna(0.0)

    if traffic_is_causal:
        # Per second counters: build a running delivery ratio from the past only.
        cum_tx = grouped["tx_packets"].transform(lambda s: s.shift(1).cumsum())
        cum_rx = grouped["rx_packets"].transform(lambda s: s.shift(1).cumsum())
        cum_delay = grouped["delay_sum"].transform(lambda s: s.shift(1).cumsum())
        df["pdr"] = np.where(cum_tx.fillna(0) > 0, cum_rx / cum_tx.replace(0, np.nan), 1.0)
        df["log_delay"] = np.log1p(cum_delay.fillna(0.0).clip(lower=0.0))
    else:
        # End of run aggregates. Not causal at t; stage 3 gates on this.
        df["pdr"] = np.where(df["tx_packets"] > 0, df["rx_packets"] / df["tx_packets"], 1.0)
        df["log_delay"] = np.log1p(df["delay_sum"].clip(lower=0.0))
    df["pdr"] = df["pdr"].fillna(1.0).clip(0.0, 1.0)

    grouped = df.groupby(["run_id", "node_id"], sort=False)
    df["rssi_trend_3"] = grouped["rssi_velocity"].transform(lambda s: _lagged(s, 3, "mean")).fillna(0.0)
    df["neighbor_trend_3"] = grouped["neighbor_velocity"].transform(lambda s: _lagged(s, 3, "mean")).fillna(0.0)
    df["rssi_std_5"] = grouped["avg_rssi"].transform(lambda s: _lagged(s, 5, "std")).fillna(0.0)
    df["neighbor_std_5"] = grouped["neighbor_count"].transform(lambda s: _lagged(s, 5, "std")).fillna(0.0)

    # The paper's factors, built from the same backward looking windows.
    df["LET"] = grouped["d_res"].transform(lambda s: _lagged(s, 5, "mean")).bfill().fillna(0.0)
    df["LS"] = grouped["avg_rssi"].transform(lambda s: _lagged(s, 5, "mean")).bfill().fillna(floor)
    df["LA"] = grouped["avg_rssi"].transform(lambda s: _lagged(s, 5, "std")).fillna(0.0)
    df["LQ_mean"] = grouped["pdr"].transform(lambda s: _lagged(s, 5, "mean")).bfill().fillna(1.0)
    df["LL_d"] = grouped["log_delay"].transform(lambda s: _lagged(s, 5, "mean")).bfill().fillna(0.0)
    df["RSSI"] = df["avg_rssi"].astype(float)
    df["T_hello"] = CFG.hello_interval

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive temporal features and paper factors.")
    parser.add_argument("--input", default=str(INPUT_FILE))
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument("--test-run-count", type=int, default=CFG.test_run_count)
    parser.add_argument("--val-run-count", type=int, default=CFG.val_run_count)
    parser.add_argument("--seed", type=int, default=CFG.random_seed)
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise FileNotFoundError(f"Missing {args.input}. Run pipeline/generate_data.py first.")

    manifest_path = Path(args.input).parent / "dataset_manifest.json"
    traffic_is_causal = False
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as handle:
            traffic_is_causal = bool(json.load(handle).get("traffic_is_causal", False))

    df = engineer(pd.read_csv(args.input), traffic_is_causal=traffic_is_causal)

    run_ids = sorted(df["run_id"].unique().astype(int).tolist())
    split = make_run_split(
        run_ids,
        seed=args.seed,
        test_run_count=args.test_run_count,
        val_run_count=args.val_run_count,
    )
    train_rows = df[df["run_id"].isin(split.train_runs)]
    stats = MinMaxStats.fit(train_rows, NORMALISED_COLUMNS)

    normalised = stats.transform(df, NORMALISED_COLUMNS)
    for column in NORMALISED_COLUMNS:
        df[column] = normalised[column]

    NORM_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    stats.save(NORM_STATS_FILE)
    with open(SPLIT_FILE, "w", encoding="utf-8") as handle:
        json.dump({"seed": args.seed, **split.as_dict(), "fitted_on": "train_runs_only"}, handle, indent=2)

    missing = [c for c in OUT_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"engineer_features produced no column for {missing}")
    df[OUT_COLS].to_csv(args.output, index=False)

    print(f"[engineer_features] wrote {len(df)} rows to {args.output}")
    print(f"[engineer_features] split -> {split.describe()}")
    print(f"[engineer_features] normalisation fitted on {len(train_rows)} training rows")
    print(f"[engineer_features] traffic features causal: {traffic_is_causal}")
    print(f"[engineer_features] model features: {len(FEATURES)}, paper factors: {len(FACTOR_COLS)}")


if __name__ == "__main__":
    main()
