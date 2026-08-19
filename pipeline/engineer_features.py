"""Stage 2: derive per node temporal features and the paper's nine link factors.

Two rules govern everything in this file.

1. Features look backwards, labels look forwards. Every rolling or differencing
   operation is preceded by `.shift(1)` inside the (run_id, node_id) group, so a
   row at time t is built only from t-1 and earlier. The label, added in stage 3,
   looks at t + horizon. The two windows meet at t and never cross it.

2. Normalisation statistics are fitted on the training runs only. The previous
   version computed min and max over the whole frame, which let the extremes of
   the held out runs influence the scaling of the training rows. The split is
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

INPUT_FILE = ROOT / "data" / "processed" / "paper_raw_dataset.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "paper_featured_dataset.csv"
NORM_STATS_FILE = ROOT / "results" / "models" / "feature_norm_stats.json"
SPLIT_FILE = ROOT / "results" / "models" / "run_split.json"

CFG = get_config()

# Columns whose min/max are fitted on training runs and reused everywhere.
NORMALISED_COLUMNS = ["RSSI", "LS", "LA", "LET", "LQ_mean", "LL_d", "ND", "d_res"]

OUT_COLS = IDENTIFIER_COLS + [
    "x", "y", "neighbor_count", "avg_rssi",
    "tx_packets", "rx_packets", "lost_packets", "delay_sum",
    "dist_to_center", "rssi_velocity", "neighbor_velocity",
    "pdr", "log_delay",
    "rssi_trend_3", "neighbor_trend_3", "rssi_std_5", "neighbor_std_5",
] + FACTOR_COLS


def _lagged(group, window, how):
    """Backward looking rolling aggregate: covers t-window .. t-1."""
    shifted = group.shift(1)
    roll = shifted.rolling(window, min_periods=1)
    return getattr(roll, how)()


def engineer(df: pd.DataFrame, traffic_is_causal: bool) -> pd.DataFrame:
    df = df.sort_values(["run_id", "node_id", "time"]).reset_index(drop=True)

    sentinel = CFG.rssi_sentinel
    floor = CFG.rssi_floor
    radius = CFG.communication_radius_default
    centre = CFG.area_center

    # Replace the isolation sentinel with a physical floor before any arithmetic.
    # Keeping -1000 in the data made every rolling mean and standard deviation
    # meaningless for isolated nodes.
    df["is_isolated"] = (df["avg_rssi"] <= sentinel).astype(int)
    df["avg_rssi"] = df["avg_rssi"].where(df["avg_rssi"] > sentinel, floor)

    df["dist_to_center"] = np.sqrt((df["x"] - centre) ** 2 + (df["y"] - centre) ** 2)
    df["d_res"] = np.clip(radius - df["dist_to_center"], 0.0, radius)
    df["ND"] = df["neighbor_count"].astype(float)

    g = df.groupby(["run_id", "node_id"], sort=False)

    df["rssi_velocity"] = g["avg_rssi"].transform(lambda s: s.diff().shift(1)).fillna(0.0)
    df["neighbor_velocity"] = g["neighbor_count"].transform(lambda s: s.diff().shift(1)).fillna(0.0)

    if traffic_is_causal:
        # Per interval counters: build a causal running delivery ratio.
        cum_tx = g["tx_packets"].transform(lambda s: s.shift(1).cumsum())
        cum_rx = g["rx_packets"].transform(lambda s: s.shift(1).cumsum())
        cum_delay = g["delay_sum"].transform(lambda s: s.shift(1).cumsum())
        df["pdr"] = np.where(cum_tx.fillna(0) > 0, cum_rx / cum_tx.replace(0, np.nan), 1.0)
        df["log_delay"] = np.log1p(cum_delay.fillna(0.0).clip(lower=0.0))
    else:
        # End of run aggregates. Not causal at time t. Stage 3 refuses to pass
        # this unless the caller explicitly opts in.
        df["pdr"] = np.where(df["tx_packets"] > 0, df["rx_packets"] / df["tx_packets"], 1.0)
        df["log_delay"] = np.log1p(df["delay_sum"].clip(lower=0.0))

    df["pdr"] = df["pdr"].fillna(1.0).clip(0.0, 1.0)

    gv = df.groupby(["run_id", "node_id"], sort=False)
    df["rssi_trend_3"] = gv["rssi_velocity"].transform(lambda s: _lagged(s, 3, "mean")).fillna(0.0)
    df["neighbor_trend_3"] = gv["neighbor_velocity"].transform(lambda s: _lagged(s, 3, "mean")).fillna(0.0)
    df["rssi_std_5"] = gv["avg_rssi"].transform(lambda s: _lagged(s, 5, "std")).fillna(0.0)
    df["neighbor_std_5"] = gv["neighbor_count"].transform(lambda s: _lagged(s, 5, "std")).fillna(0.0)

    # The paper's nine factors, all built from the same backward looking windows.
    df["LET"] = gv["d_res"].transform(lambda s: _lagged(s, 5, "mean")).bfill().fillna(0.0)
    df["LS"] = gv["avg_rssi"].transform(lambda s: _lagged(s, 5, "mean")).bfill().fillna(floor)
    df["LA"] = gv["avg_rssi"].transform(lambda s: _lagged(s, 5, "std")).fillna(0.0)
    df["LQ_mean"] = gv["pdr"].transform(lambda s: _lagged(s, 5, "mean")).bfill().fillna(1.0)
    df["LL_d"] = gv["log_delay"].transform(lambda s: _lagged(s, 5, "mean")).bfill().fillna(0.0)
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
        with open(manifest_path, encoding="utf-8") as f:
            traffic_is_causal = bool(json.load(f).get("traffic_is_causal", False))

    df = pd.read_csv(args.input)
    df = engineer(df, traffic_is_causal=traffic_is_causal)

    # Fit normalisation on training runs only, then apply everywhere.
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
    for c in NORMALISED_COLUMNS:
        df[c] = normalised[c]

    NORM_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    stats.save(NORM_STATS_FILE)
    with open(SPLIT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"seed": args.seed, **split.as_dict(), "fitted_on": "train_runs_only"},
            f,
            indent=2,
        )

    missing = [c for c in OUT_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"engineer_features produced no column for {missing}")
    df[OUT_COLS].to_csv(args.output, index=False)

    print(f"[engineer_features] wrote {len(df)} rows to {args.output}")
    print(f"[engineer_features] split -> {split.describe()}")
    print(f"[engineer_features] normalisation fitted on {len(train_rows)} training rows, "
          f"saved to {NORM_STATS_FILE.name}")
    print(f"[engineer_features] traffic features causal: {traffic_is_causal}")
    print(f"[engineer_features] model features: {len(FEATURES)}, paper factors: {len(FACTOR_COLS)}")


if __name__ == "__main__":
    main()
