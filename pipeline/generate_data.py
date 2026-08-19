"""Stage 1: turn raw NS-3 output into one flat table.

Inputs (data/raw):
  positions_run{N}.csv    per second per node mobility snapshot written by the
                          simulation: time, nodeId, x, y, neighbor_count,
                          avg_neighbor_rssi_dbm
  manet_flowmon_run{N}.xml FlowMonitor serialisation, end of run aggregates
  flowstats_run{N}.csv    optional. Written by the updated simulation when
                          periodic FlowMonitor checkpointing is enabled. This is
                          the only traffic source that is causal at time t.

Output:
  data/processed/paper_raw_dataset.csv
  data/processed/dataset_manifest.json   provenance, including whether the
                                         traffic columns are causal or end of
                                         run aggregates

Note on the FlowMonitor parse. The classifier element that NS-3 writes is
`Ipv4FlowClassifier` and the per flow statistics live at `FlowStats/Flow`.
An earlier version of this script looked for `FlowClassifier/Flow` and for any
descendant named `FlowStats`, which matched the container element and the per
probe records instead. The result was that every traffic column came out as a
constant and nobody noticed, because nothing in the pipeline checked. The
validator added in stage 3 now fails on constant feature columns.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from bootstrap import setup_paths  # noqa: E402

ROOT = setup_paths()

RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = PROC_DIR / "paper_raw_dataset.csv"
MANIFEST_FILE = PROC_DIR / "dataset_manifest.json"

IPV4_PREFIX = ["10", "1", "1"]
EMPTY_STATS = {"tx_packets": 0, "rx_packets": 0, "lost_packets": 0, "delay_sum": 0.0}


def _node_id_from_ipv4(ip: str):
    """10.1.1.k maps to node k-1. Returns None for anything else."""
    try:
        parts = ip.strip().split(".")
    except AttributeError:
        return None
    if len(parts) != 4 or parts[0:3] != IPV4_PREFIX:
        return None
    try:
        last = int(parts[3])
    except ValueError:
        return None
    return last - 1 if last > 0 else None


def _ns_to_float(value: str) -> float:
    try:
        return float(str(value).replace("ns", "").strip() or 0.0)
    except ValueError:
        return 0.0


def parse_flowmonitor_xml(xml_file: str | Path) -> Tuple[Dict[int, dict], dict]:
    """Return (per node aggregates, parse diagnostics).

    Diagnostics are returned rather than printed so the caller can fail loudly
    when a file parses to nothing, instead of silently producing zero columns.
    """
    root = ET.parse(str(xml_file)).getroot()

    classifier = root.find("Ipv4FlowClassifier")
    if classifier is None:
        classifier = root.find("FlowClassifier")  # tolerate other NS-3 versions

    endpoints: Dict[int, Tuple[int, int]] = {}
    if classifier is not None:
        for flow in classifier.findall("Flow"):
            src = _node_id_from_ipv4(flow.attrib.get("sourceAddress", ""))
            dst = _node_id_from_ipv4(flow.attrib.get("destinationAddress", ""))
            fid = flow.attrib.get("flowId")
            if fid is None or src is None or dst is None:
                continue
            try:
                endpoints[int(fid)] = (src, dst)
            except ValueError:
                continue

    stats_root = root.find("FlowStats")
    flow_records = stats_root.findall("Flow") if stats_root is not None else []

    node_stats: Dict[int, dict] = {}

    def ensure(nid: int) -> dict:
        if nid not in node_stats:
            node_stats[nid] = dict(EMPTY_STATS)
        return node_stats[nid]

    matched = 0
    for fs in flow_records:
        try:
            fid = int(fs.attrib.get("flowId", "-1"))
        except ValueError:
            continue
        if fid not in endpoints:
            continue
        matched += 1
        src, dst = endpoints[fid]
        tx = int(fs.attrib.get("txPackets", 0))
        rx = int(fs.attrib.get("rxPackets", 0))
        lost = int(fs.attrib.get("lostPackets", 0))
        delay = _ns_to_float(fs.attrib.get("delaySum", "0ns"))

        s = ensure(src)
        s["tx_packets"] += tx
        s["lost_packets"] += lost
        s["delay_sum"] += delay
        ensure(dst)["rx_packets"] += rx

    diagnostics = {
        "classifier_flows": len(endpoints),
        "stat_flows": len(flow_records),
        "matched_flows": matched,
        "nodes_with_traffic": sum(1 for v in node_stats.values() if v["tx_packets"] > 0),
    }
    return node_stats, diagnostics


def load_interval_traffic(run_id: int) -> pd.DataFrame | None:
    """Per interval traffic counters, if the simulation produced them."""
    path = RAW_DIR / f"flowstats_run{run_id}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    expected = {"time", "nodeId", "tx_packets", "rx_packets", "lost_packets", "delay_sum"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return df.rename(columns={"nodeId": "node_id"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten NS-3 output into one table.")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means use every run found")
    parser.add_argument("--max-rows-per-run", type=int, default=0, help="0 means keep every row")
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    args = parser.parse_args()

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    pos_files = sorted(
        glob.glob(str(RAW_DIR / "positions_run*.csv")),
        key=lambda p: int(Path(p).stem.replace("positions_run", "")),
    )
    if not pos_files:
        raise FileNotFoundError(
            f"No position files in {RAW_DIR}. Run simulation/run_simulation.sh first."
        )
    if args.max_runs > 0:
        pos_files = pos_files[: args.max_runs]

    xml_diag: Dict[int, dict] = {}
    flow_by_run: Dict[int, Dict[int, dict]] = {}
    for pf in pos_files:
        run_id = int(Path(pf).stem.replace("positions_run", ""))
        xmlf = RAW_DIR / f"manet_flowmon_run{run_id}.xml"
        if xmlf.exists():
            flow_by_run[run_id], xml_diag[run_id] = parse_flowmonitor_xml(xmlf)
        else:
            flow_by_run[run_id], xml_diag[run_id] = {}, {"missing_xml": True}

    frames = []
    traffic_granularity = "interval"
    for pf in pos_files:
        run_id = int(Path(pf).stem.replace("positions_run", ""))
        dfp = pd.read_csv(pf).rename(
            columns={"nodeId": "node_id", "avg_neighbor_rssi_dbm": "avg_rssi"}
        )
        dfp["run_id"] = run_id
        if args.max_rows_per_run > 0:
            keep_times = sorted(dfp["time"].unique())
            per_time = max(1, len(dfp) // max(1, len(keep_times)))
            n_times = max(2, args.max_rows_per_run // per_time)
            dfp = dfp[dfp["time"].isin(keep_times[:n_times])]

        interval = load_interval_traffic(run_id)
        if interval is not None:
            dfp = dfp.merge(interval, on=["time", "node_id"], how="left")
            for c in EMPTY_STATS:
                dfp[c] = dfp[c].fillna(0.0)
        else:
            traffic_granularity = "run_aggregate"
            stats = flow_by_run.get(run_id, {})
            for col, default in EMPTY_STATS.items():
                dfp[col] = dfp["node_id"].map(
                    lambda n: stats.get(int(n), EMPTY_STATS)[col]
                ).astype(float if isinstance(default, float) else int)

        frames.append(dfp)

    out_cols = [
        "run_id", "time", "node_id", "x", "y", "neighbor_count", "avg_rssi",
        "tx_packets", "rx_packets", "lost_packets", "delay_sum",
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["run_id", "node_id", "time"]).reset_index(drop=True)
    df[out_cols].to_csv(args.output, index=False)

    manifest = {
        "output": str(args.output),
        "rows": int(len(df)),
        "runs": sorted(int(r) for r in df["run_id"].unique()),
        "nodes_per_run": int(df.groupby("run_id")["node_id"].nunique().max()),
        "timesteps_per_run": int(df.groupby("run_id")["time"].nunique().max()),
        "traffic_granularity": traffic_granularity,
        "traffic_is_causal": traffic_granularity == "interval",
        "flowmonitor_diagnostics": xml_diag,
    }
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_matched = sum(d.get("matched_flows", 0) for d in xml_diag.values())
    print(f"[generate_data] wrote {len(df)} rows to {args.output}")
    print(f"[generate_data] runs={manifest['runs']}")
    print(f"[generate_data] traffic granularity: {traffic_granularity} "
          f"(causal={manifest['traffic_is_causal']})")
    print(f"[generate_data] FlowMonitor flows matched to node pairs: {total_matched}")
    if total_matched == 0 and traffic_granularity == "run_aggregate":
        print("[generate_data] WARNING: no FlowMonitor flow matched a node pair. "
              "Traffic derived features will be constant and stage 3 will reject them.")


if __name__ == "__main__":
    main()
