"""Stage 1: flatten raw NS-3 output into one table.

Reads from data/raw:
  positions_run{N}.csv     per second mobility snapshot per node
  manet_flowmon_run{N}.xml FlowMonitor serialisation (end of run aggregates)
  flowstats_run{N}.csv     optional per second traffic deltas, written when the
                           simulation runs with --logFlowStats. The only traffic
                           source that is causal at time t.

The FlowMonitor parse targets `Ipv4FlowClassifier` and `FlowStats/Flow`. An
earlier version looked for `FlowClassifier/Flow` and any descendant named
`FlowStats`, which matched the container element and the per probe records, so
every traffic column came out constant and nothing checked.
"""

from __future__ import annotations

import argparse
import glob
import json
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
    """10.1.1.k maps to node k-1. None for anything else."""
    parts = str(ip).strip().split(".")
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
    when a file parses to nothing.
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
            flow_id = flow.attrib.get("flowId")
            if flow_id is None or src is None or dst is None:
                continue
            try:
                endpoints[int(flow_id)] = (src, dst)
            except ValueError:
                continue

    stats_root = root.find("FlowStats")
    flow_records = stats_root.findall("Flow") if stats_root is not None else []

    node_stats: Dict[int, dict] = {}
    matched = 0
    for record in flow_records:
        try:
            flow_id = int(record.attrib.get("flowId", "-1"))
        except ValueError:
            continue
        if flow_id not in endpoints:
            continue
        matched += 1
        src, dst = endpoints[flow_id]
        for node in (src, dst):
            node_stats.setdefault(node, dict(EMPTY_STATS))

        node_stats[src]["tx_packets"] += int(record.attrib.get("txPackets", 0))
        node_stats[src]["lost_packets"] += int(record.attrib.get("lostPackets", 0))
        node_stats[src]["delay_sum"] += _ns_to_float(record.attrib.get("delaySum", "0ns"))
        node_stats[dst]["rx_packets"] += int(record.attrib.get("rxPackets", 0))

    diagnostics = {
        "classifier_flows": len(endpoints),
        "stat_flows": len(flow_records),
        "matched_flows": matched,
        "nodes_with_traffic": sum(1 for v in node_stats.values() if v["tx_packets"] > 0),
    }
    return node_stats, diagnostics


def load_interval_traffic(run_id: int) -> pd.DataFrame | None:
    """Per second traffic counters, if the simulation produced them."""
    path = RAW_DIR / f"flowstats_run{run_id}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"time", "nodeId", "tx_packets", "rx_packets", "lost_packets", "delay_sum"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return df.rename(columns={"nodeId": "node_id"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten NS-3 output into one table.")
    parser.add_argument("--max-runs", type=int, default=0, help="0 uses every run found")
    parser.add_argument("--max-rows-per-run", type=int, default=0, help="0 keeps every row")
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    args = parser.parse_args()

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    position_files = sorted(
        glob.glob(str(RAW_DIR / "positions_run*.csv")),
        key=lambda p: int(Path(p).stem.replace("positions_run", "")),
    )
    if not position_files:
        raise FileNotFoundError(
            f"No position files in {RAW_DIR}. Run simulation/run_simulation.sh first."
        )
    if args.max_runs > 0:
        position_files = position_files[: args.max_runs]

    diagnostics: Dict[int, dict] = {}
    flow_by_run: Dict[int, Dict[int, dict]] = {}
    for path in position_files:
        run_id = int(Path(path).stem.replace("positions_run", ""))
        xml_path = RAW_DIR / f"manet_flowmon_run{run_id}.xml"
        if xml_path.exists():
            flow_by_run[run_id], diagnostics[run_id] = parse_flowmonitor_xml(xml_path)
        else:
            flow_by_run[run_id], diagnostics[run_id] = {}, {"missing_xml": True}

    frames = []
    traffic_granularity = "interval"
    for path in position_files:
        run_id = int(Path(path).stem.replace("positions_run", ""))
        frame = pd.read_csv(path).rename(
            columns={"nodeId": "node_id", "avg_neighbor_rssi_dbm": "avg_rssi"}
        )
        frame["run_id"] = run_id

        if args.max_rows_per_run > 0:
            times = sorted(frame["time"].unique())
            rows_per_time = max(1, len(frame) // max(1, len(times)))
            keep = max(2, args.max_rows_per_run // rows_per_time)
            frame = frame[frame["time"].isin(times[:keep])]

        interval = load_interval_traffic(run_id)
        if interval is not None:
            frame = frame.merge(interval, on=["time", "node_id"], how="left")
            for column in EMPTY_STATS:
                frame[column] = frame[column].fillna(0.0)
        else:
            traffic_granularity = "run_aggregate"
            stats = flow_by_run.get(run_id, {})
            for column in EMPTY_STATS:
                frame[column] = (
                    frame["node_id"]
                    .map(lambda n: stats.get(int(n), EMPTY_STATS)[column])
                    .astype(float)
                )

        frames.append(frame)

    out_columns = [
        "run_id", "time", "node_id", "x", "y", "neighbor_count", "avg_rssi",
        "tx_packets", "rx_packets", "lost_packets", "delay_sum",
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["run_id", "node_id", "time"]).reset_index(drop=True)
    df[out_columns].to_csv(args.output, index=False)

    manifest = {
        "output": str(args.output),
        "rows": int(len(df)),
        "runs": sorted(int(r) for r in df["run_id"].unique()),
        "nodes_per_run": int(df.groupby("run_id")["node_id"].nunique().max()),
        "timesteps_per_run": int(df.groupby("run_id")["time"].nunique().max()),
        "traffic_granularity": traffic_granularity,
        "traffic_is_causal": traffic_granularity == "interval",
        "flowmonitor_diagnostics": diagnostics,
    }
    with open(MANIFEST_FILE, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    matched_total = sum(d.get("matched_flows", 0) for d in diagnostics.values())
    print(f"[generate_data] wrote {len(df)} rows to {args.output}")
    print(f"[generate_data] runs={manifest['runs']}")
    print(f"[generate_data] traffic granularity: {traffic_granularity} "
          f"(causal={manifest['traffic_is_causal']})")
    print(f"[generate_data] FlowMonitor flows matched to node pairs: {matched_total}")
    if matched_total == 0 and traffic_granularity == "run_aggregate":
        print("[generate_data] WARNING: no FlowMonitor flow matched a node pair, so the "
              "traffic features will be constant and stage 3 will reject them.")


if __name__ == "__main__":
    main()
