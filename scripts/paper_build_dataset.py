import glob
import os
import shutil
import xml.etree.ElementTree as ET

import pandas as pd

PAPER_RAW_DIR = "dataset/paper/raw"
PAPER_PROC_DIR = "dataset/paper/processed"


def _node_id_from_ipv4(ip: str):
    try:
        p = ip.strip().split(".")
        if len(p) != 4 or p[0:3] != ["10", "1", "1"]:
            return None
        return int(p[3]) - 1
    except Exception:
        return None


def _parse_flowmonitor_xml(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    flow_endpoints = {}
    for flow in root.findall(".//FlowClassifier/Flow"):
        flow_id = flow.attrib.get("flowId")
        src = _node_id_from_ipv4(flow.attrib.get("sourceAddress", ""))
        dst = _node_id_from_ipv4(flow.attrib.get("destinationAddress", ""))
        if flow_id is None or src is None or dst is None:
            continue
        try:
            flow_endpoints[int(flow_id)] = (src, dst)
        except Exception:
            pass

    node_stats = {}

    def ensure(nid: int):
        if nid not in node_stats:
            node_stats[nid] = {"tx_packets": 0, "rx_packets": 0, "lost_packets": 0, "delay_sum": 0.0}
        return node_stats[nid]

    for fs in root.findall(".//FlowStats"):
        try:
            fid = int(fs.attrib.get("flowId", "-1"))
        except Exception:
            continue
        if fid not in flow_endpoints:
            continue
        src, dst = flow_endpoints[fid]
        tx = int(fs.attrib.get("txPackets", 0))
        rx = int(fs.attrib.get("rxPackets", 0))
        lost = int(fs.attrib.get("lostPackets", 0))
        d = float(fs.attrib.get("delaySum", "0ns").replace("ns", "") or 0.0)

        ensure(src)["tx_packets"] += tx
        ensure(src)["lost_packets"] += lost
        ensure(src)["delay_sum"] += d
        ensure(dst)["rx_packets"] += rx

    return node_stats


def main():
    os.makedirs(PAPER_PROC_DIR, exist_ok=True)

    pos_files = sorted(glob.glob(os.path.join(PAPER_RAW_DIR, "positions_run*.csv")))
    xml_files = sorted(glob.glob(os.path.join(PAPER_RAW_DIR, "manet_flowmon_run*.xml")))

    # Fallback: use existing project data if paper/raw not generated yet.
    if not pos_files:
        os.makedirs(PAPER_RAW_DIR, exist_ok=True)
        for f in glob.glob("dataset/positions_run*.csv"):
            shutil.copy2(f, os.path.join(PAPER_RAW_DIR, os.path.basename(f)))
        for f in glob.glob("dataset/manet_flowmon_run*.xml"):
            shutil.copy2(f, os.path.join(PAPER_RAW_DIR, os.path.basename(f)))
        pos_files = sorted(glob.glob(os.path.join(PAPER_RAW_DIR, "positions_run*.csv")))
        xml_files = sorted(glob.glob(os.path.join(PAPER_RAW_DIR, "manet_flowmon_run*.xml")))

    if not pos_files:
        raise FileNotFoundError("No position files found in dataset/paper/raw or dataset/")

    flow_by_run = {}
    for xmlf in xml_files:
        run = int(os.path.basename(xmlf).replace("manet_flowmon_run", "").replace(".xml", ""))
        flow_by_run[run] = _parse_flowmonitor_xml(xmlf)

    rows = []
    for pf in pos_files:
        run_id = int(os.path.basename(pf).replace("positions_run", "").replace(".csv", ""))
        stats = flow_by_run.get(run_id, {})
        dfp = pd.read_csv(pf)
        for _, r in dfp.iterrows():
            nid = int(r["nodeId"])
            s = stats.get(nid, {"tx_packets": 0, "rx_packets": 0, "lost_packets": 0, "delay_sum": 0.0})
            rows.append(
                {
                    "run_id": run_id,
                    "time": r["time"],
                    "node_id": nid,
                    "x": r["x"],
                    "y": r["y"],
                    "neighbor_count": r["neighbor_count"],
                    "avg_rssi": r["avg_neighbor_rssi_dbm"],
                    "tx_packets": s["tx_packets"],
                    "rx_packets": s["rx_packets"],
                    "lost_packets": s["lost_packets"],
                    "delay_sum": s["delay_sum"],
                }
            )

    out = os.path.join(PAPER_PROC_DIR, "paper_raw_dataset.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
