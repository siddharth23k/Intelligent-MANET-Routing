import pandas as pd
import glob
import xml.etree.ElementTree as ET
import os

DATASET_DIR = "dataset"
dataset_rows = []

def _node_id_from_ipv4(ip: str):
    """
    Maps NS-3 IPv4 addresses to node ids for the default addressing scheme used in the simulation:
      10.1.1.(node_id + 1)
    Returns None if the mapping can't be inferred.
    """
    try:
        parts = ip.strip().split(".")
        if len(parts) != 4:
            return None
        if parts[0:3] != ["10", "1", "1"]:
            return None
        last = int(parts[3])
        if last <= 0:
            return None
        return last - 1
    except Exception:
        return None


def _parse_flowmonitor_xml(xml_file: str):
    """
    Parse FlowMonitor XML and return per-node traffic aggregates.

    We use FlowClassifier/FiveTuple mappings to associate each FlowStats entry with a
    (source_node_id, dest_node_id) pair, then aggregate tx/rx/lost/delay at the node level.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # flow_id -> (src_node_id, dst_node_id)
    flow_endpoints = {}
    for flow in root.findall(".//FlowClassifier/Flow"):
        flow_id = flow.attrib.get("flowId")
        src_ip = flow.attrib.get("sourceAddress")
        dst_ip = flow.attrib.get("destinationAddress")
        if flow_id is None or src_ip is None or dst_ip is None:
            continue
        src_nid = _node_id_from_ipv4(src_ip)
        dst_nid = _node_id_from_ipv4(dst_ip)
        if src_nid is None or dst_nid is None:
            continue
        try:
            flow_endpoints[int(flow_id)] = (int(src_nid), int(dst_nid))
        except Exception:
            continue

    # node_id -> aggregates
    node_stats = {}

    def _ensure(nid: int):
        if nid not in node_stats:
            node_stats[nid] = {"tx_packets": 0, "rx_packets": 0, "lost_packets": 0, "delay_sum": 0.0}
        return node_stats[nid]

    for fs in root.findall(".//FlowStats"):
        fid = fs.attrib.get("flowId")
        if fid is None:
            continue
        try:
            fid_i = int(fid)
        except Exception:
            continue
        endpoints = flow_endpoints.get(fid_i)
        if endpoints is None:
            continue
        src_nid, dst_nid = endpoints

        tx = int(fs.attrib.get("txPackets", 0))
        rx = int(fs.attrib.get("rxPackets", 0))
        lost = int(fs.attrib.get("lostPackets", 0))
        d_str = fs.attrib.get("delaySum", "0ns").replace("ns", "")
        try:
            delay = float(d_str)
        except Exception:
            delay = 0.0

        _ensure(src_nid)["tx_packets"] += tx
        _ensure(src_nid)["lost_packets"] += lost
        _ensure(src_nid)["delay_sum"] += delay

        _ensure(dst_nid)["rx_packets"] += rx

    return node_stats

print("Scanning dataset directory...")

# 1. Parse FlowMonitor XML files (node-level aggregates)
xml_files = glob.glob(os.path.join(DATASET_DIR, "manet_flowmon_run*.xml"))
print(f"XML files found: {len(xml_files)}")

flow_stats_by_run_and_node = {}

for xml_file in xml_files:
    filename = os.path.basename(xml_file)
    # Correctly extract run_id
    run_id = int(filename.replace("manet_flowmon_run", "").replace(".xml", ""))
    flow_stats_by_run_and_node[run_id] = _parse_flowmonitor_xml(xml_file)

print(f"Flow stats parsed for {len(flow_stats_by_run_and_node)} runs.")

# 2. Parse position CSV files 
pos_files = glob.glob(os.path.join(DATASET_DIR, "positions_run*.csv"))
print(f"Position files found: {len(pos_files)}")

for pos_file in pos_files:
    filename = os.path.basename(pos_file)
    run_id = int(filename.replace("positions_run", "").replace(".csv", ""))

    df_pos = pd.read_csv(pos_file)
    run_node_stats = flow_stats_by_run_and_node.get(run_id, {})

    for _, row in df_pos.iterrows():
        node_id = int(row["nodeId"])
        stats = run_node_stats.get(node_id, {"tx_packets": 0, "rx_packets": 0, "lost_packets": 0, "delay_sum": 0.0})
        dataset_rows.append({
            "run_id": run_id,
            "time": row["time"],
            "node_id": node_id,
            "x": row["x"],
            "y": row["y"],
            "neighbor_count": row["neighbor_count"],
            "avg_rssi": row["avg_neighbor_rssi_dbm"],
            "tx_packets": stats["tx_packets"],
            "rx_packets": stats["rx_packets"],
            "lost_packets": stats["lost_packets"],
            "delay_sum": stats["delay_sum"]
        })

df_final = pd.DataFrame(dataset_rows)
output_path = os.path.join(DATASET_DIR, "manet_raw_dataset.csv") # Save as RAW first
df_final.to_csv(output_path, index=False)

print(f"Success! Master dataset saved: {output_path} ({len(df_final)} rows)")