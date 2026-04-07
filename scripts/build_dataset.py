import pandas as pd
import glob
import xml.etree.ElementTree as ET
import os

DATASET_DIR = "dataset"
dataset_rows = []
flow_stats = {}

print("Scanning dataset directory...")

# 1. Parse FlowMonitor XML files
xml_files = glob.glob(os.path.join(DATASET_DIR, "manet_flowmon_run*.xml"))
print(f"XML files found: {len(xml_files)}")

for xml_file in xml_files:
    filename = os.path.basename(xml_file)
    # Correctly extract run_id
    run_id = int(filename.replace("manet_flowmon_run", "").replace(".xml", ""))

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Initialize totals for this run
    tx_total, rx_total, lost_total, delay_total = 0, 0, 0, 0.0

    for flow in root.findall(".//FlowStats"):
        tx_total += int(flow.attrib.get("txPackets", 0))
        rx_total += int(flow.attrib.get("rxPackets", 0))
        lost_total += int(flow.attrib.get("lostPackets", 0))
        
        d_str = flow.attrib.get("delaySum", "0ns").replace("ns", "")
        delay_total += float(d_str)

    flow_stats[run_id] = {
        "tx_packets": tx_total,
        "rx_packets": rx_total,
        "lost_packets": lost_total,
        "delay_sum": delay_total
    }

print(f"Flow stats parsed for {len(flow_stats)} runs.")

# 2. Parse position CSV files 
pos_files = glob.glob(os.path.join(DATASET_DIR, "positions_run*.csv"))
print(f"Position files found: {len(pos_files)}")

for pos_file in pos_files:
    filename = os.path.basename(pos_file)
    run_id = int(filename.replace("positions_run", "").replace(".csv", ""))

    df_pos = pd.read_csv(pos_file)
    stats = flow_stats.get(run_id, {"tx_packets": 0, "rx_packets": 0, "lost_packets": 0, "delay_sum": 0})

    for _, row in df_pos.iterrows():
        dataset_rows.append({
            "run_id": run_id,
            "time": row["time"],
            "node_id": row["nodeId"],
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