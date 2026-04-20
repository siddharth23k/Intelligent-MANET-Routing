#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SIM_SRC="$REPO_ROOT/simulation/paper_frlfp_simulation.cc"
DATA_DIR="$REPO_ROOT/data/raw"
NS3_DIR="${NS3_DIR:-$HOME/ns-3.47}"
NUM_RUNS="${NUM_RUNS:-30}"
COMM_RADIUS="${COMM_RADIUS:-150}"
NUM_NODES="${NUM_NODES:-100}"
SIM_TIME="${SIM_TIME:-300}"
AREA_SIZE="${AREA_SIZE:-1000}"
SPEED_MIN="${SPEED_MIN:-0}"
SPEED_MAX="${SPEED_MAX:-60}"
PAUSE="${PAUSE:-2}"
PACKET_SIZE="${PACKET_SIZE:-512}"
MAX_PACKETS="${MAX_PACKETS:-300}"
PKT_INTERVAL="${PKT_INTERVAL:-1.0}"
CBR_CONNECTIONS="${CBR_CONNECTIONS:-10}"

mkdir -p "$DATA_DIR"

if [[ ! -f "$SIM_SRC" ]]; then
  echo "Missing $SIM_SRC"
  exit 1
fi

if [[ ! -d "$NS3_DIR" ]]; then
  echo "Missing NS3_DIR: $NS3_DIR"
  exit 1
fi


cp "$SIM_SRC" "$NS3_DIR/scratch/paper_frlfp_simulation.cc"

for seed in $(seq 1 "$NUM_RUNS"); do
    cd "$NS3_DIR"
  ./ns3 run "scratch/paper_frlfp_simulation --RngRun=$seed --runId=$seed --outDir=$DATA_DIR --commRadius=$COMM_RADIUS --numNodes=$NUM_NODES --simTimeSeconds=$SIM_TIME --area=$AREA_SIZE --speedMin=$SPEED_MIN --speedMax=$SPEED_MAX --pause=$PAUSE --packetSize=$PACKET_SIZE --maxPackets=$MAX_PACKETS --interval=$PKT_INTERVAL --cbrConnections=$CBR_CONNECTIONS"
done

