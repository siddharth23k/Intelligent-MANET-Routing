#!/usr/bin/env bash
#
# Run the NS-3 scenario NUM_RUNS times with independent RNG substreams and drop
# the output into data/raw. Every parameter can be overridden by an environment
# variable, so a scenario sweep does not need the script edited.
#
#   NS3_DIR=~/ns-3.47 NUM_RUNS=30 bash simulation/run_simulation.sh

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

# Per second FlowMonitor deltas. Without these the traffic features are end of
# run aggregates, which are not causal at time t, and pipeline/validate_dataset.py
# will refuse to pass the dataset.
LOG_FLOW_STATS="${LOG_FLOW_STATS:-true}"

if [[ ! -f "$SIM_SRC" ]]; then
  echo "Missing simulation source: $SIM_SRC" >&2
  exit 1
fi

if [[ ! -d "$NS3_DIR" ]]; then
  echo "Missing NS3_DIR: $NS3_DIR" >&2
  echo "Set NS3_DIR to your ns-3 checkout, for example: NS3_DIR=~/ns-3.47 $0" >&2
  exit 1
fi

mkdir -p "$DATA_DIR"
cp "$SIM_SRC" "$NS3_DIR/scratch/paper_frlfp_simulation.cc"

cd "$NS3_DIR"
for seed in $(seq 1 "$NUM_RUNS"); do
  echo "[run_simulation] run $seed of $NUM_RUNS"
  # RngRun selects a non overlapping substream, which is what makes the runs
  # independent and therefore what makes the run level split legitimate.
  ./ns3 run "scratch/paper_frlfp_simulation \
    --RngRun=$seed \
    --runId=$seed \
    --outDir=$DATA_DIR \
    --commRadius=$COMM_RADIUS \
    --numNodes=$NUM_NODES \
    --simTimeSeconds=$SIM_TIME \
    --area=$AREA_SIZE \
    --speedMin=$SPEED_MIN \
    --speedMax=$SPEED_MAX \
    --pause=$PAUSE \
    --packetSize=$PACKET_SIZE \
    --maxPackets=$MAX_PACKETS \
    --interval=$PKT_INTERVAL \
    --cbrConnections=$CBR_CONNECTIONS \
    --logFlowStats=$LOG_FLOW_STATS"
done

echo "[run_simulation] wrote $NUM_RUNS run(s) to $DATA_DIR"
