#!/usr/bin/env bash
set -euo pipefail

# Find repo root (script is inside repo/scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SIM_SRC="$REPO_ROOT/simulations/manet_simulation.cc"
DATA_DIR="$REPO_ROOT/dataset"

# ns-3 installation directory
NS3_DIR="${NS3_DIR:-$HOME/ns-3.47}"

# create dataset directory if missing
mkdir -p "$DATA_DIR"

echo "Repo root: $REPO_ROOT"
echo "Using ns-3 at: $NS3_DIR"
echo "Dataset directory: $DATA_DIR"

# copy simulation file into ns-3 scratch folder
cp "$SIM_SRC" "$NS3_DIR/scratch/manet_simulation.cc"

NUM_RUNS=30

for seed in $(seq 1 $NUM_RUNS); do
  echo "-------------------------------------"
  echo "Running simulation with seed=$seed"
  echo "-------------------------------------"

  cd "$NS3_DIR"

  ./ns3 run "scratch/manet_simulation --RngRun=$seed --runId=$seed --outDir=$DATA_DIR"

done

echo "---"
echo "All simulations completed."
echo "Files saved in: $DATA_DIR"