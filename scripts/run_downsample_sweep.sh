#!/bin/bash
# Run the downsample sweep (one dataset per GPU, in parallel).
#
# Usage:
#   bash scripts/run_downsample_sweep.sh              # all datasets, auto GPU assignment
#   bash scripts/run_downsample_sweep.sh --dry-run    # print planned runs only
#   bash scripts/run_downsample_sweep.sh --cuda 2     # force all to GPU 2
#
# To run a single dataset manually:
#   python scripts/run_zarr_downsample_sweep.py \
#       --sweep-config configs/downsample_sweep.yaml \
#       --datasets Awakening_balanced --cuda 0

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=/root/miniconda3/envs/eeg-bench-new/bin/python3
SWEEP_CFG=configs/downsample_sweep.yaml
EXTRA_ARGS=("$@")          # forward any extra flags (e.g. --dry-run)

# ── Datasets and GPU assignment ───────────────────────────────────────────────
# Datasets must match the `datasets:` list in configs/downsample_sweep.yaml.
# Available GPUs on this machine: 0–3 (4× A800).
declare -A DATASET_GPU=(
    [Longitudinal_EEG_Reliability]=2
)

LOG_DIR=logs/downsample_sweep
mkdir -p "$LOG_DIR"

PIDS=()
for dataset in "${!DATASET_GPU[@]}"; do
    gpu=${DATASET_GPU[$dataset]}
    log="$LOG_DIR/${dataset}_gpu${gpu}.log"
    echo "[Launch] $dataset -> GPU $gpu  (log: $log)"
    $PYTHON scripts/run_zarr_downsample_sweep.py \
        --sweep-config "$SWEEP_CFG" \
        --datasets "$dataset" \
        --cuda "$gpu" \
        "${EXTRA_ARGS[@]}" \
        > "$log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} processes. Waiting for completion..."
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[ERROR] Process $pid exited with error."
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -eq 0 ]; then
    echo "[Done] All datasets completed successfully."
else
    echo "[Done] $FAILED dataset(s) failed. Check logs in $LOG_DIR/"
    exit 1
fi
