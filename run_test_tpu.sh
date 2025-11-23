#!/usr/bin/env bash
set -euo pipefail

# Set TPU environment
export PJRT_DEVICE=TPU

# Disable Python output buffering for immediate logs
export PYTHONUNBUFFERED=1

# Ensure outputs directory exists
mkdir -p /workspace/outputs

# Compute start time and export
START_TIME=$(date +%Y%m%d-%H%M%S)
export START_TIME

mkdir -p /workspace/outputs/${START_TIME}

# Run the Python script with the start time passed as an argument
# Use tee to output to both terminal and log file
cp /workspace/hyperparameter.json /workspace/outputs/${START_TIME}/hyperparameter.json

chmod -R a+x /workspace/outputs/${START_TIME}

python3 -u /workspace/test_tpu.py --start-time "${START_TIME}" 2>&1 | tee "/workspace/outputs/${START_TIME}/run_log_${START_TIME}.log"
