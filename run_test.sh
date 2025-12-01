#!/usr/bin/env bash
set -euo pipefail

# PWD path
PWD_PATH=$(pwd)
export PWD_PATH
# Ensure outputs directory exists
mkdir -p ${PWD_PATH}/outputs

# Compute start time and export
START_TIME=$(date +%Y%m%d-%H%M%S)
export START_TIME

mkdir -p ${PWD_PATH}/outputs/${START_TIME}

# Run the Python script with the start time passed as an argument
# Use tee to output to both terminal and log file
cp ${PWD_PATH}/hyperparameter.json ${PWD_PATH}/outputs/${START_TIME}/hyperparameter.json

chmod -R a+x ${PWD_PATH}/outputs/${START_TIME}

python3 ${PWD_PATH}/test.py --start-time "${START_TIME}" --path "${PWD_PATH}" 2>&1 | tee "${PWD_PATH}/outputs/${START_TIME}/run_log_${START_TIME}.log"