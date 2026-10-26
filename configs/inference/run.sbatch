#!/bin/bash -l

# Load environment
module load cudatoolkit/12.4
export CONDA_PREFIX=${SCRATCH}/envs/desi-cmb-fli

# Set up library paths for JAX/GPU
export LD_LIBRARY_PATH=$(echo ${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH}

# Add env bin to path
export PATH=${CONDA_PREFIX}/bin:$PATH

# Run script with config path (argument 1, default to standard config)
CONFIG_PATH=${1:-configs/inference/config.yaml}
echo "Using config: $CONFIG_PATH"

# Use absolute python path to guarantee correct environment
${CONDA_PREFIX}/bin/python -u scripts/run_inference.py --config "$CONFIG_PATH"
status=$?

# Cleanup temporary config file if it's in the submitted directory
if [[ "$CONFIG_PATH" == *"configs/inference/submitted/"* ]]; then
    echo "Cleaning up temporary config: $CONFIG_PATH"
    rm "$CONFIG_PATH"
fi
exit $status
