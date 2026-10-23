#!/usr/bin/env python3
"""Submit a SLURM batch job of run_inference.py for a config (default configs/inference/config.yaml),
with the SLURM parameters of its `slurm` block:

    ./configs/inference/submit.py configs/inference/abacus/abacus_joint_so_cosmo_box5000_m160.yaml
"""

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Load config
config_path = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/inference/config.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

slurm = cfg["slurm"]

# Create a unique copy of the config for this job
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submitted_dir = Path("configs/inference/submitted")  # run.sbatch removes the copies made here
submitted_dir.mkdir(exist_ok=True)
job_config_path = submitted_dir / f"{config_path.stem}_{timestamp}.yaml"

print(f"Copying config to: {job_config_path}")
shutil.copy(config_path, job_config_path)

# Build sbatch command
cmd = [
    "sbatch",
    f"--job-name={slurm['job_name']}",
    f"--account={slurm['account']}",
    f"--qos={slurm['qos']}",
    f"--constraint={slurm['constraint']}",
    f"--nodes={slurm['nodes']}",
    f"--time={slurm['time']}",
    "--output=logs/%x-%j.out",
]

# Add GPUs only on GPU nodes (gpu, gpu&hbm80g)
if slurm["constraint"].startswith("gpu"):
    cmd.append(f"--gpus={slurm['gpus']}")

# The actual script to run, passing the specific config file
cmd.append("configs/inference/run.sbatch")
cmd.append(str(job_config_path))

# Submit
print(f"Submitting job with: {' '.join(cmd)}")
result = subprocess.run(cmd)
sys.exit(result.returncode)
