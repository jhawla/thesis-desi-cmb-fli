"""Run storage (docs/hpc.md, "Storage"): runs are written on scratch, the fast file system, and kept
on CFS, which is not purged."""

import os
import subprocess
from pathlib import Path

SCRATCH_RUNS = Path(os.environ.get("SCRATCH", ".")) / "outputs"
KEPT_RUNS = Path("/global/cfs/cdirs/desi/users") / os.environ.get("USER", "") / "desi-cmb-fli" / "runs"


def run_path(name, kept=None):
    """A run directory by name: on scratch while it lasts, else its kept copy."""
    kept = KEPT_RUNS if kept is None else Path(kept)
    return next((d / name for d in (SCRATCH_RUNS, kept) if (d / name).is_dir()), SCRATCH_RUNS / name)


def keep_run(run_dir, kept=None):
    """Copy a run directory to the kept runs (``rsync -a``, so a second call only adds what changed).
    The kept copy itself is left alone, and a failure is reported, not raised: it never ends a job."""
    run_dir, kept = Path(run_dir).resolve(), (KEPT_RUNS if kept is None else Path(kept))
    if run_dir.parent == kept.resolve():
        return None
    try:
        kept.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", str(run_dir), f"{kept}/"], check=True)
    except (OSError, subprocess.CalledProcessError) as err:
        print(f"WARNING: run not copied to {kept} ({err}); copy it by hand (docs/hpc.md, Storage)")
        return None
    print(f"Run kept: {kept / run_dir.name}")
    return kept / run_dir.name
