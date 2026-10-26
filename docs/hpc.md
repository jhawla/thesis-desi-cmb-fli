# NERSC Perlmutter Setup

Quick guide for running on NERSC Perlmutter GPU nodes.

## Initial Setup (once)

```bash
# 1. Log in and source conda
ssh username@perlmutter.nersc.gov
source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20251214-1.0.0/conda/etc/profile.d/conda.sh

# 2. Create conda environment
mkdir -p ${SCRATCH}/envs
conda env create --solver libmamba -p ${SCRATCH}/envs/desi-cmb-fli -f env/environment.yml
conda activate ${SCRATCH}/envs/desi-cmb-fli
pip install -e .

# 3. Install JAX with GPU support
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 4. Enable git hooks
pre-commit install
```

## Running Jobs

All scripts are configured via YAML files in `configs/<script_name>/config.yaml` which contain:
- SLURM parameters (time, nodes, GPUs, constraint)
- Model configuration
- MCMC settings
- Random seeds

### Submit a job

```bash
./configs/inference/submit.py                      # configs/inference/config.yaml
./configs/inference/submit.py configs/inference/abacus/abacus_joint_so_cosmo_box5000_m160.yaml
```

`submit.py` copies the config to `configs/inference/submitted/` and submits `run.sbatch` on it with the
SLURM parameters of its `slurm` block (`job_name`, `account`, `qos`, `constraint`, `nodes`, `gpus`,
`time`); `--gpus` is passed on any GPU constraint (`gpu`, `gpu&hbm80g`).

### Check job status

```bash
squeue -u $USER
```

### View outputs

- Logs: `logs/<jobname>-<jobid>.out` (in home directory)
- Figures: `$SCRATCH/outputs/run_<timestamp>/figures/`
- Config used: `$SCRATCH/outputs/run_<timestamp>/config/`
- Samples: `$SCRATCH/outputs/run_<timestamp>/config/*.npz`

### Storage

- **Scratch** (`$SCRATCH/outputs/`) holds the run directories of `run_inference.py` and nothing else:
  it is the fastest file system for the jobs to write to, but NERSC purges files not accessed for
  8 weeks and keeps no backup. The conda environment (`$SCRATCH/envs/desi-cmb-fli`) is the one
  exception, and must not be deleted.
- **Kept runs** live on CFS, permanent with 7-day snapshots, in
  `/global/cfs/cdirs/desi/users/$USER/desi-cmb-fli/runs/`. `run_inference.py` copies its run there at
  the end (after the analysis, even if the analysis fails), and `analyze_run.py` run by hand copies
  the new figures (`desi_cmb_fli.storage.keep_run`, `rsync -a`, so only what changed is sent). A failed
  copy prints a warning and does not stop the job; then, or for a run interrupted by the time limit:
  ```bash
  rsync -a $SCRATCH/outputs/run_<timestamp>_<jobid> /global/cfs/cdirs/desi/users/$USER/desi-cmb-fli/runs/
  ```
  For many runs at once, use a data-transfer node (`ssh dtn01.nersc.gov`). `density_scan.py` looks a
  run up on scratch first, then there; the other scripts take run paths explicitly.
- **Everything else** stays in the repository: figures in `figures/<topic>/` with the small data
  behind them next to them (`.npz`, `.json` of the same name), caches in `data/cache/` (git-ignored).

### Interactive job

Runs with galaxies need the 80 GB A100s (`hbm80g`): on a 40 GB node the joint runs out of memory on
GPU 0 and hangs without an error (job 59687440). `run_inference.py` reserves JAX's default pool at
start as one block of 95 % of the GPU (`XLA_PYTHON_CLIENT_PREALLOCATE=true`,
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, each set only if the shell does not; the shell may also set
`XLA_PYTHON_CLIENT_ALLOCATOR`) and clears each warm-up's executables before the next step. Grown on
demand instead (preallocation off), the pool adds a region per large request and never returns one,
so a program whose working buffer exceeds every region already held asks for a new one the GPU no
longer has: at mesh 160 the warm-ups ran out of memory with most of the pool free (pipeline §7.19).
The platform allocator (no pool) fits but costs 1.7 × per gradient. A compiled program's
constants load outside that pool: the Fourier kernels and the particle lattice are built with `jnp`
inside the traced code, and `run_inference.py` adds `--xla_disable_hlo_passes=constant_folding` to
`XLA_FLAGS` so that XLA does not fold them back into grid-sized constants (at mesh 160 they made
2.2 GiB per trace of the log-density, 7–12 GB per warm-up program, and the 95 % pool left no room to
load them, pipeline §7.19); what loads outside the pool must fit in the remaining 5 %. A memory figure from `bench_gradient.py` is one gradient in a fresh
process; check a new grid with a short end-to-end `run_inference.py` (few warm-up steps, one batch).

```bash
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint "gpu&hbm80g" --gpus 4 --account=desi
module load cudatoolkit/12.4
source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20251214-1.0.0/conda/etc/profile.d/conda.sh
conda activate ${SCRATCH}/envs/desi-cmb-fli
export LD_LIBRARY_PATH=$(echo ${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH}
python -c "import jax; print(jax.devices())"  # Should show [CudaDevice(id=0), ...]

# Launch job directly
python scripts/run_inference.py --config configs/inference/config.yaml
```

### Long batch job (warm-up beyond 4 h)

The warm-up of `run_inference.py` (mesh, then all parameters) runs in one go and is not checkpointed;
when it does not fit an interactive allocation (the mesh-160 runs), submit the run with `submit.py`
and the config, whose `slurm` block sets the `regular` QOS (`gpu_regular`, at most 48 h), the 80 GB
GPUs (`gpu&hbm80g`) and the time. The queue wait depends strongly on the time and nodes asked: check
the real-time table at https://www.nersc.gov/users/status before submitting and size `slurm.time` to
what the job needs (the warm-up must fit; NERSC charges the time used).

```bash
./configs/inference/submit.py configs/inference/abacus/abacus_gxyonly_cosmo_box5000_m160.yaml
```

The sampler state is saved right after the warm-up (`sampler_state.pkl`), so the batches can then be
continued in an interactive allocation with `run_inference.py --resume <run_dir>` (`run.sbatch` runs
`--config` only).

## Storage

- **DESI DR1 catalogs**: `/global/cfs/cdirs/desi/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5`
- **Planck PR4 Lensing Noise ($N_\ell^{\kappa\kappa}$)**: `/global/cfs/cdirs/cmb/data/planck2020/PR4_lensing/PR4_nlkk_p.dat`
- **ACT DR6 Lensing Noise ($N_\ell^{\kappa\kappa}$)**: `/global/homes/j/jhawla/thesis-desi-cmb-fli/data/N_L_kk_act_dr6_lensing_v1_baseline.txt`
- **AbacusSummit κ maps**: `/global/cfs/cdirs/desi/cosmosim/AbacusLensing/v1/AbacusSummit_{base_c000_ph000/kappa_00047,huge_c000_ph201/kappa_00045}.asdf`
- **AbacusSummit HUGE LRG lightcone + randoms**: `/global/cfs/projectdirs/desi/mocks/cai/abacus_HF_lightcones/DR2/mock_catalogs/lrg_huge/` (reference $C_\ell^{\kappa g}$ in `../../Clkg/lrg_huge/`)
- **AbacusSummit initial conditions**: `/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/ic/<SimName>/ic_dens_N576.asdf`
- **HalfDome** (arXiv:2407.17462): `/global/cfs/cdirs/cmb/gsharing/halfdome`
