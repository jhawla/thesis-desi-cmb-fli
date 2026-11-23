# NERSC Perlmutter Setup

## Initial setup (once)

```bash
source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20251214-1.0.0/conda/etc/profile.d/conda.sh
mkdir -p ${SCRATCH}/envs
conda env create --solver libmamba -p ${SCRATCH}/envs/desi-cmb-fli -f env/environment.yml
conda activate ${SCRATCH}/envs/desi-cmb-fli
pip install -e .
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pre-commit install
```

## GPUs

Perlmutter has 40 GB and 80 GB A100 nodes. Runs with galaxies need the 80 GB ones: `--constraint "gpu&hbm80g"`.

## Interactive run (up to 4 h)

```bash
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint "gpu&hbm80g" --gpus 4 --account=desi
module load cudatoolkit/12.4
source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20251214-1.0.0/conda/etc/profile.d/conda.sh
conda activate ${SCRATCH}/envs/desi-cmb-fli
export LD_LIBRARY_PATH=$(echo ${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH}
python scripts/run_inference.py --config configs/inference/config.yaml
```

## Batch run (longer than 4 h)

```bash
./configs/inference/submit.py configs/inference/abacus/abacus_gxyonly_cosmo_box5000_m160.yaml
./configs/inference/submit.py --resume $SCRATCH/outputs/run_<timestamp>_<jobid>
```

`submit.py` submits `run.sbatch` with the config's `slurm` block (`regular` QOS, at most 48 h). Queue
waits: https://www.nersc.gov/users/status. A run resumes from `sampler_state.pkl` (after the warm-up) or
`mesh_warmup_state.npz` (after the field warm-up), interactively with `run_inference.py --resume <run_dir>`.

## CPU node (tests without sampling)

GPU hours are scarce: tests that do not sample run on CPU nodes with `JAX_PLATFORMS=cpu`.

```bash
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu --account desi \
    srun --ntasks 1 --cpus-per-task 256 bash logs/<topic>/run.sh
```

## Outputs and storage

- Logs: `logs/<jobname>-<jobid>.out`. Runs: `$SCRATCH/outputs/run_<timestamp>_<jobid>/` (`config/`, `figures/`).
- Scratch holds run directories only (purged after 8 weeks without access). Runs are copied at their end to
  `/global/cfs/cdirs/desi/users/$USER/desi-cmb-fli/runs/`; if the copy fails (it prints a warning):
  ```bash
  rsync -a $SCRATCH/outputs/run_<timestamp>_<jobid> /global/cfs/cdirs/desicollab/users/$USER/desi-cmb-fli/runs/
  ```
- Everything else stays in the repository: figures and their data in `figures/<topic>/`, caches in `data/cache/`.

## Data

- **AbacusSummit κ maps**: `/global/cfs/cdirs/desi/cosmosim/AbacusLensing/v1/AbacusSummit_<sim>/kappa_*.asdf`
- **AbacusSummit HUGE LRG light cone + randoms**: `/global/cfs/projectdirs/desi/mocks/cai/abacus_HF_lightcones/DR2/mock_catalogs/lrg_huge/`
- **AbacusSummit initial conditions**: `/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/ic/<sim>/ic_dens_N576.asdf`
- **CMB lensing noise**: Planck PR4 `/global/cfs/cdirs/cmb/data/planck2020/PR4_lensing/PR4_nlkk_p.dat`; ACT DR6 and SO in `data/`
- **DESI DR1 catalogs**: `/global/cfs/cdirs/desi/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5`
- **HalfDome**: `/global/cfs/cdirs/cmb/gsharing/halfdome`
