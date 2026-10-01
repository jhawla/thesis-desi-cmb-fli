# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Field-level Bayesian inference of cosmology, primordial non-Gaussianity (local `f_NL`), galaxy bias
and the initial density field from **galaxy clustering jointly with CMB-lensing convergence (κ)**.
Everything is JAX + NumPyro + BlackJAX and differentiable end to end.

[docs/pipeline.md](docs/pipeline.md) is the authoritative technical reference (forward model,
likelihoods, observation modes, sampling, diagnostics, measured results). **Read the relevant section
there before changing physics or likelihood code** — it explains *why* things are the way they are,
and most non-obvious choices in the code are justified there.

## Commands

```bash
pytest                                   # full suite (CPU, no GPU needed)
pytest tests/test_model.py -q            # one file
pytest tests/test_model.py::test_name    # one test
ruff format . && ruff check --fix .      # formatting / lint (line-length 100)
pre-commit run --all-files               # what CI-adjacent hooks run on commit
mkdocs serve                             # preview docs
```

NERSC Perlmutter setup (conda env, CUDA jax, data paths) is in [docs/hpc.md](docs/hpc.md).

## Running inference

```bash
python scripts/run_inference.py --config configs/inference/config.yaml
python scripts/run_inference.py --config configs/inference/config.yaml --resume $SCRATCH/outputs/run_<ts>_<jobid>
```

- **Never run inference (or any heavy JAX job) on a login node.** Runs go in a 4-hour interactive
  `salloc` (see [docs/hpc.md](docs/hpc.md) for the exact incantation) launching `run_inference.py`
  directly — not `configs/inference/submit.py`, not the 11h `regular` queue. If a run is needed,
  hand the user the `salloc` line rather than starting it.
- Outputs land in `$SCRATCH/outputs/run_<timestamp>_<jobid>/` with `config/` (copied
  `config.yaml`, `model.yaml`, `truth.npz`, `samples_batch_*.npz`, `sampler_state.pkl`) and
  `figures/`. `analyze_run.py` runs automatically at the end of a job.
- `samples_batch_*.npz` hold **scalars only**; the sampled field survives only in
  `sampler_state.pkl` — that is what the field-level figures read. With CMB lensing,
  `kappa_batch_*.npz` hold the κ observable of each chain after every batch (posterior mean/std).

Everything is driven by one YAML ([configs/inference/config.yaml](configs/inference/config.yaml)):
SLURM block, `model`, `cmb_lensing`, `truth_params`, `latents` (priors), `mcmc`, `observation_mode`
and the Abacus data paths. `get_model_from_config` ([src/desi_cmb_fli/model.py:210](src/desi_cmb_fli/model.py#L210))
turns the dict into a `FieldLevelModel`; there is no separate argument plumbing — add a knob to the
YAML and read it there.

## Architecture

Dataflow: Gaussian `init_mesh` → LPT/N-body evolution → Lagrangian bias + RSD → galaxy counts; the
**same** evolved particles are Born-integrated over HEALPix shells → κ map. Both enter a joint
likelihood; MCLMC samples the field and the scalars together in a whitened (Kaiser-preconditioned)
basis.

| module | role |
| --- | --- |
| [model.py](src/desi_cmb_fli/model.py) | `FieldLevelModel` — the whole probabilistic model: `prior`, `evolve`, `likelihood`, `reparam`, preconditioning, config loader |
| [bricks.py](src/desi_cmb_fli/bricks.py) | bias, RSD, PNG (`add_png`, `fNL_bias`), interlaced painting/deconvolution, coordinates, fiducial cosmologies |
| [nbody.py](src/desi_cmb_fli/nbody.py) | growth/distance background (`BackgroundEmulator`, bypasses jax_cosmo ODEs to avoid LLVM OOM), LPT & N-body integrators, Fourier kernels |
| [cmb_lensing.py](src/desi_cmb_fli/cmb_lensing.py) | Born projector onto HEALPix shells, masks/geometry, Abacus κ/galaxy/IC loaders, theory `C_ℓ` |
| [samplers.py](src/desi_cmb_fli/samplers.py) | MCLMC warmup/run (the only path wired into `run_inference.py`); NUTS-within-Gibbs and MAMS exist but are unreachable from the script |
| [metrics.py](src/desi_cmb_fli/metrics.py), [validation.py](src/desi_cmb_fli/validation.py), [chains.py](src/desi_cmb_fli/chains.py), [plot.py](src/desi_cmb_fli/plot.py) | spectra, diagnostic figures, `Samples`/`Chains` containers, plotting |

`scripts/` are thin CLI drivers over those modules: `run_inference.py` (3-step run: warm the mesh →
warm all params → multi-chain mini-batch sampling), `analyze_run.py` (R-hat/ESS, corner, IC and κ
reconstruction figures), `compare_runs.py` / `compare_reconstruction.py` (overlay several runs),
`quick_pk_spectra.py` / `quick_cl_spectra.py` (forward-model sanity checks without sampling).

### Things that bite

- **Observation modes.** `closure` = synthetic data from `truth_params` via `model.predict`;
  `abacus` = real N-body κ map + LRG lightcone, where `truth_params` are ignored by the likelihood
  and `abacus_truth_params` only place corner-plot markers.
- **Oversampling grids.** `init_oversamp` / `evol_oversamp` / `ptcl_oversamp` / `paint_oversamp` are
  distinct grids over the same physical box (anti-aliasing, §2.5). Mesh shapes are auto-adjusted to
  even cells — the real↔complex Gaussian repacking asserts it.
- **The galaxy likelihood is on counts, not `1+δ`** (§3.1), and the selection multiplies at *paint*
  resolution. Runs predating that refactor store overdensity and are neither resumable nor
  comparable with current ones.
- **Cosmology is inferable but fixed by default** (`mcmc.fixed_params: [Omega_m, sigma8]`,
  `high_z_mode: fixed`) so the constraint concentrates on PNG.
- **`png_type`**: use `fNL` (universality). `fNL_bias` frees the PNG amplitudes and is not
  identifiable — it is bimodal in the sign of a quadratic PNG combination.
- Upstream pins matter: `jax<0.10`, a forked `jax_cosmo`, a pinned commit of `JaxPM` (its former `41-spherical-lensing` branch, deleted upstream), a pinned `jax-healpy`
  commit. Don't bump them casually.

## Conventions

- **English only** in code, comments, plots and printed output.
- Document mechanism in [docs/pipeline.md](docs/pipeline.md), not in large comment blocks above the
  code. Write it present-tense ("what the code does and why"), with no changelog and no
  run-specific numbers; measured results with their run id and config go in §7.x.
- This repo builds on Hugo Simon's [benchmark-field-level](https://github.com/hsimonfroy/benchmark-field-level)
  / `montecosmo` (often checked out at `~/montecosmo`); mirror its conventions when porting, and
  note any deliberate divergence in pipeline.md (`b2` follows montecosmo: `b2/2 (δ² − ⟨δ²⟩)`).
- Feature branches; bump `pyproject.toml` + `CITATION.cff` together on a release.
