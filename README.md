# thesis-desi-cmb-fli

Thesis project targeting the **joint analysis of galaxy clustering and CMB lensing at the field level**.

## Abstract

We present our work towards a field-level inference (FLI) pipeline for the joint analysis of DESI
galaxy clustering and CMB lensing from Planck and ACT. We demonstrate an end-to-end Bayesian
framework that jointly samples the posterior of the initial density field, galaxy bias, and local
primordial non-Gaussianity ($f_{\mathrm{NL}}$). Our differentiable forward model propagates initial
conditions through Lagrangian Perturbation Theory (2LPT), applies a Lagrangian bias expansion, and
computes the CMB lensing convergence along the line of sight via the Born approximation. The joint
likelihood combines galaxy number counts with the convergence map observations, incorporating ACT DR6
reconstruction noise and analytically accounting for the unmodeled high-z lensing contribution beyond
the simulation volume. The high-dimensional posterior is explored efficiently with the MCLMC sampler,
the joint problem converging as well as galaxies alone. The pipeline is validated against
AbacusLensing simulations. Preliminary results show that CMB lensing improves the reconstruction of
the initial field outside the galaxy survey volume, where clustering alone carries no information.
This work is a step toward applying FLI to real data.

## Code Attribution

This repository builds upon the [benchmark-field-level](https://github.com/hsimonfroy/benchmark-field-level) framework by Hugo Simon.

The `cmb_lensing.py` module is built upon the implementation by François Lanusse (see [repository](https://github.com/EiffL/LPTLensingComparison/blob/c407fdc8c70ebc37bd213be4e79eadd3a619d848/jax_lensing/model.py)).

## Quick Start

**Local development** (CPU only):
```bash
conda env create -f env/environment.yml
conda activate desi-cmb-fli
pip install -e .
pre-commit install
```

**NERSC Perlmutter** (with GPU): See `docs/hpc.md` for complete setup.

## Development

**Run tests**: `pytest`
**Format code**: `ruff format .`
**Preview docs**: `mkdocs serve`

Git hooks automatically format code on commit. CI runs tests on push.

## Pipeline Status

**✅ Completed:** Initial conditions, gravitational evolution, galaxy bias (+ PNG / local f_NL) and
RSD modeling, curved-sky Born CMB-lensing modeling, and field-level inference validated both on
synthetic **closure** data and on **AbacusSummit** N-body data (galaxy-only, CMB-only, and joint
galaxy × κ). The current headline configuration is the **joint analysis on the AbacusSummit huge box**
(both probes), with cosmology fixed to focus the constraint on the primordial non-Gaussianity
parameters.

**🚧 Next Steps:**
- Field-level inference on **real data** (DESI LRG × Planck/ACT κ-maps).

See [`pipeline.md`](https://github.com/jhawla/thesis-desi-cmb-fli/blob/main/docs/pipeline.md) for detailed implementation roadmap.

## Citation
The project for the FLI × DESI × CMB lensing analysis should be cited
using `CITATION.cff`.

## License
MIT (see `LICENSE`).
