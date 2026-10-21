# thesis-desi-cmb-fli

Thesis project targeting the **joint analysis of galaxy clustering and CMB lensing at the field level**.

## Abstract

We present our work towards a field-level inference (FLI) pipeline for the joint analysis of galaxy
clustering and CMB lensing. We demonstrate an end-to-end Bayesian framework that jointly samples the
posterior of the initial density field, galaxy bias and cosmological parameters ($\Omega_m$,
$\sigma_8$). Our differentiable forward model propagates initial conditions through Lagrangian
Perturbation Theory (2LPT), applies a Lagrangian bias expansion, and computes the CMB lensing
convergence along the line of sight via the Born approximation. The joint likelihood combines galaxy
number counts with the convergence map, incorporating the lensing reconstruction noise and
analytically accounting, as a Gaussian variance that follows the cosmology, for the convergence the
model does not generate (beyond the simulation volume and below its resolution). The
high-dimensional posterior is explored efficiently with gradient-based sampling. Validated externally on a high-resolution N-body simulation,
with $\Omega_m$ and $\sigma_8$ free, the joint posterior is consistent with the truth, and CMB
lensing tightens the cosmological constraints relative to galaxies alone by breaking the degeneracy
between the galaxy bias and $\sigma_8$, as expected from standard analyses. This work is a step
towards a field-level analysis of real data, combining DESI galaxies with CMB-lensing maps from
Planck, ACT and the Simons Observatory.

## Code Attribution

This repository builds upon the [benchmark-field-level](https://github.com/hsimonfroy/benchmark-field-level) framework by Hugo Simon.

The curved-sky Born projection of the particles onto HEALPix shells (`cmb_lensing.py`) is built on
Wassim Kabalan's spherical lensing in [JaxPM](https://github.com/DifferentiableUniverseInitiative/JaxPM/tree/3c23f2f0527ac7bed0d265ddcac5ecf34de16568).

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

**✅ Completed:** Initial conditions, gravitational evolution, Lagrangian galaxy bias (with its
redshift evolution) and RSD modeling, curved-sky Born CMB-lensing modeling with an analytic,
cosmology-dependent term for the convergence beyond the modelled volume, and field-level inference
validated both on synthetic **closure** data and on an external high-resolution N-body simulation
(galaxies only, CMB lensing only, and joint). The current headline configuration is the **joint
galaxy × CMB-lensing analysis with $\Omega_m$ and $\sigma_8$ free**, with Simons Observatory
reconstruction noise. Local primordial non-Gaussianity ($f_{\mathrm{NL}}$) is implemented in the
forward model and is work in progress.

**🚧 Next Steps:**
- Comparison with a standard two-point analysis of the CMB-lensing map on the same data.
- First paper (joint method, convergence, map and field reconstruction, $\Omega_m$–$\sigma_8$).
- $f_{\mathrm{NL}}$ with the joint analysis; a larger simulation volume covering the full lensing depth.
- Field-level inference on **real data** (DESI galaxies × Planck/ACT/SO CMB-lensing maps).

See [`pipeline.md`](https://github.com/jhawla/thesis-desi-cmb-fli/blob/main/docs/pipeline.md) for detailed implementation roadmap.

## Citation
The project for the FLI × DESI × CMB lensing analysis should be cited
using `CITATION.cff`.

## License
MIT (see `LICENSE`).
