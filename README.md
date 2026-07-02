# thesis-desi-cmb-fli

Repository for a thesis project targeting **field-level inference (FLI)** on DESI galaxy samples cross-correlated with CMB lensing reconstructions from Planck and ACT.

## Thesis vision
- Construction of a reproducible FLI workflow that ingests DESI galaxy
  clustering data and CMB lensing maps (Planck PR4, ACT DR6).
- Delivery of joint cosmological constraints from the cross-correlation of those
  observables.

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
galaxy × κ). The current headline configuration is the **joint analysis on the AbacusSummit base box**
(both probes), with cosmology fixed to focus the constraint on the primordial non-Gaussianity
parameters.

**🚧 Next Steps:**
- Field-level inference on **real data** (DESI LRG × Planck/ACT κ-maps) — *not yet supported*; the only
  external data ingested today is AbacusSummit.

See [`pipeline.md`](https://github.com/Joletaxi19/thesis-desi-cmb-fli/blob/main/docs/pipeline.md) for detailed implementation roadmap.

## Citation
The project for the FLI × DESI × CMB lensing analysis should be cited
using `CITATION.cff`.

## License
MIT (see `LICENSE`).
