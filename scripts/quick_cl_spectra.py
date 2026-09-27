#!/usr/bin/env python
"""
Quick C_l Spectra Diagnostic Script

Generate C_l power spectra (kappa-kappa, galaxy-kappa, galaxy-galaxy) from
a simulation using the same config.yaml as run_inference.py.

This allows fast visual verification of spectra without running full inference.
Generates N realizations via model.predict(), accumulates the measured spectra,
and produces one figure with mean ± std bands.

Usage:
    python scripts/quick_cl_spectra.py --config configs/inference/config.yaml --cell_size 5.0 --n_realizations 20
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

# Memory optimization for JAX on GPU.
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if "--xla_gpu_enable_command_buffer=" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
    ).strip()

import jax
import jax.numpy as jnp
import jax.random as jr

from desi_cmb_fli import utils
from desi_cmb_fli.cmb_lensing import (
    load_abacus_galaxy_observation,
    load_abacus_kappa_observation,
    sample_healpix_gaussian,
)
from desi_cmb_fli.model import get_model_from_config
from desi_cmb_fli.validation import conditioning_params, measure_spectra, plot_cl_figure

jax.config.update("jax_enable_x64", True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quick C_l spectra diagnostic from simulated data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, default="configs/inference/config.yaml",
        help="Path to config.yaml (same format as run_inference.py)"
    )
    parser.add_argument(
        "--cell_size", type=float, default=None,
        help="Override cell size in Mpc/h (for testing higher resolution)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Base random seed (subsequent realizations use seed+1, seed+2, ...)"
    )
    parser.add_argument(
        "--n_realizations", type=int, default=1,
        help="Number of realizations to average over (for error estimation)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: figures)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show plot interactively"
    )
    parser.add_argument(
        "--no-smooth", action="store_true",
        help="Skip HEALPix alm cut in abacus mode (diagnostic only — exposes gnomview aliasing)"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 80)
    print("QUICK C_l SPECTRA DIAGNOSTIC")
    print("=" * 80)
    print(f"\nJAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    # Load config dict
    cfg_dict = utils.yload(args.config)
    observation_mode = cfg_dict.get("observation_mode", "closure")
    if args.no_smooth and observation_mode == "abacus":
        print("[quick_cl] --no-smooth is ignored in curved-sky HEALPix mode.")

    if args.cell_size is not None:
        cfg_dict["model"]["cell_size"] = args.cell_size
    model, model_config = get_model_from_config(cfg_dict)

    truth_params = cfg_dict.get("truth_params", {})
    base_seed = args.seed if args.seed is not None else cfg_dict.get("seed", 42)
    output_dir = Path(args.output_dir or "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    n_real = args.n_realizations

    spectra_list = []

    if observation_mode == "abacus":
        cmb_enabled = model.cmb_enabled
        galaxies_enabled = model.galaxies_enabled

        if not cmb_enabled and not galaxies_enabled:
            raise ValueError("[Abacus] Neither CMB nor galaxies enabled — nothing to diagnose.")

        # ── Load Abacus kappa (if CMB enabled) ──────────────────────────────
        kappa_noiseless = None
        _var_k = None
        if cmb_enabled:
            print("\n[Abacus mode] Loading Abacus kappa map...")
            abacus_cfg = cfg_dict.get("abacus_kappa", {})
            truth_cmb = load_abacus_kappa_observation(abacus_cfg, model)
            kappa_noiseless = truth_cmb["kappa_pred"]
            print(
                f"[Abacus mode] Loaded masked HEALPix kappa: "
                f"n_pix={kappa_noiseless.shape[0]}, std={float(jnp.std(kappa_noiseless)):.4f}"
            )

        # ── Load Abacus galaxy mesh (if galaxies enabled) ───────────────────
        gxy_obs_mesh = None
        if galaxies_enabled:
            abacus_gxy_cfg = cfg_dict.get("abacus_galaxy", {})
            if not abacus_gxy_cfg.get("file"):
                raise ValueError("[Abacus] galaxies_enabled=true but no abacus_galaxy.file in config.")
            gxy_truth = load_abacus_galaxy_observation(
                abacus_gxy_cfg, model,
            )
            gxy_obs_mesh = gxy_truth["obs"]
            truth_gxy_keys = ("gxy_occ_mask3d", "selec_mesh", "gxy_hp_counts", "chi_range_gxy")

        # ── Build truth dicts and measure spectra ───────────────────────────
        # The diagnostic compares the noiseless maps, which do not change between noise draws.
        print("Measuring the Abacus maps (one realisation: the simulation)...")
        for i in range(1):
            seed_i = base_seed + i
            truth_i = {}

            if cmb_enabled and kappa_noiseless is not None:
                truth_i["kappa_pred"] = kappa_noiseless
                truth_i["kappa_obs"] = kappa_noiseless + sample_healpix_gaussian(
                    jr.key(seed_i),
                    jnp.asarray(model.nell_1d),
                    nside=model.cmb_nside,
                    lmax=model.cmb_lmax,
                    # Same effective mask as load_abacus_kappa_observation (sim & external)
                    mask=getattr(model, "cmb_mask", None),
                )

            if galaxies_enabled and gxy_obs_mesh is not None:
                truth_i["obs"] = gxy_obs_mesh
                for key in truth_gxy_keys:
                    if key in gxy_truth:
                        truth_i[key] = gxy_truth[key]
            spectra_list.append(measure_spectra(truth_i, model, model_config))

    else:
        # Closure: N independent LPT realizations
        print(f"\n[Closure mode] Generating {n_real} realization(s)...")

        cond_params = conditioning_params(model, truth_params)

        @jax.jit
        def run_one_realization(seed):
            return model.predict(
                samples=cond_params,
                hide_base=False,
                hide_samp=False,
                hide_det=False,
                frombase=True,
                rng=jr.key(seed),
            )

        for i in range(n_real):
            seed_i = base_seed + i
            print(f"  Realization {i+1}/{n_real} (seed={seed_i})", end="\r")

            truth_i = run_one_realization(seed_i)

            spectra_list.append(measure_spectra(truth_i, model, model_config))

            import gc
            del truth_i
            gc.collect()
    print("")

    if not spectra_list or spectra_list[0]["ell"] is None:
        print("No spectra measured — nothing to plot.")
        return

    # Theory cosmology: the simulation's on Abacus, the closure truth otherwise; bias from truth.
    theory_params = dict(truth_params)
    if observation_mode == "abacus":
        theory_params.update({k: v for k, v in cfg_dict.get("abacus_truth_params", {}).items()
                              if k in ("Omega_m", "sigma8")})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_cl_figure(spectra_list, model, theory_params, observation_mode,
                   output_dir / f"cl_spectra_{timestamp}.png", show=args.show)


if __name__ == "__main__":
    main()
