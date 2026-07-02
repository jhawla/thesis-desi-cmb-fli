#!/usr/bin/env python
"""
Quick 3D P(k) Diagnostic Script

Two modes depending on observation_mode in config.yaml:

  closure: Measures 3D matter P(k) from the model's matter_mesh field and
           compares to nonlinear jax_cosmo P_mm(k) at a_fid.

  abacus:  Compares galaxy P(k) from two sources as seen by the likelihood:
           1. Abacus observed galaxies (painted from catalog)
           2. LPT-simulated galaxies at config (truth) cosmology

Usage:
    python scripts/quick_pk_spectra.py --n_realizations 5
    python scripts/quick_pk_spectra.py --config configs/inference/config.yaml --cell_size 10.0
"""

import argparse
import os
import types
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np

from desi_cmb_fli import utils
from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import load_abacus_galaxy_observation
from desi_cmb_fli.metrics import spectrum
from desi_cmb_fli.model import get_model_from_config

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_enable_x64", True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quick 3D P(k) diagnostic from simulated matter field or Abacus galaxies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/inference/config.yaml")
    parser.add_argument("--cell_size", type=float, default=None,
                        help="Override cell size in Mpc/h")
    parser.add_argument("--n_realizations", type=int, default=5,
                        help="Number of realizations (closure mode only)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def pk_theory_nonlinear(cosmo, k_hmpc, a):
    """Nonlinear P_mm(k) in (Mpc/h)^3 at scale factor a."""
    pk_fn = jax.vmap(lambda ki: jnp.squeeze(jc.power.nonlinear_matter_power(cosmo, ki, a)))
    return np.array(pk_fn(jnp.array(k_hmpc, dtype=float)))


def pk_theory_linear(cosmo, k_hmpc, a):
    """Linear P_mm(k) in (Mpc/h)^3 at scale factor a."""
    pk_fn = jax.vmap(lambda ki: jnp.squeeze(jc.power.linear_matter_power(cosmo, ki, a)))
    return np.array(pk_fn(jnp.array(k_hmpc, dtype=float)))


def _build_abacus_proxy(cfg_dict, cell_size):
    """Minimal proxy model for abacus mode (geometry + galaxy attrs only)."""
    model_cfg = cfg_dict.get("model", {})
    box = np.array(model_cfg["box_shape"], dtype=float)
    mesh_shape = np.array([int(round(b / cell_size)) for b in box])
    chi_max = float(box[2])
    gxy_density = float(model_cfg.get("gxy_density", 5e-4))

    proxy = types.SimpleNamespace(
        galaxies_enabled=bool(model_cfg.get("galaxies_enabled", False)),
        box_shape=box,
        mesh_shape=mesh_shape,
        box_center=np.array([0.0, 0.0, chi_max / 2]),
        cell_shape=np.array([cell_size, cell_size, cell_size]),
        gxy_density=gxy_density,
        gxy_count=gxy_density * cell_size**3,
        paint_oversamp=float(model_cfg.get("paint_oversamp", 1.0)),
        loc_fid={k: float(v["loc_fid"]) for k, v in cfg_dict.get("latents", {}).items()
                 if isinstance(v, dict) and "loc_fid" in v},
        # CMB attributes disabled — not needed for P(k)
        cmb_enabled=False,
        full_los_correction=False,
        chi_high_z_max=None,
        high_z_mode="exact",
        cl_high_z_cached=None,
        high_z_gradients=None,
    )
    return proxy, dict(model_cfg, cell_size=cell_size)


def _plot_pk(k_arr, pk_mean, pk_std, pk_th, k_nyq, label_meas, label_th,
             title, outfile, show):
    ratio = pk_mean / pk_th

    fig, (ax_pk, ax_rat) = plt.subplots(2, 1, figsize=(8, 8),
                                        gridspec_kw={"height_ratios": [3, 1]},
                                        sharex=True)

    ax_pk.fill_between(k_arr, pk_mean - pk_std, pk_mean + pk_std,
                       color="steelblue", alpha=0.3, label=r"Mean ± 1$\sigma$")
    ax_pk.loglog(k_arr, pk_mean, color="steelblue", lw=2, label=label_meas)

    ax_pk.loglog(k_arr, pk_th, "k--", lw=1.5, label=label_th)

    ax_pk.axvline(k_nyq, color="red", lw=1, ls=":", alpha=0.7,
                  label=f"$k_{{Nyq}}={k_nyq:.3f}$")
    ax_pk.axvline(0.5 * k_nyq, color="orange", lw=1, ls=":", alpha=0.7,
                  label=f"$0.5 k_{{Nyq}}={0.5*k_nyq:.3f}$")

    ax_pk.set_ylabel(r"$P(k)\;[({\rm Mpc}/h)^3]$", fontsize=12)
    ax_pk.set_title(title, fontsize=10)
    ax_pk.legend(fontsize=9, loc="lower left")
    ax_pk.grid(True, alpha=0.2)
    ax_pk.set_ylim(bottom=1e0)

    ax_rat.semilogx(k_arr, ratio, "o-", color="steelblue", ms=4)
    ax_rat.axhline(1.0, color="k", lw=1.2)
    ax_rat.axhline(1.05, color="k", lw=0.5, ls="--")
    ax_rat.axhline(0.95, color="k", lw=0.5, ls="--")
    ax_rat.axvline(k_nyq, color="red", lw=1, ls=":", alpha=0.7)
    ax_rat.axvline(0.5 * k_nyq, color="orange", lw=1, ls=":", alpha=0.7)
    ax_rat.fill_between(k_arr, ratio - pk_std / pk_th, ratio + pk_std / pk_th,
                        color="steelblue", alpha=0.25)
    ax_rat.set_xlabel(r"$k\;[h/{\rm Mpc}]$", fontsize=12)
    ax_rat.set_ylabel("Meas / Theory", fontsize=10)
    ax_rat.set_ylim(0.5, 1.5)
    ax_rat.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved: {outfile}")
    if show:
        plt.show()
    plt.close()


def _print_ratio_table(k_arr, pk_mean, pk_std, pk_th, k_nyq):
    ratio = pk_mean / pk_th
    low_k  = k_arr < 0.5 * k_nyq
    high_k = (k_arr >= 0.5 * k_nyq) & (k_arr < k_nyq)
    print(f"  {'k [h/Mpc]':>12s}  {'P(k)_meas':>12s}  {'P(k)_th':>12s}  "
          f"{'ratio':>8s}  {'std/mean':>8s}")
    for i in range(len(k_arr)):
        flag = "<-- Nyq" if not low_k[i] else ""
        std_over_mean = pk_std[i] / pk_mean[i] if pk_mean[i] > 0 else float("nan")
        print(f"  {k_arr[i]:12.4f}  {pk_mean[i]:12.4e}  {pk_th[i]:12.4e}  "
              f"  {ratio[i]:8.4f}  {std_over_mean:8.3f}  {flag}")
    print(f"\n  Mean ratio  k < 0.5 k_Nyq : {np.nanmean(ratio[low_k]):.4f}")
    print(f"  Mean ratio  k > 0.5 k_Nyq : {np.nanmean(ratio[high_k]):.4f}")


def _main_closure(args, cfg_dict, output_dir):
    """Closure mode: measure 3D matter P(k) from N forward-model realizations."""
    if args.cell_size is not None:
        cfg_dict["model"]["cell_size"] = args.cell_size

    model, _ = get_model_from_config(cfg_dict)
    truth_params = cfg_dict.get("truth_params", {})
    base_seed = args.seed if args.seed is not None else cfg_dict.get("seed", 42)
    cosmo = get_cosmology(**truth_params)

    a_eff = float(model.a_fid)
    k_fund = float(2 * np.pi / np.min(model.box_shape))
    k_nyq  = float(np.pi * np.min(model.mesh_shape / model.box_shape))

    print(f"\nEffective scale factor: a_fid={a_eff:.4f}  (z={1/a_eff - 1:.4f})")
    print(f"Mesh shape:  {tuple(model.mesh_shape)}")
    print(f"Box shape:   {tuple(model.box_shape)} Mpc/h")
    print(f"Cell size:   {float(model.cell_shape[0]):.3f} Mpc/h")
    print(f"k_fund={k_fund:.4f}  k_Nyq={k_nyq:.4f} h/Mpc")
    print(f"paint_oversamp={model.paint_oversamp}")

    n_real = args.n_realizations
    print(f"\nGenerating {n_real} realization(s)...\n")

    all_k, all_pk = [], []

    @jax.jit
    def run_one_realization(seed):
        return model.predict(
            samples=truth_params,
            hide_base=False, hide_samp=False, hide_det=False,
            frombase=True, rng=jr.key(seed),
        )

    for i in range(n_real):
        seed_i = base_seed + i
        print(f"  Realization {i+1}/{n_real} (seed={seed_i})", end="\r")
        truth_i = run_one_realization(seed_i)
        mm = np.array(truth_i["matter_mesh"])
        delta = mm / np.mean(mm) - 1.0
        k3d, pk3d = spectrum(delta, box_shape=np.array(model.box_shape), comp=(0, 0))
        all_k.append(np.array(k3d))
        all_pk.append(np.array(pk3d))

        import gc
        del truth_i, mm, delta
        gc.collect()

    print(f"  Done.{' '*40}")

    k_arr   = all_k[0]
    pk_mat  = np.array(all_pk)
    pk_mean = pk_mat.mean(axis=0)
    pk_std  = pk_mat.std(axis=0)

    pk_th = pk_theory_nonlinear(cosmo, k_arr, a_eff)

    print("\n" + "=" * 72)
    print("P(k) RATIO TABLE:  measured (mean±std) / theory [nonlinear P_mm]")
    print("=" * 72)
    _print_ratio_table(k_arr, pk_mean, pk_std, pk_th, k_nyq)
    print("  (Should be ~1 everywhere if painting is correct)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = output_dir / f"pk_spectra_{timestamp}.png"
    title = (f"3D Matter P(k) — {n_real} realization(s)\n"
             f"Mesh {tuple(model.mesh_shape)}, "
             f"Box {tuple(int(b) for b in model.box_shape)} Mpc/h, "
             f"paint_oversamp={model.paint_oversamp}")

    _plot_pk(k_arr, pk_mean, pk_std, pk_th, k_nyq,
             label_meas="Measured mean",
             label_th=f"Nonlinear theory (jax_cosmo, a={a_eff:.3f})",
             title=title, outfile=outfile, show=args.show)


def _main_abacus(args, cfg_dict, output_dir):
    """
    Abacus mode: compare galaxy P_gg(k) as seen by the likelihood from:
      1. Abacus observed galaxies (painted from catalog)
      2. LPT-simulated galaxies at config (truth) cosmology

    Both spectra use the same survey mask (gxy_occ_mask3d), exactly as the
    likelihood sees them.
    """
    if args.cell_size is not None:
        cfg_dict["model"]["cell_size"] = args.cell_size

    model, _ = get_model_from_config(cfg_dict)

    if not model.galaxies_enabled:
        raise ValueError(
            "[Abacus P(k)] galaxies_enabled=false in config — nothing to measure. "
            "Set model.galaxies_enabled=true and provide abacus_galaxy.file."
        )

    abacus_gxy_cfg = cfg_dict.get("abacus_galaxy", {})
    if not abacus_gxy_cfg.get("file"):
        raise ValueError("[Abacus P(k)] No abacus_galaxy.file in config.")

    print("\n[Abacus comparison] Loading galaxy catalog...")
    gxy_truth = load_abacus_galaxy_observation(abacus_gxy_cfg, model)
    obs_mesh    = np.array(gxy_truth["obs"])
    survey_mask = np.array(gxy_truth["gxy_occ_mask3d"])

    occ_frac = float(np.mean(survey_mask))
    print(f"[Abacus comparison] Survey mask occupancy: {occ_frac:.4f}")
    print(f"[Abacus comparison] obs_mesh shape: {obs_mesh.shape}")

    # Shot noise: Poisson noise from discrete galaxy counts.
    # obs = count/nbar -> delta = obs - 1 has variance 1/nbar_per_cell per cell
    # -> P_shot = V_cell / nbar_per_cell (flat in k-space)
    # The model (regular grid + LPT) has no shot noise, so we subtract it
    # from the Abacus P(k) for a fair comparison.
    cell_vol      = float(np.prod(model.cell_shape))
    nbar_per_cell = float(model.gxy_count)
    P_shot        = cell_vol / nbar_per_cell

    k_nyq  = float(np.pi * np.min(model.mesh_shape / model.box_shape))
    k_fund = float(2 * np.pi / np.min(model.box_shape))
    print(f"[Abacus comparison] k_fund={k_fund:.4f}  k_Nyq={k_nyq:.4f} h/Mpc")
    print(f"[Abacus comparison] Shot noise: P_shot = {P_shot:.1f} (Mpc/h)^3  "
          f"(nbar_cell={nbar_per_cell:.2f}, V_cell={cell_vol:.0f})")

    # Measure Abacus P(k) with survey mask applied
    delta_abacus = np.zeros_like(obs_mesh)
    delta_abacus[survey_mask] = obs_mesh[survey_mask] - 1.0
    k_arr, pk_abacus_raw = spectrum(delta_abacus, box_shape=np.array(model.box_shape), comp=(0, 0))
    k_arr     = np.array(k_arr)
    pk_abacus = np.array(pk_abacus_raw) / occ_frac - P_shot

    print(f"[Abacus comparison] P_shot / P_gg(k_min) = {P_shot / (pk_abacus[0] + P_shot):.1%}")

    # Model realizations at config (truth) cosmology
    truth_params = cfg_dict.get("truth_params", {})
    base_seed    = args.seed if args.seed is not None else cfg_dict.get("seed", 42)
    n_real       = args.n_realizations

    print(f"\n[Abacus comparison] Generating {n_real} model realization(s)...")
    print(f"  Cosmology: Omega_m={truth_params.get('Omega_m', '?')}, "
          f"sigma8={truth_params.get('sigma8', '?')}")

    all_pk_model = []
    for i in range(n_real):
        seed_i = base_seed + i
        print(f"  Realization {i+1}/{n_real} (seed={seed_i})", end="\r")
        obs_i = model.predict(
            samples=truth_params,
            hide_base=False, hide_samp=False, hide_det=False,
            frombase=True, rng=jr.key(seed_i),
        )
        gxy_mesh_i = np.array(obs_i["gxy_mesh"])
        delta_i    = np.zeros_like(gxy_mesh_i)
        delta_i[survey_mask] = gxy_mesh_i[survey_mask] - 1.0
        _, pk_i_raw = spectrum(delta_i, box_shape=np.array(model.box_shape), comp=(0, 0))
        all_pk_model.append(np.array(pk_i_raw) / occ_frac)

    print(f"  Done.{' '*40}")

    pk_model_mean = np.mean(all_pk_model, axis=0)
    pk_model_std  = np.std(all_pk_model,  axis=0)

    # Print comparison table
    low_k  = k_arr < 0.5 * k_nyq
    high_k = (k_arr >= 0.5 * k_nyq) & (k_arr < k_nyq)

    print("\n" + "=" * 78)
    print("Galaxy P(k) COMPARISON: Abacus vs Model (config cosmology)")
    print("=" * 78)
    print(f"  {'k [h/Mpc]':>10s}  {'P_Abacus':>12s}  {'P_Model':>12s}  {'ratio':>8s}")
    for i in range(len(k_arr)):
        ratio = pk_abacus[i] / pk_model_mean[i] if pk_model_mean[i] > 0 else float("nan")
        flag  = "" if low_k[i] else "  <- > 0.5 k_Nyq"
        print(f"  {k_arr[i]:10.4f}  {pk_abacus[i]:12.4e}  {pk_model_mean[i]:12.4e}  "
              f"{ratio:8.4f}{flag}")

    print(f"\n  Shot noise subtracted (Abacus only): P_shot = {P_shot:.1f} (Mpc/h)^3")
    print(f"  Mean ratio Abacus/Model  k < 0.5 k_Nyq : "
          f"{np.nanmean(pk_abacus[low_k] / pk_model_mean[low_k]):.4f}")
    print(f"  Mean ratio Abacus/Model  k > 0.5 k_Nyq : "
          f"{np.nanmean(pk_abacus[high_k] / pk_model_mean[high_k]):.4f}")

    # Plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = output_dir / f"pk_spectra_abacus_vs_model_{timestamp}.png"

    fig, (ax_pk, ax_rat) = plt.subplots(2, 1, figsize=(9, 8),
                                        gridspec_kw={"height_ratios": [3, 1]},
                                        sharex=True)

    ax_pk.loglog(k_arr, pk_abacus, "o-", color="purple", lw=2.5, ms=4,
                 label=f"Abacus ($-P_{{shot}}={P_shot:.0f}$)")
    ax_pk.fill_between(k_arr, pk_model_mean - pk_model_std, pk_model_mean + pk_model_std,
                       color="steelblue", alpha=0.25)
    ax_pk.loglog(k_arr, pk_model_mean, "s-", color="steelblue", lw=2, ms=3,
                 label=f"Model ({n_real} real.), config cosmology")

    ax_pk.axvline(k_nyq, color="red", lw=1, ls=":", alpha=0.6,
                  label=f"$k_{{Nyq}}={k_nyq:.3f}$")
    ax_pk.axvline(0.5 * k_nyq, color="orange", lw=1, ls=":", alpha=0.6,
                  label=f"$0.5 k_{{Nyq}}={0.5*k_nyq:.3f}$")

    Omega_m = truth_params.get("Omega_m", "?")
    sigma8  = truth_params.get("sigma8", "?")
    mesh_shape = tuple(int(x) for x in model.mesh_shape)
    title = (f"3D Galaxy $P_{{gg}}(k)$ — Abacus vs Model\n"
             f"Mesh {mesh_shape}, Box {tuple(int(b) for b in model.box_shape)} Mpc/h, "
             f"$\\Omega_m={Omega_m}$, $\\sigma_8={sigma8}$")
    ax_pk.set_title(title, fontsize=11, fontweight="bold")
    ax_pk.set_ylabel(r"$P_{gg}(k)\;[({\rm Mpc}/h)^3]$", fontsize=12)
    ax_pk.legend(fontsize=9, loc="lower left")
    ax_pk.grid(True, alpha=0.2)
    ax_pk.set_ylim(bottom=1e0)

    ratio_arr = pk_abacus / pk_model_mean
    ax_rat.semilogx(k_arr, ratio_arr, "o-", color="steelblue", ms=4)
    ax_rat.fill_between(k_arr,
                        ratio_arr - pk_model_std / pk_model_mean,
                        ratio_arr + pk_model_std / pk_model_mean,
                        color="steelblue", alpha=0.2)
    ax_rat.axhline(1.0,  color="k", lw=1.2)
    ax_rat.axhline(1.05, color="k", lw=0.5, ls="--", alpha=0.5)
    ax_rat.axhline(0.95, color="k", lw=0.5, ls="--", alpha=0.5)
    ax_rat.axvline(k_nyq,       color="red",    lw=1, ls=":", alpha=0.6)
    ax_rat.axvline(0.5 * k_nyq, color="orange", lw=1, ls=":", alpha=0.6)
    ax_rat.set_xlabel(r"$k\;[h/{\rm Mpc}]$", fontsize=12)
    ax_rat.set_ylabel("Abacus / Model", fontsize=10)
    ax_rat.set_ylim(0.7, 1.5)
    ax_rat.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved: {outfile}")
    if args.show:
        plt.show()
    plt.close()


def main():
    args = parse_args()

    print("=" * 72)
    print("QUICK 3D P(k) DIAGNOSTIC")
    print("=" * 72)
    print(f"\nJAX backend: {jax.default_backend()}")

    cfg_dict = utils.yload(args.config)
    observation_mode = cfg_dict.get("observation_mode", "closure")
    print(f"observation_mode: {observation_mode}")

    output_dir = Path(args.output_dir or "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    if observation_mode == "abacus":
        _main_abacus(args, cfg_dict, output_dir)
    else:
        _main_closure(args, cfg_dict, output_dir)


if __name__ == "__main__":
    main()
