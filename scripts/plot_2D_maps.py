#!/usr/bin/env python3
"""
Plot 2D maps: κ (convergence) and projected galaxy density.

In closure mode: generates truth via model.predict() and shows:
    - κ observed (= κ_pred + noise)
    - κ predicted (noiseless, from Born projection)
    - Galaxy density projected (if galaxies_enabled)

In abacus mode: loads the AbacusSummit κ map and shows:
    - κ observed (Abacus + N_ℓ noise)
    - κ Abacus (noiseless)
    - Galaxy density projected (if galaxies_enabled, from AbacusLensing LRG catalog)

Reads all parameters from configs/inference/config.yaml (same as run_inference.py).

Usage:
    python scripts/plot_2D_maps.py
    python scripts/plot_2D_maps.py --config configs/inference/config.yaml --seed 42
    python scripts/plot_2D_maps.py --show
    python scripts/plot_2D_maps.py --save-healpix
"""

from __future__ import annotations

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
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot 2D maps: κ and projected galaxy density",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/inference/config.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Skip HEALPix alm cut in abacus mode (diagnostic)")
    parser.add_argument("--save-healpix", action="store_true",
                        help="Save generated kappa maps as HEALPix FITS files")
    return parser.parse_args()

def _infer_flat_sky_geometry(model):
    chi_max_box = float(model.box_shape[2])
    field_size = float(2.0 * np.degrees(np.arctan(model.box_shape[0] / (2.0 * chi_max_box))))
    npix = int(model.mesh_shape[0])
    chi_center = float(model.box_center[2]) if hasattr(model, "box_center") else float(model.box_shape[2] / 2.0)
    return field_size, npix, chi_center


def _project_masked_healpix(kappa_mask, cmb_mask, cmb_nside, xsize=1200):
    import healpy as hp

    full_map = np.full(hp.nside2npix(cmb_nside), hp.UNSEEN, dtype=float)
    full_map[np.asarray(cmb_mask, dtype=bool)] = np.asarray(kappa_mask, dtype=float)
    # Remplacer cartview par mollview
    proj = hp.mollview(full_map, return_projected_map=True, xsize=xsize)
    plt.close()
    proj = np.asarray(proj, dtype=float)
    proj[proj == hp.UNSEEN] = np.nan
    return proj

def _project_full_healpix(kappa_full, cmb_mask=None, xsize=1200):
    import healpy as hp

    full_map = np.asarray(kappa_full, dtype=float).copy()
    if cmb_mask is not None and full_map.size == np.asarray(cmb_mask).size:
        full_map[~np.asarray(cmb_mask, dtype=bool)] = hp.UNSEEN
    # Remplacer cartview par mollview
    proj = hp.mollview(full_map, return_projected_map=True, xsize=xsize)
    plt.close()
    proj = np.asarray(proj, dtype=float)
    proj[proj == hp.UNSEEN] = np.nan
    return proj

def main():
    args = parse_args()

    from desi_cmb_fli import utils

    cfg = utils.yload(args.config)
    observation_mode = cfg.get("observation_mode", "closure")
    truth_params = cfg.get("truth_params", {})
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    output_dir = Path(args.output_dir or "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: {args.config}")
    print(f"Mode: {observation_mode}, seed: {seed}")

    # ── Build model ──────────────────────────────────────────────────────────
    from desi_cmb_fli.model import get_model_from_config

    model, model_config = get_model_from_config(cfg)
    cmb_enabled = model.cmb_enabled
    galaxies_enabled = model.galaxies_enabled

    # Allow running with just galaxy panel (no CMB) in abacus mode
    if not cmb_enabled and observation_mode != "abacus":
        print("CMB lensing is disabled and not in abacus mode — nothing to plot.")
        return

    field_size, npix, chi_center = _infer_flat_sky_geometry(model)

    kappa_pred = None
    kappa_obs  = None
    obs_field  = None     # galaxy mesh (3D, for projection)
    abacus_lon = None
    abacus_lat = None

    # ── Generate / load κ maps ───────────────────────────────────────────────
    if observation_mode == "abacus":
        abacus_kappa_cfg = cfg.get("abacus_kappa", {})

        if cmb_enabled:
            print("Loading AbacusSummit κ map...")
            from desi_cmb_fli.cmb_lensing import load_abacus_kappa_observation

            cmb_truth = load_abacus_kappa_observation(
                abacus_cfg={**abacus_kappa_cfg, "noise_seed": seed},
                model=model,
            )
            kappa_pred = cmb_truth["kappa_pred"]
            kappa_obs = cmb_truth["kappa_obs"]
            print(f"  κ std(noiseless): {float(jnp.std(kappa_pred)):.4f}")
            print(f"  κ std(observed):  {float(jnp.std(kappa_obs)):.4f}")

        if galaxies_enabled:
            print("Loading AbacusLensing galaxy catalog...")
            from desi_cmb_fli.cmb_lensing import load_abacus_galaxy_observation

            gxy_cfg   = cfg.get("abacus_galaxy", {})
            gxy_truth = load_abacus_galaxy_observation(gxy_cfg, model)
            obs_field = gxy_truth["obs"]
            print(f"  Galaxy mesh std(1+δ): {float(jnp.std(obs_field)):.4f}")

    else:  # closure
        print("Running forward model (closure)...")

        @jax.jit
        def generate_truth(rng):
            res = model.predict(
                samples=truth_params,
                hide_base=False, hide_samp=False, hide_det=False,
                frombase=True, rng=rng,
            )
            out = {}
            if cmb_enabled:
                out["kappa_pred"] = res.get("kappa_pred")
                out["kappa_obs"] = res.get("kappa_obs")
            if galaxies_enabled:
                out["obs"] = res.get("obs")
            return out

        truth = generate_truth(jr.key(seed))
        kappa_pred = None if truth.get("kappa_pred") is None else np.asarray(truth.get("kappa_pred"))
        # In closure mode, "kappa_obs" is the packed real a_lm observable (length
        # ~2*n_alm), not a pixel map. Reconstruct a HEALPix map for plotting/saving.
        if truth.get("kappa_obs") is None:
            kappa_obs = None
        else:
            kappa_obs = np.asarray(model.unpack_to_map(jnp.asarray(truth["kappa_obs"])))
        obs_field = np.asarray(truth.get("obs")) if galaxies_enabled else None

        if kappa_pred is not None:
            print(f"  κ std(predicted): {float(np.std(kappa_pred)):.4f}")
        if kappa_obs is not None:
            print(f"  κ std(observed):  {float(np.std(kappa_obs)):.4f}")

    # ── Save HEALPix maps if requested ───────────────────────────────────────
    if args.save_healpix and cmb_enabled:
        import healpy as hp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        def save_hp_map(arr, suffix):
            if arr is None:
                return
            arr_np = np.asarray(arr, dtype=float)
            full_map = np.full(hp.nside2npix(model.cmb_nside), hp.UNSEEN, dtype=float)
            if arr_np.size == full_map.size:
                full_map = arr_np.copy()
                if getattr(model, "cmb_sim_mask", None) is not None:
                    full_map[~np.asarray(model.cmb_sim_mask, dtype=bool)] = hp.UNSEEN
            else:
                full_map[np.asarray(model.cmb_mask, dtype=bool)] = arr_np

            outfile = output_dir / f"kappa_{suffix}_{timestamp}.fits"
            hp.write_map(str(outfile), full_map, overwrite=True)
            print(f"  Saved HEALPix map: {outfile}")

        save_hp_map(kappa_pred, "pred_noiseless")
        save_hp_map(kappa_obs, "obs_noisy")

    # ── Assemble panels ───────────────────────────────────────────────────────
    panels = []   # list of (array, title, cmap, label, extent, xlabel, ylabel)

    if cmb_enabled and kappa_obs is not None:
        obs_arr = np.asarray(kappa_obs)
        obs_extent = [0, field_size, 0, field_size]
        obs_xlabel = "θ [deg]"
        obs_ylabel = "θ [deg]"
        if obs_arr.ndim == 1:
            if obs_arr.size == 12 * int(model.cmb_nside) * int(model.cmb_nside):
                obs_arr = _project_full_healpix(obs_arr, getattr(model, "cmb_sim_mask", None))
            else:
                obs_arr = _project_masked_healpix(obs_arr, model.cmb_mask, model.cmb_nside)
            obs_extent = [180, -180, -90, 90]
            obs_xlabel = "lon [deg]"
            obs_ylabel = "lat [deg]"
        obs_title = ("κ observed (Abacus + $N_\\ell$ noise)" if observation_mode == "abacus"
                     else "κ observed (pred + noise)")
        panels.append((obs_arr, obs_title, "RdBu_r", "κ", obs_extent, obs_xlabel, obs_ylabel))

    if cmb_enabled and kappa_pred is not None:
        pred_arr = np.asarray(kappa_pred)
        pred_extent = [0, field_size, 0, field_size]
        pred_xlabel = "θ [deg]"
        pred_ylabel = "θ [deg]"
        if pred_arr.ndim == 1:
            if pred_arr.size == 12 * int(model.cmb_nside) * int(model.cmb_nside):
                pred_arr = _project_full_healpix(pred_arr, getattr(model, "cmb_sim_mask", None))
            else:
                pred_arr = _project_masked_healpix(pred_arr, model.cmb_mask, model.cmb_nside)
            pred_extent = [180, -180, -90, 90]
            pred_xlabel = "lon [deg]"
            pred_ylabel = "lat [deg]"
        pred_title = ("κ Abacus (noiseless)" if observation_mode == "abacus"
                      else "κ predicted (noiseless)")
        panels.append((pred_arr, pred_title, "RdBu_r", "κ", pred_extent, pred_xlabel, pred_ylabel))

    if galaxies_enabled and obs_field is not None:
        gxy_mesh_np = np.array(obs_field)

        from desi_cmb_fli.cmb_lensing import project_mesh_to_healpix

        gxy_proj_mask = project_mesh_to_healpix(
            gxy_mesh_np,
            model.box_shape,
            model.observer_position,
            model.cmb_nside,
            model.cmb_mask,
            chi_max=float(getattr(model, "chi_boundary", np.linalg.norm(model.box_shape))),
        )

        gxy_proj = _project_masked_healpix(gxy_proj_mask, model.cmb_mask, model.cmb_nside)

        gxy_title = ("Galaxy density projected (Abacus LRG)" if observation_mode == "abacus"
                     else "Galaxy density (projected)")

        extent = [180, -180, -90, 90]
        panels.append((gxy_proj, gxy_title, "viridis", "N (projected)", extent, "lon [deg]", "lat [deg]"))

    if not panels:
        print("No panels to plot (check cmb_enabled and galaxies_enabled).")
        return

    # ── Plot ─────────────────────────────────────────────────────────────────
    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    for ax, (arr, title, cmap, label, extent, xlabel, ylabel) in zip(axes, panels, strict=False):
        finite = np.isfinite(arr)
        if not np.any(finite):
            raise RuntimeError(f"No finite pixels available to plot for panel: {title}")
        vmax = float(np.percentile(np.abs(arr[finite]), 99))
        if vmax == 0.0:
            vmax = 1e-12
        vmin = -vmax if cmap == "RdBu_r" else None
        im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax, label=label, orientation="horizontal", pad=0.15, fraction=0.05)

    # Info footer
    box  = model.box_shape
    cell = model.cell_shape[0] if hasattr(model, "cell_shape") else 0
    loc  = (f"lon={abacus_lon:.0f}°, lat={abacus_lat:.0f}° | "
            if abacus_lon is not None else "")
    info = (f"{loc}Box: [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}] Mpc/h | "
            f"Cell: {cell:.1f} Mpc/h"
            + (f" | Diagnostic FOV: {field_size:.1f}° | {npix}×{npix} pix" if galaxies_enabled else "")
            + (f" | HEALPix nside={model.cmb_nside}" if cmb_enabled else "")
            + f" | seed={seed}")
    fig.suptitle(info, fontsize=9, y=0.02)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = output_dir / f"2D_maps_{timestamp}.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outfile}")

    if args.show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
