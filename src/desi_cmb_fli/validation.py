
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from desi_cmb_fli import metrics, plot
from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import (
    compute_cl_high_z,
    compute_shell_support_fractions,
    compute_theoretical_cl_gg,
    compute_theoretical_cl_kappa,
    compute_theoretical_cl_kappa_windowed,
    compute_theoretical_cl_kg,
    project_mesh_to_healpix,
)
from desi_cmb_fli.metrics import bin_cl_log, get_cl_healpix
from desi_cmb_fli.utils import chreshape, r2chshape


def _infer_box_field_geometry(box_shape, mesh_shape, chi_max=None):
    box_shape = np.asarray(box_shape, dtype=float)
    mesh_shape = np.asarray(mesh_shape, dtype=int)
    chi_ref = float(box_shape[2] if chi_max is None else chi_max)
    half_angle_rad = np.arctan(float(box_shape[0]) / (2.0 * chi_ref))
    field_size_deg = float(2.0 * np.degrees(half_angle_rad))
    return field_size_deg, int(mesh_shape[0])


def _diagnostic_ell_limit(model, cmb_enabled):
    if cmb_enabled:
        return float(model.cmb_lmax)
    from desi_cmb_fli.cmb_lensing import NYQUIST_FRACTION

    field_size_deg, field_npix = _infer_box_field_geometry(model.box_shape, model.mesh_shape)
    return float(NYQUIST_FRACTION * np.pi * field_npix / (field_size_deg * np.pi / 180.0))


def _project_masked_healpix(kappa_mask, cmb_mask, cmb_nside, xsize=1200):
    import healpy as hp

    full_map = np.full(hp.nside2npix(cmb_nside), hp.UNSEEN, dtype=float)
    full_map[np.asarray(cmb_mask, dtype=bool)] = np.asarray(kappa_mask, dtype=float)
    proj = hp.mollview(full_map, return_projected_map=True, xsize=xsize)
    plt.close()
    proj = np.asarray(proj, dtype=float)
    proj[proj == hp.UNSEEN] = np.nan
    return proj


def _project_full_healpix(kappa_full, cmb_mask=None, cmb_nside=None, xsize=1200):
    import healpy as hp

    full_map = np.asarray(kappa_full, dtype=float).copy()
    if cmb_nside is None:
        cmb_nside = hp.npix2nside(full_map.size)
    if cmb_mask is not None and full_map.size == np.asarray(cmb_mask).size:
        full_map[~np.asarray(cmb_mask, dtype=bool)] = hp.UNSEEN
    proj = hp.mollview(full_map, return_projected_map=True, xsize=xsize)
    plt.close()
    proj = np.asarray(proj, dtype=float)
    proj[proj == hp.UNSEEN] = np.nan
    return proj


def _project_galaxy_mesh_to_healpix(truth, model, model_config):
    """Project the galaxy field onto the CMB HEALPix footprint for pseudo-C_ell."""
    support3d = np.asarray(
        truth.get("gxy_occ_mask3d", getattr(model, "gxy_occ_mask3d", np.ones_like(truth["obs"]))),
        dtype=float,
    )
    obs_mesh = np.asarray(truth["obs"], dtype=float)
    masked_mesh = obs_mesh * support3d

    box_shape_arr = np.asarray(model_config.get("box_shape", model.box_shape), dtype=float)
    obs_pos_arr = np.asarray(
        getattr(model, "observer_position", [box_shape_arr[0] / 2.0, box_shape_arr[1] / 2.0, 0.0]),
        dtype=float,
    )

    chi_max_depth = float(getattr(model, "chi_boundary", box_shape_arr[2]))

    # Ray-cast the galaxy field and its 3D occupation support; delta = proj/coverage - 1
    # is independent of the integration step, so the two share one projector.
    proj = project_mesh_to_healpix(
        masked_mesh, box_shape_arr, obs_pos_arr, model.cmb_nside, model.cmb_mask,
        chi_max=chi_max_depth,
    )
    coverage = project_mesh_to_healpix(
        support3d, box_shape_arr, obs_pos_arr, model.cmb_nside, model.cmb_mask,
        chi_max=chi_max_depth,
    )

    cov_max = float(np.max(coverage)) if coverage.size else 0.0
    if cov_max <= 0.0:
        return None

    coverage_frac = coverage / cov_max
    local_mask = coverage_frac > 1e-4
    if not np.any(local_mask):
        return None

    delta = np.zeros_like(proj)
    delta[local_mask] = proj[local_mask] / np.maximum(coverage[local_mask], cov_max * 1e-4) - 1.0

    full_mask = np.zeros_like(np.asarray(model.cmb_mask), dtype=bool)
    full_mask[np.asarray(model.cmb_mask, dtype=bool)] = local_mask

    return {
        "delta_masked": delta[local_mask],
        "mask_full": full_mask,
        "mask_local": local_mask,
        "coverage_local": coverage_frac,
        "f_sky": float(np.mean(full_mask.astype(float))),
    }


def plot_field_slices(
    truth,
    output_dir,
    mesh_shape=None,
    show=False,
    box_shape=None,
    field_size_deg=None,
    field_npix=None,
    chi_center=None,
    observation_mode="closure",
    cmb_mask=None,
    cmb_nside=None,
    observer_position=None,
    chi_boundary=None,
):
    """
    Plot 2D slices of the generated fields (obs, kappa_obs, kappa_pred).

    Args:
        truth (dict): Dictionary containing 'obs', and optionally 'kappa_obs', 'kappa_pred'.
        output_dir (Path or str): Output directory.
        mesh_shape (tuple): Mesh shape (optional, inferred from obs).
        show (bool): Whether to show the plot.
        box_shape (tuple): Box size in Mpc/h (for galaxy projection).
        field_size_deg (float): Field size in degrees (for galaxy projection).
        field_npix (int): Number of pixels for projection.
        chi_center (float): Comoving distance to box center (for galaxy projection).
        observation_mode (str): 'closure' or 'abacus' (adjusts panel titles).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Galaxy Field Slices (XY, XZ, YZ) - only if obs is available
    if "obs" in truth:
        if mesh_shape is None:
            mesh_shape = truth["obs"].shape

        idx_z = mesh_shape[2] // 2
        idx_x = mesh_shape[0] // 2
        idx_y = mesh_shape[1] // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        # Shared color scale across all slices for direct visual comparison.
        obs_arr = np.asarray(truth["obs"])
        obs_vlim = tuple(np.quantile(obs_arr, [5e-5, 1 - 5e-5]))
        # axis=0 -> YZ (x fixed), axis=1 -> XZ (y fixed), axis=2 -> XY (z fixed)
        titles = [f"YZ (x={idx_x})", f"XZ (y={idx_y})", f"XY (z={idx_z})"]
        axes_idx = [0, 1, 2]
        slice_indices = [idx_x, idx_y, idx_z]

        for ax, axis_idx, title, sl_idx in zip(axes, axes_idx, titles, slice_indices, strict=False):
            plt.sca(ax)
            im = plot.plot_mesh(
                truth["obs"],
                sli=slice(sl_idx, sl_idx + 1),
                axis=axis_idx,
                vlim=obs_vlim,
                cmap="viridis",
            )
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label="1 + δ")

        plt.tight_layout()
        plt.savefig(output_dir / "obs_slices.png", dpi=150)
        if show:
            plt.show()
        plt.close()
    elif mesh_shape is None and "kappa_obs" in truth and np.ndim(truth["kappa_obs"]) >= 2:
        # Fallback: use kappa shape if obs not available
        mesh_shape = (truth["kappa_obs"].shape[0], truth["kappa_obs"].shape[1], truth["kappa_obs"].shape[1])

    # 2. CMB Convergence (if available)
    if "kappa_obs" in truth:
        # Determine number of panels based on available data
        has_galaxy = "obs" in truth
        has_kappa_pred = "kappa_pred" in truth

        if has_galaxy:
            # Full mode: 3 panels (kappa_obs, kappa_pred, galaxy)
            ncols = 3
            figsize = (18, 5)
        else:
            # CMB-only: 2 panels (kappa_obs, kappa_pred)
            ncols = 2
            figsize = (12, 5)

        fig, axes = plt.subplots(1, ncols, figsize=figsize)
        if ncols == 2:
            axes = list(axes)

        # Observed kappa
        _obs_arr = np.asarray(truth["kappa_obs"])
        _obs_plot = _obs_arr
        _obs_xlabel = "x [pix]"
        _obs_ylabel = "y [pix]"
        _obs_extent = None
        if _obs_arr.ndim == 1 and cmb_nside is not None:
            if cmb_mask is not None and _obs_arr.size == np.asarray(cmb_mask).size:
                _obs_plot = _project_full_healpix(_obs_arr, cmb_mask, cmb_nside)
            elif cmb_mask is not None:
                _obs_plot = _project_masked_healpix(_obs_arr, cmb_mask, cmb_nside)
            _obs_xlabel = "lon [deg]"
            _obs_ylabel = "lat [deg]"
            # Mollweide raster: longitude runs +180 (left) -> -180 (right) (astro flip),
            # latitude -90 (bottom) -> +90 (top) with origin='lower'.
            _obs_extent = [180, -180, -90, 90]
        _vmax_obs = float(np.percentile(np.abs(_obs_plot[np.isfinite(_obs_plot)]), 99))
        _obs_title = ("κ observed (Abacus + N_ℓ noise)" if observation_mode == "abacus"
                      else "CMB Convergence κ (observed)")
        im0 = axes[0].imshow(_obs_plot, origin="lower", cmap="RdBu_r",
                             vmin=-_vmax_obs, vmax=_vmax_obs, extent=_obs_extent)
        axes[0].set_title(_obs_title)
        axes[0].set_xlabel(_obs_xlabel)
        axes[0].set_ylabel(_obs_ylabel)
        plt.colorbar(im0, ax=axes[0], label="κ", orientation="horizontal", pad=0.15, fraction=0.05)

        # Predicted kappa
        _pred_title = ("κ Abacus (noiseless)" if observation_mode == "abacus"
                       else "CMB Convergence κ (predicted)")
        if has_kappa_pred:
            _pred_arr = np.asarray(truth["kappa_pred"])
            _pred_plot = _pred_arr
            _pred_xlabel = "x [pix]"
            _pred_ylabel = "y [pix]"
            _pred_extent = None
            if _pred_arr.ndim == 1 and cmb_nside is not None:
                if cmb_mask is not None and _pred_arr.size == np.asarray(cmb_mask).size:
                    _pred_plot = _project_full_healpix(_pred_arr, cmb_mask, cmb_nside)
                elif cmb_mask is not None:
                    _pred_plot = _project_masked_healpix(_pred_arr, cmb_mask, cmb_nside)
                _pred_xlabel = "lon [deg]"
                _pred_ylabel = "lat [deg]"
                _pred_extent = [180, -180, -90, 90]
            _vmax_pred = float(np.percentile(np.abs(_pred_plot[np.isfinite(_pred_plot)]), 99))
            im1 = axes[1].imshow(_pred_plot, origin="lower", cmap="RdBu_r",
                                 vmin=-_vmax_pred, vmax=_vmax_pred, extent=_pred_extent)
            axes[1].set_title(_pred_title)
            axes[1].set_xlabel(_pred_xlabel)
            axes[1].set_ylabel(_pred_ylabel)
            plt.colorbar(im1, ax=axes[1], label="κ", orientation="horizontal", pad=0.15, fraction=0.05)
        else:
            axes[1].axis('off')

        # Galaxy projected (only if available and 3-panel mode)
        if has_galaxy:
            idx = truth["obs"].shape[2] // 2 if mesh_shape is None else mesh_shape[2] // 2

            if cmb_mask is not None and cmb_nside is not None:
                _box_arr = np.asarray(box_shape if box_shape is not None else truth["obs"].shape, dtype=float)
                # Use the real observer geometry so the galaxy projection matches the
                # kappa footprint (e.g. center observer => full sky). Falling back to a
                # z=0 corner observer only covers the forward hemisphere.
                _obs_pos = (
                    np.asarray(observer_position, dtype=float)
                    if observer_position is not None
                    else np.array([_box_arr[0] / 2.0, _box_arr[1] / 2.0, 0.0], dtype=float)
                )
                _proxy_attrs = {
                    "cmb_nside": cmb_nside,
                    "cmb_mask": cmb_mask,
                    "box_shape": _box_arr,
                    "observer_position": _obs_pos,
                }
                if chi_boundary is not None:
                    _proxy_attrs["chi_boundary"] = float(chi_boundary)
                gxy_hp = _project_galaxy_mesh_to_healpix(
                    truth,
                    type("ValidationModelProxy", (), _proxy_attrs)(),
                    {"box_shape": _box_arr},
                )
                if gxy_hp is not None:
                    gxy_proj = _project_masked_healpix(
                        gxy_hp["delta_masked"], gxy_hp["mask_full"], cmb_nside
                    )
                    im2 = axes[2].imshow(gxy_proj, origin="lower", cmap="viridis",
                                         extent=[180, -180, -90, 90])
                    axes[2].set_title("Galaxy Density (HEALPix proj.)")
                    axes[2].set_xlabel("lon [deg]")
                    axes[2].set_ylabel("lat [deg]")
                    plt.colorbar(im2, ax=axes[2], label=r"$\delta_g$", orientation="horizontal", pad=0.15, fraction=0.05)
                else:
                    axes[2].axis("off")
            else:
                # Fallback to slice if projection parameters not provided
                im2 = axes[2].imshow(truth["obs"][..., idx], origin="lower", cmap="viridis")
                axes[2].set_title(f"Galaxy Density Slice (z={idx})")
                plt.colorbar(im2, ax=axes[2], label="δ", orientation="horizontal", pad=0.15, fraction=0.05)

        plt.tight_layout()
        plt.savefig(output_dir / "kappa_maps.png", dpi=150)
        if show:
            plt.show()
        plt.close()


def measure_spectra(truth, model, model_config=None):
    """
    Measure Cℓ spectra on the maps in `truth`.

    Pure measurement — no plotting, no theory curves.
    Use this to accumulate spectra over multiple realizations.

    Args:
        truth (dict): Map dictionary with keys like 'kappa_pred', 'kappa_obs', 'obs'.
        model (FieldLevelModel): Initialized model (for field geometry).
        model_config (dict, optional): Model config dict (for box_shape in galaxy projection).

    Returns:
        dict: {ell, cl_kk_pred, cl_kk_obs, cl_gg, cl_kg}. Missing keys are None.
    """
    if model_config is None:
        model_config = getattr(model, "config", {})

    cmb_enabled = model.cmb_enabled
    has_kappa_pred = "kappa_pred" in truth
    has_kappa_obs = "kappa_obs" in truth
    has_galaxies = "obs" in truth

    field_size, npix = _infer_box_field_geometry(model.box_shape, model.mesh_shape)

    ell = None
    cl_kk_pred, cl_kk_obs, cl_gg, cl_kg = None, None, None, None
    cl_mode = "flat"
    lmax_hp = int(model.cmb_lmax) if cmb_enabled else None
    npix_hp = 12 * int(model.cmb_nside) * int(model.cmb_nside) if cmb_enabled else None
    cmb_mask_eff = np.asarray(getattr(model, "cmb_mask", None), dtype=bool) if cmb_enabled and getattr(model, "cmb_mask", None) is not None else None

    # For quick diagnostic spectra, use the stable f_sky estimator on masked
    # HEALPix maps. Direct MASTER inversion of unbinned C_ell is too noisy on
    # small sky fractions and is not suitable for plotting-level diagnostics.
    kk_decouple = "fsky"
    kk_coupling_matrix = None

    # The model stores the noisy observed kappa as a packed observable ("kappa_obs":
    # pseudo-a_lm vector or KL eigenmode coefficients, size != #pixels); the diagnostic
    # needs a pixel map, so reconstruct the (noisy) observed map from the observable.
    if cmb_enabled and has_kappa_obs:
        _ko = np.asarray(truth["kappa_obs"])
        _pix_sizes = {npix_hp} | ({int(cmb_mask_eff.sum())} if cmb_mask_eff is not None else set())
        if _ko.ndim == 1 and _ko.size not in _pix_sizes:
            truth = {
                **truth,
                "kappa_obs": np.asarray(model.unpack_kappa_obs_to_map(jnp.asarray(_ko))),
            }

    # Kappa spectra
    if cmb_enabled and has_kappa_obs and np.ndim(truth["kappa_obs"]) == 1:
        kappa_obs = np.asarray(truth["kappa_obs"])
        if cmb_mask_eff is not None and kappa_obs.size == npix_hp:
            ell, cl_kk_obs, _ = get_cl_healpix(
                kappa_obs[cmb_mask_eff],
                cmb_mask_eff,
                lmax=lmax_hp,
                decouple=kk_decouple,
                coupling_matrix=kk_coupling_matrix,
            )
        else:
            ell, cl_kk_obs, _ = get_cl_healpix(
                kappa_obs,
                model.cmb_mask,
                lmax=lmax_hp,
                decouple=kk_decouple,
                coupling_matrix=kk_coupling_matrix,
            )
        ell, cl_kk_obs = np.asarray(ell), np.asarray(cl_kk_obs)
        cl_mode = "healpix"
    elif cmb_enabled and has_kappa_obs and np.ndim(truth["kappa_obs"]) == 2:
        ell, cl_kk_obs = metrics.get_cl_2d(truth["kappa_obs"], field_size_deg=field_size)
        ell, cl_kk_obs = np.asarray(ell), np.asarray(cl_kk_obs)

    if cmb_enabled and has_kappa_pred and np.ndim(truth["kappa_pred"]) == 1:
        kappa_pred = np.asarray(truth["kappa_pred"])
        if cmb_mask_eff is not None and kappa_pred.size == npix_hp:
            ell_tmp, cl_kk_pred, _ = get_cl_healpix(
                kappa_pred[cmb_mask_eff],
                cmb_mask_eff,
                lmax=lmax_hp,
                decouple=kk_decouple,
                coupling_matrix=kk_coupling_matrix,
            )
        else:
            ell_tmp, cl_kk_pred, _ = get_cl_healpix(
                kappa_pred,
                model.cmb_mask,
                lmax=lmax_hp,
                decouple=kk_decouple,
                coupling_matrix=kk_coupling_matrix,
            )
        cl_kk_pred = np.asarray(cl_kk_pred)
        if ell is None:
            ell = np.asarray(ell_tmp)
        cl_mode = "healpix"
    elif cmb_enabled and has_kappa_pred and np.ndim(truth["kappa_pred"]) == 2:
        ell_tmp, cl_kk_pred = metrics.get_cl_2d(truth["kappa_pred"], field_size_deg=field_size)
        cl_kk_pred = np.asarray(cl_kk_pred)
        if ell is None:
            ell = np.asarray(ell_tmp)

    # Galaxy spectra
    f_sky_gxy = 1.0
    if has_galaxies:
        if cmb_enabled and np.ndim(truth.get("kappa_pred", truth.get("kappa_obs"))) == 1:
            gxy_hp = _project_galaxy_mesh_to_healpix(truth, model, model_config)
            if gxy_hp is not None:
                ell_tmp, cl_gg, info_gg = get_cl_healpix(
                    gxy_hp["delta_masked"],
                    gxy_hp["mask_full"],
                    lmax=lmax_hp,
                )
                cl_gg = np.asarray(cl_gg)
                f_sky_gxy = float(info_gg["norm"])
                if ell is None:
                    ell = np.asarray(ell_tmp)
                cl_mode = "healpix"

                if has_kappa_pred:
                    kappa_pred = np.asarray(truth["kappa_pred"])
                    if cmb_mask_eff is not None and kappa_pred.size == npix_hp:
                        kappa_masked = kappa_pred[cmb_mask_eff]
                        kappa_mask_full = cmb_mask_eff
                    else:
                        kappa_masked = truth["kappa_pred"]
                        kappa_mask_full = model.cmb_mask
                    _, cl_kg, _ = get_cl_healpix(
                        kappa_masked,
                        kappa_mask_full,
                        gxy_hp["delta_masked"],
                        gxy_hp["mask_full"],
                        lmax=lmax_hp,
                    )
                    cl_kg = np.asarray(cl_kg)
            else:
                print("  [validation] Skipping galaxy HEALPix spectra: projected survey footprint is empty.")
        else:
            gxy_field = np.array(truth["obs"])
            mask3d = truth.get("gxy_occ_mask3d", None)
            if mask3d is not None:
                occ_indices = np.where(np.any(np.asarray(mask3d), axis=(0, 1)))[0]
            else:
                occ_indices = np.arange(gxy_field.shape[2])
            if occ_indices.size == 0:
                print("  [validation] Skipping projected galaxy spectra: no occupied z-slices.")
            else:
                gxy_proj = np.mean(gxy_field[:, :, occ_indices], axis=2) - 1.0
                ell_tmp, cl_gg = metrics.get_cl_2d(gxy_proj, field_size_deg=field_size)
                cl_gg = np.asarray(cl_gg)
                if ell is None:
                    ell = np.asarray(ell_tmp)

                if cmb_enabled and has_kappa_pred and np.ndim(truth["kappa_pred"]) == 2:
                    _, cl_kg = metrics.get_cl_2d(truth["kappa_pred"], gxy_proj, field_size_deg=field_size)
                    cl_kg = np.asarray(cl_kg)

    # ── Log-binned versions ────────────────────────────────────────────────
    ell_b, cl_kk_pred_b, cl_kk_obs_b, cl_gg_b, cl_kg_b = (None,) * 5
    n_modes_b = None
    if ell is not None:
        if cl_kk_obs is not None:
            ell_b, cl_kk_obs_b, n_modes_b = bin_cl_log(ell, cl_kk_obs)
        if cl_kk_pred is not None:
            ell_b_tmp, cl_kk_pred_b, n_modes_b_tmp = bin_cl_log(ell, cl_kk_pred)
            if ell_b is None:
                ell_b, n_modes_b = ell_b_tmp, n_modes_b_tmp
        if cl_gg is not None:
            ell_b_gg, cl_gg_b, _ = bin_cl_log(ell, cl_gg)
            if ell_b is None:
                ell_b = ell_b_gg
        if cl_kg is not None:
            _, cl_kg_b, _ = bin_cl_log(ell, cl_kg)

    return {"ell": ell, "cl_kk_pred": cl_kk_pred, "cl_kk_obs": cl_kk_obs,
            "cl_gg": cl_gg, "cl_kg": cl_kg, "f_sky_gxy": f_sky_gxy,
            "cl_mode": cl_mode,
            # binned versions
            "ell_b": ell_b, "cl_kk_pred_b": cl_kk_pred_b, "cl_kk_obs_b": cl_kk_obs_b,
            "cl_gg_b": cl_gg_b, "cl_kg_b": cl_kg_b, "n_modes_b": n_modes_b}


def compute_cl_theory(model, cosmo_val, ell_theory,
                      chi_range_gxy=None, bE=2.0,
                      has_galaxies=True, observation_mode="closure"):
    """
    Compute all Limber theory curves for C_l diagnostic plots.

    Single source of truth for theory calculations — called by both plot_spectra
    (validation.py) and quick_cl_spectra.py so that changes propagate everywhere.

    Returns dict with keys:
        cl_kk_theory, cl_kk_theory_coupled,
        cl_box_highz_theory_coupled, cl_total_theory_coupled,
        cl_gg_theory, cl_gg_theory_full,
        cl_kg_theory, cl_kg_theory_full,
        cl_gg_shot, nell_at_theory, cl_high_z_theory
    """
    def _couple_cl_to_mask(cl_values):
        if (
            cl_values is None
            or not cmb_enabled
            or getattr(model, "cmb_M_ll", None) is None
        ):
            return None

        ell_full = np.arange(int(model.cmb_lmax) + 1, dtype=float)
        cl_full = np.zeros_like(ell_full, dtype=float)
        valid = np.isfinite(cl_values)
        if np.any(valid):
            cl_full = np.interp(
                ell_full,
                np.asarray(ell_theory, dtype=float)[valid],
                np.asarray(cl_values, dtype=float)[valid],
                left=0.0,
                right=0.0,
            )
        cl_coupled_full = np.asarray(model.cmb_M_ll) @ cl_full
        return np.interp(
            np.asarray(ell_theory, dtype=float),
            ell_full,
            cl_coupled_full,
            left=0.0,
            right=0.0,
        )

    cmb_enabled = model.cmb_enabled
    z_source = model.cmb_z_source
    chi_min = 1.0
    # chi_boundary = radius of the inscribed sphere (box face to observer along each axis)
    chi_max = float(getattr(model, "chi_boundary", float(model.box_shape[2]) - float(model.observer_position[2])))
    chi_min_gg = float(chi_range_gxy[0]) if chi_range_gxy is not None else chi_min
    chi_max_gg = float(chi_range_gxy[1]) if chi_range_gxy is not None else chi_max
    dx_mesh = float(model.box_shape[0]) / float(model.mesh_shape[0])
    k_nyq_mesh = np.pi / dx_mesh

    cl_kk_theory = None
    cl_kk_theory_coupled = None
    cl_box_highz_theory_coupled = None
    cl_total_theory_coupled = None
    cl_gg_theory = cl_gg_theory_full = cl_kg_theory = cl_kg_theory_full = None
    cl_gg_shot = None
    nell_at_theory = np.zeros_like(ell_theory)
    cl_high_z_theory = np.zeros_like(ell_theory)

    if cmb_enabled:
        if observation_mode == "closure" and hasattr(model, "cmb_r_shells"):
            shell_weights = compute_shell_support_fractions(
                model.observer_position,
                model.box_shape,
                model.cmb_nside,
                model.cmb_r_shells,
                model.cmb_d_r,
                final_mask=model.cmb_mask,
            )
            cl_kk_theory = np.asarray(
                compute_theoretical_cl_kappa_windowed(
                    cosmo_val,
                    jnp.array(ell_theory),
                    model.cmb_r_shells,
                    model.cmb_a_shells,
                    model.cmb_d_r,
                    z_source,
                    shell_weights=shell_weights,
                    k_nyq=k_nyq_mesh,
                )
            )
        else:
            cl_kk_theory = np.asarray(compute_theoretical_cl_kappa(
                cosmo_val, jnp.array(ell_theory), chi_min, chi_max, z_source
            ))
        ell_noise = np.asarray(model.ell_1d, dtype=float)
        nell_noise = np.asarray(model.nell_1d, dtype=float)
        valid_noise = (ell_noise > 0) & np.isfinite(ell_noise) & np.isfinite(nell_noise) & (nell_noise > 0)
        nell_at_theory = np.exp(np.interp(
            np.log(np.maximum(ell_theory, 1e-5)),
            np.log(np.maximum(ell_noise[valid_noise], 1e-5)),
            np.log(nell_noise[valid_noise]),
        ))
        if model.full_los_correction:
            _high_z_mode = 'exact' if observation_mode == 'abacus' else model.high_z_mode
            cl_high_z_1d = compute_cl_high_z(
                cosmo_val, ell_noise, model.chi_boundary, model.chi_high_z_max,
                z_source, mode=_high_z_mode, cl_cached=model.cl_high_z_cached,
                gradients=model.high_z_gradients, loc_fid=model.loc_fid,
            )
            cl_high_z_theory = np.interp(
                ell_theory, ell_noise, np.asarray(cl_high_z_1d)
            )
        cl_kk_theory_coupled = _couple_cl_to_mask(cl_kk_theory)
        cl_box_highz_theory_coupled = _couple_cl_to_mask(cl_kk_theory + cl_high_z_theory)
        cl_total_theory_coupled = _couple_cl_to_mask(cl_kk_theory + cl_high_z_theory + nell_at_theory)

    if has_galaxies:
        cl_gg_theory = np.asarray(compute_theoretical_cl_gg(
            cosmo_val, jnp.array(ell_theory), chi_min_gg, chi_max_gg, bE, k_nyq=k_nyq_mesh
        ))
        cl_gg_theory_full = np.asarray(compute_theoretical_cl_gg(
            cosmo_val, jnp.array(ell_theory), chi_min_gg, chi_max_gg, bE
        ))
        # Shot noise: use effective chi range when catalog covers only part of the box
        if cmb_enabled and getattr(model, "cmb_mask", None) is not None:
            _field_area_sr = 4.0 * np.pi * float(np.mean(np.asarray(model.cmb_mask, dtype=float)))
        else:
            _field_size_deg, _ = _infer_box_field_geometry(model.box_shape, model.mesh_shape)
            _field_area_sr = (_field_size_deg * np.pi / 180.0) ** 2
        # Volume seen by the projection: inscribed sphere when no explicit range given
        if chi_range_gxy is not None:
            _v_eff = float(model.box_shape[0]) * float(model.box_shape[1]) * (chi_max_gg - chi_min_gg)
        else:
            # Cone of solid angle _field_area_sr (N_ell = Omega / N_gal_in_Omega, so f_sky cancels)
            _v_eff = _field_area_sr / 3.0 * (chi_max_gg**3 - chi_min_gg**3)
        _n_gal = float(model.gxy_density) * _v_eff
        cl_gg_shot = _field_area_sr / _n_gal
        print(f"  Galaxy shot noise: N_ell = {cl_gg_shot:.3e} sr"
              f"  (n_bar={model.gxy_density:.2e} (Mpc/h)^-3, "
              f"N_gal~{_n_gal:.0f}, Omega={_field_area_sr*1e4:.2f}e-4 sr)")
        if cmb_enabled:
            cl_kg_theory = np.asarray(compute_theoretical_cl_kg(
                cosmo_val, jnp.array(ell_theory), chi_min_gg, chi_max_gg, z_source, bE,
                k_nyq=k_nyq_mesh
            ))
            cl_kg_theory_full = np.asarray(compute_theoretical_cl_kg(
                cosmo_val, jnp.array(ell_theory), chi_min_gg, chi_max_gg, z_source, bE
            ))

    return {
        "cl_kk_theory": cl_kk_theory,
        "cl_kk_theory_coupled": cl_kk_theory_coupled,
        "cl_box_highz_theory_coupled": cl_box_highz_theory_coupled,
        "cl_total_theory_coupled": cl_total_theory_coupled,
        "cl_gg_theory": cl_gg_theory,
        "cl_gg_theory_full": cl_gg_theory_full,
        "cl_kg_theory": cl_kg_theory,
        "cl_kg_theory_full": cl_kg_theory_full,
        "cl_gg_shot": cl_gg_shot,
        "nell_at_theory": nell_at_theory,
        "cl_high_z_theory": cl_high_z_theory,
    }


def plot_spectra(truth, model, output_dir, cosmo_params=None, model_config=None,
                 observation_mode="closure", show=False, suffix=""):
    """
    Measure and plot Cℓ spectra from the maps in `truth`, compared to Limber theory.

    Works for any observation mode (closure, Abacus, real data).
    Measures spectra on the provided maps — does NOT generate new realizations.

    Panels:
        - κκ auto-spectrum:
            closure: 3 pairs (grey=obs, purple=pred+highz, blue=pred)
            abacus:  2 pairs (grey=obs, purple=pred which is full LOS)
        - Galaxy auto + κg cross-spectrum (if obs present)

    Args:
        truth (dict): Map dictionary. Recognized keys:
            'kappa_pred' (2D, noiseless κ), 'kappa_obs' (2D, with noise),
            'obs' (3D galaxy field), 'matter_mesh' (3D matter field).
        model (FieldLevelModel): Initialized model (for field geometry, N_ell, etc.).
        output_dir (Path or str): Output directory for the plot.
        cosmo_params (dict, optional): Cosmology for Limber curves {Omega_m, sigma8, ...}.
            If None, uses model.loc_fid.
        model_config (dict, optional): Model config dict (for box_shape in galaxy projection).
        observation_mode (str): 'closure' or 'abacus'. Controls which curves are shown.
        show (bool): Whether to show the plot interactively.
        suffix (str): Suffix for the output filename.

    Returns:
        dict: Measured spectra {ell, cl_kk_pred, cl_kk_obs, cl_gg, cl_kg}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_config is None:
        model_config = getattr(model, "config", {})

    print("=" * 80)
    print("VALIDATION: Power Spectra")
    print("=" * 80)

    cmb_enabled = model.cmb_enabled
    has_kappa_obs = "kappa_obs" in truth
    has_galaxies = "obs" in truth

    if not has_kappa_obs and not has_galaxies:
        print("  No maps to measure spectra on, skipping.")
        return {}

    # ── Cosmology for Limber theory ──────────────────────────────────────────
    if cosmo_params is None:
        cosmo_params = {k: v for k, v in model.loc_fid.items()
                        if k in ("Omega_m", "sigma8")}
    cosmo_val = get_cosmology(**cosmo_params)

    # ── Field geometry ───────────────────────────────────────────────────────
    field_size, _ = _infer_box_field_geometry(model.box_shape, model.mesh_shape)
    ell_max = _diagnostic_ell_limit(model, cmb_enabled)

    # ── Theoretical Limber spectra ───────────────────────────────────────────
    b1_lag = cosmo_params.get("b1", model.loc_fid.get("b1", 1.0))
    bE = 1.0 + b1_lag
    chi_range_gxy = truth.get("chi_range_gxy", None) if has_galaxies else None

    ell_theory = np.geomspace(10, ell_max, 100)

    theory = compute_cl_theory(
        model, cosmo_val, ell_theory,
        chi_range_gxy=chi_range_gxy, bE=bE,
        has_galaxies=has_galaxies, observation_mode=observation_mode,
    )
    cl_kk_theory      = theory["cl_kk_theory"]
    cl_gg_theory      = theory["cl_gg_theory"]
    cl_gg_theory_full = theory.get("cl_gg_theory_full")
    cl_kg_theory      = theory["cl_kg_theory"]
    cl_kg_theory_full = theory.get("cl_kg_theory_full")
    cl_gg_shot        = theory["cl_gg_shot"]
    nell_at_theory    = theory["nell_at_theory"]
    cl_high_z_theory  = theory["cl_high_z_theory"]

    # ── Measure spectra on the provided maps ─────────────────────────────────
    spectra = measure_spectra(truth, model, model_config)
    ell          = spectra["ell"]
    cl_kk_pred   = spectra["cl_kk_pred"]
    cl_kk_obs    = spectra["cl_kk_obs"]
    cl_gg        = spectra["cl_gg"]
    cl_kg        = spectra["cl_kg"]
    ell_b        = spectra["ell_b"]
    cl_kk_pred_b = spectra["cl_kk_pred_b"]
    cl_kk_obs_b  = spectra["cl_kk_obs_b"]
    cl_gg_b      = spectra["cl_gg_b"]
    cl_kg_b      = spectra["cl_kg_b"]
    valid_b = ell_b is not None and len(ell_b) > 0

    # ── Plot ─────────────────────────────────────────────────────────────────
    print("  Generating plot...")
    has_cmb_panel = cmb_enabled and (cl_kk_pred is not None or cl_kk_obs is not None)
    has_gxy_panel = cl_gg is not None

    if has_cmb_panel and has_gxy_panel:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_kk, ax_gg = axes
    elif has_cmb_panel:
        fig, ax_kk = plt.subplots(1, 1, figsize=(8, 6))
        ax_gg = None
    elif has_gxy_panel:
        fig, ax_gg = plt.subplots(1, 1, figsize=(8, 6))
        ax_kk = None
    else:
        print("  Nothing to plot.")
        return {}

    # ── CMB κ panel ──────────────────────────────────────────────────────────
    if has_cmb_panel:
        plt.sca(ax_kk)
        valid = (ell > 10) & (ell < ell_max)

        cl_box_highz_theory = cl_kk_theory + cl_high_z_theory
        cl_total_theory = cl_box_highz_theory + nell_at_theory

        cl_high_z_on_ell = np.interp(ell, ell_theory, cl_high_z_theory)

        # ── Raw faded lines ───────────────────────────────────────────────────
        if cl_kk_obs is not None:
            v_grey = valid & np.isfinite(cl_kk_obs)
            plt.loglog(ell[v_grey], cl_kk_obs[v_grey], "-", color="grey", lw=0.4, alpha=0.3)
        if cl_kk_pred is not None:
            _pred_raw = (cl_kk_pred + cl_high_z_on_ell) if observation_mode == "closure" else cl_kk_pred
            v_pred_raw = valid & np.isfinite(_pred_raw)
            _pred_raw_color = "steelblue" if observation_mode == "closure" else "purple"
            plt.loglog(ell[v_pred_raw], _pred_raw[v_pred_raw], "-", color=_pred_raw_color, lw=0.4, alpha=0.3)

        # ── Binned markers ────────────────────────────────────────────────────
        if valid_b:
            vb_mask = (ell_b > 10) & (ell_b < ell_max)
            if cl_kk_obs_b is not None:
                vb = vb_mask & np.isfinite(cl_kk_obs_b)
                plt.errorbar(ell_b[vb], cl_kk_obs_b[vb], fmt="o", color="grey",
                             ms=4, lw=1.2, capsize=2,
                             label=r"$C_\ell^{\kappa\kappa}$ (obs, binned)")
            if cl_kk_pred_b is not None:
                _pred_b = (cl_kk_pred_b + np.interp(ell_b, ell_theory, cl_high_z_theory)
                           if observation_mode == "closure" else cl_kk_pred_b)
                vb = vb_mask & np.isfinite(_pred_b)
                _pred_b_label = (
                    r"$C_\ell^{\kappa\kappa}$ (pred + high-$z$, binned)"
                    if observation_mode == "closure"
                    else r"$C_\ell^{\kappa\kappa}$ (Abacus, noiseless, binned)"
                )
                plt.errorbar(ell_b[vb], _pred_b[vb], fmt="s", color="purple",
                             ms=4, lw=1.2, capsize=2,
                             label=_pred_b_label)
            if observation_mode == "closure" and cl_kk_pred_b is not None:
                vb = vb_mask & np.isfinite(cl_kk_pred_b)
                plt.errorbar(ell_b[vb], cl_kk_pred_b[vb], fmt="^", color="steelblue",
                             ms=4, lw=1, capsize=2,
                             label=r"$C_\ell^{\kappa\kappa}$ (pred box, binned)")

        # ── Theory dashed lines ───────────────────────────────────────────────
        plt.plot(ell_theory, cl_total_theory, "--", color="grey", lw=1.5, alpha=0.9,
                 label=r"Limber (Box + high-$z$ + $N_\ell$)")
        plt.plot(ell_theory, cl_box_highz_theory, "--", color="purple", lw=1.5, alpha=0.9,
                 label=r"Limber (Box + high-$z$)")
        if observation_mode == "closure":
            plt.plot(ell_theory, cl_kk_theory, "--", color="steelblue", lw=1.5, alpha=0.9,
                     label=r"Limber (Box)")

        mode_label = "closure" if observation_mode == "closure" else "Abacus"
        plt.xlabel(r"$\ell$")
        plt.ylabel(r"$C_\ell$")
        plt.title(rf"$C_\ell^{{\kappa\kappa}}$ — {mode_label}")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.2)
        plt.xlim(10, ell_max)

    # ── Galaxy panel ─────────────────────────────────────────────────────────
    if has_gxy_panel:
        plt.sca(ax_gg)
        valid_gg = (ell > 10) & (ell < ell_max) & np.isfinite(cl_gg)
        plt.loglog(ell[valid_gg], cl_gg[valid_gg], "-", color="green", lw=0.4, alpha=0.3)
        if valid_b and cl_gg_b is not None:
            vb_gg = (ell_b > 10) & (ell_b < ell_max) & np.isfinite(cl_gg_b)
            plt.errorbar(ell_b[vb_gg], cl_gg_b[vb_gg], fmt="o", color="green",
                         ms=4, lw=1.2, capsize=2, label=r"$C_\ell^{gg}$ (binned)")
        if cl_gg_theory is not None:
            plt.plot(ell_theory, np.array(cl_gg_theory), "--", color="green",
                     alpha=0.8, lw=2, label=r"Limber $C_\ell^{gg}$ (res.-aware)")
            plt.plot(ell_theory, np.array(cl_gg_theory) + cl_gg_shot,
                     "-.", color="darkgreen", alpha=0.8, lw=1.5,
                     label=r"Limber $C_\ell^{gg}$ + shot noise")
            plt.axhline(cl_gg_shot, color="gray", lw=0.8, ls=":", alpha=0.6,
                        label=rf"Shot noise $N_\ell={cl_gg_shot:.1e}$")
        if cl_gg_theory_full is not None:
            plt.plot(ell_theory, np.array(cl_gg_theory_full), ":", color="green",
                     alpha=0.4, lw=1.5, label=r"Limber $C_\ell^{gg}$ (full res.)")

        if cl_kg is not None:
            valid_kg = (ell > 10) & (ell < ell_max) & np.isfinite(cl_kg)
            plt.loglog(ell[valid_kg], np.abs(cl_kg[valid_kg]), "-", color="red",
                       lw=0.4, alpha=0.3)
            if valid_b and cl_kg_b is not None:
                vb_kg = (ell_b > 10) & (ell_b < ell_max) & np.isfinite(cl_kg_b)
                plt.errorbar(ell_b[vb_kg], np.abs(cl_kg_b[vb_kg]), fmt="s", color="red",
                             ms=4, lw=1.2, capsize=2, label=r"|$C_\ell^{\kappa g}$| (binned)")
            if cl_kg_theory is not None:
                plt.plot(ell_theory, np.abs(np.array(cl_kg_theory)), "--", color="red",
                         alpha=0.8, lw=2, label=r"Limber |$C_\ell^{\kappa g}$| (res.-aware)")
            if cl_kg_theory_full is not None:
                plt.plot(ell_theory, np.abs(np.array(cl_kg_theory_full)), ":", color="red",
                         alpha=0.4, lw=1.5, label=r"Limber |$C_\ell^{\kappa g}$| (full res.)")
            plt.title(r"Galaxy Auto & Cross Spectra")
        else:
            plt.title(r"Galaxy Auto Power Spectrum")

        plt.xlabel(r"$\ell$")
        plt.ylabel(r"$C_\ell$")
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.2)
        plt.xlim(10, ell_max)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    # Info box
    box_x, box_y, box_z = (float(model.box_shape[i]) for i in range(3))
    mesh_x, mesh_y, mesh_z = (int(model.mesh_shape[i]) for i in range(3))
    cell_val = float(model.cell_shape[0]) if hasattr(model, "cell_shape") else 0
    info = (
        f"Box: [{box_x:.0f}, {box_y:.0f}, {box_z:.0f}] Mpc/h | "
        f"Mesh: ({mesh_x}, {mesh_y}, {mesh_z}) | Cell: {cell_val:.1f} Mpc/h"
    )
    if cmb_enabled:
        # Curved-sky (HEALPix) run: report the actual sky coverage, not the
        # flat-sky box FOV (which is meaningless full-sky).
        cmb_mask = getattr(model, "cmb_mask", None)
        f_sky = float(np.mean(np.asarray(cmb_mask, dtype=float))) if cmb_mask is not None else 1.0
        info += f" | f_sky={f_sky:.3f} | nside={model.cmb_nside}"
    if cosmo_params:
        info += f" | Ω_m={cosmo_params.get('Omega_m', '?')}, σ₈={cosmo_params.get('sigma8', '?')}"
    plt.figtext(0.5, 0.02, info, ha="center", fontsize=8,
                bbox={"facecolor": "oldlace", "alpha": 0.5,
                      "edgecolor": "grey", "boxstyle": "round,pad=0.5"})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = output_dir / f"cl_spectra{suffix}_{timestamp}.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {outfile}")

    if show:
        plt.show()
    plt.close()

    return {"ell": ell, "cl_kk_pred": cl_kk_pred, "cl_kk_obs": cl_kk_obs,
            "cl_gg": cl_gg, "cl_kg": cl_kg}


def plot_cmb_noise_spectrum(model, output_dir, show=False):
    """
    Plot the input CMB lensing noise spectrum N_ell.

    Args:
        model (FieldLevelModel): Initialized model containing cmb_noise_nell.
        output_dir (Path or str): Output directory.
        show (bool): Whether to show the plot.
    """
    if not (model.cmb_enabled and model.cmb_noise_nell is not None):
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("VALIDATION: Plotting CMB noise spectrum...")
    try:
        # Resolve N_ell
        if isinstance(model.cmb_noise_nell, str):
            data = np.loadtxt(model.cmb_noise_nell)
            if data.ndim == 1:
                ell_in = np.arange(len(data))
                nell_in = data
            else:
                ell_in, nell_in = data[:, 0], data[:, 1]
        elif isinstance(model.cmb_noise_nell, dict):
            ell_in, nell_in = model.cmb_noise_nell["ell"], model.cmb_noise_nell["N_ell"]
        else:
            # Tuple or list
            ell_in, nell_in = model.cmb_noise_nell[0], model.cmb_noise_nell[1]

        # Apply scaling if present
        nell_scaled = nell_in * model.cmb_noise_scaling

        # Filter for log plot
        mask = (ell_in > 0) & (nell_scaled > 0)

        plt.figure(figsize=(8, 6))
        plot.plot_cl(ell_in[mask], nell_scaled[mask], log=True, ylabel=r"$N_\ell$")

        # Title reflects scaling
        if model.cmb_noise_scaling != 1.0:
            plt.title(f"CMB Lensing Noise Power Spectrum (scaled by {model.cmb_noise_scaling:.4g})")
        else:
            plt.title("CMB Lensing Noise Power Spectrum")

        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend(["$N_\\ell$ (used in run)"])

        outfile = output_dir / "cmb_noise_spectrum.png"
        plt.savefig(outfile, dpi=150)
        if show:
            plt.show()
        plt.close()
        print(f"✓ Saved: {outfile}")
    except Exception as e:
        print(f"⚠️  Could not plot N_ell: {e}")

def plot_warmup_diagnostics(model, state, init_params, truth, output_dir, show=False):
    """
    Plot warmup diagnostics: Power Spectrum, Transfer Function, and Coherence
    comparing initial condition (init) vs warmed-up state (warm).

    Args:
        model (FieldLevelModel): The model.
        state (MCMCState): The final warmup state.
        init_params (dict): The initial parameters before warmup (all chains).
        truth (dict): The truth dictionary containing 'init_mesh'.
        output_dir (Path or str): Output directory.
        show (bool): Whether to show the plot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("VALIDATION: Warmup Diagnostics (Power/Transfer/Coherence)")

    try:
        # Extract true init_mesh from truth
        init_mesh_true = truth.get('init_mesh')

        if init_mesh_true is not None:
            # Convert init_mesh from Fourier to real space for spectrum calculation
            init_mesh_true_real = jnp.fft.irfftn(init_mesh_true)

            # Compute true spectrum
            kpow_true = model.spectrum(init_mesh_true_real)

            # Compute power/transfer/coherence for init params (all chains)
            # Need to extract init_mesh from reparametrized params and convert to real
            def compute_kptc(params_dict):
                # Reparam to get init_mesh in Fourier (base) space
                params_base = model.reparam(params_dict)
                # Convert from Fourier to real space
                init_mesh_real = jnp.fft.irfftn(params_base['init_mesh'])
                return model.powtranscoh(init_mesh_true_real, init_mesh_real)

            # Vectorize over chains
            from jax import vmap
            kptcs_init = vmap(compute_kptc)(init_params)
            kptcs_warm = vmap(compute_kptc)(state.position)

            # Fiducial linear power spectrum
            cosmo_fid = get_cosmology(**model.loc_fid)
            # Use k bins from first chain result
            from desi_cmb_fli.bricks import lin_power_interp
            kptcs_warm_0 = jax.tree.map(lambda x: x[0], kptcs_warm)
            kpow_fid = (kptcs_warm_0[0], lin_power_interp(cosmo_fid)(kptcs_warm_0[0]))

            # Create diagnostic figure
            fig = plt.figure(figsize=(12, 4))
            fig.suptitle('Warmup Diagnostics: Initial Conditions Spectrum', fontsize=14)

            def plot_kptcs(kptcs, label=None):
                """Plot power/transfer/coherence (median of chains)."""
                kptcs_median = jax.tree.map(lambda x: jnp.median(x, 0), kptcs)
                plot.plot_powtranscoh(*kptcs_median, label=label)

            # Plot init and warmup
            plot_kptcs(kptcs_init, label='init')
            plot_kptcs(kptcs_warm, label='warm')

            # Add truth and fiducial to subplot 1 (power)
            plt.subplot(131)
            plot.plot_pow(*kpow_true, 'k:', label='true')
            plot.plot_pow(*kpow_fid, 'k--', alpha=0.5, label='fiducial')
            plt.legend()

            # Add reference lines to subplot 2 (transfer)
            plt.subplot(132)
            plt.axhline(1., linestyle=':', color='k', alpha=0.5)
            # Fiducial transfer (sqrt of power ratio)
            k_true, pow_true = kpow_true
            k_fid, pow_fid = kpow_fid
            transfer_fid = (pow_fid / pow_true)**0.5
            plot.plot_trans(k_true, transfer_fid, 'k--', alpha=0.5, label='fiducial')

            # Add reference line to subplot 3 (coherence)
            plt.subplot(133)
            # Plot mean of selection mask if available
            if hasattr(model, 'selec_mesh'):
                selec_mean = float(jnp.mean(model.selec_mesh))
                plt.axhline(selec_mean, linestyle=':', color='k', alpha=0.5,
                           label=f'selec_mesh mean={selec_mean:.3f}')
            plt.axhline(1.0, linestyle=':', color='k', alpha=0.2)

            plt.tight_layout()
            outfile = output_dir / 'init_warm.png'
            plt.savefig(outfile, dpi=150)
            if show:
                plt.show()
            plt.close()
            print(f"✓ Saved: {outfile}")
        else:
            # Abacus mode: no true IC available.
            # Panel 1: per-chain P(k) vs fiducial P_lin(k)
            # Panel 2: transfer √(P/P_lin) per chain
            # Panel 3: 2D coherence of projected IC vs galaxy obs in occupied z-slices
            from jax import vmap

            from desi_cmb_fli.bricks import lin_power_interp

            def get_init_mesh_real(params_dict):
                return jnp.fft.irfftn(model.reparam(params_dict)['init_mesh'])

            init_meshes_real = np.asarray(vmap(get_init_mesh_real)(init_params))
            warm_meshes_real = np.asarray(vmap(get_init_mesh_real)(state.position))
            n_chains = warm_meshes_real.shape[0]

            # Single reference = bare linear matter power P_lin(a=1). A physically correct
            # posterior linear field sits ON this line; far above = over-amplified.
            # See docs/pipeline.md "init_warm".
            cosmo_fid = get_cosmology(**model.loc_fid)
            k0, _ = model.spectrum(warm_meshes_real[0])
            k0 = np.asarray(k0)
            plin = np.asarray(lin_power_interp(cosmo_fid)(k0))

            fig = plt.figure(figsize=(12, 4))
            fig.suptitle('Warmup Diagnostics: Initial Conditions (Abacus mode)', fontsize=12)
            colors = plt.cm.tab10(np.linspace(0, 0.9, n_chains))

            for i in range(n_chains):
                _, pow_w = model.spectrum(warm_meshes_real[i])
                _, pow_i = model.spectrum(init_meshes_real[i])
                pow_w, pow_i = np.asarray(pow_w), np.asarray(pow_i)

                plt.subplot(131)
                plt.loglog(k0, pow_w, color=colors[i], label=f'chain {i}')
                plt.loglog(k0, pow_i, color=colors[i], linestyle='--', alpha=0.3)

                plt.subplot(132)
                plt.semilogx(k0, (pow_w / plin) ** 0.5, color=colors[i], label=f'chain {i}')

            plt.subplot(131)
            plt.loglog(k0, plin, 'k-', lw=2, label='P_lin (expected field)')
            plt.xlabel('k [h/Mpc]')
            plt.ylabel('P(k) [(Mpc/h)³]')
            plt.legend(fontsize=7)

            plt.subplot(132)
            plt.axhline(1.0, c='k', ls=':', alpha=0.7)
            plt.xlabel('k [h/Mpc]')
            plt.ylabel('√(P_warm / P_lin)   [1 = healthy]')
            plt.legend(fontsize=7)

            # 2D coherence of projected warm IC vs galaxy obs in occupied z-slices
            plt.subplot(133)
            mask3d = truth.get('gxy_occ_mask3d', None)
            obs_mesh = truth.get('obs', None)
            if mask3d is not None and obs_mesh is not None:
                occ_idx = np.where(np.any(np.asarray(mask3d), axis=(0, 1)))[0]
                gxy_proj = np.mean(np.asarray(obs_mesh)[:, :, occ_idx] - 1.0, axis=2)

                if model.cmb_enabled:
                    field_size_2d, _ = _infer_box_field_geometry(
                        model.box_shape,
                        model.mesh_shape,
                        chi_max=float(model.box_center[2]),
                    )
                else:
                    chi_ctr = float(model.box_center[2])
                    field_size_2d = float(2.0 * np.degrees(
                        np.arctan(float(model.box_shape[0]) / (2.0 * chi_ctr))))

                _, cl_gg = metrics.get_cl_2d(gxy_proj, field_size_deg=field_size_2d)
                cl_gg = np.asarray(cl_gg)

                # The inferred IC may live on an oversampled grid (init_oversamp); Fourier-crop
                # it to the final grid so the 2D projection matches the (final-grid) galaxy obs.
                final_shape = tuple(int(s) for s in model.mesh_shape)

                def _ic_to_final(ic_real):
                    if tuple(ic_real.shape) == final_shape:
                        return np.asarray(ic_real)
                    ic_k = chreshape(jnp.fft.rfftn(jnp.asarray(ic_real)), r2chshape(final_shape))
                    return np.asarray(jnp.fft.irfftn(ic_k, s=final_shape))

                for i in range(n_chains):
                    ic_proj = np.mean(_ic_to_final(warm_meshes_real[i])[:, :, occ_idx], axis=2)
                    ell_c, cl_cross = metrics.get_cl_2d(ic_proj, gxy_proj, field_size_deg=field_size_2d)
                    _, cl_ii = metrics.get_cl_2d(ic_proj, field_size_deg=field_size_2d)
                    coh = np.asarray(cl_cross) / np.sqrt(np.asarray(cl_ii) * cl_gg + 1e-30)
                    ell_b, coh_b, _ = bin_cl_log(np.asarray(ell_c), coh)
                    plt.semilogx(ell_b, coh_b, color=colors[i], label=f'chain {i}')

                plt.axhline(1.0, c='k', ls=':', alpha=0.7)
                plt.xlabel('ℓ')
                plt.ylabel('coherence')
                plt.title(f'IC × gxy obs\n({len(occ_idx)} occ. z-slices)')
                plt.legend(fontsize=7)
            else:
                plt.text(0.5, 0.5, 'Galaxy obs\nnot available',
                         ha='center', va='center', transform=plt.gca().transAxes)

            plt.tight_layout()
            outfile = output_dir / 'init_warm.png'
            plt.savefig(outfile, dpi=150)
            if show:
                plt.show()
            plt.close()
            print(f"✓ Saved: {outfile}")

    except Exception as e:
        print(f"⚠️  Warning: Could not generate warmup diagnostic plot: {e}")


def diagnose_freeze(model, init_params, output_dir, scan_range=8.0, n_scan=61):
    """Localize the source of MCLMC step_size->0. See docs/pipeline.md "freeze diagnostic"."""
    from pathlib import Path

    import jax
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    pos0 = {k: jax.device_put(np.asarray(v)[0]) for k, v in init_params.items()}
    logpdf = model.logpdf

    val, grad = jax.value_and_grad(logpdf)(pos0)
    print("\n" + "=" * 64)
    print("FREEZE DIAGNOSTIC  (chain 0, STEP-2 start: warmed mesh + fiducial scalars)")
    print("=" * 64)
    print(f"logpdf = {float(val):.6e}   finite={bool(np.isfinite(float(val)))}")
    print("per-parameter gradient (sampling space):")
    for k in sorted(grad):
        g = np.asarray(grad[k])
        print(f"  {k:12s} |grad|={np.linalg.norm(g):.4e}  max|g|={np.max(np.abs(g)):.4e}  "
              f"allfinite={bool(np.all(np.isfinite(g)))}  shape={tuple(g.shape)}")

    scalar_keys = [k for k in pos0 if k != "init_mesh_"]
    ts = np.linspace(-scan_range, scan_range, n_scan)
    ncol = max(len(scalar_keys), 1)
    fig, axes = plt.subplots(1, ncol, figsize=(4 * ncol, 4), squeeze=False)
    print("\n1D logpdf scans (perturb one scalar latent by t, others/mesh fixed):")
    for ax, k in zip(axes[0], scalar_keys, strict=False):
        lp = np.array([float(logpdf({**pos0, k: pos0[k] + t})) for t in ts])
        i0 = n_scan // 2
        win = lp[i0 - 1:i0 + 2]
        curv = ((win[2] - 2 * win[1] + win[0]) / (ts[1] - ts[0]) ** 2
                if np.all(np.isfinite(win)) else np.nan)
        finite = np.isfinite(lp)
        tfin = ts[finite]
        print(f"  {k:12s} nonfinite={int(np.sum(~finite))}/{n_scan}  curv@0={curv:.3e}  "
              f"finite-t in [{tfin.min() if tfin.size else np.nan:.2f},"
              f"{tfin.max() if tfin.size else np.nan:.2f}]")
        ax.plot(ts, lp - np.nanmax(lp))
        ax.set_title(k)
        ax.set_xlabel("t (latent perturbation)")
        ax.set_ylabel("logpdf - max")
        ax.set_ylim(-5e4, 5)
        ax.axvline(0, c="k", ls=":", alpha=0.4)
    plt.tight_layout()
    out = output_dir / "freeze_diagnostic.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"saved {out}")
    print("=" * 64 + "\n")
