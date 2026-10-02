
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from desi_cmb_fli import metrics, plot
from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import (
    bilinear_window,
    compute_shell_support_fractions,
    compute_theoretical_cl_gg,
    compute_theoretical_cl_kappa,
    compute_theoretical_cl_kappa_windowed,
    compute_theoretical_cl_kg,
    kappa_radial_window,
    project_mesh_to_healpix,
)
from desi_cmb_fli.metrics import bin_cl_log, get_cl_healpix, masked_healpix_to_full
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


def kappa_pred_on_obs_sphere(kappa_pred, model):
    """Bring a predict() kappa map from the projection sphere to the observable one.

    With cmb_lensing.proj_oversamp > 1 the model scatters kappa at nside * proj_oversamp; this
    band-limits it through the likelihood's own packing and resynthesises it at cmb_nside. Any
    other input is returned unchanged.
    """
    kp = np.asarray(kappa_pred)
    proj_nside = getattr(model, "cmb_proj_nside", None) if model is not None else None
    if (kp.ndim == 1 and proj_nside and proj_nside != model.cmb_nside
            and kp.size == 12 * int(proj_nside) ** 2):
        return np.asarray(model.unpack_to_map(model.pack_kappa_map(jnp.asarray(kp))))
    return kp


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


def _galaxy_healpix_proxy(model, chi_range):
    """Stand-in model exposing the HEALPix geometry when the CMB is disabled.

    Curved-sky lightcone galaxies still need a spherical projection: the flat-sky
    fallback averages along the box z-axis, which for a centred observer runs
    through the empty interior and merges opposite sides of the sky.  nside is
    matched to the cell size seen at the mid-survey distance.
    """
    from desi_cmb_fli.cmb_lensing import compute_healpix_mask

    chi_mid = 0.5 * (float(chi_range[0]) + float(chi_range[1]))
    cell = float(np.min(np.asarray(model.box_shape, dtype=float) / np.asarray(model.mesh_shape)))
    exponent = int(np.clip(round(np.log2(np.sqrt(np.pi / 3.0) * chi_mid / cell)), 2, 9))
    nside = 2 ** exponent
    attrs = {
        "cmb_nside": nside,
        "cmb_mask": np.asarray(
            compute_healpix_mask(model.observer_position, model.box_shape, nside), dtype=bool
        ),
        "box_shape": np.asarray(model.box_shape, dtype=float),
        "observer_position": np.asarray(model.observer_position, dtype=float),
        "chi_boundary": float(getattr(model, "chi_boundary", model.box_shape[2])),
    }
    return type("ValidationModelProxy", (), attrs)(), 3 * nside - 1


def _truth_as_overdensity(truth, model=None):
    """Return ``truth`` with 'obs' converted from galaxy counts to 1 + delta_g.

    The likelihood works on counts; every plot below reads an overdensity. The mean density
    n̄·S comes from truth.npz when it carries the model state, otherwise from ``model``.

    Idempotent: converting twice would divide by n̄·S twice and silently shrink every spectrum,
    so the result is tagged and a second call is a no-op.
    """
    if "obs" not in truth or "_obs_overdensity" in truth:
        return truth

    def _get(key):
        v = truth[key] if key in truth else None
        return getattr(model, key, None) if v is None else v

    gxy_count = _get("gxy_count")
    if gxy_count is None:
        return truth
    selec = _get("selec_mesh")
    nbar = float(gxy_count) * (1.0 if selec is None else np.asarray(selec, dtype=float))
    dens = np.asarray(truth["obs"], dtype=float) / np.maximum(nbar, 1e-10)
    mask = _get("gxy_occ_mask3d")
    if mask is not None:
        dens = np.where(np.asarray(mask).astype(bool), dens, 1.0)
    return dict(truth) | {"obs": dens, "_obs_overdensity": True}


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
    model=None,
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
        model (FieldLevelModel): needed to bring a proj_oversamp > 1 kappa_pred to cmb_nside.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    truth = _truth_as_overdensity(truth)  # counts -> 1 + delta_g

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
        _obs_arr = kappa_pred_on_obs_sphere(truth["kappa_obs"], model)  # Abacus: projection sphere
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
            _pred_arr = kappa_pred_on_obs_sphere(truth["kappa_pred"], model)
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


def _expected_hp_map(truth, model, nside, chi_max):
    """Unclustered expectation of the galaxy angular map, and its radial kernel.

    The survey selection (x the survey mask) is integrated along each pixel centre with the r^2
    volume weight; the same samples summed over the sky give dN/dchi. The expectation is smooth,
    so casting rays through the selection grid costs no resolution.
    """
    import healpy as hp
    from scipy.ndimage import map_coordinates

    box = np.asarray(model.box_shape, dtype=float)
    obs_pos = np.asarray(model.observer_position, dtype=float)
    occ = truth.get("gxy_occ_mask3d", getattr(model, "gxy_occ_mask3d", None))
    selp = truth.get("selec_paint", getattr(model, "selec_paint", None))
    occ = None if occ is None else np.asarray(occ, dtype=float)
    selp = None if selp is None else np.asarray(selp, dtype=float)

    npix = hp.nside2npix(nside)
    nvec = np.array(hp.pix2vec(nside, np.arange(npix))).T
    step = float(np.min(box / np.asarray(model.paint_shape))) / 2.0
    r = np.arange(step / 2.0, chi_max, step)
    E, nz = np.zeros(npix), np.zeros(r.size)
    for i, ri in enumerate(r):
        x = obs_pos + ri * nvec
        w = np.ones(npix)
        if selp is not None:
            w *= map_coordinates(selp, (x / (box / selp.shape)).T, order=1, mode="grid-wrap")
        if occ is not None:
            w *= map_coordinates(occ, (x / (box / occ.shape)).T, order=0, mode="grid-wrap")
        E += w * ri**2 * step
        nz[i] = w.sum() * ri**2
    return E, (r, nz)


def _particle_hp_counts(truth, model, nside, chi_max):
    """Galaxy counts per pixel from the model's particles: bias weight x selection x survey mask."""
    from scipy.ndimage import map_coordinates

    from desi_cmb_fli.cmb_lensing import healpix_counts

    box = np.asarray(model.box_shape, dtype=float)
    x = np.asarray(truth["rsd_pos"], dtype=float) * box / np.asarray(model.evol_shape)
    w = np.asarray(truth["gxy_weights"], dtype=float).ravel()
    for key, order in (("selec_paint", 1), ("gxy_occ_mask3d", 0)):
        grid = truth.get(key, getattr(model, key, None))
        if grid is not None:
            grid = np.asarray(grid, dtype=float)
            w = w * map_coordinates(grid, (x / (box / grid.shape)).T, order=order, mode="grid-wrap")
    rel = x - np.asarray(model.observer_position, dtype=float)
    r = np.linalg.norm(rel, axis=1)
    keep = (r > 0) & (r <= chi_max)
    return healpix_counts(rel[keep], nside, weights=w[keep], bilinear=True)


def galaxy_healpix_delta(truth, model, nside, chi_max):
    """Galaxy overdensity on HEALPix pixels, built from galaxies rather than from the mesh.

    Data: the catalogue count map the Abacus loader stores (``gxy_hp_counts``), summed down to
    ``nside``. Model: its particles weighted like the painted galaxy field, spread bilinearly over
    the pixels as the Born projector spreads them for kappa. Both are divided by the unclustered
    expectation of ``_expected_hp_map``. No mesh interpolation enters, so the only window is the
    HEALPix pixel. Returns None when neither source is in ``truth``.
    """
    import healpy as hp

    if "gxy_hp_counts" in truth:
        G = hp.ud_grade(np.asarray(truth["gxy_hp_counts"], dtype=float), nside, power=-2)
        poisson = True
    elif "rsd_pos" in truth and "gxy_weights" in truth:
        G = _particle_hp_counts(truth, model, nside, chi_max)
        poisson = False  # the particles are a displaced lattice, not a Poisson sample
    elif "obs" in truth:
        # Fallback (kaiser evolution, truths without particles): the mesh nodes as points. The
        # cell window is then left in the map.
        from desi_cmb_fli.cmb_lensing import healpix_counts

        print("  [validation] galaxy map from mesh nodes: cell window not removed")
        obs = np.asarray(truth["obs"], dtype=float)
        w = obs.copy()
        for key in ("selec_mesh", "gxy_occ_mask3d"):
            grid = truth.get(key, getattr(model, key, None))
            if grid is not None:
                w = w * np.asarray(grid, dtype=float)
        cell = np.asarray(model.box_shape, dtype=float) / np.asarray(obs.shape)
        rel = np.indices(obs.shape).reshape(3, -1).T * cell - np.asarray(model.observer_position)
        r = np.linalg.norm(rel, axis=1)
        keep = (r > 0) & (r <= chi_max)
        G = healpix_counts(rel[keep], nside, weights=w.ravel()[keep], bilinear=True)
        poisson = False
    else:
        return None
    if not poisson:
        # Bilinear spreading gives some pixels more weight than others (the polar caps 17 %):
        # divide it out, as the Born projector does, so that the expectation below applies.
        from desi_cmb_fli.cmb_lensing import bilinear_weight_norm

        G = G / bilinear_weight_norm(nside)
    E, kernel = _expected_hp_map(truth, model, nside, chi_max)
    mask = E > 1e-6 * E.max()
    g_exp = E / E[mask].sum() * G[mask].sum()
    shot = 4 * np.pi / E.size * float(np.mean(1.0 / g_exp[mask])) if poisson else 0.0
    # A catalogue histogram carries the pixel window; particles and mesh nodes are spread by the
    # bilinear kernel, whose window is not the pixel one.
    return {"delta_masked": G[mask] / g_exp[mask] - 1.0, "mask_full": mask,
            "kernel": kernel, "shot": shot, "bilinear": not poisson}


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
    truth = _truth_as_overdensity(truth, model)  # counts -> 1 + delta_g

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
    # The Abacus loader gives it as a map on the projection sphere, like kappa_pred.
    if cmb_enabled and has_kappa_obs:
        _ko = kappa_pred_on_obs_sphere(truth["kappa_obs"], model)
        truth = {**truth, "kappa_obs": _ko}
        _pix_sizes = {npix_hp} | ({int(cmb_mask_eff.sum())} if cmb_mask_eff is not None else set())
        if _ko.ndim == 1 and _ko.size not in _pix_sizes:
            truth = {
                **truth,
                "kappa_obs": np.asarray(model.unpack_kappa_obs_to_map(jnp.asarray(_ko))),
            }

    if cmb_enabled and has_kappa_pred:
        truth = {**truth, "kappa_pred": kappa_pred_on_obs_sphere(truth["kappa_pred"], model)}

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

    # Galaxy spectra: map from the galaxies themselves (catalogue or particles), pixel window
    # removed, shot noise subtracted, so it compares directly with Limber.
    f_sky_gxy, gxy_kernel, gxy_shot = 1.0, None, None
    if has_galaxies:
        import healpy as hp

        gxy_model, gxy_lmax = None, lmax_hp
        if cmb_enabled and np.ndim(truth.get("kappa_pred", truth.get("kappa_obs"))) == 1:
            gxy_model = model
        elif getattr(model, "curved_sky", False) and truth.get("chi_range_gxy") is not None:
            gxy_model, gxy_lmax = _galaxy_healpix_proxy(model, truth["chi_range_gxy"])

        gmap = None
        if gxy_model is not None:
            nside_g = 2 * int(gxy_model.cmb_nside)
            chi_max_g = float(getattr(model, "chi_boundary", np.min(model.box_shape) / 2))
            gmap = galaxy_healpix_delta(truth, model, nside_g, chi_max_g)
        if gmap is not None:
            wpix = (bilinear_window(nside_g, gxy_lmax) if gmap.get("bilinear")
                    else hp.pixwin(nside_g, lmax=gxy_lmax))
            ell_tmp, cl_gg, info_gg = get_cl_healpix(
                gmap["delta_masked"], gmap["mask_full"], lmax=gxy_lmax)
            gxy_shot = gmap["shot"]
            wpix = wpix[np.asarray(ell_tmp, dtype=int)]
            cl_gg = (np.asarray(cl_gg) - gxy_shot) / wpix**2
            f_sky_gxy = float(info_gg["norm"])
            gxy_kernel = gmap["kernel"]
            if ell is None:
                ell = np.asarray(ell_tmp)
            cl_mode = "healpix"

            if cmb_enabled and has_kappa_pred and np.ndim(truth["kappa_pred"]) == 1:
                # kappa is band-limited to lmax, so moving it to the galaxy nside loses nothing.
                kmask = np.asarray(model.cmb_mask, dtype=bool)
                kfull = np.asarray(truth["kappa_pred"], dtype=float)
                if kfull.size != kmask.size:
                    kfull = masked_healpix_to_full(kfull, kmask, fill_value=0.0)
                k_up = hp.alm2map(hp.map2alm(kfull * kmask, lmax=lmax_hp), nside_g, lmax=lmax_hp)
                kmask_up = hp.ud_grade(kmask.astype(float), nside_g) > 0.5
                ell_kg, cl_kg, _ = get_cl_healpix(
                    k_up[kmask_up], kmask_up, gmap["delta_masked"], gmap["mask_full"],
                    lmax=lmax_hp)
                cl_kg = np.asarray(cl_kg) / wpix[: len(ell_kg)]
        elif gxy_model is None:
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
        else:
            print("  [validation] No galaxy catalogue map or particles in truth: galaxy spectra skipped.")

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
            "cl_mode": cl_mode, "gxy_kernel": gxy_kernel, "gxy_shot": gxy_shot,
            # binned versions
            "ell_b": ell_b, "cl_kk_pred_b": cl_kk_pred_b, "cl_kk_obs_b": cl_kk_obs_b,
            "cl_gg_b": cl_gg_b, "cl_kg_b": cl_kg_b, "n_modes_b": n_modes_b}


def conditioning_params(model, *param_dicts):
    """Fiducial value for every latent, overridden by the given config truth dicts.

    ``predict`` draws any latent missing from ``samples`` straight from its prior, so a
    partial dict silently randomises the model: with ``png_type: fNL_bias`` that means
    fNL_bp and fNL_bpd at scale 1e4, whose 1/k^2 scale-dependent bias swamps P(k).
    """
    samples = dict(model.loc_fid)
    given = set()
    for d in param_dicts:
        for k, v in (d or {}).items():
            if k in samples:
                samples[k] = v
                given.add(k)
    filled = sorted(set(samples) - given)
    if filled:
        print(f"  [conditioning] not in config truth, held at loc_fid: "
              f"{ {k: samples[k] for k in filled} }")
    return samples


def compute_cl_theory(model, cosmo_val, ell_theory, bE=2.0, gxy_kernel=None,
                      has_galaxies=True, observation_mode="closure"):
    """The one Limber curve per spectrum that the C_ell diagnostic compares with.

    ``measure_spectra`` returns noiseless maps with the galaxy pixel window removed and the galaxy
    shot noise subtracted, so no noise term enters here:

    * kappa-kappa, closure: the model's Born shells, chi_min to chi_boundary, with k_perp below the
      init-grid Nyquist -- the inferred linear field has no power above it.
    * kappa-kappa, abacus: the line of sight from ``low_z_matter_start`` to ``chi_high_z_max`` at
      full resolution, times
      the pixel window of the ud_grade that brings the simulation map to ``cmb_nside``.
    * gg and kappa-g: the survey's dN/dchi (``gxy_kernel``), constant Eulerian bias ``bE``,
      nonlinear P(k); the same k cut in closure, none on data. In closure kappa-g also carries the
      radial window of the Born shells (``kappa_radial_window``): where ``chi_min`` cuts into the
      galaxies, the model map does not hold their matter.
    """

    ell_j = jnp.asarray(ell_theory, dtype=float)
    k_cut = None
    if observation_mode == "closure":
        init_cell = np.asarray(model.box_shape, dtype=float) / np.asarray(model.init_shape)
        k_cut = float(np.pi / np.max(init_cell))
    out = {"cl_kk": None, "cl_gg": None, "cl_kg": None, "k_cut": k_cut}

    kappa_window, w_kappa = None, None
    if model.cmb_enabled:
        z_source = model.cmb_z_source
        # Model and (since the loader resamples it through the same kernel) Abacus kappa both
        # carry the Born projector's bilinear window at the projection nside.
        w_proj = bilinear_window(int(model.cmb_proj_nside), int(np.ceil(np.max(ell_theory))))
        w_kappa = np.interp(ell_theory, np.arange(w_proj.size), w_proj)
        if observation_mode == "closure":
            support = compute_shell_support_fractions(
                model.observer_position, model.box_shape, model.cmb_nside,
                model.cmb_r_shells, model.cmb_d_r, final_mask=model.cmb_mask,
            )
            kappa_window = kappa_radial_window(model.cmb_r_shells, model.cmb_d_r,
                                               model.cmb_shell_weights, support)
            out["cl_kk"] = np.asarray(compute_theoretical_cl_kappa_windowed(
                cosmo_val, ell_j, model.cmb_r_shells, model.cmb_a_shells, model.cmb_d_r,
                z_source, shell_weights=support, k_nyq=k_cut,
            )) * w_kappa**2
        else:
            out["cl_kk"] = np.asarray(compute_theoretical_cl_kappa(
                cosmo_val, ell_j, model.low_z_matter_start, float(model.chi_high_z_max), z_source,
            )) * w_kappa**2

    if has_galaxies and gxy_kernel is not None:
        r, nz = (np.asarray(a, dtype=float) for a in gxy_kernel)
        occupied = r[nz > 0]
        chi0, chi1 = float(occupied.min()), float(occupied.max())
        out["cl_gg"] = np.asarray(compute_theoretical_cl_gg(
            cosmo_val, ell_j, chi0, chi1, bE, n_steps=400, k_nyq=k_cut, nz=(r, nz)))
        if model.cmb_enabled:
            out["cl_kg"] = np.asarray(compute_theoretical_cl_kg(
                cosmo_val, ell_j, chi0, chi1, model.cmb_z_source, bE, n_steps=400,
                k_nyq=k_cut, nz=(r, nz), kappa_window=kappa_window)) * w_kappa
    return out


def plot_cl_figure(spectra_list, model, cosmo_params, observation_mode, outfile, show=False):
    """Binned C_ell of one or several realisations against the Limber curve of each spectrum.

    Points are the mean over realisations with the error of that mean (none for a single
    realisation). Also prints measured/theory per bin. Returns the theory dict.
    """
    sp0 = spectra_list[0]
    ell_b = np.asarray(sp0["ell_b"])
    ell_theory = np.geomspace(2.0, float(np.max(sp0["ell"])), 100)
    b1 = (cosmo_params or {}).get("b1", model.loc_fid.get("b1", 1.0))
    cosmo = {k: v for k, v in (cosmo_params or model.loc_fid).items() if k in ("Omega_m", "sigma8")}
    theory = compute_cl_theory(
        model, get_cosmology(**cosmo), ell_theory, bE=1.0 + float(b1),
        gxy_kernel=sp0.get("gxy_kernel"), has_galaxies=sp0.get("cl_gg") is not None,
        observation_mode=observation_mode,
    )

    n = len(spectra_list)

    def stack(key):
        arrs = [np.asarray(sp[key]) for sp in spectra_list if sp.get(key) is not None]
        if not arrs:
            return None, None
        a = np.array(arrs)
        return a.mean(0), (a.std(0, ddof=1) / np.sqrt(n) if n > 1 else None)

    src = "model" if observation_mode == "closure" else "Abacus"
    series = [
        ("kk", "cl_kk_pred_b", "cl_kk", "C0", rf"$C_\ell^{{\kappa\kappa}}$ ({src})"),
        ("gal", "cl_gg_b", "cl_gg", "C2", rf"$C_\ell^{{gg}}$ ({src})"),
        ("gal", "cl_kg_b", "cl_kg", "C3", rf"$C_\ell^{{\kappa g}}$ ({src})"),
    ]
    panels = [p for p in ("kk", "gal")
              if any(pan == p and stack(k)[0] is not None and theory[t] is not None
                     for pan, k, t, _, _ in series)]
    if not panels:
        print("  Nothing to plot.")
        return theory
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5), squeeze=False)
    ax_of = dict(zip(panels, axes[0], strict=True))
    vb = (ell_b >= 2) & (ell_b <= ell_theory.max())

    print(f"  measured / Limber per bin ({n} realisation(s), {src}):")
    for pan, key, tkey, color, label in series:
        mean, err = stack(key)
        if mean is None or theory[tkey] is None:
            continue
        ax = ax_of[pan]
        ax.errorbar(ell_b[vb], mean[vb], yerr=None if err is None else err[vb], fmt="o",
                    color=color, ms=4, capsize=2, label=label)
        ax.plot(ell_theory, theory[tkey], color=color, lw=1.5)
        th_b = np.interp(ell_b[vb], ell_theory, theory[tkey])
        print(f"    {tkey[3:]:>3s}: " + " ".join(f"{x:.2f}" for x in mean[vb] / th_b))

    for pan, ax in ax_of.items():
        ax.plot([], [], color="k", lw=1.5, label="Limber")
        ax.set(xscale="log", yscale="log", xlabel=r"$\ell$", ylabel=r"$C_\ell$",
               title=r"$\kappa\kappa$" if pan == "kk" else r"galaxies: $gg$ and $\kappa g$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
    nside = f", nside {model.cmb_nside}" if model.cmb_enabled else ""
    fig.suptitle(f"{src}, box {model.box_shape[0]:.0f} Mpc/h, cell {float(model.cell_shape[0]):.1f}"
                 f" Mpc/h{nside}, {n} realisation(s)", fontsize=10)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {outfile}")
    if show:
        plt.show()
    plt.close(fig)
    return theory


def plot_spectra(truth, model, output_dir, cosmo_params=None, model_config=None,
                 observation_mode="closure", show=False, suffix=""):
    """Startup C_ell check of one truth: see ``plot_cl_figure``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("VALIDATION: Power Spectra")
    print("=" * 80)
    spectra = measure_spectra(truth, model, model_config)
    if spectra["ell"] is None:
        print("  No maps to measure spectra on, skipping.")
        return {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_cl_figure([spectra], model, cosmo_params, observation_mode,
                   output_dir / f"cl_spectra{suffix}_{timestamp}.png", show=show)
    return spectra


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

def plot_warmup_diagnostics(model, state, init_params, truth, output_dir, show=False,
                            fixed_latents=None):
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
        fixed_latents (dict): The scalar latents the run holds fixed (``mcmc.fixed_params``),
            which the sampler state does not carry; without them ``reparam`` falls back on the
            default cosmology.
    """
    fixed_latents = dict(fixed_latents or {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    truth = _truth_as_overdensity(truth, model)  # counts -> 1 + delta_g

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
                params_base = model.reparam({**params_dict, **fixed_latents})
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
            kpow_fid = (kptcs_warm_0[0],
                        lin_power_interp(cosmo_fid, pk_ratio=getattr(model, "pk_ratio", None))(kptcs_warm_0[0]))

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
                return jnp.fft.irfftn(model.reparam({**params_dict, **fixed_latents})['init_mesh'])

            init_meshes_real = np.asarray(vmap(get_init_mesh_real)(init_params))
            warm_meshes_real = np.asarray(vmap(get_init_mesh_real)(state.position))
            n_chains = warm_meshes_real.shape[0]

            # Single reference = bare linear matter power P_lin(a=1). A physically correct
            # posterior linear field sits ON this line; far above = over-amplified.
            # See docs/pipeline.md "init_warm".
            cosmo_fid = get_cosmology(**model.loc_fid)
            k0, _ = model.spectrum(warm_meshes_real[0])
            k0 = np.asarray(k0)
            plin = np.asarray(lin_power_interp(cosmo_fid, pk_ratio=getattr(model, "pk_ratio", None))(k0))

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
