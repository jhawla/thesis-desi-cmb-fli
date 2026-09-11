"""
CMB Lensing Module

Pipeline overview
-----------------
LPT -> particle positions (cell units)
  -> single-pass bilinear scatter onto (n_shells, npix) HEALPix rows
     -> delta on HEALPix mask (normalised on footprint)
        -> convergence_Born_spherical (Born integral)
           -> kappa_hp : (n_pix_mask,)

Observation:
  AbacusSummit kappa_*.asdf -> hp.ud_grade -> full-sky map + noise on cmb_mask

Likelihood:
    Native dist.Normal over the packed masked pseudo-a_lm (see FieldLevelModel).
"""

import os
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import jax_cosmo as jc
import jax_cosmo.constants as constants
import numpy as np

NYQUIST_FRACTION = 0.5

# Where the reduced randoms meshes are cached. Set abacus_galaxy.randoms_cache_dir
# in the config to override, or to null to disable caching.
_DEFAULT_RANDOMS_CACHE_DIR = (
    Path(os.environ.get("SCRATCH", tempfile.gettempdir())) / "desi_cmb_fli_cache"
)


# =========================================================================
# Geometry helpers
# =========================================================================


def project_mesh_to_healpix(mesh, box_shape, observer_position, nside, mask,
                            chi_max=None, order=1):
    """Integrate a 3D Cartesian mesh along the line of sight onto masked HEALPix pixels.

    Ray-casts radially outward from ``observer_position`` through each unmasked
    pixel, sampling ``mesh`` with ``order``-th order interpolation and
    accumulating ``value * step_size`` (step = smallest cell size). Works for any
    observer (e.g. a centre observer gives full sky). Returns the projected
    column for the masked pixels only, in ``np.where(mask)[0]`` order.

    Single source of truth for the galaxy/mesh -> HEALPix projection used by both
    validation.py (diagnostic spectra & maps) and plot_2D_maps.py.
    """
    import healpy as hp
    from scipy.ndimage import map_coordinates

    mesh = np.asarray(mesh, dtype=float)
    box_shape = np.asarray(box_shape, dtype=float)
    observer_position = np.asarray(observer_position, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    pix_idx = np.where(mask)[0]
    theta, phi = hp.pix2ang(nside, pix_idx)
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)

    mesh_shape = np.array(mesh.shape)
    dx = box_shape[0] / mesh_shape[0]
    dy = box_shape[1] / mesh_shape[1]
    dz = box_shape[2] / mesh_shape[2]

    r_max = float(chi_max) if chi_max is not None else float(np.linalg.norm(box_shape))
    step_size = float(min(dx, dy, dz))
    r_steps = np.arange(step_size / 2.0, r_max, step_size)

    proj_map = np.zeros(len(pix_idx))
    for r in r_steps:
        x = observer_position[0] + r * nx
        y = observer_position[1] + r * ny
        z = observer_position[2] + r * nz
        valid = (
            (x >= 0) & (x < box_shape[0])
            & (y >= 0) & (y < box_shape[1])
            & (z >= 0) & (z < box_shape[2])
        )
        if not np.any(valid):
            continue
        coords = np.stack(
            [x[valid] / dx - 0.5, y[valid] / dy - 0.5, z[valid] / dz - 0.5], axis=0
        )
        vals = map_coordinates(mesh, coords, order=order, mode="constant", cval=0.0)
        proj_map[valid] += vals * step_size
    return proj_map


def compute_healpix_mask(observer_position_mpc, box_shape, nside):
    """Return a boolean HEALPix mask of pixels whose LOS intersects the box.

    The box occupies [0, Lx] x [0, Ly] x [0, Lz] in Mpc/h.  A LOS ray
    from ``observer_position_mpc`` in direction n_hat(theta, phi) is accepted
    when it intersects the box interior for any positive distance.

    Parameters
    ----------
    observer_position_mpc : array-like (3,)
        Observer position in Mpc/h -- typically (Lx/2, Ly/2, 0).
    box_shape : array-like (3,)
        Box size (Lx, Ly, Lz) in Mpc/h.
    nside : int
        HEALPix nside.

    Returns
    -------
    mask : np.ndarray, shape (npix_full,)
        Boolean mask; True = pixel inside the box footprint.
    """

    t_enter, t_exit = _box_ray_intervals(observer_position_mpc, box_shape, nside)

    mask = (t_exit > t_enter) & (t_exit > 0.0)
    return mask


def _box_ray_intervals(observer_position_mpc, box_shape, nside):
    """Return entry/exit distances for each HEALPix ray through the box."""
    import healpy as hp

    npix = hp.nside2npix(nside)
    obs = np.asarray(observer_position_mpc, dtype=float)
    Lx, Ly, Lz = float(box_shape[0]), float(box_shape[1]), float(box_shape[2])

    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=False)
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)

    eps = 1e-12

    def slab_t(n_i, obs_i, L_i):
        t_near = np.where(
            np.abs(n_i) > eps,
            (0.0 - obs_i) / np.where(np.abs(n_i) > eps, n_i, eps),
            np.where(obs_i >= 0.0, -np.inf, np.inf),
        )
        t_far = np.where(
            np.abs(n_i) > eps,
            (L_i - obs_i) / np.where(np.abs(n_i) > eps, n_i, eps),
            np.where(obs_i <= L_i, np.inf, -np.inf),
        )
        return np.minimum(t_near, t_far), np.maximum(t_near, t_far)

    tx0, tx1 = slab_t(nx, obs[0], Lx)
    ty0, ty1 = slab_t(ny, obs[1], Ly)
    tz0, tz1 = slab_t(nz, obs[2], Lz)

    t_enter = np.maximum(np.maximum(tx0, ty0), tz0)
    t_exit = np.minimum(np.minimum(tx1, ty1), tz1)
    return t_enter, t_exit


def compute_shell_support_fractions(observer_position_mpc, box_shape, nside, r_shells, d_r, final_mask=None):
    """Return the relative angular support of each radial shell inside the box.

    The returned fractions are normalised by the mean of ``final_mask`` so they
    can be used with pseudo-C_ell measurements that divide by ``f_sky``.
    """
    t_enter, t_exit = _box_ray_intervals(observer_position_mpc, box_shape, nside)
    if final_mask is None:
        final_mask = compute_healpix_mask(observer_position_mpc, box_shape, nside)
    final_mask = np.asarray(final_mask, dtype=bool)
    f_final = max(float(np.mean(final_mask.astype(float))), 1e-12)

    r_shells = np.asarray(r_shells, dtype=float)
    if np.ndim(d_r) == 0:
        d_r_arr = np.full(r_shells.shape, float(d_r), dtype=float)
    else:
        d_r_arr = np.asarray(d_r, dtype=float)

    weights = np.empty_like(r_shells, dtype=float)
    for i, (chi_i, dr_i) in enumerate(zip(r_shells, d_r_arr, strict=False)):
        r_min = chi_i - 0.5 * dr_i
        r_max = chi_i + 0.5 * dr_i
        # Same criterion as convergence_Born_spherical: the shell segment must be
        # fully inside the box along the ray (not merely intersecting it).
        shell_mask = final_mask & (t_enter <= r_min) & (t_exit >= r_max)
        weights[i] = float(np.mean(shell_mask.astype(float)) / f_final)
    return weights


def max_box_radius(observer_position_mpc, box_shape):
    """Maximum distance from the observer to any corner of the simulation box."""
    obs = np.asarray(observer_position_mpc, dtype=float)
    Lx, Ly, Lz = map(float, box_shape)
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [Lx, 0.0, 0.0],
            [0.0, Ly, 0.0],
            [0.0, 0.0, Lz],
            [Lx, Ly, 0.0],
            [Lx, 0.0, Lz],
            [0.0, Ly, Lz],
            [Lx, Ly, Lz],
        ],
        dtype=float,
    )
    return float(np.max(np.linalg.norm(corners - obs[None, :], axis=1)))


def load_healpix_mask(mask_spec, nside):
    """Load an optional HEALPix mask and convert it to a boolean map at ``nside``."""
    import healpy as hp

    if mask_spec is None:
        return np.ones(hp.nside2npix(nside), dtype=bool)

    if isinstance(mask_spec, list | tuple | np.ndarray):
        mask = np.asarray(mask_spec, dtype=float)
    else:
        mask_path = str(mask_spec)
        if mask_path.endswith(".npy"):
            mask = np.load(mask_path)
        else:
            mask = hp.read_map(mask_path, dtype=float, verbose=False)

    if hp.get_nside(mask) != nside:
        mask = hp.ud_grade(mask, nside, order_in="RING", order_out="RING")
    return np.asarray(mask) > 0.5


# =========================================================================
# Lensing kernel
# =========================================================================


def lensing_kernel(cosmo, chi, a, chi_source):
    """Born/Limber lensing kernel W_kappa(chi) for a source plane at chi_source."""
    prefactor = 1.5 * cosmo.Omega_m * (constants.H0 / constants.c) ** 2
    geometry = jnp.clip((chi_source - chi) / chi_source, 0.0, jnp.inf)
    return prefactor * (chi / a) * geometry


# =========================================================================
# Spherical Born convergence
# =========================================================================


def linear_shell_volumes(r_shells, d_r, npix):
    """Per-pixel volume of each shell under the 'linear' shell weights.

    The weight of shell ``i`` is the tent ``max(0, 1 - |r - c_i| / d_r)``: it rises from
    zero at ``c_i - d_r`` and falls back to zero at ``c_i + d_r``, so a particle enters
    and leaves the integration continuously at both ends of the radial range. Shell ``i``
    sees the volume ``Omega_pix * int w_i(r) r^2 dr``, computed here in closed form.
    """
    r_shells = np.asarray(r_shells, dtype=float)

    def piece(a, b, alpha, beta):
        a, b = max(a, 0.0), max(b, 0.0)
        return alpha * (b**3 - a**3) / 3.0 + beta * (b**4 - a**4) / 4.0

    vol = np.array([
        piece(c - d_r, c, -(c - d_r) / d_r, 1.0 / d_r)
        + piece(c, c + d_r, (c + d_r) / d_r, -1.0 / d_r)
        for c in r_shells
    ])
    return (4.0 * np.pi / npix) * vol


def convergence_Born_spherical(
    cosmo,
    pos,
    box_shape,
    mesh_shape,
    observer_pos_mpc,
    r_shells,
    a_shells,
    d_r,
    nside,
    mask,
    z_source,
    t_enter=None,
    t_exit=None,
    return_full=False,
    shell_weights="nearest",
):
    """Compute CMB convergence on a HEALPix mask via the Born approximation.

    Each particle is assigned radially to its shell(s) and bilinearly scattered
    onto the HEALPix row(s) in a single pass, giving per-shell densities
    ``rho``; then ``kappa = sum_i (rho_i / n_bar - 1) * d_r_i * W_kappa(chi_i,
    a_i)`` over the shells whose segment lies fully inside the box along each
    ray. Shells must tile the radial range exactly. See docs/pipeline.md
    "Single-pass shell scatter" and "Shell weights" (2.6).

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
    pos : array, shape (N, 3)
        Particle positions in simulation mesh cell units (0 ... mesh_shape-1).
    box_shape : array-like (3,)
        Box size in Mpc/h.
    mesh_shape : tuple of int (3,)
        Simulation mesh shape.
    observer_pos_mpc : array-like (3,)
        Observer position in Mpc/h.
    r_shells : array (n_shells,)
        Comoving distance at the centre of each radial shell (Mpc/h).
    a_shells : array (n_shells,)
        Scale factor at the centre of each radial shell.
    d_r : float or array (n_shells,)
        Radial width of each shell (Mpc/h).
    nside : int
        HEALPix nside for the output map.
    mask : array (npix_full,) bool
        True for pixels inside the physical support of the simulation volume.
    z_source : float
        CMB source redshift.
    return_full : bool, optional
        If True, return a full-sky HEALPix map with zeros outside ``mask``.
        If False, return only the active pixels.
    shell_weights : {'nearest', 'linear'}
        'nearest': each particle goes entirely to the shell containing it.
        'linear': its weight is split linearly between the two shells whose
        centres bracket its radius, ramping up from zero over the half shell
        below the first edge (uniform shells only), so ``kappa`` is continuous
        in the particle radii, inner edge included.

    Returns
    -------
    kappa_hp : array
        Full-sky HEALPix map if ``return_full`` is True, otherwise the masked pixels.
    """
    import healpy as hp
    import jax_healpy as jhp

    chi_s = jc.background.radial_comoving_distance(cosmo, 1.0 / (1.0 + z_source))[0]
    observer_pos_mpc = jnp.asarray(observer_pos_mpc, dtype=float)
    box_size_jnp = jnp.asarray(box_shape, dtype=float)
    mesh_shape_tuple = tuple(int(x) for x in mesh_shape)
    n_shells = len(r_shells)
    npix_full = hp.nside2npix(nside)
    if mask is None:
        mask = np.ones(npix_full, dtype=bool)
    mask_jnp = jnp.asarray(mask)
    # Must be JAX arrays so they can be indexed by a JAX tracer inside lax.scan
    r_shells_jnp = jnp.asarray(r_shells)
    a_shells_jnp = jnp.asarray(a_shells)

    if not hasattr(d_r, "__len__"):
        d_r_arr = jnp.full(n_shells, d_r)
        d_r_np = np.full(n_shells, float(d_r))
    else:
        d_r_arr = jnp.asarray(d_r)
        d_r_np = np.asarray(d_r, dtype=float)

    if t_enter is None or t_exit is None:
        t_enter, t_exit = _box_ray_intervals(observer_pos_mpc, box_shape, nside)

    t_enter_jnp = jnp.asarray(t_enter)
    t_exit_jnp = jnp.asarray(t_exit)

    n_total = float(pos.shape[0])
    v_box = float(np.prod(np.asarray(box_shape, dtype=float)))
    n_bar = n_total / v_box

    lower = np.asarray(r_shells, dtype=float) - 0.5 * d_r_np
    upper = np.asarray(r_shells, dtype=float) + 0.5 * d_r_np
    if not np.allclose(upper[:-1], lower[1:]):
        raise ValueError(
            "convergence_Born_spherical requires contiguous shells: "
            "r_shells +/- d_r/2 must tile without gaps or overlaps."
        )

    pos_phys = pos * box_size_jnp / jnp.asarray(mesh_shape_tuple, dtype=float)
    rel = pos_phys - observer_pos_mpc
    r = jnp.linalg.norm(rel, axis=-1)
    r_safe = jnp.where(r > 1e-10, r, 1e-10)
    theta, phi = jhp.vec2ang(rel / r_safe[..., None])
    pixels, interp_weights = jhp.get_interp_weights(nside, theta, phi)

    if shell_weights == "nearest":
        edges = jnp.asarray(np.concatenate([lower, upper[-1:]]))
        shell = jnp.searchsorted(edges, r, side="right") - 1
        inside = (shell >= 0) & (shell < n_shells)
        assignment = [(shell, jnp.where(inside, 1.0, 0.0))]
        shell_vol = (
            (4.0 * jnp.pi / npix_full)
            * (jnp.asarray(upper) ** 3 - jnp.asarray(lower) ** 3)
            / 3.0
        )
        mask_lower, mask_upper = lower, upper
    elif shell_weights == "linear":
        if not np.allclose(d_r_np, d_r_np[0]):
            raise ValueError("shell_weights='linear' requires uniform shells.")
        h = d_r_np[0]
        f = (r - float(r_shells[0])) / h
        i0 = jnp.floor(f)
        w1 = f - i0
        i0 = i0.astype(jnp.int32)

        def tent(idx, w):
            ok = (idx >= 0) & (idx < n_shells)
            return idx, jnp.where(ok, w, 0.0)

        assignment = [tent(i0, 1.0 - w1), tent(i0 + 1, w1)]
        shell_vol = jnp.asarray(linear_shell_volumes(r_shells, h, npix_full))
        # The tent reaches d_r beyond each centre; nudge the mask bounds inwards so that
        # a shell whose support ends exactly on the box boundary is not lost to rounding.
        tol = 1e-6 * (float(np.max(r_shells)) + h)
        mask_lower = np.asarray(r_shells) - h + tol
        mask_upper = np.asarray(r_shells) + h - tol
    else:
        raise ValueError(f"shell_weights must be 'nearest' or 'linear', got {shell_weights!r}")

    counts = jnp.zeros((n_shells, npix_full))
    for idx, w in assignment:
        shell_idx = jnp.broadcast_to(jnp.clip(idx, 0, n_shells - 1), pixels.shape)
        counts = counts.at[shell_idx, pixels].add(interp_weights * w)

    rho = counts / shell_vol[:, None]

    is_fully_inside = (t_enter_jnp[None, :] <= jnp.asarray(mask_lower)[:, None]) & (
        t_exit_jnp[None, :] >= jnp.asarray(mask_upper)[:, None]
    )
    shell_mask = mask_jnp[None, :] & is_fully_inside
    delta = jnp.where(shell_mask & jnp.isfinite(rho), rho / n_bar - 1.0, 0.0)

    W = lensing_kernel(cosmo, r_shells_jnp, a_shells_jnp, chi_s)
    kappa_hp = (delta * (d_r_arr * W)[:, None]).sum(0)
    if return_full:
        return kappa_hp
    return kappa_hp[mask_jnp]


# =========================================================================
# Pixel noise variance
# =========================================================================


def compute_sigma_hp(ell_in, nell_in, nside, cl_extra_1d=None):
    """Compute pixel-space noise std dev for HEALPix convergence.

    Parameters
    ----------
    ell_in : array-like
        Multipole values.
    nell_in : array-like
        Noise power spectrum N_ell at each ell.
    nside : int
        HEALPix nside (sets ell_max = 2*nside and Omega_pix).
    cl_extra_1d : array-like (ell_max+1,), optional
        Additional Cl contribution indexed from ell=0.

    Returns
    -------
    sigma_hp : float
    """
    ell_max = 2 * nside
    ell_arr = np.arange(ell_max + 1, dtype=float)

    ell_in_arr = np.asarray(ell_in, dtype=float)
    nell_in_arr = np.asarray(nell_in, dtype=float)
    valid = (ell_in_arr > 0) & np.isfinite(nell_in_arr) & (nell_in_arr > 0)
    ell_v = ell_in_arr[valid]
    nell_v = nell_in_arr[valid]
    nell_interp = np.exp(
        np.interp(np.log(np.maximum(ell_arr, 1e-5)), np.log(ell_v), np.log(nell_v))
    )
    nell_interp[0] = 0.0

    power = nell_interp.copy()
    if cl_extra_1d is not None:
        cl_extra = np.asarray(cl_extra_1d, dtype=float)
        n_extra = min(len(cl_extra), ell_max + 1)
        power[:n_extra] += cl_extra[:n_extra]

    weights = (2.0 * ell_arr + 1.0) / (4.0 * np.pi)
    sigma2 = float(np.sum(weights * power))
    return float(np.sqrt(max(sigma2, 0.0)))


def compute_sigma_hp_from_cl(ell_1d, power_1d, nside):
    """JAX-compatible sigma_hp from a precomputed 1-D power spectrum."""
    ell = jnp.asarray(ell_1d, dtype=float)
    power = jnp.asarray(power_1d, dtype=float)
    weights = (2.0 * ell + 1.0) / (4.0 * jnp.pi)
    sigma2 = jnp.sum(weights * power)
    return jnp.sqrt(jnp.maximum(sigma2, 0.0))


def sample_healpix_gaussian(key, cl_1d, nside, lmax=None, mask=None):
    """Draw a scalar HEALPix Gaussian realisation from an input C_ell."""
    import healpy as hp
    import jax_healpy as jhp

    cl_1d = jnp.asarray(cl_1d, dtype=float)
    if lmax is None:
        lmax = len(cl_1d) - 1

    ell = np.asarray(hp.Alm.getlm(lmax)[0], dtype=int)
    emm = np.asarray(hp.Alm.getlm(lmax)[1], dtype=int)
    power_lm = jnp.maximum(cl_1d[ell], 1e-30)

    key_re, key_im = jr.split(key)
    re = jr.normal(key_re, shape=(ell.size,), dtype=cl_1d.dtype)
    im = jr.normal(key_im, shape=(ell.size,), dtype=cl_1d.dtype)

    m0 = jnp.asarray(emm == 0)
    alm_mp = (re + 1j * im) * jnp.sqrt(power_lm / 2.0)
    alm_m0 = re * jnp.sqrt(power_lm)
    alm = jnp.where(m0, alm_m0, alm_mp)
    noise = jhp.alm2map(
        alm,
        nside=nside,
        lmax=lmax,
        pol=False,
        healpy_ordering=True,
    )
    noise = jnp.real(noise)  # alm2map may return complex; κ is a real scalar field
    if mask is not None:
        noise = noise * jnp.asarray(mask, dtype=noise.dtype)
    return noise



# =========================================================================
# Theoretical power spectra (Limber)
# =========================================================================


def compute_theoretical_cl_kappa(cosmo, ell, chi_min, chi_max, z_source, n_steps=100, linear_pk=False, n_interp=48):
    """C_ell^{kappa kappa} via Limber approximation."""
    chi_s = jc.background.radial_comoving_distance(cosmo, 1.0 / (1.0 + z_source))[0]
    chi_max = jnp.minimum(chi_max, chi_s)
    chi = jnp.linspace(chi_min, chi_max, n_steps)
    a = jc.background.a_of_chi(cosmo, chi)
    w_val = lensing_kernel(cosmo, chi, a, chi_s)

    def get_cl_per_ell(ell_val):
        k = (ell_val + 0.5) / chi
        if linear_pk:
            pk = jax.vmap(
                lambda ki, ai: jnp.squeeze(jc.power.linear_matter_power(cosmo, ki, ai))
            )(k, a)
        else:
            pk = jax.vmap(
                lambda ki, ai: jnp.squeeze(jc.power.nonlinear_matter_power(cosmo, ki, ai))
            )(k, a)
        integrand = (w_val**2 / chi**2) * pk
        return jax.scipy.integrate.trapezoid(integrand, x=chi)

    ell = jax.lax.stop_gradient(ell)
    ell_min_v = jnp.maximum(jnp.min(ell), 1.0)
    ell_max_v = jnp.maximum(jnp.max(ell), ell_min_v + 1.0)
    ell_interp = jax.lax.stop_gradient(jnp.geomspace(ell_min_v, ell_max_v, n_interp))
    cl_interp = jnp.squeeze(jax.vmap(get_cl_per_ell)(ell_interp))
    # Interpolate in log-log (smooth, positive C_ell) and clip the unused ell<1.
    log_cl = jnp.interp(
        jnp.log(jnp.maximum(ell, 1.0)),
        jnp.log(ell_interp),
        jnp.log(jnp.maximum(cl_interp, 1e-40)),
    )
    return jnp.exp(log_cl)


def compute_theoretical_cl_kappa_windowed(
    cosmo,
    ell,
    r_shells,
    a_shells,
    d_r,
    z_source,
    shell_weights=None,
    k_nyq=None,
):
    """Discrete shell theory matching the closure projector geometry.

    This approximates the Born integral using the same shell centres and widths
    as the forward model, with optional per-shell angular support weights.
    """
    chi = jnp.asarray(r_shells, dtype=float)
    a = jnp.asarray(a_shells, dtype=float)
    if np.ndim(d_r) == 0:
        dr = jnp.full(chi.shape, float(d_r), dtype=float)
    else:
        dr = jnp.asarray(d_r, dtype=float)
    if shell_weights is None:
        shell_weights = jnp.ones_like(chi)
    else:
        shell_weights = jnp.asarray(shell_weights, dtype=float)

    chi_s = jc.background.radial_comoving_distance(cosmo, 1.0 / (1.0 + z_source))[0]
    w_val = lensing_kernel(cosmo, chi, a, chi_s)

    def get_cl_per_ell(ell_val):
        k = (ell_val + 0.5) / chi
        pk = jax.vmap(
            lambda ki, ai: jnp.squeeze(jc.power.nonlinear_matter_power(cosmo, ki, ai))
        )(k, a)
        integrand = shell_weights * (w_val**2 / chi**2) * pk
        if k_nyq is not None:
            integrand = integrand * (k <= k_nyq)
        return jnp.sum(integrand * dr)

    return jax.vmap(get_cl_per_ell)(jnp.asarray(ell, dtype=float))


def compute_theoretical_cl_gg(
    cosmo, ell, chi_min, chi_max, b1, n_steps=100, bias_of_a=None, k_nyq=None
):
    """C_ell^{gg} via Limber approximation (uniform galaxy distribution)."""
    chi = jnp.linspace(chi_min, chi_max, n_steps)
    a = jc.background.a_of_chi(cosmo, chi)
    delta_chi = chi_max - chi_min
    bias = jnp.ones_like(a) * b1 if bias_of_a is None else bias_of_a(a)
    w_g = bias / delta_chi

    def get_cl_per_ell(ell_val):
        k = (ell_val + 0.5) / chi
        pk = jax.vmap(
            lambda ki, ai: jnp.squeeze(jc.power.nonlinear_matter_power(cosmo, ki, ai))
        )(k, a)
        integrand = (w_g**2 / chi**2) * pk
        if k_nyq is not None:
            integrand = integrand * (chi >= (ell_val + 0.5) / k_nyq)
        dx = chi[1] - chi[0]
        return (jnp.sum(integrand) - 0.5 * (integrand[0] + integrand[-1])) * dx

    return jax.vmap(get_cl_per_ell)(ell)


def compute_theoretical_cl_kg(
    cosmo, ell, chi_min, chi_max, z_source, b1, n_steps=100, bias_of_a=None, k_nyq=None
):
    """C_ell^{kappa g} cross-spectrum via Limber approximation."""
    chi_s = jc.background.radial_comoving_distance(cosmo, 1.0 / (1.0 + z_source))[0]
    chi = jnp.linspace(chi_min, chi_max, n_steps)
    a = jc.background.a_of_chi(cosmo, chi)
    w_kappa = lensing_kernel(cosmo, chi, a, chi_s)
    delta_chi = chi_max - chi_min
    bias = jnp.ones_like(a) * b1 if bias_of_a is None else bias_of_a(a)
    w_g = bias / delta_chi

    def get_cl_per_ell(ell_val):
        k = (ell_val + 0.5) / chi
        pk = jax.vmap(
            lambda ki, ai: jnp.squeeze(jc.power.nonlinear_matter_power(cosmo, ki, ai))
        )(k, a)
        integrand = (w_kappa * w_g / chi**2) * pk
        if k_nyq is not None:
            integrand = integrand * (chi >= (ell_val + 0.5) / k_nyq)
        dx = chi[1] - chi[0]
        return (jnp.sum(integrand) - 0.5 * (integrand[0] + integrand[-1])) * dx

    return jax.vmap(get_cl_per_ell)(ell)


# =========================================================================
# High-z correction (1-D ell array, not 2-D FFT grid)
# =========================================================================


def compute_cl_high_z(
    cosmo,
    ell_1d,
    chi_min,
    chi_max,
    z_source,
    mode="fixed",
    cosmo_fid=None,
    cl_cached=None,
    gradients=None,
    loc_fid=None,
    n_steps=100,
):
    """High-z C_ell^{kappa kappa} correction.

    Modes:
      'fixed'  : returns cached C_ell at fiducial cosmology.
      'taylor' : first-order Taylor in (Omega_m, sigma_8).
      'exact'  : dynamic Limber integral for current cosmo (nonlinear Pk).
      'exact_linear' : dynamic Limber integral for current cosmo (linear Pk).
    """
    if mode == "fixed":
        if cl_cached is None:
            raise ValueError("mode='fixed' requires cl_cached")
        return cl_cached

    elif mode == "taylor":
        if cl_cached is None or gradients is None or loc_fid is None:
            raise ValueError("mode='taylor' requires cl_cached, gradients, loc_fid")
        Om_curr = cosmo.Omega_c + cosmo.Omega_b
        s8_curr = cosmo.sigma8
        dOm = Om_curr - loc_fid["Omega_m"]
        ds8 = s8_curr - loc_fid["sigma8"]
        cl = cl_cached + gradients["dCl_dOm"] * dOm + gradients["dCl_ds8"] * ds8
        return jnp.maximum(cl, 1e-30)

    elif mode == "exact":
        chi_source = jc.background.radial_comoving_distance(cosmo, 1.0 / (1.0 + z_source))[0]
        chi_max_eff = chi_source if chi_max is None else chi_max
        return compute_theoretical_cl_kappa(
            cosmo, ell_1d, chi_min, chi_max_eff, z_source, n_steps=n_steps
        )

    elif mode == "exact_linear":
        chi_source = jc.background.radial_comoving_distance(cosmo, 1.0 / (1.0 + z_source))[0]
        chi_max_eff = chi_source if chi_max is None else chi_max
        return compute_theoretical_cl_kappa(
            cosmo, ell_1d, chi_min, chi_max_eff, z_source, n_steps=n_steps, linear_pk=True
        )

    else:
        raise ValueError(f"Unknown mode '{mode}'. Must be 'fixed', 'taylor', 'exact', or 'exact_linear'")


# =========================================================================
# AbacusSummit kappa observation -- HEALPix direct extraction
# =========================================================================


def prepare_abacus_kappa_hp(healpix_map, nside, mask=None):
    """Degrade an AbacusSummit HEALPix kappa map.

    Parameters
    ----------
    healpix_map : np.ndarray
        Full-sky HEALPix map (RING ordering, any nside).
    nside : int
        Target HEALPix nside.
    mask : np.ndarray (npix_full,) bool, optional
        If provided, return only the masked pixels.

    Returns
    -------
    kappa_out : np.ndarray
    """
    import healpy as hp

    nside_in = hp.get_nside(healpix_map)
    if nside_in != nside:
        healpix_map = hp.ud_grade(healpix_map, nside, order_in="RING", order_out="RING")
    healpix_map = np.asarray(healpix_map)
    if mask is None:
        return healpix_map
    return healpix_map[np.asarray(mask, dtype=bool)]


# =========================================================================
# AbacusSummit observation loaders
# =========================================================================


def load_abacus_kappa_observation(abacus_cfg: dict, model) -> dict:
    """Load an AbacusSummit kappa HEALPix map and build an observed kappa map.

    Pipeline:
      1. Load the full-sky HEALPix ASDF map.
      2. ud_grade to model.cmb_nside (full-sky map at the model resolution).
      3. Add a harmonic Gaussian noise realisation on the effective mask.

    Parameters
    ----------
    abacus_cfg : dict
        Sub-dict with keys:
            file (str)       : Path to the ASDF convergence file.
            noise_seed (int) : JAX PRNGKey seed (default 7777).
    model : FieldLevelModel
        Required attributes: cmb_enabled, cmb_nside, cmb_lmax, cmb_mask, nell_1d.

    Returns
    -------
    dict: 'kappa_obs' (npix_full,), 'kappa_pred' (npix_full,).
    """
    if not model.cmb_enabled:
        raise ValueError("load_abacus_kappa_observation requires cmb_enabled=True")

    try:
        import healpy as hp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("healpy is required for observation_mode='abacus'.") from exc
    try:
        import asdf as _asdf
    except ImportError as exc:
        raise RuntimeError("asdf is required for observation_mode='abacus'.") from exc

    _DEFAULT_FILE = (
        "/global/cfs/cdirs/desi/cosmosim/AbacusLensing/"
        "v1/AbacusSummit_base_c000_ph000/kappa_00047.asdf"
    )
    abacus_file = abacus_cfg.get("file", _DEFAULT_FILE)
    noise_seed = int(abacus_cfg.get("noise_seed", 7777))

    if "noise_seed" not in abacus_cfg:
        print(
            "[Abacus] WARNING: noise_seed not set. Using default 7777. "
            "Set it explicitly for independent noise draws."
        )

    nside = model.cmb_nside
    # Effective analysis mask = simulation footprint & external kappa footprint.
    eff_mask = np.asarray(getattr(model, "cmb_mask", getattr(model, "cmb_sim_mask", None)), dtype=bool)

    print(f"[Abacus] Loading {abacus_file} ...")
    _kappa_hp = None
    try:
        with _asdf.open(abacus_file) as _f:
            _kappa_hp = np.array(_f.tree["data"]["kappa"])
        kappa_full_np = prepare_abacus_kappa_hp(_kappa_hp, nside)
    finally:
        del _kappa_hp

    kappa_support = kappa_full_np[eff_mask]
    print("[Abacus] kappa map stats (noiseless, on cmb_mask):")
    print(f"  n_pix_support = {kappa_support.size}, nside = {nside}")
    print(f"  mean  = {float(kappa_support.mean()):.6f}")
    print(f"  std   = {float(kappa_support.std()):.6f}")

    kappa_signal = jnp.asarray(kappa_full_np, dtype=float)
    del kappa_full_np, kappa_support

    noise = sample_healpix_gaussian(
        jr.key(noise_seed),
        jnp.asarray(model.nell_1d),
        nside=model.cmb_nside,
        lmax=model.cmb_lmax,
        mask=eff_mask,
    )
    kappa_obs = kappa_signal + noise

    print(
        f"[Abacus] Noise added (seed={noise_seed}): "
        f"sigma_signal={float(jnp.std(kappa_signal)):.4f}, "
        f"sigma_noise={float(model.sigma_hp):.4f}, "
        f"sigma_total={float(jnp.std(kappa_obs)):.4f}"
    )

    return {"kappa_obs": kappa_obs, "kappa_pred": kappa_signal}


def _is_cubic_box_catalog(path) -> bool:
    """True if `path` is an AbacusSummit CubicBox snapshot (cartesian x,y,z, no RA/DEC)."""
    from pathlib import Path as _P
    path = _P(path)
    if path.suffix == ".asdf":
        return False  # AbacusLensing lightcone mocks are asdf with RA/DEC
    import fitsio
    cols = fitsio.FITS(str(path))[1].get_colnames()
    return "x" in cols and "RA" not in cols


def _load_abacus_cubic_box(file_paths, model, final_shape, paint_oversamp) -> dict:
    """Load AbacusSummit CubicBox snapshot (periodic full box) -> galaxy obs dict.

    No lightcone, no survey window: selec=1 everywhere, n_bar uniform (matches the
    montecosmo unbiased-f_NL snapshot setup). Positions are cartesian in [-L/2, L/2],
    shifted to [0, L) and wrapped (periodic). Real-space (no RSD) unless los is set.
    Requires model.lightcone=false and model.a_obs set. See docs/pipeline.md.
    """
    import fitsio

    from desi_cmb_fli.bricks import paint_arbitrary

    box = np.asarray(model.box_shape, dtype=float)
    mesh = np.asarray(final_shape, dtype=float)
    pos_list = []
    for fp in file_paths:
        d = fitsio.read(str(fp), columns=["x", "y", "z"])
        pos_list.append(np.column_stack([d["x"], d["y"], d["z"]]).astype(float))
    pos = np.concatenate(pos_list, axis=0)
    # CubicBox is centered at the origin; shift to [0, L) and wrap into cell units.
    cell = jnp.asarray(((pos + box / 2.0) / (box / mesh)) % mesh)
    n_gal = int(cell.shape[0])
    print(f"[Abacus-snapshot] {n_gal} galaxies (CubicBox, full periodic box) -> mesh {final_shape}")

    mesh_plain = paint_arbitrary(jnp.zeros(final_shape), cell)
    if paint_oversamp > 1.0:
        from desi_cmb_fli.bricks import interlace_paint_deconv
        from desi_cmb_fli.utils import get_scaled_shape

        paint_shape = get_scaled_shape(final_shape, paint_oversamp)
        scale = jnp.asarray([paint_shape[i] / final_shape[i] for i in range(3)], dtype=cell.dtype)
        mesh_count = interlace_paint_deconv(cell * scale, paint_shape, final_shape)
        print(f"[Abacus-snapshot] Interlaced painting (oversamp={paint_oversamp})")
    else:
        mesh_count = mesh_plain

    nbar = float(jnp.mean(mesh_count))
    obs_mesh = mesh_count
    survey_mask = jnp.ones(final_shape, dtype=bool)
    selec_mesh = jnp.ones(final_shape, dtype=jnp.float32)

    model.gxy_count = nbar
    model.selec_mesh = selec_mesh
    model.selec_paint = None
    model.gxy_occ_mask3d = survey_mask
    if getattr(model, "lightcone", False):
        raise ValueError(
            "[Abacus-snapshot] CubicBox catalog requires model.lightcone=false and a fixed "
            "model.a_obs (e.g. 0.5556 for z=0.8)."
        )
    print(
        f"[Abacus-snapshot] Full box: n_bar={nbar:.2f} gal/cell, "
        f"a_obs={model.a_obs:.4f} (z={1.0 / model.a_obs - 1.0:.3f}), selec=1 everywhere"
    )

    return {
        "obs": obs_mesh,
        "chi_range_gxy": None,
        "gxy_occ_mask3d": survey_mask,
        "selec_mesh": selec_mesh,
        "mesh_plain": mesh_plain,
        "rand_mesh_plain": None,
    }


def _randoms_cache_path(cache_dir, rand_paths, z_range, box_shape, mesh_shape,
                        observer_pos, cosmo, max_rows, chunk_rows, paint_shape):
    """Content-addressed path for the reduced randoms meshes.

    The randoms catalogue (tens of GB) is only ever used to produce two
    ``mesh_shape`` arrays and one count, so that reduction is cached on disk.
    The key covers every input that changes it. The catalogue files are
    identified by (path, size, mtime) rather than hashed, and the cosmology by
    the leaves of the ``Cosmology`` pytree -- i.e. the background parameters
    that set the z -> distance conversion, and not the nuisance latents.
    """
    import hashlib
    import json

    stamp = {
        "version": 3,
        "files": [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in rand_paths],
        "z_range": None if z_range is None else [float(z) for z in z_range],
        "box_shape": [float(x) for x in np.asarray(box_shape).ravel()],
        "mesh_shape": [int(x) for x in mesh_shape],
        "observer": [float(x) for x in np.asarray(observer_pos).ravel()],
        "cosmo": [float(np.asarray(x)) for x in jax.tree_util.tree_leaves(cosmo)],
        "max_rows": None if max_rows is None else int(max_rows),
        "chunk_rows": int(chunk_rows),
        "paint_shape": None if paint_shape is None else [int(x) for x in paint_shape],
    }
    digest = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()[:16]
    return Path(cache_dir) / f"randoms_{digest}.npz"


def load_abacus_galaxy_observation(
    abacus_gxy_cfg: dict,
    model,
) -> dict:
    """Load AbacusSummit lightcone galaxy catalogs -> {'obs': mesh}.

    Curved-sky version: uses observer_position instead of box_center + rotation.

    Parameters
    ----------
    abacus_gxy_cfg : dict
        Config sub-dict with 'file' (str | list[str]), and optionally
        'randoms' (standalone randoms file(s); otherwise the RAND_* columns of
        'file' are used), 'z_range' (overrides the per-shell path-derived
        redshift bounds), 'randoms_max_rows' and 'randoms_chunk_rows'
        (FITS randoms streaming).  FKP weights, if present, are deliberately
        not applied: the likelihood is Poisson on raw counts.
    model : FieldLevelModel
        Required: observer_position, box_shape, mesh_shape, loc_fid.
        Modified in-place: gxy_count, selec_mesh, gxy_occ_mask3d.

    Returns
    -------
    dict with keys: 'obs', 'chi_range_gxy', 'gxy_occ_mask3d', 'selec_mesh',
        'mesh_plain', 'rand_mesh_plain'.
    """
    import re as _re
    from pathlib import Path as _Path

    from desi_cmb_fli.bricks import (
        INTERLACE_ORDER,
        catalog2positions,
        get_cosmology,
        interlace_accumulate,
        interlace_combine,
        interlace_finalize,
        paint_arbitrary,
        radius_mesh,
        randoms2positions,
        randoms_num_rows,
    )
    from desi_cmb_fli.utils import get_scaled_shape

    def _as_paths(raw):
        return (
            [_Path(p) for p in raw]
            if isinstance(raw, list | tuple)
            else [_Path(raw)]
        )

    file_paths = _as_paths(abacus_gxy_cfg["file"])
    rand_file = abacus_gxy_cfg.get("randoms")
    rand_paths = _as_paths(rand_file) if rand_file else None

    cosmo_fid = get_cosmology(**model.loc_fid)
    final_shape = tuple(model.mesh_shape)
    observer_pos = np.asarray(model.observer_position)
    paint_oversamp = float(getattr(model, "paint_oversamp", 1.0))

    # CubicBox snapshot (cartesian x,y,z, periodic full box) vs lightcone (RA/DEC) — auto-detect.
    # Snapshot path: no observer/octant/randoms/selection. Requires model.lightcone=false + a_obs.
    # See docs/pipeline.md "Abacus CubicBox snapshot".
    if _is_cubic_box_catalog(file_paths[0]):
        return _load_abacus_cubic_box(file_paths, model, final_shape, paint_oversamp)

    def _nominal_z(fp):
        m = _re.search(r"/z(\d+\.\d+)/", str(fp))
        return float(m.group(1)) if m else None

    cfg_z_range = abacus_gxy_cfg.get("z_range")
    if cfg_z_range is not None:
        cfg_z_range = (float(cfg_z_range[0]), float(cfg_z_range[1]))

    nominal_zs = [_nominal_z(fp) for fp in file_paths]
    if cfg_z_range is not None:
        z_ranges = [cfg_z_range] * len(file_paths)
        print(f"[Abacus-gxy] z_range from config: [{cfg_z_range[0]:.3f}, {cfg_z_range[1]:.3f}]")
    elif all(z is not None for z in nominal_zs) and len(file_paths) > 1:
        zs = sorted(set(nominal_zs))
        z_boundaries = {}
        for z in zs:
            idx = zs.index(z)
            z_lo = (zs[idx - 1] + z) / 2 if idx > 0 else 0.0
            z_hi = (z + zs[idx + 1]) / 2 if idx < len(zs) - 1 else 100.0
            z_boundaries[z] = (z_lo, z_hi)
        z_ranges = [z_boundaries[z] for z in nominal_zs]
        print(f"[Abacus-gxy] Shell dedup: {len(zs)} shells, z=[{zs[0]:.3f}, {zs[-1]:.3f}]")
    else:
        z_ranges = [None] * len(file_paths)

    all_gxy_positions = []
    n_gxy_per_file = []
    n_gxy_total = 0
    chi_min_gxy, chi_max_gxy = float("inf"), float("-inf")

    for fp, z_range in zip(file_paths, z_ranges, strict=False):
        pos_i, n_i, chi_min_i, chi_max_i = catalog2positions(
            path=fp,
            cosmo=cosmo_fid,
            observer_position=observer_pos,
            box_shape=model.box_shape,
            mesh_shape=final_shape,
            z_range=z_range,
        )
        if n_i > 0:
            chi_min_gxy = min(chi_min_gxy, chi_min_i)
            chi_max_gxy = max(chi_max_gxy, chi_max_i)
            all_gxy_positions.append(np.asarray(pos_i))
        n_gxy_per_file.append(n_i)
        n_gxy_total += n_i

    if n_gxy_total == 0:
        raise ValueError(
            "[Abacus-gxy] No galaxies inside the box. "
            "Check observer_position and box_shape."
        )

    all_gxy_pos = jnp.concatenate(all_gxy_positions, axis=0)
    del all_gxy_positions

    chunk_rows = int(abacus_gxy_cfg.get("randoms_chunk_rows", 50_000_000))
    max_rows = abacus_gxy_cfg.get("randoms_max_rows")

    def _random_sources():
        if rand_paths is None:
            yield from zip(file_paths, z_ranges, [None] * len(file_paths),
                           n_gxy_per_file, strict=False)
            return
        for rp in rand_paths:
            n_rows = randoms_num_rows(rp)
            if n_rows is None:
                yield rp, cfg_z_range, None, None
                continue
            if max_rows:
                n_rows = min(n_rows, int(max_rows))
            for start in range(0, n_rows, chunk_rows):
                yield rp, cfg_z_range, (start, min(start + chunk_rows, n_rows)), None

    rand_mesh_support = jnp.zeros(final_shape)
    rand_mesh_window = None
    rand_mesh_paint = None
    n_rand_total = 0
    paint_chunk = int(abacus_gxy_cfg.get("randoms_paint_chunk", 4_194_304))
    rand_paint_shape = (
        get_scaled_shape(final_shape, paint_oversamp) if paint_oversamp > 1.0 else None
    )
    if rand_paint_shape is not None and tuple(rand_paint_shape) != tuple(
        int(s) for s in model.paint_shape
    ):
        raise ValueError(
            f"[Abacus-gxy] randoms paint grid {tuple(rand_paint_shape)} differs from the model's "
            f"{tuple(int(s) for s in model.paint_shape)}: the selection could not multiply the "
            "predicted field."
        )

    # The randoms reduce to these two meshes and one count; cache that reduction.
    cache_dir = abacus_gxy_cfg.get("randoms_cache_dir", _DEFAULT_RANDOMS_CACHE_DIR)
    cache_path = None
    if rand_paths is not None and cache_dir is not None:
        cache_path = _randoms_cache_path(
            cache_dir, rand_paths, cfg_z_range, model.box_shape, final_shape,
            observer_pos, cosmo_fid, max_rows, chunk_rows, rand_paint_shape,
        )

    if cache_path is not None and cache_path.exists():
        with np.load(cache_path) as cached:
            rand_mesh_support = jnp.asarray(cached["support"])
            rand_mesh_window = jnp.asarray(cached["window"])
            if "window_paint" in cached:
                rand_mesh_paint = jnp.asarray(cached["window_paint"])
            n_rand_total = int(cached["n_rand_total"])
        print(f"[Abacus-gxy] Randoms reduction from cache: {cache_path}")
    else:
        rand_shifted, rand_mesh_cic = None, jnp.zeros(final_shape)
        if rand_paint_shape is not None:
            rand_shifted = [jnp.zeros(rand_paint_shape) for _ in range(INTERLACE_ORDER)]
            rand_scale = jnp.asarray(
                [rand_paint_shape[i] / final_shape[i] for i in range(3)], dtype=float
            )
        for rp, z_range, row_range, n_i in _random_sources():
            rpos_i, rn_i = randoms2positions(
                path=rp,
                cosmo=cosmo_fid,
                observer_position=observer_pos,
                box_shape=model.box_shape,
                mesh_shape=final_shape,
                z_range=z_range,
                row_range=row_range,
            )
            n_rand_total += rn_i
            if rn_i == 0:
                continue
            w_i = None if n_i is None else n_i / rn_i
            rand_mesh_support = paint_arbitrary(rand_mesh_support, rpos_i, chunk_size=paint_chunk)
            if rand_shifted is not None:
                rand_shifted = interlace_accumulate(
                    rand_shifted, rpos_i * rand_scale, w_i, chunk_size=paint_chunk
                )
            else:
                rand_mesh_cic = paint_arbitrary(
                    rand_mesh_cic, rpos_i, w_i, chunk_size=paint_chunk
                )
        if rand_shifted is not None:
            rand_mesh_window = interlace_finalize(rand_shifted, rand_paint_shape, final_shape)
            rand_mesh_paint = interlace_combine(rand_shifted, rand_paint_shape)
            del rand_shifted
        else:
            rand_mesh_window = rand_mesh_cic
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            to_cache = {
                "support": np.asarray(rand_mesh_support),
                "window": np.asarray(rand_mesh_window),
                "n_rand_total": np.int64(n_rand_total),
            }
            if rand_mesh_paint is not None:
                to_cache["window_paint"] = np.asarray(rand_mesh_paint, dtype=np.float32)
            np.savez(cache_path, **to_cache)
            print(f"[Abacus-gxy] Randoms reduction cached: {cache_path}")

    # No global alpha = n_gal/n_rand: the unit-mean normalisation below divides any constant out.
    # The ASDF path's per-shell alpha (w_i above) does survive, since it varies from shell to shell.
    print(f"[Abacus-gxy] {n_gxy_total} galaxies, {n_rand_total} randoms -> mesh {final_shape}")

    mesh_plain = paint_arbitrary(jnp.zeros(final_shape), all_gxy_pos)

    if paint_oversamp > 1.0:
        from desi_cmb_fli.bricks import interlace_paint_deconv
        from desi_cmb_fli.utils import get_scaled_shape

        paint_shape = get_scaled_shape(final_shape, paint_oversamp)
        scale = jnp.asarray(
            [paint_shape[i] / final_shape[i] for i in range(3)],
            dtype=all_gxy_pos.dtype,
        )
        mesh_count = interlace_paint_deconv(all_gxy_pos * scale, paint_shape, final_shape)
        print(f"[Abacus-gxy] Interlaced painting (oversamp={paint_oversamp})")
    else:
        mesh_count = mesh_plain

    del all_gxy_pos

    if n_rand_total == 0:
        raise ValueError(
            "[Abacus-gxy] No randoms found. Set abacus_galaxy.randoms, or use "
            "ASDF catalogs carrying RAND_* columns."
        )

    # Survey mask = random occupancy + completeness cut (docs/pipeline.md "Completeness cut").
    completeness_min = float(abacus_gxy_cfg.get("completeness_min", 0.8))
    survey_mask = jnp.asarray(np.asarray(rand_mesh_support > 0))
    del rand_mesh_support

    # Unit mean over the occupied cells, so the selection is a dimensionless completeness and
    # n_eff_per_cell below carries the density. selec_paint is the same field on the paint grid:
    # band_limit(selec_paint) == selec_mesh inside the survey, by construction.
    norm = jnp.mean(jnp.asarray(rand_mesh_window)[survey_mask])
    selec_mesh = jnp.asarray(rand_mesh_window / norm, dtype=jnp.float32)
    survey_mask = survey_mask & (selec_mesh > completeness_min)
    if rand_mesh_paint is None:
        selec_paint = None
    else:
        paint_per_final = float(np.prod(rand_paint_shape) / np.prod(final_shape))
        selec_paint = jnp.asarray(rand_mesh_paint * paint_per_final / norm, dtype=jnp.float32)
    selec_mesh = jnp.where(survey_mask, selec_mesh, 1.0)
    n_survey_cells = int(jnp.sum(survey_mask))

    print(
        f"[Abacus-gxy] Selection: "
        f"{n_survey_cells}/{int(np.prod(final_shape))} cells "
        f"({float(n_survey_cells) / int(np.prod(final_shape)):.1%}), "
        f"completeness_min={completeness_min}"
    )

    n_gal_in_survey = float(jnp.sum(mesh_count[survey_mask]))
    sum_selec_survey = float(jnp.sum(selec_mesh[survey_mask]))
    n_eff_per_cell = n_gal_in_survey / sum_selec_survey

    obs_mesh = mesh_count

    model.gxy_count = n_eff_per_cell
    model.selec_mesh = selec_mesh
    model.selec_paint = selec_paint
    model.gxy_occ_mask3d = survey_mask

    rmesh = np.asarray(radius_mesh(model.box_center, model.box_shape, final_shape,
                                   curved_sky=model.curved_sky, los=model.los))

    n_rbins = int(getattr(model, "n_gxy_shells", 0))
    if getattr(model, "gxy_ngbar_free", False) and n_rbins >= 1:
        rsurv = rmesh[np.asarray(survey_mask)]
        rmin, rmax = float(rsurv.min()), float(rsurv.max())
        eps = (rmax - rmin) / n_rbins * 1e-3 + 1e-6
        redges = np.linspace(rmin - eps, rmax + eps, n_rbins + 1)
        sid = np.clip(np.digitize(rmesh, redges) - 1, 0, n_rbins - 1)
        sid = np.where(np.asarray(survey_mask), sid, -1)
        model.gxy_shell_id = jnp.asarray(sid, dtype=jnp.int32)
        print(f"[ngbars] {n_rbins} fine radial bins (dr≈{(rmax - rmin) / n_rbins:.0f} Mpc/h, "
              f"chi=[{rmin:.0f},{rmax:.0f}] Mpc/h), survey cells/bin="
              f"{[int(np.sum(sid == b)) for b in range(n_rbins)]}")

    in_survey_z = jnp.any(survey_mask, axis=(0, 1))
    if getattr(model, "lightcone", False):
        from desi_cmb_fli.nbody import a2g, chi2a, g2a

        chi_survey = jnp.asarray(rmesh[np.asarray(survey_mask)])
        a_survey = chi2a(cosmo_fid, chi_survey)
        a_fid_new = float(g2a(cosmo_fid, jnp.mean(a2g(cosmo_fid, a_survey))))
        a_fid_old = getattr(model, "a_fid", None)
        if a_fid_old is not None:
            print(
                f"[Abacus-gxy] Updating a_fid: {a_fid_old:.4f} "
                f"-> {a_fid_new:.4f} [survey cells]"
            )
        model.a_fid = a_fid_new

    survey_frac = float(n_survey_cells) / float(int(np.prod(final_shape)))
    n_survey_z = int(jnp.sum(in_survey_z))
    nz = final_shape[2]
    print(
        f"[Abacus-gxy] Survey: {n_survey_cells} cells ({survey_frac:.1%}), "
        f"{n_survey_z}/{nz} z-slices, "
        f"chi=[{chi_min_gxy:.0f}, {chi_max_gxy:.0f}] Mpc/h, "
        f"n_bar={n_eff_per_cell:.2f} gal/cell"
    )

    return {
        "obs": obs_mesh,
        "chi_range_gxy": (chi_min_gxy, chi_max_gxy),
        "gxy_occ_mask3d": survey_mask,
        "selec_mesh": selec_mesh,
        "mesh_plain": mesh_plain,
        "rand_mesh_plain": rand_mesh_window,
    }


def load_abacus_ic_truth(abacus_ic_cfg: dict, model) -> dict:
    """Load the AbacusSummit linear IC field -> {'init_mesh': rfftn field}.

    Validation reference only.  The sampler is warm-started from the *observed*
    galaxy field (``model.kaiser_post``); this field is never a starting point,
    it only lets ``plot_warmup_diagnostics`` score the reconstruction against
    the true modes instead of against a P_lin reference.

    The published ``ic_dens_N*.asdf`` holds delta_L at ``InitialRedshift``
    (z=99) over the full simulation box -- not at ``CLASS_Redshift``, which is
    only the redshift of the CLASS power spectrum the ICs were drawn from.  It
    is grown to a=1 with the header ``GrowthTable`` (normalised to D=1 at
    ``InitialRedshift``) and Fourier-resampled onto ``model.init_shape``, so
    that ``model.spectrum(irfftn(init_mesh))`` matches
    ``lin_power_interp(cosmo)``.
    """
    import asdf as _asdf

    from desi_cmb_fli.utils import chreshape, r2chshape

    with _asdf.open(str(abacus_ic_cfg["file"])) as f:
        header = f.tree["header"]
        box_ic = float(header["BoxSize"])
        z_ic = float(header["InitialRedshift"])
        growth = dict(header["GrowthTable"])
        dens = np.asarray(f.tree["data"]["density"], dtype=np.float32)

    if not np.allclose(np.asarray(model.box_shape, dtype=float), box_ic):
        raise ValueError(
            f"[Abacus-IC] Box mismatch: IC box is {box_ic} Mpc/h but "
            f"model.box_shape={tuple(model.box_shape)}. The published IC covers the "
            f"whole simulation box; set model.box_shape to {box_ic} to use it, or "
            f"drop abacus_ic.file to run without the IC reference."
        )

    dens = dens * (float(growth[0.0]) / float(growth[z_ic]))

    init_shape = tuple(int(s) for s in model.init_shape)
    meshk = jnp.fft.rfftn(jnp.asarray(dens))
    if tuple(dens.shape) != init_shape:
        meshk = chreshape(meshk, r2chshape(init_shape))

    print(
        f"[Abacus-IC] delta_L from z={z_ic:g} grown to a=1 "
        f"(x{float(growth[0.0]) / float(growth[z_ic]):.4f}), "
        f"{dens.shape} -> {init_shape}"
    )
    return {"init_mesh": meshk}
