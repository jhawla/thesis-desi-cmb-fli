from functools import partial
from pathlib import Path

import jax_cosmo as jc
import numpy as np
from jax import numpy as jnp
from jax_cosmo import Cosmology, constants
from jaxpm.painting import cic_read, scatter

from desi_cmb_fli.nbody import a2chi, a2f, a2g, chi2a, invlaplace_kernel, paint_kernel, rfftk
from desi_cmb_fli.utils import (
    cgh2rg,
    ch2rshape,
    chreshape,
    get_scaled_shape,
    r2chshape,
    radecrad2cart,
    rg2cgh,
    safe_div,
    std2trunc,
    trunc2std,
)

# [Planck2015 XIII](https://arxiv.org/abs/1502.01589) Table 4 final column (best fit)
Planck15 = partial(
    Cosmology,
    Omega_c=0.2589,
    Omega_b=0.04860,
    Omega_k=0.0,
    h=0.6774,
    n_s=0.9667,
    sigma8=0.8159,
    w0=-1.0,
    wa=0.0,
)

# [Planck 2018 VI](https://arxiv.org/abs/1807.06209) Table 2 final column (best fit)
Planck18 = partial(
    Cosmology,
    # Omega_m = 0.3111
    Omega_c=0.2607,
    Omega_b=0.0490,
    Omega_k=0.0,
    h=0.6766,
    n_s=0.9665,
    sigma8=0.8102,
    w0=-1.0,
    wa=0.0,
)

# AbacusSummit fiducial cosmology (from cosmoprimo.fiducial)
AbacusSummit0 = partial(
    Cosmology,
    Omega_c=0.26447041,
    Omega_b=0.04930169,
    Omega_k=0.0,
    h=0.6736,
    n_s=0.9649,
    sigma8=0.8076353990239834,
    w0=-1.0,
    wa=0.0,
)


def lin_power_interp(cosmo=Cosmology, a=1.0, n_interp=256):
    """
    Return a light emulation of the linear matter power spectrum.
    """
    ks = jnp.logspace(-4, 1, n_interp)
    logpows = jnp.log(jc.power.linear_matter_power(cosmo, ks, a=a))

    # Interpolate in semilogy space with logspaced k values, correctly handles k==0,
    # as interpolation in loglog space can produce nan gradients
    def pow_fn(x):
        return jnp.exp(
            jnp.interp(x.reshape(-1), ks, logpows, left=-jnp.inf, right=-jnp.inf)
        ).reshape(x.shape)

    return pow_fn


def lin_power_mesh(cosmo: Cosmology, mesh_shape, box_shape, a=1.0, n_interp=256):
    """
    Return linear matter power spectrum field.
    """
    pow_fn = lin_power_interp(cosmo, a=a, n_interp=n_interp)
    kvec = rfftk(mesh_shape)
    kmesh = (
        sum(
            (ki * (m / box_l)) ** 2
            for ki, m, box_l in zip(kvec, mesh_shape, box_shape, strict=False)
        )
        ** 0.5
    )
    return pow_fn(kmesh) * (mesh_shape / box_shape).prod()  # from [Mpc/h]^3 to cell units


def _kmesh_phys(mesh_shape, box_shape):
    """
    Return |k| field in physical h/Mpc units for an rfftn-shaped mesh.

    Our ``rfftk`` returns wavevectors in cell units, so physical wavenumbers are
    recovered as ``ki * (mesh_shape / box_shape)`` (same convention as ``lin_power_mesh``).
    """
    kvec = rfftk(mesh_shape)
    return (
        sum(
            (ki * (m / box_l)) ** 2
            for ki, m, box_l in zip(kvec, mesh_shape, box_shape, strict=False)
        )
        ** 0.5
    )


def trans_phi2delta_interp(cosmo: Cosmology, a=1.0, n_interp=256):
    """
    Return a light emulation of the transfer function M(k) from the primordial
    potential phi to the linear matter density field: delta(k) = M(k) phi(k).

    jax_cosmo has no A_s, so fallback on
    https://arxiv.org/pdf/1904.08859:

        M(k) = 2 rh^2 k^2 T_lin(k) D_norm(a) / (3 Omega_m)

    with T_lin(k) = sqrt(P_lin(k) / k^{n_s}) normalized to 1 as k -> 0, and
    D_norm(a) = D(a) / D(a_norm) * a_norm with a_norm in the matter-dominated era.
    """
    pow_fn = lin_power_interp(cosmo, a, n_interp=n_interp)
    ks = jnp.logspace(-4, 1, n_interp)
    pow_large = ks**cosmo.n_s  # primordial power spectrum on large scales
    pow_lin = pow_fn(ks)
    trans_lin = (pow_lin / pow_large / (pow_lin[0] / pow_large[0])) ** 0.5

    z_norm = 10.0  # in matter-dominated era
    a_norm = 1.0 / (1.0 + z_norm)
    normalized_growth_factor = a2g(cosmo, a) / a2g(cosmo, a_norm) * a_norm
    trans = (
        2.0 * constants.rh**2 * ks**2 * trans_lin * normalized_growth_factor / (3.0 * cosmo.Omega_m)
    )
    return lambda x: jnp.interp(x.reshape(-1), ks, trans, left=0.0, right=0.0).reshape(x.shape)


def add_png(cosmo: Cosmology, fNL, init_mesh, box_shape):
    """
    Add local-type Primordial Non-Gaussianity (PNG) to the linear matter density field.

    Faithful port of montecosmo: in the primordial potential,
    phi -> phi + fNL (phi^2 - <phi^2>), then map back to the density field.

    ``init_mesh`` is the Gaussian linear density field in Fourier space (rfftn, cell units).
    """
    mesh_shape = ch2rshape(init_mesh.shape)
    kmesh = _kmesh_phys(mesh_shape, box_shape)
    trans_phi2delta = trans_phi2delta_interp(cosmo)(kmesh)

    phi = jnp.fft.irfftn(safe_div(init_mesh, trans_phi2delta))
    phi2 = phi**2
    phi = phi + fNL * (phi2 - phi2.mean())
    init_mesh = trans_phi2delta * jnp.fft.rfftn(phi)

    return init_mesh


def b_phi(b1, p=1.0, delta_c=1.686):
    """
    Primordial scale-dependent bias parameter b_phi = 2 delta_c (b1 + 1 - p).

    ``b1`` is the Lagrangian linear bias (b1E = 1 + b1), matching ``lagrangian_weights``.
    p=1 for halos, p~1.6 for recent mergers, p~0.55 for stellar-mass-selected samples.
    See [Giannantonio&Porciani](http://arxiv.org/abs/0911.0017), [Barreira2022](https://arxiv.org/pdf/2107.06887).
    """
    return 2 * delta_c * (b1 + 1 - p)


def b_phi_delta(b1, b2, delta_c=1.686):
    """
    Primordial-density scale-dependent bias parameter b_{phi delta} = 2 (delta_c b2 - b1).

    See [Giannantonio&Porciani](http://arxiv.org/abs/0911.0017), [Barreira2022](https://arxiv.org/pdf/2107.06887).
    """
    return 2 * (delta_c * b2 - b1)


def fNL_bias(fNL, b1, b2, p=1.0, png_type=None, fNL_bp=0.0, fNL_bpd=0.0):
    """
    Return the effective scale-dependent bias amplitudes (fNL_bp, fNL_bpd)
    = (fNL b_phi, fNL b_{phi delta}) entering the galaxy bias.

    - png_type='fNL': derive them from the bias via the universality relation
      (b_phi = 2 delta_c (b1+1-p), b_phi_delta = 2 (delta_c b2 - b1)).
    - png_type='fNL_bias': use the free latents ``fNL_bp``, ``fNL_bpd`` directly.
    """
    if png_type == "fNL":
        return fNL * b_phi(b1, p), fNL * b_phi_delta(b1, b2)
    if png_type == "fNL_bias":
        return fNL_bp, fNL_bpd
    return 0.0, 0.0


def kaiser_posterior(delta_obs, cosmo: Cosmology, bE, a, box_shape, gxy_count, los=None):
    """
    Return posterior mean and std fields of the linear matter field (at a=1) given the observed field,
    by assuming Kaiser model. All fields are in fourier space.

    Noise level is fixed to 1 / gxy_count (pure shot noise).
    """
    # Compute linear matter power spectrum
    mesh_shape = ch2rshape(delta_obs.shape)
    pmeshk = lin_power_mesh(cosmo, mesh_shape, box_shape)
    boost = kaiser_boost(cosmo, a, bE, mesh_shape, los)

    gxy_count = jnp.asarray(gxy_count)
    noise = jnp.where(gxy_count > 0, 1.0 / gxy_count, jnp.inf)
    stds = (pmeshk / (1 + boost**2 * pmeshk / noise)) ** 0.5
    means = stds**2 / noise * boost * delta_obs
    return means, stds


def get_cosmology(**cosmo) -> Cosmology:
    """
    Return full cosmology object from cosmological params.

    The reference cosmology is AbacusSummit0 so that the fiducial background
    matches the AbacusSummit c000 mocks and the Abacus-calibrated background
    emulator (see nbody._is_abacus_background).
    """
    ref_cosmo = AbacusSummit0
    kwargs = ref_cosmo.keywords.copy()
    valid_keys = set(kwargs) | {"Omega_m"}
    nuisance_keys = {"b1", "b2", "bs2", "bn2", "bnpar", "fNL", "fNL_bp", "fNL_bpd", "s_e"}
    unknown = set(cosmo) - valid_keys - nuisance_keys
    unknown = {k for k in unknown if not str(k).startswith("ngbar")}  # per-shell ngbars
    if unknown:
        raise ValueError(f"Unknown cosmology parameter(s): {sorted(unknown)}")

    if "Omega_m" in cosmo:
        kwargs["Omega_c"] = cosmo["Omega_m"] - kwargs["Omega_b"]

    for key, value in cosmo.items():
        if key in kwargs:
            kwargs[key] = value

    return Cosmology(**kwargs)


def regular_pos(mesh_shape: tuple, ptcl_shape: tuple | None = None):
    """
    Return regularly spaced particle positions in cell coordinates.
    """
    if ptcl_shape is None:
        ptcl_shape = mesh_shape

    pos = [
        np.linspace(0, m, p, endpoint=False) for m, p in zip(mesh_shape, ptcl_shape, strict=False)
    ]
    return jnp.stack(np.meshgrid(*pos, indexing="ij"), axis=-1).reshape(-1, 3)


def cic_paint_2d(mesh, positions, weight):
    """
    CIC (Cloud-In-Cell) painting onto a 2D grid.

    Parameters
    ----------
    mesh : array [Nx, Ny]
        2D grid to paint onto (typically zero-initialized).
    positions : array [N, 2]
        Particle positions in pixel units.
    weight : array [N]
        Per-particle weights.

    Returns
    -------
    mesh : array [Nx, Ny]
        Painted 2D grid.
    """
    nx, ny = mesh.shape
    pos_floor = jnp.floor(positions).astype(int)
    frac = positions - pos_floor

    # 4 CIC neighbors: (0,0), (1,0), (0,1), (1,1)
    dx = jnp.stack([1.0 - frac[:, 0], frac[:, 0]], axis=-1)  # (N, 2)
    dy = jnp.stack([1.0 - frac[:, 1], frac[:, 1]], axis=-1)  # (N, 2)

    # Non-periodic BC: particles outside the grid are discarded (not wrapped).
    for di in range(2):
        for dj in range(2):
            ix = pos_floor[:, 0] + di
            iy = pos_floor[:, 1] + dj
            valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            w = jnp.where(valid, weight * dx[:, di] * dy[:, dj], 0.0)
            ix = jnp.clip(ix, 0, nx - 1)
            iy = jnp.clip(iy, 0, ny - 1)
            mesh = mesh.at[ix, iy].add(w)

    return mesh


def cell2phys_pos(pos, box_center, box_shape, mesh_shape):
    """
    Convert cell coordinates to physical coordinates in Mpc/h.
    """
    pos = pos * (box_shape / mesh_shape)
    pos = pos - box_shape / 2.0
    pos = pos + box_center
    return pos


def phys2cell_pos(pos, box_center, box_shape, mesh_shape):
    """
    Convert physical coordinates in Mpc/h to cell coordinates.
    """
    pos = pos - box_center
    pos = pos + box_shape / 2.0
    pos = pos / (box_shape / mesh_shape)
    return pos


def cell2phys_vel(vel, box_shape, mesh_shape):
    """
    Convert cell velocities to physical velocities in Mpc/h.
    """
    return vel * (box_shape / mesh_shape)


def phys2cell_vel(vel, box_shape, mesh_shape):
    """
    Convert physical velocities in Mpc/h to cell velocities.
    """
    return vel / (box_shape / mesh_shape)


def radius_mesh(box_center, box_shape, mesh_shape, curved_sky=True, los=None):
    """
    Return radial comoving distance for each mesh cell center.
    """
    pos = regular_pos(mesh_shape)
    pos = cell2phys_pos(pos, box_center, box_shape, mesh_shape).reshape(tuple(mesh_shape) + (3,))

    if curved_sky:
        return jnp.linalg.norm(pos, axis=-1)

    if los is None:
        los = safe_div(box_center, jnp.linalg.norm(box_center))
    los = jnp.asarray(los) / jnp.linalg.norm(los)
    return jnp.abs((pos * los).sum(-1))


def tophysical_pos(
    pos, box_center, box_shape, mesh_shape, cosmo: Cosmology, a_obs=None, curved_sky=True, los=None
):
    """
    Return physical positions, radial distance, line-of-sight, and scale factor.
    """
    pos = cell2phys_pos(pos, box_center, box_shape, mesh_shape)
    if curved_sky:
        los_vec = safe_div(pos, jnp.linalg.norm(pos, axis=-1, keepdims=True))
        rpos = jnp.linalg.norm(pos, axis=-1, keepdims=True)
    else:
        if los is None:
            los = safe_div(box_center, jnp.linalg.norm(box_center))
        los = jnp.asarray(los) / jnp.linalg.norm(los)
        los_vec = los
        rpos = jnp.abs((pos * los).sum(-1, keepdims=True))

    if a_obs is None:
        a = chi2a(cosmo, rpos)
    else:
        a = a_obs
    return pos, rpos, los_vec, a


def tophysical_mesh(
    box_center, box_shape, mesh_shape, cosmo: Cosmology, a_obs=None, curved_sky=True, los=None
):
    """
    Return line-of-sight and scale-factor mesh.
    """
    pos = regular_pos(mesh_shape)
    pos = cell2phys_pos(pos, box_center, box_shape, mesh_shape).reshape(tuple(mesh_shape) + (3,))

    if curved_sky:
        los_mesh = safe_div(pos, jnp.linalg.norm(pos, axis=-1, keepdims=True))
        rmesh = jnp.linalg.norm(pos, axis=-1)
    else:
        if los is None:
            los = safe_div(box_center, jnp.linalg.norm(box_center))
        los = jnp.asarray(los) / jnp.linalg.norm(los)
        los_mesh = los
        rmesh = jnp.abs((pos * los).sum(-1))

    if a_obs is None:
        a = chi2a(cosmo, rmesh)
    else:
        a = a_obs
    return los_mesh, a


def samp2base(params: dict, config, inv=False, temp=1.0) -> dict:
    """
    Transform sample params into base params.
    """
    out = {}
    for in_name, value in params.items():
        name = in_name if inv else in_name[:-1]
        out_name = in_name + "_" if inv else in_name[:-1]

        conf = config[name]
        low, high = conf.get("low", -jnp.inf), conf.get("high", jnp.inf)
        loc_fid, scale_fid = conf["loc_fid"], conf["scale_fid"]
        scale_fid *= temp**0.5

        # Reparametrize
        if not inv:
            if low != -jnp.inf or high != jnp.inf:
                out[out_name] = std2trunc(value, loc_fid, scale_fid, low, high)
            else:
                out[out_name] = value * scale_fid + loc_fid
        else:
            if low != -jnp.inf or high != jnp.inf:
                out[out_name] = trunc2std(value, loc_fid, scale_fid, low, high)
            else:
                out[out_name] = (value - loc_fid) / scale_fid
    return out


def samp2base_mesh(init: dict, precond=False, transfer=None, inv=False, temp=1.0) -> dict:
    """
    Transform sample mesh into base mesh, i.e. initial wavevector coefficients at a=1.
    """
    assert len(init) <= 1, "init dict should only have one or zero key"
    for in_name, mesh in init.items():
        out_name = in_name + "_" if inv else in_name[:-1]
        transfer *= temp**0.5

        # Reparametrize
        if not inv:
            if precond == "direct":
                # Sample in direct space
                mesh = jnp.fft.rfftn(mesh)

            elif precond in ["fourier", "kaiser", "kaiser_dyn"]:
                # Sample in fourier space
                mesh = rg2cgh(mesh)

            mesh *= transfer  # ~ CN(0, P)
        else:
            mesh = safe_div(mesh, transfer)

            if precond == "direct":
                mesh = jnp.fft.irfftn(mesh)

            elif precond in ["fourier", "kaiser", "kaiser_dyn"]:
                mesh = cgh2rg(mesh)

        return {out_name: mesh}
    return {}


def lagrangian_weights(
    cosmo: Cosmology, a, pos, box_shape, b1, b2, bs2, bn2, init_mesh,
    fNL_bp=0.0, fNL_bpd=0.0, png_type=None,
):
    """
    Return Lagrangian bias expansion weights as in [Modi+2020](http://arxiv.org/abs/1910.07097).
    .. math::

        w = 1 + b_1 \\delta + \\frac{b_2}{2} \\left(\\delta^2 - \\braket{\\delta^2}\\right) + b_{s^2} \\left(s^2 - \\braket{s^2}\\right) + b_{\\nabla^2} \\nabla^2 \\delta + b_{\\phi} f_\\mathrm{NL} \\phi + b_{\\phi \\delta} f_\\mathrm{NL} \\left(\\phi \\delta - \\braket{\\phi \\delta}\\right)

    When ``png_type`` is set, ``init_mesh`` must be the original Gaussian linear field
    (before ``add_png``), so the primordial potential phi is recovered correctly.
    """
    delta_k = init_mesh
    delta = jnp.fft.irfftn(delta_k)
    growths = a2g(cosmo, a)
    if jnp.ndim(growths) > 1:
        growths = growths.squeeze()
    # Smooth field to mitigate negative weights or TODO: use gaussian lagrangian biases
    # k_nyquist = jnp.pi * jnp.min(mesh_shape / box_shape)
    # delta_k = delta_k * jnp.exp( - kk_box / k_nyquist**2)
    # delta = jnp.fft.irfftn(delta_k)

    mesh_shape = delta.shape
    kvec = rfftk(mesh_shape)
    kk_box = sum(
        (ki * (m / box_l)) ** 2 for ki, m, box_l in zip(kvec, mesh_shape, box_shape, strict=False)
    )  # minus laplace kernel in h/Mpc physical units

    # Init weights
    weights = 1.0

    # Apply b1, punctual term
    delta_part = cic_read(delta, pos) * growths
    weights = weights + b1 * delta_part

    # Apply primordial non-Gaussianity scale-dependent bias terms
    if png_type is not None:
        trans_phi2delta = trans_phi2delta_interp(cosmo)(kk_box**0.5)
        phi = jnp.fft.irfftn(safe_div(delta_k, trans_phi2delta))

        # Apply bphi, primordial term (phi is primordial: no growth factor)
        phi_part = cic_read(phi, pos)
        weights = weights + fNL_bp * phi_part

        # Apply bphidelta, primordial-density cross term
        phi_delta_part = phi_part * delta_part
        weights = weights + fNL_bpd * (phi_delta_part - phi_delta_part.mean())

    # Apply b2, punctual term (montecosmo convention: b2/2 * (delta^2 - <delta^2>))
    delta2_part = delta_part**2
    weights = weights + b2 * (delta2_part - delta2_part.mean()) / 2

    # Apply bshear2, non-punctual term. Use physical wavevectors so the ratios
    # k_i k_j / k^2 stay correct even for anisotropic cells.
    kvec_phys = [
        ki * (m / box_l) for ki, m, box_l in zip(kvec, mesh_shape, box_shape, strict=False)
    ]
    pot_k = delta_k * invlaplace_kernel(kvec_phys)

    shear2 = 0
    for i, ki in enumerate(kvec_phys):
        # Add diagonal terms
        shear2 = shear2 + jnp.fft.irfftn(-(ki**2) * pot_k - delta_k / 3) ** 2
        for kj in kvec_phys[i + 1 :]:
            # Add strict-up-triangle terms (counted twice)
            shear2 = shear2 + 2 * jnp.fft.irfftn(-ki * kj * pot_k) ** 2

    shear2_part = cic_read(shear2, pos) * growths**2
    weights = weights + bs2 * (shear2_part - shear2_part.mean())

    # Apply bnabla2, non-punctual term
    delta_nl = jnp.fft.irfftn(-kk_box * delta_k)

    delta_nl_part = cic_read(delta_nl, pos) * growths
    weights = weights + bn2 * delta_nl_part

    return weights


def lagrangian_fog_velocity(cosmo: Cosmology, a, pos, box_shape, bnpar, init_mesh):
    r"""
    Finger-of-God higher-derivative velocity term (b_{\nabla_\parallel}).
    """
    delta_k = init_mesh
    mesh_shape = ch2rshape(init_mesh.shape)
    growths = a2g(cosmo, a)
    if jnp.ndim(growths) > 1:
        growths = growths.squeeze()
    kvec = rfftk(mesh_shape)
    grad_parts = []
    for ki, m, box_l in zip(kvec, mesh_shape, box_shape, strict=False):
        k_phys = ki * (m / box_l)  # physical wavevector component in h/Mpc
        grad = jnp.fft.irfftn(1j * k_phys * delta_k)
        grad_parts.append(cic_read(grad, pos))
    grad_pos = jnp.stack(grad_parts, axis=-1)  # (N, 3) = grad(delta_L) at particle positions
    g = jnp.asarray(growths)
    g = g[..., None] if g.ndim else g
    return bnpar * grad_pos * g


def rsd(cosmo: Cosmology, vel, los, a, box_shape, mesh_shape, dvel=0.0):
    """
    Redshift-Space Distortion (RSD) displacement from cosmology and growth-time integrator velocity.
    Computed with respect to scale factor(s) and line-of-sight(s).

    ``dvel`` is an optional extra physical-velocity 3-vector (per particle) added before the
    LOS projection — used for the Finger-of-God b_nabla_parallel term (see
    ``lagrangian_fog_velocity``). It is projected onto the same observer-dependent ``los``.

    No RSD if los is None.
    """
    if los is None:
        return jnp.zeros_like(vel)

    los = jnp.asarray(los)
    vel = cell2phys_vel(vel, box_shape, mesh_shape)

    growth = a2g(cosmo, a) * a2f(cosmo, a)
    if jnp.ndim(growth) == 1:
        growth = growth[:, None]
    vel = vel * growth

    vel = vel + dvel  # Finger-of-God higher-derivative velocity (physical h/Mpc)

    dpos = (vel * los).sum(-1, keepdims=True) * los
    return phys2cell_vel(dpos, box_shape, mesh_shape)


def naive_mu2_delta(delta_k, los, mesh_shape):
    """
    Return mu^2 * delta for a possibly spatially varying line-of-sight field.

    This implements n_i n_j d_i d_j nabla^{-2} delta in real space.
    """
    los = jnp.asarray(los)
    if los.ndim == 1:
        los = los / jnp.linalg.norm(los)
    else:
        los = safe_div(los, jnp.linalg.norm(los, axis=-1, keepdims=True))

    kvec = rfftk(mesh_shape)
    pot_k = delta_k * invlaplace_kernel(kvec)

    mu2_delta = jnp.zeros(tuple(mesh_shape))
    for i, ki in enumerate(kvec):
        mu2_delta = mu2_delta + los[..., i] ** 2 * jnp.fft.irfftn(-(ki**2) * pot_k)
        for j in range(i + 1, 3):
            mu2_delta = mu2_delta + 2.0 * los[..., i] * los[..., j] * jnp.fft.irfftn(
                -ki * kvec[j] * pot_k
            )

    return mu2_delta


def kaiser_boost(
    cosmo: Cosmology, a, bE, mesh_shape, los: np.ndarray = None,
    fNL_bp=0.0, png_type=None, box_shape=None,
):
    """
    Return Eulerian Kaiser boost including linear growth, Eulerian linear bias, RSD,
    and (optionally) the local-PNG scale-dependent bias fNL_bp / M(k).

    No RSD if los is None.
    """
    if los is None:
        boost = a2g(cosmo, a) * bE
    else:
        los = jnp.asarray(los)
        los = safe_div(los, jnp.linalg.norm(los))
        kvec = rfftk(mesh_shape)
        kmesh = sum(kk**2 for kk in kvec) ** 0.5  # in cell units
        mumesh = sum(ki * losi for ki, losi in zip(kvec, los, strict=False))
        mumesh = safe_div(mumesh, kmesh)

        boost = a2g(cosmo, a) * (bE + a2f(cosmo, a) * mumesh**2)

    if png_type is not None:
        trans = trans_phi2delta_interp(cosmo)(_kmesh_phys(mesh_shape, box_shape))
        boost = boost + safe_div(fNL_bp, trans)  # delta += fNL_bp * phi in Fourier space
    return boost


def kaiser_model(
    cosmo: Cosmology, a, bE, init_mesh, los: np.ndarray = None,
    fNL_bp=0.0, png_type=None, box_shape=None,
):
    """
    Kaiser model, with linear growth, Eulerian linear bias, RSD, and (optionally) the
    local-PNG scale-dependent bias fNL_bp / M(k).
    """
    mesh_shape = ch2rshape(init_mesh.shape)

    # Fast path: scalar time and fixed LOS (standard snapshot Kaiser in Fourier space).
    if jnp.ndim(a) == 0 and (los is None or jnp.ndim(los) == 1):
        boost = kaiser_boost(cosmo, a, bE, mesh_shape, los, fNL_bp, png_type, box_shape)
        return 1 + jnp.fft.irfftn(init_mesh * boost)  # 1 + delta

    delta_lin = jnp.fft.irfftn(init_mesh)
    if los is None:
        mu2_delta = jnp.zeros_like(delta_lin)
    elif jnp.ndim(los) == 1:
        los = jnp.asarray(los)
        los = safe_div(los, jnp.linalg.norm(los))
        kvec = rfftk(mesh_shape)
        kmesh = sum(kk**2 for kk in kvec) ** 0.5
        mumesh = sum(ki * losi for ki, losi in zip(kvec, los, strict=False))
        mumesh = safe_div(mumesh, kmesh)
        mu2_delta = jnp.fft.irfftn(mumesh**2 * init_mesh)
    else:
        mu2_delta = naive_mu2_delta(init_mesh, los, mesh_shape)

    delta = bE * delta_lin + a2f(cosmo, a) * mu2_delta
    delta = a2g(cosmo, a) * delta

    if png_type is not None:
        trans = trans_phi2delta_interp(cosmo)(_kmesh_phys(mesh_shape, box_shape))
        phi = jnp.fft.irfftn(safe_div(init_mesh, trans))
        delta = delta + fNL_bp * phi  # scale-dependent bias (lightcone / curved-sky)
    return 1 + delta


# =====================================================================
# Abacus catalog loading
# =====================================================================


def paint_arbitrary(mesh, positions, weights=None):
    """CIC-paint an arbitrary set of N particles into a mesh.

    Parameters
    ----------
    mesh : jnp.ndarray
        Zero-initialised mesh of shape ``(Nx, Ny, Nz)``.
    positions : array (N, 3)
        Particle positions in cell units (float).
    weights : array (N,) or scalar, optional
        Per-particle weights.  If None, every particle contributes 1.

    Returns
    -------
    jnp.ndarray  shape (Nx, Ny, Nz)
    """
    positions = jnp.asarray(positions)
    pmid = jnp.floor(positions).astype(jnp.int32) % jnp.array(mesh.shape)
    disp = positions - jnp.floor(positions)
    val = 1.0 if weights is None else jnp.asarray(weights)
    return scatter(pmid, disp, mesh, val=val)


def interlace_paint_deconv(pos_paint, paint_shape, final_shape, weights=None):
    """CIC paint with interlacing (order 2), CIC deconvolution, and Fourier crop.

    Shared implementation used by both ``FieldLevelModel.paint_and_deconv`` and
    ``catalog2mesh``.  Positions must already be in ``paint_shape`` cell units.
    """
    kvec = rfftk(paint_shape)
    kvec_sum = sum(kvec)  # precomputed — constant across interlace shifts
    meshk = jnp.zeros(r2chshape(paint_shape), dtype=complex)
    interlace_order = 2
    for i_shift in range(interlace_order):
        shift = i_shift / interlace_order  # 0.0, then 0.5
        mesh_i = paint_arbitrary(
            jnp.zeros(paint_shape), pos_paint + shift, weights
        )
        phase = jnp.exp(1j * shift * kvec_sum)
        meshk = meshk + jnp.fft.rfftn(mesh_i) * phase / interlace_order
    meshk = meshk / paint_kernel(kvec, order=2)
    meshk = meshk * jnp.prod(jnp.array(paint_shape) / jnp.array(final_shape))
    meshk = chreshape(meshk, r2chshape(final_shape))
    return jnp.fft.irfftn(meshk, s=final_shape)


def radecz2cart(cosmo: Cosmology, radecz: dict):
    """
    Convert {RA, DEC, Z} dictionary (degrees / redshift) to Cartesian (Mpc/h).

    Follows montecosmo convention: comoving distance via ``a2chi``.
    """
    ra = jnp.asarray(radecz["RA"], dtype=float)
    dec = jnp.asarray(radecz["DEC"], dtype=float)
    radius = a2chi(cosmo, 1.0 / (1.0 + jnp.asarray(radecz["Z"], dtype=float)))
    return radecrad2cart(ra, dec, radius)


def catalog2positions(
    path: str | Path,
    cosmo: Cosmology,
    observer_position,
    box_shape,
    mesh_shape,
    z_range: tuple[float, float] | None = None,
):
    """Read a galaxy catalog and return positions in cell-units, filtered to the box.

    This is the low-level reader used by both ``catalog2mesh`` (single-file
    painting) and ``load_abacus_galaxy_observation`` (multi-file accumulation
    before a single deconvolution pass).

    The observer is placed at ``observer_position`` within the simulation box
    (in Mpc/h).  Typically ``observer_position = [Lx/2, Ly/2, 0]`` so the
    box occupies ``[0, Lx] x [0, Ly] x [0, Lz]`` in comoving Cartesian
    coordinates with the observer at the z=0 face.

    Parameters
    ----------
    observer_position : array-like (3,)
        Observer position in Mpc/h inside the simulation box.
    z_range : tuple[float, float] | None
        If provided, keep only objects with z_min <= z < z_max.
        Used to avoid double-counting at AbacusLensing shell boundaries.

    Returns
    -------
    pos : jnp.ndarray, shape (N, 3)
        Galaxy positions in cell-unit coordinates.
    n_in_box : int
        Number of galaxies inside the volume.
    chi_inbox_min, chi_inbox_max : float
        Comoving distance range of accepted galaxies.
    """
    path = Path(path)
    if path.suffix == ".asdf":
        import asdf as _asdf

        with _asdf.open(str(path)) as f:
            data = {
                "RA": np.array(f["data"]["RA"]),
                "DEC": np.array(f["data"]["DEC"]),
                "Z": np.array(f["data"].get("Z_RSD", f["data"].get("Z_COSMO"))),
            }
            if z_range is not None and "Z_COSMO" in f["data"]:
                data["Z_COSMO"] = np.array(f["data"]["Z_COSMO"])
    else:
        import fitsio

        data = fitsio.read(str(path), columns=["RA", "DEC", "Z"])

    if z_range is not None:
        z_min, z_max = z_range
        z_dedup = data.get("Z_COSMO", data["Z"])
        z_sel = (z_dedup >= z_min) & (z_dedup < z_max)
        data = {k: v[z_sel] for k, v in data.items()}

    pos = radecz2cart(cosmo, data)

    box_shape_arr = jnp.asarray(box_shape, dtype=float)
    mesh_shape_arr = jnp.asarray(mesh_shape, dtype=float)
    box_center = box_shape_arr / 2.0 - jnp.asarray(observer_position, dtype=float)

    chi_all = jnp.linalg.norm(pos, axis=-1)
    pos = phys2cell_pos(pos, box_center, box_shape_arr, mesh_shape_arr)

    inside = jnp.all((pos >= 0) & (pos < mesh_shape_arr - 1), axis=-1)

    chi_in = chi_all[inside]
    pos = pos[inside]
    n_in_box = int(inside.sum())
    chi_inbox_min = float(jnp.min(chi_in)) if n_in_box > 0 else 0.0
    chi_inbox_max = float(jnp.max(chi_in)) if n_in_box > 0 else 0.0

    return pos, n_in_box, chi_inbox_min, chi_inbox_max


def randoms2positions(
    path: str | Path,
    cosmo: Cosmology,
    observer_position,
    box_shape,
    mesh_shape,
    z_range: tuple[float, float] | None = None,
):
    """Read randoms from an AbacusLensing ASDF file (RAND_RA/DEC/Z columns).

    Same geometry pipeline as ``catalog2positions`` but reads the
    ``RAND_*`` columns instead of the galaxy columns.  Uses
    ``observer_position`` to derive the box_center (see ``catalog2positions``).

    Parameters
    ----------
    observer_position : array-like (3,)
        Observer position in Mpc/h inside the simulation box.
    z_range : tuple[float, float] | None
        If provided, keep only objects with z_min <= z < z_max.

    Returns
    -------
    pos : jnp.ndarray, shape (N, 3)
        Random positions in cell-unit coordinates.
    n_in_box : int
        Number of randoms inside the volume.
    """
    path = Path(path)
    if path.suffix != ".asdf":
        raise ValueError(f"randoms2positions only supports ASDF files, got: {path}")

    import asdf as _asdf

    with _asdf.open(str(path)) as f:
        data = {
            "RA": np.array(f["data"]["RAND_RA"]),
            "DEC": np.array(f["data"]["RAND_DEC"]),
            "Z": np.array(f["data"]["RAND_Z"]),
        }

    if z_range is not None:
        z_min, z_max = z_range
        z_sel = (data["Z"] >= z_min) & (data["Z"] < z_max)
        data = {k: v[z_sel] for k, v in data.items()}

    pos = radecz2cart(cosmo, data)

    box_shape_arr = jnp.asarray(box_shape, dtype=float)
    mesh_shape_arr = jnp.asarray(mesh_shape, dtype=float)
    box_center = box_shape_arr / 2.0 - jnp.asarray(observer_position, dtype=float)

    pos = phys2cell_pos(pos, box_center, box_shape_arr, mesh_shape_arr)
    inside = jnp.all((pos >= 0) & (pos < mesh_shape_arr - 1), axis=-1)

    pos = pos[inside]
    n_in_box = int(inside.sum())
    return pos, n_in_box


def paint_catalog_fits(
    path: str | Path,
    cosmo: Cosmology,
    observer_position,
    box_shape,
    mesh_shape,
    paint_oversamp: float = 1.0,
):
    """Paint a galaxy FITS catalog (RA/DEC/Z) onto a 3-D mesh.

    FITS → Cartesian → cell coords → CIC.
    Galaxies outside the mesh are silently discarded (non-periodic boundaries).

    When ``paint_oversamp > 1.0`` the painting is done on an oversampled grid
    and then passed through the same interlace + deconvolution + Fourier-crop
    pipeline as ``model.paint_and_deconv``.

    Parameters
    ----------
    path : str or Path
        Path to a FITS file with columns ``RA``, ``DEC``, ``Z``.
    cosmo : Cosmology
        Fiducial cosmology for the redshift→distance conversion.
    observer_position : array-like (3,)
        Observer position in Mpc/h inside the simulation box.
    box_shape : array-like (3,)
        Physical box size in Mpc/h.
    mesh_shape : array-like (3,)
        Number of cells along each axis.
    paint_oversamp : float, optional
        Oversampling factor for the paint grid.  ``1.0`` (default) = plain CIC.
        ``2.0`` = interlace + deconvolve (recommended).

    Returns
    -------
    mesh : jnp.ndarray, shape ``mesh_shape``
        Painted galaxy count field.
    n_in_box : int
        Number of galaxies that fell inside the mesh volume.
    mesh_plain : jnp.ndarray, shape ``mesh_shape``
        Plain CIC count field (always ≥ 0, used for occupancy masking).
    chi_inbox_min, chi_inbox_max : float
        Minimum and maximum comoving distance of galaxies inside the mesh.
    """
    path = Path(path)
    pos, n_in_box, chi_inbox_min, chi_inbox_max = catalog2positions(
        path=path,
        cosmo=cosmo,
        observer_position=observer_position,
        box_shape=box_shape,
        mesh_shape=mesh_shape,
    )

    final_shape = tuple(int(n) for n in mesh_shape)
    mesh_plain = paint_arbitrary(jnp.zeros(final_shape), pos)

    if paint_oversamp <= 1.0:
        return mesh_plain, n_in_box, mesh_plain, chi_inbox_min, chi_inbox_max

    paint_shape = get_scaled_shape(tuple(int(n) for n in mesh_shape), paint_oversamp)
    scale = jnp.asarray([paint_shape[i] / final_shape[i] for i in range(3)], dtype=pos.dtype)
    pos_paint = pos * scale  # positions in paint_shape cell units
    mesh = interlace_paint_deconv(pos_paint, paint_shape, final_shape)

    return mesh, n_in_box, mesh_plain, chi_inbox_min, chi_inbox_max
