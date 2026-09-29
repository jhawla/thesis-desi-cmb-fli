"""
Unit tests for the curved-sky CMB lensing module.

Covers: compute_healpix_mask, convergence_Born_spherical,
        compute_sigma_hp, lensing_kernel, prepare_abacus_kappa_hp.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax_cosmo import Cosmology

from desi_cmb_fli.bricks import regular_pos
from desi_cmb_fli.cmb_lensing import (
    _box_ray_intervals,
    compute_healpix_mask,
    compute_shell_support_fractions,
    compute_sigma_hp,
    convergence_Born_spherical,
    lensing_kernel,
    prepare_abacus_kappa_hp,
)

jax.config.update("jax_enable_x64", True)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def cosmo():
    return Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.70,
        sigma8=0.80,
        n_s=0.96,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
    )


@pytest.fixture
def small_config():
    """Minimal box config for fast tests."""
    Lx, Ly, Lz = 450.0, 450.0, 800.0   # Mpc/h — small Lz for speed
    mesh = (32, 32, 64)
    nside = 32
    n_shells = 8
    observer_pos = np.array([Lx / 2, Ly / 2, 0.0])
    box_shape = np.array([Lx, Ly, Lz])
    return {
        "box_shape": box_shape,
        "mesh_shape": mesh,
        "nside": nside,
        "n_shells": n_shells,
        "observer_pos": observer_pos,
    }


@pytest.fixture
def shells(cosmo, small_config):
    """Radial shells for Born integration."""
    import jax_cosmo as jc

    Lz = float(small_config["box_shape"][2])
    n = small_config["n_shells"]
    dr = Lz / n
    r_shells = np.linspace(dr / 2.0, Lz - dr / 2.0, n)
    a_shells = np.array([float(jc.background.a_of_chi(cosmo, r)[0]) for r in r_shells])
    return r_shells, a_shells, dr


@pytest.fixture
def mask(small_config):
    return compute_healpix_mask(
        small_config["observer_pos"],
        small_config["box_shape"],
        small_config["nside"],
    )


@pytest.fixture
def ray_intervals(small_config):
    return _box_ray_intervals(
        small_config["observer_pos"],
        small_config["box_shape"],
        small_config["nside"],
    )


@pytest.fixture
def uniform_pos(small_config):
    """Uniform particle grid in cell units."""
    pos = regular_pos(tuple(small_config["mesh_shape"]))
    return pos.astype(jnp.float64)


# =========================================================================
# compute_healpix_mask
# =========================================================================


def test_healpix_mask_nonzero(small_config):
    """Mask must cover at least some pixels."""
    mask = compute_healpix_mask(
        small_config["observer_pos"],
        small_config["box_shape"],
        small_config["nside"],
    )
    n_pix = int(np.sum(mask))
    assert n_pix > 0, "Mask is empty"


def test_healpix_mask_fraction(small_config):
    """Footprint fraction should be << 1 (small box, not full sky)."""
    import healpy as hp

    mask = compute_healpix_mask(
        small_config["observer_pos"],
        small_config["box_shape"],
        small_config["nside"],
    )
    frac = float(np.sum(mask)) / hp.nside2npix(small_config["nside"])
    # Observer at z=0 face center sees ~50% of sky (upper hemisphere);
    # allow up to 55% for pixel discretisation at the equator.
    assert frac < 0.55, f"Mask fraction too large: {frac:.2%}"
    assert frac > 0.001, f"Mask fraction too small: {frac:.2%}"


def test_healpix_mask_bool_dtype(small_config):
    """Mask must be boolean."""
    mask = compute_healpix_mask(
        small_config["observer_pos"],
        small_config["box_shape"],
        small_config["nside"],
    )
    assert mask.dtype == bool, f"Expected bool mask, got {mask.dtype}"


# =========================================================================
# lensing_kernel
# =========================================================================


def test_lensing_kernel_positive(cosmo):
    """Kernel should be non-negative in [0, chi_s)."""
    import jax_cosmo as jc

    chi_s = float(jc.background.radial_comoving_distance(cosmo, 1.0 / 1101.0)[0])
    chi = jnp.linspace(10.0, chi_s * 0.99, 50)
    a = jc.background.a_of_chi(cosmo, chi)
    W = lensing_kernel(cosmo, chi, a, chi_s)
    assert jnp.all(W >= 0), "Lensing kernel has negative values"


def test_lensing_kernel_zero_at_source(cosmo):
    """Kernel should be 0 at the source plane."""
    import jax_cosmo as jc

    chi_s = float(jc.background.radial_comoving_distance(cosmo, 1.0 / 1101.0)[0])
    a_s = float(jc.background.a_of_chi(cosmo, jnp.array([chi_s]))[0])
    W = lensing_kernel(cosmo, jnp.array(chi_s), jnp.array(a_s), chi_s)
    assert float(W) == pytest.approx(0.0, abs=1e-12)


# =========================================================================
# convergence_Born_spherical
# =========================================================================


def test_born_spherical_shape(cosmo, small_config, shells, mask, ray_intervals, uniform_pos):
    """Output shape must be (n_pix_mask,)."""
    r_shells, a_shells, dr = shells
    t_enter, t_exit = ray_intervals
    kappa = convergence_Born_spherical(
        cosmo,
        uniform_pos,
        small_config["box_shape"],
        small_config["mesh_shape"],
        small_config["observer_pos"],
        r_shells,
        a_shells,
        dr,
        small_config["nside"],
        mask,
        z_source=1100.0,
        t_enter=t_enter,
        t_exit=t_exit,
    )
    n_pix_mask = int(np.sum(mask))
    assert kappa.shape == (n_pix_mask,), f"Expected ({n_pix_mask},), got {kappa.shape}"


def test_born_spherical_uniform_zero(cosmo, small_config, shells, mask, ray_intervals, uniform_pos):
    import healpy as hp

    r_shells, a_shells, dr = shells
    t_enter, t_exit = ray_intervals
    nside = small_config["nside"]
    kappa_full = convergence_Born_spherical(
        cosmo,
        uniform_pos,
        small_config["box_shape"],
        small_config["mesh_shape"],
        small_config["observer_pos"],
        r_shells,
        a_shells,
        dr,
        nside,
        mask,
        z_source=1100.0,
        t_enter=t_enter,
        t_exit=t_exit,
        return_full=True,
    )
    # Masked pseudo-spectrum, mirroring the likelihood's masked-alm construction.
    w_kappa = np.asarray(mask, dtype=float) * np.asarray(kappa_full)
    cl = hp.anafast(w_kappa, lmax=3 * nside - 1)
    rms_l0 = float(np.sqrt(cl[0]))                 # monopole (discarded)
    rms_l2 = float(np.sqrt(np.max(cl[2:])))        # worst l>=2 mode (kept)
    # A uniform density must give kappa ~ 0: bound the leakage in absolute terms
    # (typical kappa signal ~1e-2). A ratio-to-monopole test would wrongly pass a
    # map with a large spurious monopole from partially covered shells.
    assert rms_l0 < 1e-3, f"Uniform field leaks a large monopole: sqrt(C_0)={rms_l0:.3e}"
    assert rms_l2 < 1e-3, (
        f"Uniform field leaks too much l>=2 power: sqrt(max C_l>=2)={rms_l2:.3e}"
    )


def test_born_spherical_finite(cosmo, small_config, shells, mask, ray_intervals):
    """Output should be finite for random positions."""
    r_shells, a_shells, dr = shells
    t_enter, t_exit = ray_intervals
    key = jr.key(0)
    Nx, Ny, Nz = small_config["mesh_shape"]
    N = 1000
    pos = jr.uniform(key, shape=(N, 3)) * jnp.array([Nx, Ny, Nz], dtype=float)

    kappa = convergence_Born_spherical(
        cosmo,
        pos,
        small_config["box_shape"],
        small_config["mesh_shape"],
        small_config["observer_pos"],
        r_shells,
        a_shells,
        dr,
        small_config["nside"],
        mask,
        z_source=1100.0,
        t_enter=t_enter,
        t_exit=t_exit,
    )
    assert jnp.all(jnp.isfinite(kappa)), "kappa has NaN/Inf for random positions"


def test_born_spherical_gradient_nonzero(cosmo, small_config, shells, mask, ray_intervals):
    """The curved-sky projector must provide usable position gradients."""
    r_shells, a_shells, dr = shells
    t_enter, t_exit = ray_intervals
    pos = jnp.array(
        [
            [3.2, 3.1, 1.5],
            [4.4, 4.7, 3.0],
            [2.1, 5.0, 6.2],
        ],
        dtype=jnp.float64,
    )

    def loss(pos_):
        kappa = convergence_Born_spherical(
            cosmo,
            pos_,
            small_config["box_shape"],
            small_config["mesh_shape"],
            small_config["observer_pos"],
            r_shells,
            a_shells,
            dr,
            small_config["nside"],
            mask,
            z_source=1100.0,
            t_enter=t_enter,
            t_exit=t_exit,
        )
        return jnp.sum(kappa**2)

    grad = jax.grad(loss)(pos)

    assert jnp.all(jnp.isfinite(grad)), "gradient has NaN/Inf"
    assert float(jnp.sum(jnp.abs(grad))) > 0.0, "gradient is identically zero"


def test_shell_support_fractions_bounded(small_config, shells, mask):
    """Relative shell support factors must be finite and bounded."""
    r_shells, _, dr = shells
    w = compute_shell_support_fractions(
        small_config["observer_pos"],
        small_config["box_shape"],
        small_config["nside"],
        r_shells,
        dr,
        final_mask=mask,
    )
    assert np.all(np.isfinite(w))
    assert np.all(w >= 0.0)
    assert np.all(w <= 1.0 + 1e-6)


# =========================================================================
# compute_sigma_hp
# =========================================================================


def test_sigma_hp_positive():
    """sigma_hp must be strictly positive for a realistic N_ell."""
    ell = np.arange(2, 500, dtype=float)
    nell = 1e-7 * (ell / 100.0) ** (-2.0)  # simple power-law noise
    sigma = compute_sigma_hp(ell, nell, nside=64)
    assert sigma > 0.0, f"sigma_hp should be positive, got {sigma}"


def test_sigma_hp_increases_with_noise():
    """More noise power -> larger sigma_hp."""
    ell = np.arange(2, 500, dtype=float)
    nell_lo = 1e-7 * np.ones_like(ell)
    nell_hi = 1e-6 * np.ones_like(ell)
    sig_lo = compute_sigma_hp(ell, nell_lo, nside=64)
    sig_hi = compute_sigma_hp(ell, nell_hi, nside=64)
    assert sig_hi > sig_lo, "Higher N_ell should give larger sigma_hp"


def test_sigma_hp_with_extra_cl():
    """Adding cl_extra_1d must increase sigma_hp."""
    ell = np.arange(2, 300, dtype=float)
    nell = 1e-7 * np.ones_like(ell)
    nside = 64
    ell_max = 2 * nside
    cl_extra = 1e-8 * np.ones(ell_max + 1)

    sig_base = compute_sigma_hp(ell, nell, nside)
    sig_with = compute_sigma_hp(ell, nell, nside, cl_extra_1d=cl_extra)
    assert sig_with >= sig_base, "Adding cl_extra should not decrease sigma_hp"


# =========================================================================
# prepare_abacus_kappa_hp
# =========================================================================


def test_prepare_abacus_kappa_hp_shape():
    """Output shape should match n_pix_mask."""
    import healpy as hp

    nside = 32
    npix = hp.nside2npix(nside)
    healpix_map = np.random.randn(npix)
    mask = np.zeros(npix, dtype=bool)
    mask[:npix // 4] = True  # first quarter active

    out = prepare_abacus_kappa_hp(healpix_map, nside, mask)
    assert out.shape == (npix // 4,), f"Expected ({npix // 4},), got {out.shape}"


def test_prepare_abacus_kappa_hp_ud_grade():
    """Function should handle input nside != target nside."""
    import healpy as hp

    nside_in = 64
    nside_out = 32
    npix_in = hp.nside2npix(nside_in)
    npix_out = hp.nside2npix(nside_out)
    healpix_map = np.random.randn(npix_in)
    mask = np.ones(npix_out, dtype=bool)

    out = prepare_abacus_kappa_hp(healpix_map, nside_out, mask)
    assert out.shape == (npix_out,)
    assert np.all(np.isfinite(out))


def test_prepare_abacus_kappa_hp_values():
    """Extracted values should be a subset of the map values."""
    import healpy as hp

    nside = 16
    npix = hp.nside2npix(nside)
    healpix_map = np.arange(npix, dtype=float)
    mask = np.zeros(npix, dtype=bool)
    mask[10:20] = True

    out = prepare_abacus_kappa_hp(healpix_map, nside, mask)
    expected = healpix_map[mask]
    np.testing.assert_array_equal(out, expected)


def test_project_mesh_to_healpix_uses_the_model_grid_convention():
    """Mesh node i sits at i * dx (painting, Born projector); the galaxy projection must agree,
    or the galaxy map is displaced from the kappa map by half a cell per axis. On the field
    f = x - x_obs the projection is then exactly odd under n -> -n."""
    import healpy as hp

    from desi_cmb_fli.cmb_lensing import project_mesh_to_healpix

    n, L, nside = 16, 1600.0, 4
    dx = L / n
    x = np.arange(n) * dx - L / 2
    mesh = np.broadcast_to(x[:, None, None], (n, n, n)).copy()
    box = np.full(3, L)
    npix = hp.nside2npix(nside)
    proj = project_mesh_to_healpix(mesh, box, box / 2, nside, np.ones(npix, bool),
                                   chi_max=L / 2 - 2 * dx)
    antipode = hp.vec2pix(nside, *(-np.array(hp.pix2vec(nside, np.arange(npix)))))
    np.testing.assert_allclose(proj + proj[antipode], 0.0, atol=1e-6 * np.abs(proj).max())


def test_healpix_counts_conserves_weight_in_both_assignments():
    from desi_cmb_fli.cmb_lensing import healpix_counts

    rng = np.random.default_rng(1)
    pos = rng.normal(size=(5000, 3)) * 100.0
    w = rng.uniform(0.5, 1.5, size=5000)
    for bilinear in (False, True):
        m = healpix_counts(pos, 8, weights=w, bilinear=bilinear)
        np.testing.assert_allclose(m.sum(), w.sum(), rtol=1e-12)


# ---------------------------------------------------------------------------
# Abacus kappa through the projector's own angular kernel
# ---------------------------------------------------------------------------


def test_the_resampling_kernel_is_the_born_projector_kernel():
    """bilinear_resample_healpix (healpy) and convergence_Born_spherical (jax_healpy) must spread a
    point over the same four pixels with the same weights, or data and model see different
    angular windows."""
    import healpy as hp
    import jax_healpy as jhp

    rng = np.random.default_rng(3)
    theta = np.arccos(rng.uniform(-1, 1, 500))
    phi = rng.uniform(0, 2 * np.pi, 500)
    p_hp, w_hp = hp.get_interp_weights(16, theta, phi)
    p_j, w_j = jhp.get_interp_weights(16, jnp.asarray(theta), jnp.asarray(phi))
    order_hp, order_j = np.argsort(p_hp, axis=0), np.argsort(np.asarray(p_j), axis=0)
    np.testing.assert_array_equal(np.take_along_axis(p_hp, order_hp, 0),
                                  np.take_along_axis(np.asarray(p_j), order_j, 0))
    np.testing.assert_allclose(np.take_along_axis(w_hp, order_hp, 0),
                               np.take_along_axis(np.asarray(w_j), order_j, 0), atol=1e-6)


def test_bilinear_resampling_keeps_a_constant_everywhere():
    """Including the polar-cap pixels, where HEALPix's bilinear weights sum 16.6 % high."""
    import healpy as hp

    from desi_cmb_fli.cmb_lensing import bilinear_resample_healpix

    const = np.full(hp.nside2npix(64), 0.37)
    np.testing.assert_allclose(bilinear_resample_healpix(const, 8), 0.37, rtol=1e-12)


def test_the_abacus_observation_lives_on_the_projection_sphere(tmp_path):
    """The loader returns maps on the projection sphere, and they pack through the fine branch,
    like the model's own kappa."""
    import asdf
    import healpy as hp

    from desi_cmb_fli.cmb_lensing import load_abacus_kappa_observation
    from desi_cmb_fli.model import FieldLevelModel, default_config

    cfg = default_config.copy()
    cfg.update({
        "mesh_shape": (8, 8, 8), "box_shape": (400.0, 400.0, 400.0), "evolution": "lpt",
        "lpt_order": 1, "a_obs": 1.0, "paint_oversamp": 1.0, "galaxies_enabled": False,
        "cmb_enabled": True, "cmb_nside": 4, "cmb_n_shells": 4, "cmb_observer_mode": "center",
        "cmb_proj_oversamp": 2, "cmb_shell_weights": "linear",
        "cmb_noise_nell": {"ell": np.arange(64, dtype=float), "N_ell": np.full(64, 1e-9)},
        "full_los_correction": True, "high_z_mode": "fixed", "chi_high_z_max": 300.0,
    })
    model = FieldLevelModel(**cfg)
    rng = np.random.default_rng(1)
    kappa_hi = rng.normal(scale=1e-2, size=hp.nside2npix(64))
    path = tmp_path / "kappa.asdf"
    asdf.AsdfFile({"data": {"kappa": kappa_hi}, "header": {}}).write_to(path)

    out = load_abacus_kappa_observation({"file": str(path), "noise_seed": 5}, model)
    npix_proj = hp.nside2npix(model.cmb_proj_nside)
    assert np.asarray(out["kappa_obs"]).size == npix_proj
    assert np.asarray(out["kappa_pred"]).size == npix_proj
    packed = np.asarray(model.pack_kappa_obs(jnp.asarray(out["kappa_obs"])))
    assert packed.size == model.cmb_u_dim and np.all(np.isfinite(packed))
