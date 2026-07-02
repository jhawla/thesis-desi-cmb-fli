"""Curved-sky validation integration tests + physics regression tests.

This file covers:
1. Cosmological constants and unit consistency
2. Lensing kernel amplitude and sign
3. C_ell^{kappa kappa} scaling with Omega_m and ell
4. Growth factor / growth rate
5. Packed pseudo-a_lm Gaussian observable (native dist.Normal consistency)
6. Gradient consistency: finite difference vs analytic (critical for HMC)
7. Model geometry (observer position, chi_boundary)
8. Curved-sky integration smoke tests
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import jax_cosmo as jc
import jax_cosmo.constants as constants
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from desi_cmb_fli.bricks import Planck18, get_cosmology  # noqa: E402
from desi_cmb_fli.cmb_lensing import (  # noqa: E402
    compute_theoretical_cl_kappa,
    lensing_kernel,
)
from desi_cmb_fli.model import FieldLevelModel, default_config  # noqa: E402
from desi_cmb_fli.nbody import a2f, a2g  # noqa: E402
from desi_cmb_fli.validation import plot_field_slices  # noqa: E402


@pytest.fixture(scope="module")
def planck18():
    cosmo = Planck18()
    cosmo._workspace = {}
    return cosmo


# =============================================================================
# 1. COSMOLOGICAL CONSTANTS AND UNITS
# =============================================================================


def test_jax_cosmo_constants_consistency():
    """Verify jax_cosmo constants match expected values."""
    assert jnp.isclose(constants.H0, 100.0, rtol=1e-5), f"H0={constants.H0}, expected 100"
    c_over_H0 = constants.c / constants.H0
    c_expected = 299792.458
    assert jnp.isclose(c_over_H0, c_expected / 100.0, rtol=1e-3), (
        f"c/H0 = {c_over_H0:.2f}, expected {c_expected/100:.2f}"
    )


def test_omega_m_sum_consistency(planck18):
    """Verify Omega_m = Omega_c + Omega_b."""
    omega_m_computed = planck18.Omega_c + planck18.Omega_b
    assert jnp.isclose(omega_m_computed, planck18.Omega_m, rtol=1e-6), (
        f"Omega_m mismatch: computed={float(omega_m_computed):.6f}, property={float(planck18.Omega_m):.6f}"
    )


def test_get_cosmology_omega_m_consistency():
    """Verify get_cosmology preserves Omega_m and sigma8 correctly."""
    cosmo = get_cosmology(Omega_m=0.3111, sigma8=0.81)
    assert jnp.isclose(cosmo.Omega_m, 0.3111, rtol=1e-6), (
        f"Omega_m mismatch: input=0.3111, got={float(cosmo.Omega_m):.6f}"
    )
    assert jnp.isclose(cosmo.sigma8, 0.81, rtol=1e-6), (
        f"sigma8 mismatch: input=0.81, got={float(cosmo.sigma8):.6f}"
    )


# =============================================================================
# 2. LENSING KERNEL AMPLITUDE AND SIGN
# =============================================================================


def test_lensing_prefactor_amplitude(planck18):
    """The lensing prefactor 3/2 * Omega_m * (H0/c)^2 should be ~5e-8 (Mpc/h)^-2."""
    prefactor = 1.5 * planck18.Omega_m * (constants.H0 / constants.c) ** 2
    assert 1e-9 < float(prefactor) < 1e-6, (
        f"Lensing prefactor outside expected range: {float(prefactor):.3e}"
    )


def test_lensing_kernel_positive_before_source(planck18):
    """W_kappa(chi) > 0 for chi < chi_source."""
    chi_source = 3000.0
    chi = 500.0
    a = 0.6
    w = lensing_kernel(planck18, chi, a, chi_source)
    assert float(w) > 0.0, f"Lensing kernel should be positive before source plane, got {float(w)}"


def test_lensing_kernel_zero_beyond_source(planck18):
    """W_kappa(chi) = 0 for chi >= chi_source (geometry factor clipped to 0)."""
    chi_source = 500.0
    chi_beyond = 800.0
    a = 0.4
    w = lensing_kernel(planck18, chi_beyond, a, chi_source)
    assert float(w) == 0.0, f"Lensing kernel should be zero beyond source, got {float(w)}"


def test_lensing_kernel_depends_on_scale_factor(planck18):
    """At fixed chi, kernel grows with decreasing a (higher z -> more lensing)."""
    chi = 500.0
    chi_source = 3000.0
    w_a1 = lensing_kernel(planck18, chi, 1.0, chi_source)
    w_a05 = lensing_kernel(planck18, chi, 0.5, chi_source)
    assert float(w_a05) > float(w_a1), (
        f"W_kappa(a=0.5)={float(w_a05):.3e} should exceed W_kappa(a=1.0)={float(w_a1):.3e}"
    )


# =============================================================================
# 3. C_ELL SCALING
# =============================================================================


def test_theoretical_cl_kappa_scaling_with_omega_m():
    """C_ell^{kappa} scales roughly as Omega_m^2 at fixed sigma8."""
    ell = jnp.linspace(100.0, 1000.0, 10)
    chi_min, chi_max, z_source = 100.0, 500.0, 1100.0

    cosmo1 = jc.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96,
        Omega_k=0.0, w0=-1.0, wa=0.0,
    )
    cosmo2 = jc.Cosmology(
        Omega_c=0.375, Omega_b=0.075, h=0.7, sigma8=0.8, n_s=0.96,
        Omega_k=0.0, w0=-1.0, wa=0.0,
    )
    cl1 = compute_theoretical_cl_kappa(cosmo1, ell, chi_min, chi_max, z_source)
    cl2 = compute_theoretical_cl_kappa(cosmo2, ell, chi_min, chi_max, z_source)

    ratio = float(jnp.mean(cl2) / jnp.mean(cl1))
    expected = (0.45 / 0.30) ** 2  # = 2.25
    assert 1.5 < ratio < 4.0, (
        f"C_ell Omega_m^2 scaling unexpected: ratio={ratio:.2f}, expected~{expected:.2f}"
    )


def test_cl_kappa_ell_scaling():
    """C_ell^{kappa} should be larger at low ell than at high ell."""
    cosmo = jc.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96,
        Omega_k=0.0, w0=-1.0, wa=0.0,
    )
    cl_low = compute_theoretical_cl_kappa(cosmo, jnp.array([100.0]), 100.0, 1000.0, 1100.0)
    cl_high = compute_theoretical_cl_kappa(cosmo, jnp.array([1000.0]), 100.0, 1000.0, 1100.0)
    assert float(cl_low[0]) > float(cl_high[0]), (
        f"C_ell not decreasing: C_ell(100)={float(cl_low[0]):.3e}, C_ell(1000)={float(cl_high[0]):.3e}"
    )


# =============================================================================
# 4. GROWTH FACTOR / GROWTH RATE
# =============================================================================


def test_growth_factor_limits(planck18):
    """Growth factor D(a) should be monotonically increasing and small at high z."""
    a_values = jnp.array([0.001, 0.1, 0.5, 1.0])
    g_values = jax.vmap(lambda a: a2g(planck18, a))(a_values)
    for i in range(len(g_values) - 1):
        assert float(g_values[i + 1]) > float(g_values[i]), (
            f"Growth factor not monotonic: D(a={float(a_values[i])})={float(g_values[i]):.4f} "
            f">= D(a={float(a_values[i+1])})={float(g_values[i+1]):.4f}"
        )
    assert float(g_values[0]) / float(g_values[-1]) < 0.1, (
        f"Growth factor ratio D(0.001)/D(1) = {float(g_values[0]/g_values[-1]):.3f} should be < 0.1"
    )


def test_growth_rate_f_range(planck18):
    """Growth rate f = d ln D / d ln a should lie in [0.4, 1.1] for LCDM."""
    a_values = jnp.array([0.1, 0.5, 0.8, 1.0])
    f_values = jax.vmap(lambda a: a2f(planck18, a))(a_values)
    for a, f in zip(a_values, f_values, strict=False):
        assert 0.4 < float(f) < 1.1, (
            f"Growth rate f out of range at a={float(a):.2f}: f={float(f):.3f}"
        )


# =============================================================================
# 5. HARMONIC SPACE GAUSSIAN (replaces FourierSpaceGaussian)
# =============================================================================


@pytest.fixture(scope="module")
def _packed_model():
    """Small curved-sky model exposing the packed pseudo-a_lm observable."""
    return FieldLevelModel(**_curved_sky_config())


def test_packed_normal_reproduces_pseudo_cl(_packed_model):
    """A native Normal on the packed components reproduces the per-l pseudo-Cl.

    With M_ll = identity, the per-component variance s^2 must satisfy
    <|a_lm|^2> = (M_ll @ C_l)[l] = C_l[l] for every multipole l>=2.
    """
    model = _packed_model
    lmax = model.cmb_lmax
    cl = np.full(lmax + 1, 1e-6, dtype=float)
    var_l = np.eye(lmax + 1) @ cl  # M_ll = identity
    l_of_u = np.asarray(model.cmb_l_of_u)
    u_half = np.asarray(model.cmb_u_half)
    s = np.sqrt(np.maximum(var_l[l_of_u], 1e-30)) * u_half

    n = 4000
    u_samples = jr.normal(jr.key(0), (n, s.size)) * s
    emp_var = np.var(np.asarray(u_samples), axis=0)
    assert np.allclose(emp_var, s**2, rtol=0.15), "Component variances do not match s^2"

    # Reconstruct a_lm and check per-l <|a_lm|^2> ~ C_l for a few multipoles.
    alm_samples = np.asarray(jax.vmap(model.unpack_u)(u_samples))
    alm_l = np.asarray(model.cmb_alm_l)
    for ell in [2, 5, lmax // 2, lmax]:
        sel = alm_l == ell
        mean_power = float(np.mean(np.abs(alm_samples[:, sel]) ** 2))
        assert np.isclose(mean_power, cl[ell], rtol=0.2), (
            f"l={ell}: <|a_lm|^2>={mean_power:.3e} vs C_l={cl[ell]:.3e}"
        )


def test_pack_unpack_roundtrip(_packed_model):
    """pack_alm / unpack_u are mutual inverses on the valid (l>=2) modes."""
    model = _packed_model
    rng = np.random.default_rng(0)
    alm_l = np.asarray(model.cmb_alm_l)
    alm_m = np.asarray(model.cmb_alm_m)
    n_alm = alm_l.shape[0]
    alm = rng.normal(size=n_alm) + 1j * rng.normal(size=n_alm)
    alm[alm_l < 2] = 0.0  # only l>=2 are representable
    alm[alm_m == 0] = np.real(alm[alm_m == 0])  # m=0 modes are real
    alm = jnp.asarray(alm)

    u = model.pack_alm(alm)
    assert u.shape == (model.cmb_u_dim,)
    assert np.all(np.asarray(model.cmb_l_of_u) >= 2)
    alm_rt = np.asarray(model.unpack_u(u))
    assert np.allclose(alm_rt, np.asarray(alm), atol=1e-10)


def test_pack_kappa_map_matches_and_differentiable(_packed_model):
    """pack_kappa_map matches an independent map2alm+pack and is differentiable."""
    import jax_healpy as jhp

    model = _packed_model
    npix = 12 * model.cmb_nside**2
    rng = np.random.default_rng(1)
    kmap = jnp.asarray(rng.normal(size=npix))

    W = jnp.asarray(model.cmb_mask, dtype=kmap.dtype)
    alm = jhp.map2alm(W * kmap, lmax=model.cmb_lmax, pol=False, iter=0, healpy_ordering=True)
    re_idx = np.asarray(model.cmb_pack_re_idx)
    im_idx = np.asarray(model.cmb_pack_im_idx)
    u_ref = np.concatenate(
        [np.real(np.asarray(alm))[re_idx], np.imag(np.asarray(alm))[im_idx]]
    )
    u = np.asarray(model.pack_kappa_map(kmap))
    assert np.allclose(u, u_ref, atol=1e-8)

    grad = np.asarray(jax.grad(lambda k: jnp.sum(model.pack_kappa_map(k)))(kmap))
    assert np.all(np.isfinite(grad))
    assert np.any(grad != 0.0)


# =============================================================================
# 6. GRADIENT CONSISTENCY — finite difference vs analytic
# =============================================================================


def test_logpdf_gradient_finite_difference():
    """Analytic gradient of log_prob matches finite differences.

    Uses the galaxy-only (cmb_enabled=False) Kaiser model at mesh=16^3 to keep
    the test fast.  Relative error < 1% is required for each scalar latent.
    """
    cfg = default_config.copy()
    cfg.update(
        {
            "mesh_shape": (16, 16, 16),
            "box_shape": (100.0, 100.0, 100.0),
            "evolution": "kaiser",
            "a_obs": 1.0,
            "cmb_enabled": False,
            "galaxies_enabled": True,
        }
    )
    model = FieldLevelModel(**cfg)

    truth_params = {
        "Omega_m": 0.31, "sigma8": 0.81,
        "b1": 1.0, "b2": 0.0, "bs2": 0.0, "bn2": 0.0, "bnpar": 0.0,
    }
    truth = model.predict(samples=truth_params, frombase=True, rng=42)
    model.reset()
    model.condition({"obs": truth["obs"]})

    # logpdf traces the model without a seed, so every sampling latent must be
    # supplied. The bias group includes bnpar (Finger-of-God) by default.
    init_mesh = jr.normal(jr.key(123), (16, 16, 16))
    test_params = {
        "Omega_m_": jnp.array(0.0),
        "sigma8_": jnp.array(0.0),
        "b1_": jnp.array(0.0),
        "b2_": jnp.array(0.0),
        "bs2_": jnp.array(0.0),
        "bn2_": jnp.array(0.0),
        "bnpar_": jnp.array(0.0),
        "init_mesh_": init_mesh,
    }

    analytic_grad = jax.grad(model.logpdf)(test_params)

    eps = 1e-4
    for key in ("Omega_m_", "sigma8_", "b1_"):
        p_plus = {**test_params, key: test_params[key] + eps}
        p_minus = {**test_params, key: test_params[key] - eps}
        fd = float((model.logpdf(p_plus) - model.logpdf(p_minus)) / (2 * eps))
        analytic = float(analytic_grad[key])
        rel_err = abs(analytic - fd) / (abs(fd) + 1e-8)
        assert rel_err < 0.01, (
            f"Gradient mismatch for {key}: analytic={analytic:.6f}, FD={fd:.6f}, "
            f"rel_error={rel_err:.4f}"
        )


# =============================================================================
# 7. MODEL GEOMETRY
# =============================================================================


def test_model_observer_position_face_mode():
    """In face mode the observer sits at (Lx/2, Ly/2, 0)."""
    cfg = default_config.copy()
    cfg.update(
        {
            "mesh_shape": (16, 16, 32),
            "box_shape": (200.0, 200.0, 400.0),
            "evolution": "kaiser",
            "a_obs": 1.0,
            "cmb_enabled": False,
            "galaxies_enabled": True,
        }
    )
    model = FieldLevelModel(**cfg)
    obs = model.observer_position
    assert np.isclose(obs[0], 100.0, rtol=1e-6), f"observer_x={obs[0]:.2f}, expected 100.0"
    assert np.isclose(obs[1], 100.0, rtol=1e-6), f"observer_y={obs[1]:.2f}, expected 100.0"
    assert np.isclose(obs[2], 0.0, atol=1e-10), f"observer_z={obs[2]:.2f}, expected 0.0"


def test_model_chi_boundary_face_mode():
    """In face mode chi_boundary = Lz."""
    cfg = default_config.copy()
    cfg.update(
        {
            "mesh_shape": (16, 16, 32),
            "box_shape": (200.0, 200.0, 400.0),
            "evolution": "kaiser",
            "a_obs": 1.0,
            "cmb_enabled": False,
            "galaxies_enabled": True,
        }
    )
    model = FieldLevelModel(**cfg)
    assert np.isclose(model.chi_boundary, 400.0, rtol=1e-6), (
        f"chi_boundary={model.chi_boundary:.2f}, expected 400.0"
    )


def test_model_chi_boundary_center_mode():
    """In center mode chi_boundary = Lz / 2."""
    cfg = default_config.copy()
    cfg.update(
        {
            "mesh_shape": (8, 8, 8),
            "box_shape": (80.0, 80.0, 120.0),
            "evolution": "kaiser",
            "a_obs": 1.0,
            "cmb_enabled": True,
            "cmb_observer_mode": "center",
            "cmb_nside": 8,
            "cmb_n_shells": 4,
            "cmb_noise_nell": {
                "ell": np.arange(64, dtype=float),
                "N_ell": np.full(64, 1e-8, dtype=float),
            },
            "galaxies_enabled": False,
        }
    )
    model = FieldLevelModel(**cfg)
    assert np.isclose(model.chi_boundary, 60.0, rtol=1e-6), (
        f"chi_boundary={model.chi_boundary:.2f}, expected 60.0 (Lz/2)"
    )


# =============================================================================
# 8. CURVED-SKY INTEGRATION SMOKE TESTS
# =============================================================================


def _curved_sky_config():
    cfg = default_config.copy()
    cfg.update(
        {
            "mesh_shape": (8, 8, 8),
            "box_shape": (80.0, 80.0, 120.0),
            "evolution": "kaiser",
            "a_obs": 1.0,
            "paint_oversamp": 1.0,
            "galaxies_enabled": True,
            "cmb_enabled": True,
            "cmb_nside": 8,
            "cmb_n_shells": 4,
            "cmb_noise_nell": {
                "ell": np.arange(64, dtype=float),
                "N_ell": np.full(64, 1e-8, dtype=float),
            },
            "full_los_correction": False,
        }
    )
    return cfg


@pytest.fixture(scope="module")
def curved_sky_case():
    model = FieldLevelModel(**_curved_sky_config())
    rng = np.random.default_rng(1)
    kappa_pred = rng.normal(scale=1e-4, size=int(np.sum(model.cmb_mask)))
    kappa_obs = kappa_pred + rng.normal(scale=float(model.sigma_hp), size=kappa_pred.shape)
    truth = {
        "obs": 1.0 + 0.05 * rng.normal(size=tuple(model.mesh_shape)),
        "kappa_pred": kappa_pred,
        "kappa_obs": kappa_obs,
    }
    return model, truth


def test_field_level_model_curved_sky_predicts_masked_healpix(curved_sky_case):
    model, truth = curved_sky_case

    assert truth["obs"].shape == tuple(model.mesh_shape)
    assert truth["kappa_pred"].ndim == 1
    assert truth["kappa_obs"].ndim == 1
    assert truth["kappa_pred"].shape == truth["kappa_obs"].shape
    assert truth["kappa_pred"].shape[0] == int(np.sum(model.cmb_mask))
    assert np.all(np.isfinite(np.asarray(truth["kappa_pred"])))
    assert np.all(np.isfinite(np.asarray(truth["kappa_obs"])))


def test_plot_field_slices_accepts_masked_healpix(curved_sky_case, tmp_path):
    model, truth = curved_sky_case
    output_dir = Path(tmp_path)

    plot_field_slices(
        truth,
        output_dir=output_dir,
        mesh_shape=tuple(model.mesh_shape),
        box_shape=np.asarray(model.box_shape),
        chi_center=float(model.box_center[2]),
        cmb_mask=model.cmb_mask,
        cmb_nside=model.cmb_nside,
    )

    assert (output_dir / "obs_slices.png").exists()
    assert (output_dir / "kappa_maps.png").exists()
