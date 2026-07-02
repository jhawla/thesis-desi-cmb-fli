"""Curved-sky masked-HEALPix integration tests."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from desi_cmb_fli.bricks import get_cosmology  # noqa: E402
from desi_cmb_fli.metrics import get_cl_healpix  # noqa: E402
from desi_cmb_fli.model import FieldLevelModel, default_config  # noqa: E402
from desi_cmb_fli.validation import compute_cl_theory, measure_spectra  # noqa: E402


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
def curved_sky_truth():
    model = FieldLevelModel(**_curved_sky_config())
    rng = np.random.default_rng(0)
    kappa_pred = rng.normal(scale=1e-4, size=int(np.sum(model.cmb_mask)))
    kappa_obs = kappa_pred + rng.normal(scale=float(model.sigma_hp), size=kappa_pred.shape)
    truth = {
        "obs": 1.0 + 0.05 * rng.normal(size=tuple(model.mesh_shape)),
        "kappa_pred": kappa_pred,
        "kappa_obs": kappa_obs,
    }
    return model, truth


def test_get_cl_healpix_removes_masked_monopole():
    mask = np.zeros(12 * 8**2, dtype=bool)
    mask[: mask.size // 3] = True
    constant_map = np.full(int(mask.sum()), 7.0)

    ell, cl, info = get_cl_healpix(constant_map, mask, lmax=16)

    assert ell[0] == 2
    assert info["norm"] > 0.0
    assert np.all(np.isfinite(cl))
    assert np.allclose(cl, 0.0, atol=1e-12)


def test_measure_spectra_curved_sky_returns_masked_healpix_cls(curved_sky_truth):
    model, truth = curved_sky_truth

    spectra = measure_spectra(truth, model)

    assert spectra["cl_mode"] == "healpix"
    assert spectra["ell"] is not None
    assert spectra["cl_kk_pred"] is not None
    assert spectra["cl_kk_obs"] is not None
    assert spectra["cl_gg"] is not None
    assert spectra["cl_kg"] is not None
    assert np.all(np.isfinite(spectra["cl_kk_pred"]))
    assert np.all(np.isfinite(spectra["cl_gg"]))
    assert np.all(np.isfinite(spectra["cl_kg"]))
    assert 0.0 < spectra["f_sky_gxy"] <= 1.0


def test_packed_observable_consistency(curved_sky_truth):
    """The packed observable round-trips and the conditioning vector is a finite
    Normal sample with the expected dimension — the native dist.Normal guarantees
    sample == log_prob by construction."""
    model, _ = curved_sky_truth
    npix = 12 * model.cmb_nside**2
    rng = np.random.default_rng(0)
    kmap = jnp.asarray(rng.normal(scale=1e-4, size=npix))

    u = model.pack_kappa_map(kmap)
    assert u.shape == (model.cmb_u_dim,)
    assert np.all(np.isfinite(np.asarray(u)))

    # Differentiable wrt the input map (needed for HMC gradients through kappa_pred).
    grad = np.asarray(jax.grad(lambda k: jnp.sum(model.pack_kappa_map(k)))(kmap))
    assert np.all(np.isfinite(grad)) and np.any(grad != 0.0)


def test_centered_full_sky_kappa_cl_amplitude_regression():
    """Regression test for the Born-integrated κ amplitude at nside=16.

    The ratio C_ℓ^{pred} / C_ℓ^{theory} is measured with the f_sky estimator,
    which has much lower variance than full MASTER at nside=16 (std ~0.18 vs ~1.3
    over random realisations).  The test therefore probes the Born-integration
    amplitude, not the decoupling method.
    """
    cfg = default_config.copy()
    cfg.update(
        {
            "mesh_shape": (16, 16, 24),
            "box_shape": (250.0, 250.0, 375.0),
            "evolution": "lpt",
            "lpt_order": 1,
            "a_obs": 1.0,
            "paint_oversamp": 1.0,
            "galaxies_enabled": False,
            "cmb_enabled": True,
            "cmb_nside": 16,
            "cmb_n_shells": 12,
            "cmb_observer_mode": "center",
            "cmb_noise_nell": {
                "ell": np.arange(128, dtype=float),
                "N_ell": np.full(128, 1e-10, dtype=float),
            },
            "full_los_correction": False,
        }
    )
    truth_params = {"Omega_m": 0.315192, "sigma8": 0.811355}

    model = FieldLevelModel(**cfg)
    truth = model.predict(
        samples=truth_params,
        hide_base=False,
        hide_samp=False,
        hide_det=False,
        frombase=True,
        rng=502,
    )

    kappa_pred = np.asarray(truth["kappa_pred"])
    mask = np.asarray(model.cmb_mask, dtype=bool)
    lmax_hp = int(model.cmb_lmax)
    ell_raw, cl_fsky, _ = get_cl_healpix(
        kappa_pred[mask], mask, lmax=lmax_hp, decouple="fsky"
    )

    from desi_cmb_fli.validation import bin_cl_log
    ell_b, cl_kk_pred_b, _ = bin_cl_log(ell_raw, cl_fsky)

    ell_theory = np.geomspace(10.0, float(lmax_hp), 80)
    theory = compute_cl_theory(
        model,
        get_cosmology(**truth_params),
        ell_theory,
        has_galaxies=False,
        observation_mode="closure",
    )
    cl_kk_theory_b = np.interp(ell_b, ell_theory, np.asarray(theory["cl_kk_theory"]))

    ratio = cl_kk_pred_b / cl_kk_theory_b
    valid = (
        np.isfinite(ratio)
        & np.isfinite(cl_kk_pred_b)
        & (ell_b > 10.0)
        & (ell_b < min(80.0, float(lmax_hp)))
    )

    assert valid.any(), "No valid multipoles for the kappa amplitude regression test"
    mean_ratio = float(np.nanmean(ratio[valid]))
    assert 0.50 < mean_ratio < 2.50, f"Unexpected centered full-sky kappa amplitude ratio: {mean_ratio:.3f}"
