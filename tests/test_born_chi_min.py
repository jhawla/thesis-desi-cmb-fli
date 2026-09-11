"""Model-level behaviour of ``cmb_lensing.chi_min`` and ``cmb_lensing.shell_weights``."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from desi_cmb_fli.model import FieldLevelModel, default_config

jax.config.update("jax_enable_x64", True)


def _cfg(**overrides):
    cfg = default_config.copy()
    cfg.update(
        {
            "mesh_shape": (8, 8, 8),
            "box_shape": (400.0, 400.0, 400.0),
            "evolution": "lpt",
            "lpt_order": 1,
            "a_obs": 1.0,
            "paint_oversamp": 1.0,
            "galaxies_enabled": False,
            "cmb_enabled": True,
            "cmb_nside": 4,
            "cmb_n_shells": 4,
            "cmb_observer_mode": "center",
            "cmb_noise_nell": {
                "ell": np.arange(64, dtype=float),
                "N_ell": np.full(64, 1e-9, dtype=float),
            },
            "full_los_correction": True,
            "high_z_mode": "fixed",
            "chi_high_z_max": 300.0,
        }
    )
    cfg.update(overrides)
    return cfg


def test_shells_tile_chi_min_to_chi_boundary_and_low_z_cl_enters_the_covariance():
    ref = FieldLevelModel(**_cfg())
    cut = FieldLevelModel(**_cfg(cmb_chi_min=60.0))
    dr = cut.cmb_d_r
    np.testing.assert_allclose(cut.cmb_r_shells[0] - dr / 2, 60.0)
    np.testing.assert_allclose(cut.cmb_r_shells[-1] + dr / 2, cut.chi_boundary)
    np.testing.assert_allclose(ref.cmb_r_shells[0] - ref.cmb_d_r / 2, 0.0)
    extra = np.asarray(cut.cl_high_z_cached) - np.asarray(ref.cl_high_z_cached)
    assert np.all(extra[2:] > 0)


def test_chi_min_requires_the_los_correction():
    with pytest.raises(ValueError, match="full_los_correction"):
        FieldLevelModel(**_cfg(cmb_chi_min=60.0, full_los_correction=False))
    with pytest.raises(ValueError, match="high_z_mode"):
        FieldLevelModel(**_cfg(cmb_chi_min=60.0, high_z_mode="exact"))


def test_chi_min_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="chi_min"):
        FieldLevelModel(**_cfg(cmb_chi_min=1e4))


def test_unknown_shell_weights_is_rejected():
    with pytest.raises(ValueError, match="shell_weights"):
        FieldLevelModel(**_cfg(cmb_shell_weights="cubic"))


@pytest.mark.parametrize("weights", ["nearest", "linear"])
def test_forward_and_gradient_run_with_chi_min(weights):
    model = FieldLevelModel(**_cfg(cmb_chi_min=60.0, cmb_shell_weights=weights))
    truth = model.predict(
        samples={"Omega_m": 0.315192, "sigma8": 0.811355},
        hide_base=False, hide_samp=False, hide_det=False, frombase=True, rng=3,
    )
    assert np.all(np.isfinite(np.asarray(truth["kappa_pred"])))
    model.reset()
    model.condition({"kappa_obs": truth["kappa_obs"]})
    model.condition({"Omega_m": 0.315192, "sigma8": 0.811355}, frombase=True)
    model.block()
    pos = {"init_mesh_": jnp.asarray(truth["init_mesh_"])}
    g = jax.grad(model.logpdf)(pos)["init_mesh_"]
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.abs(g).max()) > 0


# ---------------------------------------------------------------------------
# cmb_lensing.proj_oversamp (anti-aliasing of the Born projection)
# ---------------------------------------------------------------------------


def test_proj_oversamp_1_leaves_every_projection_attribute_untouched():
    m = FieldLevelModel(**_cfg())
    assert m.cmb_proj_nside == m.cmb_nside
    assert m.cmb_proj_mask is m.cmb_mask and m.cmb_proj_sim_mask is m.cmb_sim_mask


def test_proj_oversamp_refines_the_sphere_but_not_the_observable():
    ref = FieldLevelModel(**_cfg())
    fine = FieldLevelModel(**_cfg(cmb_proj_oversamp=2))
    assert fine.cmb_proj_nside == 2 * fine.cmb_nside
    assert fine.cmb_proj_mask.size == hp.nside2npix(2 * fine.cmb_nside)
    # the band, the mode-coupling matrix and the packed observable are unchanged
    assert fine.cmb_lmax == ref.cmb_lmax
    assert fine.cmb_u_dim == ref.cmb_u_dim
    assert fine.cmb_mask.size == ref.cmb_mask.size
    np.testing.assert_array_equal(fine.cmb_mask, ref.cmb_mask)


def test_a_band_limited_field_packs_the_same_at_both_projection_resolutions():
    """The defining property of the refinement: it must only remove aliasing. On a field that
    is already band-limited to the observable band there is nothing to alias, so both the coarse
    and the refined path must return the a_lm the map was built from, and the refined one must
    not be worse. (The residual is HEALPix quadrature error, large at this test's tiny nside.)"""
    fine = FieldLevelModel(**_cfg(cmb_proj_oversamp=2))
    lmax = int(fine.cmb_lmax)
    rng = np.random.default_rng(0)
    n_alm = hp.Alm.getsize(lmax)
    alm = (rng.normal(size=n_alm) + 1j * rng.normal(size=n_alm)).astype(np.complex128)
    alm[hp.Alm.getlm(lmax)[1] == 0] = alm[hp.Alm.getlm(lmax)[1] == 0].real

    u_true = np.asarray(fine.pack_alm(jnp.asarray(alm)))          # the exact answer
    u_coarse = np.asarray(fine.pack_kappa_map(
        jnp.asarray(hp.alm2map(alm, nside=fine.cmb_nside, lmax=lmax))))
    u_fine = np.asarray(fine.pack_kappa_map(
        jnp.asarray(hp.alm2map(alm, nside=fine.cmb_proj_nside, lmax=lmax))))

    assert u_true.shape == u_coarse.shape == u_fine.shape == (fine.cmb_u_dim,)
    scale = np.abs(u_true).max()
    assert scale > 0, "degenerate input: the test would prove nothing"
    err_coarse = np.abs(u_coarse - u_true).max() / scale
    err_fine = np.abs(u_fine - u_true).max() / scale
    # Both recover the input; the refined path is not allowed to be the worse of the two.
    assert err_coarse < 0.1 and err_fine <= err_coarse


def test_the_refined_transform_selects_the_same_lm_pairs():
    """The refined path transforms at its own larger lmax; the indices it then selects must
    address exactly the (l, m) pairs the observable is defined on, in the same order."""
    fine = FieldLevelModel(**_cfg(cmb_proj_oversamp=2))
    assert fine.cmb_proj_lmax == 2 * fine.cmb_proj_nside > fine.cmb_lmax
    for coarse_idx, fine_idx in ((fine.cmb_pack_re_idx, fine.cmb_proj_pack_re_idx),
                                 (fine.cmb_pack_im_idx, fine.cmb_proj_pack_im_idx)):
        c, f = np.asarray(coarse_idx), np.asarray(fine_idx)
        assert c.shape == f.shape
        np.testing.assert_array_equal(hp.Alm.getlm(fine.cmb_lmax, c),
                                      hp.Alm.getlm(fine.cmb_proj_lmax, f))


def test_proj_oversamp_rejects_a_non_power_of_two():
    for bad in (0, 3, 6):
        with pytest.raises(ValueError, match="power of two"):
            FieldLevelModel(**_cfg(cmb_proj_oversamp=bad))


@pytest.mark.parametrize("oversamp", [1, 2])
def test_forward_and_gradient_run_with_proj_oversamp(oversamp):
    model = FieldLevelModel(**_cfg(cmb_proj_oversamp=oversamp, cmb_shell_weights="linear"))
    truth = model.predict(
        samples={"Omega_m": 0.315192, "sigma8": 0.811355},
        hide_base=False, hide_samp=False, hide_det=False, frombase=True, rng=3,
    )
    assert np.asarray(truth["kappa_pred"]).size == hp.nside2npix(model.cmb_proj_nside)
    assert np.asarray(truth["kappa_obs"]).size == model.cmb_u_dim
    model.reset()
    model.condition({"kappa_obs": truth["kappa_obs"]})
    model.condition({"Omega_m": 0.315192, "sigma8": 0.811355}, frombase=True)
    model.block()
    g = jax.grad(model.logpdf)({"init_mesh_": jnp.asarray(truth["init_mesh_"])})["init_mesh_"]
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.abs(g).max()) > 0
