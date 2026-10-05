"""The per-shell multipole cut of the Born convergence, ``cmb_lensing.shell_kmax``."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_born_shells import _cfg

from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import (
    compute_cl_high_z,
    compute_cl_shell_cut,
    compute_theoretical_cl_kappa,
    convergence_Born_spherical,
    cut_shells,
    shell_ell_taper,
)
from desi_cmb_fli.model import FieldLevelModel

jax.config.update("jax_enable_x64", True)

FID = {"Omega_m": 0.315192, "sigma8": 0.811355}


def _truth(model, rng=3):
    model.reset()
    return model.predict(samples=dict(FID), hide_base=False, hide_samp=False, hide_det=False,
                         frombase=True, rng=rng)


def test_the_taper_follows_the_resolved_multipole_of_each_shell():
    r, h = np.array([50.0, 150.0, 400.0, 1000.0]), 100.0
    taper = shell_ell_taper(r, h, "nearest", k_max=0.05, ell_max=32, lmax=64)
    r_lo, r_hi = r - h / 2, r + h / 2
    ell_res = 0.05 * 0.75 * (r_hi**4 - r_lo**4) / (r_hi**3 - r_lo**3)  # W. Kabalan's r_eff
    assert np.all(taper[3] == 1.0)  # ell_res 50 >= ell_max: untouched
    for s in range(3):
        l_cut = int(np.floor(ell_res[s]))
        l_width = max(l_cut // 4, 1)
        assert np.all(taper[s, : l_cut - l_width + 1] == 1.0)
        assert np.all(taper[s, l_cut:] == 0.0)
        mid = taper[s, l_cut - l_width + 1 : l_cut]
        assert np.all((mid > 0) & (mid < 1)) and np.all(np.diff(mid) < 0)
    # a shell that resolves no multipole is removed whole
    assert np.all(shell_ell_taper([10.0], 10.0, "linear", 0.05, 32, 64) == 0.0)


def test_cut_shells_sums_untouched_shells_and_keeps_a_band_limited_shell():
    nside, lmax = 16, 32
    rng = np.random.default_rng(0)
    alm = hp.synalm(np.ones(lmax + 1), lmax=lmax, new=True)
    alm[hp.Alm.getlm(lmax)[0] > 8] = 0.0
    low = hp.alm2map(alm, nside, lmax=lmax)
    shells = jnp.asarray(np.stack([low, rng.normal(size=low.size)]))
    taper = np.ones((2, lmax + 1))
    np.testing.assert_array_equal(np.asarray(cut_shells(shells, taper, nside)),
                                  np.asarray(shells.sum(0)))
    taper[0, 16:] = 0.0  # cuts above everything shell 0 holds
    out = np.asarray(cut_shells(shells, taper, nside)) - np.asarray(shells[1])
    # the iter=0 round trip: small in rms, its worst pixels at the poles
    assert np.std(out - low) < 5e-3 * np.std(low)
    taper[0, :] = 0.0
    np.testing.assert_allclose(np.asarray(cut_shells(shells, taper, nside)), shells[1], atol=1e-12)


def test_per_shell_born_maps_sum_to_the_born_map():
    model = FieldLevelModel(**_cfg(cmb_shell_weights="linear"))
    pos = jnp.asarray(np.random.default_rng(1).uniform(0, 8, size=(4000, 3)))
    args = (get_cosmology(**FID), pos, model.box_shape, model.mesh_shape, model.observer_position,
            model.cmb_r_shells, model.cmb_a_shells, model.cmb_d_r, model.cmb_proj_nside,
            model.cmb_proj_sim_mask, model.cmb_z_source, model.t_enter, model.t_exit)
    kw = {"return_full": True, "shell_weights": "linear"}
    full = convergence_Born_spherical(*args, **kw)
    shells = convergence_Born_spherical(*args, per_shell=True, **kw)
    assert shells.shape == (model.cmb_n_shells, full.size)
    np.testing.assert_allclose(np.asarray(shells.sum(0)), np.asarray(full), rtol=1e-12, atol=1e-15)


def test_a_cut_that_touches_no_shell_leaves_the_model_bit_identical():
    ref = FieldLevelModel(**_cfg(cmb_shell_kmax=0.0))
    same = FieldLevelModel(**_cfg(cmb_shell_kmax=10.0))  # every ell_res far above 2 nside
    assert not (same.cmb_shell_taper < 1).any()
    np.testing.assert_array_equal(np.asarray(same.cl_high_z_cached), np.asarray(ref.cl_high_z_cached))
    np.testing.assert_array_equal(np.asarray(_truth(same)["kappa_pred"]),
                                  np.asarray(_truth(ref)["kappa_pred"]))


def test_the_covariance_term_is_the_limber_power_the_cut_removes():
    model = FieldLevelModel(**_cfg(cmb_shell_kmax=0.05))
    ref = FieldLevelModel(**_cfg(cmb_shell_kmax=0.0))
    assert 0 < (model.cmb_shell_taper < 1).any(axis=1).sum() < model.cmb_n_shells
    extra = np.asarray(model.cl_high_z_cached) - np.asarray(ref.cl_high_z_cached)
    cosmo = get_cosmology(**model.loc_fid)
    term = np.asarray(compute_cl_shell_cut(cosmo, model.ell_1d, **model.cmb_shell_cut))
    np.testing.assert_allclose(extra, term * np.asarray(model.cmb_los_window2), rtol=1e-8, atol=1e-30)
    assert np.all(term[2:] >= 0) and term[-1] > 0
    # every shell removed whole: the Limber C_l of the matter the shells hold, up to the Riemann
    # sum over the shells' lensing weights
    cut = dict(model.cmb_shell_cut, shell_weights="nearest", taper=np.zeros_like(model.cmb_shell_taper))
    r, h = model.cmb_r_shells, model.cmb_d_r
    whole = np.asarray(compute_cl_shell_cut(cosmo, model.ell_1d, **cut, n_per_shell=200))
    limber = np.asarray(compute_theoretical_cl_kappa(cosmo, model.ell_1d, max(1.0, r[0] - h / 2),
                                                     r[-1] + h / 2, model.cmb_z_source, n_steps=800))
    np.testing.assert_allclose(whole[2:], limber[2:], rtol=0.1)
    nothing = dict(model.cmb_shell_cut, taper=np.ones_like(model.cmb_shell_taper))
    assert np.all(np.asarray(compute_cl_shell_cut(cosmo, model.ell_1d, **nothing)) == 0.0)


def test_every_high_z_mode_carries_the_cut_term_alike():
    models = {mode: FieldLevelModel(**_cfg(cmb_shell_kmax=0.05, high_z_mode=mode))
              for mode in ("fixed", "taylor", "exact")}
    cosmo = get_cosmology(**models["fixed"].loc_fid)
    cl = {mode: np.asarray(compute_cl_high_z(
        cosmo, m.ell_1d, m.chi_boundary, m.chi_high_z_max, m.cmb_z_source, mode=mode,
        cl_cached=m.cl_high_z_cached, gradients=m.high_z_gradients, loc_fid=m.loc_fid,
        shell_cut=m.cmb_shell_cut, window2=m.cmb_los_window2))[2:] for mode, m in models.items()}
    np.testing.assert_allclose(cl["exact"], cl["fixed"], rtol=1e-8)
    np.testing.assert_allclose(cl["taylor"], cl["fixed"], rtol=1e-8)
    ref = FieldLevelModel(**_cfg(high_z_mode="taylor", cmb_shell_kmax=0.0))
    d = np.asarray(models["taylor"].high_z_gradients["dCl_ds8"]) - np.asarray(ref.high_z_gradients["dCl_ds8"])
    assert np.all(d[2:] > 0)  # the removed power grows with sigma8


def test_the_cut_lowers_the_model_power_at_the_top_of_the_band():
    ref, cut = FieldLevelModel(**_cfg(cmb_shell_kmax=0.0)), FieldLevelModel(**_cfg(cmb_shell_kmax=0.05))
    lmax = 2 * ref.cmb_nside
    c_ref = hp.anafast(np.asarray(_truth(ref)["kappa_pred"]), lmax=lmax)
    c_cut = hp.anafast(np.asarray(_truth(cut)["kappa_pred"]), lmax=lmax)
    assert c_cut[lmax] < c_ref[lmax]


def test_the_cut_defaults_to_the_init_grid_nyquist_and_requires_the_los_correction():
    m = FieldLevelModel(**_cfg())
    init_cell = np.asarray(m.box_shape, dtype=float) / np.asarray(m.init_shape)
    assert m.cmb_shell_kmax == pytest.approx(np.pi / init_cell.max())
    assert m.cmb_shell_taper is not None
    off = FieldLevelModel(**_cfg(full_los_correction=False))
    assert off.cmb_shell_kmax == 0.0 and off.cmb_shell_taper is None
    with pytest.raises(ValueError, match="full_los_correction"):
        FieldLevelModel(**_cfg(cmb_shell_kmax=0.05, full_los_correction=False))


@pytest.mark.parametrize("oversamp", [1, 2])
def test_forward_and_gradient_run_with_the_cut(oversamp):
    model = FieldLevelModel(**_cfg(cmb_shell_kmax=0.05, cmb_proj_oversamp=oversamp,
                                   cmb_shell_weights="linear"))
    truth = _truth(model)
    assert np.all(np.isfinite(np.asarray(truth["kappa_pred"])))
    model.reset()
    model.condition({"kappa_obs": truth["kappa_obs"]})
    model.condition(dict(FID), frombase=True)
    model.block()
    g = jax.grad(model.logpdf)({"init_mesh_": jnp.asarray(truth["init_mesh_"])})["init_mesh_"]
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.abs(g).max()) > 0
