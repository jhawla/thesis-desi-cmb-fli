"""The per-shell multipole cut of the Born convergence, ``cmb_lensing.shell_kmax``."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_born_shells import _cfg, _limber

from desi_cmb_fli.bricks import get_cosmology, lin_power_interp
from desi_cmb_fli.cmb_lensing import (
    compute_cl_out,
    compute_cl_outside_model,
    convergence_Born_spherical,
    cut_shells,
    lensing_kernel,
    limber_ratio_outside_model,
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
    rng = np.random.default_rng(0)  # hp.synalm draws from the unseeded global state
    ell, m = hp.Alm.getlm(lmax)
    alm = rng.normal(size=ell.size) + 1j * np.where(m > 0, rng.normal(size=ell.size), 0.0)
    alm[ell > 8] = 0.0
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
    np.testing.assert_array_equal(np.asarray(same.cl_out_cached), np.asarray(ref.cl_out_cached))
    np.testing.assert_array_equal(np.asarray(_truth(same)["kappa_pred"]),
                                  np.asarray(_truth(ref)["kappa_pred"]))


def test_the_covariance_term_is_the_matter_the_shells_leave_out():
    model = FieldLevelModel(**_cfg(cmb_shell_kmax=0.05))
    assert 0 < (model.cmb_shell_taper < 1).any(axis=1).sum() < model.cmb_n_shells
    cosmo = get_cosmology(**model.loc_fid)
    term = np.asarray(compute_cl_outside_model(cosmo, model.ell_1d, **model.cmb_shells,
                                               chi_max=model.chi_high_z_max))
    np.testing.assert_allclose(np.asarray(model.cl_out_cached),
                               term * np.asarray(model.cmb_los_window2 * model.cmb_los_limber),
                               rtol=1e-8, atol=1e-30)
    assert np.all(term[2:] > 0)
    r, h, end = model.cmb_r_shells, model.cmb_d_r, model.chi_boundary

    def outside(chi_max, **kw):
        return np.asarray(compute_cl_outside_model(cosmo, model.ell_1d, **(model.cmb_shells | kw),
                                                   chi_max=chi_max, n_per_shell=200))

    def limber(lo, hi):
        return _limber(cosmo, model.ell_1d, lo, hi, model.cmb_z_source)

    full = limber(r[0] - h / 2, end)
    # nearest bins, no cut, the map stopping at the box: only the Riemann steps of the shells
    held = outside(end, taper=None)
    assert np.all(held[2:] < 0.05 * full[2:])
    # the map deeper than the box: the matter beyond it, as its own Limber power
    np.testing.assert_allclose((outside(model.chi_high_z_max, taper=None) - held)[2:],
                               limber(end, model.chi_high_z_max)[2:], rtol=1e-2)
    # every shell removed whole: all the matter the shells hold
    np.testing.assert_allclose(outside(end, taper=np.zeros_like(model.cmb_shell_taper))[2:],
                               full[2:], rtol=0.1)
    # the cut adds what it removes at the top of the band
    assert outside(end)[-1] > held[-1]


def test_linear_tents_leave_their_end_ramps_to_the_covariance():
    model = FieldLevelModel(**_cfg(cmb_shell_weights="linear", cmb_shell_kmax=0.0))
    cosmo = get_cosmology(**model.loc_fid)
    nearest = FieldLevelModel(**_cfg(cmb_shell_kmax=0.0))
    end = model.chi_boundary
    ramps = np.asarray(compute_cl_outside_model(cosmo, model.ell_1d, **model.cmb_shells, chi_max=end,
                                                n_per_shell=200))
    steps = np.asarray(compute_cl_outside_model(cosmo, nearest.ell_1d, **nearest.cmb_shells,
                                                chi_max=end, n_per_shell=200))
    assert np.all(ramps[2:] > steps[2:])


def test_every_cl_out_mode_carries_the_cut_term_alike():
    models = {mode: FieldLevelModel(**_cfg(cmb_shell_kmax=0.05, cl_out_mode=mode))
              for mode in ("fixed", "exact")}
    ref = FieldLevelModel(**_cfg(cl_out_mode="exact", cmb_shell_kmax=0.0))
    fid = models["fixed"].loc_fid

    def cl(m, ds8=0.0):
        cosmo = get_cosmology(Omega_m=fid["Omega_m"], sigma8=fid["sigma8"] + ds8)
        return np.asarray(compute_cl_out(cosmo, m.ell_1d, m.chi_high_z_max, m.cmb_shells,
                                         mode=m.cl_out_mode, cl_cached=m.cl_out_cached,
                                         window2=m.cmb_los_window2 * m.cmb_los_limber))[2:]

    np.testing.assert_allclose(cl(models["exact"]), cl(models["fixed"]), rtol=1e-6)
    removed = cl(models["exact"], 0.01) - cl(ref, 0.01) - (cl(models["exact"]) - cl(ref))
    assert np.all(removed > 0)  # the removed power grows with sigma8


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


def test_the_covariance_term_is_the_exact_projection_where_limber_fails():
    """A slab the shells leave out (one shell removed whole, the map 1000-2000 Mpc/h): exact over
    Limber equals a direct evaluation on other grids, is well above 1 at the lowest multipoles
    (Limber misses the radial modes) and goes to 1 at high ell."""
    import jax_cosmo as jc
    from scipy.special import spherical_jn

    cosmo = get_cosmology(Omega_m=0.315192, sigma8=0.811355)
    r = np.array([1025.0])
    a_r = np.asarray(jc.background.a_of_chi(cosmo, jnp.asarray(r)))
    ell = np.array([2, 5, 10, 60])
    ratio = limber_ratio_outside_model(cosmo, ell, r, a_r, 50.0, "nearest", 1100.0, 2000.0,
                                       taper=np.zeros((1, 61)))
    chi = np.linspace(1000.0, 2000.0, 801)
    a = jc.background.a_of_chi(cosmo, jnp.asarray(chi))
    chi_s = jc.background.radial_comoving_distance(cosmo, 1.0 / 1101.0)[0]
    kern = np.asarray(lensing_kernel(cosmo, jnp.asarray(chi), a, chi_s) *
                      jc.background.growth_factor(cosmo, a))
    k = np.geomspace(1e-4, 1.0, 20000)
    pk = np.asarray(lin_power_interp(cosmo)(jnp.asarray(k)))  # the model's spectrum
    for i, ell_val in enumerate(ell):
        proj = np.trapezoid(kern[None] * spherical_jn(ell_val, np.outer(k, chi)), chi, axis=1)
        exact = 2.0 / np.pi * np.trapezoid(k**2 * pk * proj**2, k)
        limber = np.trapezoid(kern**2 / chi**2 * np.interp((ell_val + 0.5) / chi, k, pk), chi)
        np.testing.assert_allclose(ratio[i], exact / limber, rtol=0.02)
    assert ratio[0] > 1.5 and ratio[1] > 1.1
    assert abs(ratio[-1] - 1.0) < 0.02
