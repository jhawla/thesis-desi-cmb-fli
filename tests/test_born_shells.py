"""Model-level behaviour of the Born shells: ``cmb_lensing.chi_matter_min``, ``shell_weights``
and ``proj_oversamp``."""

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


def test_shells_tile_the_matter_start_to_chi_boundary():
    ref = FieldLevelModel(**_cfg())
    late = FieldLevelModel(**_cfg(cmb_chi_matter_min=60.0))
    dr = late.cmb_d_r
    np.testing.assert_allclose(late.cmb_r_shells[0] - dr / 2, 60.0)
    np.testing.assert_allclose(late.cmb_r_shells[-1] + dr / 2, late.chi_boundary)
    np.testing.assert_allclose(ref.cmb_r_shells[0] - ref.cmb_d_r / 2, 0.0)
    assert late.low_z_matter_start == 60.0 and ref.low_z_matter_start == 1.0


def test_without_the_cut_the_los_covariance_is_the_matter_beyond_the_box_and_the_shell_steps():
    """No low-z term: the matter below the shells is the matter the map does not hold. Without the
    cut, nearest bins leave out the matter beyond the box and the steps of the shells only."""
    from desi_cmb_fli.bricks import get_cosmology
    from desi_cmb_fli.cmb_lensing import compute_theoretical_cl_kappa

    for start in (0.0, 60.0):
        m = FieldLevelModel(**_cfg(cmb_chi_matter_min=start, cmb_shell_kmax=0.0))
        cosmo = get_cosmology(**m.loc_fid)
        beyond = np.asarray(compute_theoretical_cl_kappa(cosmo, m.ell_1d, m.chi_boundary, 300.0,
                                                         m.cmb_z_source))
        shells = np.asarray(compute_theoretical_cl_kappa(cosmo, m.ell_1d, max(start, 1.0),
                                                         m.chi_boundary, m.cmb_z_source))
        term = np.asarray(m.cl_high_z_cached) / np.asarray(m.cmb_los_window2)
        assert np.all(np.abs(term - beyond)[2:] < 0.05 * shells[2:])


def test_the_kappa_radial_window_is_one_inside_and_ramps_at_the_ends():
    from desi_cmb_fli.cmb_lensing import kappa_radial_window

    r = np.array([100.0, 150.0, 200.0])
    chi, w = kappa_radial_window(r, 50.0, "linear")
    inside = (chi >= r[0]) & (chi <= r[-1])
    np.testing.assert_allclose(w[inside], 1.0, atol=1e-12)
    np.testing.assert_allclose(np.interp([75.0, 225.0], chi, w), 0.5, atol=1e-3)
    chi, w = kappa_radial_window(r, 50.0, "nearest", support=[1.0, 0.5, 1.0])
    np.testing.assert_allclose(np.interp([110.0, 150.0, 190.0], chi, w), [1.0, 0.5, 1.0])


def test_the_kappa_window_removes_only_the_matter_the_map_does_not_hold():
    from desi_cmb_fli.bricks import get_cosmology
    from desi_cmb_fli.cmb_lensing import compute_theoretical_cl_kg

    cosmo, ell = get_cosmology(), jnp.array([10.0, 50.0])
    g = np.linspace(1000.0, 2000.0, 200)
    args = (cosmo, ell, 1000.0, 2000.0, 1100.0, 2.0)
    ref = np.asarray(compute_theoretical_cl_kg(*args, nz=(g, np.ones_like(g))))
    covering = (np.array([500.0, 3000.0]), np.ones(2))
    np.testing.assert_allclose(
        np.asarray(compute_theoretical_cl_kg(*args, nz=(g, np.ones_like(g)),
                                             kappa_window=covering)), ref, rtol=1e-10)
    half = (np.array([500.0, 1499.9, 1500.0, 3000.0]), np.array([0.0, 0.0, 1.0, 1.0]))
    cut = np.asarray(compute_theoretical_cl_kg(*args, nz=(g, np.ones_like(g)), kappa_window=half))
    assert np.all((cut > 0.2 * ref) & (cut < 0.8 * ref))


def test_every_high_z_mode_carries_the_los_term_alike():
    """fixed, taylor and exact agree at the fiducial; away from it taylor follows exact to first
    order and fixed does not move."""
    from desi_cmb_fli.bricks import get_cosmology
    from desi_cmb_fli.cmb_lensing import compute_cl_high_z, compute_cl_outside_model

    def los(m, cosmo, linear=False):
        return np.asarray(compute_cl_outside_model(cosmo, m.ell_1d, **m.cmb_shells, chi_max=300.0,
                                                   linear_pk=linear)) * np.asarray(m.cmb_los_window2)

    models = {mode: FieldLevelModel(**_cfg(cmb_shell_kmax=0.0, high_z_mode=mode))
              for mode in ("fixed", "taylor", "exact", "exact_linear")}
    fid = models["fixed"].loc_fid
    for ds8 in (0.0, 0.01):
        cosmo = get_cosmology(Omega_m=fid["Omega_m"], sigma8=fid["sigma8"] + ds8)
        cl = {mode: np.asarray(compute_cl_high_z(
            cosmo, m.ell_1d, m.chi_boundary, m.chi_high_z_max, m.cmb_z_source, mode=mode,
            cl_cached=m.cl_high_z_cached, gradients=m.high_z_gradients, loc_fid=m.loc_fid,
            shells=m.cmb_shells, window2=m.cmb_los_window2))[2:] for mode, m in models.items()}
        np.testing.assert_allclose(cl["exact"], los(models["exact"], cosmo)[2:], rtol=1e-10)
        np.testing.assert_allclose(cl["exact_linear"],
                                   los(models["exact"], cosmo, linear=True)[2:], rtol=1e-10)
        np.testing.assert_allclose(cl["taylor"], cl["exact"], rtol=2e-3)
        np.testing.assert_allclose(cl["fixed"], los(models["fixed"], get_cosmology(**fid))[2:],
                                   rtol=1e-10)


def test_the_los_covariance_carries_the_squared_bilinear_window_of_the_projection_sphere():
    """The data map's matter outside the model reaches the observable through the projector's
    bilinear window at proj_nside, as the model map does; finer projection, window closer to 1."""
    from desi_cmb_fli.cmb_lensing import bilinear_window

    for ov in (1, 2):
        m = FieldLevelModel(**_cfg(cmb_proj_oversamp=ov))
        w2 = bilinear_window(int(m.cmb_proj_nside), int(m.cmb_lmax)) ** 2
        np.testing.assert_allclose(np.asarray(m.cmb_los_window2), w2, rtol=1e-12)
    coarse = FieldLevelModel(**_cfg(cmb_proj_oversamp=1)).cmb_los_window2
    fine = FieldLevelModel(**_cfg(cmb_proj_oversamp=2)).cmb_los_window2
    lmax = coarse.size - 1
    assert 0 < coarse[lmax] < fine[lmax] < 1
    assert FieldLevelModel(**_cfg(full_los_correction=False)).cmb_los_window2 is None


def test_removed_config_keys_are_rejected_with_kappa_on():
    from pathlib import Path

    import yaml

    from desi_cmb_fli.model import get_model_from_config

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(open(root / "configs/inference/scan/closure_d1p00_joint.yaml"))
    for key in ("chi_min", "chi_low_z_min"):
        bad = {**cfg, "cmb_lensing": {**cfg["cmb_lensing"], key: 700.0}}
        with pytest.raises(ValueError, match="chi_matter_min"):
            get_model_from_config(bad)


def test_chi_matter_min_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="chi_matter_min"):
        FieldLevelModel(**_cfg(cmb_chi_matter_min=1e4))


def test_unknown_shell_weights_is_rejected():
    with pytest.raises(ValueError, match="shell_weights"):
        FieldLevelModel(**_cfg(cmb_shell_weights="cubic"))


@pytest.mark.parametrize("weights", ["nearest", "linear"])
def test_forward_and_gradient_run_with_a_late_matter_start(weights):
    model = FieldLevelModel(**_cfg(cmb_chi_matter_min=60.0, cmb_shell_weights=weights))
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


@pytest.mark.parametrize("table", [None, "data/abacus_cosm000_CLASS_power.txt"])
def test_sampling_omega_m_puts_no_jax_cosmo_callback_in_the_graph(table):
    """The jax_cosmo fork computes growth and distances through host callbacks that re-solve for
    every new cosmology; with Omega_m sampled they exhaust the compiler's memory on a GPU run. Every
    background quantity of the model must come from the emulator or be evaluated at a = 1."""
    from pathlib import Path

    table = None if table is None else str(Path(__file__).resolve().parents[1] / table)
    model = FieldLevelModel(**_cfg(galaxies_enabled=True, png_type="fNL", a_obs=None, cmb_chi_matter_min=60.0,
                                   cmb_shell_weights="linear", lin_pk_table=table))
    truth = model.predict(samples={"Omega_m": 0.315192, "sigma8": 0.811355, "b1": 1.0, "fNL": 10.0},
                          hide_base=False, hide_samp=False, hide_det=False, frombase=True, rng=0)
    model.reset()
    model.condition({k: truth[k] for k in ("obs", "kappa_obs")})
    model.condition({n: truth[n] for n in model.sampled_scalar_latents()
                     if n not in ("Omega_m", "sigma8")}, frombase=True)
    model.block()
    pos = {k: jnp.asarray(truth[k]) for k in ("init_mesh_", "Omega_m_", "sigma8_")}
    assert "pure_callback" not in str(jax.make_jaxpr(jax.grad(model.logpdf))(pos))
