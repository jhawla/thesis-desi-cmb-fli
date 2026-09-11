"""The galaxy likelihood works on counts, with the selection multiplying the model.

Dividing the data by the selection blows up wherever the survey only partly covers a cell,
which is exactly where a realistic footprint lives. Multiplying the prediction instead is the
same likelihood -- these tests pin that equality -- but stays finite as the completeness goes
to zero, and applies the selection at paint resolution so a cell that is half covered is
predicted as half covered rather than as a full cell of half the density.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import condition, seed, trace

from desi_cmb_fli.bricks import (
    INTERLACE_ORDER,
    band_limit,
    interlace_accumulate,
    interlace_combine,
    interlace_finalize,
)
from desi_cmb_fli.model import FieldLevelModel, default_config

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def model():
    config = default_config.copy()
    config["mesh_shape"] = (16, 16, 16)
    config["box_shape"] = (400.0, 400.0, 400.0)
    config["paint_oversamp"] = 1.5
    return FieldLevelModel(**config)


def _selection(shape, seed_=0, low=0.05):
    """A smooth selection in [low, ~1], as a survey edge would look."""
    rng = np.random.default_rng(seed_)
    s = np.abs(band_limit(jnp.asarray(rng.normal(size=shape)), shape))
    s = s / s.mean()
    return jnp.asarray(np.clip(s, low, None), dtype=float)


def _obs_logprob(model, counts, gxy_mesh, gxy_paint=None, ngbars=None):
    """Log-probability of the 'obs' site alone, at a given predicted field."""
    with trace() as tr, seed(rng_seed=0), condition(data={"obs": counts}):
        model.likelihood(gxy_mesh=gxy_mesh, gxy_paint=gxy_paint, ngbars=ngbars)
    site = tr["obs"]
    return float(jnp.sum(site["fn"].log_prob(jnp.asarray(counts))))


def test_counts_likelihood_matches_the_ratio_formulation(model):
    """chi2 is invariant: dividing the data or multiplying the model is one likelihood.

    The two differ only by a parameter-independent Jacobian, so the *difference* between two
    predicted fields must agree exactly.
    """
    shape = tuple(int(s) for s in model.mesh_shape)
    rng = np.random.default_rng(3)
    model.selec_mesh = _selection(shape, 1)
    model.gxy_occ_mask3d = jnp.ones(shape, dtype=bool)
    model.gxy_count = 250.0

    nbar = float(model.gxy_count) * np.asarray(model.selec_mesh)
    counts = jnp.asarray(nbar * (1.0 + 0.1 * rng.normal(size=shape)))

    def chi2_ratio(g):
        """The old form: obs = N/(n̄S) compared to 1+delta_g with variance 1/(n̄S)."""
        obs = np.asarray(counts) / nbar
        return float(np.sum((obs - np.asarray(g)) ** 2 * nbar))

    lps, chi2s = [], []
    for _ in range(3):
        g = jnp.asarray(1.0 + 0.05 * rng.normal(size=shape))
        lps.append(_obs_logprob(model, counts, g))
        chi2s.append(chi2_ratio(g))

    for i in (1, 2):
        np.testing.assert_allclose(
            lps[i] - lps[0], -0.5 * (chi2s[i] - chi2s[0]), rtol=1e-9
        )


def test_ngbars_rescale_counts_and_shot_noise_together(model):
    """A per-shell amplitude alpha must move the mean and the variance by the same factor."""
    shape = tuple(int(s) for s in model.mesh_shape)
    model.selec_mesh = _selection(shape, 2)
    model.gxy_occ_mask3d = jnp.ones(shape, dtype=bool)
    model.gxy_count = 100.0
    model.gxy_shell_id = jnp.zeros(shape, dtype=jnp.int32)
    model.n_gxy_shells = 1

    g = jnp.ones(shape)
    counts = jnp.asarray(float(model.gxy_count) * np.asarray(model.selec_mesh))
    alpha = 1.3
    with trace() as tr, seed(rng_seed=0), condition(data={"obs": counts}):
        model.likelihood(gxy_mesh=g, ngbars=jnp.array([alpha]))
    fn = tr["obs"]["fn"]

    expected_mean = alpha * float(model.gxy_count) * np.asarray(model.selec_mesh)
    np.testing.assert_allclose(np.asarray(fn.mean), expected_mean, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(fn.variance), expected_mean, rtol=1e-10)


def test_the_selection_multiplies_at_paint_resolution():
    """The predicted total count is right at paint resolution and biased at final resolution.

    Multiplying two coarse fields drops the sub-cell correlation between the galaxy density and
    the survey coverage, which is exactly what a partly covered edge cell is made of.
    """
    shape = (16, 16, 16)
    paint = (32, 32, 32)
    rng = np.random.default_rng(7)
    g_paint = jnp.asarray(1.0 + 0.2 * rng.normal(size=paint))
    # Coverage correlated with the density inside each cell: half the sub-cells are observed.
    s_paint = jnp.asarray(np.where(np.asarray(g_paint) > 1.0, 1.8, 0.2))

    true_total = float(jnp.mean(g_paint * s_paint))
    fine = float(jnp.mean(band_limit(g_paint * s_paint, shape)))
    coarse = float(jnp.mean(band_limit(g_paint, shape) * band_limit(s_paint, shape)))

    np.testing.assert_allclose(fine, true_total, rtol=1e-10)
    assert abs(coarse - true_total) > 0.05 * abs(true_total)


def test_a_constant_selection_makes_both_likelihood_branches_agree(model):
    """The paint-grid branch must reduce to the final-grid one when the selection is uniform."""
    shape = tuple(int(s) for s in model.mesh_shape)
    paint = tuple(int(s) for s in model.paint_shape)
    assert paint != shape, "fixture must have paint oversampling"
    rng = np.random.default_rng(11)

    model.selec_mesh = jnp.full(shape, 0.7)
    model.gxy_occ_mask3d = jnp.ones(shape, dtype=bool)
    model.gxy_count = 80.0

    g_paint = jnp.asarray(1.0 + 0.1 * rng.normal(size=paint))
    g_mesh = band_limit(g_paint, shape)
    counts = jnp.asarray(float(model.gxy_count) * 0.7 * (1.0 + 0.05 * rng.normal(size=shape)))

    lp_final = _obs_logprob(model, counts, g_mesh)
    model.selec_paint = jnp.full(paint, 0.7)
    lp_paint = _obs_logprob(model, counts, g_mesh, gxy_paint=g_paint)
    np.testing.assert_allclose(lp_paint, lp_final, rtol=1e-8)


def test_low_completeness_stays_finite(model):
    """Where the old ratio diverged, the counts likelihood must stay finite and well scaled."""
    shape = tuple(int(s) for s in model.mesh_shape)
    selec = jnp.full(shape, 1.0).at[0, 0, 0].set(1e-3)
    model.selec_mesh = selec
    model.gxy_occ_mask3d = jnp.ones(shape, dtype=bool)
    model.gxy_count = 500.0

    counts = jnp.asarray(float(model.gxy_count) * np.asarray(selec))
    lp = _obs_logprob(model, counts, jnp.ones(shape))
    assert np.isfinite(lp)

    with trace() as tr, seed(rng_seed=0), condition(data={"obs": counts}):
        model.likelihood(gxy_mesh=jnp.ones(shape))
    fn = tr["obs"]["fn"]
    # Poisson scaling all the way down: variance == expected counts, here 0.5 of them.
    np.testing.assert_allclose(float(np.asarray(fn.variance)[0, 0, 0]), 0.5, rtol=1e-10)


def test_obs_to_delta_inverts_the_likelihood_mean(model):
    shape = tuple(int(s) for s in model.mesh_shape)
    rng = np.random.default_rng(13)
    model.selec_mesh = _selection(shape, 9)
    mask = np.zeros(shape, dtype=bool)
    mask[2:, 2:, 2:] = True
    model.gxy_occ_mask3d = jnp.asarray(mask)
    model.gxy_count = 300.0

    delta = jnp.asarray(0.1 * rng.normal(size=shape))
    counts = float(model.gxy_count) * jnp.asarray(model.selec_mesh) * (1.0 + delta)
    back = np.asarray(model.obs_to_delta(counts))
    np.testing.assert_allclose(back[mask], np.asarray(delta)[mask], rtol=1e-9, atol=1e-12)
    assert np.all(back[~mask] == 0.0)


def test_predicted_counts_have_the_right_mean_and_shot_noise(model):
    """End-to-end: a closure draw is a count mesh, not an overdensity."""
    truth = model.predict(
        samples={"Omega_m": 0.3, "sigma8": 0.8, "b1": 1.0, "b2": 0.0, "bs2": 0.0, "bn2": 0.0},
        hide_base=False, frombase=True,
    )
    obs = np.asarray(truth["obs"])
    nbar = float(model.gxy_count)
    assert nbar > 10.0
    np.testing.assert_allclose(obs.mean() / nbar, 1.0, rtol=0.05)
    # Counts, not an overdensity: the scatter sits between the shot-noise floor sqrt(n̄) and
    # the clustering scale n̄, orders of magnitude away from the O(1) spread of 1 + delta_g.
    assert np.sqrt(nbar) < obs.std() < 5.0 * nbar


def test_painting_on_the_paint_grid_then_cropping_is_the_old_forward_model(model):
    """The refactor must not move gxy_mesh: crop-last equals crop-inside, exactly."""
    paint = tuple(int(s) for s in model.paint_shape)
    final = tuple(int(s) for s in model.mesh_shape)
    if paint == final:
        pytest.skip("model has no paint oversampling")
    rng = np.random.default_rng(17)
    n = 4000
    pos = jnp.asarray(rng.uniform(0.0, final[0], size=(n, 3)))
    w = jnp.asarray(1.0 + 0.3 * rng.normal(size=n))

    old = model.paint_and_deconv(pos, weights=w, from_shape=final) / (n / np.prod(final))
    new = band_limit(
        model.paint_and_deconv(pos, weights=w, from_shape=final, crop=False)
        / (n / np.prod(paint)),
        final,
    )
    np.testing.assert_allclose(np.asarray(new), np.asarray(old), rtol=0, atol=1e-10)


def test_selec_paint_crops_onto_selec_mesh():
    """The loader's two selection meshes must be one field seen at two resolutions."""
    paint, final = (14, 14, 14), (8, 8, 8)
    rng = np.random.default_rng(19)
    pos = jnp.asarray(rng.uniform(0.0, paint[0], size=(2000, 3)))
    shifted = interlace_accumulate([jnp.zeros(paint)] * INTERLACE_ORDER, pos)

    window = interlace_finalize(shifted, paint, final)
    combined = interlace_combine(shifted, paint) * float(np.prod(paint) / np.prod(final))
    np.testing.assert_allclose(
        np.asarray(band_limit(combined, final)), np.asarray(window), rtol=0, atol=1e-10
    )
