"""End-to-end shape of the selection built from randoms, with the catalog readers faked.

The likelihood multiplies the model by ``selec_paint`` on the paint grid and scales the shot
noise by ``selec_mesh`` on the final grid. If those two ever stop being the same field seen at
two resolutions, the mean and the variance describe different surveys and nothing downstream
would say so -- hence this test on the loader itself.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import desi_cmb_fli.bricks as bricks
import desi_cmb_fli.cmb_lensing as cl
from desi_cmb_fli.bricks import band_limit
from desi_cmb_fli.model import FieldLevelModel, default_config

jax.config.update("jax_enable_x64", True)

MESH = (16, 16, 16)
N_GXY, N_RAND = 40_000, 400_000


@pytest.fixture
def model():
    config = default_config.copy()
    config["mesh_shape"] = MESH
    config["box_shape"] = (400.0, 400.0, 400.0)
    config["paint_oversamp"] = 1.5
    config["gxy_ngbar_free"] = False
    return FieldLevelModel(**config)


def _cell_positions(n, seed, mesh):
    """Positions filling a wedge of the box, so part of the mesh stays outside the survey."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, mesh[0], size=(4 * n, 3))
    keep = pos[:, 0] + pos[:, 1] < 1.4 * mesh[0]
    return jnp.asarray(pos[keep][:n])


@pytest.fixture
def faked(monkeypatch):
    def catalog2positions(path, cosmo, observer_position, box_shape, mesh_shape, z_range=None):
        return _cell_positions(N_GXY, 0, mesh_shape), N_GXY, 100.0, 900.0

    def randoms2positions(path, cosmo, observer_position, box_shape, mesh_shape,
                          z_range=None, row_range=None):
        lo, hi = (0, N_RAND) if row_range is None else row_range
        return _cell_positions(N_RAND, 1, mesh_shape)[lo:hi], hi - lo

    monkeypatch.setattr(bricks, "catalog2positions", catalog2positions)
    monkeypatch.setattr(bricks, "randoms2positions", randoms2positions)
    monkeypatch.setattr(bricks, "randoms_num_rows", lambda p: N_RAND)
    monkeypatch.setattr(cl, "_is_cubic_box_catalog", lambda p: False)


def _load(model, tmp_path, completeness_min=0.8):
    cfg = {
        "file": str(tmp_path / "gxy.fits"),
        "randoms": str(tmp_path / "rand.fits"),
        "z_range": [0.4, 1.1],
        "randoms_chunk_rows": 150_000,  # forces the streaming path over several chunks
        "randoms_cache_dir": None,
        "completeness_min": completeness_min,
    }
    return cl.load_abacus_galaxy_observation(cfg, model)


def test_selec_paint_band_limits_onto_selec_mesh(model, faked, tmp_path):
    truth = _load(model, tmp_path)
    mask = np.asarray(truth["gxy_occ_mask3d"])
    selec = np.asarray(model.selec_mesh)
    fine = np.asarray(band_limit(jnp.asarray(model.selec_paint), MESH))

    assert model.selec_paint.shape == tuple(int(s) for s in model.paint_shape)
    np.testing.assert_allclose(fine[mask], selec[mask], rtol=1e-5, atol=1e-5)


def test_the_observation_is_counts(model, faked, tmp_path):
    truth = _load(model, tmp_path)
    mask = np.asarray(truth["gxy_occ_mask3d"])
    obs = np.asarray(truth["obs"])

    # Total galaxies preserved by the painting, and the mean count matches n̄·S.
    assert 0.0 < mask.mean() < 1.0
    np.testing.assert_allclose(obs.sum(), N_GXY, rtol=1e-6)
    expected = float(model.gxy_count) * np.asarray(model.selec_mesh)[mask]
    np.testing.assert_allclose(obs[mask].mean(), expected.mean(), rtol=0.02)


def test_obs_to_delta_is_centred_on_zero(model, faked, tmp_path):
    truth = _load(model, tmp_path)
    delta = np.asarray(model.obs_to_delta(truth["obs"]))
    mask = np.asarray(truth["gxy_occ_mask3d"])
    assert abs(float(delta[mask].mean())) < 0.05
    assert np.all(delta[~mask] == 0.0)


def test_lowering_the_completeness_cut_keeps_the_likelihood_finite(model, faked, tmp_path):
    """The point of counts: a cut at 0.05 must widen the survey without any blow-up."""
    truth = _load(model, tmp_path, completeness_min=0.05)
    mask = np.asarray(truth["gxy_occ_mask3d"])
    selec = np.asarray(model.selec_mesh)[mask]
    assert selec.min() < 0.5, "the low-completeness edge cells must be kept"

    nbar = float(model.gxy_count) * np.asarray(model.selec_mesh)
    assert np.all(np.isfinite(np.asarray(truth["obs"])))
    assert np.all(nbar[mask] > 0.0)
