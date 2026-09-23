"""The field-level figures must be produced, and be correct, in every run geometry.

They are the figures the paper shows to claim the field is reconstructed, so the things that
would silently make them wrong -- an observer that is not at the centre of the box, a refined
projection sphere, a missing covariance -- are exercised here rather than assumed.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_run import _bands, plot_initial_conditions, plot_kappa_maps  # noqa: E402

from desi_cmb_fli.model import FieldLevelModel, default_config  # noqa: E402

jax.config.update("jax_enable_x64", True)

N_CHAINS = 2


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


def _run(model):
    """A truth and a set of chain positions, as the figures receive them from a real run."""
    cosmo = {"Omega_m": 0.315192, "sigma8": 0.811355}
    truth = model.predict(samples=cosmo, hide_base=False, hide_samp=False, hide_det=False,
                          frombase=True, rng=3)
    truth = {k: np.asarray(v) for k, v in truth.items() if v is not None}
    model.reset()
    model.condition(cosmo, frombase=True)
    rng = np.random.default_rng(0)
    field = np.asarray(truth["init_mesh_"])
    positions = {"init_mesh_": np.stack(
        [field + 0.01 * rng.normal(size=field.shape) for _ in range(N_CHAINS)])}
    return truth, positions


@pytest.mark.parametrize("observer_mode", ["center", "corner"])
def test_the_initial_conditions_figure_is_drawn_around_the_observer(observer_mode, tmp_path):
    """The slice must pass through the observer and the radii must be measured from it, so that
    the dashed shells and the radial profile mean the same thing wherever the observer sits."""
    model = FieldLevelModel(**_cfg(cmb_observer_mode=observer_mode))
    truth, positions = _run(model)
    out = tmp_path / "initial_conditions.png"
    plot_initial_conditions(model, truth, positions, out)
    assert out.exists() and out.stat().st_size > 0

    box = np.asarray(model.box_shape, dtype=float)
    shape = tuple(int(s) for s in model.init_shape)
    center = np.asarray(model.box_center, dtype=float)
    coord = [np.arange(shape[d]) * (box[d] / shape[d]) - box[d] / 2 + center[d] for d in range(3)]
    # The observer is at the origin of these coordinates, and inside the drawn plane.
    sl = int(np.argmin(np.abs(coord[2])))
    assert abs(coord[2][sl]) <= box[2] / shape[2]
    assert coord[0].min() <= 0.0 <= coord[0].max()
    if observer_mode == "corner":
        assert coord[0].min() >= -box[0] / shape[0]   # the corner really is at the edge
    else:
        np.testing.assert_allclose(coord[0].min(), -coord[0].max(), atol=box[0] / shape[0])


def test_the_initial_conditions_figure_refuses_mismatched_grids(tmp_path, capsys):
    model = FieldLevelModel(**_cfg())
    truth, positions = _run(model)
    truth["init_mesh"] = np.asarray(truth["init_mesh"])[:-1]
    out = tmp_path / "initial_conditions.png"
    plot_initial_conditions(model, truth, positions, out)
    assert "different grids" in capsys.readouterr().out
    assert not out.exists()


@pytest.mark.parametrize("proj_oversamp", [1, 2])
def test_the_convergence_figure_compares_maps_in_the_likelihood_band(proj_oversamp, tmp_path):
    """With proj_oversamp > 1 the model map lives on a finer sphere than the data. The figure has
    to bring it back to the observable band instead of comparing two different resolutions."""
    model = FieldLevelModel(**_cfg(cmb_proj_oversamp=proj_oversamp, cmb_shell_weights="linear"))
    truth, positions = _run(model)
    import healpy as hp
    assert np.asarray(truth["kappa_pred"]).size == hp.nside2npix(model.cmb_proj_nside)
    out = tmp_path / "kappa_maps.png"
    plot_kappa_maps(model, truth, positions, out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.parametrize("proj_oversamp", [1, 2])
def test_the_startup_slices_accept_a_refined_convergence_map(proj_oversamp, tmp_path):
    """In closure kappa_pred comes out of predict() on the refined sphere; the startup figure of
    run_inference must bring it to cmb_nside rather than scatter it into the observable mask."""
    from desi_cmb_fli.validation import plot_field_slices

    model = FieldLevelModel(**_cfg(cmb_proj_oversamp=proj_oversamp, cmb_shell_weights="linear"))
    truth, _ = _run(model)
    truth["kappa_obs"] = np.asarray(model.unpack_kappa_obs_to_map(truth["kappa_obs"]))
    plot_field_slices(truth, output_dir=tmp_path, box_shape=model.box_shape,
                      cmb_mask=model.cmb_mask, cmb_nside=model.cmb_nside, model=model)
    assert (tmp_path / "kappa_maps.png").exists()


def test_the_convergence_figure_is_skipped_without_a_map(tmp_path, capsys):
    model = FieldLevelModel(**_cfg())
    truth, positions = _run(model)
    truth.pop("kappa_pred")
    out = tmp_path / "kappa_maps.png"
    plot_kappa_maps(model, truth, positions, out)
    assert "skipping" in capsys.readouterr().out
    assert not out.exists()


def test_the_bands_cover_the_whole_multipole_range():
    for lmax in (8, 64, 256):
        bands = _bands(lmax)
        assert bands[0][0] == 2 and bands[-1][1] == lmax + 1
        assert all(b > a for a, b in bands)
