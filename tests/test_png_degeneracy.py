"""How much of ``add_png`` is really the ``phi delta`` operator?

montecosmo absorbs an ``fNL <-> fNL_bpd`` degeneracy by subtracting ``2 fNL`` from ``fNL_bpd``,
on the argument that the squeezed part of the ``fNL phi^2`` that ``add_png`` injects into the
linear field is the same operator ``fNL_bpd`` multiplies in the Lagrangian weights. Its author
flagged the coefficient as unverified. These tests measure it on this forward model: the answer
is well below 2, and most of the ``add_png`` response is not that operator at all, so we do not
apply the subtraction. If anyone re-derives it, this is the check to re-run.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

from desi_cmb_fli.bricks import b_phi, b_phi_delta, fNL_bias
from desi_cmb_fli.model import FieldLevelModel, default_config

jax.config.update("jax_enable_x64", True)


def test_the_free_latents_are_passed_through():
    assert fNL_bias(fNL=7.0, b1=1.0, b2=0.0, png_type="fNL_bias",
                    fNL_bp=3.0, fNL_bpd=100.0) == (3.0, 100.0)


def test_universality_derives_them_from_the_bias():
    bp, bpd = fNL_bias(fNL=5.0, b1=1.2, b2=0.3, png_type="fNL")
    assert bp == pytest.approx(5.0 * b_phi(1.2, 1.0))
    assert bpd == pytest.approx(5.0 * b_phi_delta(1.2, 0.3))
    assert fNL_bias(fNL=5.0, b1=1.2, b2=0.3, png_type=None) == (0.0, 0.0)


@pytest.fixture(scope="module")
def model():
    config = default_config.copy()
    config["mesh_shape"] = (16, 16, 16)
    config["box_shape"] = (7500.0, 7500.0, 7500.0)
    config["png_type"] = "fNL_bias"
    config["a_obs"] = 0.5556  # snapshot: no lightcone geometry in the way
    config["curved_sky"] = False
    config["latents"] = {k: dict(v) for k, v in config["latents"].items()}
    for k in ("fNL", "fNL_bp", "fNL_bpd"):
        config["latents"][k] |= {"loc": 0.0, "scale": 1e4, "scale_fid": 1.0, "loc_fid": 0.0}
    return FieldLevelModel(**config)


def _gxy(model, fNL, fNL_bpd):
    p = {"Omega_m": 0.3, "sigma8": 0.8, "b1": 1.0, "b2": 0.0, "bs2": 0.0, "bn2": 0.0,
         "fNL": fNL, "fNL_bp": 0.0, "fNL_bpd": fNL_bpd}
    return np.asarray(model.predict(samples=p, hide_base=False, hide_det=False,
                                    frombase=True, rng=jax.random.key(0))["gxy_mesh"])


def test_add_png_is_mostly_not_the_phi_delta_operator(model):
    """Project the add_png response onto the phi*delta response, at a fixed initial field.

    Measured across box 1200-7500 and mesh 16-64 (k_Nyq/k_fund from 8 to 32): coefficient
    0.72-0.82 and residual ~78%, both flat in resolution. The squeezed-limit value 2 is not
    reached, and the operator overlap is small, so absorbing it would rename the latent to a
    combination that is not the one the model is degenerate along.
    """
    T = 10.0
    base = _gxy(model, 0.0, 0.0)
    assert np.array_equal(base, _gxy(model, 0.0, 0.0)), "prediction is not deterministic"
    d_bpd = _gxy(model, 0.0, T) - base           # pure phi*delta, amplitude T
    d_fnl = _gxy(model, T, 0.0) - base           # pure add_png

    assert np.std(d_bpd) > 0, "the phi*delta operator has no effect; the test is vacuous"
    coeff = float(np.sum(d_fnl * d_bpd) / np.sum(d_bpd * d_bpd))
    residual = float(np.std(d_fnl - coeff * d_bpd) / np.std(d_fnl))

    assert 0.3 < coeff < 1.5, f"coefficient {coeff:.3f} (montecosmo assumes 2)"
    assert residual > 0.5, f"residual {residual:.1%}: the overlap is larger than measured"
