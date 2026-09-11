"""
Tests for the field-level model.
Adapted from benchmark-field-level to validate model construction and inference.
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from desi_cmb_fli.model import FieldLevelModel, default_config


@pytest.fixture
def small_model():
    """Create a small model for quick testing."""
    config = default_config.copy()
    config["mesh_shape"] = (16, 16, 16)
    config["box_shape"] = (100.0, 100.0, 100.0)
    return FieldLevelModel(**config)


def test_model_creation():
    """Test that model can be created with default config."""
    model = FieldLevelModel(**default_config)
    assert model.mesh_shape.shape == (3,)
    assert model.box_shape.shape == (3,)
    assert model.evolution in ["kaiser", "lpt", "nbody"]
    assert model.observable == "field"
    assert model.lightcone is True
    assert model.a_obs is None
    assert 0.0 < model.a_fid <= 1.0


def test_model_str(small_model):
    """Test model string representation."""
    model_str = str(small_model)
    assert "# CONFIG" in model_str
    assert "# INFOS" in model_str
    assert "mean_gxy_count" in model_str


def test_model_predict(small_model):
    """Test model prediction (forward simulation)."""
    truth_params = {
        "Omega_m": 0.3,
        "sigma8": 0.8,
        "b1": 1.0,
        "b2": 0.0,
        "bs2": 0.0,
        "bn2": 0.0,
    }

    result = small_model.predict(samples=truth_params, hide_base=False, frombase=True)

    assert "obs" in result
    assert result["obs"].shape == tuple(small_model.mesh_shape)
    assert jnp.all(jnp.isfinite(result["obs"]))


def test_model_conditioning(small_model):
    """Test model conditioning on observed data."""
    # Generate synthetic data
    truth_params = {"Omega_m": 0.3, "sigma8": 0.8, "b1": 1.0, "b2": 0.0, "bs2": 0.0, "bn2": 0.0}
    truth = small_model.predict(samples=truth_params, frombase=True)

    # Condition on observation
    small_model.reset()
    small_model.condition({"obs": truth["obs"]})

    # Model should now be conditioned
    assert small_model.model != small_model._model


def test_model_block(small_model):
    """Test model blocking."""
    small_model.reset()
    small_model.block()

    # After blocking, model should be wrapped
    assert small_model.model != small_model._model


def test_kaiser_posterior_init(small_model):
    """Test Kaiser posterior initialization."""
    # Generate synthetic data
    truth_params = {"Omega_m": 0.3, "sigma8": 0.8, "b1": 1.0, "b2": 0.0, "bs2": 0.0, "bn2": 0.0}
    truth = small_model.predict(samples=truth_params, frombase=True)

    # Get Kaiser posterior samples
    rng = jr.key(0)
    init_params = small_model.kaiser_post(rng, small_model.obs_to_delta(truth["obs"]), base=True)

    assert "init_mesh" in init_params
    assert init_params["init_mesh"].shape == (16, 16, 9)  # Hermitian symmetry


def test_model_latents_config():
    """Test latents configuration structure."""
    model = FieldLevelModel(**default_config)

    # Check cosmology latents
    assert "Omega_m" in model.latents
    assert "sigma8" in model.latents
    assert model.latents["Omega_m"]["group"] == "cosmo"

    # Check bias latents
    assert "b1" in model.latents
    assert "b2" in model.latents
    assert model.latents["b1"]["group"] == "bias"

    # Check init latents
    assert "init_mesh" in model.latents
    assert model.latents["init_mesh"]["group"] == "init"


def test_model_groups():
    """Test parameter grouping."""
    model = FieldLevelModel(**default_config)

    # Base groups
    assert "cosmo" in model.groups
    assert "bias" in model.groups
    assert "init" in model.groups

    # Sample groups (with underscore suffix)
    assert "cosmo_" in model.groups_
    assert "bias_" in model.groups_


def test_evolution_options():
    """Test different evolution options."""
    for evolution in ["kaiser", "lpt"]:
        config = default_config.copy()
        config["evolution"] = evolution
        config["mesh_shape"] = (16, 16, 16)
        config["box_shape"] = (100.0, 100.0, 100.0)

        model = FieldLevelModel(**config)
        assert model.evolution == evolution

        # Test forward simulation
        truth = model.predict(
            samples={"Omega_m": 0.3, "sigma8": 0.8, "b1": 1.0, "b2": 0.0, "bs2": 0.0, "bn2": 0.0},
            frombase=True,
        )
        assert "obs" in truth


def test_nbody_lightcone_not_implemented():
    config = default_config.copy()
    config["mesh_shape"] = (8, 8, 8)
    config["box_shape"] = (80.0, 80.0, 80.0)
    config["evolution"] = "nbody"
    config["a_obs"] = None

    model = FieldLevelModel(**config)
    with pytest.raises(NotImplementedError):
        model.predict(
            samples={"Omega_m": 0.3, "sigma8": 0.8, "b1": 1.0, "b2": 0.0, "bs2": 0.0, "bn2": 0.0},
            frombase=True,
        )


def test_precond_options(small_model):
    """Test different preconditioning options."""
    for precond in ["direct", "fourier", "kaiser", "kaiser_dyn"]:
        config = default_config.copy()
        config["precond"] = precond
        config["mesh_shape"] = (16, 16, 16)
        config["box_shape"] = (100.0, 100.0, 100.0)

        model = FieldLevelModel(**config)
        assert model.precond == precond


def test_model_cell_parameters():
    """Test derived cell parameters."""
    model = FieldLevelModel(**default_config)

    expected_cell_shape = model.box_shape / model.mesh_shape
    np.testing.assert_array_almost_equal(model.cell_shape, expected_cell_shape)

    assert model.k_funda > 0
    assert model.k_nyquist > model.k_funda
    assert model.gxy_count > 0
