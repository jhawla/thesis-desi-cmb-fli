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


def _minimal_config(**model_overrides):
    return {
        "model": {"box_shape": [100.0, 100.0, 100.0], "cell_size": 12.5, "evolution": "lpt",
                  "lpt_order": 1, "gxy_density": 1e-3, "lightcone": False, "a_obs": 1.0,
                  **model_overrides},
    }


def test_an_unknown_model_key_is_an_error():
    """A typo must not silently fall back on the default."""
    from desi_cmb_fli.model import get_model_from_config

    with pytest.raises(ValueError, match="paint_oversmp"):
        get_model_from_config(_minimal_config(paint_oversmp=1.75))
    cfg = _minimal_config()
    cfg["cmb_lensing"] = {"enabled": False, "chi_minn": 700}
    with pytest.raises(ValueError, match="chi_minn"):
        get_model_from_config(cfg)


def test_sampled_scalar_latents_lists_what_the_prior_samples(small_model):
    names = small_model.sampled_scalar_latents()
    assert "Omega_m" in names and "b1" in names and "init_mesh" not in names


def test_cmb_lensing_needs_particles():
    config = default_config.copy()
    config.update({"mesh_shape": (8, 8, 8), "box_shape": (400.0, 400.0, 400.0),
                   "evolution": "kaiser", "a_obs": 1.0, "cmb_enabled": True,
                   "cmb_noise_nell": {"ell": np.arange(64.0), "N_ell": np.full(64, 1e-9)}})
    with pytest.raises(ValueError, match="kaiser"):
        FieldLevelModel(**config)


def test_mclmc_run_can_drop_the_field_from_the_samples():
    import jax
    from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
    from blackjax.mcmc import mclmc

    from desi_cmb_fli.samplers import get_mclmc_run

    def logdf(x):
        return -0.5 * (jnp.sum(x["init_mesh_"] ** 2) + x["b1_"] ** 2)

    pos = {"init_mesh_": jnp.zeros(8), "b1_": jnp.array(0.0)}
    state = mclmc.init(position=pos, logdensity_fn=logdf, rng_key=jr.key(0))
    config = MCLMCAdaptationState(L=1.0, step_size=0.3, inverse_mass_matrix=jnp.ones(9))
    for drop, expected in (((), {"init_mesh_", "b1_"}), (("init_mesh_",), {"b1_"})):
        run = jax.jit(get_mclmc_run(logdf, n_samples=4, progress_bar=False, drop_keys=drop))
        _, samples = run(jr.key(1), state, config)
        assert expected <= set(samples) and not (set(drop) & set(samples))


def test_omega_m_bounds_outside_the_linear_power_emulator_range_are_an_error():
    """The emulator is validated for Omega_m in ACE_OMEGA_M_RANGE only (docs/pipeline.md §7.11)."""
    from desi_cmb_fli.model import get_model_from_config

    cfg = _minimal_config()
    cfg["latents"] = {"Omega_m": {"low": 0.1, "high": 0.6}}
    with pytest.raises(ValueError, match="Omega_m prior bounds"):
        get_model_from_config(cfg)


def test_forward_model_with_png_and_its_gradient_run():
    import jax

    from desi_cmb_fli.model import get_model_from_config

    model, _ = get_model_from_config(_minimal_config(png_type="fNL"))
    scalars = {"Omega_m": 0.315192, "sigma8": 0.811355, "b1": 1.0, "fNL": 50.0}
    truth = model.predict(samples=scalars, hide_base=False, hide_samp=False, hide_det=False,
                          frombase=True, rng=0)
    assert np.all(np.isfinite(np.asarray(truth["obs"])))
    model.reset()
    model.condition({"obs": truth["obs"]})
    model.condition({n: truth[n] for n in model.sampled_scalar_latents()}, frombase=True)
    model.block()
    g = jax.grad(model.logpdf)({"init_mesh_": jnp.asarray(truth["init_mesh_"])})["init_mesh_"]
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.abs(g).max()) > 0


def test_bias_evolution_samples_b1_alpha_on_the_light_cone_only():
    """bias_evolution 'redshift' samples b1_alpha on the light cone only; 'none' never does."""
    from desi_cmb_fli.model import FieldLevelModel, default_config

    def bias_group(**kw):
        cfg = default_config.copy()
        cfg.update({"mesh_shape": (8, 8, 8), "box_shape": (400.0, 400.0, 400.0), "evolution": "kaiser",
                    "cmb_enabled": False, **kw})
        return FieldLevelModel(**cfg).groups["bias"]

    assert "b1_alpha" in bias_group(a_obs=None)
    assert "b1_alpha" not in bias_group(a_obs=None, bias_evolution="none")
    assert "b1_alpha" not in bias_group(a_obs=1.0)


def test_b1_sigma8_same_posterior():
    """b1_sigma8 changes the sampled coordinate, not the posterior: reparam goes back and forth,
    and at one physical point the log-density in the new coordinates differs from the one in b1 by
    the Jacobian -log(sigma8 / sigma8_fid) alone (docs/pipeline.md 5)."""
    cfg = default_config | {"mesh_shape": (16, 16, 16), "box_shape": (100.0, 100.0, 100.0)}
    old = FieldLevelModel(**cfg)
    new = FieldLevelModel(**(cfg | {"b1_sigma8": True}))
    scalars = {"Omega_m": 0.3, "sigma8": 0.8, "b1": 1.0, "b2": 0.0, "bs2": 0.0, "bn2": 0.0}
    truth = old.predict(samples=scalars, frombase=True, hide_base=False, hide_samp=False)
    base = old.reparam({k: v for k, v in truth.items() if k.endswith("_")})
    base = {k: v for k, v in base.items() if k in old.latents or k == "init_mesh"}
    for m in (old, new):
        m.reset()
        m.condition({"obs": truth["obs"]})
        m.block()
    s8_fid = old.loc_fid["sigma8"]
    for s8, b1 in [(0.6, 1.3), (1.0, 0.7), (s8_fid, 1.1)]:
        point = base | {"sigma8": s8, "b1": b1}
        z_old, z_new = old.reparam(point, inv=True), new.reparam(point, inv=True)
        back = new.reparam(z_new)
        assert np.isclose(float(back["b1"]), b1, rtol=1e-5)
        assert np.isclose(float(back["sigma8"]), s8, rtol=1e-6)
        diff = float(new.logpdf(z_new) - old.logpdf(z_old))
        assert np.isclose(diff, -np.log(s8 / s8_fid), atol=2e-2), (s8, diff)
