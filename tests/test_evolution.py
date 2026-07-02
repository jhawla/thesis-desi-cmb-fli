"""Unit tests for gravitational evolution."""

import jax.numpy as jnp
import jax.random as jr
import jax_cosmo as jc
import numpy as np

from desi_cmb_fli.bricks import lin_power_mesh
from desi_cmb_fli.nbody import (
    a2f,
    a2g,
    invlaplace_kernel,
    lpt,
    pm_forces,
    rfftk,
)
from desi_cmb_fli.utils import (
    ch2rshape,
    safe_div,
)


def planck18():
    """Return a lightweight Planck-like cosmology for tests."""
    return jc.Cosmology(
        Omega_c=0.2607,
        Omega_b=0.0490,
        Omega_k=0.0,
        h=0.6766,
        n_s=0.9665,
        sigma8=0.8102,
        w0=-1.0,
        wa=0.0,
    )


def test_safe_div_handles_zero_denominator():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([2.0, 0.0, 3.0])
    result = safe_div(x, y)
    expected = jnp.array([0.5, 0.0, 1.0])
    assert jnp.allclose(result, expected)


def test_ch2rshape_inverts_hermitian_convention():
    kshape = (4, 4, 3)  # Hermitian shape from (4,4,4) real field
    rshape = ch2rshape(kshape)
    assert rshape == (4, 4, 4)


def test_rfftk_returns_correct_shapes():
    mesh_shape = (8, 8, 8)
    kx, ky, kz = rfftk(mesh_shape)
    assert kx.shape == (8, 1, 1)
    assert ky.shape == (1, 8, 1)
    assert kz.shape == (1, 1, 5)  # rfft along last axis


def test_invlaplace_kernel_zero_at_k_zero():
    mesh_shape = (4, 4, 4)
    kvec = rfftk(mesh_shape)
    kernel = invlaplace_kernel(kvec, fd=False)
    # k=0 mode should give zero (safe division)
    assert kernel[0, 0, 0] == 0.0


def test_pm_forces_shape_matches_positions():
    mesh_shape = np.array([8, 8, 8])
    n_particles = int(np.prod(mesh_shape))
    key = jr.PRNGKey(42)
    pos = jr.uniform(key, (n_particles, 3)) * mesh_shape
    forces = pm_forces(pos, mesh_shape)
    assert forces.shape == (n_particles, 3)


def test_a2g_growth_factor_increases_with_scale_factor():
    cosmo = planck18()
    cosmo._workspace = {}
    a_early = 0.5
    a_late = 1.0
    g_early = a2g(cosmo, a_early)
    g_late = a2g(cosmo, a_late)
    assert g_late > g_early


def test_a2f_growth_rate_positive():
    cosmo = planck18()
    cosmo._workspace = {}
    a = 0.8
    f = a2f(cosmo, a)
    assert f > 0


def test_lpt_produces_displacements_and_velocities():
    cosmo = planck18()
    cosmo._workspace = {}
    mesh_shape = np.array([4, 4, 4])
    box_shape = np.array([100.0, 100.0, 100.0])

    # Generate initial field
    pmesh = lin_power_mesh(cosmo, mesh_shape, box_shape, a=1.0)
    key = jr.PRNGKey(1)
    init_mesh_real = jr.normal(key, shape=mesh_shape)
    init_mesh = jnp.fft.rfftn(init_mesh_real) * jnp.sqrt(pmesh)

    # Create particle positions
    pos = jnp.indices(mesh_shape, dtype=float).reshape(3, -1).T

    # Run 1LPT
    a_obs = 0.8
    dpos, vel = lpt(cosmo, init_mesh, pos, a=a_obs, order=1)

    # Check outputs
    assert dpos.shape == pos.shape
    assert vel.shape == pos.shape


def test_lpt_order2_includes_second_order_correction():
    mesh_shape = np.array([4, 4, 4])
    box_shape = np.array([100.0, 100.0, 100.0])

    pmesh_template = lin_power_mesh(planck18(), mesh_shape, box_shape, a=1.0)
    key = jr.PRNGKey(2)
    init_mesh_real = jr.normal(key, shape=mesh_shape)
    init_mesh = jnp.fft.rfftn(init_mesh_real) * jnp.sqrt(pmesh_template)

    pos = jnp.indices(mesh_shape, dtype=float).reshape(3, -1).T
    a_obs = 0.8

    # Use fresh cosmology objects for each order to avoid cache issues
    cosmo1 = planck18()
    cosmo1._workspace = {}
    dpos1, vel1 = lpt(cosmo1, init_mesh, pos, a=a_obs, order=1)

    cosmo2 = planck18()
    cosmo2._workspace = {}
    dpos2, vel2 = lpt(cosmo2, init_mesh, pos, a=a_obs, order=2)

    # 2LPT should differ from 1LPT
    assert not jnp.allclose(dpos1, dpos2)
    assert not jnp.allclose(vel1, vel2)


def test_lpt_accepts_particlewise_scale_factors():
    cosmo = planck18()
    cosmo._workspace = {}
    mesh_shape = np.array([4, 4, 4])
    box_shape = np.array([100.0, 100.0, 100.0])

    key = jr.PRNGKey(3)
    init_mesh_real = jr.normal(key, shape=mesh_shape)
    pmesh = lin_power_mesh(cosmo, mesh_shape, box_shape, a=1.0)
    init_mesh = jnp.fft.rfftn(init_mesh_real) * jnp.sqrt(pmesh)

    pos = jnp.indices(mesh_shape, dtype=float).reshape(3, -1).T
    a_part = jnp.linspace(0.6, 0.9, pos.shape[0])[:, None]

    dpos, vel = lpt(cosmo, init_mesh, pos, a=a_part, order=2)
    assert dpos.shape == pos.shape
    assert vel.shape == pos.shape
    assert jnp.all(jnp.isfinite(dpos))
    assert jnp.all(jnp.isfinite(vel))
