"""Unit tests for initial conditions generation."""

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import jax_cosmo as jc
import numpy as np

from desi_cmb_fli.bricks import (
    lin_power_interp,
    lin_power_mesh,
)
from desi_cmb_fli.nbody import rfftk
from desi_cmb_fli.utils import (
    cgh2rg,
    rg2cgh,
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


def test_lin_power_mesh_consistent_with_interp():
    cosmo = planck18()
    mesh_shape = np.array([4, 4, 4])
    box_shape = np.array([100.0, 100.0, 100.0])
    mesh = lin_power_mesh(cosmo, mesh_shape, box_shape, a=1.0, n_interp=64)
    kvec = rfftk(mesh_shape)
    kmesh = (
        sum(
            (ki * (m / box_len)) ** 2
            for ki, m, box_len in zip(kvec, mesh_shape, box_shape, strict=False)
        )
        ** 0.5
    )
    interp = lin_power_interp(cosmo, a=1.0, n_interp=64)
    expected = interp(kmesh) * (mesh_shape / box_shape).prod()
    assert np.allclose(np.asarray(mesh), np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_rg2cgh_cgh2rg_roundtrip_recovers_real_field():
    key = jr.PRNGKey(0)
    mesh = jr.normal(key, (4, 4, 4), dtype=jnp.float32)
    meshk = rg2cgh(mesh, norm="backward")
    recovered = cgh2rg(meshk, norm="backward")
    assert jnp.allclose(recovered, mesh, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Linear power: the ACE emulator (docs/pipeline.md §2.1)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CLASS_TABLE = ROOT / "data/abacus_cosm000_CLASS_power.txt"


def _true_sigma8_sq(pow_fn):
    from desi_cmb_fli.bricks import _tophat8_variance

    k = np.logspace(-4.5, 1.3, 20000)
    return float(_tophat8_variance(k, np.asarray(pow_fn(jnp.asarray(k)))))


def test_ace_evaluator_reproduces_jaxmapse():
    """The pure-JAX evaluation of the network equals jaxmapse's Pk_lin_cb (reference written by
    jaxmapse 0.1.1 at z = 1 with D = 1, Abacus c000 with Omega_m moved)."""
    from desi_cmb_fli.bricks import _ace_power, get_cosmology

    ref = np.load(ROOT / "tests/data/ace_jaxmapse_reference.npz")
    for i, om in enumerate(ref["omega_m"]):
        got = np.asarray(_ace_power(get_cosmology(Omega_m=float(om)), jnp.asarray(ref["k"])))
        np.testing.assert_allclose(got, ref[f"pk_{i}"], rtol=1e-4)


def test_lin_power_matches_the_class_spectrum_of_the_abacus_ics():
    """At the Abacus cosmology the shape is the CLASS table's (cb, z = 1) over the likelihood band."""
    from desi_cmb_fli.bricks import get_cosmology

    kt, pt = np.loadtxt(CLASS_TABLE, unpack=True)
    table = lambda k: jnp.exp(jnp.interp(jnp.log(k), np.log(kt), np.log(pt)))  # noqa: E731
    pow_fn = lin_power_interp(get_cosmology(Omega_m=0.315192, sigma8=0.811355), n_interp=4096)
    k = jnp.asarray(np.logspace(-3, np.log10(0.058), 50))
    got = np.asarray(pow_fn(k)) / 0.811355**2
    want = np.asarray(table(k)) / _true_sigma8_sq(table)
    np.testing.assert_allclose(got / want, 1.0, rtol=2e-3)


def test_lin_power_keeps_sigma8_exact_and_grows_with_the_emulator_growth():
    from desi_cmb_fli.bricks import get_cosmology
    from desi_cmb_fli.nbody import a2g

    k = jnp.asarray(np.logspace(-3, 0, 20))
    for om, s8 in ((0.25, 0.75), (0.40, 0.90)):
        cosmo = get_cosmology(Omega_m=om, sigma8=s8)
        np.testing.assert_allclose(_true_sigma8_sq(lin_power_interp(cosmo, n_interp=4096)), s8**2, rtol=2e-4)
        growth2 = float(np.squeeze(a2g(cosmo, jnp.asarray(0.5)))) ** 2
        np.testing.assert_allclose(np.asarray(lin_power_interp(cosmo, a=0.5)(k)),
                                   np.asarray(lin_power_interp(cosmo)(k)) * growth2, rtol=1e-4)


def test_lin_power_is_differentiable_in_omega_m():
    import jax

    from desi_cmb_fli.bricks import get_cosmology, trans_phi2delta_interp

    k = jnp.asarray([0.005, 0.02, 0.05])

    def f(om):
        cosmo = get_cosmology(Omega_m=om, sigma8=0.81)
        return jnp.sum(jnp.log(lin_power_interp(cosmo)(k))) + jnp.sum(
            jnp.log(trans_phi2delta_interp(cosmo)(k)))

    g, eps = float(jax.grad(f)(0.3)), 1e-3
    assert np.isfinite(g) and g != 0.0
    np.testing.assert_allclose(g, float(f(0.3 + eps) - f(0.3 - eps)) / (2 * eps), rtol=2e-2)
