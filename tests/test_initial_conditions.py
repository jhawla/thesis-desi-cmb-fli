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


def test_lin_power_interp_matches_jax_cosmo():
    cosmo = planck18()
    ks = jnp.logspace(-3, 0, 8)
    interp = lin_power_interp(cosmo, a=0.8, n_interp=128)
    expected = jc.power.linear_matter_power(cosmo, ks, a=0.8)
    got = interp(ks)
    assert jnp.allclose(got, expected, rtol=2e-2)


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
# model.lin_pk_table: Eisenstein-Hu times the shape ratio of a tabulated spectrum
# ---------------------------------------------------------------------------

CLASS_TABLE = Path(__file__).resolve().parents[1] / "data/abacus_cosm000_CLASS_power.txt"


def _true_sigma8_sq(pow_fn):
    from desi_cmb_fli.bricks import _tophat8_variance

    k = np.logspace(-4.5, 1.3, 20000)
    return float(_tophat8_variance(k, np.asarray(pow_fn(jnp.asarray(k)))))


def test_lin_pk_table_gives_the_table_at_its_cosmology():
    """At the cosmology of the table the spectrum is the table's, at any a, normalised to sigma8."""
    from desi_cmb_fli.bricks import get_cosmology, lin_pk_ratio

    cosmo = get_cosmology(Omega_m=0.315192, sigma8=0.811355)
    ratio = lin_pk_ratio(CLASS_TABLE, cosmo)
    kt, pt = np.loadtxt(CLASS_TABLE, unpack=True)
    table = lambda k: jnp.exp(jnp.interp(jnp.log(k), np.log(kt), np.log(pt)))  # noqa: E731
    nodes = jnp.asarray(np.logspace(-4, 1, 256)[20:240:11])
    for a in (1.0, 0.5):
        got = np.asarray(lin_power_interp(cosmo, a=a, pk_ratio=ratio)(nodes))
        growth2 = float(jc.background.growth_factor(cosmo, jnp.atleast_1d(a))[0]) ** 2
        want = np.asarray(table(nodes)) * 0.811355**2 / _true_sigma8_sq(table) * growth2
        np.testing.assert_allclose(got, want, rtol=1e-4)


def test_lin_pk_table_keeps_sigma8_exact_and_eh_omega_m_shape_elsewhere():
    from desi_cmb_fli.bricks import get_cosmology, lin_pk_ratio

    ratio = lin_pk_ratio(CLASS_TABLE, get_cosmology(Omega_m=0.315192, sigma8=0.811355))
    k = jnp.asarray(np.logspace(-3, 0, 40))
    for om, s8 in ((0.25, 0.75), (0.40, 0.90)):
        cosmo = get_cosmology(Omega_m=om, sigma8=s8)
        new = lin_power_interp(cosmo, pk_ratio=ratio, n_interp=4096)
        np.testing.assert_allclose(_true_sigma8_sq(new), s8**2, rtol=2e-4)
        # the shape change with Omega_m is EH's: new / EH is the fixed ratio, up to a constant
        rel = np.asarray(new(k) / lin_power_interp(cosmo, n_interp=4096)(k))
        fixed = np.interp(np.log(np.asarray(k)), *ratio)
        np.testing.assert_allclose(rel / fixed, (rel / fixed).mean(), rtol=2e-3)


def test_lin_pk_table_is_differentiable_in_omega_m():
    import jax

    from desi_cmb_fli.bricks import get_cosmology, lin_pk_ratio, trans_phi2delta_interp

    ratio = lin_pk_ratio(CLASS_TABLE, get_cosmology(Omega_m=0.315192, sigma8=0.811355))
    k = jnp.asarray([0.005, 0.02, 0.05])

    def f(om):
        cosmo = get_cosmology(Omega_m=om, sigma8=0.81)
        return jnp.sum(lin_power_interp(cosmo, pk_ratio=ratio)(k)) + jnp.sum(
            trans_phi2delta_interp(cosmo, pk_ratio=ratio)(k))

    g = jax.grad(f)(0.3)
    assert np.isfinite(float(g)) and float(g) != 0.0
