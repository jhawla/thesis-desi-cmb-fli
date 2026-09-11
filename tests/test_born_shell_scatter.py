"""Equivalence test for the single-pass Born shell scatter.

Pins convergence_Born_spherical against a literal transcription of the previous
per-shell implementation (jaxpm paint_particles_spherical inside a lax.scan).
The two must agree to floating-point tolerance on the kappa map itself, not on
any intermediate.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import (
    _box_ray_intervals,
    convergence_Born_spherical,
    lensing_kernel,
    linear_shell_volumes,
)

jax.config.update("jax_enable_x64", True)

NSIDE = 8
N_SHELLS = 6
BOX = 1000.0
MESH = (16, 16, 16)
CHI_MAX = 400.0
Z_SOURCE = 1089.28


def reference_born(cosmo, pos, box_shape, mesh_shape, observer, r_shells,
                   a_shells, d_r, nside, mask, z_source, t_enter, t_exit):
    """The previous implementation, transcribed verbatim."""
    import jax_cosmo as jc
    from jaxpm.spherical import paint_particles_spherical

    chi_s = jc.background.radial_comoving_distance(cosmo, 1.0 / (1.0 + z_source))[0]
    observer = jnp.asarray(observer, dtype=float)
    box_size_jnp = jnp.asarray(box_shape, dtype=float)
    npix_full = hp.nside2npix(nside)
    mask_jnp = jnp.asarray(mask)
    r_shells_jnp, a_shells_jnp = jnp.asarray(r_shells), jnp.asarray(a_shells)
    d_r_arr = jnp.full(len(r_shells), d_r)
    t_enter_jnp, t_exit_jnp = jnp.asarray(t_enter), jnp.asarray(t_exit)
    n_bar = float(pos.shape[0]) / float(np.prod(np.asarray(box_shape, dtype=float)))

    def scan_fn(kappa_acc, i):
        chi_i, a_i, dr_i = r_shells_jnp[i], a_shells_jnp[i], d_r_arr[i]
        R_min, R_max = chi_i - 0.5 * dr_i, chi_i + 0.5 * dr_i
        rho_full = paint_particles_spherical(
            positions=pos, nside=nside, observer_position=observer,
            R_min=R_min, R_max=R_max, box_size=box_size_jnp,
            mesh_shape=tuple(int(x) for x in mesh_shape), method="bilinear",
        )
        is_fully_inside = (t_enter_jnp <= R_min) & (t_exit_jnp >= R_max)
        shell_mask = mask_jnp & is_fully_inside
        delta_full = jnp.where(
            jnp.isfinite(rho_full) & shell_mask, rho_full / n_bar - 1.0, 0.0
        )
        W_i = lensing_kernel(cosmo, chi_i, a_i, chi_s)
        return kappa_acc + delta_full * dr_i * W_i, None

    kappa_hp, _ = jax.lax.scan(scan_fn, jnp.zeros(npix_full), jnp.arange(len(r_shells)))
    return kappa_hp[mask_jnp]


@pytest.fixture
def setup():
    cosmo = get_cosmology(Omega_m=0.315192, sigma8=0.811355)
    observer = np.array([BOX / 2] * 3)
    d_r = CHI_MAX / N_SHELLS
    r_shells = np.linspace(d_r / 2, CHI_MAX - d_r / 2, N_SHELLS)
    a_shells = np.full(N_SHELLS, 0.8)
    mask = np.ones(hp.nside2npix(NSIDE), dtype=bool)
    t_enter, t_exit = _box_ray_intervals(observer, [BOX] * 3, NSIDE)

    rng = np.random.default_rng(0)
    pos = jnp.asarray(rng.uniform(0, MESH[0], size=(20_000, 3)))
    return {
        "cosmo": cosmo, "pos": pos, "observer": observer, "r_shells": r_shells,
        "a_shells": a_shells, "d_r": d_r, "mask": mask,
        "t_enter": t_enter, "t_exit": t_exit,
    }


def _call(s, return_full=False, shell_weights="nearest"):
    return convergence_Born_spherical(
        s["cosmo"], s["pos"], [BOX] * 3, MESH, s["observer"], s["r_shells"],
        s["a_shells"], s["d_r"], NSIDE, s["mask"], Z_SOURCE,
        t_enter=s["t_enter"], t_exit=s["t_exit"], return_full=return_full,
        shell_weights=shell_weights,
    )


def _move_particle_radially(s, i, r_new):
    pos = np.asarray(s["pos"]).copy()
    obs_cell = s["observer"] * MESH[0] / BOX
    direction = pos[i] - obs_cell
    direction /= np.linalg.norm(direction)
    pos[i] = obs_cell + direction * r_new * MESH[0] / BOX
    return jnp.asarray(pos)


def test_matches_the_per_shell_implementation(setup):
    new = np.asarray(_call(setup))
    ref = np.asarray(reference_born(
        setup["cosmo"], setup["pos"], [BOX] * 3, MESH, setup["observer"],
        setup["r_shells"], setup["a_shells"], setup["d_r"], NSIDE,
        setup["mask"], Z_SOURCE, setup["t_enter"], setup["t_exit"],
    ))

    assert new.shape == ref.shape
    assert np.all(np.isfinite(new))
    scale = np.abs(ref).max()
    assert scale > 0, "degenerate reference: the test would prove nothing"
    np.testing.assert_allclose(new, ref, rtol=1e-10, atol=1e-12 * scale)


def test_gradient_matches_the_per_shell_implementation(setup):
    """The sampler differentiates through this, so pin the gradient too.

    A plain kappa.sum() is degenerate here: each particle contributes a total
    bilinear weight of 1 to its shell wherever it sits inside that shell, so
    the summed map is piecewise constant and both gradients vanish. Weighting
    the pixels breaks that conservation and exercises the real scatter.
    """
    weights = jnp.asarray(
        np.random.default_rng(1).normal(size=int(setup["mask"].sum()))
    )

    def loss(use_new):
        def inner(pos):
            if use_new:
                out = _call(dict(setup, pos=pos))
            else:
                out = reference_born(
                    setup["cosmo"], pos, [BOX] * 3, MESH, setup["observer"],
                    setup["r_shells"], setup["a_shells"], setup["d_r"], NSIDE,
                    setup["mask"], Z_SOURCE, setup["t_enter"], setup["t_exit"],
                )
            return (out * weights).sum()
        return inner

    g_new = np.asarray(jax.grad(loss(True))(setup["pos"]))
    g_ref = np.asarray(jax.grad(loss(False))(setup["pos"]))
    scale = np.abs(g_ref).max()
    assert scale > 0, "degenerate loss: the test would prove nothing"
    np.testing.assert_allclose(g_new, g_ref, rtol=1e-8, atol=1e-10 * scale)


def test_rejects_non_contiguous_shells(setup):
    s = dict(setup)
    s["r_shells"] = setup["r_shells"] * 1.5
    with pytest.raises(ValueError, match="contiguous shells"):
        _call(s)


def test_works_under_jit(setup):
    """model.py always calls with return_full=True, from inside jax.jit
    (pmap(jit(...)) in samplers.py) -- reproduces that, since jax.jit traces
    branches bare jax.grad leaves concrete and a prior version only failed here.
    """
    jitted = jax.jit(lambda pos: _call({**setup, "pos": pos}, return_full=True))
    out = np.asarray(jitted(setup["pos"]))
    np.testing.assert_allclose(out, np.asarray(_call(setup, return_full=True)), rtol=1e-10)


def test_gradient_works_under_jit(setup):
    weights = jnp.asarray(np.random.default_rng(2).normal(size=int(hp.nside2npix(NSIDE))))

    def loss(pos):
        return (_call({**setup, "pos": pos}, return_full=True) * weights).sum()

    g_eager = np.asarray(jax.grad(loss)(setup["pos"]))
    g_jit = np.asarray(jax.jit(jax.grad(loss))(setup["pos"]))
    np.testing.assert_allclose(g_jit, g_eager, rtol=1e-10)


# ---------------------------------------------------------------------------
# shell_weights='linear'
# ---------------------------------------------------------------------------


def test_linear_volumes_match_the_tent_integral(setup):
    """Closed form against numerical quadrature of Omega_pix * int tent_i(r) r^2 dr, and
    against the algebraic value r_i^2 d_r + d_r^3 / 6 for the shells whose tent clears
    the origin (a tent overlapping r=0 is truncated there, and must be)."""
    r_shells, d_r = setup["r_shells"], setup["d_r"]
    npix = hp.nside2npix(NSIDE)
    vol = linear_shell_volumes(r_shells, d_r, npix)
    omega = 4.0 * np.pi / npix
    clears = r_shells >= d_r
    assert clears.sum() and not clears.all(), "fixture must exercise both cases"
    np.testing.assert_allclose(
        vol[clears], omega * (r_shells[clears] ** 2 * d_r + d_r**3 / 6.0), rtol=1e-12
    )
    for i, c in enumerate(r_shells):
        r = np.linspace(max(c - d_r, 0.0), c + d_r, 200001)
        w = np.clip(1.0 - np.abs(r - c) / d_r, 0.0, 1.0)
        np.testing.assert_allclose(vol[i], omega * np.trapezoid(w * r**2, r), rtol=1e-7)


def test_linear_conserves_the_particle_weight(setup):
    """Every in-range particle contributes a total weight of one, whichever shells it is
    split between: the two maps carry the same mass, so the kappa sums agree once the
    per-shell volumes are the matching ones."""
    hard = np.asarray(_call(setup, return_full=True))
    lin = np.asarray(_call(setup, return_full=True, shell_weights="linear"))
    assert np.all(np.isfinite(lin))
    assert np.corrcoef(hard, lin)[0, 1] > 0.98
    np.testing.assert_allclose(lin.std(), hard.std(), rtol=0.1)


def test_linear_is_continuous_across_a_shell_edge_where_nearest_jumps(setup):
    i = 0
    edge = setup["r_shells"][2] + setup["d_r"] / 2
    radii = np.linspace(edge - 1.0, edge + 1.0, 9)

    def steps(shell_weights):
        maps = [np.asarray(_call({**setup, "pos": _move_particle_radially(setup, i, r)},
                                 return_full=True, shell_weights=shell_weights)) for r in radii]
        return np.array([np.abs(maps[j + 1] - maps[j]).max() for j in range(len(radii) - 1)])

    hard, lin = steps("nearest"), steps("linear")
    assert hard.max() > 100 * np.median(hard)
    assert lin.max() < 3 * np.median(lin)
    assert lin.max() < 0.05 * hard.max()


def test_linear_gradient_works_under_jit(setup):
    weights = jnp.asarray(np.random.default_rng(3).normal(size=int(hp.nside2npix(NSIDE))))

    def loss(pos):
        return (_call({**setup, "pos": pos}, return_full=True, shell_weights="linear") * weights).sum()

    g_eager = np.asarray(jax.grad(loss)(setup["pos"]))
    g_jit = np.asarray(jax.jit(jax.grad(loss))(setup["pos"]))
    scale = np.abs(g_eager).max()
    assert np.all(np.isfinite(g_eager)) and scale > 0
    np.testing.assert_allclose(g_jit, g_eager, rtol=1e-10, atol=1e-12 * scale)


def test_linear_rejects_non_uniform_shells(setup):
    s = dict(setup)
    s["d_r"] = np.array([setup["d_r"] * (1.1 if k == 0 else 1.0) for k in range(N_SHELLS)])
    s["r_shells"] = np.cumsum(s["d_r"]) - s["d_r"] / 2
    with pytest.raises(ValueError, match="uniform"):
        _call(s, shell_weights="linear")


def test_unknown_shell_weights_is_rejected(setup):
    with pytest.raises(ValueError, match="shell_weights"):
        _call(setup, shell_weights="cubic")


@pytest.mark.parametrize("end", ["inner", "outer"])
def test_linear_switches_particles_on_and_off_continuously_at_both_ends(setup, end):
    """Both ends of the radial range sit inside the particle cloud: chi_min at the inner
    one, the box boundary at the outer one. With hard bins a particle crossing either end
    switches its whole weight at once; the tent must ramp it. The outer end carries ~100x
    more crossings than the inner one, so it dominates the sampler's energy error."""
    s = dict(setup)
    s["r_shells"] = setup["r_shells"] + 100.0  # move the inner end into the cloud
    span = setup["d_r"]
    centre = s["r_shells"][0] if end == "inner" else s["r_shells"][-1]
    radii = np.linspace(centre - 1.2 * span, centre + 1.2 * span, 25)

    def steps(shell_weights):
        maps = [np.asarray(_call({**s, "pos": _move_particle_radially(s, 0, r)},
                                 return_full=True, shell_weights=shell_weights)) for r in radii]
        return np.array([np.abs(maps[j + 1] - maps[j]).max() for j in range(len(radii) - 1)])

    hard, lin = steps("nearest"), steps("linear")
    assert hard.max() > 20 * np.median(hard), "the reference scan shows no hard-bin jump"
    assert lin.max() < 0.2 * hard.max()
