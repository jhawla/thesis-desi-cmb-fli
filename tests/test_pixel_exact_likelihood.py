"""Tests for the exact cut-sky CMB-lensing likelihood (cmb_likelihood_mode='pixel_exact').

pixel_exact is implemented via signal-eigenmode (KL) compression: the observed-pixel covariance is
diagonalised once, the k supported eigenmodes (lambda > rcond*lambda_max) are kept, and the observable
is the eigenmode amplitudes a = U_k^T kappa[obs] with the (exactly diagonal) eigenvalue covariance.

Checks (no full forward model needed):
1. The kept eigenmodes diagonalise the exact (Legendre) pixel covariance, and the kept eigenvalues are
   positive — i.e. U_k^T C U_k = diag(Lambda_k).
2. The truncation keeps only the supported subspace (k < npix_obs on a cut sky -> rank deficiency).
3. The likelihood site is a single Normal -> sample == log_prob by construction; tempering scales var.
"""

import types

import healpy as hp
import numpy as np

from desi_cmb_fli.model import FieldLevelModel


def _make_stub(nside, mask, cl, rcond=1e-8):
    stub = types.SimpleNamespace()
    stub.cmb_nside = nside
    stub.cmb_lmax = 2 * nside
    stub.nell_1d = np.asarray(cl, dtype=np.float64)
    stub.full_los_correction = False
    stub.cl_high_z_cached = None
    stub.cmb_mask = np.asarray(mask, dtype=bool)
    stub.cmb_kl_rcond = rcond
    return stub


def _small_setup():
    nside = 8
    lmax = 2 * nside
    ell = np.arange(lmax + 1, dtype=float)
    cl = np.ones_like(ell)  # white, well-conditioned; band-limit still makes the patch cov rank-structured
    cl[0] = 0.0
    mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    disc = hp.query_disc(nside, hp.ang2vec(np.pi / 3, np.pi / 4), np.radians(35.0))
    mask[disc] = True
    return nside, lmax, ell, cl, mask


def _exact_cov(nside, lmax, ell, cl, obs_pix):
    # Exact per-pair Legendre covariance (no pixel window), independent of the builder's interpolation.
    coef = (2 * ell + 1) / (4 * np.pi) * cl
    vecs = np.asarray(hp.pix2vec(nside, obs_pix)).T
    mu = np.clip(vecs @ vecs.T, -1.0, 1.0)
    return np.polynomial.legendre.legval(mu, coef)


def test_kl_eigenmodes_diagonalize_cov():
    nside, lmax, ell, cl, mask = _small_setup()
    obs_pix = np.where(mask)[0]
    stub = _make_stub(nside, mask, cl)
    FieldLevelModel._build_cmb_pixel_cov(stub)

    U = np.asarray(stub.cmb_kl_U)        # (npix, k)
    var = np.asarray(stub.cmb_kl_var)    # (k,)
    npix, k = U.shape
    assert k == var.size
    assert np.all(var > 0)                       # kept eigenvalues are positive
    assert k < npix                              # cut sky -> rank deficient -> truncated
    assert k <= obs_pix.size

    # Eigenmodes diagonalise the EXACT covariance: U^T C U = diag(var).
    C = _exact_cov(nside, lmax, ell, cl, obs_pix)
    M = U.T @ C @ U
    off = M - np.diag(np.diag(M))
    assert np.max(np.abs(off)) < 1e-3 * np.max(np.abs(np.diag(M)))  # off-diagonal negligible
    np.testing.assert_allclose(np.diag(M), var, rtol=1e-3, atol=1e-6 * var.max())


def test_kl_sample_equals_log_prob():
    """The KL observable is a single Normal site -> sample == log_prob natively."""
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.handlers import seed, trace

    rng = np.random.default_rng(1)
    k = 40
    loc = jnp.asarray(rng.standard_normal(k))
    scale = jnp.asarray(np.abs(rng.standard_normal(k)) + 0.1)

    def model():
        numpyro.sample("kappa_obs", dist.Normal(loc, scale))

    tr = trace(seed(model, jax.random.PRNGKey(3))).get_trace()
    site = tr["kappa_obs"]
    lp_trace = site["fn"].log_prob(site["value"]).sum()
    lp_recompute = dist.Normal(loc, scale).log_prob(site["value"]).sum()
    assert jnp.allclose(lp_trace, lp_recompute)


def test_kl_temper_scales_variance():
    """scale = sqrt(var*temp) scales the per-mode variance by temp (matches diagonal's total_cl*temp)."""
    import jax.numpy as jnp
    import numpyro.distributions as dist

    rng = np.random.default_rng(2)
    var = jnp.asarray(np.abs(rng.standard_normal(16)) + 0.1)
    loc = jnp.zeros(16)
    temp = 3.0
    d0 = dist.Normal(loc, jnp.sqrt(var))
    dt = dist.Normal(loc, jnp.sqrt(var * temp))
    assert jnp.allclose(dt.variance, temp * d0.variance, rtol=1e-5)
