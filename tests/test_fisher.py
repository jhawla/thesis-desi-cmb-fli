"""Fisher forecasts of the density scan (desi_cmb_fli.fisher) against closed forms.

A toy background (constant power, no growth, no RSD, M(k) = c k^2) makes the 3-D galaxy Fisher
integrable by hand: each check is the textbook mode-counting result for one parameter.
"""

import numpy as np
import pytest
import yaml

from desi_cmb_fli import fisher as fi

P0, C_M = 2.0e4, 50.0


class ToyBackground:
    """chi = 1000 + 1000 z, D = 1, f = 0, P_lin = P0, M(k) = C_M k^2."""

    def __init__(self):
        self._z = np.linspace(0.0, 3.0, 3001)
        self._chi = 1000.0 + 1000.0 * self._z

    def chi_of_z(self, z):
        return 1000.0 + 1000.0 * np.asarray(z)

    def z_of_chi(self, chi):
        return (np.asarray(chi) - 1000.0) / 1000.0

    def growth_rate(self, z):
        return 0.0 * np.asarray(z)

    def plin(self, k, z):
        return P0 + 0.0 * k * z

    def m_phi(self, k, z):
        return C_M * k**2 + 0.0 * z

    def w_kappa(self, chi):
        return 1e-4 * np.ones_like(chi)

    def c_kappa(self, ells, lo, hi):
        return 1e-8 * (hi - lo) / 1000.0 * np.ones(np.size(ells))


def survey(n_total):
    return {
        "zmin": 0.405,
        "zmax": 1.095,
        "fsky": 1.0,
        "N_gal": n_total,  # the histogram edges
        "zbins": 0.405 + (np.arange(15) + 0.5) * 0.046,
        "nbins": np.full(15, n_total / 15.0),
    }


def shell_volume(s):
    lo, hi = 1000.0 * (1 + s["zmin"]), 1000.0 * (1 + s["zmax"])
    return 4 * np.pi / 3 * (hi**3 - lo**3)


KMIN, KMAX = 2e-3, 3e-2


def test_b1_information_is_the_mode_count():
    """No shot noise, no RSD: P = b^2 P0, so F_b1b1 = N_modes (2/b)^2 = V dk^3 / (3 pi^2 b^2)."""
    s, b1 = survey(1e14), 1.2
    F = fi.galaxy_fisher_3d(
        ToyBackground(), s, b1, 0.0, KMAX, kmin=KMIN, rsd=False, n_chi=400, n_k=2000
    )
    expected = shell_volume(s) * (KMAX**3 - KMIN**3) / (3 * np.pi**2 * (1 + b1) ** 2)
    np.testing.assert_allclose(F[1, 1], expected, rtol=2e-3)


def test_fnl_information_follows_the_inverse_square_response():
    """dP/dfNL = 2 b P0 b_phi / (C_M k^2): F_fNLfNL = V b_phi^2 / (pi^2 b^2 C_M^2) (1/kmin - 1/kmax)."""
    s, b1 = survey(1e14), 1.2
    F = fi.galaxy_fisher_3d(
        ToyBackground(), s, b1, 0.0, KMAX, kmin=KMIN, rsd=False, n_chi=400, n_k=4000
    )
    bphi, b = 2 * fi.DELTA_C * b1, 1 + b1
    expected = shell_volume(s) * bphi**2 / (np.pi**2 * b**2 * C_M**2) * (1 / KMIN - 1 / KMAX)
    np.testing.assert_allclose(F[0, 0], expected, rtol=2e-3)


def test_shot_noise_limited_information_scales_as_the_density_squared():
    s = survey(10.0)  # nbar P ~ 1e-5
    F1 = fi.galaxy_fisher_3d(ToyBackground(), s, 1.2, 50.0, KMAX, kmin=KMIN, density_scale=1.0)
    F2 = fi.galaxy_fisher_3d(ToyBackground(), s, 1.2, 50.0, KMAX, kmin=KMIN, density_scale=2.0)
    np.testing.assert_allclose(F2, 4 * F1, rtol=1e-4)


def test_the_density_is_the_catalogue_histogram_piecewise():
    """n(z) is flat over each bin up to its edges, not interpolated between bin centres."""
    bg, s = ToyBackground(), survey(1.5e7)
    z = np.array([0.406, 0.428, 0.45, 1.094])
    n = fi.nbar_of_chi(bg, bg.chi_of_z(z), s)
    expected = 1e6 / 0.046 / 1000.0 / (4 * np.pi * bg.chi_of_z(z) ** 2)
    np.testing.assert_allclose(n, expected, rtol=1e-6)
    assert fi.nbar_of_chi(bg, bg.chi_of_z(np.array([0.40, 1.10])), s).max() == 0.0


def test_galaxy_fisher_is_symmetric_positive_definite():
    F = fi.galaxy_fisher_3d(ToyBackground(), survey(2e7), 1.2, 78.0, KMAX, kmin=KMIN)
    np.testing.assert_allclose(F, F.T)
    assert np.all(np.linalg.eigvalsh(F) > 0)


def test_kappa_adds_nothing_when_its_noise_is_infinite():
    kappa = {
        "chi_min": 700.0,
        "chi_low_z_min": 292.6,
        "chi_box": 3750.0,
        "chi_high_z_max": 3942.0,
        "lmax": 16,
    }
    nell = np.column_stack([np.arange(2.0, 200.0), np.full(198, 1e-8)])
    kw = {"b1": 1.2, "bn2": 78.0, "n_shells": 4, "nell": nell}
    dF = fi.kappa_fisher_increment(ToyBackground(), survey(2e7), kappa, noise_scaling=1.0, **kw)
    dF_inf = fi.kappa_fisher_increment(
        ToyBackground(), survey(2e7), kappa, noise_scaling=1e12, **kw
    )
    np.testing.assert_allclose(dF, dF.T, atol=1e-12 * np.abs(dF).max())
    assert np.all(np.linalg.eigvalsh(dF) > -1e-10 * np.abs(dF).max())
    assert np.abs(dF).max() > 0
    assert np.abs(dF_inf).max() < 1e-6 * np.abs(dF).max()


def test_paired_ratio_recovers_a_known_width_ratio():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(4, 20000))
    j = 0.8 * g + 0.1 * rng.normal(size=g.shape)  # sigma ratio sqrt(0.64 + 0.01)
    j[:, :10000] += 50.0  # a burn-in the estimator must discard
    r, err, per_chain = fi.paired_sigma_ratio(g, j, burn_in=0.5)
    assert per_chain.shape == (4,)
    np.testing.assert_allclose(r, np.sqrt(0.65), rtol=0.01)
    assert err < 0.01


def test_scalar_chains_are_read_back_in_physical_units(tmp_path):
    """Sampled latents are (x - loc_fid) / scale_fid; n_batches keeps the first batches only."""
    cdir = tmp_path / "config"
    cdir.mkdir()
    lat = {p: {"loc_fid": 1.0 + i, "scale_fid": 2.0 + i} for i, p in enumerate(fi.PARAMS)}
    yaml.safe_dump({"latents": lat}, open(cdir / "model.yaml", "w"))
    for b in range(3):
        np.savez(
            cdir / f"samples_batch_{b}.npz",
            **{p + "_": np.full((4, 5), float(b)) for p in fi.PARAMS},
        )
    out = fi.load_scalar_chains(tmp_path, n_batches=2)
    assert fi.n_batches(tmp_path) == 3
    for p in fi.PARAMS:
        assert out[p].shape == (4, 10)
        np.testing.assert_allclose(out[p][:, 5:], 1.0 * lat[p]["scale_fid"] + lat[p]["loc_fid"])


def test_chain_fisher_is_the_inverse_covariance():
    rng = np.random.default_rng(1)
    cov = np.array([[4.0, 0.3, 1.0], [0.3, 0.5, 0.2], [1.0, 0.2, 9.0]])
    x = rng.multivariate_normal(np.zeros(3), cov, size=(4, 50000))
    chains = {p: x[..., i] for i, p in enumerate(fi.PARAMS)}
    np.testing.assert_allclose(np.linalg.inv(fi.chain_fisher(chains)), cov, rtol=0.03, atol=0.02)
    assert fi.sigma(fi.chain_fisher(chains)) == pytest.approx(2.0, rel=0.02)
