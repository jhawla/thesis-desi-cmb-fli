"""Fisher forecasts on (f_NL, b1, b_nabla2) for the tracer-density scan, and the measured paired
ratio they are compared with. Formulas and approximations: docs/pipeline.md §6 ("Fisher forecasts").

- ``galaxy_fisher_3d``: the galaxies alone, from the redshift-space power spectrum of a full-sky
  light-cone shell (Gaussian field: at linear order the field level carries the same information).
- ``kappa_fisher_increment``: what kappa adds to the galaxies, a tomographic Limber Fisher
  F[galaxy shells + kappa] - F[galaxy shells] over kappa's band.
- ``paired_sigma_ratio``: sigma_joint / sigma_gxy per chain of two runs sharing seed and warm start.
"""

from pathlib import Path

import numpy as np
import yaml

PARAMS = ("fNL", "b1", "bn2")
DELTA_C = 1.686
C_KMS_OVER_H100 = 2997.92458  # c / (100 km/s/Mpc), Mpc/h
NELL_FILE = Path(__file__).resolve().parents[2] / "data/N_L_kk_act_dr6_lensing_v1_baseline.txt"

# AbacusSummit HUGE c000_ph201 LRG light cone: redshift range, full sky, and the catalogue n(z)
# (15 bins of width 0.046 from z = 0.405). The kappa geometry is set from the run config.
HUGE_LRG = {
    "zmin": 0.4,
    "zmax": 1.1,
    "fsky": 1.0,
    "N_gal": 22_239_543,
    "zbins": 0.405 + (np.arange(15) + 0.5) * (1.095 - 0.405) / 15,
    "nbins": np.array(
        [
            956218,
            1082438,
            1270883,
            1452080,
            1594423,
            1723189,
            1819472,
            2007973,
            2322955,
            2214695,
            1924098,
            1556193,
            1116871,
            730871,
            467184,
        ],
        float,
    ),
}


class Background:
    """Distances, growth, growth rate, linear power and the phi -> delta transfer, tabulated once."""

    def __init__(self, Omega_m=0.315192, sigma8=0.811355, z_source=1089.28):
        import jax
        from jax import numpy as jnp

        from desi_cmb_fli.bricks import get_cosmology, lin_power_interp, trans_phi2delta_interp
        from desi_cmb_fli.nbody import a2chi, a2f, a2g

        jax.config.update("jax_enable_x64", True)
        self.cosmo = get_cosmology(Omega_m=Omega_m, sigma8=sigma8)
        self.Omega_m = float(Omega_m)
        self._z = np.linspace(0.0, 1200.0, 30000)
        a = 1.0 / (1.0 + self._z)
        self._chi = np.asarray(a2chi(self.cosmo, a))
        self._D = np.asarray(a2g(self.cosmo, a))
        self._f = np.asarray(a2f(self.cosmo, a))
        self.D0 = float(np.asarray(a2g(self.cosmo, np.array([1.0])))[0])
        self._k = np.logspace(-4, 1, 512)
        self._P0 = np.asarray(lin_power_interp(self.cosmo, a=1.0)(jnp.asarray(self._k)))
        self._M0 = np.asarray(trans_phi2delta_interp(self.cosmo, a=1.0)(jnp.asarray(self._k)))
        self.chi_s = float(self.chi_of_z(z_source))

    def chi_of_z(self, z):
        return np.interp(z, self._z, self._chi)

    def z_of_chi(self, chi):
        return np.interp(chi, self._chi, self._z)

    def growth(self, z):
        return np.interp(z, self._z, self._D) / self.D0

    def growth_rate(self, z):
        return np.interp(z, self._z, self._f)

    def plin(self, k, z):
        return np.interp(k, self._k, self._P0) * self.growth(z) ** 2

    def m_phi(self, k, z):
        """delta_lin(k, z) = M(k, z) phi(k); phi is primordial, so M carries the growth."""
        return np.interp(k, self._k, self._M0) * self.growth(z)

    def w_kappa(self, chi):
        z = self.z_of_chi(chi)
        return (
            1.5
            * self.Omega_m
            / C_KMS_OVER_H100**2
            * chi
            * (1 + z)
            * np.clip(self.chi_s - chi, 0, None)
            / self.chi_s
        )

    def c_kappa(self, ells, chi_lo, chi_hi, n=800):
        """Limber C_l^kk of the matter between chi_lo and chi_hi."""
        chi = np.linspace(max(chi_lo, 1.0), chi_hi, n)
        z = self.z_of_chi(chi)
        k = (np.atleast_1d(ells)[:, None] + 0.5) / chi[None, :]
        P = np.interp(k, self._k, self._P0) * (self.growth(z) ** 2)[None, :]
        return np.trapezoid(self.w_kappa(chi)[None, :] ** 2 / chi[None, :] ** 2 * P, chi, axis=1)


def shell_cut_taper(ell, chi, k_cut, lmax):
    """The per-shell multipole cut of the Born projector (``cmb_lensing.shell_kmax``) as a weight
    on the matter at distance chi for multipole ell: the cosine taper of ``shell_ell_taper`` with
    ell_res = k_cut chi, 1 where ell_res >= lmax or without a cut (``k_cut`` 0 or None)."""
    chi = np.asarray(chi, dtype=float)
    if not k_cut:
        return np.ones_like(chi)
    ell_res = k_cut * chi
    l_cut = np.floor(ell_res)
    l_width = np.maximum(l_cut // 4, 1)
    x = (ell - (l_cut - l_width)) / l_width
    t = np.where(ell <= l_cut - l_width, 1.0, np.where(ell >= l_cut, 0.0, 0.5 * (1 + np.cos(np.pi * x))))
    return np.where(ell_res >= lmax, 1.0, t)


def kappa_spectra(bg, ells, kappa, n=800):
    """Limber C_l^kk of the model's shells (kappa["chi_min"] to kappa["chi_box"], times the squared
    per-shell taper) and of the line of sight the likelihood puts in its covariance (beyond the box to
    kappa["chi_high_z_max"], plus the power the cut removes, as independent noise like the model)."""
    lo, hi, k_cut = kappa["chi_min"], kappa["chi_box"], kappa.get("k_cut")
    los = bg.c_kappa(ells, hi, kappa["chi_high_z_max"])
    if not k_cut:
        return bg.c_kappa(ells, lo, hi), los
    chi = np.linspace(max(lo, 1.0), hi, n)
    base = bg.w_kappa(chi) ** 2 / chi**2
    z = bg.z_of_chi(chi)
    kept, removed = np.zeros(np.size(ells)), np.zeros(np.size(ells))
    for i, ell in enumerate(np.atleast_1d(ells)):
        t = shell_cut_taper(ell, chi, k_cut, kappa["lmax"])
        p = base * bg.plin((ell + 0.5) / chi, z)
        kept[i], removed[i] = np.trapezoid(p * t**2, chi), np.trapezoid(p * (1 - t) ** 2, chi)
    return kept, los + removed


def nbar_of_chi(bg, chi, survey, density_scale=1.0):
    """Mean comoving number density (h/Mpc)^3 from the catalogue n(z) histogram, piecewise
    constant over its bins (``zbins`` are the bin centres), zero outside them."""
    z = bg.z_of_chi(chi)
    dz = survey["zbins"][1] - survey["zbins"][0]
    i = np.floor((z - (survey["zbins"][0] - dz / 2)) / dz).astype(int)
    inside = (i >= 0) & (i < len(survey["nbins"]))
    dNdz = np.where(inside, survey["nbins"][np.clip(i, 0, len(survey["nbins"]) - 1)] / dz, 0.0)
    dchidz = np.gradient(bg._chi, bg._z)
    dNdchi = dNdz / np.interp(z, bg._z, dchidz)
    return density_scale * dNdchi / (4 * np.pi * survey["fsky"] * chi**2)


def galaxy_fisher_3d(
    bg, survey, b1, bn2, kmax, kmin=None, density_scale=1.0, rsd=True, n_chi=160, n_k=240, n_mu=48
):
    """Fisher matrix on PARAMS from the galaxy power spectrum of a light-cone shell.

    F_ab = sum over the shell volume of  1 / (2 (2 pi)^2) int k^2 dk int_{-1}^{1} dmu
           dP/da dP/db / (P + 1/nbar)^2,
    P(k, mu, z) = (b(k, z) + f mu^2)^2 P_lin(k, z),  b = 1 + b1 - bn2 k^2 + b_phi fNL / M(k, z),
    b_phi = 2 delta_c b1, at the fiducial fNL = 0. ``kmin`` defaults to 2 pi / V^(1/3).
    """
    chi_lo, chi_hi = float(bg.chi_of_z(survey["zmin"])), float(bg.chi_of_z(survey["zmax"]))
    V = 4 * np.pi / 3 * survey["fsky"] * (chi_hi**3 - chi_lo**3)
    kmin = 2 * np.pi / V ** (1 / 3) if kmin is None else kmin
    chi = np.linspace(chi_lo, chi_hi, n_chi)
    dV = 4 * np.pi * survey["fsky"] * chi**2 * np.gradient(chi)
    z = bg.z_of_chi(chi)
    nbar = nbar_of_chi(bg, chi, survey, density_scale)
    lnk = np.linspace(np.log(kmin), np.log(kmax), n_k)
    k = np.exp(lnk)
    mu, wmu = np.polynomial.legendre.leggauss(n_mu)  # on [-1, 1]

    K, Z, MU = k[None, :, None], z[:, None, None], mu[None, None, :]
    f = bg.growth_rate(Z) if rsd else 0.0
    b = 1 + b1 - bn2 * K**2
    amp = b + f * MU**2
    Pl = bg.plin(K, Z)
    P = amp**2 * Pl
    dP = {
        "fNL": 2 * amp * Pl * (2 * DELTA_C * b1) / bg.m_phi(K, Z),
        "b1": 2 * amp * Pl,
        "bn2": 2 * amp * Pl * (-(K**2)),
    }
    n = nbar[:, None, None]
    inv_var = (n / (n * P + 1.0)) ** 2  # 1 / (P + 1/nbar)^2, finite where nbar = 0
    weight = (
        dV[:, None, None] * (k**3)[None, :, None] * wmu[None, None, :] / (2 * (2 * np.pi) ** 2)
    )  # k^2 dk = k^3 dlnk
    F = np.zeros((len(PARAMS), len(PARAMS)))
    for i, a in enumerate(PARAMS):
        for j, c in enumerate(PARAMS[: i + 1]):
            integrand = weight * dP[a] * dP[c] * inv_var
            F[i, j] = F[j, i] = np.trapezoid(integrand.sum(axis=(0, 2)), lnk)
    return F


def kappa_fisher_increment(
    bg, survey, kappa, b1, bn2, noise_scaling=1.0, density_scale=1.0, n_shells=10, nell=None
):
    """Information kappa adds on PARAMS on top of the galaxies, tomographic Limber.

    Delta_F = F[galaxy shells + kappa] - F[galaxy shells], sum over ell_min <= ell <= kappa["lmax"]
    of (2 ell + 1)/2 tr(C^-1 dC_a C^-1 dC_b). Galaxy shells: equal-number bins of the catalogue
    n(z), bias b_E - bn2 k^2 + b_phi fNL / M (no Kaiser term: broad shells). Kappa: the matter
    between kappa["chi_min"] and kappa["chi_box"] under the per-shell cut (``kappa_spectra``), plus
    as noise N_l x noise_scaling and the line of sight. ell_min is the survey's fundamental mode,
    k_F chi_eff.
    """
    if nell is None:
        nell = np.loadtxt(NELL_FILE)
    chi_lo, chi_hi = float(bg.chi_of_z(survey["zmin"])), float(bg.chi_of_z(survey["zmax"]))
    V = (4 * np.pi / 3) * survey["fsky"] * (chi_hi**3 - chi_lo**3)
    chi_eff = 0.5 * (chi_lo + chi_hi)
    ell_min = max(2, int(np.ceil(2 * np.pi / V ** (1 / 3) * chi_eff)))
    bE, bphi = 1 + b1, 2 * DELTA_C * b1

    ells = np.arange(ell_min, int(kappa["lmax"]) + 1)
    ck_mod, ck_los = kappa_spectra(bg, ells, kappa)

    chi = np.linspace(chi_lo, chi_hi, 400)
    z = bg.z_of_chi(chi)
    nz_m = np.interp(z, survey["zbins"], survey["nbins"], left=0, right=0)
    dNdchi = nz_m / np.trapezoid(nz_m, chi)
    cdf = np.cumsum(dNdchi) / np.sum(dNdchi)
    edges = np.interp(np.linspace(0, 1, n_shells + 1), cdf, chi)
    ng_sr = density_scale * survey["N_gal"] / (4 * np.pi * survey["fsky"])
    wk = bg.w_kappa(chi)

    W, frac = [], []
    for i in range(n_shells):
        w = np.where((chi >= edges[i]) & (chi < edges[i + 1]), dNdchi, 0.0)
        frac.append(np.trapezoid(w, chi))
        W.append(w / max(frac[-1], 1e-30))
    W = np.array(W)

    npar, ns = len(PARAMS), n_shells
    Fg, Fj = np.zeros((npar, npar)), np.zeros((npar, npar))
    gi = np.arange(ns)
    for idx, ell in enumerate(ells):
        k = (ell + 0.5) / chi
        Pm = bg.plin(k, z)
        B = bE - bn2 * k**2
        dB = {"fNL": bphi / bg.m_phi(k, z), "b1": np.ones_like(k), "bn2": -(k**2)}

        def ig(a, b, e, Pm=Pm):
            return np.trapezoid(a * b / chi**2 * Pm * e, chi)

        C = np.zeros((ns + 1, ns + 1))
        dC = {p: np.zeros((ns + 1, ns + 1)) for p in PARAMS}
        for i in range(ns):
            for j in range(ns):
                C[i, j] = ig(W[i], W[j], B**2)
                for p in PARAMS:
                    dC[p][i, j] = ig(W[i], W[j], 2 * B * dB[p])
            wkt = wk * shell_cut_taper(ell, chi, kappa.get("k_cut"), kappa["lmax"])
            C[i, ns] = C[ns, i] = ig(W[i], wkt, B)
            for p in PARAMS:
                dC[p][i, ns] = dC[p][ns, i] = ig(W[i], wkt, dB[p])
        C[ns, ns] = ck_mod[idx]

        Cg = C.copy()
        for i in range(ns):
            Cg[i, i] += 1.0 / (ng_sr * frac[i])
        Cj = Cg.copy()
        Cj[ns, ns] += noise_scaling * np.interp(ell, nell[:, 0], nell[:, 1]) + ck_los[idx]

        pre = (2 * ell + 1) / 2.0
        Cgi = np.linalg.inv(Cg[np.ix_(gi, gi)])
        Cji = np.linalg.inv(Cj)
        for a, pa in enumerate(PARAMS):
            for b, pb in enumerate(PARAMS):
                Fg[a, b] += pre * np.trace(
                    Cgi @ dC[pa][np.ix_(gi, gi)] @ Cgi @ dC[pb][np.ix_(gi, gi)]
                )
                Fj[a, b] += pre * np.trace(Cji @ dC[pa] @ Cji @ dC[pb])
    return (Fj - Fg) * survey["fsky"]


def kappa_geometry(cfg):
    """Kappa geometry of a run config: "chi_min" where the shells start (``chi_matter_min``), box
    edge, high-z end, band, and the per-shell cut "k_cut" (``shell_kmax``, default the init-grid
    Nyquist pi oversamp / cell when the line-of-sight correction is on)."""
    cmb = cfg.get("cmb_lensing", {}) or {}
    model = cfg["model"]
    box = np.asarray(model["box_shape"], dtype=float)
    k_cut = cmb.get("shell_kmax")
    if k_cut is None and cmb.get("full_los_correction", False):
        k_cut = np.pi * float(model.get("init_oversamp", 1.0)) / float(model["cell_size"])
    return {
        "chi_min": float(cmb.get("chi_matter_min", 0.0)),
        "chi_box": float(box.min() / 2 if cmb.get("observer_mode", "face") == "center" else box[2]),
        "chi_high_z_max": float(cmb.get("chi_high_z_max", 3942.0)),
        "lmax": 2 * int(cmb.get("nside", 32)),
        "k_cut": float(k_cut or 0.0),
    }


def sigma(F, param="fNL"):
    """Marginal width of ``param`` from a Fisher matrix."""
    return float(np.sqrt(np.linalg.inv(F)[PARAMS.index(param), PARAMS.index(param)]))


# ---------------------------------------------------------------------------------------------
# Measured chains
# ---------------------------------------------------------------------------------------------


def load_scalar_chains(run_dir, params=PARAMS, n_batches=None):
    """Physical scalar samples (n_chains, n_samples) of a run, from its batch files.

    The sampled latents are ``(x - loc_fid) / scale_fid`` (model.yaml); ``n_batches`` keeps the
    first batches only, to match two runs of different length.
    """
    c = Path(run_dir) / "config"
    batches = sorted(c.glob("samples_batch_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
    if n_batches is not None:
        batches = batches[:n_batches]
    if not batches:
        raise FileNotFoundError(f"no samples_batch_*.npz in {c}")
    lat = yaml.safe_load(open(c / "model.yaml"))["latents"]
    out = {}
    for p in params:
        x = np.concatenate([np.load(b)[p + "_"] for b in batches], axis=1)
        out[p] = x * lat[p]["scale_fid"] + lat[p]["loc_fid"]
    return out


def n_batches(run_dir):
    return len(list((Path(run_dir) / "config").glob("samples_batch_*.npz")))


def paired_sigma_ratio(gxy_chains, joint_chains, burn_in=0.5):
    """sigma_joint / sigma_gxy per chain (second part of each chain), mean and error of the mean.

    The two runs share their seed and warm start, so chain i of one is paired with chain i of the
    other; the error is the spread of the per-chain ratios over sqrt(n_chains).
    """
    g, j = np.asarray(gxy_chains), np.asarray(joint_chains)
    g = g[:, int(burn_in * g.shape[1]) :]
    j = j[:, int(burn_in * j.shape[1]) :]
    r = j.std(axis=1) / g.std(axis=1)
    return float(r.mean()), float(r.std(ddof=1) / np.sqrt(len(r))), r


def chain_fisher(chains, burn_in=0.5):
    """Galaxy Fisher measured from a chain: inv(Cov[PARAMS]) over all chains, second part."""
    cols = [
        np.asarray(chains[p])[:, int(burn_in * np.shape(chains[p])[1]) :].ravel() for p in PARAMS
    ]
    return np.linalg.inv(np.cov(np.array(cols)))


# ---------------------------------------------------------------------------------------------
# Cosmology: (Omega_m, sigma8) with the biases, fNL optional
# ---------------------------------------------------------------------------------------------

COSMO_PARAMS = ("Omega_m", "sigma8", "b1", "bn2")
FD_STEPS = {"Omega_m": 0.005, "sigma8": 0.005, "b1": 0.01, "bn2": 2.0, "fNL": 1.0}


class BackgroundCache:
    """One background per (Omega_m, sigma8), built on first use; ``factory(Omega_m, sigma8)``."""

    def __init__(self, factory=None, z_source=1089.28):
        self.factory = factory or (lambda om, s8: Background(om, s8, z_source))
        self._cache = {}

    def __call__(self, theta):
        key = (round(float(theta["Omega_m"]), 10), round(float(theta["sigma8"]), 10))
        if key not in self._cache:
            self._cache[key] = self.factory(*key)
        return self._cache[key]


def _central_differences(fn, fid, params, steps):
    out = {}
    for p in params:
        h = steps[p]
        up, dn = dict(fid), dict(fid)
        up[p] += h
        dn[p] -= h
        out[p] = (fn(up) - fn(dn)) / (2 * h)
    return out


def _galaxy_bias(bg, theta, k, z):
    """Eulerian galaxy bias b(k, z) = 1 + b1 - bn2 k^2 + 2 delta_c b1 fNL / M(k, z)."""
    b = 1 + theta["b1"] - theta.get("bn2", 0.0) * k**2
    if theta.get("fNL", 0.0) != 0.0:
        b = b + 2 * DELTA_C * theta["b1"] * theta["fNL"] / bg.m_phi(k, z)
    return b


def galaxy_fisher_3d_cosmo(
    bgs, fid, survey, kmax, params=COSMO_PARAMS, kmin=None, density_scale=1.0, rsd=True,
    steps=None, n_chi=160, n_k=240, n_mu=48,
):
    """3-D redshift-space galaxy Fisher on ``params`` by central differences, cosmology included.

    As at the field level, the galaxies sit at their fiducial comoving distances (the data are
    painted once): volume, n(chi) and the k grid are those of the fiducial background, and a
    cosmology moves only the redshift z(chi) of each distance, hence the growth, the growth rate and
    the linear power there. No Alcock-Paczynski information. Same integral as ``galaxy_fisher_3d``.
    """
    steps = {**FD_STEPS, **(steps or {})}
    bg0 = bgs(fid)
    chi_lo, chi_hi = float(bg0.chi_of_z(survey["zmin"])), float(bg0.chi_of_z(survey["zmax"]))
    V = 4 * np.pi / 3 * survey["fsky"] * (chi_hi**3 - chi_lo**3)
    kmin = 2 * np.pi / V ** (1 / 3) if kmin is None else kmin
    chi = np.linspace(chi_lo, chi_hi, n_chi)
    dV = 4 * np.pi * survey["fsky"] * chi**2 * np.gradient(chi)
    nbar = nbar_of_chi(bg0, chi, survey, density_scale)
    lnk = np.linspace(np.log(kmin), np.log(kmax), n_k)
    k = np.exp(lnk)
    mu, wmu = np.polynomial.legendre.leggauss(n_mu)
    K, MU = k[None, :, None], mu[None, None, :]

    def power(theta):
        bg = bgs(theta)
        Z = bg.z_of_chi(chi)[:, None, None]
        f = bg.growth_rate(Z) if rsd else 0.0
        return (_galaxy_bias(bg, theta, K, Z) + f * MU**2) ** 2 * bg.plin(K, Z)

    P = power(fid)
    dP = _central_differences(power, fid, params, steps)
    n = nbar[:, None, None]
    inv_var = (n / (n * P + 1.0)) ** 2
    weight = dV[:, None, None] * (k**3)[None, :, None] * wmu[None, None, :] / (2 * (2 * np.pi) ** 2)
    F = np.zeros((len(params), len(params)))
    for i, a in enumerate(params):
        for j, c in enumerate(params[: i + 1]):
            F[i, j] = F[j, i] = np.trapezoid((weight * dP[a] * dP[c] * inv_var).sum(axis=(0, 2)), lnk)
    return F


def angular_fisher_cosmo(
    bgs, fid, survey, kappa, probes="gk", params=COSMO_PARAMS, density_scale=1.0,
    noise_scaling=1.0, nell=None, los="noise", n_shells=10, ell_min=2, steps=None, n_chi=400,
    mismatch=None,
):
    """Tomographic Limber Fisher of galaxy shells ('g'), kappa ('k') or both ('gk') on ``params``.

    Galaxy shells: equal-number bins of the catalogue n(z) at fiducial distances, bias
    ``_galaxy_bias`` (no Kaiser term, as in ``kappa_fisher_increment``). Kappa signal: the matter
    between kappa["chi_min"] and kappa["chi_box"] at fixed comoving distance, a cosmology moving
    z(chi), the growth, Omega_m and chi_s in the kernel, under the per-shell cut (``kappa_spectra``).
    Kappa noise: N_l x noise_scaling, plus the line of sight (chi_box -> chi_high_z_max and the power
    the cut removes) either at the fiducial as noise (``los='noise'``, high_z_mode fixed) or
    following the cosmology as signal ('signal').
    Sum over ell_min <= ell <= kappa["lmax"] of (2 ell + 1)/2 tr(C^-1 dC_a C^-1 dC_b), times fsky.

    With ``mismatch``, the kappa-kappa power the data hold beyond the model on ``0 ... lmax`` (one
    value per ell), also returns ``b_a = fsky sum (2 ell + 1)/2 tr(C^-1 dC_a C^-1 Delta C)``: the
    first-order shift of the maximum-likelihood parameters is ``F_total^-1 b``.
    """
    if nell is None:
        nell = np.loadtxt(NELL_FILE)
    steps = {**FD_STEPS, **(steps or {})}
    bg0 = bgs(fid)
    chi_lo, chi_hi = float(bg0.chi_of_z(survey["zmin"])), float(bg0.chi_of_z(survey["zmax"]))
    chi = np.linspace(chi_lo, chi_hi, n_chi)
    nz = np.interp(bg0.z_of_chi(chi), survey["zbins"], survey["nbins"], left=0, right=0)
    dNdchi = nz / np.trapezoid(nz, chi)
    cdf = np.cumsum(dNdchi) / np.sum(dNdchi)
    edges = np.interp(np.linspace(0, 1, n_shells + 1), cdf, chi)
    W, frac = [], []
    for i in range(n_shells):
        w = np.where((chi >= edges[i]) & (chi < edges[i + 1]), dNdchi, 0.0)
        frac.append(np.trapezoid(w, chi))
        W.append(w / max(frac[-1], 1e-30))
    W, frac = np.array(W), np.array(frac)
    ells = np.arange(ell_min, int(kappa["lmax"]) + 1)

    def c_los(bg):
        return kappa_spectra(bg, ells, kappa)[1]

    def spectra(theta):
        """Signal matrices (n_ell, n_shells + 1, n_shells + 1), kappa last."""
        bg = bgs(theta)
        z = bg.z_of_chi(chi)
        wk = bg.w_kappa(chi)
        ck = kappa_spectra(bg, ells, kappa)[0]
        if los == "signal":
            ck = ck + c_los(bg)
        out = np.zeros((len(ells), n_shells + 1, n_shells + 1))
        for n, ell in enumerate(ells):
            k = (ell + 0.5) / chi
            kern = _galaxy_bias(bg, theta, k, z) * W  # (n_shells, n_chi)
            pm = bg.plin(k, z) / chi**2
            out[n, :n_shells, :n_shells] = np.trapezoid(kern[:, None] * kern[None] * pm, chi, axis=2)
            wkt = wk * shell_cut_taper(ell, chi, kappa.get("k_cut"), kappa["lmax"])
            out[n, :n_shells, n_shells] = out[n, n_shells, :n_shells] = np.trapezoid(
                kern * wkt * pm, chi, axis=1
            )
            out[n, n_shells, n_shells] = ck[n]
        return out

    C = spectra(fid)
    dC = _central_differences(spectra, fid, params, steps)
    noise = np.zeros(n_shells + 1)
    noise[:n_shells] = 1.0 / (density_scale * survey["N_gal"] / (4 * np.pi * survey["fsky"]) * frac)
    kappa_noise = noise_scaling * np.interp(ells, nell[:, 0], nell[:, 1])
    if los == "noise":
        kappa_noise = kappa_noise + c_los(bg0)
    idx = {"g": list(range(n_shells)), "k": [n_shells], "gk": list(range(n_shells + 1))}[probes]
    F, B = np.zeros((len(params), len(params))), np.zeros(len(params))
    for n, ell in enumerate(ells):
        Cn = C[n] + np.diag(noise)
        Cn[n_shells, n_shells] += kappa_noise[n]
        Ci = np.linalg.inv(Cn[np.ix_(idx, idx)])
        D = [Ci @ dC[p][n][np.ix_(idx, idx)] for p in params]
        for a in range(len(params)):
            for b in range(a + 1):
                F[a, b] = F[b, a] = F[a, b] + (2 * ell + 1) / 2 * np.trace(D[a] @ D[b])
        if mismatch is not None and n_shells in idx:
            dmis = np.zeros((n_shells + 1, n_shells + 1))
            dmis[n_shells, n_shells] = mismatch[int(ell)]
            Dm = Ci @ dmis[np.ix_(idx, idx)]
            B += [(2 * ell + 1) / 2 * np.trace(D[a] @ Dm) for a in range(len(params))]
    if mismatch is None:
        return F * survey["fsky"]
    return F * survey["fsky"], B * survey["fsky"]


def prior_fisher(latents, params):
    """Gaussian prior of the run config's latents ('scale') on ``params``, as a Fisher matrix."""
    return np.diag([1.0 / float(latents[p]["scale"]) ** 2 for p in params])


def marginal(F, params, name):
    return float(np.sqrt(np.linalg.inv(F)[params.index(name), params.index(name)]))
