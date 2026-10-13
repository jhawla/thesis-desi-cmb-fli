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
NELL_FILE = Path(__file__).resolve().parents[2] / "data/N_L_kk_so_v3_1_1_baseline_mv.txt"

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


def kappa_shells(kappa):
    """Radii, width and multipole taper of the model's Born shells over [kappa["chi_min"],
    kappa["chi_box"]], built as ``FieldLevelModel`` builds them: ``linear`` tents apex to apex,
    ``nearest`` bins tiling the range; taper rows on l = 0 ... kappa["lmax"] (None without a cut)."""
    from desi_cmb_fli.cmb_lensing import shell_ell_taper

    lo, hi, n = float(kappa["chi_min"]), float(kappa["chi_box"]), int(kappa["n_shells"])
    linear = kappa["shell_weights"] == "linear"
    dr = (hi - lo) / (n + (1 if linear else 0))
    half = dr if linear else dr / 2.0
    r = np.linspace(lo + half, hi - half, n)
    taper = None
    if kappa.get("k_cut"):
        taper = shell_ell_taper(r, dr, kappa["shell_weights"], kappa["k_cut"], ell_max=int(kappa["lmax"]),
                                lmax=int(kappa["lmax"]))
    return r, dr, taper


def held_fraction(bg, kappa, chi, ells):
    """Fraction of the map's lensing kernel the model's shells hold at ``chi`` for each of ``ells``,
    shape (n_ell, n_chi): sum_s w_s(l) q_s(chi) / W(chi) with q_s = d_r W(r_s) p_s(chi) chi^2 / V_s,
    the kernels of ``cmb_lensing.compute_cl_outside_model``; 1 - it is what the covariance holds."""
    from desi_cmb_fli.cmb_lensing import shell_profiles

    r, dr, taper = kappa_shells(kappa)
    chi_f, p_f = shell_profiles(r, dr, kappa["shell_weights"], n_per_shell=200)
    vol = np.trapezoid(p_f * chi_f**2, chi_f, axis=1)
    chi = np.asarray(chi, dtype=float)
    if kappa["shell_weights"] == "linear":
        p = np.clip(1.0 - np.abs(chi[None] - r[:, None]) / dr, 0.0, None)
    else:
        p = ((chi[None] >= (r - dr / 2)[:, None]) & (chi[None] < (r + dr / 2)[:, None])).astype(float)
    q = dr * bg.w_kappa(r)[:, None] * p * chi[None] ** 2 / vol[:, None]
    w = np.ones((r.size, np.size(ells))) if taper is None else taper[:, np.asarray(ells).astype(int)]
    return (w.T @ q) / np.maximum(bg.w_kappa(chi), 1e-300)[None]


def kappa_window(kappa, ells):
    """Angular window of the kappa observable at ``ells``: kappa["window"] (per ell from 0, the
    bilinear window of the projection sphere) if given, 1 otherwise."""
    if kappa.get("window") is None:
        return np.ones(np.shape(ells))
    return np.asarray(kappa["window"])[np.asarray(ells).astype(int)]


def kappa_spectra(bg, ells, kappa, n=1600):
    """Limber C_l^kk of what the model's shells hold (``held_fraction`` of the lensing kernel) and of
    the line-of-sight term the likelihood puts in its covariance (the rest of the kernel from
    kappa["chi_min"] to kappa["chi_high_z_max"]: the tents' end ramps, what the cut removes, the
    matter beyond the box), the latter times the exact-over-Limber ratio kappa["los_ratio"] (per ell
    from 0; 1 if absent), both times the squared window (``kappa_window``), as the model does."""
    ells = np.atleast_1d(ells)
    hi = max(float(kappa["chi_box"]), float(kappa["chi_high_z_max"]))
    chi = np.linspace(max(float(kappa["chi_min"]), 1.0), hi, n)
    base = bg.w_kappa(chi) ** 2 / chi**2
    z = bg.z_of_chi(chi)
    held = held_fraction(bg, kappa, chi, ells)
    kept, los = np.zeros(ells.size), np.zeros(ells.size)
    for i, ell in enumerate(ells):
        pk = base * bg.plin((ell + 0.5) / chi, z)
        kept[i], los[i] = np.trapezoid(pk * held[i] ** 2, chi), np.trapezoid(pk * (1 - held[i]) ** 2, chi)
    ratio = np.ones(ells.size) if kappa.get("los_ratio") is None else (
        np.asarray(kappa["los_ratio"])[ells.astype(int)])
    w2 = kappa_window(kappa, ells) ** 2
    return kept * w2, los * ratio * w2


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


def cube_mode_fraction(x, n_u=2001):
    """Fraction of the directions at |k| = x k_N inside the Fourier cube |k_i| <= k_N of a grid:
    1 up to x = 1, 0 from x = sqrt(3). The cell likelihood holds every mode of that cube, so the
    3-D Fisher counts the modes up to sqrt(3) k_N with this weight (docs/pipeline.md §6). Exact: with
    u = |n_z| uniform, the azimuth keeps |n_x|, |n_y| <= t = 1/x, a fraction 1 - (4/pi) arccos(t/s)
    of it for t < s <= sqrt(2) t (s = sqrt(1 - u^2)), all of it for s <= t and none beyond."""
    t = 1.0 / np.maximum(np.atleast_1d(np.asarray(x, dtype=float)), 1e-12)
    u = np.linspace(0.0, 1.0, n_u)[None, :]
    s = np.sqrt(1.0 - u**2)
    tt = t[:, None]
    ratio = np.clip(tt / np.maximum(s, 1e-300), 0.0, 1.0)
    g = np.where(s <= tt, 1.0, np.where(s <= np.sqrt(2.0) * tt, 1.0 - 4.0 / np.pi * np.arccos(ratio), 0.0))
    frac = np.trapezoid(np.where(u <= tt, g, 0.0), u, axis=1)
    return frac.reshape(np.shape(x))


def galaxy_fisher_3d(
    bg, survey, b1, bn2, kmax, kmin=None, density_scale=1.0, rsd=True, n_chi=160, n_k=240, n_mu=48
):
    """Fisher matrix on PARAMS from the galaxy power spectrum of a light-cone shell.

    F_ab = sum over the shell volume of  1 / (2 (2 pi)^2) int k^2 dk c(k / kmax) int_{-1}^{1} dmu
           dP/da dP/db / (P + 1/nbar)^2,
    P(k, mu, z) = (b(k, z) + f mu^2)^2 P_lin(k, z),  b = 1 + b1 - bn2 k^2 + b_phi fNL / M(k, z),
    b_phi = 2 delta_c b1, at the fiducial fNL = 0. ``kmax`` is the Nyquist frequency pi / cell of the
    grid: the modes are those of its Fourier cube, ``c = cube_mode_fraction`` up to sqrt(3) kmax.
    ``kmin`` defaults to 2 pi / V^(1/3).
    """
    chi_lo, chi_hi = float(bg.chi_of_z(survey["zmin"])), float(bg.chi_of_z(survey["zmax"]))
    V = 4 * np.pi / 3 * survey["fsky"] * (chi_hi**3 - chi_lo**3)
    kmin = 2 * np.pi / V ** (1 / 3) if kmin is None else kmin
    chi = np.linspace(chi_lo, chi_hi, n_chi)
    dV = 4 * np.pi * survey["fsky"] * chi**2 * np.gradient(chi)
    z = bg.z_of_chi(chi)
    nbar = nbar_of_chi(bg, chi, survey, density_scale)
    lnk = np.linspace(np.log(kmin), np.log(np.sqrt(3.0) * kmax), n_k)
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
        dV[:, None, None] * (k**3 * cube_mode_fraction(k / kmax))[None, :, None] * wmu[None, None, :]
        / (2 * (2 * np.pi) ** 2)
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
    held = held_fraction(bg, kappa, chi, ells)

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
            wkt = wk * held[idx] * kappa_window(kappa, ell)
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
    edge, high-z end, band, the shells ("n_shells", "shell_weights"), the per-shell cut "k_cut"
    (``shell_kmax``, default the init-grid Nyquist pi oversamp / cell when the line-of-sight
    correction is on), the "window" of the observable, the bilinear window of the projection sphere
    (nside x proj_oversamp), and "los_ratio" (``with_los_ratio``) at the config's truth cosmology."""
    from desi_cmb_fli.cmb_lensing import bilinear_window

    cmb = cfg.get("cmb_lensing", {}) or {}
    model = cfg["model"]
    box = np.asarray(model["box_shape"], dtype=float)
    k_cut = cmb.get("shell_kmax")
    if k_cut is None and cmb.get("full_los_correction", False):
        k_cut = np.pi * float(model.get("init_oversamp", 1.0)) / float(model["cell_size"])
    geometry = {
        "chi_min": float(cmb.get("chi_matter_min", 0.0)),
        "chi_box": float(box.min() / 2 if cmb.get("observer_mode", "face") == "center" else box[2]),
        "chi_high_z_max": float(cmb.get("chi_high_z_max", 3942.0)),
        "lmax": 2 * int(cmb.get("nside", 32)),
        "n_shells": int(cmb.get("n_shells", 189)),
        "shell_weights": str(cmb.get("shell_weights", "nearest")),
        "k_cut": float(k_cut or 0.0),
        "window": bilinear_window(int(cmb.get("nside", 32)) * int(cmb.get("proj_oversamp", 1)),
                                  2 * int(cmb.get("nside", 32))),
    }
    truth = {**(cfg.get("truth_params") or {}), **(cfg.get("abacus_truth_params") or {})}
    return with_los_ratio(geometry, float(truth.get("Omega_m", 0.315192)),
                          float(truth.get("sigma8", 0.811355)), float(cmb.get("z_source", 1089.28)))


def with_los_ratio(kappa, Omega_m, sigma8, z_source=1089.28):
    """``kappa`` with "los_ratio": the exact-over-Limber ratio of the line-of-sight kernel of its
    geometry (``cmb_lensing.limber_ratio_outside_model``), as the model computes it at its fiducial."""
    import jax
    import jax_cosmo as jc
    from jax import numpy as jnp

    from desi_cmb_fli.bricks import get_cosmology
    from desi_cmb_fli.cmb_lensing import limber_ratio_outside_model

    cosmo = get_cosmology(Omega_m=Omega_m, sigma8=sigma8)
    r, dr, taper = kappa_shells(kappa)
    a = np.asarray(jax.jit(lambda c: jc.background.a_of_chi(c, jnp.asarray(r)))(cosmo))
    ratio = limber_ratio_outside_model(cosmo, np.arange(int(kappa["lmax"]) + 1), r, a, dr,
                                       kappa["shell_weights"], z_source,
                                       max(float(kappa["chi_box"]), float(kappa["chi_high_z_max"])),
                                       taper=taper)
    return {**kappa, "los_ratio": ratio}


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

COSMO_PARAMS = ("Omega_m", "sigma8", "b1", "bn2", "bnpar")
FD_STEPS = {"Omega_m": 0.005, "sigma8": 0.005, "b1": 0.01, "bn2": 2.0, "fNL": 1.0, "b1_alpha": 0.05,
            "bnpar": 5.0}


def bias_reference_z(bg, survey):
    """Redshift where ``b1`` is defined when the bias evolves: the volume-weighted mean growth over
    the survey shell, as the loader's ``a_fid`` (growth-weighted mean over the survey cells)."""
    chi = np.linspace(bg.chi_of_z(survey["zmin"]), bg.chi_of_z(survey["zmax"]), 2000)
    D = bg.growth(bg.z_of_chi(chi))
    D_mean = np.sum(chi**2 * D) / np.sum(chi**2)
    return float(np.interp(D_mean, bg.growth(bg._z)[::-1], bg._z[::-1]))


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
    """Eulerian galaxy bias b(k, z) = (1 + b1) (D(z_b1)/D(z))^b1_alpha - bn2 k^2 + 2 delta_c b1 fNL / M(k, z),
    the growth at the cosmology of ``theta``, ``z_b1`` fixed (``bias_reference_z``)."""
    b = 1 + theta["b1"]
    if theta.get("b1_alpha", 0.0) != 0.0:
        b = b * (bg.growth(theta["z_b1"]) / bg.growth(z)) ** theta["b1_alpha"]
    b = b - theta.get("bn2", 0.0) * k**2
    if theta.get("fNL", 0.0) != 0.0:
        b = b + 2 * DELTA_C * theta["b1"] * theta["fNL"] / bg.m_phi(k, z)
    return b


def galaxy_fisher_3d_cosmo(
    bgs, fid, survey, kmax, params=COSMO_PARAMS, kmin=None, density_scale=1.0, rsd=True,
    steps=None, n_chi=160, n_k=240, n_mu=48,
):
    """3-D redshift-space galaxy Fisher on ``params`` by central differences, cosmology included,
    with the Finger-of-God term of the model: P = (b + f mu^2 + bnpar k^2 mu^2)^2 P_lin (the velocity
    ``bnpar D grad delta_L`` projected on the line of sight adds bnpar k_par^2 delta).

    As at the field level, the galaxies sit at their fiducial comoving distances (the data are
    painted once): volume, n(chi) and the k grid are those of the fiducial background, and a
    cosmology moves only the redshift z(chi) of each distance, hence the growth, the growth rate and
    the linear power there. No Alcock-Paczynski information. Same integral as ``galaxy_fisher_3d``,
    over the Fourier cube of the grid whose Nyquist frequency is ``kmax``.
    """
    steps = {**FD_STEPS, **(steps or {})}
    bg0 = bgs(fid)
    chi_lo, chi_hi = float(bg0.chi_of_z(survey["zmin"])), float(bg0.chi_of_z(survey["zmax"]))
    V = 4 * np.pi / 3 * survey["fsky"] * (chi_hi**3 - chi_lo**3)
    kmin = 2 * np.pi / V ** (1 / 3) if kmin is None else kmin
    chi = np.linspace(chi_lo, chi_hi, n_chi)
    dV = 4 * np.pi * survey["fsky"] * chi**2 * np.gradient(chi)
    nbar = nbar_of_chi(bg0, chi, survey, density_scale)
    lnk = np.linspace(np.log(kmin), np.log(np.sqrt(3.0) * kmax), n_k)
    k = np.exp(lnk)
    mu, wmu = np.polynomial.legendre.leggauss(n_mu)
    K, MU = k[None, :, None], mu[None, None, :]

    def power(theta):
        bg = bgs(theta)
        Z = bg.z_of_chi(chi)[:, None, None]
        rsd_mu2 = (bg.growth_rate(Z) + theta.get("bnpar", 0.0) * K**2) * MU**2 if rsd else 0.0
        return (_galaxy_bias(bg, theta, K, Z) + rsd_mu2) ** 2 * bg.plin(K, Z)

    P = power(fid)
    dP = _central_differences(power, fid, params, steps)
    n = nbar[:, None, None]
    inv_var = (n / (n * P + 1.0)) ** 2
    weight = (dV[:, None, None] * (k**3 * cube_mode_fraction(k / kmax))[None, :, None]
              * wmu[None, None, :] / (2 * (2 * np.pi) ** 2))
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
    the cut removes) either at the fiducial as noise (``los='noise'``, cl_out_mode fixed) or
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
        held = held_fraction(bg, kappa, chi, ells)
        ck = kappa_spectra(bg, ells, kappa)[0]
        if los == "signal":
            ck = ck + c_los(bg)
        out = np.zeros((len(ells), n_shells + 1, n_shells + 1))
        for n, ell in enumerate(ells):
            k = (ell + 0.5) / chi
            kern = _galaxy_bias(bg, theta, k, z) * W  # (n_shells, n_chi)
            pm = bg.plin(k, z) / chi**2
            out[n, :n_shells, :n_shells] = np.trapezoid(kern[:, None] * kern[None] * pm, chi, axis=2)
            wkt = wk * held[n] * kappa_window(kappa, ell)
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
