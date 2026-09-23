#!/usr/bin/env python
"""
What can kappa add on f_NL, given what the galaxies already measure? (AbacusSummit HUGE geometry)

The Fisher information Delta_F on (fNL, b1, bn2) that kappa adds on top of the galaxies is a
tomographic Limber Fisher restricted to kappa's own band (ell <= 64 = 2 nside) and to the modes the
survey contains (ell >= k_F chi_eff): Delta_F = F[galaxy shells + kappa] - F[galaxy shells]. It is
added to the galaxy Fisher measured from a galaxy-only chain, F_gxy = inv(Cov_MCMC[fNL, b1, bn2]),
which carries the 3-D information a two-point proxy cannot reproduce.

The Fisher gain per --density_scale is the reference curve of the density-scan figure
(docs/pipeline.md §7.3). sigma_kappa = sqrt(inv(Delta_F)[fNL, fNL]) is kappa's *incremental*
information on f_NL through the galaxies, not a kappa-only constraint (kappa alone sees f_NL only
through the weak matter phi^2 channel), which is why it grows as the galaxies thin out.

Usage:
    python scripts/fisher_kappa_gain.py                                   # Abacus reference, x1
    python scripts/fisher_kappa_gain.py --gxy_run run_<...> --density_scale 0.1
"""
import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path

import jax
import numpy as np
import yaml
from jax import numpy as jnp

from desi_cmb_fli.bricks import get_cosmology, lin_power_interp, trans_phi2delta_interp
from desi_cmb_fli.nbody import a2chi, a2f, a2g

jax.config.update("jax_enable_x64", True)

DELTA_C, RH = 1.686, 2997.92458
PARAMS = ["fNL", "b1", "bn2"]
NELL_FILE = Path(__file__).resolve().parents[1] / "data/N_L_kk_act_dr6_lensing_v1_baseline.txt"

# AbacusSummit HUGE c000_ph201: full-sky LRG lightcone and CMB-source kappa, baseline analysis
# geometry (box 7500, cell 93.75, centred observer, chi_min 350, matter in the map to chi 3942).
HUGE = {
    "zmin": 0.4, "zmax": 1.1, "fsky_g": 1.0, "fsky_k": 1.0, "N_gal": 22_239_543,
    "chi_min_kappa": 350.0, "chi_box": 3750.0, "chi_high_z_max": 3942.0,
    "zbins": 0.405 + (np.arange(15) + 0.5) * (1.095 - 0.405) / 15,
    "nbins": np.array([956218, 1082438, 1270883, 1452080, 1594423, 1723189, 1819472,
                       2007973, 2322955, 2214695, 1924098, 1556193, 1116871, 730871,
                       467184], float),
}

cosmo = get_cosmology(Omega_m=0.315192, sigma8=0.811355)
_ztab = np.linspace(0.0, 1200.0, 30000)
_chitab = np.asarray(a2chi(cosmo, 1.0 / (1.0 + _ztab)))
_Dtab = np.asarray(a2g(cosmo, 1.0 / (1.0 + _ztab)))
D0 = float(np.asarray(a2g(cosmo, np.array([1.0])))[0])
_ks = np.logspace(-4, 1, 512)
_P0 = np.asarray(lin_power_interp(cosmo, a=1.0)(jnp.asarray(_ks)))
_M0 = np.asarray(trans_phi2delta_interp(cosmo, a=1.0)(jnp.asarray(_ks)))


def chi_of_z(z):
    return np.interp(z, _ztab, _chitab)


def z_of_chi(c):
    return np.interp(c, _chitab, _ztab)


def D_of_z(z):
    return np.interp(z, _ztab, _Dtab)


def Plin(k, z):
    return np.interp(k, _ks, _P0) * (D_of_z(z) / D0) ** 2


def Mk(k, z):
    return np.interp(k, _ks, _M0) * (D_of_z(z) / D0)


def f_of_z(z):
    return np.asarray(a2f(cosmo, 1.0 / (1.0 + np.atleast_1d(z))))


NELL = np.loadtxt(NELL_FILE)
CHI_S = float(chi_of_z(1089.28))


def N_ell(ell, scaling):
    return scaling * np.interp(ell, NELL[:, 0], NELL[:, 1])


def W_kappa(chi):
    return 1.5 * cosmo.Omega_m / RH**2 * chi * (1 + z_of_chi(chi)) * np.clip(CHI_S - chi, 0, None) / CHI_S


def C_kappa(ells, lo, hi, n=800):
    chi = np.linspace(max(lo, 1.0), hi, n)
    z = z_of_chi(chi)
    k = (np.atleast_1d(ells)[:, None] + 0.5) / chi[None, :]
    P = np.interp(k, _ks, _P0) * ((D_of_z(z) / D0) ** 2)[None, :]
    return np.trapezoid(W_kappa(chi)[None, :] ** 2 / chi[None, :] ** 2 * P, chi, axis=1)


def chain_stats(run_dir, burn=0.5):
    """Covariance of (fNL, b1, bn2) from a galaxy-only run, and the b1 / bn2 means."""
    c = Path(run_dir) / "config"
    bs = sorted(c.glob("samples_batch_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
    lat = yaml.safe_load(open(c / "model.yaml"))["latents"]
    cols = []
    for p in PARAMS:
        x = np.concatenate([np.load(b)[p + "_"] for b in bs], axis=1)
        x = x * lat[p]["scale_fid"] + lat[p]["loc_fid"]
        cols.append(x[:, int(burn * x.shape[1]):].ravel())
    cols = np.array(cols)
    return np.cov(cols), float(cols[1].mean()), float(cols[2].mean())


def delta_F_kappa(g, b1, bn2, noise_scaling, density_scale, nz=10):
    """Fisher information on (fNL, b1, bn2) that kappa adds on top of the galaxy shells."""
    chi_lo, chi_hi = float(chi_of_z(g["zmin"])), float(chi_of_z(g["zmax"]))
    V = (4 * np.pi / 3) * g["fsky_g"] * (chi_hi**3 - chi_lo**3)
    chi_eff = 0.5 * (chi_lo + chi_hi)
    ell_min = max(2, int(np.ceil(2 * np.pi / V ** (1 / 3) * chi_eff)))
    bE, bphi = 1 + b1, 2 * DELTA_C * b1

    ells = np.arange(ell_min, 65)
    Ck_mod = C_kappa(ells, g["chi_min_kappa"], g["chi_box"])
    Ck_corr = C_kappa(ells, g["chi_box"], g["chi_high_z_max"])
    if g["chi_min_kappa"] > 0:
        Ck_corr = Ck_corr + C_kappa(ells, 1.0, g["chi_min_kappa"])

    chi = np.linspace(chi_lo, chi_hi, 400)
    z = z_of_chi(chi)
    nz_m = np.interp(z, g["zbins"], g["nbins"], left=0, right=0)
    dNdchi = nz_m / np.trapezoid(nz_m, chi)
    cdf = np.cumsum(dNdchi) / np.sum(dNdchi)
    edges = np.interp(np.linspace(0, 1, nz + 1), cdf, chi)
    Ng_sr = density_scale * g["N_gal"] / (4 * np.pi * g["fsky_g"])
    Wk = W_kappa(chi)
    fg = f_of_z(z)

    W, frac = [], []
    for i in range(nz):
        w = np.where((chi >= edges[i]) & (chi < edges[i + 1]), dNdchi, 0.0)
        frac.append(np.trapezoid(w, chi))
        W.append(w / max(frac[-1], 1e-30))
    W = np.array(W)

    Fg, Fj = np.zeros((3, 3)), np.zeros((3, 3))
    gi = np.arange(nz)
    for idx, ell in enumerate(ells):
        k = (ell + 0.5) / chi
        Pm = Plin(k, z)
        B = bE - bn2 * k**2 + fg / 3.0
        dB = {"fNL": bphi / Mk(k, z), "b1": np.ones_like(k), "bn2": -(k**2)}

        def ig(a, b, e, Pm=Pm):
            return np.trapezoid(a * b / chi**2 * Pm * e, chi)

        C = np.zeros((nz + 1, nz + 1))
        dC = {p: np.zeros((nz + 1, nz + 1)) for p in PARAMS}
        for i in range(nz):
            for j in range(nz):
                C[i, j] = ig(W[i], W[j], B**2)
                for p in PARAMS:
                    dC[p][i, j] = ig(W[i], W[j], 2 * B * dB[p])
            C[i, nz] = C[nz, i] = ig(W[i], Wk, B)
            for p in PARAMS:
                dC[p][i, nz] = dC[p][nz, i] = ig(W[i], Wk, dB[p])
        C[nz, nz] = Ck_mod[idx]

        Cg = C.copy()
        for i in range(nz):
            Cg[i, i] += 1.0 / (Ng_sr * frac[i])
        Cj = Cg.copy()
        Cj[nz, nz] += N_ell(ell, noise_scaling) + Ck_corr[idx]

        pre = (2 * ell + 1) / 2.0
        Cgi = np.linalg.inv(Cg[np.ix_(gi, gi)])
        Cji = np.linalg.inv(Cj)
        for a, pa in enumerate(PARAMS):
            for b, pb in enumerate(PARAMS):
                Fg[a, b] += pre * np.trace(Cgi @ dC[pa][np.ix_(gi, gi)] @ Cgi @ dC[pb][np.ix_(gi, gi)])
                Fj[a, b] += pre * np.trace(Cji @ dC[pa] @ Cji @ dC[pb])
    return (Fj - Fg) * g["fsky_k"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gxy_run", default=os.path.expandvars(
        "$SCRATCH/outputs/run_20260910_033019_58153868"),
        help="galaxy-only run whose chain gives F_gxy (and b1, bn2)")
    ap.add_argument("--density_scale", type=float, default=1.0,
                    help="multiplier on the catalogue n-bar (closure_gxy_density_scale)")
    ap.add_argument("--noise_scalings", type=float, nargs="+", default=[1.0, 0.4, 0.1, 0.01, 0.0])
    args = ap.parse_args()

    cov, b1, bn2 = chain_stats(args.gxy_run)
    F_g = np.linalg.inv(cov)
    s_g = np.sqrt(cov[0, 0])
    print(f"galaxy-only run: {args.gxy_run}")
    print(f"density scale {args.density_scale}, b1 = {b1:.4f}, bn2 = {bn2:.2f}")
    print(f"galaxy-only sigma(fNL) from the chain = {s_g:.3f}")
    print(f"{'N_l scaling':>12} {'sigma_kappa':>12} {'sigma_gxy/sigma_kappa':>22} "
          f"{'Fisher sigma_joint':>19} {'Fisher gain':>12} {'indep.-probe gain':>18}")
    for s in args.noise_scalings:
        dF = delta_F_kappa(HUGE, b1, bn2, s, args.density_scale)
        s_j = np.sqrt(np.linalg.inv(F_g + dF)[0, 0])
        s_k = np.sqrt(np.linalg.inv(dF)[0, 0]) if np.linalg.det(dF) > 0 else np.inf
        ratio = s_g / s_k
        line = 1 - (1 + ratio**2) ** -0.5
        print(f"{'x' + str(s):>12} {s_k:12.1f} {ratio:22.3f} {s_j:19.3f} "
              f"{100 * (1 - s_j / s_g):11.2f}% {100 * line:17.2f}%")


if __name__ == "__main__":
    main()
