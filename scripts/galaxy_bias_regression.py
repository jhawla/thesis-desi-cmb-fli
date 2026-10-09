#!/usr/bin/env python
"""Linear galaxy bias against the true initial conditions, per k band and radial shell (pipeline §7.14).

Regression of the counts of a run's `truth.npz` on its true linear field (the `init_mesh` of the
Abacus ICs, cropped to the final grid), linear Kaiser with the radial line of sight:

    obs = n̄S [1 + Σ_b B_b D δ_b + F_b f D (∂²_r + (2/r) ∂_r) ∇⁻² δ_b]

by weighted least squares (weights n̄S) over the occupied cells, `D(χ)`, `f(χ)` at the fiducial
`Omega_m`; B is the linear Eulerian bias, F the RSD amplitude relative to the fiducial f.

    python scripts/galaxy_bias_regression.py --runs "Abacus LRG=RUN_DIR" closure=RUN_DIR2 \\
        [--profiles "Abacus LRG=PROFILE_ALPHA0.json,PROFILE_ALPHA.json" ...]

Figure `figures/galaxy_bias_evolution/bias_evolution_true_ic.png` and its `.json`: B averaged over
0.01 < k < 0.034 in six equal shells of the LRG range against distance, with the 1/D(χ) shape, and,
with `--profiles` (outputs of `galaxy_likelihood_profile.py --mode omega_m`, with and without
`--alpha0`), χ² − χ²(fiducial) against `Omega_m`.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.integrate import quad  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures" / "galaxy_bias_evolution"
OM_FID = 0.315192
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def growth_tables(om):
    """χ(a), D(a) (= 1 today), f(a) of flat ΛCDM, tabulated in increasing χ."""
    a = np.linspace(0.2, 1.0, 801)
    E = lambda x: np.sqrt(om / x**3 + 1 - om)  # noqa: E731
    D = E(a) * np.array([quad(lambda x: 1 / (x * E(x)) ** 3, 0, ai)[0] for ai in a])
    D /= D[-1]
    f = np.gradient(np.log(D), np.log(a))
    chi = np.array([quad(lambda zz: 2997.92458 / E(1 / (1 + zz)), 0, 1 / ai - 1)[0] for ai in a])
    return chi[::-1], D[::-1], f[::-1]


def crop_rfft(dk, n_out):
    """Crop an rfftn mesh to n_out³ (dropping the output Nyquist plane), amplitude per cell kept."""
    n_in, h = dk.shape[0], n_out // 2
    idx = np.r_[0:h, n_in - h + 1 : n_in]
    out = np.zeros((n_out, n_out, h + 1), complex)
    keep = np.r_[0:h, n_out - h + 1 : n_out]
    out[np.ix_(keep, keep, np.arange(h))] = dk[np.ix_(idx, idx, np.arange(h))]
    return out * (n_out / n_in) ** 3


def regression(
    run_dir,
    shells,
    om=OM_FID,
    edges=(0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.0335, 0.042, 0.06),
):
    cfg = yaml.safe_load(open(Path(run_dir) / "config" / "config.yaml"))
    t = np.load(Path(run_dir) / "config" / "truth.npz")
    obs, S, mask, nbar = (
        t["obs"],
        t["selec_mesh"].astype(float),
        t["gxy_occ_mask3d"],
        float(t["gxy_count"]),
    )
    n, box = obs.shape[0], float(cfg["model"]["box_shape"][0])
    dk = crop_rfft(np.asarray(t["init_mesh"]), n)
    kf = 2 * np.pi / box
    kx, kz = np.fft.fftfreq(n, 1 / n) * kf, np.fft.rfftfreq(n, 1 / n) * kf
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K, K2s = np.sqrt(K2), np.where(K2 > 0, K2, 1)
    x = np.arange(n) * box / n - box / 2
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    Rs = np.where(R > 0, R, 1)
    rh = [X / Rs, Y / Rs, Z / Rs]
    chi_t, D_t, f_t = growth_tables(om)
    D, f = np.interp(R, chi_t, D_t), np.interp(R, chi_t, f_t)
    kv, cols = [KX, KY, KZ], []
    for b in range(len(edges) - 1):
        db = ((K > edges[b]) & (K <= edges[b + 1])) * dk
        d2 = sum(
            rh[i] * rh[j] * np.fft.irfftn(db * kv[i] * kv[j] / K2s, s=(n,) * 3)
            for i in range(3)
            for j in range(3)
        )
        d1 = sum(rh[i] * np.fft.irfftn(db * (-1j) * kv[i] / K2s, s=(n,) * 3) for i in range(3))
        cols += [D * np.fft.irfftn(db, s=(n,) * 3), f * D * (d2 + 2 / Rs * d1)]
    keff = np.array([K[(K > edges[b]) & (K <= edges[b + 1])].mean() for b in range(len(edges) - 1)])
    y, w = obs / np.where(S > 0, nbar * S, 1) - 1, nbar * S
    out = []
    for lo, hi in shells:
        m = mask & (R >= lo) & (R < hi)
        Xm = np.stack([c[m] for c in cols], 1)
        A = Xm.T @ (Xm * w[m][:, None])
        coef = np.linalg.solve(A, Xm.T @ (y[m] * w[m]))
        chi2 = ((y[m] - Xm @ coef) ** 2 * w[m]).sum() / (m.sum() - len(coef))
        err = np.sqrt(np.diag(np.linalg.inv(A) * chi2))
        out.append({"k": keff, "B": coef[0::2], "F": coef[1::2], "eB": err[0::2], "eF": err[1::2]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--runs", nargs="+", required=True, help="label=run_dir (truth.npz with obs and the ICs)"
    )
    ap.add_argument("--profiles", nargs="*", default=[], help="label=alpha0.json,alpha_free.json")
    ap.add_argument("--chi_range", type=float, nargs=2, default=[1093.0, 2447.0])
    ap.add_argument("--n_shells", type=int, default=6)
    ap.add_argument("--k_range", type=float, nargs=2, default=[0.01, 0.034])
    args = ap.parse_args()
    edges_chi = np.linspace(*args.chi_range, args.n_shells + 1)
    shells = list(zip(edges_chi[:-1], edges_chi[1:], strict=True))
    chi_mid = 0.5 * (edges_chi[1:] + edges_chi[:-1])
    data = {}
    for item in args.runs:
        label, run = item.split("=", 1)
        bands = regression(run, shells)
        B, eB = [], []
        for v in bands:
            sel = (v["k"] > args.k_range[0]) & (v["k"] < args.k_range[1])
            wt = 1 / v["eB"][sel] ** 2
            B.append(float((v["B"][sel] * wt).sum() / wt.sum()))
            eB.append(float(wt.sum() ** -0.5))
        data[label] = {"run": Path(run).name, "chi": chi_mid.tolist(), "bE": B, "bE_err": eB}
        print(label, np.round(B, 3).tolist(), "±", np.round(eB, 3).tolist())
    for item in args.profiles:
        label, files = item.split("=", 1)
        data.setdefault(label, {})["profile"] = {
            Path(f).stem: json.load(open(f))["profile"] for f in files.split(",")
        }

    c0, D0, _ = growth_tables(OM_FID)
    fig, ax = plt.subplots(
        1, 2 if args.profiles else 1, figsize=(11 if args.profiles else 5.6, 4.2), squeeze=False
    )
    ax = ax[0]
    for c, (label, d) in zip(COLORS, data.items(), strict=False):
        if "bE" in d:
            ax[0].errorbar(
                d["chi"],
                d["bE"],
                d["bE_err"],
                fmt="o",
                ms=6,
                color=c,
                lw=1.5,
                capsize=3,
                label=label,
            )
    Dm = np.interp(chi_mid, c0, D0)
    ref = np.average(next(d["bE"] for d in data.values() if "bE" in d))
    ax[0].plot(
        chi_mid,
        Dm.mean() / Dm * ref,
        color="0.4",
        ls=":",
        lw=1.5,
        label=r"$b_E \propto 1/D(\chi)$ (scaled)",
    )
    ax[0].set_xlabel(r"comoving distance $\chi$ [Mpc/$h$]")
    ax[0].set_ylabel(
        rf"linear bias $b_E$ against the true ICs (${args.k_range[0]}<k<{args.k_range[1]}$)"
    )
    ax[0].legend(frameon=False)
    if args.profiles:
        for c, (label, d) in zip(COLORS, data.items(), strict=False):
            for (stem, prof), ls in zip(d.get("profile", {}).items(), ("-", "--"), strict=False):
                pts = sorted((p["value"], p["chi2"]) for p in prof)
                om, c2 = np.array(pts).T
                ax[1].plot(
                    om,
                    c2 - c2[np.argmin(np.abs(om - OM_FID))],
                    ls,
                    marker="o",
                    ms=5,
                    color=c,
                    lw=2,
                    label=f"{label}, {'α = 0' if stem.endswith('alpha0') else 'α free'}",
                )
        ax[1].axvline(OM_FID, color="0.6", lw=1)
        ax[1].axhline(0, color="0.8", lw=1)
        ax[1].set_xlabel(r"$\Omega_m$")
        ax[1].set_ylabel(
            r"$\chi^2 - \chi^2(\Omega_m^\mathrm{fid})$ at the true ICs, biases profiled"
        )
        ax[1].legend(frameon=False, fontsize=9)
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "bias_evolution_true_ic.png", dpi=150)
    (OUT / "bias_evolution_true_ic.json").write_text(json.dumps(data, indent=1) + "\n")
    print(f"Saved {OUT / 'bias_evolution_true_ic.png'}")


if __name__ == "__main__":
    main()
