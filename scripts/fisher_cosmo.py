"""Fisher forecast on (Omega_m, sigma8) with the biases: galaxies alone, kappa alone, joint.

The geometry, band, noise, truth biases and priors come from a run config (default: the closure at
the Abacus HUGE geometry, whose truth biases are the galaxy-only Abacus posterior means). Formulas
and approximations in docs/pipeline.md §6 ("Fisher forecasts"):

- galaxies: 3-D redshift-space Fisher (``galaxy_fisher_3d_cosmo``), k in [2 pi / V^(1/3), pi / cell];
- kappa alone: Limber Fisher of kappa over l = 2 ... 2 nside (``angular_fisher_cosmo``, probes 'k');
- joint: galaxies + the kappa increment F[shells + kappa] - F[shells] (tomographic Limber);
- the line of sight outside the model as noise at the fiducial (high_z_mode fixed) or as signal;
- the Gaussian priors of the config's latents on every parameter.

Usage:
  python scripts/fisher_cosmo.py [--config CFG] [--densities 1.0 0.1 0.03] [--fnl]
  python scripts/fisher_cosmo.py --resolution_scan   # box / cell trade-offs
  python scripts/fisher_cosmo.py --mismatch FILE      # first-order shift from a kappa power mismatch

``--mismatch`` takes the data/model ratio of the kappa power (model = the box kappa plus the line of
sight): the ``kappa_from_ic.npz`` of ``validate_kappa_from_ic.py`` (C^tt / (C^mm + C^LOS) at the
config's shell_kmax), or a text file of two columns, ell and ratio. The data then hold
(ratio - 1) (C_box + C_LOS) beyond the model, and the script prints the shift F^-1 b of
(Omega_m, sigma8) it causes, kappa alone and joint, in units of the marginal sigma.
Figure: figures/fisher_diagnostic/fisher_cosmo_contours.png; numbers:
figures/fisher_diagnostic/fisher_cosmo.json (next to the figure).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from desi_cmb_fli import fisher as fi

ROOT = Path(__file__).resolve().parents[1]
BLUE, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"


def ellipse(F2, center, nsig2=2.30, n=200):
    """68 % contour (Delta chi^2 = 2.30) of a 2x2 Fisher matrix."""
    C = np.linalg.inv(F2)
    vals, vecs = np.linalg.eigh(C)
    t = np.linspace(0, 2 * np.pi, n)
    xy = vecs @ (np.sqrt(nsig2 * vals)[:, None] * np.array([np.cos(t), np.sin(t)]))
    return center[0] + xy[0], center[1] + xy[1]


def marginalised_2x2(F, params):
    C = np.linalg.inv(F)[:2, :2]
    assert params[:2] == ("Omega_m", "sigma8")
    return np.linalg.inv(C)


# (label, box side, cell, final mesh); the shells start at the config's chi_matter_min and the
# per-shell cut sits at the init-grid Nyquist of each cell.
RESOLUTIONS = [
    ("7500 / 93.75 (current)", 7500.0, 93.75, 80),
    ("7500 / 62.5", 7500.0, 62.5, 120),
    ("7500 / 46.875", 7500.0, 46.875, 160),
    ("5000 / 62.5", 5000.0, 62.5, 80),
    ("5000 / 41.67", 5000.0, 5000.0 / 120, 120),
    ("5000 / 31.25", 5000.0, 31.25, 160),
]


def resolution_scan(cfg, bgs, fid, params, prior, nell, noise_scaling, out_path):
    """Galaxies, kappa alone and joint at each (box, cell) of RESOLUTIONS, density 1.

    The box enters through the kappa box edge (observer at the centre: box / 2), the cell through
    the galaxies' k_max = pi / cell and the per-shell cut of kappa at the init-grid Nyquist; kappa
    keeps l <= 2 nside of the config and the data-map depth; the matter between the box edge and
    chi_high_z_max, and what the cut removes, are line-of-sight noise.
    """
    base = fi.kappa_geometry(cfg)
    oversamp = float(cfg["model"].get("init_oversamp", 1.0))
    rows, out = [], {}
    for label, box, cell, mesh in RESOLUTIONS:
        kappa = {**base, "chi_box": box / 2, "k_cut": np.pi * oversamp / cell}
        Fg = fi.galaxy_fisher_3d_cosmo(bgs, fid, fi.HUGE_LRG, np.pi / cell, params=params) + prior
        kw = {"params": params, "noise_scaling": noise_scaling, "nell": nell, "los": "noise"}
        dF = fi.angular_fisher_cosmo(bgs, fid, fi.HUGE_LRG, kappa, "gk", **kw) - (
            fi.angular_fisher_cosmo(bgs, fid, fi.HUGE_LRG, kappa, "g", **kw))
        Fk = fi.angular_fisher_cosmo(bgs, fid, fi.HUGE_LRG, kappa, "k", params=("Omega_m", "sigma8"),
                                     noise_scaling=noise_scaling, nell=nell, los="noise")
        Fk = Fk + fi.prior_fisher(cfg["latents"], ("Omega_m", "sigma8"))
        sk = np.sqrt(np.diag(np.linalg.inv(Fk)))
        sg = [fi.marginal(Fg, params, p) for p in ("Omega_m", "sigma8")]
        sj = [fi.marginal(Fg + dF, params, p) for p in ("Omega_m", "sigma8")]
        cost = (mesh / 80) ** 3
        rows.append((label, cost, sg, sk, sj))
        out[label] = {"box": box, "cell": cell, "mesh": mesh, "cost_vs_current": cost,
                      "galaxies": sg, "kappa": list(sk), "joint": sj}
    print("\nResolution scan, density 1, LOS as noise; sigma(Omega_m) / sigma(sigma8):")
    print(f"  {'box / cell':30s} {'cost':>5s}  {'galaxies':>15s}  {'kappa alone':>15s}  "
          f"{'joint':>15s}  gain Om, s8")
    for label, cost, sg, sk, sj in rows:
        print(f"  {label:30s} {cost:5.1f}  {sg[0]:.4f} / {sg[1]:.4f}  {sk[0]:.4f} / {sk[1]:.4f}  "
              f"{sj[0]:.4f} / {sj[1]:.4f}  {1 - sj[0] / sg[0]:.0%}, {1 - sj[1] / sg[1]:.0%}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=1) + "\n")
    print(f"Saved {out_path}")


def load_mismatch_ratio(path, k_cut, lmax):
    """Data/model kappa power ratio on l = 0 ... lmax (1 outside the file's range); from a
    ``kappa_from_ic.npz``, the spectra at the config's per-shell cut ``k_cut``."""
    if str(path).endswith(".npz"):
        d = np.load(path)
        keys = [k for k in d.files if k.startswith("tt_")]
        key = next((k for k in keys if np.isclose(float(k[3:]), k_cut, rtol=1e-3, atol=1e-6)), None)
        if key is None:
            raise ValueError(f"{path} holds no spectra at shell_kmax {k_cut:.4g}: {keys}")
        c = key[3:]
        ell, ratio = np.arange(d[key].size), d[f"tt_{c}"] / (d[f"mm_{c}"] + d[f"los_{c}"])
    else:
        ell, ratio = np.loadtxt(path, unpack=True)
    out = np.ones(lmax + 1)
    ok = (ell >= 2) & (ell <= lmax)
    out[ell[ok].astype(int)] = ratio[ok]
    return out


def mismatch_shift(args, cfg, bgs, fid, params, prior, kappa, kmax, nell, noise_scaling):
    """First-order shift of the parameters from the kappa power mismatch of ``args.mismatch``."""
    lmax, ells = int(kappa["lmax"]), np.arange(int(kappa["lmax"]) + 1)
    bg0 = bgs(fid)
    model_power = sum(fi.kappa_spectra(bg0, np.maximum(ells, 1), kappa))
    ratio = load_mismatch_ratio(args.mismatch, kappa["k_cut"], lmax)
    dC = (ratio - 1.0) * model_power
    print(f"\nmismatch {args.mismatch}: data/model kappa power, l 2-{lmax}: "
          + " ".join(f"{x:.3f}" for x in ratio[2:]))
    kw = {"noise_scaling": noise_scaling, "nell": nell, "los": "noise", "mismatch": dC}
    p2 = ("Omega_m", "sigma8")
    Fk, bk = fi.angular_fisher_cosmo(bgs, fid, fi.HUGE_LRG, kappa, "k", params=p2, **kw)
    Fk = Fk + fi.prior_fisher(cfg["latents"], p2)
    Fg = fi.galaxy_fisher_3d_cosmo(bgs, fid, fi.HUGE_LRG, kmax, params=params) + prior
    Fgk, bj = fi.angular_fisher_cosmo(bgs, fid, fi.HUGE_LRG, kappa, "gk", params=params, **kw)
    Fj = Fg + Fgk - fi.angular_fisher_cosmo(bgs, fid, fi.HUGE_LRG, kappa, "g", params=params,
                                            noise_scaling=noise_scaling, nell=nell, los="noise")
    out = {}
    for name, F, b, names in (("kappa alone", Fk, bk, p2), ("joint", Fj, bj, params)):
        C = np.linalg.inv(F)
        shift = C @ b
        out[name] = {}
        for i, p in enumerate(names[:2]):
            out[name][p] = {"shift": float(shift[i]), "sigma": float(np.sqrt(C[i, i]))}
        print(f"  {name:12s}: " + "  ".join(
            f"d{p} {v['shift']:+.4f} ({v['shift'] / v['sigma']:+.2f} sigma)" for p, v in out[name].items()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--config", default=str(ROOT / "configs/inference/scan/closure_d1p00_joint.yaml")
    )
    ap.add_argument("--densities", type=float, nargs="+", default=[1.0])
    ap.add_argument("--fnl", action="store_true", help="free fNL too (default: fixed at its truth)")
    ap.add_argument("--b1_alpha", action="store_true",
                    help="free the bias evolution (1 + b1)(D(z_b1)/D(z))^b1_alpha (default: fixed at its "
                         "truth, 0 if absent); prior from the config latents, N(1, 1) if absent; "
                         "outputs get a _b1alpha suffix")
    ap.add_argument("--resolution_scan", action="store_true",
                    help="only the (box, cell) scan of RESOLUTIONS, density 1")
    ap.add_argument("--mismatch", default=None, metavar="FILE",
                    help="only the first-order shift from a kappa power mismatch (see the docstring)")
    ap.add_argument("--fig", default=str(ROOT / "figures/fisher_diagnostic/fisher_cosmo_contours.png"))
    ap.add_argument(
        "--out",
        default=str(ROOT / "figures/fisher_diagnostic/fisher_cosmo.json"),
    )
    args = ap.parse_args()
    if args.b1_alpha:
        args.out = str(Path(args.out).with_name(Path(args.out).stem + "_b1alpha.json"))
        args.fig = str(Path(args.fig).with_name(Path(args.fig).stem + "_b1alpha.png"))

    cfg = yaml.safe_load(open(args.config))
    cmb = cfg["cmb_lensing"]
    truth = {**cfg["truth_params"], **cfg.get("abacus_truth_params", {})}
    params = fi.COSMO_PARAMS + (("fNL",) if args.fnl else ()) + (("b1_alpha",) if args.b1_alpha else ())
    fid = {p: float(truth.get(p, 0.0)) for p in ("Omega_m", "sigma8", "b1", "bn2", "fNL", "b1_alpha")}
    cfg["latents"].setdefault("b1_alpha", {"loc": 1.0, "scale": 1.0})
    kappa = fi.kappa_geometry(cfg)
    kmax = np.pi / float(cfg["model"]["cell_size"])
    nell_tab = np.loadtxt(cmb.get("cmb_noise_nell") or fi.NELL_FILE)
    noise_scaling = float(cmb.get("cmb_noise_scaling", 1.0))
    prior = fi.prior_fisher(cfg["latents"], params)
    bgs = fi.BackgroundCache(z_source=float(cmb.get("z_source", 1089.28)))
    fid["z_b1"] = fi.bias_reference_z(bgs(fid), fi.HUGE_LRG)
    shown = {k: v for k, v in kappa.items() if k != "window"}
    print(f"config {Path(args.config).name}: kmax {kmax:.4f} h/Mpc, kappa {shown}, N_l x{noise_scaling}")
    print(f"fiducial {fid}; free {params}; priors (config latents) on every parameter")

    if args.mismatch:
        shift = mismatch_shift(args, cfg, bgs, fid, params, prior, kappa, kmax, nell_tab, noise_scaling)
        out = Path(args.out).with_name(f"fisher_cosmo_mismatch_{Path(args.mismatch).stem}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"mismatch": args.mismatch, "config": args.config, **shift}, indent=1) + "\n")
        print(f"Saved {out}")
        return

    if args.resolution_scan:
        resolution_scan(cfg, bgs, fid, params, prior, nell_tab, noise_scaling,
                        Path(args.out).with_name("fisher_cosmo_resolution.json"))
        return

    results = {}
    kappa_only = {}
    for los in ("noise", "signal"):
        Fk = fi.angular_fisher_cosmo(bgs, fid, fi.HUGE_LRG, kappa, "k", params=("Omega_m", "sigma8"),
                                     noise_scaling=noise_scaling, nell=nell_tab, los=los)
        kappa_only[los] = Fk + fi.prior_fisher(cfg["latents"], ("Omega_m", "sigma8"))
    for d in args.densities:
        Fg = fi.galaxy_fisher_3d_cosmo(bgs, fid, fi.HUGE_LRG, kmax, params=params, density_scale=d)
        row = {"galaxies": Fg + prior}
        for los in ("noise", "signal"):
            kw = {"params": params, "density_scale": d, "noise_scaling": noise_scaling,
                  "nell": nell_tab, "los": los}
            dF = fi.angular_fisher_cosmo(bgs, fid, fi.HUGE_LRG, kappa, "gk", **kw) - (
                fi.angular_fisher_cosmo(bgs, fid, fi.HUGE_LRG, kappa, "g", **kw)
            )
            row[f"joint (LOS {los})"] = Fg + dF + prior
        results[d] = row

    out = {"config": args.config, "params": params, "fiducial": fid, "densities": {}}
    print("\nmarginal 1-sigma (with the config priors):")
    for los, Fk in kappa_only.items():
        s = [np.sqrt(np.linalg.inv(Fk)[i, i]) for i in range(2)]
        print(f"  kappa alone (LOS {los}): sigma(Omega_m) {s[0]:.4f}  sigma(sigma8) {s[1]:.4f}")
        out[f"kappa alone (LOS {los})"] = {"Omega_m": s[0], "sigma8": s[1]}
    for d, row in results.items():
        print(f"  density {d}:")
        out["densities"][str(d)] = {}
        s_g = {p: fi.marginal(row["galaxies"], params, p) for p in params}
        for name, F in row.items():
            s = {p: fi.marginal(F, params, p) for p in params}
            C = np.linalg.inv(F)
            r = C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])
            gain = "" if name == "galaxies" else "  gain " + ", ".join(
                f"{p} {1 - s[p] / s_g[p]:.1%}" for p in ("Omega_m", "sigma8"))
            print(f"    {name:18s}: " + "  ".join(f"sigma({p}) {v:.4g}" for p, v in s.items())
                  + f"  corr(Om, s8) {r:+.2f}{gain}")
            out["densities"][str(d)][name] = {**s, "corr_Om_s8": r}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")

    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.6), squeeze=False)
    c0 = (fid["Omega_m"], fid["sigma8"])
    for ax, (d, row) in zip(axes[0], results.items(), strict=True):
        for F2, color, ls, label in (
            (marginalised_2x2(row["galaxies"], params), BLUE, "-", "galaxies (3-D, RSD)"),
            (kappa_only["noise"], AQUA, "--", "κ alone"),
            (marginalised_2x2(row["joint (LOS noise)"], params), ORANGE, "-", "joint"),
        ):
            x, y = ellipse(F2, c0)
            ax.plot(x, y, color=color, ls=ls, lw=2, label=label)
        ax.plot(*c0, "k+", ms=10)
        ax.set_xlabel(r"$\Omega_m$")
        ax.set_ylabel(r"$\sigma_8$")
        ax.set_title(f"density {d:g} × LRG, 68 % (LOS as noise)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=150)
    print(f"\nSaved {args.fig} and {args.out}")


if __name__ == "__main__":
    main()
