#!/usr/bin/env python
"""Density scan of what CMB lensing adds on sigma(f_NL): measured paired gains against Fisher forecasts.

For each density of --runs, the measured galaxy-only and joint widths of f_NL and the paired gain
1 - sigma_joint/sigma_gxy (mean of the per-chain ratios at matched batches, second half). Against
them, on a grid of densities, the Fisher forecasts of desi_cmb_fli.fisher (formulas:
docs/pipeline.md §6 "Fisher forecasts"): sigma_gxy from the 3-D galaxy Fisher, and the gain kappa
adds, Delta_F from the tomographic Limber Fisher added to that 3-D galaxy Fisher. The kappa
geometry, the cell (kmax = pi / cell) and the fiducial biases come from the `fisher_config` of --runs.

The printed table also gives the 3-D Fisher width and the Fisher gain computed on the galaxy Fisher
measured from each galaxy-only chain. --noise_table prints, for one galaxy-only run, sigma_kappa and
the Fisher gain per N_l scaling (docs/pipeline.md §7.2).

    python scripts/density_scan.py
    python scripts/density_scan.py --noise_table run_20260910_033019_58153868 \\
        --config configs/inference/abacus/abacus_joint_Nl1p0.yaml
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from desi_cmb_fli import fisher as fi

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = Path(os.environ.get("SCRATCH", ".")) / "outputs"
GXY_COLOR, JOINT_COLOR = "#2a78d6", "#eb6834"


def fiducial(cfg):
    t = cfg.get("truth_params", {})
    return (
        float(t.get("Omega_m", 0.315192)),
        float(t.get("sigma8", 0.811355)),
        float(t.get("b1", 1.19)),
        float(t.get("bn2", 78.0)),
    )


def measure(entry, burn_in=0.5):
    """Measured widths and paired gain of one registry entry (keys absent where a run is missing)."""
    out = dict(entry)
    if not entry.get("gxy"):
        return out
    gxy = fi.load_scalar_chains(OUTPUTS / entry["gxy"])
    half = gxy["fNL"].shape[1] // 2
    out["sigma_gxy"] = float(gxy["fNL"][:, half:].std())
    out["b1"] = float(gxy["b1"][:, half:].mean())
    out["bn2"] = float(gxy["bn2"][:, half:].mean())
    out["F_gxy"] = fi.chain_fisher(gxy, burn_in)
    if entry.get("joint"):
        n = min(fi.n_batches(OUTPUTS / entry["gxy"]), fi.n_batches(OUTPUTS / entry["joint"]))
        g = fi.load_scalar_chains(OUTPUTS / entry["gxy"], ("fNL",), n_batches=n)["fNL"]
        j = fi.load_scalar_chains(OUTPUTS / entry["joint"], ("fNL",), n_batches=n)["fNL"]
        ratio, err, _ = fi.paired_sigma_ratio(g, j, burn_in)
        out.update(
            matched_batches=n,
            ratio=ratio,
            ratio_err=err,
            sigma_joint=float(j[:, j.shape[1] // 2 :].std()),
        )
    return out


def noise_table(args, cfg):
    om, s8, *_ = fiducial(cfg)
    bg = fi.Background(om, s8)
    kappa = fi.kappa_geometry(cfg)
    chains = fi.load_scalar_chains(OUTPUTS / args.noise_table)
    F_g = fi.chain_fisher(chains)
    half = chains["b1"].shape[1] // 2
    b1, bn2 = chains["b1"][:, half:].mean(), chains["bn2"][:, half:].mean()
    s_g = fi.sigma(F_g)
    shown = {k: v for k, v in kappa.items() if k != "window"}
    print(
        f"kappa geometry {shown}; galaxy-only {args.noise_table}: sigma(fNL) {s_g:.3f}, "
        f"b1 {b1:.4f}, bn2 {bn2:.2f}, density scale {args.density}"
    )
    print(f"{'N_l scaling':>12} {'sigma_kappa':>12} {'Fisher sigma_joint':>19} {'Fisher gain':>12}")
    for s in args.noise_scalings:
        dF = fi.kappa_fisher_increment(bg, fi.HUGE_LRG, kappa, b1, bn2, s, args.density)
        s_k = fi.sigma(dF) if np.linalg.det(dF) > 0 else np.inf
        s_j = fi.sigma(F_g + dF)
        print(f"{'x' + str(s):>12} {s_k:12.1f} {s_j:19.3f} {100 * (1 - s_j / s_g):11.2f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=str(ROOT / "configs/inference/scan/density_scan_runs.yaml"))
    ap.add_argument(
        "--densities",
        type=float,
        nargs=3,
        default=[0.01, 1.5, 40],
        metavar=("MIN", "MAX", "N"),
        help="log grid of the Fisher curves",
    )
    ap.add_argument("--fig", default=str(ROOT / "figures/results/density_scan.png"))
    ap.add_argument("--out", default=str(OUTPUTS / "density_scan" / "density_scan.json"))
    ap.add_argument(
        "--noise_table",
        default=None,
        metavar="GXY_RUN",
        help="print sigma_kappa and the Fisher gain per N_l scaling for this run",
    )
    ap.add_argument("--config", default=None, help="with --noise_table: the kappa geometry")
    ap.add_argument("--density", type=float, default=1.0, help="with --noise_table")
    ap.add_argument("--noise_scalings", type=float, nargs="+", default=[1.0, 0.4, 0.1, 0.01, 0.0])
    args = ap.parse_args()

    reg = yaml.safe_load(open(args.runs))
    cfg = yaml.safe_load(open(ROOT / (args.config or reg["fisher_config"])))
    if args.noise_table:
        return noise_table(args, cfg)

    om, s8, b1_fid, bn2_fid = fiducial(cfg)
    bg = fi.Background(om, s8)
    kappa = fi.kappa_geometry(cfg)
    kmax = np.pi / float(cfg["model"]["cell_size"])
    survey = fi.HUGE_LRG
    shown = {k: v for k, v in kappa.items() if k != "window"}
    print(
        f"Fisher: {reg['fisher_config']}, kappa {shown}, kmax {kmax:.4f}, "
        f"fiducial b1 {b1_fid}, bn2 {bn2_fid}"
    )

    grid = np.geomspace(args.densities[0], args.densities[1], int(args.densities[2]))
    curve = {"density": grid.tolist(), "sigma_gxy": [], "sigma_joint": [], "gain": []}
    for d in grid:
        F3 = fi.galaxy_fisher_3d(bg, survey, b1_fid, bn2_fid, kmax, density_scale=d)
        dF = fi.kappa_fisher_increment(bg, survey, kappa, b1_fid, bn2_fid, 1.0, d)
        s3, sj = fi.sigma(F3), fi.sigma(F3 + dF)
        curve["sigma_gxy"].append(s3)
        curve["sigma_joint"].append(sj)
        curve["gain"].append(1 - sj / s3)

    points = []
    print(
        f"\n{'density':>8} {'batches':>7} {'sigma_gxy':>9} {'3-D Fisher':>10} {'sigma_joint':>11} "
        f"{'paired ratio':>15} {'gain':>11} {'Fisher gain (chain)':>19}"
    )
    for entry in reg["pairs"]:
        m = measure(entry)
        if "sigma_gxy" in m:
            d = m["density"]
            F3 = fi.galaxy_fisher_3d(bg, survey, m["b1"], m["bn2"], kmax, density_scale=d)
            dF = fi.kappa_fisher_increment(bg, survey, kappa, m["b1"], m["bn2"], 1.0, d)
            m["sigma_gxy_fisher"] = fi.sigma(F3)
            m["fisher_gain_chain"] = 1 - fi.sigma(m["F_gxy"] + dF) / fi.sigma(m["F_gxy"])
        m.pop("F_gxy", None)
        points.append(m)

        def fmt(key, spec, m=m):
            return format(m[key], spec) if key in m else "-"

        ratio = f"{m['ratio']:.3f}±{m['ratio_err']:.3f}" if "ratio" in m else "-"
        gain = f"{100 * (1 - m['ratio']):5.1f}±{100 * m['ratio_err']:.1f}%" if "ratio" in m else "-"
        fg = f"{100 * m['fisher_gain_chain']:.2f}%" if "fisher_gain_chain" in m else "-"
        print(
            f"{m['density']:8.2f} {fmt('matched_batches', 'd'):>7} {fmt('sigma_gxy', '.2f'):>9} "
            f"{fmt('sigma_gxy_fisher', '.2f'):>10} {fmt('sigma_joint', '.2f'):>11} "
            f"{ratio:>15} {gain:>11} {fg:>19}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"curve": curve, "points": points, "fisher_config": reg["fisher_config"], "kmax": kmax},
        open(args.out, "w"),
        indent=1,
    )
    plot(curve, points, args.fig)
    print(f"\nSaved {args.fig} and {args.out}")


def plot(curve, points, path):
    d = np.asarray(curve["density"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(d, curve["sigma_gxy"], color=GXY_COLOR, lw=2, label="galaxies only, 3-D Fisher")
    for m in points:
        if "sigma_gxy" in m:
            ax1.plot(m["density"], m["sigma_gxy"], "o", color=GXY_COLOR, ms=8, mec="white", mew=1.5)
        if "sigma_joint" in m:
            ax1.plot(
                m["density"], m["sigma_joint"], "s", color=JOINT_COLOR, ms=8, mec="white", mew=1.5
            )
    ax1.plot([], [], "o", color=GXY_COLOR, ms=8, label="galaxies only, field level")
    ax1.plot([], [], "s", color=JOINT_COLOR, ms=8, label=r"galaxies + $\kappa$, field level")
    ax1.set(
        xscale="log",
        yscale="log",
        xlabel=r"galaxy density / $\bar n_\mathrm{LRG}$",
        ylabel=r"$\sigma(f_\mathrm{NL})$",
        title=r"Width of $f_\mathrm{NL}$",
    )
    ax1.set_yticks([4, 6, 10, 20, 40])
    ax1.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.plot(d, 100 * np.asarray(curve["gain"]), color=JOINT_COLOR, lw=2, label="two-point Fisher")
    for m in points:
        if "ratio" in m:
            ax2.errorbar(
                m["density"],
                100 * (1 - m["ratio"]),
                yerr=100 * m["ratio_err"],
                fmt="s",
                ms=8,
                capsize=3,
                color=JOINT_COLOR,
                mec="white",
                mew=1.5,
            )
    ax2.errorbar(
        [], [], yerr=[], fmt="s", color=JOINT_COLOR, ms=8, label="field level, paired runs"
    )
    ax2.set(
        xscale="log",
        xlabel=r"galaxy density / $\bar n_\mathrm{LRG}$",
        ylabel=r"gain $1-\sigma_\mathrm{joint}/\sigma_\mathrm{gxy}$ [%]",
        title=r"What CMB lensing adds on $\sigma(f_\mathrm{NL})$",
    )
    for ax in (ax1, ax2):
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
