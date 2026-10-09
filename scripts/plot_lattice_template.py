#!/usr/bin/env python
"""Figures of the κ lattice template (pipeline §7.15), from the outputs of kappa_lattice_template.py.

    python scripts/plot_lattice_template.py --config CFG --grid 'data/cache/template/CFGSTEM_*.npz' \\
        --runs ph201=RUN_DIR ph202=RUN_DIR2 --geometries 'figures/kappa_lattice_template/geometry_*.json'

1. `kappa_only_template_contours.png`: for each run, the κ-only MCMC (second half of the batches)
   against the Gaussian κ likelihood of its data with the template as mean and without it, 68/95 %.
2. `kappa_template_map_and_spacing.png`: the template at the grid point nearest the fiducial (the
   config gives the packing), its S/N against the covariance and its ratio to the undisplaced lattice
   against the particle spacing (one `analyse --json` per geometry), with the Debye–Waller form
   exp[−(2π/d)²σ²/2] fitted as an empirical description.
Outputs and their numbers (.json) go to figures/kappa_lattice_template/.
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from kappa_lattice_template import OM_FID, S8_FID, covariance, data_vector, load_grid  # noqa: E402
from scipy.interpolate import RectBivariateSpline  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures" / "kappa_lattice_template"
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#1f1f1e",
    "#6b6a65",
)
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def mcmc(run):
    c = yaml.safe_load(open(f"{run}/config/config.yaml"))["latents"]
    fs = sorted(
        glob.glob(f"{run}/config/samples_batch_*.npz"),
        key=lambda f: int(re.findall(r"_(\d+)\.npz", f)[0]),
    )
    fs = fs[len(fs) // 2 :]
    return [
        np.concatenate([np.load(f)[p + "_"] for f in fs], 1).ravel() * c[p]["scale_fid"]
        + c[p]["loc_fid"]
        for p in ("Omega_m", "sigma8")
    ]


def mass_levels(w, masses=(0.95, 0.68)):
    s = np.sort(w.ravel())[::-1]
    c = np.cumsum(s)
    return sorted(s[np.searchsorted(c, m)] for m in masses)


def gauss_posterior(g, y, with_t, omf, s8f, prior_scale=0.15):
    sel = g["l"] >= 2
    om_g = np.unique([p[0] for p in g["points"]])
    s8_g = np.unique([p[1] for p in g["points"]])
    idx = {p: i for i, p in enumerate(g["points"])}
    L = np.zeros((len(om_g), len(s8_g)))
    for a, o in enumerate(om_g):
        for b, s in enumerate(s8_g):
            i = idx[(o, s)]
            C = covariance(g, i)
            r = y - g["pairs"][i].mean(0) if with_t else y
            L[a, b] = np.sum(r[sel] ** 2 / C[sel] + np.log(C[sel]))
    Om, S8 = np.meshgrid(omf, s8f, indexing="ij")
    M = (
        RectBivariateSpline(om_g, s8_g, L)(omf, s8f)
        + ((Om - OM_FID) / prior_scale) ** 2
        + ((S8 - S8_FID) / prior_scale) ** 2
    )
    w = np.exp(-0.5 * (M - M.min()))
    return w / w.sum()


def fig_contours(g, runs):
    om_g = np.unique([p[0] for p in g["points"]])
    s8_g = np.unique([p[1] for p in g["points"]])
    omf, s8f = np.linspace(om_g.min(), om_g.max(), 240), np.linspace(s8_g.min(), s8_g.max(), 240)
    fig, axes = plt.subplots(
        1, len(runs), figsize=(4.8 * len(runs), 4.3), sharey=True, squeeze=False
    )
    out = {}
    for ax, (label, run) in zip(axes[0], runs.items(), strict=True):
        om, s8 = mcmc(run)
        H, xe, ye = np.histogram2d(
            om, s8, bins=[np.linspace(0.10, 0.42, 65), np.linspace(0.50, 0.95, 65)]
        )
        H = gaussian_filter(H, 1.2)
        H /= H.sum()
        ax.contour(
            0.5 * (xe[1:] + xe[:-1]),
            0.5 * (ye[1:] + ye[:-1]),
            H.T,
            levels=mass_levels(H),
            colors=BLUE,
            linewidths=2,
        )
        out[label] = {
            "run": Path(run).name,
            "mcmc": {"Omega_m": [om.mean(), om.std()], "sigma8": [s8.mean(), s8.std()]},
        }
        y = data_vector(run, g["u_half"])
        Om, S8 = np.meshgrid(omf, s8f, indexing="ij")
        for with_t, color, ls, key in (
            (True, ORANGE, "--", "with template"),
            (False, AQUA, ":", "without template"),
        ):
            w = gauss_posterior(g, y, with_t, omf, s8f)
            ax.contour(
                omf, s8f, w.T, levels=mass_levels(w), colors=color, linewidths=2, linestyles=ls
            )
            mo, ms = (w * Om).sum(), (w * S8).sum()
            out[label][key] = {
                "Omega_m": [mo, np.sqrt((w * (Om - mo) ** 2).sum())],
                "sigma8": [ms, np.sqrt((w * (S8 - ms) ** 2).sum())],
            }
        ax.plot(OM_FID, S8_FID, "+", color=INK, ms=12, mew=2)
        ax.set_title(
            f"{label}, κ alone (box {g['box']:.0f}, spacing {g['spacing']:.1f} Mpc/h)",
            fontsize=10,
            color=INK,
        )
        ax.set_xlabel(r"$\Omega_m$")
        ax.set_xlim(0.15, 0.40)
        ax.set_ylim(0.58, 0.90)
        ax.grid(alpha=0.2)
    axes[0][0].set_ylabel(r"$\sigma_8$")
    handles = [
        plt.Line2D([], [], color=BLUE, lw=2, label="field-level MCMC"),
        plt.Line2D(
            [],
            [],
            color=ORANGE,
            lw=2,
            ls="--",
            label="Gaussian likelihood with the model's lattice template",
        ),
        plt.Line2D([], [], color=AQUA, lw=2, ls=":", label="Gaussian likelihood, template removed"),
        plt.Line2D([], [], color=INK, marker="+", ls="", ms=10, mew=2, label="truth"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(
        "κ-only posteriors and the lattice template: 68 and 95 % contours (Gaussian curves on the grid's range)",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.97))
    fig.savefig(OUT / "kappa_only_template_contours.png", dpi=150)
    (OUT / "kappa_only_template_contours.json").write_text(
        json.dumps(out, indent=1, default=float) + "\n"
    )


def fig_template(g, cfg_path, geometries):
    import healpy as hp
    import jax.numpy as jnp

    from desi_cmb_fli.model import get_model_from_config

    model, _ = get_model_from_config(yaml.safe_load(open(cfg_path)))
    fid = min(
        range(len(g["points"])),
        key=lambda i: abs(g["points"][i][0] - OM_FID) + abs(g["points"][i][1] - S8_FID),
    )
    T = g["pairs"][fid].mean(0) * g["u_half"]
    tmap = np.asarray(model.unpack_kappa_obs_to_map(jnp.asarray(T)))
    pts = sorted(
        (
            {
                "box": d["box"],
                "d": d["particle_spacing"],
                "sn": d["S/N"]["l<=64"],
                "damp": d["template/undisplaced"],
            }
            for d in (json.load(open(f)) for f in sorted(glob.glob(geometries)))
        ),
        key=lambda p: -p["d"],
    )
    fig = plt.figure(figsize=(14.5, 4.4))
    lim = 3 * tmap.std()
    hp.mollview(
        tmap,
        fig=fig.number,
        sub=(1, 3, 1),
        cmap="RdBu_r",
        min=-lim,
        max=lim,
        cbar=True,
        unit="κ",
        format="%.4f",
        title=f"Model template E[κ] at the truth, ℓ ≤ 64\nbox {g['box']:.0f}, spacing {g['spacing']:.1f} Mpc/h",
    )
    colors = {7500: BLUE, 6000: ORANGE, 5000: AQUA}
    ax = fig.add_subplot(1, 3, 2)
    for p in pts:
        ax.plot(
            p["d"],
            p["sn"],
            "o",
            ms=8,
            color=colors.get(round(p["box"] / 500) * 500, YELLOW),
            mec="white",
            mew=1.5,
        )
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    for b in sorted({round(p["box"]) for p in pts}, reverse=True):
        ax.plot([], [], "o", color=colors.get(round(b / 500) * 500, YELLOW), ms=8, label=f"box {b}")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("particle spacing d [Mpc/h]")
    ax.set_ylabel("template S/N, ℓ ≤ 64")
    ax.set_title("Template signal-to-noise against the covariance", fontsize=10, color=INK)
    ax.grid(alpha=0.2)
    ax = fig.add_subplot(1, 3, 3)
    d = np.array([p["d"] for p in pts])
    damp = np.array([p["damp"] for p in pts])
    sigma = float(np.sqrt(np.mean(-2 * np.log(damp) / (2 * np.pi / d) ** 2)))
    for p in pts:
        ax.plot(
            p["d"],
            p["damp"],
            "o",
            ms=8,
            color=colors.get(round(p["box"] / 500) * 500, YELLOW),
            mec="white",
            mew=1.5,
        )
    dd = np.linspace(15, 40, 200)
    ax.plot(
        dd,
        np.exp(-0.5 * (2 * np.pi / dd) ** 2 * sigma**2),
        color=INK,
        lw=2,
        label=rf"empirical fit $\exp[-(2\pi/d)^2\sigma^2/2]$, $\sigma$ = {sigma:.1f} Mpc/h"
        + "\n(the form only: the linear displacements are smaller, pipeline 7.15)",
    )
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("particle spacing d [Mpc/h]")
    ax.set_ylabel("template / undisplaced lattice")
    ax.set_title("Template over the undisplaced-lattice map", fontsize=10, color=INK)
    ax.grid(alpha=0.2)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.14, top=0.86, wspace=0.32)
    fig.savefig(OUT / "kappa_template_map_and_spacing.png", dpi=150)
    (OUT / "kappa_template_map_and_spacing.json").write_text(
        json.dumps({"points": pts, "sigma_fit": sigma}, indent=1) + "\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--config", required=True, help="the config of the grid (for the packing of the map)"
    )
    ap.add_argument(
        "--grid", required=True, help="glob of kappa_lattice_template.py measure outputs"
    )
    ap.add_argument("--runs", nargs="+", default=[], help="label=run_dir of κ-only runs")
    ap.add_argument(
        "--geometries", default=None, help="glob of analyse --json outputs, one per geometry"
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    g = load_grid(args.grid)
    if args.runs:
        fig_contours(g, dict(r.split("=", 1) for r in args.runs))
    if args.geometries:
        fig_template(g, args.config, args.geometries)
    print(f"Saved in {OUT}")


if __name__ == "__main__":
    main()
