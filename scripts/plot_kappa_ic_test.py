#!/usr/bin/env python
"""Paper figure fig:kappa_from_ic: kappa from the true initial conditions against the AbacusLensing map.

Reads the `.npz` of `validate_kappa_from_ic.py` run with two `--shell_kmax` (the per-shell cut and
none) and draws, in bins of 4 multipoles: the angular power of the map and of the model (dotted: the
map minus the out-of-model term), the coherence with the map (dotted: the most a model exact on the
matter it holds reaches given that term), the error power over N_l (dotted: the term), and the model's
power plus the term over the map's. The per-shell cut, the analysis setting, is the coloured curve.

    python scripts/plot_kappa_ic_test.py \\
        --npz figures/spectra_diagnostic/kappa_from_abacus_ic_box5000_m160_ph202_cut_nocut.npz

Writes figures/spectra_diagnostic/fig_kappa_ic.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures" / "spectra_diagnostic"
BLUE, INK, GRAY = "#2a78d6", "#1f1f1e", "#a3a39e"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", required=True, type=Path)
    args = ap.parse_args()
    d = np.load(args.npz)
    cuts = sorted({k.split("_", 1)[1] for k in d.files}, key=float)
    lmax = d[f"tt_{cuts[0]}"].size - 1
    edges = np.arange(2, lmax + 2, 4)
    ell = 0.5 * (edges[:-1] + edges[1:] - 1)

    def binned(x):
        return np.array([x[a:b].sum() for a, b in zip(edges[:-1], edges[1:], strict=False)])

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for cut in cuts[::-1]:  # no cut first, the analysis curve on top
        b = {k: binned(d[f"{k}_{cut}"]) for k in ("tt", "mm", "tm", "err", "los", "nell")}
        on = float(cut) > 0
        c, lab = (BLUE, "model, per-shell cut") if on else (GRAY, "model, no cut")
        axes[0].plot(ell, b["mm"] / 4, color=c, lw=1.5, label=lab)
        axes[0].plot(ell, (b["tt"] - b["los"]) / 4, color=c, ls=":", lw=1.2)
        axes[1].plot(ell, b["tm"] / np.sqrt(b["tt"] * b["mm"]), color=c, lw=1.5)
        axes[1].plot(ell, np.sqrt(np.clip(1 - b["los"] / b["tt"], 0, 1)), color=c, ls=":", lw=1.2)
        axes[2].plot(ell, b["err"] / b["nell"], color=c, lw=1.5)
        axes[2].plot(ell, b["los"] / b["nell"], color=c, ls=":", lw=1.2)
        axes[3].plot(ell, (b["mm"] + b["los"]) / b["tt"], color=c, lw=1.5)
    axes[0].plot(ell, b["tt"] / 4, color=INK, lw=1.8, label="AbacusLensing")
    axes[0].plot([], [], color=INK, ls=":", lw=1.2, label="expected")
    axes[0].set(yscale="log", ylabel=r"$C_\ell$")
    axes[1].set(ylim=(0, 1), ylabel=r"coherence $r_\ell$")
    axes[2].set(yscale="log", ylabel=r"$C_\ell^{\rm err}/N_\ell$")
    axes[3].set(ylabel=r"$(C_\ell^{mm}+C_\ell^{\rm out})/C_\ell^{tt}$")
    for ax in axes[2:]:
        ax.axhline(1, color=INK, lw=0.7)
    for ax in axes:
        ax.set(xlabel=r"$\ell$", xlim=(0, lmax))
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_kappa_ic.png", dpi=200)
    print(f"Saved {OUT / 'fig_kappa_ic.png'}")


if __name__ == "__main__":
    main()
