#!/usr/bin/env python
"""
Plot the CMB lensing reconstruction noise of Planck PR4, ACT DR6 and the Simons Observatory baseline.

Top: N_ell^kappa_kappa of the three; bottom: each over ACT DR6. The shaded band is the multipole range
of the likelihood (ell <= 2 nside = 64). The SO curve is the file the paper's runs use (pipeline 3.2).

Usage:
    python scripts/plot_cmb_noise_comparison.py [--output OUTPUT_DIR] [--lmax_band 64]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PLANCK = "/global/cfs/cdirs/cmb/data/planck2020/PR4_lensing/PR4_nlkk_p.dat"
ACT = "data/N_L_kk_act_dr6_lensing_v1_baseline.txt"
SO = "data/N_L_kk_so_v3_1_1_baseline_mv.txt"


def load_planck_nlkk(filepath=PLANCK):
    """Planck PR4 N_ell^kappa_kappa (one column, indexed by ell)."""
    data = np.loadtxt(filepath)
    return np.arange(len(data)), data


def load_two_columns(filepath):
    """N_ell^kappa_kappa from a file of columns ell, N_ell (ACT DR6, SO)."""
    data = np.loadtxt(filepath)
    return data[:, 0].astype(int), data[:, 1]


def plot_comparison(output_dir="figures", lmax_band=64):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curves = {
        "Planck PR4": load_planck_nlkk(),
        "ACT DR6 baseline": load_two_columns(ACT),
        "Simons Observatory baseline": load_two_columns(SO),
    }
    colors = {"Planck PR4": "C0", "ACT DR6 baseline": "C1", "Simons Observatory baseline": "C2"}
    act_ell, act_nl = curves["ACT DR6 baseline"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax in axes:
        ax.axvspan(2, lmax_band, color="0.85", alpha=0.6, lw=0)
    ax = axes[0]
    for label, (ell, nl) in curves.items():
        m = (ell >= 2) & (ell <= 3000) & (nl > 0)
        ax.loglog(ell[m], nl[m], label=label, lw=2.5, alpha=0.85, color=colors[label])
    ax.set_ylabel(r"$N_\ell^{\kappa\kappa}$", fontsize=13)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=11)
    ax.grid(True, which="both", alpha=0.3, ls="-", lw=0.5)
    ax.set_title(
        f"CMB lensing reconstruction noise (shaded: likelihood band $\\ell \\leq {lmax_band}$)",
        fontsize=14,
        pad=10,
    )

    ax = axes[1]
    print(f"Mean ratio to ACT DR6 over ell 2-{lmax_band} and 65-2000:")
    for label in ("Planck PR4", "Simons Observatory baseline"):
        ell, nl = curves[label]
        common = np.arange(2, min(2001, act_ell.max() + 1, ell.max() + 1))
        ratio = np.interp(common, ell, nl) / np.interp(common, act_ell, act_nl)
        band = common <= lmax_band
        ax.semilogx(
            common,
            ratio,
            lw=2.5,
            color=colors[label],
            label=f"{label} / ACT DR6 (mean {ratio[band].mean():.2f} at $\\ell \\leq {lmax_band}$)",
        )
        print(f"  {label}: {ratio[band].mean():.3f}, {ratio[~band].mean():.3f}")
    ax.axhline(1, color="black", ls="--", lw=1.5, alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"Multipole $\ell$", fontsize=13)
    ax.set_ylabel(r"$N_\ell / N_\ell^{\rm ACT\,DR6}$", fontsize=13)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=10)
    ax.grid(True, which="both", alpha=0.3, ls="-", lw=0.5)

    plt.tight_layout()
    outfile = output_dir / "cmb_noise_comparison.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument(
        "--output",
        "-o",
        default="figures/spectra_diagnostic",
        help="output directory (default: figures/spectra_diagnostic/)",
    )
    parser.add_argument(
        "--lmax_band",
        type=int,
        default=64,
        help="top of the likelihood band, 2 nside (default: 64)",
    )
    args = parser.parse_args()
    plot_comparison(output_dir=args.output, lmax_band=args.lmax_band)


if __name__ == "__main__":
    main()
