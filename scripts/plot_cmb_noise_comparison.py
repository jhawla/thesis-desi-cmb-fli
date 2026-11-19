#!/usr/bin/env python
"""
Plot the CMB lensing noise of a run config, and the noise of Planck PR4, ACT DR6 and the SO baseline.

`cmb_noise_so.png` (the paper's fig:noise): the config's N_ell (the SO baseline in every run of the
paper), N_ell + C_ell^out (the variance the diagonal likelihood adds per multipole, at the config's
fiducial, pipeline 3.2) and the likelihood band ell <= 2 nside; the curves go to the npz.
`cmb_noise_comparison.png` (thesis): N_ell of Planck PR4, ACT DR6 and SO, each over ACT DR6, with the
same band.

Usage:
    python scripts/plot_cmb_noise_comparison.py [--config CONFIG] [--output OUTPUT_DIR]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from desi_cmb_fli import utils
from desi_cmb_fli.model import get_model_from_config

PLANCK = "/global/cfs/cdirs/cmb/data/planck2020/PR4_lensing/PR4_nlkk_p.dat"
ACT = "data/N_L_kk_act_dr6_lensing_v1_baseline.txt"
SO = "data/N_L_kk_so_v3_1_1_baseline_mv.txt"
CONFIG = "configs/inference/abacus/abacus_kappaonly_so_cosmo_box5000_m160_ph202.yaml"


def load_planck_nlkk(filepath=PLANCK):
    """Planck PR4 N_ell^kappa_kappa (one column, indexed by ell)."""
    data = np.loadtxt(filepath)
    return np.arange(len(data)), data


def load_two_columns(filepath):
    """N_ell^kappa_kappa from a file of columns ell, N_ell (ACT DR6, SO)."""
    data = np.loadtxt(filepath)
    return data[:, 0].astype(int), data[:, 1]


def plot_run_noise(config, output_dir):
    """N_ell and N_ell + C_ell^out of the config, with its likelihood band. Returns ell_max."""
    model, _ = get_model_from_config(utils.yload(config))
    ell = np.asarray(model.ell_1d)
    nell = np.asarray(model.nell_1d)
    cl_out = np.asarray(model.cl_out_cached)
    lmax = int(model.cmb_lmax)
    so_ell, so_nl = load_two_columns(SO)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axvspan(2, lmax, color="0.88", lw=0, label=rf"likelihood band $2\leq\ell\leq{lmax}$")
    m = (so_ell >= 2) & (so_ell <= 1000)
    ax.loglog(so_ell[m], so_nl[m], color="C2", lw=2, label=r"$N_\ell$ (Simons Observatory baseline)")
    s = ell >= 2
    ax.loglog(ell[s], nell[s] + cl_out[s], color="C3", lw=2,
              label=r"$N_\ell + C_\ell^{\rm out}$")
    ax.set_xlabel(r"Multipole $\ell$", fontsize=12)
    ax.set_ylabel(r"$C_\ell$", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, which="both", alpha=0.3, lw=0.5)
    plt.tight_layout()
    outfile = output_dir / "cmb_noise_so.png"
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close()
    np.savez(outfile.with_suffix(".npz"), ell=ell, nell=nell, cl_out=cl_out, config=config)
    print(f"C_out / N over ell 2-{lmax}: {cl_out[s].sum() / nell[s].sum():.3f} (sums), "
          f"at ell 2, 64, 128, {lmax}: "
          + ", ".join(f"{cl_out[i] / nell[i]:.2f}" for i in (2, 64, 128, lmax)))
    print(f"Saved: {outfile} and .npz")
    return lmax


def plot_comparison(output_dir, lmax_band):
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
    print(f"Mean ratio to ACT DR6 over ell 2-{lmax_band} and above:")
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
        print(f"  {label}: {ratio[band].mean():.3f} (range {ratio[band].min():.3f}-"
              f"{ratio[band].max():.3f}), {ratio[~band].mean():.3f}")
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
    parser.add_argument("--config", default=CONFIG, help=f"run config (default: {CONFIG})")
    parser.add_argument(
        "--output",
        "-o",
        default="figures/spectra_diagnostic",
        help="output directory (default: figures/spectra_diagnostic/)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    lmax = plot_run_noise(args.config, output_dir)
    plot_comparison(output_dir, lmax)


if __name__ == "__main__":
    main()
