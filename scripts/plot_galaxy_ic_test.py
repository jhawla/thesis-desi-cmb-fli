#!/usr/bin/env python
"""Paper figure fig:gxy_ic: galaxy counts from the true initial conditions against the Abacus counts.

Reads the JSON of `validate_galaxy_from_ic.py` (all survey cells) and the Newton JSON of
`galaxy_likelihood_profile.py --mode newton` (its fitted noise level) and draws, against k: the
transfer P_dm / P_mm, the power of the whitened residual (d - m) / sqrt(nbar S) off the Nyquist planes,
and the shape of the effective bias against the matter of the same ICs (no bias fit), each with the band
of the closure draws (the model plus the fitted noise law, the likelihood's), to sqrt(3) k_N, the
corner modes of the cube beyond the dotted line at k_N.

    python scripts/plot_galaxy_ic_test.py \\
        --ic figures/galaxy_ic_test/abacus_gxyonly_cosmo_box5000_m100_ph202_fog.json \\
        --newton figures/galaxy_likelihood_profile/abacus_gxyonly_cosmo_box5000_m100_ph202_fog_newton.json

Writes figures/galaxy_ic_test/fig_gxy_ic.png.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures" / "galaxy_ic_test"
BLUE, INK, GRAY = "#2a78d6", "#1f1f1e", "#8a8a85"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ic", required=True, type=Path)
    ap.add_argument("--newton", required=True, type=Path)
    args = ap.parse_args()
    d = json.loads(args.ic.read_text())
    s_e = json.loads(args.newton.read_text())["noise_fit"]["s_e"]
    row = d["shells"][0]
    k = np.asarray(d["k"])
    k_nyq = d["k_nyquist"]
    panels = [("T", "transfer $P_{dm}/P_{mm}$", (0.95, 1.03)),
              ("residual", "whitened residual power", (0.7, 1.15)),
              ("B", "effective bias shape (no fit)", (0.9, 1.04))]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    # modes per bin off the Nyquist planes (the residual's): the last corner bin holds a handful
    dk = k[1] - k[0]
    kf = k[0] - dk / 2
    n = int(round(k_nyq / kf * 2))
    ix = np.fft.fftfreq(n) * n
    iz = np.fft.rfftfreq(n) * n
    IX, IY, IZ = np.meshgrid(ix, ix, iz, indexing="ij")
    kk = kf * np.sqrt(IX**2 + IY**2 + IZ**2)
    off = ~((np.abs(IX) == n // 2) | (np.abs(IY) == n // 2) | (IZ == n // 2))
    n_off = np.array([np.sum(off & (np.abs(kk - c) < dk / 2)) for c in k])
    keep = {"residual": n_off >= 50}
    for ax, (key, label, ylim) in zip(axes, panels, strict=True):
        sel = keep.get(key, np.ones(k.size, bool))
        mean, std = np.asarray(row["closure_mean"][key])[sel], np.asarray(row["closure_std"][key])[sel]
        ax.fill_between(k[sel], mean - std, mean + std, color=GRAY, alpha=0.35, lw=0, label="model + fitted noise")
        ax.plot(k[sel], np.asarray(row["data"][key])[sel], color=BLUE, lw=1.6, label="Abacus")
        ax.axhline(1, color=INK, lw=0.7)
        ax.axvline(k_nyq, color=INK, lw=0.8, ls=":", label=r"$k_N$ (cube corners beyond)")
        if key == "residual":
            ax.axhline(s_e**2, color=BLUE, lw=0.8, ls="--", label="fitted noise level")
        ax.set(xlabel=r"$k\;[h\,{\rm Mpc}^{-1}]$", ylabel=label, ylim=ylim, xlim=(0, k.max() * 1.02))
        ax.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_gxy_ic.png", dpi=200)
    print(f"Saved {OUT / 'fig_gxy_ic.png'}")


if __name__ == "__main__":
    main()
