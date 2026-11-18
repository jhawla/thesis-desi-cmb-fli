#!/usr/bin/env python
"""Paper figure fig:shell_cut: the model's κ at the true ICs with and without the per-shell multipole cut.

Reads the .npz of `validate_kappa_from_ic.py --shell_kmax 0 <cut>` (one run, both cuts) and draws the
model's power plus its covariance term over the map's power:
without the cut the near shells add the particles' discreteness above ℓ_res; with it the ratio stays at 1.

    python scripts/plot_shell_cut.py \\
        --npz figures/spectra_diagnostic/kappa_from_abacus_ic_box5000_m160_ph202_cut_nocut.npz --cut 0.1508

Writes figures/spectra_diagnostic/fig_shell_cut.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures" / "spectra_diagnostic"
BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#1f1f1e"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--cut", type=float, required=True, help="shell_kmax of the analysis (h/Mpc)")
    ap.add_argument("--dl", type=int, default=8, help="ℓ bin width")
    args = ap.parse_args()
    d = np.load(args.npz)
    lmax = d["tt_0.0"].size - 1
    edges = np.arange(2, lmax + 2, args.dl)
    ell = 0.5 * (edges[:-1] + edges[1:] - 1)

    def b(key, c):
        x = d[f"{key}_{c}"]
        return np.array([x[a:z].mean() for a, z in zip(edges[:-1], edges[1:], strict=False)])

    cut = f"{args.cut:g}"
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for c, color, label in (("0.0", BLUE, "no cut"), (cut, ORANGE, "per-shell cut")):
        ax.plot(ell, (b("mm", c) + b("los", c)) / b("tt", c), color=color, label=label)
    ax.axhline(1, color=INK, lw=0.8)
    ax.set(xlabel=r"$\ell$", ylabel=r"$(C_\ell^{\rm model} + C_\ell^{\rm out}) / C_\ell^{\rm map}$")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_shell_cut.png", dpi=200)
    print(f"Saved {OUT / 'fig_shell_cut.png'}")


if __name__ == "__main__":
    main()
