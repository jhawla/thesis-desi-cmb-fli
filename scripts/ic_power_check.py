#!/usr/bin/env python
"""Power of the published AbacusSummit initial conditions against their own input spectrum.

The IC field is loaded as `load_abacus_ic_truth` does (delta_L at InitialRedshift grown to a=1 with the
header GrowthTable) and its P(k) is divided by the CLASS table the ICs were drawn from (at
CLASS_Redshift, grown to a=1 with the same table), over the full box and over the central cube of the
model box. A loading error (growth, units, FFT normalisation) is a constant factor at every k; a
deficit only at low k is the realisation. Numpy only.

    python scripts/ic_power_check.py [--ic FILE] [--crop 5000] [--json figures/ic_power/ph201.json]
"""

import argparse
import json
from pathlib import Path

import asdf
import numpy as np
import scipy.fft as sf

ROOT = Path(__file__).resolve().parents[1]
IC = "/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/ic/AbacusSummit_huge_c000_ph201/ic_dens_N576.asdf"
EDGES = [0.0, 0.01, 0.02, 0.034, 0.06, 0.1, 0.15]


def spectrum_ratio(d, box, p_in):
    n = d.shape[0]
    p = np.abs(sf.rfftn(d, workers=4)) ** 2 * box**3 / n**6
    kx, kz = 2 * np.pi * sf.fftfreq(n, box / n), 2 * np.pi * sf.rfftfreq(n, box / n)
    k = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2)
    w = np.full(kz.shape, 2.0)
    w[0] = w[-1] = 1.0  # rfft half-plane weights (n even)
    w = np.broadcast_to(w, k.shape)
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:], strict=True):
        s = (k > lo) & (k < hi)
        modes = w[s].sum() / 2
        ratio = np.sum(w[s] * p[s]) / np.sum(w[s] * p_in(k[s]))
        out.append({"k": [lo, hi], "ratio": float(ratio), "cosmic_variance": float(modes**-0.5)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ic", default=IC)
    ap.add_argument("--crop", type=float, default=5000.0, help="side of the central cube (Mpc/h)")
    ap.add_argument("--json", default=str(ROOT / "figures" / "ic_power" / "ph201.json"))
    args = ap.parse_args()

    with asdf.open(args.ic) as f:
        h = f.tree["header"]
        g = dict(h["GrowthTable"])
        box, z_ic, z_class = (
            float(h["BoxSize"]),
            float(h["InitialRedshift"]),
            float(h["CLASS_Redshift"]),
        )
        d = np.asarray(f.tree["data"]["density"], np.float32) * np.float32(g[0.0] / g[z_ic])
    tab = np.loadtxt(ROOT / "data" / "abacus_cosm000_CLASS_power.txt")
    grow2 = (g[0.0] / g[z_class]) ** 2

    def p_in(k):
        return np.exp(np.interp(np.log(k), np.log(tab[:, 0]), np.log(tab[:, 1]))) * grow2

    n = d.shape[0]
    c = int(round(n * args.crop / box))
    s = (n - c) // 2
    res = {
        "ic": args.ic,
        f"full box {box:g}": spectrum_ratio(d, box, p_in),
        f"central cube {args.crop:g}": spectrum_ratio(
            np.ascontiguousarray(d[s : s + c, s : s + c, s : s + c]), args.crop, p_in
        ),
    }
    for name, rows in res.items():
        if name == "ic":
            continue
        print(f"{name}: P_IC / P_input")
        for r in rows:
            print(
                f"  k {r['k'][0]:.3f}-{r['k'][1]:.3f}  {r['ratio']:.4f} +- {r['cosmic_variance']:.4f}"
            )
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(res, indent=1) + "\n")
    print(f"Saved {args.json}")


if __name__ == "__main__":
    main()
