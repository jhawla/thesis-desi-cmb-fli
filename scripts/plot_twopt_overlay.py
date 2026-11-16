#!/usr/bin/env python
"""Overlay the field-level and the two-point (C. Payerne) Omega_m-sigma8 posteriors of one kappa map (fig:twopt).

The two-point chains use CCL's conventions: Omega_m without the massive neutrino (Omega_cb) and sigma8 of the
total matter. They are brought to ours (Omega_m total, sigma8 of CDM + baryons) with Omega_nu and the ratio
sigma8_cb / sigma8_tot of the fiducial cosmology (docs/pipeline.md 7.20). The field-level samples are the
second halves of the run's batches, the latents mapped through their priors as the model does (`samp2base`).

    python scripts/plot_twopt_overlay.py --run $SCRATCH/outputs/run_20261106_143937_60800106 \\
        --name kappaonly_closure_ph201ic_nside128 \\
        --twopt ~/Cosmological_Inference_from_CMBLensing_maps/mcmc_chains/mcmc_samples_kappa_closure_ph201ic_noise6482_nside128_obs_noisy_ell_min2_ell_max256.npz

Writes figures/twopt_payerne/twopt_overlay_<name>.{png,npz}.
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from getdist import MCSamples, plots  # noqa: E402

from desi_cmb_fli.bricks import samp2base  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TRUTH = {"Om": 0.315192, "s8": 0.811355}


def fieldlevel_samples(run):
    """Omega_m and sigma8 of the second half of a run's batches, all chains, latents mapped by samp2base."""
    latents = yaml.safe_load((run / "config/config.yaml").read_text())["latents"]
    files = sorted(
        (run / "config").glob("samples_batch_*.npz"),
        key=lambda f: int(re.findall(r"(\d+)\.npz", f.name)[0]),
    )
    lat = {
        k + "_": np.concatenate([np.load(f)[k + "_"] for f in files], 1)
        for k in ("Omega_m", "sigma8")
    }
    base = samp2base(lat, latents)
    half = lat["Omega_m_"].shape[1] // 2
    return {k: np.asarray(base[k])[:, half:].ravel() for k in ("Omega_m", "sigma8")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--twopt", required=True, type=Path, help="two-point samples (npz with flat_samples)"
    )
    ap.add_argument("--run", required=True, type=Path, help="field-level run directory (κ only)")
    ap.add_argument(
        "--name", required=True, help="output tag, e.g. kappaonly_abacus_ph202_nside128"
    )
    ap.add_argument(
        "--omega_nu", type=float, default=0.001421, help="Omega_nu of one 0.06 eV neutrino (CAMB)"
    )
    ap.add_argument(
        "--s8_cb_over_tot", type=float, default=1.00427, help="sigma8_cb / sigma8_tot (CAMB)"
    )
    args = ap.parse_args()

    tp = np.load(args.twopt)["flat_samples"]
    om_tp = tp[:, 0] + args.omega_nu
    s8_tp = tp[:, 1] * args.s8_cb_over_tot
    fl = fieldlevel_samples(args.run)

    names, labels = ["Om", "s8"], [r"\Omega_m", r"\sigma_8"]
    settings = {"smooth_scale_2D": 0.5, "smooth_scale_1D": 0.5}
    samples = [
        MCSamples(
            samples=np.c_[fl["Omega_m"], fl["sigma8"]],
            names=names,
            labels=labels,
            label="field level",
            settings=settings,
        ),
        MCSamples(
            samples=np.c_[om_tp, s8_tp],
            names=names,
            labels=labels,
            label=r"two-point $C_\ell^{\kappa\kappa}$",
            settings=settings,
        ),
    ]
    for s in samples:
        print(f"{s.label}: " + ", ".join(f"{n} {s.mean(n):.4f} ± {s.std(n):.4f}" for n in names))

    g = plots.get_subplot_plotter(subplot_size=3)
    colors = ["#2a78d6", "#d6452a"]
    g.triangle_plot(
        samples,
        names,
        filled=[True, False],
        markers=TRUTH,
        contour_colors=colors,
        line_args=[{"color": c} for c in colors],
    )
    stem = REPO / "figures/twopt_payerne" / f"twopt_overlay_{args.name}"
    plt.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    np.savez(
        stem.with_suffix(".npz"),
        om_twopt=om_tp,
        s8_twopt=s8_tp,
        om_fieldlevel=fl["Omega_m"],
        s8_fieldlevel=fl["sigma8"],
        twopt=str(args.twopt),
        run=args.run.name,
        omega_nu=args.omega_nu,
        s8_cb_over_tot=args.s8_cb_over_tot,
    )
    print(f"Saved {stem.with_suffix('.png')} and .npz")


if __name__ == "__main__":
    main()
