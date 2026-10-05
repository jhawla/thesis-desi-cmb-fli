"""How much of the CMB-lensing convergence the runs hold, against the depth of the line of sight.

Limber C_l^kk (non-linear P(k), the fiducial cosmology of a run config) from the observer to the
CMB, split by comoving distance. The integrand depends on the bounds only through the integration
range, so it is tabulated once on a fine chi grid and every range is a cumulative sum of it (checked
against cmb_lensing.compute_theoretical_cl_kappa at start). Two figures, both over the likelihood's
band l = 2 ... 2 nside:

- lensing_fraction_vs_z.png: power fraction C_l(chi_a -> chi(z_max)) / C_l(0 -> chi_s) against
  z_max, from the observer and from the model's chi_matter_min; red line at the model's box edge.
- lensing_spectra_comparison.png: total C_l, what the model shells hold (before the per-shell
  multipole cut, which moves part of it to the covariance), what the data map holds (Abacus or
  closure), the beyond-the-box line-of-sight term of the likelihood covariance, the part in neither,
  and N_l.

Usage: python scripts/plot_lensing_fraction.py [--config configs/inference/abacus/abacus_joint_Nl1p0_chimin700.yaml]
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np
import yaml

from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import compute_theoretical_cl_kappa, lensing_kernel

ROOT = Path(__file__).resolve().parents[1]
BLUE, AQUA, ORANGE, RED, GREY = "#2a78d6", "#1baf7a", "#eb6834", "#e34948", "#8a8984"


def limber_integrand(cosmo, ell, chi, chi_s):
    """(W^2 / chi^2) P_NL((l + 1/2)/chi, a(chi)), shape (n_ell, n_chi)."""
    a = jc.background.a_of_chi(cosmo, chi)
    w2 = (lensing_kernel(cosmo, chi, a, chi_s) / chi) ** 2

    def one(ell_val):
        k = (ell_val + 0.5) / chi
        pk = jax.vmap(lambda ki, ai: jnp.squeeze(jc.power.nonlinear_matter_power(cosmo, ki, ai)))(k, a)
        return w2 * pk

    return np.asarray(jax.lax.map(one, jnp.asarray(ell, dtype=float)))


class LineOfSight:
    """C_l over any [chi_a, chi_b] from one tabulated integrand (trapezoid, cumulative)."""

    def __init__(self, cosmo, ell, chi_s, n_chi=3000):
        self.chi = np.linspace(1.0, chi_s, n_chi)
        f = limber_integrand(cosmo, ell, jnp.asarray(self.chi), chi_s)
        steps = 0.5 * (f[:, 1:] + f[:, :-1]) * np.diff(self.chi)
        self.cum = np.concatenate([np.zeros((f.shape[0], 1)), np.cumsum(steps, axis=1)], axis=1)

    def cl(self, chi_a, chi_b):
        at = lambda c: np.array([np.interp(c, self.chi, row) for row in self.cum])  # noqa: E731
        return at(min(chi_b, self.chi[-1])) - at(max(chi_a, self.chi[0]))


def run_geometry(cfg):
    cmb, model = cfg["cmb_lensing"], cfg["model"]
    box = float(np.atleast_1d(model["box_shape"])[-1])
    chi_boundary = box / 2 if cmb.get("observer_mode", "center") == "center" else box
    truth = {**cfg.get("truth_params", {}), **cfg.get("abacus_truth_params", {})}
    return {
        "cosmo": {k: float(truth[k]) for k in ("Omega_m", "sigma8") if k in truth},
        "z_source": float(cmb.get("z_source", 1089.28)),
        "lmax": 2 * int(cmb["nside"]),
        "chi_min": max(1.0, float(cmb.get("chi_matter_min", 0.0))),
        "chi_boundary": chi_boundary,
        "chi_high_z_max": float(cmb["chi_high_z_max"]),
        "chi_map_min": max(1.0, float(cmb.get("chi_matter_min", 0.0))),
        "z_gxy": cfg.get("abacus_galaxy", {}).get("z_range"),
        "nell_file": cmb.get("cmb_noise_nell"),
        "nell_scale": float(cmb.get("cmb_noise_scaling", 1.0)),
        "data": "Abacus map" if cfg.get("observation_mode") == "abacus" else "closure data map",
    }


def weighted(ell, x):
    return float(np.sum((2 * ell + 1) * x))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--config", default=str(ROOT / "configs/inference/abacus/abacus_joint_Nl1p0_chimin700.yaml")
    )
    ap.add_argument("--out_dir", default=str(ROOT / "figures/lensing_fraction"))
    args = ap.parse_args()

    g = run_geometry(yaml.safe_load(open(args.config)))
    cosmo = get_cosmology(**g["cosmo"])
    chi_s = float(jc.background.radial_comoving_distance(cosmo, jnp.atleast_1d(1 / (1 + g["z_source"])))[0])
    ell = np.arange(2, g["lmax"] + 1).astype(float)
    los = LineOfSight(cosmo, ell, chi_s)

    # Check the cumulative table against the direct Limber integral of the model.
    ref = np.asarray(compute_theoretical_cl_kappa(cosmo, jnp.asarray(ell), g["chi_min"], g["chi_boundary"],
                                                  g["z_source"], n_steps=1000))
    dev = np.max(np.abs(los.cl(g["chi_min"], g["chi_boundary"]) / ref - 1))
    print(f"chi_s = {chi_s:.1f} Mpc/h; cumulative table vs compute_theoretical_cl_kappa: max |rel. dev.| {dev:.2e}")

    z_of_chi = lambda c: float(1 / jc.background.a_of_chi(cosmo, jnp.atleast_1d(c))[0] - 1)  # noqa: E731
    tot = los.cl(1.0, chi_s)
    parts = {
        "model shells": los.cl(g["chi_min"], g["chi_boundary"]),
        "data map": los.cl(g["chi_map_min"], g["chi_high_z_max"]),
        "LOS in the covariance": los.cl(g["chi_boundary"], g["chi_high_z_max"]),
        "in neither": los.cl(1.0, g["chi_map_min"]) + los.cl(g["chi_high_z_max"], chi_s),
    }
    nell = None
    if g["nell_file"]:
        tab = np.loadtxt(g["nell_file"])
        nell = g["nell_scale"] * np.interp(ell, tab[:, 0], tab[:, 1])

    bands = [(2, 4), (5, 10), (11, 20), (21, 36), (37, g["lmax"])]
    print(f"Power fraction of C_l(0 -> chi_s), (2l+1)-weighted per band; data-map matter from chi = "
          f"{g['chi_map_min']:g} to {g['chi_high_z_max']:g} Mpc/h:")
    for lo, hi in bands:
        m = (ell >= lo) & (ell <= hi)
        row = "  ".join(f"{k} {weighted(ell[m], v[m]) / weighted(ell[m], tot[m]):.3f}" for k, v in parts.items())
        extra = f"  model/data map {weighted(ell[m], parts['model shells'][m]) / weighted(ell[m], parts['data map'][m]):.3f}"
        if nell is not None:
            cov = weighted(ell[m], nell[m] + parts["LOS in the covariance"][m])
            extra += f"  N_l {weighted(ell[m], nell[m]) / weighted(ell[m], tot[m]):.2f}"
            extra += f"  (N_l + LOS) / (N_l + LOS + neither) {cov / (cov + weighted(ell[m], parts['in neither'][m])):.2f}"
        print(f"  l {lo}-{hi}: {row}{extra}")

    # ---- fraction vs depth -------------------------------------------------------------------
    z_max = np.linspace(0.05, 5.0, 100)
    chi_max = np.asarray(jc.background.radial_comoving_distance(cosmo, jnp.asarray(1 / (1 + z_max))))
    z_min_model, z_box = z_of_chi(g["chi_min"]), z_of_chi(g["chi_boundary"])

    def curve(chi_a):
        frac = np.array([los.cl(chi_a, c) / tot if c > chi_a else np.zeros_like(tot) for c in chi_max])
        mean = np.array([weighted(ell, f * tot) / weighted(ell, tot) for f in frac])
        return mean, frac.min(axis=1), frac.max(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    if g["z_gxy"]:
        ax.axvspan(*g["z_gxy"], color=GREY, alpha=0.15, lw=0)
        ax.text(np.mean(g["z_gxy"]), 1.02, "LRG shell", ha="center", va="bottom", fontsize=9, color="0.3")
    for chi_a, color, ls, label in (
        (1.0, BLUE, "-", r"matter from the observer to $z_\mathrm{max}$"),
        (g["chi_min"], AQUA, "--",
         rf"matter from {g['chi_min']:.0f} Mpc/$h$ (where the map starts) to $z_\mathrm{{max}}$ (model)"),
    ):
        mean, lo, hi = curve(chi_a)
        ok = chi_max > chi_a
        ax.plot(z_max[ok], mean[ok], color=color, ls=ls, lw=2, label=label)
        ax.fill_between(z_max[ok], lo[ok], hi[ok], color=color, alpha=0.15, lw=0)
    f_model = weighted(ell, parts["model shells"]) / weighted(ell, tot)
    ax.axvline(z_box, color=RED, ls="--", lw=1.5,
               label=rf"box edge of the runs, $\chi$ = {g['chi_boundary']:.0f} Mpc/$h$ ($z$ = {z_box:.2f})")
    ax.plot([z_box], [f_model], "o", color=RED, ms=7)
    ax.annotate(f"model holds {f_model:.0%}", (z_box, f_model), xytext=(10, -14),
                textcoords="offset points", fontsize=9, color="0.15")
    f_map = weighted(ell, parts["data map"]) / weighted(ell, tot)
    z_map = z_of_chi(g["chi_high_z_max"])
    ax.plot([z_map], [f_map], "s", color=AQUA, ms=7, mec="k", mew=0.6)
    ax.annotate(f"{g['data']} holds {f_map:.0%}", (z_map, f_map), xytext=(10, 6),
                textcoords="offset points", fontsize=9, color="0.15")
    ax.axvline(z_min_model, color=GREY, ls=":", lw=1)
    ax.text(z_min_model, 0.02, r" map start", fontsize=9, color="0.3", ha="left")
    ax.axhline(1.0, color="0.5", ls=":", lw=1)
    ax.set_xlim(0, z_max[-1])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel(r"depth of the integration $z_\mathrm{max}$")
    ax.set_ylabel(r"fraction of the full CMB-lensing power $C_\ell^{\kappa\kappa}(0 \to \chi_s)$")
    ax.set_title(rf"CMB-lensing power captured, $\ell$ = 2–{g['lmax']} (band: spread over $\ell$)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "lensing_fraction_vs_z.png", dpi=150)
    plt.close(fig)

    # ---- spectra -----------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.loglog(ell, tot, color="k", lw=2, label=r"total, observer $\to$ CMB")
    ax.loglog(ell, parts["model shells"], color=BLUE, lw=2,
              label=rf"model shells, {g['chi_min']:.0f}–{g['chi_boundary']:.0f} Mpc/$h$")
    ax.loglog(ell, parts["data map"], color=AQUA, ls="--", lw=2,
              label=rf"{g['data']}, {g['chi_map_min']:.1f}–{g['chi_high_z_max']:.0f} Mpc/$h$")
    ax.loglog(ell, parts["LOS in the covariance"], color=ORANGE, ls="-.", lw=2,
              label=r"line of sight in the covariance, $C_\ell^\mathrm{LOS}$")
    ax.loglog(ell, parts["in neither"], color=RED, ls=":", lw=2, label=f"in neither (absent from the {g['data']})")
    if nell is not None:
        ax.loglog(ell, nell, color=GREY, lw=1.5, label=rf"$N_\ell$ (ACT DR6 $\times${g['nell_scale']:g})")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell^{\kappa\kappa}$ (Limber, non-linear $P$)")
    ax.set_title("What each part of the line of sight contributes")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(out / "lensing_spectra_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out / 'lensing_fraction_vs_z.png'} and {out / 'lensing_spectra_comparison.png'}")


if __name__ == "__main__":
    main()
