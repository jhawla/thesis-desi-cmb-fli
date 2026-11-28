"""How much of the CMB-lensing convergence the runs hold, per multipole and against depth.

One figure, `lensing_budget.png` (and its `.npz`), over the likelihood's band l = 2 ... 2 nside:

- left: the observed variance per multipole, C_l of the data map (Limber, non-linear P(k), the
  config's cosmology) plus N_l, split into the matter the model holds, the parts the out-of-model
  term C_l^out holds (what the per-shell multipole cut removes, the end ramps of the shells, the matter
  beyond the box) and the noise. The split of the signal comes from the kernels of the term
  (`cmb_lensing.compute_cl_outside_model`, linear Limber on the model's spectrum, cross terms in
  the part they are added with): the term with the cut, without it, and without the matter beyond the box.
- right: power fraction C_l(chi_a -> chi(z_max)) / C_l(0 -> chi_s) against z_max (band: spread over
  l) from the observer; markers at the box edge and at the end of the map's matter.

The Limber integrand depends on the bounds only through the integration range, so it is tabulated
once on a fine chi grid and every range is a cumulative sum of it (checked against
cmb_lensing.compute_theoretical_cl_kappa at start). Prints the shares per band.

Usage: python scripts/plot_lensing_fraction.py [--config configs/inference/abacus/abacus_joint_so_cosmo_box5000_m100_ph202.yaml]
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np
import yaml

from desi_cmb_fli import utils
from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import (
    compute_cl_outside_model,
    compute_theoretical_cl_kappa,
    lensing_kernel,
    nonlinear_power,
)
from desi_cmb_fli.model import get_model_from_config

ROOT = Path(__file__).resolve().parents[1]
BLUE, AQUA, ORANGE, RED, GREY, INK = "#2a78d6", "#1baf7a", "#eb6834", "#e34948", "#8a8984", "#1f1f1e"


def limber_integrand(cosmo, ell, chi, chi_s):
    """(W^2 / chi^2) P_NL((l + 1/2)/chi, a(chi)), shape (n_ell, n_chi)."""
    a = jc.background.a_of_chi(cosmo, chi)
    w2 = (lensing_kernel(cosmo, chi, a, chi_s) / chi) ** 2
    pk_fn = nonlinear_power(cosmo)

    def one(ell_val):
        k = (ell_val + 0.5) / chi
        pk = jax.vmap(pk_fn)(k, a)
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
        "--config", default=str(ROOT / "configs/inference/abacus/abacus_joint_so_cosmo_box5000_m100_ph202.yaml")
    )
    ap.add_argument("--out_dir", default=str(ROOT / "figures/lensing_fraction"))
    args = ap.parse_args()

    g = run_geometry(yaml.safe_load(open(args.config)))
    cfg = utils.yload(args.config)
    cfg["model"]["galaxies_enabled"] = False  # the kappa likelihood's attributes only
    model, _ = get_model_from_config(cfg)
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
    c_map = los.cl(g["chi_map_min"], g["chi_high_z_max"])
    nell = None
    if g["nell_file"]:
        tab = np.loadtxt(g["nell_file"])
        nell = g["nell_scale"] * np.interp(ell, tab[:, 0], tab[:, 1])

    # Shares of the data map's signal, from the kernels of the out-of-model term (linear Limber,
    # the model's spectrum): map W (no shell held), left out with the per-shell cut (the term), left
    # out without the cut, and the same without the matter beyond the box.
    sh = dict(model.cmb_shells)
    li = ell.astype(int)

    def left(taper, chi_max):
        return np.asarray(compute_cl_outside_model(cosmo, li, **{**sh, "taper": taper}, chi_max=chi_max))

    lin_map = left(np.zeros_like(np.asarray(sh["taper"])), g["chi_high_z_max"])
    lin_out = left(sh["taper"], g["chi_high_z_max"])
    lin_uncut = left(None, g["chi_high_z_max"])
    lin_ends = left(None, 0.0)
    shares = {
        "held by the model": 1 - lin_out / lin_map,
        "per-shell cut": (lin_out - lin_uncut) / lin_map,
        "shell ends": lin_ends / lin_map,
        "beyond the box": (lin_uncut - lin_ends) / lin_map,
    }
    var = c_map + (nell if nell is not None else 0.0)
    frac = {k: v * c_map / var for k, v in shares.items()}
    if nell is not None:
        frac["noise $N_\\ell$"] = nell / var

    bands = [(2, 10), (11, 64), (65, 128), (129, 192), (193, g["lmax"])]
    print("Share of the data map's signal, (2l+1)-weighted per band (linear Limber kernels); then of the"
          " observed variance C_map + N_l:")
    for lo, hi in bands + [(2, g["lmax"])]:
        m = (ell >= lo) & (ell <= hi)
        sig = "  ".join(f"{k} {weighted(ell[m], v[m] * c_map[m]) / weighted(ell[m], c_map[m]):.3f}"
                        for k, v in shares.items())
        obs = "  ".join(f"{k} {weighted(ell[m], v[m] * var[m]) / weighted(ell[m], var[m]):.3f}"
                        for k, v in frac.items())
        print(f"  l {lo}-{hi}: signal: {sig}\n           observed: {obs}")
    f_full = weighted(ell, los.cl(g["chi_min"], g["chi_boundary"])) / weighted(ell, tot)
    f_map = weighted(ell, c_map) / weighted(ell, tot)
    chi_of_z = lambda z: float(jc.background.radial_comoving_distance(cosmo, jnp.atleast_1d(1 / (1 + z)))[0])  # noqa: E731
    f_obs_box = weighted(ell, los.cl(1.0, g["chi_boundary"])) / weighted(ell, tot)
    f_lrg = (weighted(ell, los.cl(*map(chi_of_z, g["z_gxy"]))) / weighted(ell, tot)) if g["z_gxy"] else np.nan
    print(f"Of the full lensing power (observer to CMB), l 2-{g['lmax']}: the box's matter "
          f"{f_full:.3f} (from the observer {f_obs_box:.3f}), the data map's {f_map:.3f}, the LRG range {f_lrg:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    colors = {"held by the model": BLUE, "per-shell cut": ORANGE, "shell ends": "#f2b38f",
              "beyond the box": RED, "noise $N_\\ell$": "0.82"}
    ax.stackplot(ell, *frac.values(), colors=[colors[k] for k in frac], labels=list(frac), lw=0)
    ax.set_xscale("log")
    ax.set(xlim=(2, g["lmax"]), ylim=(0, 1), xlabel=r"$\ell$",
           ylabel=r"share of $C_\ell^{\rm map} + N_\ell$")
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h[::-1], lab[::-1], loc="lower left", fontsize=8, frameon=True, framealpha=0.85)

    ax = axes[1]
    z_max = np.linspace(0.05, 5.0, 100)
    chi_max = np.asarray(jc.background.radial_comoving_distance(cosmo, jnp.asarray(1 / (1 + z_max))))
    if g["z_gxy"]:
        ax.axvspan(*g["z_gxy"], color=GREY, alpha=0.15, lw=0, label="LRGs")
    fr = np.array([los.cl(1.0, c) / tot for c in chi_max])
    mean = np.array([weighted(ell, f * tot) / weighted(ell, tot) for f in fr])
    ax.plot(z_max, mean, color=BLUE, lw=2, label="matter to $z_{\\rm max}$")
    ax.fill_between(z_max, fr.min(axis=1), fr.max(axis=1), color=BLUE, alpha=0.12, lw=0)
    z_box, z_map = z_of_chi(g["chi_boundary"]), z_of_chi(g["chi_high_z_max"])
    ax.axvline(z_box, color=RED, ls="--", lw=1.4, label=f"box edge ({f_full:.0%})")
    ax.axvline(z_map, color=INK, ls="--", lw=1.4, label=f"end of the map ({f_map:.0%})")
    ax.set(xlim=(0, z_max[-1]), ylim=(0, 1), xlabel=r"depth $z_{\rm max}$",
           ylabel=r"share of $C_\ell^{\kappa\kappa}$ (observer to CMB)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    for a in axes:
        a.tick_params(labelsize=9)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "lensing_budget.png", dpi=200)
    plt.close(fig)
    np.savez(out / "lensing_budget.npz", ell=ell, c_map=c_map, nell=np.zeros_like(ell) if nell is None else nell,
             tot=tot, **{k.split(" $")[0].replace(" ", "_"): v for k, v in frac.items()})
    print(f"Saved {out / 'lensing_budget.png'} and .npz")

if __name__ == "__main__":
    main()
