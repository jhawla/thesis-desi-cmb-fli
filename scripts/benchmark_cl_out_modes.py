#!/usr/bin/env python
"""The out-of-model term C_l^out under `cl_out_mode: fixed` against `exact`, over a grid of
(Omega_m, sigma8), at the geometry of a run config (docs/pipeline.md 3.2); and, at the fiducial, what
the linear spectrum of the term leaves out (halofit against linear, jax_cosmo, same kernel).

    python scripts/benchmark_cl_out_modes.py --config configs/inference/abacus/abacus_kappaonly_so_cosmo_box5000.yaml

The term is the model's own: the shells, their per-shell cut, the window and the exact-over-Limber
ratio of the config. Per point: the largest relative error on C_l^out over the bands, and the
expected error on ln L, KL = 1/2 sum_l (2l + 1) f_sky [r - 1 - ln r] with
r = (N_l + C_l^out exact) / (N_l + C_l^out fixed), in nats. `fixed` moves the posterior only where
its KL is not small against 1/2 across the posterior's range.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from desi_cmb_fli import utils
from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import compute_cl_out, lensing_kernel, shell_profiles
from desi_cmb_fli.model import get_model_from_config

BANDS = [(2, 4), (4, 6), (6, 11), (11, 20), (20, 36), (36, 65)]


def jax_cosmo_term(cosmo, ell, shells, chi_max, power, n_per_shell=40, n_beyond=200):
    """The term's Limber power on the same kernel with a jax_cosmo spectrum and background."""
    r = np.asarray(shells["r_shells"], dtype=float)
    h = np.broadcast_to(np.asarray(shells["d_r"], dtype=float), r.shape)
    chi_in, p = shell_profiles(r, h, shells["shell_weights"], n_per_shell)
    keep = chi_in > 1.0
    chi_in, p = chi_in[keep], p[:, keep]
    vol = np.trapezoid(p * chi_in**2, chi_in, axis=1)
    t = np.linspace(0.0, 1.0, n_beyond)[1:]
    chi = jnp.asarray(np.concatenate([chi_in, chi_in[-1] + max(chi_max - chi_in[-1], 0.0) * t]))
    p = np.concatenate([p, np.zeros((r.size, t.size))], axis=1)
    chi_s = jc.background.radial_comoving_distance(cosmo, 1.0 / (1.0 + shells["z_source"]))[0]
    w = lensing_kernel(cosmo, jnp.asarray(r), jnp.asarray(shells["a_shells"]), chi_s)
    q = (jnp.asarray(h) * w)[:, None] * jnp.asarray(p) * chi**2 / jnp.asarray(vol)[:, None]
    a = jc.background.a_of_chi(cosmo, chi)
    w_map = lensing_kernel(cosmo, chi, a, chi_s)
    ell_idx = np.asarray(ell).astype(int)
    held = (np.ones((r.size, ell_idx.size)) if shells["taper"] is None
            else np.asarray(shells["taper"], dtype=float)[:, ell_idx])
    def per_ell(ell_val, held_l):
        kern = w_map - (held_l[:, None] * q).sum(0)
        pk = jax.vmap(lambda ki, ai: jnp.squeeze(power(cosmo, ki, ai)))(
            (ell_val + 0.5) / chi, a)
        return jnp.trapezoid(kern**2 / chi**2 * pk, chi)

    return np.asarray(jax.vmap(per_ell)(jnp.asarray(ell_idx, dtype=float), jnp.asarray(held.T)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="run config with CMB lensing")
    p.add_argument("--omega_m", type=float, nargs="+",
                   default=[0.22, 0.25, 0.28, 0.315192, 0.34, 0.37, 0.40, 0.43, 0.46, 0.50])
    p.add_argument("--sigma8", type=float, nargs="+", default=[0.73, 0.77, 0.811355, 0.85, 0.89])
    p.add_argument("--out", default=None, help="output stem (default figures/cl_out_modes/<config>)")
    args = p.parse_args()

    cfg = utils.yload(args.config)
    cfg["cmb_lensing"].pop("high_z_mode", None)
    cfg["cmb_lensing"]["cl_out_mode"] = "exact"
    model, _ = get_model_from_config(cfg)

    ell = np.asarray(model.ell_1d, dtype=float)
    noise = np.asarray(model.nell_1d)
    f_sky = float(np.mean(np.asarray(model.cmb_mask)))
    chi_max = float(model.chi_high_z_max)
    window2 = model.cmb_los_window2 * model.cmb_los_limber
    exact_fn = jax.jit(lambda om, s8: compute_cl_out(
        get_cosmology(Omega_m=om, sigma8=s8), model.ell_1d, chi_max, model.cmb_shells,
        window2=window2))
    fixed = np.asarray(model.cl_out_cached)
    sel = ell >= 2

    def kl(c_exact, c_other):
        r = (noise + c_exact) / (noise + c_other)
        return float(0.5 * np.sum(((2 * ell + 1) * f_sky * (r - 1 - np.log(r)))[sel]))

    def band_err(c_exact, c_other):
        return float(max(abs(c_other[(ell >= a) & (ell < b)].sum() / c_exact[(ell >= a) & (ell < b)].sum() - 1)
                         for a, b in BANDS))

    rows, curves, exacts = [], {}, []
    for om in args.omega_m:
        for s8 in args.sigma8:
            exact = np.asarray(exact_fn(om, s8))
            row = {"Omega_m": om, "sigma8": s8,
                   "out_over_noise": float(exact[sel].sum() / noise[sel].sum()),
                   "fixed_band_err": band_err(exact, fixed), "fixed_kl": kl(exact, fixed)}
            rows.append(row)
            curves[(om, s8)] = fixed / exact
            exacts.append(exact)
            print(f"Omega_m {om:.3f} sigma8 {s8:.3f}: C_out/N {row['out_over_noise']:.2f}; "
                  f"fixed: band {row['fixed_band_err']:.3f}, KL {row['fixed_kl']:.3f} nats", flush=True)

    # what the linear spectrum leaves out: halofit against linear at the fiducial
    fid = get_cosmology(**{k: float(model.loc_fid[k]) for k in ("Omega_m", "sigma8")})
    nl, lin = (jax_cosmo_term(fid, ell, model.cmb_shells, chi_max, pw) * np.asarray(window2)
               for pw in (jc.power.nonlinear_matter_power, jc.power.linear_matter_power))
    nonlinear = {"kl": kl(nl, lin), "max_rel": float(np.max(np.abs(lin[sel] / nl[sel] - 1)))}
    print(f"fiducial, halofit against linear: KL {nonlinear['kl']:.4f} nats, "
          f"max |lin/halofit - 1| {nonlinear['max_rel']:.4f}")

    out = Path(args.out) if args.out else Path("figures/cl_out_modes") / Path(args.config).stem
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(
        {"config": args.config, "fiducial": {k: float(model.loc_fid[k]) for k in ("Omega_m", "sigma8")},
         "f_sky": f_sky, "bands": BANDS, "nonlinear_at_fiducial": nonlinear, "points": rows},
        indent=1))
    np.savez(out.with_suffix(".npz"), ell=ell, noise=noise, fixed=fixed, exact=np.array(exacts),
             omega_m=np.repeat(args.omega_m, len(args.sigma8)),
             sigma8=np.tile(args.sigma8, len(args.omega_m)))

    oms, s8s = np.array(args.omega_m), np.array(args.sigma8)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    grid = np.array([[next(r["fixed_kl"] for r in rows if r["Omega_m"] == om and r["sigma8"] == s8)
                      for om in oms] for s8 in s8s])
    im = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto",
                   extent=[-0.5, len(oms) - 0.5, -0.5, len(s8s) - 0.5])
    for i in range(len(s8s)):
        for j in range(len(oms)):
            v = grid[i, j]
            ax.text(j, i, "0" if v < 1e-6 else (f"{v:.0f}" if v >= 10 else f"{v:.1f}"), ha="center",
                    va="center", fontsize=8, color="w" if v < 0.5 * grid.max() else "k")
    ax.set_xticks(range(len(oms)), [f"{x:.3g}" for x in oms], fontsize=7)
    ax.set_yticks(range(len(s8s)), [f"{x:.3g}" for x in s8s])
    ax.set_xlabel(r"$\Omega_m$")
    ax.set_ylabel(r"$\sigma_8$")
    ax.set_title("fixed against exact: expected error on ln L [nats]", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[1]
    for (om, s8), f in curves.items():
        if om in (oms.min(), oms.max()) and s8 in (s8s.min(), s8s.max()):
            ax.plot(ell[sel], f[sel], label=rf"$\Omega_m$ {om:.2f}, $\sigma_8$ {s8:.2f}")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell^{\rm out}$ fixed / exact")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_title("Corners of the grid", fontsize=10)
    fig.suptitle(f"Out-of-model term C_l^out: fixed against exact — {Path(args.config).name}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"Saved {out.with_suffix('.png')} and .json")


if __name__ == "__main__":
    main()
