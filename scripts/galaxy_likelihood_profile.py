#!/usr/bin/env python
"""Galaxy likelihood of the Abacus counts at the true initial conditions, biases profiled
(pipeline §7.14, §7.15).

The true ICs of the config's `abacus_ic` (cropped to the model box when it is the central cube of the
simulation, §4) go through the forward model; the counts likelihood of §3.1 is minimised over the
biases (`b1`, `b2`, `b_{s²}`, `b∇²`, `bnpar`, and `b1_alpha` unless `--alpha0`) by L-BFGS, as a function
of the field amplitude A at the fiducial cosmology (`--mode amplitude`: A·init_mesh, i.e. `sigma8` =
A × the fiducial for the likelihood) or of `Omega_m` at the true field (`--mode omega_m`). A model
consistent with the data has its minimum at A = 1 and at the true `Omega_m`.

`--mode newton` finds the joint minimum in (A, biases) directly: the biases by L-BFGS at A = 1, then
Gauss–Newton steps in A and the biases (the third-order `b3`, `b_{δs²}`, `b_{s³}` too when the config has
`bias_order: 3`), each with the model's Jacobian by forward-mode differentiation, until the step in A is
below 1e-4; σ_A from Δχ² = 1; then the noise law of the configs (`s_e`, `s_ed`) fitted to the
residual at the minimum. It is the check of a grid or a selection before a run (minutes on a CPU
node; pipeline §2.5, §7.16): A_min within about σ_A of 1.

    python scripts/galaxy_likelihood_profile.py --config configs/inference/abacus/abacus_gxyonly_cosmo_box5000.yaml \\
        --mode amplitude --values 1.0 0.9 0.95 1.05
    python scripts/galaxy_likelihood_profile.py --config configs/inference/abacus/abacus_gxyonly_cosmo_box5000_m100_ph202.yaml \\
        --mode newton

Prints χ² per value (split between the survey cells within 300 Mpc/h of a box face and the others)
and the profiled biases; writes `figures/galaxy_likelihood_profile/<config stem>_<mode>.json`. Needs a
compute node (the randoms of a new geometry take ~30–40 min; then cached in `data/cache/`).
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from scipy.optimize import minimize

from desi_cmb_fli.bricks import band_limit, get_cosmology
from desi_cmb_fli.cmb_lensing import load_abacus_galaxy_observation, load_abacus_ic_truth
from desi_cmb_fli.model import get_model_from_config

NAMES = ["b1", "b2", "bs2", "bn2", "bnpar", "b1_alpha"]
SCALE = np.array([0.05, 0.3, 0.3, 30.0, 30.0, 0.3])
X0 = np.array([1.15, 0.6, -0.4, 0.0, -45.0, 1.0])
OUT = Path(__file__).resolve().parents[1] / "figures" / "galaxy_likelihood_profile"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["amplitude", "omega_m", "newton"], default="amplitude")
    ap.add_argument(
        "--values", type=float, nargs="+", default=[1.0], help="A or Omega_m values (not newton)"
    )
    ap.add_argument(
        "--alpha0", action="store_true", help="z-independent bias (b1_alpha fixed at 0)"
    )
    ap.add_argument(
        "--edge", type=float, default=300.0, help="Mpc/h from a box face for the edge split"
    )
    ap.add_argument("--box", type=float, default=None, help="cubic box side (Mpc/h), with --mesh")
    ap.add_argument("--mesh", type=int, default=None, help="final mesh cells per side, with --box")
    ap.add_argument("--particles", type=int, default=None, help="particles per side")
    ap.add_argument("--init_oversamp", type=float, default=None)
    ap.add_argument("--paint_oversamp", type=float, default=None)
    ap.add_argument("--tag", default="", help="suffix of the output file")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    m = cfg["model"]
    m["galaxies_enabled"] = True
    cfg.get("cmb_lensing", {})["enabled"] = False
    if args.box is not None:
        m["box_shape"] = [args.box] * 3
        m["cell_size"] = args.box / args.mesh
    n_final = round(float(m["box_shape"][0]) / float(m["cell_size"]))
    if args.particles is not None:
        m["evol_oversamp"] = m["ptcl_oversamp"] = args.particles / n_final
    for key in ("init_oversamp", "paint_oversamp"):
        if getattr(args, key) is not None:
            m[key] = getattr(args, key)
    model, _ = get_model_from_config(cfg)
    print(
        "grids: final",
        tuple(int(x) for x in model.mesh_shape),
        "init",
        tuple(int(x) for x in model.init_shape),
        "particles",
        tuple(int(x) for x in model.evol_shape),
        "paint",
        tuple(int(x) for x in model.paint_shape),
        flush=True,
    )
    t0 = time.time()
    tr = load_abacus_galaxy_observation(abacus_gxy_cfg=cfg["abacus_galaxy"], model=model)
    print(f"galaxies loaded in {time.time() - t0:.0f} s", flush=True)
    obs, mask = jnp.asarray(tr["obs"]), jnp.asarray(tr["gxy_occ_mask3d"])
    selec_paint, nbar = (
        jnp.asarray(model.selec_paint),
        model.gxy_count * jnp.asarray(model.selec_mesh),
    )
    init_true = jnp.asarray(load_abacus_ic_truth(cfg["abacus_ic"], model)["init_mesh"])
    truth = {**cfg.get("truth_params", {}), **cfg.get("abacus_truth_params", {})}
    om0, s80 = float(truth["Omega_m"]), float(truth["sigma8"])

    box = float(model.box_shape[0])
    n = int(model.mesh_shape[0])
    x = (np.arange(n) + 0.5) * box / n - box / 2
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    edge = jnp.asarray(np.max(np.abs(np.stack([X, Y, Z])), 0) > box / 2 - args.edge)
    free = np.arange(5) if args.alpha0 else np.arange(6)

    def r2(theta, om, amp):
        b = dict(zip(NAMES, theta, strict=True))
        f = model.evolve(
            (get_cosmology(Omega_m=om, sigma8=s80), b, {"init_mesh": amp * init_true}, {"fNL": 0.0})
        )
        intens = band_limit(f["gxy_paint"] * selec_paint, model._sim_mesh)
        return jnp.where(
            mask, (obs - model.gxy_count * intens) ** 2 / jnp.maximum(nbar, 1e-10), 0.0
        )

    vg = jax.jit(jax.value_and_grad(lambda th, om, amp: r2(th, om, amp).sum()))
    parts = jax.jit(
        lambda th, om, amp: jnp.stack(
            [
                jnp.where(edge, r2(th, om, amp), 0.0).sum(),
                jnp.where(edge, 0.0, r2(th, om, amp)).sum(),
            ]
        )
    )
    x0 = X0.copy()
    if args.alpha0:
        x0[5] = 0.0
    if args.mode == "newton":
        newton(
            args,
            model,
            obs,
            mask,
            nbar,
            selec_paint,
            init_true,
            get_cosmology(Omega_m=om0, sigma8=s80),
            vg,
            x0,
            free,
            t0,
        )
        return
    res = []
    for v in args.values:
        om, amp = (om0, v) if args.mode == "amplitude" else (v, 1.0)

        def fun(u, om=om, amp=amp):
            th = x0.copy()
            th[free] = u * SCALE[free]
            val, g = vg(jnp.asarray(th, jnp.float32), om, amp)
            return float(val), np.asarray(g, float)[free] * SCALE[free]

        r = minimize(
            fun,
            x0[free] / SCALE[free],
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 200, "gtol": 1e-3},
        )
        th = x0.copy()
        th[free] = r.x * SCALE[free]
        pe = np.asarray(parts(jnp.asarray(th, jnp.float32), om, amp))
        res.append(
            {
                "value": v,
                "chi2": float(r.fun),
                "chi2_edge": float(pe[0]),
                "chi2_rest": float(pe[1]),
                "biases": dict(zip(NAMES, map(float, th), strict=True)),
            }
        )
        print(
            f"{args.mode} {v}: chi2 {r.fun:.1f} (within {args.edge:.0f} Mpc/h of a face {pe[0]:.1f}) "
            f"{ {k: round(b, 3) for k, b in res[-1]['biases'].items()} }  t {time.time() - t0:.0f}s",
            flush=True,
        )
    OUT.mkdir(parents=True, exist_ok=True)
    out = (
        OUT
        / f"{Path(args.config).stem}_{args.mode}{'_alpha0' if args.alpha0 else ''}{args.tag}.json"
    )
    out.write_text(
        json.dumps(
            {
                "config": args.config,
                "mode": args.mode,
                "cells": int(mask.sum()),
                "cells_near_face": int((mask & edge).sum()),
                "profile": res,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"Saved {out}")


def newton(
    args, model, obs, mask, nbar, selec_paint, init_true, cosmo, vg, x0, free, t0, max_iter=8
):
    """Joint minimum in (A, biases): L-BFGS on the biases at A = 1, then Gauss–Newton steps."""
    om = float(cosmo.Omega_m)

    def fun(u):
        th = x0.copy()
        th[free] = u * SCALE[free]
        val, g = vg(jnp.asarray(th, jnp.float32), om, 1.0)
        return float(val), np.asarray(g, float)[free] * SCALE[free]

    r = minimize(
        fun,
        x0[free] / SCALE[free],
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 200, "gtol": 1e-3},
    )
    start = x0.copy()
    start[free] = r.x * SCALE[free]
    names = ["A"] + [NAMES[i] for i in free]
    third = getattr(model, "bias_order", 2) == 3
    if third:
        names += ["b3", "bds2", "bs3"]
    fixed = {NAMES[i]: float(x0[i]) for i in range(len(NAMES)) if i not in free}
    sd = jnp.sqrt(jnp.maximum(nbar, 1e-10))

    def white(th):
        b = {**fixed, **dict(zip(names[1:], th[1:], strict=True))}
        f = model.evolve((cosmo, b, {"init_mesh": th[0] * init_true}, {"fNL": 0.0}))
        m = model.gxy_count * band_limit(f["gxy_paint"] * selec_paint, model._sim_mesh)
        return jnp.where(mask, m / sd, 0.0)

    wj = jax.jit(white)
    tangent = jax.jit(lambda th, v: jax.jvp(white, (th,), (v,))[1])
    ow = jnp.where(mask, obs / sd, 0.0)
    th = np.array([1.0] + [float(start[i]) for i in free] + [0.0] * (3 * third))
    eye = np.eye(len(names), dtype=np.float32)
    hist = []
    for it in range(max_iter):
        thj = jnp.asarray(th, jnp.float32)
        res = np.asarray(ow - wj(thj), np.float64).ravel()
        jac = np.stack([np.asarray(tangent(thj, jnp.asarray(e)), np.float64).ravel() for e in eye])
        fish = jac @ jac.T
        step = np.linalg.solve(fish, jac @ res)
        sigma = np.sqrt(np.diag(np.linalg.inv(fish)))
        hist.append(
            {
                "theta": dict(zip(names, map(float, th), strict=True)),
                "chi2": float(res @ res),
                "sigma": dict(zip(names, map(float, sigma), strict=True)),
                "step_A": float(step[0]),
            }
        )
        print(
            f"newton it {it}: chi2 {res @ res:.1f}; "
            + ", ".join(f"{n} {v:.4f}" for n, v in zip(names, th, strict=True))
            + f"; step A {step[0]:+.5f} (sigma_A {sigma[0]:.4f})  t {time.time() - t0:.0f}s",
            flush=True,
        )
        th = th + step
        if it > 0 and abs(step[0]) < 1e-4:
            break
    last = hist[-1]
    print(f"A_min {last['theta']['A'] + last['step_A']:.4f} ± {last['sigma']['A']:.4f}", flush=True)
    # The noise law of the configs, std = s_e (1 + s_ed max(delta_m, -1)) sqrt(nbar S), fitted to the
    # residual at the minimum by maximum likelihood over the survey cells (pipeline §3.1).
    thj = jnp.asarray(th, jnp.float32)
    mdl = np.asarray(wj(thj) * sd)[np.asarray(mask)]
    nb = np.asarray(nbar)[np.asarray(mask)]
    dev = (np.asarray(obs)[np.asarray(mask)] - mdl) / np.sqrt(nb)
    dm = np.maximum(mdl / nb - 1, -1)

    def nll(p):
        s = p[0] * (1 + p[1] * dm)
        return np.sum((dev / s) ** 2 + 2 * np.log(np.abs(s)))

    noise = minimize(
        nll, [1.0, 0.2], method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-3}
    ).x
    print(f"noise fit at the minimum: s_e {noise[0]:.4f}, s_ed {noise[1]:.4f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{Path(args.config).stem}_newton{'_alpha0' if args.alpha0 else ''}{args.tag}.json"
    out.write_text(
        json.dumps(
            {
                "config": args.config,
                "mode": "newton",
                "cells": int(mask.sum()),
                "A_min": last["theta"]["A"] + last["step_A"],
                "sigma_A": last["sigma"]["A"],
                "noise_fit": {"s_e": float(noise[0]), "s_ed": float(noise[1])},
                "iterations": hist,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
