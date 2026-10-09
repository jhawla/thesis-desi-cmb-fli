#!/usr/bin/env python
"""Galaxy likelihood of the Abacus counts at the true initial conditions, biases profiled
(pipeline §7.14, §7.15).

The true ICs of the config's `abacus_ic` (cropped to the model box when it is the central cube of the
simulation, §4) go through the forward model; the counts likelihood of §3.1 is minimised over the
biases (`b1`, `b2`, `b_{s²}`, `b∇²`, `bnpar`, and `b1_alpha` unless `--alpha0`) by L-BFGS, as a function
of the field amplitude A at the fiducial cosmology (`--mode amplitude`: A·init_mesh, i.e. `sigma8` =
A × the fiducial for the likelihood) or of `Omega_m` at the true field (`--mode omega_m`). A model
consistent with the data has its minimum at A = 1 and at the true `Omega_m`.

    python scripts/galaxy_likelihood_profile.py --config configs/inference/abacus/abacus_gxyonly_cosmo_box5000.yaml \\
        --mode amplitude --values 1.0 0.9 0.95 1.05

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
    ap.add_argument("--mode", choices=["amplitude", "omega_m"], default="amplitude")
    ap.add_argument("--values", type=float, nargs="+", required=True)
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


if __name__ == "__main__":
    main()
