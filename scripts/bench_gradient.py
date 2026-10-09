#!/usr/bin/env python
"""Cost of one log-density gradient: time and device peak memory (pipeline §7.15).

The config's closure data (generated from `truth_params`) condition the model, which is then
differentiated at that point: one jitted value_and_grad, compiled once and timed over five calls.
`--particles` and `--box --mesh` override the particle grid and the geometry as in
`kappa_lattice_template.py`. On a GPU node, one process per device:

    CUDA_VISIBLE_DEVICES=0 python scripts/bench_gradient.py --config configs/inference/scan/closure_d1p00_joint_cosmo.yaml --particles 240
"""

import argparse
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from kappa_lattice_template import build_model
from numpyro.handlers import seed, trace

from desi_cmb_fli.utils import restore_model_state_from_truth


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--particles", type=int, default=None)
    ap.add_argument("--box", type=float, default=None)
    ap.add_argument("--mesh", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()
    cfg_path = args.config
    model, cfg = build_model(cfg_path, args.box, args.mesh, args.particles)
    cfg["closure_geometry_from_abacus"] = False
    truth = model.predict(
        samples=cfg["truth_params"],
        hide_base=False,
        hide_samp=False,
        hide_det=False,
        frombase=True,
        rng=jr.key(0),
    )
    restore_model_state_from_truth(model, truth)
    cond = {}
    if model.galaxies_enabled:
        cond["obs"] = truth["obs"]
    if model.cmb_enabled:
        cond["kappa_obs"] = truth["kappa_obs"]
    model.condition(cond)
    fixed = {
        p: float(cfg["truth_params"].get(p, model.loc_fid[p]))
        for p in cfg["mcmc"].get("fixed_params", [])
    }
    if fixed:
        model.condition(fixed, frombase=True)
    model.block()
    tr = trace(seed(model.model, 0)).get_trace()
    params = {
        k: jnp.asarray(v["value"])
        for k, v in tr.items()
        if v["type"] == "sample" and not v.get("is_observed", False)
    }
    f = jax.jit(jax.value_and_grad(model.logpdf))
    t0 = time.time()
    jax.block_until_ready(f(params)[1])
    compile_s = time.time() - t0
    ts = []
    for _ in range(args.repeat):
        t0 = time.time()
        jax.block_until_ready(f(params)[1])
        ts.append(time.time() - t0)
    stats = jax.devices()[0].memory_stats() or {}
    peak = stats.get("peak_bytes_in_use", float("nan")) / 2**30
    print(
        f"{cfg_path}: particles {tuple(int(x) for x in model.evol_shape)}, box {float(model.box_shape[0]):.0f}, "
        f"gradient {np.median(ts):.3f} s (compile {compile_s:.0f} s), peak {peak:.1f} GiB, latents {sorted(params)}"
    )


if __name__ == "__main__":
    main()
