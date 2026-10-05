#!/usr/bin/env python
"""How stiff does CMB lensing make the field posterior, per radial set-up of the Born projector?

The MCLMC step size is capped by the stiffest direction of the log-density in the sampled basis,
roughly as 1/sqrt(lambda_max) of the Hessian of -log p. This script measures lambda_max with respect
to the field (`init_mesh_`, the basis the sampler moves in) by power iteration on exact
Hessian-vector products (reverse-over-reverse), for several `shell_weights:shell_kmax` variants. The
scalars are held fixed as in the field warmup, and galaxies are off so the likelihood is κ alone:
in the whitened basis the prior alone gives lambda = 1 (checked by a variant with the κ noise
scaled by 1e8). The truth is a closure draw at the configuration's geometry, and the Hessian is
taken there.

For the top eigenvector it also reports where it lives: the fraction of its variance, in the real
space initial field, within given distances of the observer.

    python scripts/kappa_stiffness.py --config configs/inference/abacus/abacus_joint_Nl1p0.yaml \\
        --variants linear:auto linear:0 nearest:auto linear:auto:1e8
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
from jax import numpy as jnp  # noqa: E402
from jax import random as jr  # noqa: E402

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from desi_cmb_fli.model import get_model_from_config  # noqa: E402
from desi_cmb_fli.validation import conditioning_params  # noqa: E402

RADII = (350.0, 700.0, 1100.0)


def build_model(cfg, weights, shell_kmax, noise_scaling):
    cfg = copy.deepcopy(cfg)
    cfg["model"]["galaxies_enabled"] = False
    cmb = cfg["cmb_lensing"]
    cmb.update(enabled=True, shell_weights=weights)
    if shell_kmax is not None:
        cmb["shell_kmax"] = float(shell_kmax)
    cmb["cmb_noise_scaling"] = float(noise_scaling)
    model, _ = get_model_from_config(cfg)
    return model


def radial_fractions(model, scalars_lat, x0, v):
    """Fraction of the variance of the real-space field perturbation along v within RADII."""

    def field(x):
        return model.reparam({**scalars_lat, "init_mesh_": x})["init_mesh"]

    # The map is linear in init_mesh_ at fixed scalars (a per-mode scale), so a difference is exact.
    dk = field(x0 + v) - field(x0)
    shape = tuple(int(s) for s in model.init_shape)
    d = np.asarray(jnp.fft.irfftn(dk, s=shape))
    cell = np.asarray(model.box_shape, dtype=float) / np.asarray(shape)
    grids = np.meshgrid(*[np.arange(n) * c for n, c in zip(shape, cell, strict=True)],
                        indexing="ij")
    r = np.sqrt(sum((g - o) ** 2 for g, o in zip(grids, model.observer_position, strict=True)))
    tot = float(np.sum(d**2))
    return {f"<{rad:g}": float(np.sum(d[r < rad] ** 2) / tot) for rad in RADII}, {
        f"<{rad:g}": float(np.mean(r < rad)) for rad in RADII}


def stiffness(model, scalars, key, n_iter):
    model.reset()
    truth = model.predict(samples=scalars, hide_base=False, hide_samp=False, hide_det=False,
                          frombase=True, rng=key)
    x0 = jnp.asarray(truth["init_mesh_"])
    model.reset()
    model.condition({"kappa_obs": truth["kappa_obs"]} | scalars, frombase=True)
    model.block()

    grad = jax.grad(lambda x: model.logpdf({"init_mesh_": x}))
    # Reverse-over-reverse: the model holds custom_vjp functions, which forward mode cannot enter.
    hvp = jax.jit(lambda v: -jax.grad(lambda x: jnp.vdot(grad(x), v))(x0))

    v = jr.normal(jr.key(1), x0.shape, dtype=x0.dtype)
    v = v / jnp.linalg.norm(v)
    history = []
    jax.block_until_ready(hvp(v))  # compile outside the timing
    t0 = time.perf_counter()
    for it in range(n_iter):
        w = hvp(v)
        lam = float(jnp.vdot(v, w))
        v = w / jnp.linalg.norm(w)
        history.append(lam)
        if (it + 1) % 10 == 0:
            print(f"    iter {it + 1:3d}: lambda = {lam:.4g}", flush=True)
    cost = {"s_per_hvp": (time.perf_counter() - t0) / n_iter}
    stats = jax.devices()[0].memory_stats() or {}
    if "peak_bytes_in_use" in stats:
        cost["peak_gb"] = stats["peak_bytes_in_use"] / 1e9
    scalars_lat = dict(model.reparam(scalars, inv=True))
    return history, radial_fractions(model, scalars_lat, x0, v), cost


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/inference/abacus/abacus_joint_Nl1p0.yaml")
    ap.add_argument("--variants", nargs="+",
                    default=["linear:auto", "linear:0", "nearest:auto", "linear:auto:1e8"],
                    help="shell_weights:shell_kmax[:cmb_noise_scaling]; shell_kmax in h/Mpc, 0 = no "
                         "cut, auto = the config's (default: the init-grid Nyquist)")
    ap.add_argument("--n_iter", type=int, default=60)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out_dir or Path(__file__).resolve().parents[1] / "figures" / "conditioning")
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for spec in args.variants:
        parts = spec.split(":")
        weights, kmax = parts[0], None if parts[1] == "auto" else float(parts[1])
        noise = float(parts[2]) if len(parts) > 2 else 1.0
        model = build_model(cfg, weights, kmax, noise)
        print(f"\n=== shell_weights {weights}, shell_kmax {model.cmb_shell_kmax:.4g}, noise x{noise:g}",
              flush=True)
        scalars = {k: v for k, v in conditioning_params(
            model, cfg.get("truth_params", {}), cfg.get("abacus_truth_params", {})).items()
            if k != "init_mesh"}
        history, (frac, vol), cost = stiffness(model, scalars, jr.key(args.seed), args.n_iter)
        results[spec] = {"lambda": history[-1], "lambda_history": history,
                         "variance_fraction": frac, "volume_fraction": vol, **cost}
        print(f"  lambda_max = {history[-1]:.4g} (last 5: {np.round(history[-5:], 3).tolist()})")
        print(f"  top eigenvector, variance within r of the observer: {frac}  (volume: {vol})")
        print(f"  cost: {cost}")

    print("\nSummary: lambda_max of -log p w.r.t. init_mesh_ (prior alone = 1)")
    for spec, res in results.items():
        print(f"  {spec:18s} {res['lambda']:10.4g}   variance <350/<700/<1100: "
              + " / ".join(f"{res['variance_fraction'][k]:.2f}" for k in ("<350", "<700", "<1100"))
              + f"   {res['s_per_hvp']:.3f} s/HVP" + (f", peak {res['peak_gb']:.1f} GB" if "peak_gb" in res else ""))
    (out / "kappa_stiffness.json").write_text(json.dumps(results, indent=1) + "\n")
    print(f"Saved {out / 'kappa_stiffness.json'}")


if __name__ == "__main__":
    main()
