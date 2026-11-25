#!/usr/bin/env python
"""Forward-model the AbacusSummit initial conditions and compare the galaxy counts with the Abacus LRGs.

The galaxy counterpart of `validate_kappa_from_ic.py`: the true ICs of the config's `abacus_ic` go
through the forward model (LPT, light cone, Lagrangian bias, RSD, paint, selection) at the Abacus
cosmology and the biases of the galaxy profile at A = 1 (`--profile`, output of
`galaxy_likelihood_profile.py --mode amplitude`) or of its joint minimum (`--mode newton`: the last
iterate's biases, third order, `bfog`, `b2_slope`, `b1_curv` included and held in the shell refits, with the field kept at the true ICs), and the predicted counts m are compared with the
Abacus counts d over the survey cells, per k band (to sqrt(3) k_N: the likelihood keeps the corner modes of the cube) and per radial
shell. No sampling is involved.

Per shell and k band it reports the coherence r = P_dm / sqrt(P_dd P_mm), the transfer T = P_dm / P_mm
and the power of the whitened residual (d − m)/sqrt(n̄S) (1 for a model exact up to the Poisson
variance), off the Nyquist planes of the final grid: there the band limit of a noise that stops at the
survey boundary moves part of the variance out of the survey cells, a property of the paint chain that
the per-cell noise of a closure does not have (docs/pipeline.md 7.19). The same numbers for a closure
realisation, d = m + Gaussian noise with the noise law of the Newton JSON (s_e (1 + s_ed max(δ_m, −1))
sqrt(n̄S), the likelihood's; Poisson for a profile JSON), averaged over `--n_closure` draws with their
scatter, give what an exact model reaches with this noise and this mask. The biases are then fitted again in each shell alone (b1_alpha held at its
global value): biases that differ between shells are a redshift evolution the model's bias does not
have, and the statistics with the shell biases show what that evolution accounts for.

T depends on where the bias fit lands; the effective bias against the matter of the same ICs (all
biases and b1_alpha at 0: the model's matter through the same selection, painting and RSD),
b_eff = P_dm0 / P_m0m0, does not. Its shape b_eff(k) / b_eff(top quarter of the k up to k_N) is read against the
closure's (a biased model field is not flat against the matter): the data's scale dependence with no fit.

    python scripts/validate_galaxy_from_ic.py \\
        --config configs/inference/abacus/abacus_gxyonly_cosmo_box5000_m160.yaml \\
        --profile figures/galaxy_likelihood_profile/abacus_gxyonly_cosmo_box5000_amplitudem160p280.json

Writes `figures/galaxy_ic_test/<config stem>.{png,json}`. GPU, or CPU node with JAX_PLATFORMS=cpu.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import yaml
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from desi_cmb_fli.bricks import band_limit, get_cosmology  # noqa: E402
from desi_cmb_fli.cmb_lensing import (  # noqa: E402
    load_abacus_galaxy_observation,
    load_abacus_ic_truth,
)
from desi_cmb_fli.model import get_model_from_config  # noqa: E402

jax.config.update("jax_enable_x64", True)
OUT = Path(__file__).resolve().parents[1] / "figures" / "galaxy_ic_test"
NAMES = ["b1", "b2", "bs2", "bn2", "bnpar", "b1_alpha"]
SCALE = np.array([0.05, 0.3, 0.3, 30.0, 30.0, 0.3])
COLORS = ["#2a78d6"]


def binned_spectra(a, b, cells, kbins, box, off_nyquist=False):
    """P_aa, P_bb, P_ab of two meshes over the given cells, in spherical k bins (no window
    correction: the ratios r and T are what the test reads); without the modes on a Nyquist plane
    of the grid if ``off_nyquist``."""
    n = a.shape[0]
    fa, fb = np.fft.rfftn(np.where(cells, a, 0.0)), np.fft.rfftn(np.where(cells, b, 0.0))
    kx = np.fft.fftfreq(n, box / n) * 2 * np.pi
    kz = np.fft.rfftfreq(n, box / n) * 2 * np.pi
    k = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2)
    idx = np.digitize(k.ravel(), kbins) - 1
    ok = (idx >= 0) & (idx < len(kbins) - 1) & (k.ravel() > 0)
    if off_nyquist:
        k_n = np.pi * n / box
        on = np.isclose(np.abs(kx[:, None, None]), k_n) | np.isclose(np.abs(kx[None, :, None]), k_n)
        ok &= ~(on | np.isclose(kz[None, None, :], k_n)).ravel()

    def binned(x):
        return np.bincount(idx[ok], x.ravel()[ok], len(kbins) - 1) / np.maximum(
            np.bincount(idx[ok], minlength=len(kbins) - 1), 1
        )

    return (
        binned(np.abs(fa) ** 2),
        binned(np.abs(fb) ** 2),
        binned((fa * np.conj(fb)).real),
    )


def statistics(d, m, nbar, cells, kbins, box, m0=None, k_top=None):
    paa, pbb, pab = binned_spectra(d - nbar, m - nbar, cells, kbins, box)
    e = (d - m) / np.sqrt(np.maximum(nbar, 1e-10))
    pee = binned_spectra(e, e, cells, kbins, box, off_nyquist=True)[0] / cells.sum()
    out = {"r": pab / np.sqrt(paa * pbb), "T": pab / pbb, "residual": pee}
    if m0 is not None:
        k_top = kbins[-1] if k_top is None else k_top
        top = np.array([0.75 * k_top, k_top + 1e-9])
        _, p00, pd0 = binned_spectra(d - nbar, m0 - nbar, cells, kbins, box)
        _, t00, td0 = binned_spectra(d - nbar, m0 - nbar, cells, top, box)
        out["B"] = (pd0 / p00) / (td0[0] / t00[0])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--profile", required=True, type=Path)
    ap.add_argument("--n_shells", type=int, default=3)
    ap.add_argument("--n_kbins", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="closure noise draws")
    ap.add_argument("--n_closure", type=int, default=20, help="closure noise draws")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["galaxies_enabled"] = True
    cfg.get("cmb_lensing", {})["enabled"] = False
    model, _ = get_model_from_config(cfg)
    t0 = time.time()
    tr = load_abacus_galaxy_observation(abacus_gxy_cfg=cfg["abacus_galaxy"], model=model)
    obs, mask = np.asarray(tr["obs"]), np.asarray(tr["gxy_occ_mask3d"], bool)
    selec_paint = jnp.asarray(model.selec_paint)
    nbar = np.asarray(model.gxy_count * jnp.asarray(model.selec_mesh))
    init_true = jnp.asarray(load_abacus_ic_truth(cfg["abacus_ic"], model)["init_mesh"])
    truth = {**cfg.get("truth_params", {}), **cfg.get("abacus_truth_params", {})}
    cosmo = get_cosmology(Omega_m=float(truth["Omega_m"]), sigma8=float(truth["sigma8"]))
    src = json.loads(args.profile.read_text())
    if src.get("mode") == "newton":
        biases = {k: v for k, v in src["iterations"][-1]["theta"].items() if k != "A"}
    else:
        biases = next(p for p in src["profile"] if p["value"] == 1.0)["biases"]
    names = [k for k in NAMES + ["b3", "bds2", "bs3", "bfog", "b2_slope", "b1_curv"] if k in biases]
    theta0 = np.array([biases[k] for k in names])
    print(f"data loaded in {time.time() - t0:.0f} s; biases {biases}", flush=True)

    @jax.jit
    def predict(theta):
        f = model.evolve(
            (cosmo, dict(zip(names, theta, strict=True)), {"init_mesh": init_true}, {"fNL": 0.0})
        )
        return model.gxy_count * band_limit(f["gxy_paint"] * selec_paint, model._sim_mesh)

    box, n = float(model.box_shape[0]), int(model.mesh_shape[0])
    x = (np.arange(n) + 0.5) * box / n - box / 2
    r = np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)
    edges = np.linspace(r[mask].min(), r[mask].max() + 1e-6, args.n_shells + 1)
    shells = [mask & (r >= lo) & (r < hi) for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    # bins of equal width up to the Nyquist k_N, continued over the corner modes to sqrt(3) k_N
    k_nyq = np.pi * n / box
    dk = (k_nyq - 2 * np.pi / box) / args.n_kbins
    kbins = 2 * np.pi / box + dk * np.arange(int(np.ceil((np.sqrt(3) * k_nyq - 2 * np.pi / box) / dk)) + 1)
    kc = 0.5 * (kbins[1:] + kbins[:-1])

    m = np.asarray(predict(jnp.asarray(theta0)))
    m0 = np.asarray(predict(jnp.zeros(len(names))))  # the matter: b1(a) = 0 needs b1_alpha = 0 too
    rng = np.random.default_rng(args.seed)
    noise = src.get("noise_fit", {"s_e": 1.0, "s_ed": 0.0})
    dm = np.maximum(m / np.maximum(nbar, 1e-10) - 1.0, -1.0)
    noise_sd = noise["s_e"] * (1.0 + noise["s_ed"] * dm) * np.sqrt(np.maximum(nbar, 0.0))
    res = {
        "config": args.config,
        "profile": str(args.profile),
        "k": kc.tolist(),
        "k_nyquist": k_nyq,
        "shell_edges": edges.tolist(),
        "biases_global": biases,
        "closure_draws": args.n_closure,
        "closure_noise": noise,
        "residual_modes": "off the Nyquist planes",
        "shells": [],
    }

    nbar_j, obs_j = jnp.asarray(nbar), jnp.asarray(obs)

    def chi2_fn(cells):
        cells = jnp.asarray(cells)

        def chi2(theta):
            r2 = (obs_j - predict(theta)) ** 2 / jnp.maximum(nbar_j, 1e-10)
            return jnp.where(cells, r2, 0.0).sum()

        return jax.jit(jax.value_and_grad(chi2))

    keys = ["r", "T", "residual", "B"]
    free = np.arange(5)  # b1_alpha held at its global value within one shell
    for i, cells in enumerate([mask] + shells):
        name = "all" if i == 0 else f"{edges[i - 1]:.0f}-{edges[i]:.0f}"
        draws = [
            statistics(m + noise_sd * rng.standard_normal(m.shape), m, nbar, cells, kbins, box, m0, k_top=k_nyq)
            for _ in range(args.n_closure)
        ]
        row = {
            "shell": name,
            "cells": int(cells.sum()),
            "data": {
                k: v.tolist() for k, v in statistics(obs, m, nbar, cells, kbins, box, m0, k_top=k_nyq).items()
            },
            "closure_mean": {k: np.mean([d[k] for d in draws], 0).tolist() for k in keys},
            "closure_std": {k: np.std([d[k] for d in draws], 0, ddof=1).tolist() for k in keys},
        }
        vg = chi2_fn(cells)
        chi2_glob = float(vg(jnp.asarray(theta0))[0])
        row["chi2_per_cell_global_biases"] = chi2_glob / row["cells"]
        if i > 0:

            def fun(u, vg=vg):
                th = theta0.copy()
                th[free] = u * SCALE[free]
                val, g = vg(jnp.asarray(th))
                return float(val), np.asarray(g)[free] * SCALE[free]

            fit = minimize(
                fun,
                theta0[free] / SCALE[free],
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 200, "gtol": 1e-3},
            )
            th = theta0.copy()
            th[free] = fit.x * SCALE[free]
            row["biases_fit"] = dict(zip(names, map(float, th), strict=True))
            row["chi2_gain_refit"] = chi2_glob - float(fit.fun)
            m_fit = np.asarray(predict(jnp.asarray(th)))
            row["data_refit"] = {
                k: v.tolist() for k, v in statistics(obs, m_fit, nbar, cells, kbins, box).items()
            }
        res["shells"].append(row)
        print(
            f"shell {name} ({row['cells']} cells): chi2/cell {row['chi2_per_cell_global_biases']:.4f}"
            + (
                f"; refit { ({k: round(v, 3) for k, v in row['biases_fit'].items()}) }, "
                f"chi2 gain {row['chi2_gain_refit']:.0f}"
                if i > 0
                else ""
            ),
            flush=True,
        )

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / Path(args.config).stem
    stem.with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
    rows = res["shells"]
    fig, axes = plt.subplots(
        len(keys), len(rows), figsize=(4 * len(rows), 3 * len(keys)), sharex=True, sharey="row"
    )
    ylims = {"r": (0.90, 1.0), "T": (0.95, 1.05), "residual": (0.7, 1.3), "B": (0.9, 1.1)}
    titles = {
        "r": "coherence r(k)",
        "T": "transfer T(k) = P_dm / P_mm",
        "residual": "whitened residual power",
        "B": "b_eff shape vs IC matter\n(no bias fit)",
    }
    for j, row in enumerate(rows):
        for i, key in enumerate(keys):
            ax = axes[i, j]
            mu, sd = np.array(row["closure_mean"][key]), np.array(row["closure_std"][key])
            ax.fill_between(kc, mu - sd, mu + sd, color="0.75", label="closure ±1σ")
            ax.plot(kc, row["data"][key], color="k", label="Abacus, global biases")
            if key in row.get("data_refit", {}):
                ax.plot(kc, row["data_refit"][key], color=COLORS[0], label="Abacus, shell biases")
            ax.axhline(1, color="0.5", lw=0.6)
            ax.axvspan(k_nyq, kc[-1], color="0.9", zorder=0)
            ax.set_ylim(*ylims[key])
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(titles[key])
            if i == len(keys) - 1:
                ax.set_xlabel("k [h/Mpc]")
        axes[0, j].set_title(
            f"χ {row['shell']} Mpc/h" if row["shell"] != "all" else "all survey cells"
        )
    axes[0, -1].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Galaxy counts from the true Abacus ICs ({Path(args.config).stem})")
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    print(f"Saved {stem.with_suffix('.png')} and .json")


if __name__ == "__main__":
    main()
