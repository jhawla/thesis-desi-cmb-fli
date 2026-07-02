#!/usr/bin/env python3
"""
Compare linear (Kaiser) vs. N-body evolved matter field from the same initial Gaussian field.

Left panel:  δ_linear  — linear growth (Kaiser, bE=1, no RSD)  → still Gaussian
Right panel: δ_nbody   — full N-body evolution                  → non-Gaussian cosmic web

Usage:
    python scripts/plot_linear_vs_nbody.py
    python scripts/plot_linear_vs_nbody.py --seed 42 --steps 20 --show
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if "--xla_gpu_enable_command_buffer=" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
    ).strip()

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mesh", type=int, default=128, help="Cubic mesh resolution")
    p.add_argument("--box", type=float, default=1024.0, help="Box size in Mpc/h")
    p.add_argument("--a-obs", type=float, default=1.0, help="Scale factor (z=0 → a=1)")
    p.add_argument("--steps", type=int, default=20, help="N-body time steps")
    p.add_argument("--output-dir", type=str, default="figures")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def make_initial_conditions(cosmo, mesh_shape, box_shape, seed):
    from desi_cmb_fli.bricks import lin_power_mesh

    key = jr.PRNGKey(seed)
    noise = jr.normal(key, shape=mesh_shape)
    pmesh = lin_power_mesh(cosmo, mesh_shape, box_shape, a=1.0)
    init_mesh = jnp.fft.rfftn(noise) * jnp.sqrt(pmesh)
    return init_mesh


def evolve_linear(cosmo, init_mesh, a_obs):
    from desi_cmb_fli.bricks import kaiser_model

    # bE = 1 (unbiased matter, no RSD) → 1 + δ_linear
    field = kaiser_model(cosmo, a=a_obs, bE=1.0, init_mesh=init_mesh, los=None)
    return field - 1.0  # return δ


def evolve_nbody(cosmo, init_mesh, mesh_shape, a_obs, n_steps):
    from jaxpm.painting import cic_paint

    from desi_cmb_fli.bricks import regular_pos
    from desi_cmb_fli.nbody import nbody_bf

    cosmo._workspace = {}
    pos0 = regular_pos(tuple(mesh_shape))

    states = nbody_bf(cosmo, init_mesh, pos0, a=a_obs, n_steps=n_steps)
    pos_final = states[0][-1]  # (N_particles, 3) — last snapshot

    mesh = cic_paint(jnp.zeros(tuple(mesh_shape)), pos_final)
    mesh = mesh / jnp.mean(mesh)
    return mesh - 1.0  # return δ


def plot_comparison(delta_lin, delta_nbody, box_mpc, output_dir, show=False):
    # Take a central 2D slice along z-axis
    nz = delta_lin.shape[2]
    sl = nz // 2
    lin_slice = delta_lin[:, :, sl]
    nbody_slice = delta_nbody[:, :, sl]

    # Common colorscale: symmetric, anchored to 99.5th percentile of the linear field
    vmax = float(np.percentile(np.abs(lin_slice), 99.5))
    vmin = -vmax

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    fig.patch.set_facecolor("white")

    extent = [0, box_mpc, 0, box_mpc]
    cmap = "RdBu_r"

    for ax, data in [(axes[0], lin_slice), (axes[1], nbody_slice)]:
        ax.imshow(
            data.T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            interpolation="bicubic",
        )
        ax.set_xlabel(r"$x\;[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$y\;[\mathrm{Mpc}/h]$")

    plt.tight_layout(pad=0.5)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = output_dir / f"linear_vs_nbody_{timestamp}.png"
    fig.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {outfile}")
    if show:
        plt.show()
    plt.close(fig)
    return outfile


def plot_pk(delta_lin, delta_nbody, cosmo, box_shape, a_obs, output_dir, show=False):
    import jax_cosmo as jc

    from desi_cmb_fli.metrics import spectrum

    box_shape = np.asarray(box_shape)

    k_lin, pk_lin = spectrum(jnp.asarray(1.0 + delta_lin), box_shape=box_shape, comp=1)
    k_nb,  pk_nb  = spectrum(jnp.asarray(1.0 + delta_nbody), box_shape=box_shape, comp=1)
    pk_lin = np.asarray(pk_lin)
    pk_nb  = np.asarray(pk_nb)
    k_lin  = np.asarray(k_lin)
    k_nb   = np.asarray(k_nb)

    # Theoretical nonlinear P(k) from jax_cosmo for reference
    k_th = np.geomspace(k_lin[1], k_lin[-1], 200)
    pk_th_nl  = np.asarray(jc.power.nonlinear_matter_power(cosmo, k_th, a=a_obs))
    pk_th_lin = np.asarray(jc.power.linear_matter_power(cosmo, k_th, a=a_obs))

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    })

    fig, axes = plt.subplots(2, 1, figsize=(8, 8),
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
                             sharex=True)

    ax, ax_r = axes

    ax.loglog(k_th, pk_th_lin, "k--", lw=1.5, label=r"$P_\mathrm{lin}(k)$ theory", zorder=1)
    ax.loglog(k_th, pk_th_nl,  "k-",  lw=1.5, label=r"$P_\mathrm{nl}(k)$ theory (Halofit)", zorder=1)
    ax.loglog(k_lin, pk_lin, "C0o-", ms=4, lw=1.5, label="Linear evolution (measured)", zorder=2)
    ax.loglog(k_nb,  pk_nb,  "C3s-", ms=4, lw=1.5, label="N-body evolution (measured)",  zorder=2)

    ax.set_ylabel(r"$P(k)\;[(\mathrm{Mpc}/h)^3]$")
    ax.legend(fontsize=12, framealpha=0.9)
    ax.set_title(r"Matter power spectrum: linear vs N-body", pad=8)

    # Ratio panel
    k_ratio = k_nb[1:]
    ratio = pk_nb[1:] / np.interp(k_ratio, k_lin[1:], pk_lin[1:])
    ax_r.semilogx(k_ratio, ratio, "C3s-", ms=4, lw=1.5)
    ax_r.axhline(1.0, color="C0", lw=1.5, ls="--")
    ax_r.set_xlabel(r"$k\;[h/\mathrm{Mpc}]$")
    ax_r.set_ylabel(r"$P_\mathrm{nbody}/P_\mathrm{lin}$")
    ax_r.set_ylim(0.5, None)

    plt.tight_layout()
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = output_dir / f"pk_linear_vs_nbody_{timestamp}.png"
    fig.savefig(outfile, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {outfile}")
    if show:
        plt.show()
    plt.close(fig)
    return outfile


def _skewness(x):
    x = x - x.mean()
    return np.mean(x**3) / np.mean(x**2) ** 1.5


def _kurtosis(x):
    x = x - x.mean()
    return np.mean(x**4) / np.mean(x**2) ** 2 - 3.0


def main():
    args = parse_args()

    from desi_cmb_fli.bricks import get_cosmology

    print(f"Seed={args.seed}  mesh={args.mesh}³  box={args.box} Mpc/h  a={args.a_obs}  steps={args.steps}")

    cosmo = get_cosmology(Omega_m=0.315192, sigma8=0.811355)
    cosmo._workspace = {}

    mesh_shape = np.array([args.mesh, args.mesh, args.mesh])
    box_shape = np.array([args.box, args.box, args.box])

    print("Generating initial conditions...")
    init_mesh = make_initial_conditions(cosmo, mesh_shape, box_shape, args.seed)

    print("Running linear evolution (Kaiser)...")
    delta_lin = np.asarray(evolve_linear(cosmo, init_mesh, args.a_obs))
    print(f"  δ_linear  std={float(np.std(delta_lin)):.4f}  skew={float(_skewness(delta_lin)):.4f}")

    print(f"Running N-body ({args.steps} steps)...")
    delta_nbody = np.asarray(evolve_nbody(cosmo, init_mesh, mesh_shape, args.a_obs, args.steps))
    print(f"  δ_nbody   std={float(np.std(delta_nbody)):.4f}  skew={float(_skewness(delta_nbody)):.4f}")

    print("Plotting maps...")
    plot_comparison(
        np.asarray(delta_lin),
        np.asarray(delta_nbody),
        box_mpc=args.box,
        output_dir=args.output_dir,
        show=args.show,
    )

    print("Plotting P(k)...")
    plot_pk(
        delta_lin, delta_nbody,
        cosmo=cosmo,
        box_shape=box_shape,
        a_obs=args.a_obs,
        output_dir=args.output_dir,
        show=args.show,
    )
    print("Done.")


if __name__ == "__main__":
    main()
