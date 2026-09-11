#!/usr/bin/env python
"""Where is the initial field actually reconstructed, and does CMB lensing extend it?

Galaxies constrain the field only where there are galaxies. The Born integration runs over a much
longer line of sight, so beyond the galaxy shell CMB lensing is the only thing informing the field.
This script measures that directly: it correlates the true simulation IC with the sampled one in
radial shells around the observer, and overlays several runs (joint vs galaxy-only) on one figure.

    python scripts/compare_reconstruction.py RUN_JOINT RUN_GXY --labels "joint" "galaxies only"

The correlation is computed per chain and shown as mean +/- spread across chains; the fields are
never averaged together, since averaging independent posterior samples suppresses the amplitude
wherever the data does not constrain the field.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from desi_cmb_fli.bricks import radius_mesh  # noqa: E402
from desi_cmb_fli.model import get_model_from_config  # noqa: E402
from desi_cmb_fli.utils import restore_model_state_from_truth  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_run import IC_SMOOTHING_MPC, _load_field_state  # noqa: E402


def _smoothing_kernel(shape, box, scale_mpc):
    """Gaussian low-pass at ``scale_mpc``, on the rfftn grid of ``shape``."""
    kx = 2 * np.pi * np.fft.fftfreq(shape[0], d=box[0] / shape[0])
    ky = 2 * np.pi * np.fft.fftfreq(shape[1], d=box[1] / shape[1])
    kz = 2 * np.pi * np.fft.rfftfreq(shape[2], d=box[2] / shape[2])
    k2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    return jnp.asarray(np.exp(-0.5 * k2 * scale_mpc**2))


def load_run(run_dir, smoothing):
    """Return everything the two panels need, or None if the run cannot be compared."""
    run_dir = Path(run_dir)
    model, _ = get_model_from_config(run_dir / "config" / "config.yaml")
    positions, truth = _load_field_state(run_dir)
    if positions is None:
        return None
    if "init_mesh" not in truth:
        print(f"{run_dir.name}: truth.npz has no init_mesh (no abacus_ic in the config); skipping.")
        return None
    restore_model_state_from_truth(model, truth)

    box = np.asarray(model.box_shape, dtype=float)
    shape = tuple(int(s) for s in model.init_shape)
    true_k = jnp.asarray(truth["init_mesh"])
    n_chains = int(np.shape(positions["init_mesh_"])[0])
    rec_k = [model.reparam({k: jnp.asarray(np.asarray(v)[c]) for k, v in positions.items()})["init_mesh"]
             for c in range(n_chains)]
    if tuple(np.shape(true_k)) != tuple(np.shape(rec_k[0])):
        print(f"{run_dir.name}: truth init_mesh {np.shape(true_k)} and the sampled field "
              f"{np.shape(rec_k[0])} are on different grids; skipping.")
        return None

    smooth = _smoothing_kernel(shape, box, smoothing)
    true_r = np.asarray(jnp.fft.irfftn(true_k * smooth))
    rec_r = [np.asarray(jnp.fft.irfftn(r * smooth)) for r in rec_k]
    coh = [np.asarray(model.powtranscoh(jnp.fft.irfftn(true_k), jnp.fft.irfftn(r))[3])
           for r in rec_k]
    ks = np.asarray(model.powtranscoh(jnp.fft.irfftn(true_k), jnp.fft.irfftn(rec_k[0]))[0])

    rmesh = np.asarray(radius_mesh(model.box_center, box, np.asarray(shape),
                                   curved_sky=True))
    chi_gxy = (tuple(float(x) for x in np.asarray(truth["chi_range_gxy"]))
               if "chi_range_gxy" in truth else None)
    return {
        "name": run_dir.name, "true": true_r, "rec": rec_r, "rmesh": rmesh, "box": box,
        "ks": ks, "coh": coh, "chi_gxy": chi_gxy,
        "cmb": bool(getattr(model, "cmb_enabled", False)),
        "chi_min": float(getattr(model, "cmb_chi_min", 0.0)),
        "chi_max": float(getattr(model, "chi_boundary", box[2])),
    }


def radial_correlation(run, edges, min_cells=200):
    """Pearson r between true and reconstructed field, per radial shell, per chain."""
    out = np.full((len(run["rec"]), len(edges) - 1), np.nan)
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=False)):
        sel = (run["rmesh"] >= lo) & (run["rmesh"] < hi)
        if sel.sum() < min_cells:
            continue
        t = run["true"][sel]
        t = t - t.mean()
        nt = np.sqrt((t**2).sum())
        if nt == 0:
            continue
        for c, rec in enumerate(run["rec"]):
            r = rec[sel]
            r = r - r.mean()
            nr = np.sqrt((r**2).sum())
            if nr > 0:
                out[c, b] = float((t * r).sum() / (nt * nr))
    return out


def plot_slices(runs, out_path, smoothing, chain=0):
    """The same comparison seen directly: one plane through the observer.

    One posterior sample per run, never a mean over chains -- averaging independent samples
    suppresses the amplitude wherever the data does not constrain the field, which would read as a
    reconstruction that lost power when every sample carries the right power.
    """
    ref = runs[0]
    rmesh = ref["rmesh"]
    obs_idx = np.unravel_index(int(np.argmin(rmesh)), rmesh.shape)
    sl = int(obs_idx[2])
    cell = float(ref["box"][0] / rmesh.shape[0])
    nx, ny = rmesh.shape[0], rmesh.shape[1]
    extent = [(-obs_idx[0] - 0.5) * cell, (nx - obs_idx[0] - 0.5) * cell,
              (-obs_idx[1] - 0.5) * cell, (ny - obs_idx[1] - 0.5) * cell]

    truth = ref["true"][:, :, sl].T
    recs = [r["rec"][chain][:, :, sl].T for r in runs]
    lim = float(np.percentile(np.abs(truth), 99.5))
    rlim = max(float(np.percentile(np.abs(r - truth), 99.5)) for r in recs)

    panels = [(truth, "True initial field", "RdBu_r", lim)]
    panels += [(r, f"Reconstructed — {run['label']}", "RdBu_r", lim)
               for r, run in zip(recs, runs, strict=False)]
    panels += [(r - truth, f"Residual — {run['label']}", "PuOr_r", rlim)
               for r, run in zip(recs, runs, strict=False)]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.9))
    chi_gxy = next((r["chi_gxy"] for r in runs if r["chi_gxy"] is not None), None)
    chi_born = next((r["chi_max"] for r in runs if r["cmb"]), None)
    chi_min = next((r["chi_min"] for r in runs if r["cmb"]), None)
    for ax, (img, title, cmap, v) in zip(axes, panels, strict=False):
        im = ax.imshow(img, origin="lower", extent=extent, cmap=cmap, vmin=-v, vmax=v)
        for radius, style, lbl in ((chi_gxy[0] if chi_gxy else None, "-", "galaxy shell"),
                                   (chi_gxy[1] if chi_gxy else None, "-", None),
                                   (chi_min, ":", "Born range"),
                                   (chi_born, ":", None)):
            if radius is None:
                continue
            ax.add_patch(plt.Circle((0, 0), radius, fill=False, ec="k", ls=style,
                                    lw=1.1, alpha=0.75, label=lbl))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x [Mpc/h]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("y [Mpc/h]")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Initial field in the plane through the observer, Gaussian smoothing "
                 f"{smoothing:.0f} Mpc/h — one posterior sample (chain {chain}). "
                 f"Solid: galaxy shell. Dotted: Born integration range.", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    # The slice above is one chain and one plane, so quote the residual over the whole 3D volume
    # and over every chain instead -- the picture illustrates, these numbers measure.
    m_in = ((rmesh >= chi_gxy[0]) & (rmesh <= chi_gxy[1])) if chi_gxy else np.ones_like(rmesh, bool)
    m_out = (~m_in) & (rmesh < (chi_born if chi_born else rmesh.max()))
    print("\nrms residual over the full volume, mean +- spread over chains:")
    for run in runs:
        res = [(rec - run["true"]) for rec in run["rec"]]
        ins = np.array([float(np.std(r[m_in])) for r in res])
        out = np.array([float(np.std(r[m_out])) for r in res])
        print(f"   {run['label']:28s} inside the galaxy shell {ins.mean():.4g} ±{ins.std():.1g}"
              f"   outside {out.mean():.4g} ±{out.std():.1g}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run directories to overlay")
    p.add_argument("--labels", nargs="*", default=None, help="one legend label per run")
    p.add_argument("--out", default=None, help="output png (default: <first run>/figures/...)")
    p.add_argument("--nbins", type=int, default=18, help="number of radial shells")
    p.add_argument("--smoothing", type=float, default=IC_SMOOTHING_MPC,
                   help="Gaussian smoothing of both fields, Mpc/h")
    args = p.parse_args()

    labels = args.labels or [Path(r).name for r in args.runs]
    if len(labels) != len(args.runs):
        p.error("--labels must give one label per run")

    runs = []
    for path, label in zip(args.runs, labels, strict=False):
        print(f"\nLoading {path} ...")
        r = load_run(path, args.smoothing)
        if r is not None:
            r["label"] = label
            runs.append(r)
    if not runs:
        print("Nothing to compare.")
        return

    rmax = min(float(r["rmesh"].max()) for r in runs)
    edges = np.linspace(0.0, rmax, args.nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    print(f"\n{'shell [Mpc/h]':>16s}" + "".join(f"{r['label']:>22s}" for r in runs))
    corrs = [radial_correlation(r, edges) for r in runs]
    for b in range(len(centres)):
        if all(np.all(np.isnan(c[:, b])) for c in corrs):
            continue
        row = f"{edges[b]:7.0f}-{edges[b + 1]:7.0f}"
        for c in corrs:
            row += f"{np.nanmean(c[:, b]):15.3f} ±{np.nanstd(c[:, b]):5.3f}"
        print(row)

    for i, (r, c) in enumerate(zip(runs, corrs, strict=False)):
        m, s = np.nanmean(c, axis=0), np.nanstd(c, axis=0)
        axes[0].plot(centres, m, "-o", ms=3, color=colors[i % len(colors)], label=r["label"])
        axes[0].fill_between(centres, m - s, m + s, color=colors[i % len(colors)], alpha=0.2)
        kk = np.asarray(r["ks"])
        cc = np.nanmean(np.asarray(r["coh"]), axis=0)
        axes[1].semilogx(kk, cc, "-", color=colors[i % len(colors)], label=r["label"])

    chi_gxy = next((r["chi_gxy"] for r in runs if r["chi_gxy"] is not None), None)
    if chi_gxy is not None:
        axes[0].axvspan(chi_gxy[0], chi_gxy[1], color="0.85", zorder=0,
                        label=f"galaxy shell ({chi_gxy[0]:.0f}-{chi_gxy[1]:.0f})")
    for r in runs:
        if r["cmb"]:
            axes[0].axvline(r["chi_min"], color="k", ls=":", lw=1)
            axes[0].axvline(r["chi_max"], color="k", ls=":", lw=1)
            break

    axes[0].set_xlabel(r"$\chi$ from the observer [Mpc/h]")
    axes[0].set_ylabel(r"corr(true, reconstructed)")
    axes[0].set_title(f"Initial field, per radial shell\n(Gaussian smoothing {args.smoothing:.0f} Mpc/h)")
    axes[0].axhline(0.0, color="k", lw=0.8, alpha=0.5)
    axes[0].set_ylim(-0.1, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel(r"$k$ [h/Mpc]")
    axes[1].set_ylabel("coherence")
    axes[1].set_title("Initial field, all scales (whole box)")
    axes[1].axhline(0.0, color="k", lw=0.8, alpha=0.5)
    axes[1].set_ylim(-0.1, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    out = Path(args.out) if args.out else Path(args.runs[0]) / "figures" / "reconstruction_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")

    plot_slices(runs, out.with_name(out.stem + "_slices" + out.suffix), args.smoothing)


if __name__ == "__main__":
    main()
