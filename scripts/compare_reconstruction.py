#!/usr/bin/env python
"""Where is the initial field actually reconstructed, and does CMB lensing extend it?

Galaxies constrain the field only where there are galaxies. The Born integration runs over a much
longer line of sight, so beyond the galaxy shell CMB lensing is the only thing informing the field.
This script measures that directly: it correlates the true simulation IC with the sampled one in
radial shells around the observer, and overlays several runs (joint vs galaxy-only) on one figure.

    python scripts/compare_reconstruction.py RUN_JOINT RUN_GXY --labels "joint" "galaxies only"

Reconstruction is a property of the posterior: the correlation is computed for every posterior
sample (the fields run_inference.py records per batch and chain, after burn-in; the last state of
each chain for runs that predate them) and shown as its posterior median and 16-84 %, beside the
correlation of the posterior mean, which has less amplitude than the truth wherever the data do not
constrain the field.
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
from analyze_run import (  # noqa: E402
    IC_SMOOTHING_MPC,
    _field_samples,
    _fixed_latents,
    _load_field_state,
    _quantiles,
)


def _smoothing_kernel(shape, box, scale_mpc):
    """Gaussian low-pass at ``scale_mpc``, on the rfftn grid of ``shape``."""
    kx = 2 * np.pi * np.fft.fftfreq(shape[0], d=box[0] / shape[0])
    ky = 2 * np.pi * np.fft.fftfreq(shape[1], d=box[1] / shape[1])
    kz = 2 * np.pi * np.fft.rfftfreq(shape[2], d=box[2] / shape[2])
    k2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    return jnp.asarray(np.exp(-0.5 * k2 * scale_mpc**2))


def load_run(run_dir):
    """The model, truth and geometry of a run, or None if it cannot be compared."""
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
    rmesh = np.asarray(radius_mesh(model.box_center, box, np.asarray(shape), curved_sky=True))
    chi_gxy = (tuple(float(x) for x in np.asarray(truth["chi_range_gxy"]))
               if "chi_range_gxy" in truth else None)
    return {
        "dir": run_dir, "name": run_dir.name, "model": model, "truth": truth,
        "positions": positions, "fixed": _fixed_latents(run_dir, model),  # absent from the state
        "rmesh": rmesh, "box": box, "shape": shape, "chi_gxy": chi_gxy,
        "cmb": bool(getattr(model, "cmb_enabled", False)),
        "chi_min": float(getattr(model, "cmb_chi_matter_min", 0.0)),
        "chi_max": float(getattr(model, "chi_boundary", box[2])),
    }


def measure(run, edges, smoothing, burn_in, chi_gxy, min_cells=200):
    """Per posterior sample: the Pearson r between true and sampled field per radial shell, the
    whole-box coherence, and the residual rms inside and outside the galaxy shell; the posterior
    mean of the smoothed field. ``chi_gxy`` is the galaxy shell of the comparison, the same for every
    run. Samples are streamed, never held together."""
    model, true_k = run["model"], jnp.asarray(run["truth"]["init_mesh"])
    samples, run["what"] = _field_samples(run["dir"], model, run["positions"], run["fixed"],
                                          burn_in)
    smooth = _smoothing_kernel(run["shape"], run["box"], smoothing)
    true_r = np.asarray(jnp.fft.irfftn(true_k * smooth))
    rmesh = run["rmesh"]
    sels = [(rmesh >= lo) & (rmesh < hi) for lo, hi in zip(edges[:-1], edges[1:], strict=False)]
    m_in = (rmesh >= chi_gxy[0]) & (rmesh <= chi_gxy[1]) if chi_gxy else np.ones_like(rmesh, bool)
    m_out = (~m_in) & (rmesh < (run["chi_max"] if run["cmb"] else rmesh.max()))

    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        den = np.sqrt((a**2).sum() * (b**2).sum())
        return float((a * b).sum() / den) if den > 0 else np.nan

    run.update(true=true_r, corr=[], coh=[], rms_in=[], rms_out=[])
    total, n = np.zeros_like(true_r), 0
    for rec_k, _ in samples:
        if tuple(np.shape(rec_k)) != tuple(np.shape(true_k)):
            print(f"{run['name']}: truth init_mesh {np.shape(true_k)} and the sampled field "
                  f"{np.shape(rec_k)} are on different grids; skipping.")
            return False
        rec_r = np.asarray(jnp.fft.irfftn(rec_k * smooth))
        run["corr"].append([corr(true_r[m], rec_r[m]) if m.sum() >= min_cells else np.nan
                            for m in sels])
        ks, *_, coh = model.powtranscoh(jnp.fft.irfftn(true_k), jnp.fft.irfftn(rec_k))
        run["coh"].append(np.asarray(coh))
        run["rms_in"].append(float(np.std((rec_r - true_r)[m_in])))
        run["rms_out"].append(float(np.std((rec_r - true_r)[m_out])))
        total, n = total + rec_r, n + 1
    run["ks"], run["mean"] = np.asarray(ks), total / n
    run["corr_mean"] = [corr(true_r[m], run["mean"][m]) if m.sum() >= min_cells else np.nan
                        for m in sels]
    print(f"{run['name']}: posterior from {run['what']}")
    return True


def plot_slices(runs, out_path, smoothing):
    """The same comparison seen directly: one plane through the observer, the truth against each
    run's posterior mean and its residual. The mean is the posterior's best estimate; it has less
    amplitude than the truth wherever the data do not constrain the field, which the radial
    profile of the other figure quantifies."""
    ref = runs[0]
    rmesh = ref["rmesh"]
    obs_idx = np.unravel_index(int(np.argmin(rmesh)), rmesh.shape)
    sl = int(obs_idx[2])
    cell = float(ref["box"][0] / rmesh.shape[0])
    nx, ny = rmesh.shape[0], rmesh.shape[1]
    extent = [(-obs_idx[0] - 0.5) * cell, (nx - obs_idx[0] - 0.5) * cell,
              (-obs_idx[1] - 0.5) * cell, (ny - obs_idx[1] - 0.5) * cell]

    truth = ref["true"][:, :, sl].T
    means = [r["mean"][:, :, sl].T for r in runs]
    lim = float(np.percentile(np.abs(truth), 99.5))
    rlim = max(float(np.percentile(np.abs(m - truth), 99.5)) for m in means)

    panels = [(truth, "True initial field", "RdBu_r", lim)]
    panels += [(m, f"Posterior mean — {run['label']}", "RdBu_r", lim)
               for m, run in zip(means, runs, strict=False)]
    panels += [(m - truth, f"Posterior mean − truth — {run['label']}", "PuOr_r", rlim)
               for m, run in zip(means, runs, strict=False)]

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
                 f"{smoothing:.0f} Mpc/h — posterior means. "
                 f"Solid: galaxy shell. Dotted: Born integration range.", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    # The slice is one plane, so quote the residual over the whole 3D volume, per sample.
    print("\nrms of (sample - truth) over the full volume, posterior median [16, 84 %]:")
    for run in runs:
        q_in, q_out = _quantiles(run["rms_in"]), _quantiles(run["rms_out"])
        print(f"   {run['label']:28s} inside the galaxy shell {q_in[0]:.4g} [{q_in[1]:.4g}, "
              f"{q_in[2]:.4g}]   outside {q_out[0]:.4g} [{q_out[1]:.4g}, {q_out[2]:.4g}]")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run directories to overlay")
    p.add_argument("--labels", nargs="*", default=None, help="one legend label per run")
    p.add_argument("--out", default=None, help="output png (default: <first run>/figures/...)")
    p.add_argument("--nbins", type=int, default=18, help="number of radial shells")
    p.add_argument("--smoothing", type=float, default=IC_SMOOTHING_MPC,
                   help="Gaussian smoothing of both fields, Mpc/h")
    p.add_argument("--burn_in", type=float, default=0.5,
                   help="fraction of the recorded batches dropped (analyze_run's default)")
    args = p.parse_args()

    labels = args.labels or [Path(r).name for r in args.runs]
    if len(labels) != len(args.runs):
        p.error("--labels must give one label per run")

    runs = []
    for path, label in zip(args.runs, labels, strict=False):
        print(f"\nLoading {path} ...")
        r = load_run(path)
        if r is not None:
            r["label"] = label
            runs.append(r)
    if not runs:
        print("Nothing to compare.")
        return

    rmax = min(float(r["rmesh"].max()) for r in runs)
    edges = np.linspace(0.0, rmax, args.nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    chi_gxy = next((r["chi_gxy"] for r in runs if r["chi_gxy"] is not None), None)
    runs = [r for r in runs if measure(r, edges, args.smoothing, args.burn_in, chi_gxy)]
    if not runs:
        print("Nothing to compare.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    print("\ncorrelation with the truth per shell: a posterior sample, median [16, 84 %]; "
          "the posterior mean")
    print(f"{'shell [Mpc/h]':>16s}" + "".join(f"{r['label']:>32s}" for r in runs))
    corrs = [np.asarray(r["corr"]) for r in runs]
    for b in range(len(centres)):
        if all(np.all(np.isnan(c[:, b])) for c in corrs):
            continue
        row = f"{edges[b]:7.0f}-{edges[b + 1]:7.0f}"
        for r, c in zip(runs, corrs, strict=False):
            q = _quantiles(c[:, b])
            row += f"{q[0]:10.3f} [{q[1]:6.3f}, {q[2]:6.3f}]{r['corr_mean'][b]:7.3f}"
        print(row)

    for i, (r, c) in enumerate(zip(runs, corrs, strict=False)):
        color = colors[i % len(colors)]
        med, lo, hi = _quantiles(c)
        axes[0].plot(centres, med, "-o", ms=3, color=color, label=f"{r['label']}: a sample")
        axes[0].fill_between(centres, lo, hi, color=color, alpha=0.2, lw=0)
        axes[0].plot(centres, r["corr_mean"], "--", color=color,
                     label=f"{r['label']}: posterior mean")
        med, lo, hi = _quantiles(r["coh"])
        axes[1].semilogx(r["ks"], med, "-", color=color, label=r["label"])
        axes[1].fill_between(r["ks"], lo, hi, color=color, alpha=0.2, lw=0)

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
    axes[0].set_title(f"Initial field, per radial shell (Gaussian smoothing {args.smoothing:.0f} "
                      f"Mpc/h)\nposterior median and 16-84 % of a sample, and the posterior mean")
    axes[0].axhline(0.0, color="k", lw=0.8, alpha=0.5)
    axes[0].set_ylim(-0.1, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel(r"$k$ [h/Mpc]")
    axes[1].set_ylabel("coherence")
    axes[1].set_title("Initial field, all scales (whole box), a posterior sample")
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
