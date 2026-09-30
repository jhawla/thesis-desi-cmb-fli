#!/usr/bin/env python
"""Forward-model the AbacusSummit initial conditions and compare the convergence with AbacusLensing.

The most direct test of the κ model: the published HUGE ICs go through our forward model (LPT, light
cone, Born projector) at the Abacus cosmology, and the resulting κ is compared with the AbacusLensing
map of the same simulation, mode by mode, in the band the likelihood uses (ℓ ≤ 2·cmb_nside). No
sampling is involved, so any difference is model error (resolution, LPT vs N-body, discreteness,
projector) plus the matter the model does not hold (below `chi_min`, beyond the box), which the
likelihood already treats as Gaussian noise (`C_ℓ^LOS`).

Per ℓ bin it reports the coherence r = C_tm / sqrt(C_tt C_mm), the transfer sqrt(C_mm / C_tt), and the
power of the error t − m against N_ℓ and against C_ℓ^LOS. For a model that is exact on the matter it
holds, t = m + u with u the unmodelled line of sight: C_err = C_LOS and r = sqrt(1 − C_LOS / C_tt).

    python scripts/validate_kappa_from_ic.py --config configs/inference/abacus/abacus_joint_Nl1p0_chimin700.yaml \\
        --chi_min 292.6 350 500 700

Needs a compute node the first time (the nside-16384 κ map does not fit a login node); the map
resampled to the projection sphere is cached in the output directory and reused. The figure
goes to figures/spectra_diagnostic/ (the pole maps of --maps to figures/maps/), the spectra
(npz) to the output directory with the caches. `--abacus_map`
takes a map already on disk instead (any nside; e.g. `kappa_pred` of a run's truth.npz), in which
case data and model need not share their angular window and the transfer and error columns carry
the ratio of the two windows; the coherence does not.
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Memory settings for JAX on GPU, as in quick_cl_spectra.py.
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
from jax import numpy as jnp  # noqa: E402
from jax import random as jr  # noqa: E402

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from desi_cmb_fli.cmb_lensing import (  # noqa: E402
    load_abacus_ic_truth,
    load_abacus_kappa_observation,
)
from desi_cmb_fli.model import get_model_from_config  # noqa: E402
from desi_cmb_fli.validation import conditioning_params  # noqa: E402

BIN_EDGES = [2, 12, 24, 36, 48, 56]


def build_model(cfg, chi_min, cell_size=None):
    cfg = copy.deepcopy(cfg)
    if cell_size:
        cfg["model"]["cell_size"] = float(cell_size)
    cfg["model"]["galaxies_enabled"] = False
    cfg["cmb_lensing"]["enabled"] = True
    cfg["cmb_lensing"]["chi_min"] = float(chi_min)
    model, _ = get_model_from_config(cfg)
    return model


def abacus_map(cfg, model, cache, given):
    if given:
        data = np.load(given)
        m = np.asarray(data["kappa_pred"] if hasattr(data, "files") else data, dtype=float)
        print(f"Abacus kappa from {given} (nside {hp.npix2nside(m.size)})")
        return m
    if cache.exists():
        print(f"Abacus kappa from the cache {cache}")
        return np.load(cache)
    m = np.asarray(load_abacus_kappa_observation(cfg.get("abacus_kappa", {}), model)["kappa_pred"])
    np.save(cache, m)
    return m


def spectra(t, m, lmax, mask_t, mask_m):
    at = hp.map2alm(t * mask_t, lmax=lmax)
    am = hp.map2alm(m * mask_m, lmax=lmax)
    fs = float(np.mean(mask_m))
    return {
        "tt": hp.alm2cl(at) / fs,
        "mm": hp.alm2cl(am) / fs,
        "tm": hp.alm2cl(at, am) / fs,
        "err": hp.alm2cl(at - am) / fs,
    }


def binned(x, edges):
    return np.array([np.sum(x[a:b]) for a, b in zip(edges[:-1], edges[1:], strict=False)])


def band_limit(m, lmax):
    return hp.alm2map(hp.map2alm(m, lmax=lmax), hp.npix2nside(m.size), lmax=lmax)


def polar_report(t, m, label):
    """Model − Abacus in the four pixels of each polar cap against the rest of the sphere."""
    nside = hp.npix2nside(m.size)
    caps = np.r_[np.arange(4), m.size - 4 + np.arange(4)]  # RING order: first and last ring
    d = m - t
    rest = np.delete(d, caps)
    print(f"  {label} (nside {nside}): model − Abacus in the 8 polar-cap pixels")
    print("    north", np.round(d[caps[:4]], 4), " south", np.round(d[caps[4:]], 4))
    print(
        f"    mean {d[caps].mean():+.4f}; elsewhere mean {rest.mean():+.4f}, rms {rest.std():.4f}"
        f" -> polar mean = {(d[caps].mean() - rest.mean()) / rest.std():+.1f} rms"
    )
    print(f"    Abacus map rms {t.std():.4f}")


def plot_polar_maps(t, m, lmax, chi_min, cell, fig_path, half_width_deg=25.0):
    """Abacus, model and model − Abacus around both poles, at full pixel resolution and at ℓ ≤ lmax."""
    rows = [("full resolution", t, m), (f"ℓ ≤ {lmax}", band_limit(t, lmax), band_limit(m, lmax))]
    vmax = 3 * float(np.std(t))
    xsize = 400
    reso = 2 * half_width_deg * 60 / xsize
    fig = plt.figure(figsize=(16, 16))
    n = 0
    for tag, tt, mm in rows:
        for pole, lat in (("north", 90), ("south", -90)):
            for name, x in (
                ("AbacusLensing", tt),
                ("model from the Abacus IC", mm),
                ("model − AbacusLensing", mm - tt),
            ):
                n += 1
                hp.gnomview(
                    x,
                    rot=(0, lat),
                    reso=reso,
                    xsize=xsize,
                    min=-vmax,
                    max=vmax,
                    cmap="RdBu_r",
                    sub=(4, 3, n),
                    notext=True,
                    cbar=True,
                    title=f"{name}\n{pole} pole, {tag}",
                    fig=fig.number,
                )
    fig.suptitle(
        f"κ around the poles (±{half_width_deg:g}°, gnomonic, pole at the centre), "
        f"chi_min {chi_min:g}, cell {cell} Mpc/h; colour range ±3 × the Abacus map rms",
        fontsize=12,
        y=1.02,
    )
    fig.savefig(fig_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_spectra(results, lmax, config_name, cell_size, fig=None):
    """Spectra, coherence and error power per chi_min; dotted: what a model exact on the matter it
    holds would give, since the Abacus map also holds the line of sight the model does not."""
    fine = np.arange(2, lmax + 2, 4)
    ell = 0.5 * (fine[:-1] + fine[1:] - 1)
    fig_, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (chi_min, cl) in enumerate(results.items()):
        b = {k: binned(v, fine) for k, v in cl.items()}
        c = f"C{i}"
        if i == 0:
            axes[0].plot(ell, b["tt"] / 4, "k", lw=2, label="AbacusLensing")
        axes[0].plot(ell, b["mm"] / 4, c, label=f"model from the Abacus IC, chi_min {chi_min:g}")
        axes[0].plot(ell, (b["tt"] - b["los"]) / 4, c, ls=":")
        axes[1].plot(ell, b["tm"] / np.sqrt(b["tt"] * b["mm"]), c, label=f"chi_min {chi_min:g}")
        axes[1].plot(ell, np.sqrt(np.clip(1 - b["los"] / b["tt"], 0, 1)), c, ls=":")
        axes[2].plot(ell, b["err"] / b["nell"], c, label=f"model error, chi_min {chi_min:g}")
        axes[2].plot(ell, b["los"] / b["nell"], c, ls=":")
    axes[0].set(
        yscale="log",
        ylabel=r"$C_\ell$",
        title="Convergence spectra (bins of 4; dotted: AbacusLensing minus the LOS)",
    )
    axes[1].set(
        ylim=(0, 1.02),
        ylabel=r"$r_\ell$",
        title="Coherence with AbacusLensing (dotted: exact model but for the LOS)",
    )
    axes[2].set(
        yscale="log",
        ylabel=r"$C_\ell^{\rm err}/N_\ell$",
        title="Error power over the ACT DR6 noise (dotted: LOS covariance)",
    )
    axes[2].axhline(1, color="k", lw=0.6)
    for ax in axes:
        ax.set_xlabel(r"$\ell$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig_.suptitle(
        f"κ from the AbacusSummit HUGE ICs through the forward model vs AbacusLensing — "
        f"{config_name}, cell {cell_size:.2f} Mpc/h",
        fontsize=11,
    )
    fig_.tight_layout()
    cell = f"{cell_size:g}".replace(".", "p")
    fig_path = Path(
        fig
        or Path(__file__).resolve().parents[1]
        / "figures"
        / "spectra_diagnostic"
        / f"kappa_from_abacus_ic_cell{cell}.png"
    )
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig_.savefig(fig_path, dpi=130)
    plt.close(fig_)
    return fig_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--config", default="configs/inference/abacus/abacus_joint_Nl1p0_chimin700.yaml"
    )
    ap.add_argument("--chi_min", type=float, nargs="+", default=[292.6, 350.0, 500.0, 700.0])
    ap.add_argument(
        "--cell_size",
        type=float,
        default=None,
        help="override model.cell_size (the Abacus map cache does not depend on it)",
    )
    ap.add_argument("--abacus_map", default=None, help=".npy map or .npz with kappa_pred")
    ap.add_argument(
        "--polar_cut_deg",
        type=float,
        default=0.0,
        help="also drop |b| > 90 - this (the projector's polar-cap excess, §2.6)",
    )
    ap.add_argument(
        "--maps",
        action="store_true",
        help="also save the model maps and plot model vs Abacus around the poles (§8 item 11)",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="caches and the spectra npz; default $SCRATCH/outputs/kappa_from_ic",
    )
    ap.add_argument(
        "--fig",
        default=None,
        help="figure path; default figures/spectra_diagnostic/kappa_from_abacus_ic_cell<cell>.png",
    )
    ap.add_argument(
        "--replot",
        default=None,
        metavar="NPZ",
        help="only redraw the figure from a kappa_from_ic.npz (with --cell_size for its title)",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.replot:
        d = np.load(args.replot)
        chis = sorted({float(k.split("_", 1)[1]) for k in d.files})
        results = {
            c: {k.split("_", 1)[0]: d[k] for k in d.files if k.endswith(f"_{c}")} for c in chis
        }
        lmax = results[chis[0]]["tt"].size - 1
        cell = float(args.cell_size or cfg["model"]["cell_size"])
        print(f"Saved {plot_spectra(results, lmax, Path(args.config).name, cell, args.fig)}")
        return
    out = Path(args.out_dir or Path(os.environ.get("SCRATCH", ".")) / "outputs" / "kappa_from_ic")
    out.mkdir(parents=True, exist_ok=True)
    truth_params = conditioning_params(
        build_model(cfg, args.chi_min[0], args.cell_size),
        cfg.get("truth_params", {}),
        cfg.get("abacus_truth_params", {}),
    )

    results, t_map, ic = {}, None, None
    for chi_min in args.chi_min:
        print(f"\n=== chi_min {chi_min}")
        model = build_model(cfg, chi_min, args.cell_size)
        lmax, nside = int(model.cmb_lmax), int(model.cmb_proj_nside)
        if ic is None:
            ic = load_abacus_ic_truth(cfg["abacus_ic"], model)["init_mesh"]
            t_map = abacus_map(cfg, model, out / f"abacus_kappa_nside{nside}.npy", args.abacus_map)
        model.reset()
        pred = model.predict(
            samples={**truth_params, "init_mesh": jnp.asarray(ic)},
            hide_base=False,
            hide_samp=False,
            hide_det=False,
            frombase=True,
            rng=jr.key(0),
        )
        m_map = np.asarray(pred["kappa_pred"], dtype=float)
        if args.maps and t_map.size == m_map.size:
            cell = f"{float(model.cell_shape[0]):g}".replace(".", "p")
            np.save(out / f"model_kappa_chimin{chi_min:g}_cell{cell}.npy", m_map)
            polar_report(t_map, m_map, "projection sphere")
            polar_report(band_limit(t_map, lmax), band_limit(m_map, lmax), f"ℓ ≤ {lmax}")
            plot_polar_maps(
                t_map,
                m_map,
                lmax,
                chi_min,
                cell,
                Path(__file__).resolve().parents[1]
                / "figures"
                / "maps"
                / f"kappa_from_abacus_ic_poles_cell{cell}_chimin{chi_min:g}.png",
            )

        mask_m = np.asarray(model.cmb_proj_mask, dtype=float)
        mask_t = mask_m if t_map.size == m_map.size else np.asarray(model.cmb_mask, dtype=float)
        if args.polar_cut_deg > 0:
            for mk in (mask_m, mask_t):
                th = hp.pix2ang(hp.npix2nside(mk.size), np.arange(mk.size))[0]
                mk *= (th > np.radians(args.polar_cut_deg)) & (
                    th < np.pi - np.radians(args.polar_cut_deg)
                )
        cl = spectra(t_map, m_map, lmax, mask_t, mask_m)
        los = np.zeros(lmax + 1)
        if getattr(model, "cl_high_z_cached", None) is not None:
            los = np.asarray(model.cl_high_z_cached, dtype=float)[: lmax + 1]
        nell = np.asarray(model.nell_1d, dtype=float)[: lmax + 1]
        cl.update(los=los, nell=nell)
        results[chi_min] = cl

    edges = np.array(BIN_EDGES + [lmax + 1])
    labels = [f"{a}-{b - 1}" for a, b in zip(edges[:-1], edges[1:], strict=False)]
    print(f"\nPer ℓ bin {labels} (sums over the bin; truth = AbacusLensing)")
    for chi_min, cl in results.items():
        b = {k: binned(v, edges) for k, v in cl.items()}
        r = b["tm"] / np.sqrt(b["tt"] * b["mm"])
        r_exp = np.sqrt(np.clip(1 - b["los"] / b["tt"], 0, 1))
        print(f"\nchi_min {chi_min}")
        print("  coherence r            ", np.round(r, 3))
        print("  r if exact but for LOS ", np.round(r_exp, 3))
        print("  transfer sqrt(Cmm/Ctt) ", np.round(np.sqrt(b["mm"] / b["tt"]), 3))
        print("  error / N_ell          ", np.round(b["err"] / b["nell"], 3))
        print("  LOS / N_ell            ", np.round(b["los"] / b["nell"], 3))
        print("  (error - LOS) / N_ell  ", np.round((b["err"] - b["los"]) / b["nell"], 3))
    fig_path = plot_spectra(
        results, lmax, Path(args.config).name, float(model.cell_shape[0]), args.fig
    )
    # The spectra behind the figure go with the caches; --replot redraws from them.
    npz_path = out / "kappa_from_ic.npz"
    np.savez(npz_path, **{f"{k}_{c}": v for c, cl in results.items() for k, v in cl.items()})
    print(f"\nSaved {fig_path} and {npz_path}")


if __name__ == "__main__":
    main()
