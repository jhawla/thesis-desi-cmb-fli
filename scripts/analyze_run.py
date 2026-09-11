#!/usr/bin/env python
"""
Analyze run results with custom filtering.
Can be used as a CLI script or imported as a module.
Burn-in by default is 0% (no burn-in).

Usage:
    python scripts/analyze_run.py --run_dir <RUN_DIR> [--burn_in 0.0] [--exclude_chains 0 2] [--output_subdir analysis_burn0_excl0_2]
"""

import argparse
import os
import pickle

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path

import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from desi_cmb_fli import utils
from desi_cmb_fli.bricks import radius_mesh
from desi_cmb_fli.chains import Chains
from desi_cmb_fli.utils import ObservationMode, restore_model_state_from_truth

# Enable x64 (needed for some operations)
jax.config.update("jax_enable_x64", True)

IC_SMOOTHING_MPC = 100.0


def _merge_batches(run_dir):
    """
    Merge separate batch output files into a single samples.npz.

    Args:
        run_dir (Path or str): Path to the run directory.

    Returns:
        dict: The merged samples dictionary if successful, None if no batches found.
    """
    run_dir = Path(run_dir)
    config_dir = run_dir / "config"
    batches = sorted(config_dir.glob("samples_batch_*.npz"))

    if not batches:
        print("No batches found.")
        return None

    # Sort numerically by batch index
    batches.sort(key=lambda p: int(p.stem.split('_')[-1]))

    print(f"Found {len(batches)} batches.")

    samples = {}

    for batch_file in batches:
        print(f"Loading {batch_file.name}...")
        data = np.load(batch_file)
        for k in data.files:
            if k not in samples:
                samples[k] = []
            samples[k].append(data[k])

    # Concatenate
    merged = {}
    for k, v in samples.items():
        # v is list of (n_chains, n_samples)
        # We need to concat along axis 1 (n_samples)
        merged[k] = np.concatenate(v, axis=1)
        print(f"Merged {k}: {merged[k].shape}")

    out_path = config_dir / "samples.npz"
    np.savez(out_path, **merged)
    print(f"Saved {out_path}")

    return merged

def load_and_process_run(run_dir, burn_in=0.0, exclude_chains=None):
    """
    Load samples from a run directory, apply burn-in and chain exclusion,
    and reparameterize to physical space.

    Args:
        run_dir (Path or str): Path to the run directory.
        burn_in (float): Fraction of samples to discard (0.0-1.0).
        exclude_chains (list[int]): Indices of chains to exclude.

    Returns:
        dict: A dictionary containing:
            - physical_samples (dict): Reparameterized physical samples (numpy arrays).
            - physical_chains (Chains): The Chains object (for plotting).
            - truth_vals (dict): Truth values from config.
            - available_params (list): List of available physical parameter names.
            - n_eff (dict): Effective sample size per parameter (if computed).
            - r_hat (dict): R-hat per parameter (if computed).
    """
    run_dir = Path(run_dir)
    config_dir = run_dir / "config"
    if exclude_chains is None:
        exclude_chains = []

    print(f"Loading run: {run_dir}")
    print(f"Burn-in: {burn_in:.0%}")
    if exclude_chains:
        print(f"Excluding chains: {exclude_chains}")

    # 0. Auto-merge batches if samples.npz doesn't exist or is older than batches
    samples_path = config_dir / "samples.npz"
    batch_files = sorted(config_dir.glob("samples_batch_*.npz"))

    if batch_files:
        needs_merge = False
        if not samples_path.exists():
            print("samples.npz not found, will merge batches...")
            needs_merge = True
        else:
            # Check if any batch is newer than samples.npz
            samples_mtime = samples_path.stat().st_mtime
            newest_batch_mtime = max(b.stat().st_mtime for b in batch_files)
            if newest_batch_mtime > samples_mtime:
                print("Found newer batches, re-merging...")
                needs_merge = True

        if needs_merge:
            _merge_batches(run_dir)

    # 1. Load Model Config
    model_config_path = config_dir / "model.yaml"
    if not model_config_path.exists():
        raise FileNotFoundError(f"Could not find model.yaml at {model_config_path}")

    # 1. Load Model Config
    from desi_cmb_fli.model import get_model_from_config

    config_yaml_path = config_dir / "config.yaml"
    print("Instantiating FieldLevelModel from config...")
    model, cfg = get_model_from_config(config_yaml_path) # Loads config internally

    # 2. Load Samples & Truth
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    samples_raw = jnp.load(samples_path)

    # 3. Apply Burn-in and Chain Exclusion
    samples_burned = {}

    # Determine chains to keep
    first_key = samples_raw.files[0]
    n_chains_total = samples_raw[first_key].shape[0]
    keep_indices = [i for i in range(n_chains_total) if i not in exclude_chains]

    if len(keep_indices) == 0:
        raise ValueError("All chains excluded!")

    for k in samples_raw.files:
        data = samples_raw[k] # (n_chains, n_samples)
        # 1. Exclude chains
        data_kept = data[keep_indices, :]

        # 2. Burn-in
        _, n_total = data_kept.shape
        n_keep = int(n_total * (1 - burn_in))
        start_idx = n_total - n_keep

        # Handle case where n_keep is 0
        if n_keep <= 0:
            raise ValueError(f"Burn-in {burn_in} is too high, 0 samples kept!")

        samples_burned[k] = jnp.array(data_kept[:, start_idx:])

    print(f"Original chains: {n_chains_total} -> Kept: {len(keep_indices)} (indices {keep_indices})")
    print(f"Samples kept per chain: {n_keep} (discarded first {start_idx})")

    # 4. Reparameterize
    print("Reparametrizing to physical space...")
    # Filter for scalar latent parameters
    samples_jax = {
        k: v for k, v in samples_burned.items()
        if k in model.loc_fid or k.endswith('_')
    }

    chain_obj = Chains(samples_jax, model.groups | model.groups_)
    physical_chains = model.reparam_chains(chain_obj, fourier=False, batch_ndim=2)
    physical_samples = {k: np.array(v) for k, v in physical_chains.data.items()}

    # Load Truth values
    config_yaml_path = config_dir / "config.yaml"
    truth_vals = {}
    if config_yaml_path.exists():
        full_cfg = utils.yload(config_yaml_path)
        obs_mode = ObservationMode.validate(full_cfg.get("observation_mode", "closure"))
        if obs_mode == ObservationMode.ABACUS:
            # In abacus mode, use the AbacusSummit cosmology as truth markers
            truth_vals = full_cfg.get("abacus_truth_params", {})
        else:
            truth_vals = full_cfg.get("truth_params", {})

    # Identify available scalar parameters (avoid showing fixed bias params in CMB-only)
    priority_params = ["Omega_m", "sigma8", "fNL", "fNL_bp", "fNL_bpd", "b1", "b2", "bs2", "bn2", "bnpar", "s_e"]
    if model.galaxies_enabled:
        available_params = [p for p in priority_params if p in physical_samples]
        extra_params = [p for p in physical_samples.keys() if p not in available_params]
        available_params += sorted(extra_params)
    else:
        # CMB-only: show cosmology + fNL, but not the (fixed/unused) bias params.
        available_params = [p for p in ["Omega_m", "sigma8", "fNL"] if p in physical_samples]

    # Filter chains/samples to available params only
    physical_chains = Chains({k: physical_chains.data[k] for k in available_params},
                             physical_chains.groups, physical_chains.labels)
    physical_samples = {k: physical_samples[k] for k in available_params}

    return {
        "physical_samples": physical_samples,
        "physical_chains": physical_chains,
        "truth_vals": truth_vals,
        "available_params": available_params,
        "n_kept": n_keep,
        "model": model,  # Return model for validation plots
        "model_config": cfg
    }

def _load_field_state(run_dir):
    """The sampled field is not written to samples_batch_*.npz (too large), so the only place
    it survives is the final sampler state. Returns (positions, truth) or (None, None)."""
    config_dir = Path(run_dir) / "config"
    state_path, truth_path = config_dir / "sampler_state.pkl", config_dir / "truth.npz"
    if not state_path.exists() or not truth_path.exists():
        print("No sampler_state.pkl / truth.npz: skipping the field figures.")
        return None, None
    with open(state_path, "rb") as f:
        positions = pickle.load(f)["state"].position
    truth_data = np.load(truth_path)
    return positions, {k: truth_data[k] for k in truth_data.files}


def _bands(lmax, n=6):
    """Logarithmic multipole bands covering [2, lmax], for the printed diagnostics."""
    edges = np.unique(np.round(np.geomspace(2, lmax + 1, n + 1)).astype(int))
    return list(zip(edges[:-1], edges[1:], strict=False))


def plot_initial_conditions(model, truth, positions, out_path, chain=0):
    """True vs reconstructed initial conditions, their difference, and the reconstruction quality.

    One posterior sample, never a mean over chains: averaging independent samples suppresses the
    amplitude wherever the data does not constrain the field, which reads as a reconstruction that
    lost power when in fact every sample carries the right power.
    """
    if "init_mesh" not in truth:
        print("truth.npz has no init_mesh (no abacus_ic in the config): skipping.")
        return

    n_chains = int(np.shape(positions["init_mesh_"])[0])
    box = np.asarray(model.box_shape, dtype=float)
    shape = tuple(int(s) for s in model.init_shape)
    true_k = jnp.asarray(truth["init_mesh"])

    rec_k = [model.reparam({k: jnp.asarray(np.asarray(v)[c]) for k, v in positions.items()})["init_mesh"]
             for c in range(n_chains)]
    if tuple(np.shape(true_k)) != tuple(np.shape(rec_k[chain])):
        print(f"truth init_mesh {np.shape(true_k)} and the sampled field "
              f"{np.shape(rec_k[chain])} are on different grids: skipping.")
        return

    ks, _, trans, coh = model.powtranscoh(jnp.fft.irfftn(true_k), jnp.fft.irfftn(rec_k[chain]))
    ks, trans, coh = np.asarray(ks), np.asarray(trans), np.asarray(coh)

    kx = 2 * np.pi * np.fft.fftfreq(shape[0], d=box[0] / shape[0])
    ky = 2 * np.pi * np.fft.fftfreq(shape[1], d=box[1] / shape[1])
    kz = 2 * np.pi * np.fft.rfftfreq(shape[2], d=box[2] / shape[2])
    k2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    smooth = jnp.asarray(np.exp(-0.5 * k2 * IC_SMOOTHING_MPC**2))
    true_r = np.asarray(jnp.fft.irfftn(true_k * smooth))
    rec_r = np.asarray(jnp.fft.irfftn(rec_k[chain] * smooth))

    # Slice the plane that contains the observer, in observer-centred coordinates, so that the
    # figure is right for every observer_mode (centre of the box, a face, a corner).
    center = np.asarray(model.box_center, dtype=float)
    coord = [np.arange(shape[d]) * (box[d] / shape[d]) - box[d] / 2 + center[d] for d in range(3)]
    sl = int(np.clip(np.argmin(np.abs(coord[2])), 0, shape[2] - 1))
    a, b = true_r[:, :, sl], rec_r[:, :, sl]
    d = b - a
    lim = float(np.percentile(np.abs(a), 99.5))
    extent = [coord[0][0], coord[0][-1], coord[1][0], coord[1][-1]]
    chi_lo, chi_hi = ((float(x) for x in np.asarray(truth["chi_range_gxy"]))
                      if "chi_range_gxy" in truth else (None, None))

    fig, axes = plt.subplots(1, 5, figsize=(25, 4.8),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.15, 1.15]})
    for ax, img, title, cmap in (
        (axes[0], a, "True initial field (simulation IC)", "RdBu_r"),
        (axes[1], b, f"Reconstructed (one posterior sample, chain {chain})", "RdBu_r"),
        (axes[2], d, f"Difference  (rms {d.std() / a.std():.0%} of the true rms)", "PuOr_r"),
    ):
        im = ax.imshow(img.T, origin="lower", extent=extent, cmap=cmap, vmin=-lim, vmax=lim)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x [Mpc/h]")
        if chi_lo is not None:
            for radius in (chi_lo, chi_hi):
                ax.add_patch(plt.Circle((0, 0), radius, fill=False, ls="--", lw=1.0,
                                        color="k", alpha=0.6))
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("y [Mpc/h]")

    ax = axes[3]
    ax.plot(ks, trans, label=r"transfer $T(k)=\sqrt{P_{\rm rec}/P_{\rm true}}$")
    ax.plot(ks, coh, label=r"cross-correlation $r(k)$")
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylim(0, 1.35)
    ax.legend(fontsize=9)
    ax.set_title("Reconstruction quality", fontsize=11)
    ax.grid(alpha=0.3)

    ax = axes[4]
    rad = np.asarray(radius_mesh(model.box_center, box, np.asarray(shape), curved_sky=True))
    edges = np.arange(0.0, float(rad.max()) + 200.0, 200.0)
    all_rec = [np.asarray(jnp.fft.irfftn(k * smooth)) for k in rec_k]
    prof = {"true": [], "rec": [], "diff": [], "r": [], "rr": []}
    mid = []
    for r0, r1 in zip(edges[:-1], edges[1:], strict=False):
        m = (rad >= r0) & (rad < r1)
        # A shell can be empty once the observer sits on a face or a corner of the box.
        if m.sum() < 2:
            continue
        t, q = true_r[m], rec_r[m]
        mid.append(0.5 * (r0 + r1))
        prof["true"].append(t.std())
        prof["rec"].append(q.std())
        prof["diff"].append((q - t).std())
        prof["r"].append(np.corrcoef(t, q)[0, 1])
        prof["rr"].append(np.mean([np.corrcoef(all_rec[i][m], all_rec[j][m])[0, 1]
                                   for i in range(n_chains) for j in range(i + 1, n_chains)])
                          if n_chains > 1 else np.nan)
    mid = np.asarray(mid)
    ax.plot(mid, np.array(prof["rec"]) / np.array(prof["true"]), label="rms ratio rec/true")
    ax.plot(mid, np.array(prof["diff"]) / np.array(prof["true"]), label="rms of the difference")
    ax.plot(mid, prof["r"], label="correlation with the truth")
    ax.plot(mid, prof["rr"], label="agreement between chains", ls="--")
    if chi_lo is not None:
        ax.axvspan(chi_lo, chi_hi, color="0.85", zorder=0, label="galaxy coverage")
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("distance from the observer [Mpc/h]")
    ax.set_ylim(-0.05, 1.6)
    ax.legend(fontsize=8)
    ax.set_title("Reconstruction vs radius", fontsize=11)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Initial conditions — {Path(out_path).parent.parent.name}, Cartesian plane z=0 through "
        f"the observer, Gaussian smoothing {IC_SMOOTHING_MPC:.0f} Mpc/h; "
        f"dashed circles = galaxy radial range",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"  difference rms / true rms: {d.std() / a.std():.3f} in this slice, "
          f"{(rec_r - true_r).std() / true_r.std():.3f} over the full box")
    lo = ks < 0.02
    if lo.any():
        print(f"  k < 0.02 h/Mpc: transfer {trans[lo].mean():.3f}, "
              f"cross-correlation {coh[lo].mean():.3f}")
    half = 0.5 * (edges[1] - edges[0])
    print(f"  {'radius Mpc/h':>13} {'rms/true':>9} {'r(truth)':>9} {'r(chains)':>10}")
    for c, t_, q_, r_, rr_ in zip(mid, prof["true"], prof["rec"], prof["r"], prof["rr"], strict=False):
        inside = ("  <-- galaxies" if chi_lo is not None
                  and c - half >= chi_lo and c + half <= chi_hi else "")
        print(f"  {c - half:5.0f}-{c + half:5.0f} {q_ / t_:9.2f} {r_:9.3f} {rr_:10.3f}{inside}")


def plot_kappa_maps(model, truth, positions, out_path, chain=0):
    """Observed convergence, the model's convergence, and the model error, with their spectra.

    Everything is shown in the band the likelihood actually uses (l <= 2*cmb_nside, at cmb_nside),
    so this is the reconstruction as the inference sees it, not a prettier high-resolution version
    of it. With cmb_lensing.proj_oversamp > 1 the model map is scattered on a finer sphere and is
    band-limited here, exactly as pack_kappa_map does before the likelihood.
    """
    if "kappa_pred" not in truth:
        print("truth.npz has no kappa_pred: skipping the convergence figure.")
        return

    nside, lmax = int(model.cmb_nside), int(model.cmb_lmax)
    mask = np.asarray(model.cmb_mask)
    proj_mask = np.asarray(model.cmb_proj_mask)
    f_sky = float(np.mean(mask))
    sigma = float(model.sigma_hp)
    bands = _bands(lmax)

    kappa_true = np.asarray(truth["kappa_pred"])            # noiseless truth
    # kappa_obs is a map in a run, but the packed a_lm vector when the truth came straight out of
    # predict(); the noise realisation can only be drawn when it is a map on the same sphere.
    kappa_obs = np.asarray(truth["kappa_obs"]) if "kappa_obs" in truth else None
    if kappa_obs is not None and kappa_obs.shape != kappa_true.shape:
        kappa_obs = None

    pos = {k: jnp.asarray(np.asarray(v)[chain]) for k, v in positions.items()}
    model.reset()
    k_model = np.asarray(model.predict(samples=pos, hide_det=False, hide_base=True,
                                       frombase=False, rng=0)["kappa_pred"])

    def to_band(m):
        """Mask, band-limit to the likelihood's band and resynthesise at the observable nside."""
        w = proj_mask if m.size == proj_mask.size else mask
        return hp.alm2map(hp.map2alm(np.asarray(m) * w, lmax=lmax), nside)

    true_b, model_b = to_band(kappa_true), to_band(k_model)
    error_b = true_b - model_b
    noise_b = to_band(kappa_obs - kappa_true) if kappa_obs is not None else None

    def cl(m):
        return hp.anafast(m, lmax=lmax) / f_sky

    cl_tt, cl_mm, cl_err = cl(true_b), cl(model_b), cl(error_b)

    # The covariance the likelihood assumes exists only in the modes that build it.
    assumed = None
    if getattr(model, "cmb_M_ll", None) is not None:
        total = np.asarray(model.nell_1d)
        if getattr(model, "cl_high_z_cached", None) is not None:
            total = total + np.asarray(model.cl_high_z_cached)
        assumed = np.asarray(model.cmb_M_ll) @ total

    fig = plt.figure(figsize=(22, 5.2))
    vmax = float(np.percentile(np.abs(true_b), 99)) / sigma
    for i, (m, title) in enumerate((
        (true_b, "Observed κ (truth, noiseless)"),
        (model_b, f"Model κ (one posterior sample, chain {chain})"),
        (error_b, "Model error = truth − model"),
    )):
        hp.mollview(m / sigma, sub=(1, 4, i + 1), title=title, min=-vmax, max=vmax,
                    cmap="RdBu_r", unit=r"$\kappa/\sigma_{\rm hp}$", fig=fig.number)
    ax = fig.add_subplot(1, 4, 4)
    ell = np.arange(len(cl_tt))
    curves = [(cl_tt, "truth κ", {"color": "k"}), (cl_mm, "model κ", {"color": "C0"}),
              (cl_err, "model error", {"color": "C3", "lw": 2})]
    if assumed is not None:
        curves.append((assumed, "covariance assumed by the likelihood",
                       {"color": "C2", "ls": "--"}))
    if noise_b is not None:
        curves.append((cl(noise_b), "the noise realisation", {"color": "0.6", "ls": ":"}))
    for y, lab, kw in curves:
        ax.plot(ell[2:], np.asarray(y)[2:], label=lab, **kw)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Is the model error inside the assumed covariance?", fontsize=11)
    fig.suptitle(f"CMB lensing — {Path(out_path).parent.parent.name}, maps band-limited to the "
                 f"likelihood's own l <= {lmax} at nside {nside}", fontsize=11)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"  rms in units of sigma_hp: truth {true_b.std() / sigma:.2f}, "
          f"model {model_b.std() / sigma:.2f}, model error {error_b.std() / sigma:.2f}"
          + (f", noise {noise_b.std() / sigma:.2f}" if noise_b is not None else ""))
    print(f"  {'band':>32}" + "".join(f"  {a}-{b}".rjust(10) for a, b in bands))
    ratios = [("model error / signal", cl_err / np.maximum(cl_tt, 1e-30))]
    if assumed is not None:
        ratios.insert(0, ("model error / assumed covariance",
                          cl_err / np.maximum(assumed, 1e-30)))
    for name, y in ratios:
        v = [np.asarray(y)[(ell >= a) & (ell < b)].mean() for a, b in bands]
        print(f"  {name:>32}" + "".join(f"  {x:8.2f}" for x in v))


def plot_field_reconstruction(run_dir, model, fig_dir):
    """Both field-level figures, from the final sampler state."""
    positions, truth = _load_field_state(run_dir)
    if positions is None:
        return
    restore_model_state_from_truth(model, truth)
    print("\n" + "=" * 40)
    print("FIELD RECONSTRUCTION")
    print("=" * 40)
    plot_initial_conditions(model, truth, positions, Path(fig_dir) / "initial_conditions.png")
    if model.cmb_enabled:
        plot_kappa_maps(model, truth, positions, Path(fig_dir) / "kappa_reconstruction.png")


def analyze_run(run_dir, burn_in=0.0, exclude_chains=None, output_subdir=None, field_plots=True):
    """
    Perform full analysis: load, diagnostics, plots.

    Args:
        run_dir (Path or str): Run directory.
        burn_in (float): Burn-in fraction.
        exclude_chains (list[int]): Chains to exclude.
        output_subdir (str, optional): Name of output directory. If None, generated automatically.
        field_plots (bool): Draw the field-level figures (initial conditions, and for a run with
            CMB lensing the convergence maps) from the final sampler state. One forward model each.
    """
    run_dir = Path(run_dir)

    # Generate default output directory name
    if output_subdir is None:
        suffix_parts = [f"burn{int(burn_in*100)}"]
        if exclude_chains:
            excl_str = "_".join(map(str, exclude_chains))
            suffix_parts.append(f"excl{excl_str}")
        else:
            suffix_parts.append("allchains")
        output_subdir = f"analysis_{'_'.join(suffix_parts)}"

    fig_dir = run_dir / output_subdir
    fig_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {fig_dir}")

    # Load data
    data = load_and_process_run(run_dir, burn_in, exclude_chains)
    physical_chains = data["physical_chains"]
    physical_samples = data["physical_samples"]
    truth_vals = data["truth_vals"]
    available_params = data["available_params"]

    # 5. Diagnostics & Plotting
    print("\n" + "="*40)
    print("DIAGNOSTICS (R-hat & ESS)")
    print("="*40)

    if hasattr(physical_chains, "print_summary"):
        physical_chains.print_summary()
    else:
        print("print_summary method not found on Chains object.")

    # Traces
    print("Plotting Traces...")
    plt.figure(figsize=(10, 2 * len(available_params)))
    physical_chains.plot(names=available_params, grid=True)
    # No tight_layout: Chains.plot lays out via fig.subfigures(), which tight_layout collapses.
    plt.savefig(fig_dir / "traces.png", dpi=150)
    plt.close()

    # Posteriors
    print("Plotting Posteriors...")
    rows = (len(available_params) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4*rows))
    axes = axes.flatten()

    # Calculate mean parameters for posterior check (Posterior Mean)
    mean_params = {}

    for i, p in enumerate(available_params):
        ax = axes[i]
        vals = physical_samples[p].reshape(-1)
        mean_val = float(np.mean(vals))
        mean_params[p] = mean_val

        ax.hist(vals, bins=30, density=True, alpha=0.6, color="blue", edgecolor="black")

        if p in truth_vals:
            ax.axvline(truth_vals[p], color="red", ls="--", lw=2, label="Truth")
        ax.axvline(mean_val, color="green", ls="-", lw=2, label="Mean")

        ax.set_title(p)
        ax.legend()

    # Hide unused subplots
    for j in range(len(available_params), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(fig_dir / "posteriors.png", dpi=150)
    plt.close()

    # GetDist
    try:
        from getdist import plots
        print("Generating Triangle Plot...")
        # Exclude the per-shell ngbars from the triangle (too many params); they remain in the
        # diagnostics table, traces.png and posteriors.png.
        corner_params = [p for p in available_params if not str(p).startswith("ngbar")]
        corner_chains = Chains({k: physical_chains.data[k] for k in corner_params},
                               physical_chains.groups, physical_chains.labels)
        samples_gd = corner_chains.to_getdist()

        g = plots.get_subplot_plotter()
        g.triangle_plot([samples_gd], filled=True, title_limit=1, markers=truth_vals)
        g.export(str(fig_dir / "corner.png"))

    except ImportError:
        print("GetDist not installed, skipping corner plot.")

    if field_plots:
        plot_field_reconstruction(run_dir, data["model"], fig_dir)

    print(f"✓ Analysis complete. Figures in {fig_dir}")

def main():
    parser = argparse.ArgumentParser(description="Re-analyze run results with custom filtering.")
    parser.add_argument("--run_dir", required=True, type=Path, help="Path to the run directory")
    parser.add_argument("--burn_in", type=float, default=0.0, help="Fraction of samples to discard (0.0-1.0)")
    parser.add_argument("--exclude_chains", type=int, nargs="*", default=[], help="Indices of chains to exclude (0-indexed)")
    parser.add_argument("--no_field_plots", action="store_true",
                        help="Skip the initial-conditions and convergence figures (each runs one forward model)")
    args = parser.parse_args()

    analyze_run(args.run_dir, args.burn_in, args.exclude_chains,
                field_plots=not args.no_field_plots)

if __name__ == "__main__":
    main()
