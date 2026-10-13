#!/usr/bin/env python
"""
Analyze run results with custom filtering.
Can be used as a CLI script or imported as a module.
Burn-in by default is 0% (no burn-in).

Usage:
    python scripts/analyze_run.py --run_dir <RUN_DIR> [--burn_in 0.0] [--exclude_chains 0 2] [--output_subdir analysis_burn0_excl0_2]
"""

import argparse
import itertools
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
from desi_cmb_fli.bricks import get_cosmology, lin_power_interp, radius_mesh
from desi_cmb_fli.chains import Chains
from desi_cmb_fli.storage import keep_run
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
    # Every scalar can be fixed (e.g. a kappa-only run with f_NL fixed): only the field is sampled.
    if samples_jax:
        physical_chains = model.reparam_chains(chain_obj, fourier=False, batch_ndim=2)
    else:
        physical_chains = chain_obj
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
    priority_params = ["Omega_m", "sigma8", "fNL", "fNL_bp", "fNL_bpd", "b1", "b1_alpha", "b2", "bs2", "bn2", "bnpar", "s_e"]
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
    """The final sampler state and the truth, (positions, truth) or (None, None). The field per
    batch is in field_batch_*.npz (``_field_samples``); runs that predate them have only this."""
    config_dir = Path(run_dir) / "config"
    state_path, truth_path = config_dir / "sampler_state.pkl", config_dir / "truth.npz"
    if not state_path.exists() or not truth_path.exists():
        print("No sampler_state.pkl / truth.npz: skipping the field figures.")
        return None, None
    with open(state_path, "rb") as f:
        positions = pickle.load(f)["state"].position
    truth_data = np.load(truth_path)
    return positions, {k: truth_data[k] for k in truth_data.files}


def _truth_values(cfg):
    """The true parameter values: ``truth_params`` in closure, ``abacus_truth_params`` otherwise."""
    obs_mode = ObservationMode.validate(cfg.get("observation_mode", "closure"))
    return (cfg.get("truth_params", {}) if obs_mode == ObservationMode.CLOSURE
            else cfg.get("abacus_truth_params", {}))


def _fixed_latents(run_dir, model):
    """The scalar latents the run held fixed (``mcmc.fixed_params``), at the values
    run_inference.py conditioned on. The sampler state does not carry them, and a predict() on
    the state alone would draw them from their priors."""
    cfg = utils.yload(Path(run_dir) / "config" / "config.yaml")
    fixed = list(cfg.get("mcmc", {}).get("fixed_params", []))
    if not fixed:
        return {}
    src = _truth_values(cfg)
    base = {p: float(src.get(p, model.loc_fid[p])) for p in fixed}
    latents = model.reparam(base, inv=True)
    return {p + "_": jnp.asarray(latents[p + "_"]) for p in fixed}


def _bands(lmax, n=6):
    """Logarithmic multipole bands covering [2, lmax], for the printed diagnostics."""
    edges = np.unique(np.round(np.geomspace(2, lmax + 1, n + 1)).astype(int))
    return list(zip(edges[:-1], edges[1:], strict=False))


def _state_samples(model, positions, fixed=None):
    """The final state of each chain, as (Fourier field, cosmology) pairs."""
    for c in range(int(np.shape(positions["init_mesh_"])[0])):
        base = model.reparam({k: jnp.asarray(np.asarray(v)[c]) for k, v in positions.items()}
                             | (fixed or {}))
        yield base["init_mesh"], {k: float(base.get(k, model.loc_fid[k]))
                                  for k in model.groups["cosmo"]}


def _field_samples(run_dir, model, positions, fixed=None, burn_in=0.0, exclude_chains=None):
    """Posterior samples of the initial field as (Fourier field, cosmology) pairs, and a label: the
    states run_inference.py records per batch and chain (``field_batch_*.npz``) after burn-in, or,
    for runs that predate them, the final state of each chain."""
    files = sorted((Path(run_dir) / "config").glob("field_batch_*.npz"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    files = files[int(len(files) * burn_in):]
    if not files:
        n = int(np.shape(positions["init_mesh_"])[0])
        return _state_samples(model, positions, fixed), f"the last state of each of {n} chains"
    n_chains = int(np.load(files[0])["init_mesh"].shape[0])
    chains = [c for c in range(n_chains) if c not in (exclude_chains or [])]

    def samples():
        for f in files:
            d = np.load(f)
            for c in chains:
                yield (jnp.asarray(d["init_mesh"][c]),
                       {k: float(d[k][c]) for k in model.groups["cosmo"]})

    return samples(), f"{len(files) * len(chains)} posterior samples"


def _quantiles(x):
    """Median and 16-84 % of per-sample curves (n_samples, ...)."""
    return np.nanpercentile(np.asarray(x), [50, 16, 84], axis=0)


def plot_initial_conditions(model, truth, samples, out_path, truth_cosmo=None,
                            what="posterior samples"):
    """True initial field against its posterior: posterior mean and spread in the plane through the
    observer, posterior of the power and of the coherence, posterior of the reconstruction per radius.

    ``samples`` yields (Fourier field, cosmology) pairs (``_field_samples``). Reconstruction is a
    property of the posterior, so each quality curve is the distribution over the samples of a
    per-sample metric (median, 16-84 %), with the posterior mean's beside it; the agreement between
    chains answers convergence and is not shown. Each field's power is over the linear spectrum of
    its own cosmology (``truth_cosmo`` for the truth), all expected at 1: with the cosmology free, a
    sample carries the spectrum of its own Omega_m, sigma8.
    """
    if "init_mesh" not in truth:
        print("truth.npz has no init_mesh (no abacus_ic in the config): skipping.")
        return

    box = np.asarray(model.box_shape, dtype=float)
    shape = tuple(int(s) for s in model.init_shape)
    true_k = jnp.asarray(truth["init_mesh"])
    samples = iter(samples)
    first = next(samples, None)
    if first is None:
        print("No field sample: skipping.")
        return
    if tuple(np.shape(true_k)) != tuple(np.shape(first[0])):
        print(f"truth init_mesh {np.shape(true_k)} and the sampled field "
              f"{np.shape(first[0])} are on different grids: skipping.")
        return
    truth_cosmo = {k: float((truth_cosmo or {}).get(k, model.loc_fid[k]))
                   for k in model.groups["cosmo"]}

    def prior_pow(cosmo, k):
        return np.asarray(lin_power_interp(get_cosmology(**cosmo))(jnp.asarray(k)))

    kx = 2 * np.pi * np.fft.fftfreq(shape[0], d=box[0] / shape[0])
    ky = 2 * np.pi * np.fft.fftfreq(shape[1], d=box[1] / shape[1])
    kz = 2 * np.pi * np.fft.rfftfreq(shape[2], d=box[2] / shape[2])
    k2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    smooth = jnp.asarray(np.exp(-0.5 * k2 * IC_SMOOTHING_MPC**2))
    # rms the prior gives the smoothed field at each cosmology, the reference of the radial profile
    kmod, w2 = np.sqrt(k2).ravel(), np.asarray(smooth).ravel() ** 2
    true_prior_rms2 = (prior_pow(truth_cosmo, kmod) * w2).sum()
    true_real = jnp.fft.irfftn(true_k)
    true_r = np.asarray(jnp.fft.irfftn(true_k * smooth))

    rad = np.asarray(radius_mesh(model.box_center, box, np.asarray(shape), curved_sky=True))
    edges = np.arange(0.0, float(rad.max()) + 200.0, 200.0)
    # A shell can be empty once the observer sits on a face or a corner of the box.
    shells = [(0.5 * (r0 + r1), m) for r0, r1 in zip(edges[:-1], edges[1:], strict=False)
              if (m := (rad >= r0) & (rad < r1)).sum() >= 2]
    mid = np.array([c for c, _ in shells])
    true_rms = np.array([true_r[m].std() for _, m in shells])

    amp_true, amp_rec, coh, ratio, diff, corr = None, [], [], [], [], []
    s1, s2, n = np.zeros(shape), np.zeros(shape), 0
    for rec_k, cosmo in itertools.chain([first], samples):
        ks, pow_rec, trans, c = (np.asarray(x) for x in model.powtranscoh(true_real,
                                                                          jnp.fft.irfftn(rec_k)))
        if amp_true is None:
            amp_true = np.sqrt(pow_rec / trans**2 / prior_pow(truth_cosmo, ks))
        amp_rec.append(np.sqrt(pow_rec / prior_pow(cosmo, ks)))
        coh.append(c)
        prior_rms_ratio = np.sqrt((prior_pow(cosmo, kmod) * w2).sum() / true_prior_rms2)
        rec_r = np.asarray(jnp.fft.irfftn(rec_k * smooth))
        ratio.append([rec_r[m].std() for _, m in shells] / true_rms / prior_rms_ratio)
        diff.append([(rec_r[m] - true_r[m]).std() for _, m in shells] / true_rms)
        corr.append([np.corrcoef(true_r[m], rec_r[m])[0, 1] for _, m in shells])
        s1, s2, n = s1 + rec_r, s2 + rec_r**2, n + 1
    mean_r = s1 / n
    std_r = np.sqrt(np.maximum(s2 / n - mean_r**2, 0.0))
    corr_mean = np.array([np.corrcoef(true_r[m], mean_r[m])[0, 1] for _, m in shells])

    # Slice the plane that contains the observer, in observer-centred coordinates, so that the
    # figure is right for every observer_mode (centre of the box, a face, a corner).
    center = np.asarray(model.box_center, dtype=float)
    coord = [np.arange(shape[d]) * (box[d] / shape[d]) - box[d] / 2 + center[d] for d in range(3)]
    sl = int(np.clip(np.argmin(np.abs(coord[2])), 0, shape[2] - 1))
    lim = float(np.percentile(np.abs(true_r[:, :, sl]), 99.5))
    extent = [coord[0][0], coord[0][-1], coord[1][0], coord[1][-1]]
    chi_lo, chi_hi = ((float(x) for x in np.asarray(truth["chi_range_gxy"]))
                      if "chi_range_gxy" in truth else (None, None))

    fig, axes = plt.subplots(1, 5, figsize=(25, 4.8),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.15, 1.15]})
    for ax, img, title, cmap, vlim_ in (
        (axes[0], true_r[:, :, sl], "True initial field (simulation IC)", "RdBu_r", (-lim, lim)),
        (axes[1], mean_r[:, :, sl], "Posterior mean", "RdBu_r", (-lim, lim)),
        (axes[2], std_r[:, :, sl], "Posterior standard deviation", "viridis", (0.0, lim)),
    ):
        im = ax.imshow(img.T, origin="lower", extent=extent, cmap=cmap, vmin=vlim_[0], vmax=vlim_[1])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x [Mpc/h]")
        if chi_lo is not None:
            for radius in (chi_lo, chi_hi):
                ax.add_patch(plt.Circle((0, 0), radius, fill=False, ls="--", lw=1.0,
                                        color="w" if cmap == "viridis" else "k", alpha=0.6))
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("y [Mpc/h]")

    def band(ax, x, curves, color, label):
        med, lo, hi = _quantiles(curves)
        ax.plot(x, med, color=color, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.25, lw=0)

    ax = axes[3]
    ax.plot(ks, amp_true, color="k",
            label=r"truth $\sqrt{P_{\rm true}/P_{\rm lin}(\theta_{\rm true})}$")
    band(ax, ks, amp_rec, "C0", r"posterior $\sqrt{P_{\rm rec}/P_{\rm lin}(\theta_{\rm sample})}$")
    band(ax, ks, coh, "C1", r"posterior cross-correlation $r(k)$ with the truth")
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylim(0, 1.35)
    ax.legend(fontsize=9)
    ax.set_title("Each field over its own prior (expected at 1)", fontsize=11)
    ax.grid(alpha=0.3)

    ax = axes[4]
    band(ax, mid, ratio, "C0", "rms ratio sample/true over the prior's")
    band(ax, mid, diff, "C1", "rms of (sample − truth) / true rms")
    band(ax, mid, corr, "C2", "correlation of a sample with the truth")
    ax.plot(mid, corr_mean, color="C2", ls="--", label="correlation of the posterior mean")
    if chi_lo is not None:
        ax.axvspan(chi_lo, chi_hi, color="0.85", zorder=0, label="galaxy coverage")
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("distance from the observer [Mpc/h]")
    ax.set_ylim(-0.05, 1.6)
    ax.legend(fontsize=8)
    ax.set_title("Reconstruction vs radius (median, 16-84 %)", fontsize=11)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Initial conditions — {Path(out_path).parent.parent.name}, posterior from {what}; "
        f"Cartesian plane z=0 through the observer, Gaussian smoothing {IC_SMOOTHING_MPC:.0f} Mpc/h; "
        f"dashed circles = galaxy radial range",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"  posterior from {what}")
    lo = ks < 0.02
    if lo.any():
        print(f"  k < 0.02 h/Mpc, each field over its own prior: truth {amp_true[lo].mean():.3f}, "
              f"posterior median {np.median(np.asarray(amp_rec)[:, lo].mean(1)):.3f}; "
              f"cross-correlation {np.median(np.asarray(coh)[:, lo].mean(1)):.3f}")
    half = 0.5 * (edges[1] - edges[0])
    q_ratio, q_diff, q_corr = _quantiles(ratio), _quantiles(diff), _quantiles(corr)
    print(f"  {'radius Mpc/h':>13} {'rms/true/prior':>15} {'rms(diff)/true':>15} "
          f"{'r(sample) median [16, 84]':>27} {'r(mean)':>8}")
    for i, c in enumerate(mid):
        inside = ("  <-- galaxies" if chi_lo is not None
                  and c - half >= chi_lo and c + half <= chi_hi else "")
        print(f"  {c - half:5.0f}-{c + half:5.0f} {q_ratio[0, i]:15.2f} {q_diff[0, i]:15.2f} "
              f"{q_corr[0, i]:11.3f} [{q_corr[1, i]:6.3f}, {q_corr[2, i]:6.3f}] "
              f"{corr_mean[i]:8.3f}{inside}")


def plot_kappa_maps(model, truth, positions, out_path, fixed=None, truth_has_los=False,
                    kappa_samples=None):
    """Observed convergence against its posterior, with their spectra and harmonic coherence.

    The posterior samples are ``kappa_samples`` (n_samples, n_obs), the kappa_obs observable
    run_inference.py records per batch and chain; without them, the final state of each chain.
    Each curve is the distribution over the samples of a per-sample quantity (median, 16-84 %) or a
    posterior summary (the mean); the agreement between chains answers convergence and is not shown.
    The model map is the Born convergence of the shells, the mean of the likelihood; the likelihood
    holds the out-of-model term C_l^out (the matter the data map holds and the shells do not) and
    the noise as Gaussian variance. The expected curves are those of a linear Gaussian posterior
    from the convergence alone (a Wiener filter), drawn only without galaxies, which otherwise fix
    the field too.

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
    kappa_only = not model.galaxies_enabled

    kappa_true = np.asarray(truth["kappa_pred"])            # noiseless truth
    # kappa_obs is a map in a run, but the packed a_lm vector when the truth came straight out of
    # predict(); the noise realisation can only be drawn when it is a map on the same sphere.
    kappa_obs = np.asarray(truth["kappa_obs"]) if "kappa_obs" in truth else None
    if kappa_obs is not None and kappa_obs.shape != kappa_true.shape:
        kappa_obs = None

    def to_band(m):
        """Mask, band-limit to the likelihood's band and resynthesise at the observable nside."""
        w = proj_mask if m.size == proj_mask.size else mask
        return hp.alm2map(hp.map2alm(np.asarray(m) * w, lmax=lmax), nside)

    if kappa_samples is not None and len(kappa_samples) > 1:
        post = np.stack([np.asarray(model.unpack_kappa_obs_to_map(v)) for v in kappa_samples])
        what = f"{len(post)} posterior samples"
    else:
        post = []
        for c in range(int(np.shape(next(iter(positions.values())))[0])):
            pos = {k: jnp.asarray(np.asarray(v)[c]) for k, v in positions.items()} | (fixed or {})
            model.reset()
            k_model = model.predict(samples=pos, hide_det=False, hide_base=True, frombase=False,
                                    rng=0)["kappa_pred"]
            post.append(to_band(np.asarray(k_model)))
        post = np.stack(post)
        what = f"the last state of each of {len(post)} chains"
    true_b = to_band(kappa_true)
    mean_b, std_b = post.mean(0), post.std(0)
    noise_b = to_band(kappa_obs - kappa_true) if kappa_obs is not None else None

    def cl(m, m2=None):
        return hp.anafast(m, m2, lmax=lmax) / f_sky

    cl_tt = cl(true_b)
    cl_mm = np.array([cl(m) for m in post])
    cl_err = np.array([cl(true_b - m) for m in post])

    # The covariance the likelihood assumes exists only in the modes that build it.
    assumed = None
    if getattr(model, "cmb_M_ll", None) is not None:
        total = np.asarray(model.nell_1d)
        out = np.zeros_like(total)
        if getattr(model, "cl_out_cached", None) is not None:
            out = np.asarray(model.cl_out_cached)
        assumed = np.asarray(model.cmb_M_ll) @ (total + out)
        out = np.asarray(model.cmb_M_ll) @ out

    # Linear Gaussian posterior from the convergence alone: the model generates S_m = S_t - L (L, the
    # out-of-model part, only in an external map); one sample's error power is then
    # S_t + S_m - 2 W S_t, W = S_m/(S_m + N_a) (pipeline 6).
    out_t = out if (assumed is not None and truth_has_los) else np.zeros_like(cl_tt)
    s_m_l = np.maximum(cl_tt - out_t, 1e-30)
    exp_err = (cl_tt + s_m_l - 2 * s_m_l / (s_m_l + assumed) * cl_tt
               if assumed is not None and kappa_only else None)
    ell = np.arange(len(cl_tt))
    exp_rms = float(np.sqrt(((2 * ell + 1) * s_m_l)[2:].sum() / ((2 * ell + 1) * cl_tt)[2:].sum()))

    # Harmonic coherence r_l = C_tm / sqrt(C_tt C_mm), in linear l bins. Expected for that linear
    # Gaussian posterior, N_a (noise + out-of-model term) the noise it assumes: a sample has the
    # power S_m and reaches r = W sqrt(S_t/S_m) with the truth, the posterior mean sqrt(W S_t/S_m);
    # S_t = S_m + L when the truth holds matter the model does not generate, L = 0 in closure.
    edges = np.unique(np.append(np.arange(2, lmax + 1, 4), lmax + 1))

    def coherence(a, b):
        cab, caa, cbb = cl(a, b), cl(a), cl(b)
        return np.array([cab[lo:hi].sum() / np.sqrt(caa[lo:hi].sum() * cbb[lo:hi].sum())
                         for lo, hi in zip(edges[:-1], edges[1:], strict=False)])

    ell_bin = 0.5 * (edges[:-1] + edges[1:] - 1)
    r_samples = np.array([coherence(true_b, m) for m in post])
    r_mean = coherence(true_b, mean_b)
    exp_truth = exp_mean = None
    if assumed is not None and kappa_only:
        def binned(x):
            return np.array([np.asarray(x)[lo:hi].sum()
                             for lo, hi in zip(edges[:-1], edges[1:], strict=False)])
        s_t = binned(cl_tt)
        s_m = np.maximum(s_t - (binned(out) if truth_has_los else 0.0), 1e-30)
        w = s_m / (s_m + binned(assumed))
        exp_truth = w * np.sqrt(s_t / s_m)
        exp_mean = np.sqrt(w * s_t / s_m)

    fig = plt.figure(figsize=(27, 5.2))
    vmax = float(np.percentile(np.abs(true_b), 99)) / sigma
    for i, (m, title) in enumerate((
        (true_b, "Observed κ (truth, noiseless)"),
        (mean_b, f"Posterior mean of the Born κ ({what})"),
        (true_b - mean_b, "Truth − posterior mean"),
    )):
        hp.mollview(m / sigma, sub=(1, 5, i + 1), title=title, min=-vmax, max=vmax,
                    cmap="RdBu_r", unit=r"$\kappa/\sigma_{\rm hp}$", fig=fig.number)

    def band(ax, x, curves, label, **kw):
        med, lo, hi = _quantiles(curves)
        ax.plot(x, med, label=label, **kw)
        ax.fill_between(x, lo, hi, color=kw.get("color"), alpha=0.25, lw=0)

    ax = fig.add_subplot(1, 5, 4)
    ax.plot(ell[2:], cl_tt[2:], color="k", label="truth κ")
    band(ax, ell[2:], cl_mm[:, 2:], "Born κ of a sample (model mean)", color="C0")
    band(ax, ell[2:], cl_err[:, 2:], "truth − sample", color="C3", lw=2)
    if truth_has_los and assumed is not None:
        ax.plot(ell[2:], (np.median(cl_mm, 0) + out)[2:], color="C0", ls="--",
                label=r"Born κ + $C_\ell^{\rm out}$ (expected: truth κ)")
    if assumed is not None:
        ax.plot(ell[2:], assumed[2:], color="C2", ls="--",
                label=r"$N_\ell + C_\ell^{\rm out}$ assumed by the likelihood")
    if exp_err is not None:
        ax.plot(ell[2:], exp_err[2:], color="C3", ls=":", lw=2,
                label="expected truth − sample (linear Gaussian, κ alone)")
    if noise_b is not None:
        ax.plot(ell[2:], cl(noise_b)[2:], color="0.6", ls=":", label="the noise realisation")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Is the model error inside the assumed covariance?", fontsize=11)

    ax = fig.add_subplot(1, 5, 5)
    band(ax, ell_bin, r_samples, "a posterior sample vs truth (median, 16-84 %)", color="C0")
    ax.plot(ell_bin, r_mean, color="C1", lw=2, label="posterior mean vs truth")
    if exp_truth is not None:
        ax.plot(ell_bin, exp_truth, color="C0", ls=":", lw=2,
                label="expected, sample vs truth (linear Gaussian, κ alone)")
        ax.plot(ell_bin, exp_mean, color="C1", ls=":", lw=2,
                label="expected, posterior mean vs truth (linear Gaussian, κ alone)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(-0.2, 1.0)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$r_\ell = C_\ell^{tm}/\sqrt{C_\ell^{tt}C_\ell^{mm}}$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Harmonic coherence with the truth", fontsize=11)
    fig.suptitle(f"CMB lensing — {Path(out_path).parent.parent.name}, posterior from {what}, maps "
                 f"band-limited to the likelihood's own l <= {lmax} at nside {nside}", fontsize=11)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"  posterior from {what}")
    print(f"  rms in units of sigma_hp: truth {true_b.std() / sigma:.2f}, posterior mean "
          f"{mean_b.std() / sigma:.2f}, a sample (median) {np.median(post.std(1)) / sigma:.2f}"
          + (f" (expected {exp_rms * true_b.std() / sigma:.2f} without the out-of-model part)"
             if truth_has_los else "")
          + f", truth - mean {(true_b - mean_b).std() / sigma:.2f}"
          + (f", noise {noise_b.std() / sigma:.2f}" if noise_b is not None else ""))
    print(f"  {'band':>40}" + "".join(f"  {a}-{b}".rjust(10) for a, b in bands))
    err = cl_err.mean(0)
    ratios = [("truth - sample / signal (posterior mean)", err / np.maximum(cl_tt, 1e-30))]
    if assumed is not None:
        ratios[:0] = [("truth - sample / covariance (post. mean)", err / np.maximum(assumed, 1e-30))]
    if exp_err is not None:
        ratios.insert(1, ("expected (linear Gaussian, kappa alone)",
                          exp_err / np.maximum(assumed, 1e-30)))
    for name, y in ratios:
        v = [np.asarray(y)[(ell >= a) & (ell < b)].mean() for a, b in bands]
        print(f"  {name:>40}" + "".join(f"  {x:8.2f}" for x in v))
    print(f"  harmonic coherence r_l with the truth, in l bins of 4 (l = {int(ell_bin[0])} ... "
          f"{int(ell_bin[-1])}):")
    q = _quantiles(r_samples)
    rows = [("sample, median", q[0]), ("sample, 16 %", q[1]), ("sample, 84 %", q[2]),
            ("posterior mean", r_mean)]
    if exp_truth is not None:
        rows += [("expected sample", exp_truth), ("expected mean", exp_mean)]
    for name, y in rows:
        print(f"  {name:>20}" + "".join(f" {x:5.2f}" for x in y))

    if len(post) > 1:
        _plot_kappa_posterior_mean(true_b, mean_b, std_b, what, sigma, lmax, nside,
                                   Path(out_path).with_name("kappa_posterior_mean.png"))
    return {"ell": ell_bin, "r_samples": r_samples, "r_mean": r_mean, "exp_truth": exp_truth,
            "exp_mean": exp_mean}


def _plot_kappa_posterior_mean(true_b, mean_b, std_b, what, sigma, lmax, nside, out_path):
    """Truth, posterior mean, per-pixel posterior spread and the normalised residual. The mean is
    the best estimate of the map but has less power than the truth where the data are weak; the
    spread says where. Only l >= 2 is recorded, so the truth is shown without its monopole and
    dipole as well."""
    alm = hp.map2alm(true_b, lmax=lmax)
    for ell in (0, 1):
        for m in range(ell + 1):
            alm[hp.Alm.getidx(lmax, ell, m)] = 0.0
    true_l2 = hp.alm2map(alm, nside)
    resid = (true_l2 - mean_b) / np.maximum(std_b, 1e-30)
    fig = plt.figure(figsize=(22, 5.2))
    vmax = float(np.percentile(np.abs(true_l2), 99)) / sigma
    for i, (m, title, lim, cmap, unit) in enumerate((
        (true_l2 / sigma, "Observed κ (truth, noiseless)", vmax, "RdBu_r", r"$\kappa/\sigma_{\rm hp}$"),
        (mean_b / sigma, f"Posterior mean ({what})", vmax, "RdBu_r", r"$\kappa/\sigma_{\rm hp}$"),
        (std_b / sigma, "Posterior standard deviation per pixel", None, "viridis",
         r"$\kappa/\sigma_{\rm hp}$"),
        (resid, "(truth − mean) / posterior std", 3.0, "RdBu_r", "σ"),
    )):
        kw = {"min": -lim, "max": lim} if lim is not None else {}
        hp.mollview(m, sub=(1, 4, i + 1), title=title, cmap=cmap, unit=unit, fig=fig.number, **kw)
    fig.suptitle(f"CMB lensing posterior — {Path(out_path).parent.parent.name}, l = 2 ... {lmax} "
                 f"at nside {nside}", fontsize=11)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"  posterior mean rms / truth rms {mean_b.std() / true_l2.std():.2f}; mean posterior std / "
          f"truth rms {std_b.mean() / true_l2.std():.2f}; rms of (truth - mean)/std {resid.std():.2f}")


def _load_kappa_samples(run_dir, burn_in=0.0, exclude_chains=None):
    """The kappa_obs observable run_inference.py records per batch and chain, after burn-in, as
    (n_samples, n_obs); None for runs that predate it or have no CMB lensing."""
    files = sorted((Path(run_dir) / "config").glob("kappa_batch_*.npz"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        return None
    files = files[int(len(files) * burn_in):]
    stack = np.stack([np.load(f)["kappa_obs"] for f in files])      # (batches, chains, n_obs)
    keep = [c for c in range(stack.shape[1]) if c not in (exclude_chains or [])]
    return stack[:, keep].reshape(-1, stack.shape[-1])


def plot_field_reconstruction(run_dir, model, fig_dir, burn_in=0.0, exclude_chains=None):
    """Both field-level figures, from the fields and maps recorded per batch and chain after burn-in,
    or, for runs that predate them, from the final sampler state."""
    positions, truth = _load_field_state(run_dir)
    if positions is None:
        return
    restore_model_state_from_truth(model, truth)
    fixed = _fixed_latents(run_dir, model)
    print("\n" + "=" * 40)
    print("FIELD RECONSTRUCTION")
    print("=" * 40)
    if fixed:
        print(f"Fixed latents (held by the run, not in the sampler state): "
              f"{ {k: float(v) for k, v in fixed.items()} }")
    cfg = utils.yload(Path(run_dir) / "config" / "config.yaml")
    samples, what = _field_samples(run_dir, model, positions, fixed, burn_in, exclude_chains)
    plot_initial_conditions(model, truth, samples, Path(fig_dir) / "initial_conditions.png",
                            truth_cosmo=_truth_values(cfg), what=what)
    if model.cmb_enabled:
        external = (ObservationMode.validate(cfg.get("observation_mode", "closure"))
                    != ObservationMode.CLOSURE)
        plot_kappa_maps(model, truth, positions, Path(fig_dir) / "kappa_reconstruction.png",
                        fixed=fixed, truth_has_los=external,
                        kappa_samples=_load_kappa_samples(run_dir, burn_in, exclude_chains))


def analyze_run(run_dir, burn_in=0.0, exclude_chains=None, output_subdir=None, field_plots=True):
    """
    Perform full analysis: load, diagnostics, plots.

    Args:
        run_dir (Path or str): Run directory.
        burn_in (float): Burn-in fraction.
        exclude_chains (list[int]): Chains to exclude.
        output_subdir (str, optional): Name of output directory. If None, generated automatically.
        field_plots (bool): Draw the field-level figures (initial conditions, and for a run with
            CMB lensing the convergence maps) from the posterior samples recorded per batch.
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

    if not available_params:
        print("\nNo sampled scalar parameter: skipping diagnostics, traces and corner plot.")
        if field_plots:
            plot_field_reconstruction(run_dir, data["model"], fig_dir, burn_in, exclude_chains)
        print(f"✓ Analysis complete. Figures in {fig_dir}")
        return

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
        plot_field_reconstruction(run_dir, data["model"], fig_dir, burn_in, exclude_chains)

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
    keep_run(args.run_dir)  # bring the new figures to the kept copy on CFS

if __name__ == "__main__":
    main()
