#!/usr/bin/env python
"""
Joint Field-Level Inference: Galaxies + CMB Lensing

When cmb_lensing.enabled=true: uses joint likelihood p(obs_gal, kappa_obs | δ, cosmo, bias)

Supports --resume to continue from a previous run.
"""

import argparse
import copy
import gc
import os
import pickle
import shutil
import sys

import yaml

# Memory optimization for JAX on GPU
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from jax import jit, pmap

from desi_cmb_fli import utils
from desi_cmb_fli.cmb_lensing import load_abacus_galaxy_observation, load_abacus_kappa_observation
from desi_cmb_fli.model import get_model_from_config
from desi_cmb_fli.samplers import get_mclmc_run, get_mclmc_warmup
from desi_cmb_fli.utils import ObservationMode

try:
    from scripts.analyze_run import analyze_run
except ImportError:
    from analyze_run import analyze_run

jax.config.update("jax_enable_x64", True)

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/inference/config.yaml")
parser.add_argument("--resume", type=str, default=None,
                    help="Path to existing run_dir to resume from")
args = parser.parse_args()

# Resume mode detection
RESUME_MODE = args.resume is not None

# Load config and setup directories
scratch_dir = os.environ.get("SCRATCH", "/pscratch/sd/j/jhawla")
start_time = datetime.now()

if RESUME_MODE:
    run_dir = Path(args.resume)
    if not run_dir.exists():
        raise ValueError(f"Resume directory not found: {run_dir}")
    config_dir = run_dir / "config"
    fig_dir = run_dir / "figures"
    cfg = utils.yload(config_dir / "config.yaml")
    existing_batches = sorted(config_dir.glob("samples_batch_*.npz"))
    start_batch = len(existing_batches)
    print(f"🔄 RESUME MODE: Loading from {run_dir}")
    print(f"🔄 Found {start_batch} existing batches, will resume from batch {start_batch}")
else:
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_dir = Path(scratch_dir) / "outputs" / f"run_{timestamp}_{job_id}"
    fig_dir = run_dir / "figures"
    config_dir = run_dir / "config"
    for d in [fig_dir, config_dir]:
        d.mkdir(parents=True, exist_ok=True)
    cfg = utils.yload(args.config)
    shutil.copy(args.config, config_dir / "config.yaml")
    start_batch = 0

print(f"JAX version: {jax.__version__}")
print(f"NumPyro version: {numpyro.__version__}")
print(f"Backend: {jax.default_backend()}")
print(f"Run dir: {run_dir}")

devices = jax.devices()
print(f"\nDevices: {len(devices)}")
for i, d in enumerate(devices):
    print(f"  {i}: {d}")

# Model setup
print("\n" + "=" * 80)
print("MODEL CONFIGURATION")
print("=" * 80)

model, model_config = get_model_from_config(cfg) # Passes dict directly

print(model)
model.save(config_dir / "model.yaml")

# Check enabled observables
cmb_enabled = model.cmb_enabled
galaxies_enabled = model.galaxies_enabled

if cmb_enabled and galaxies_enabled:
    print("\n✓ Joint inference: Galaxies + CMB lensing")
elif cmb_enabled:
    print("\n✓ CMB lensing ENABLED (CMB-only mode)")
elif galaxies_enabled:
    print("\n✓ Galaxies ENABLED (galaxies-only mode)")

if cmb_enabled:
    print(f"  z_source: {model.cmb_z_source}")
    if model_config.get("full_los_correction"):
         print("  ✓ full_los_correction ENABLED")

    if hasattr(model, "cmb_noise_nell") and model.cmb_noise_nell is not None:
         print("  Noise: Using N_ell from config")

# Plot N_ell if CMB is enabled
if model.cmb_enabled and model.cmb_noise_nell is not None:
    from desi_cmb_fli.validation import plot_cmb_noise_spectrum
    plot_cmb_noise_spectrum(model, fig_dir)


# =============================================================================
# RESUME MODE: Load saved state and skip to sampling
# =============================================================================
if RESUME_MODE:
    print("\n" + "=" * 80)
    print("🔄 LOADING SAVED STATE")
    print("=" * 80)

    # Load truth
    print("\nLoading truth.npz...")
    truth_data = jnp.load(config_dir / "truth.npz")
    truth = {k: truth_data[k] for k in truth_data.files}
    print(f"  Loaded keys: {list(truth.keys())}")

    # Load sampler state
    print("\nLoading sampler_state.pkl...")
    with open(config_dir / "sampler_state.pkl", "rb") as f:
        saved_state = pickle.load(f)
    state = saved_state["state"]
    config = saved_state["config"]
    print("  ✓ State loaded")

# =============================================================================
# NORMAL MODE: Generate truth and run warmup
# =============================================================================
else:
    print("\n" + "=" * 80)
    print("GENERATING OBSERVATION")
    print("=" * 80)

    seed = cfg["seed"]
    observation_mode = ObservationMode.validate(cfg.get("observation_mode", "closure"))

    # ── Branch by observation mode ──────────────────────────────────────────
    if observation_mode == ObservationMode.CLOSURE:
        # Closure test: generate synthetic observation from truth_params
        truth_params = cfg["truth_params"]
        print(f"[Closure test] Truth params: {truth_params}")
        print(f"Seed: {seed}")

        truth = model.predict(
            samples=truth_params,
            hide_base=False,
            hide_samp=False,
            hide_det=False,
            frombase=True,
            rng=jr.key(seed),
        )
        print("[Closure test] kappa_obs generated synthetically from truth_params")

        if model.cmb_enabled and "kappa_obs" in truth:
            truth["kappa_obs_packed"] = truth["kappa_obs"]
            truth["kappa_obs"] = model.unpack_kappa_obs_to_map(truth["kappa_obs"])

    elif observation_mode == ObservationMode.ABACUS:
        truth = {}

        # ── Galaxy observation ──────────────────────────────────────────
        if model.galaxies_enabled:
            abacus_gxy_cfg = cfg.get("abacus_galaxy", None)
            if abacus_gxy_cfg is None:
                raise ValueError(
                    "observation_mode='abacus' with galaxies_enabled=True requires "
                    "an 'abacus_galaxy' section in config.yaml."
                )
            truth_gxy = load_abacus_galaxy_observation(
                abacus_gxy_cfg=abacus_gxy_cfg,
                model=model,
            )
            truth.update(truth_gxy)

        # ── CMB kappa observation ───────────────────────────────────────
        if model.cmb_enabled:
            truth_cmb = load_abacus_kappa_observation(
                abacus_cfg=cfg.get("abacus_kappa", {}),
                model=model,
            )
            truth.update(truth_cmb)
            truth["kappa_obs_packed"] = model.pack_kappa_obs(jnp.asarray(truth["kappa_obs"]))

        if not truth:
            raise ValueError(
                "observation_mode='abacus' but neither galaxies_enabled nor "
                "cmb_enabled is True — nothing to load."
            )

        abacus_truth = cfg.get("abacus_truth_params", {})
        if abacus_truth:
            print(f"[Abacus] Corner plot markers (abacus_truth_params): {abacus_truth}")
        else:
            print(
                "[Abacus] WARNING: abacus_truth_params not set in config — "
                "corner plot will have no truth markers."
            )

    else:
        raise NotImplementedError(f"observation_mode {observation_mode!r} not handled here")
    # ─────────────────────────────────────────────────────────────────────────

    jnp.savez(config_dir / "truth.npz", **truth)

    effective_cfg = copy.deepcopy(cfg)
    effective_cfg["observation_mode"] = observation_mode.value
    with open(config_dir / "config.yaml", "w") as _f:
        yaml.dump(effective_cfg, _f, default_flow_style=False, sort_keys=False)

    if model.galaxies_enabled and 'obs' in truth:
        print(f"\nGalaxy obs shape: {truth['obs'].shape}")
        print(f"Mean count: {float(jnp.mean(truth['obs'])):.4f}")
        print(f"Std: {float(jnp.std(truth['obs'])):.4f}")

    if model.cmb_enabled and 'kappa_obs' in truth:
        print(f"\nCMB kappa obs shape: {truth['kappa_obs'].shape}")
        print(f"Mean: {float(jnp.mean(truth['kappa_obs'])):.4e}")
        print(f"Std: {float(jnp.std(truth['kappa_obs'])):.4e}")

    # Validation: Plot Field Slices
    print("\n" + "-" * 40)
    print("VALIDATION: Field Slices")
    print("-" * 40)
    from desi_cmb_fli.bricks import get_cosmology
    from desi_cmb_fli.validation import plot_field_slices, plot_spectra

    chi_center = float(model.box_center[2]) if hasattr(model, "box_center") else float(model.box_shape[2] / 2.0)

    plot_field_slices(
        truth,
        output_dir=fig_dir,
        box_shape=model.box_shape,
        chi_center=chi_center,
        observation_mode=observation_mode,
        cmb_mask=getattr(model, "cmb_mask", None),
        cmb_nside=getattr(model, "cmb_nside", None),
        observer_position=getattr(model, "observer_position", None),
        chi_boundary=getattr(model, "chi_boundary", None),
    )
    print(f"✓ Saved field slices to {fig_dir}")

    # Validation: Power Spectra
    cosmo_for_spectra = (
        cfg.get("truth_params") if observation_mode == ObservationMode.CLOSURE
        else cfg.get("abacus_truth_params")
    )
    plot_spectra(
        truth=truth,
        model=model,
        output_dir=fig_dir,
        cosmo_params=cosmo_for_spectra,
        model_config=model_config,
        observation_mode=observation_mode,
        suffix="_truth_check",
    )

# MCMC config
print("\n" + "=" * 80)
print("MCMC CONFIGURATION")
print("=" * 80)

num_warmup = cfg["mcmc"]["num_warmup"]
num_samples = cfg["mcmc"]["num_samples"]  # Samples PER BATCH
num_batches = cfg["mcmc"].get("num_batches", 1)  # Number of batches
num_chains_req = cfg["mcmc"]["num_chains"]

# Auto-adjust num_chains if fewer devices are available
if len(devices) < num_chains_req:
    print(f"\n⚠️  WARNING: Requested {num_chains_req} chains but only {len(devices)} devices available.")
    print(f"   Adjusting num_chains to {len(devices)} to avoid pmap crash.")
    num_chains = max(1, len(devices))
else:
    num_chains = num_chains_req

mclmc_cfg = cfg["mcmc"].get("mclmc", {})
desired_energy_var = float(mclmc_cfg.get("desired_energy_var", 5e-4))
diagonal_precond = bool(mclmc_cfg.get("diagonal_preconditioning", True))
thinning = int(mclmc_cfg.get("thinning", 1))

print("Sampler: MCLMC (Multi-chain + Mini-batch)")
print(f"Chains: {num_chains}")
print(f"Warmup: {num_warmup}")
print(f"Samples per batch: {num_samples}")
print(f"Number of batches: {num_batches}")
print(f"Total samples per chain: {num_samples * num_batches}")
print(f"Total samples (all chains): {num_samples * num_batches * num_chains}")
print(f"Energy var: {desired_energy_var}, Diag precond: {diagonal_precond}")
print(f"Thinning: {thinning}")

if len(devices) >= num_chains:
    print(f"\n✓ {num_chains} chains // across {len(devices)} GPUs")
    if len(devices) > num_chains:
        print(f"  (Note: {len(devices) - num_chains} GPUs idle)")
else:
    print(f"\n⚠️  WARNING: Only {len(devices)} GPUs available for {num_chains} chains.")
    print("   Some chains will share GPUs, which may slow down sampling significantly.")

# Condition model on observations
condition_dict = {}
if model.galaxies_enabled and "obs" in truth:
    condition_dict["obs"] = truth["obs"]
    model.gxy_occ_mask3d = truth.get("gxy_occ_mask3d", None)
    if "selec_mesh" in truth:
        model.selec_mesh = truth["selec_mesh"]

if cmb_enabled and "kappa_obs_packed" in truth:
    condition_dict["kappa_obs"] = truth["kappa_obs_packed"]

fixed_params = list(cfg.get("mcmc", {}).get("fixed_params", []))
fixed_dict = {}
if fixed_params:
    _truth_src = (
        cfg.get("truth_params", {}) if observation_mode == ObservationMode.CLOSURE
        else cfg.get("abacus_truth_params", {})
    )
    for _p in fixed_params:
        fixed_dict[_p] = float(_truth_src.get(_p, model.loc_fid[_p]))
    print(f"\n🔒 Fixing parameters to their truth/fiducial value: {fixed_dict}")

fixed_latent_keys = [p + "_" for p in fixed_params]

def apply_conditioning():
    """Condition the model on the data and on any fixed scalar latents, then block."""
    model.condition(condition_dict)
    if fixed_dict:
        model.condition(fixed_dict, frombase=True)
    model.block()

if model.galaxies_enabled and cmb_enabled:
    print("\n✓ Joint inference: Galaxies + CMB lensing")
elif model.galaxies_enabled:
    print("\n✓ Galaxy-only inference")
elif cmb_enabled:
    print("\n✓ CMB-only inference (galaxies disabled)")
else:
    raise ValueError("At least one observable must be enabled")

# =============================================================================
# WARMUP (skip if resuming)
# =============================================================================
if not RESUME_MODE:
    # STEP 1: Warmup mesh only (benchmark approach)
    print("\n" + "=" * 80, flush=True)
    print("STEP 1: WARMUP MESH ONLY", flush=True)
    print("=" * 80, flush=True)

    model.reset()
    model.condition(condition_dict | model.loc_fid, frombase=True)
    model.block()

    # Initialize (multi-chains)
    if model.galaxies_enabled and "obs" in truth:
        # Use galaxy overdensity for initialization
        delta_obs = truth["obs"] - 1
        rngs = jr.split(jr.key(45), num_chains)
        scale_field = 2/3

        # Define a wrapper to handle the method call cleanly with pmap
        def init_fn(rng, delta):
            return model.kaiser_post(rng, delta, scale_field=scale_field)

        init_params_ = pmap(init_fn, in_axes=(0, None))(rngs, delta_obs)
    else:
        # CMB-only mode: initialize from fiducial with random init_mesh
        rngs = jr.split(jr.key(45), num_chains)
        print("\n⚠️  CMB-only: Initializing from fiducial cosmology with random init_mesh")

        def init_from_fid(rng):
            # Sample random init_mesh_ in sampling space (real-valued gaussian)
            scale, _ = model._precond_scale_and_transfer(
                get_cosmology(**model.loc_fid),
                1 + model.loc_fid["b1"]
            )
            init_mesh_samp = jr.normal(rng, scale.shape) * scale

            # Reparametrize loc_fid from base space to sampling space
            loc_fid_samp = model.reparam(model.loc_fid, inv=True)
            return loc_fid_samp | {"init_mesh_": init_mesh_samp}

        init_params_ = pmap(init_from_fid)(rngs)
    init_mesh_ = {k: init_params_[k] for k in ["init_mesh_"]}

    # Store init params for diagnostic plot
    params_start = init_params_.copy()

    print(f"Warming mesh ({cfg['mcmc'].get('mesh_warmup_steps', 2**13)} steps, cosmo/bias fixed)...", flush=True)
    import time as _time
    _t0 = _time.time()
    warmup_mesh_fn = pmap(jit(get_mclmc_warmup(
        model.logpdf,
        n_steps=cfg["mcmc"].get("mesh_warmup_steps", 2**13),
        config=None,
        desired_energy_var=1e-6,
        diagonal_preconditioning=True,
    )))

    _t0 = _time.time()
    state_mesh, config_mesh = warmup_mesh_fn(jr.split(jr.key(43), num_chains), init_mesh_)
    # Force sync to get accurate wall time
    jnp.array(0.).block_until_ready()
    print(f"  [mesh warmup call: {_time.time()-_t0:.1f}s]", flush=True)

    print("✓ Mesh warmup done", flush=True)
    print(f"  Logdens (median): {float(jnp.median(state_mesh.logdensity)):.2f}", flush=True)
    print(f"  L (median): {float(jnp.median(config_mesh.L)):.6f}", flush=True)
    print(f"  step_size (median): {float(jnp.median(config_mesh.step_size)):.6e}", flush=True)

    # Update only init_mesh_ from mesh warmup, keep cosmo/bias from initial params
    init_params_["init_mesh_"] = state_mesh.position["init_mesh_"]
    init_params_ = {k: v for k, v in init_params_.items() if k not in fixed_latent_keys}

    # STEP 2: Warmup all params
    print("\n" + "=" * 80, flush=True)
    print("STEP 2: WARMUP ALL PARAMS", flush=True)
    print("=" * 80, flush=True)

    model.reset()
    apply_conditioning()

    if os.environ.get("DIAGNOSE_FREEZE"):
        from desi_cmb_fli.validation import diagnose_freeze
        diagnose_freeze(model, init_params_, fig_dir)
        sys.exit(0)

    print(f"Warming all params ({num_warmup} steps)...", flush=True)
    print(
        f"  Initial scalar params: fiducial (Omega_m={model.loc_fid['Omega_m']:.4f}, "
        f"sigma8={model.loc_fid['sigma8']:.4f}, b1={model.loc_fid['b1']:.2f}, "
        "b2/bs2/bn2 from latents)",
        flush=True,
    )
    _t0 = _time.time()
    if bool(cfg["mcmc"].get("scalar_precond", False)):
        from desi_cmb_fli.samplers import scalar_precond_mass
        _pos0 = {k: jax.device_put(np.asarray(v)[0]) for k, v in init_params_.items()}
        _scalar_keys = [k for k in _pos0 if k != "init_mesh_"]
        print("Building scalar-curvature preconditioner (mesh=1, scalars=1/curv)...", flush=True)
        _inv_mass, _ = scalar_precond_mass(model.logpdf, _pos0, _scalar_keys)
        _init_config = {
            "L": float(jnp.median(config_mesh.L)),
            "step_size": float(jnp.median(config_mesh.step_size)),
            "inverse_mass_matrix": _inv_mass,
        }
        warmup_all_fn = pmap(jit(get_mclmc_warmup(
            model.logpdf,
            n_steps=num_warmup,
            config=_init_config,
            desired_energy_var=desired_energy_var,
            diagonal_preconditioning=True,
        )))
    else:
        warmup_all_fn = pmap(jit(get_mclmc_warmup(
            model.logpdf,
            n_steps=num_warmup,
            config=None,
            desired_energy_var=desired_energy_var,
            diagonal_preconditioning=diagonal_precond,
        )))

    _t0 = _time.time()
    state, config = warmup_all_fn(jr.split(jr.key(43), num_chains), init_params_)
    jnp.array(0.).block_until_ready()
    print(f"  [all-params warmup call: {_time.time()-_t0:.1f}s]", flush=True)

    print("✓ Full warmup done")

    # ========================================================================
    # WARMUP DIAGNOSTIC PLOTS (Power/Transfer/Coherence)
    # ========================================================================
    print("\n" + "=" * 80)
    print("WARMUP DIAGNOSTIC: POWER/TRANSFER/COHERENCE")
    print("=" * 80)

    from desi_cmb_fli.validation import plot_warmup_diagnostics
    plot_warmup_diagnostics(model, state, params_start, truth, fig_dir)

    # Save raw per-chain values
    raw_L = jnp.array(config.L)
    raw_step_size = jnp.array(config.step_size)

    median_L_adapted = float(jnp.median(raw_L))
    median_ss = float(jnp.median(raw_step_size))
    median_imm = jnp.median(config.inverse_mass_matrix, axis=0)
    print(f"  Logdens (median): {float(jnp.median(state.logdensity)):.2f}")
    print(f"  Adapted L (median): {median_L_adapted:.6f}")
    print(f"  step_size (median): {median_ss:.6f}")

    if median_ss < 1.0:
        print("\n⚠️  WARNING: Step size is very small! This indicates poor conditioning or mixing issues.")
        print("   Check if diagonal_preconditioning is enabled or if priors are too tight.")


    # Recalculate L (benchmark approach)
    eval_per_ess = 1e3
    recalc_L = float(0.4 * eval_per_ess / 2 * median_ss)

    devices = jax.local_devices()[:num_chains]
    config = MCLMCAdaptationState(
        L=jax.device_put_replicated(jnp.array(recalc_L), devices),
        step_size=jax.device_put_replicated(jnp.array(median_ss), devices),
        inverse_mass_matrix=jax.device_put_replicated(median_imm, devices),
    )

    print(f"\n  Recalculated L: {recalc_L:.6f} (was: {median_L_adapted:.6f})")

    jnp.savez(config_dir / "warmup_state.npz", **state.position)
    utils.ydump(
        {"L": recalc_L, "step_size": median_ss, "eval_per_ess": eval_per_ess},
        config_dir / "warmup_config.yaml",
    )

    # Save sampler state for resume (allows resuming even if job crashes before first batch)
    with open(config_dir / "sampler_state.pkl", "wb") as f:
        pickle.dump({"state": state, "config": config}, f)
    print("  ✓ Saved sampler state for resume")

    # ========================
    # WARMUP VALIDATION TESTS
    # ========================
    print("\n" + "=" * 80)
    print("WARMUP VALIDATION")
    print("=" * 80)

    # Test 1: Step size consistency across chains
    ss_std = float(jnp.std(raw_step_size))
    ss_mean = float(jnp.mean(raw_step_size))
    ss_rel_std = ss_std / ss_mean if ss_mean > 0 else float('inf')
    print("\n1. Step Size:")
    print(f"   Values: {raw_step_size}")
    print(f"   Mean: {ss_mean:.6f}, Std: {ss_std:.6f}, Rel Std: {ss_rel_std:.4f}")

    # Test 2: Logdensity spread (info only - expected to vary after warmup)
    logdens = jnp.array(state.logdensity)
    ld_std = float(jnp.std(logdens))
    print("\n2. Logdensity Spread:")
    print(f"   Values: {logdens}")
    print(f"   Std: {ld_std:.2f}")

    # Test 3: Post-warmup scalar params — physical values via std2trunc
    from desi_cmb_fli.utils import std2trunc as _std2trunc
    print("\n3. Scalar Parameters (post-warmup):")
    abacus_truth = cfg.get("abacus_truth_params", {})
    closure_truth = cfg.get("truth_params", {})
    _scalar_table = ["Omega_m_", "sigma8_", "b1_", "b2_", "bs2_", "bn2_", "bnpar_", "fNL_", "fNL_bp_", "fNL_bpd_", "s_e_"]
    _scalar_table += [f"ngbar_{b}_" for b in range(int(getattr(model, "n_gxy_shells", 0)))]
    for param in _scalar_table:
        if param not in state.position:
            continue
        base_name = param.rstrip("_")
        lat_cfg = model.latents.get(base_name)
        if lat_cfg is None:
            continue
        latent_vals = jnp.array(state.position[param])   # shape (num_chains,)
        low = lat_cfg.get("low", -jnp.inf)
        high = lat_cfg.get("high", jnp.inf)
        if low == -jnp.inf and high == jnp.inf:
            phys_vals = [float(v * lat_cfg["scale_fid"] + lat_cfg["loc_fid"]) for v in latent_vals]
        else:
            phys_vals = [float(_std2trunc(v,
                                          loc=lat_cfg["loc_fid"],
                                          scale=lat_cfg["scale_fid"],
                                          low=low,
                                          high=high))
                         for v in latent_vals]
        fid_val   = lat_cfg["loc_fid"]
        phys_str  = "  ".join(f"{v:.4f}" for v in phys_vals)
        print(f"   {base_name:8s}: [{phys_str}]  (fid={fid_val:.4f})")


# STEP 3: Multi-Chain Mini-Batch Sampling
print("\n" + "=" * 80)
print("STEP 3: MULTI-CHAIN MINI-BATCH SAMPLING")
print("=" * 80)

# Condition model for sampling
model.reset()
apply_conditioning()

# Setup sampling function (pmap for parallel chains)
run_fn = pmap(jit(get_mclmc_run(model.logpdf, n_samples=num_samples, thinning=thinning, progress_bar=False)))

print(f"\nRunning {num_chains} chains in parallel, each with {num_batches} sequential batches")
print(f"   Samples per batch: {num_samples}")
print(f"   Total per chain: {num_samples * num_batches}")
print(f"   Total all chains: {num_samples * num_batches * num_chains}")
print(f"   Total evaluations per chain: {num_samples * num_batches * thinning}")
if RESUME_MODE:
    print(f"   Starting from batch: {start_batch}")
print()



# Storage
samples_scalars = {}  # For small parameters (scalars)
param_names = None

# Optional: control saving of large fields
mcmc_io_cfg = cfg.get("mcmc", {})
save_large_fields = bool(mcmc_io_cfg.get("save_large_fields", False))

# Load existing batches if resuming
if RESUME_MODE and start_batch > 0:
    print(f"Loading {start_batch} existing batches...")
    large_params = {
        "init_mesh_", "init_mesh",
        "gxy_mesh", "matter_mesh",
        "lpt_pos", "rsd_pos", "nbody_pos",
        "a_part", "obs"
    }
    for batch_idx in range(start_batch):
        batch_file = config_dir / f"samples_batch_{batch_idx}.npz"
        batch_data = jnp.load(batch_file)
        if param_names is None:
            param_names = list(batch_data.files)
            scalar_params = [p for p in param_names if p not in large_params]
            for p in scalar_params:
                samples_scalars[p] = []
        for p in scalar_params:
            if p in batch_data.files:
                samples_scalars[p].append(batch_data[p])
        print(f"  Loaded {batch_file.name}")

key = jr.key(42 + start_batch * 1000)  # Offset key for reproducibility when resuming

for batch_idx in range(start_batch, num_batches):
    print(f"Batch {batch_idx + 1}/{num_batches}:")

    # Run sampling for this batch (parallel across chains)
    keys = jr.split(key, num_chains + 1)
    key, run_keys = keys[0], keys[1:]
    state, samples_dict = run_fn(run_keys, state, config)

    # Identify parameters on first batch
    if param_names is None:
        param_names = [k for k in samples_dict.keys() if k not in ["logdensity", "mse_per_dim"]]
        # Identify large fields
        large_params = {
            "init_mesh_", "init_mesh",
            "gxy_mesh", "matter_mesh",
            "lpt_pos", "rsd_pos", "nbody_pos",
            "a_part", "obs"
        }
        scalar_params = [p for p in param_names if p not in large_params]

        # Initialize storage for scalars
        for p in scalar_params:
            samples_scalars[p] = []

    # Process batch
    batch_scalars = {}
    batch_large = {}

    cpu_samples = jax.device_get(samples_dict)
    for p in param_names:
        val = cpu_samples[p]
        if p in scalar_params:
            samples_scalars[p].append(val)
            batch_scalars[p] = val
        elif save_large_fields:
            batch_large[p] = val

    # Save batch to disk
    batch_content = {**batch_scalars, **batch_large}
    jnp.savez(config_dir / f"samples_batch_{batch_idx}.npz", **batch_content)

    # Save sampler state for resume
    with open(config_dir / "sampler_state.pkl", "wb") as f:
        pickle.dump({"state": state, "config": config}, f)

    print(f"  ✓ {num_samples} samples × {num_chains} chains")

    # Show MSE per dim for EACH chain
    mse_per_chain = jnp.mean(samples_dict['mse_per_dim'], axis=1)  # shape: (num_chains,)
    print(f"  MSE/dim per chain: {mse_per_chain}")
    print(f"  MSE/dim (median):  {float(jnp.median(mse_per_chain)):.6e}")
    print(f"  Logdensity (median): {float(jnp.median(state.logdensity)):.2f}")
    print(f"  Saved: samples_batch_{batch_idx}.npz")

    # Energy variance validation on first batch of this run
    if batch_idx == start_batch:
        diag_lines = []
        for i, mse_val in enumerate(mse_per_chain):
            ratio = float(mse_val) / desired_energy_var
            status = "✓" if ratio < 2.0 else "⚠️"
            line = f"\n  📊 Chain {i} Energy Variance: {float(mse_val):.2e} (ratio: {ratio:.2f}) {status}"
            print(line)
            diag_lines.append(line)

        median_mse = float(jnp.median(mse_per_chain))
        median_ratio = median_mse / desired_energy_var
        line = f"\n  📊 Overall (median): {median_mse:.2e}, Desired: {desired_energy_var:.2e}, Ratio: {median_ratio:.2f}"
        print(line)
        diag_lines.append(line)
        if median_ratio < 2.0:
            line = "     ✓ PASS: Energy variance matches desired"
        else:
            line = f"     ⚠️  WARN: Ratio = {median_ratio:.1f}x - consider reducing desired_energy_var"
        print(line)
        diag_lines.append(line)

        with open(config_dir / "energy_variance_diag.txt", "w") as _f:
            _f.write(f"step_size per chain: {list(jnp.array(config.step_size))}\n")
            _f.write("".join(diag_lines) + "\n")

    # Explicitly delete large arrays to free memory
    del batch_large, batch_content, batch_scalars
    gc.collect()

print("\n✓ All batches complete!")

# Merge batches
print("\n" + "-" * 40)
print("ANALYSIS")
print("-" * 40)

print("\nRunning final analysis...")
analyze_run(run_dir, burn_in=0.5)

print("\nAnalysis complete. Check figures folder.")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nOutputs: {run_dir}")
print(f"  Config: {config_dir}")
print(f"  Figures: {fig_dir}")
print(f"  Samples: {config_dir / 'samples.npz'}")
print(f"  Truth: {config_dir / 'truth.npz'}")
if cmb_enabled:
    print("\n✓ Joint inference: Galaxies + CMB lensing")
else:
    print("\n✓ Galaxy-only inference")
print("\n" + "=" * 80)
end_time = datetime.now()
duration = end_time - start_time
print(f"Total run time: {duration}")
print("DONE")
print("=" * 80)
