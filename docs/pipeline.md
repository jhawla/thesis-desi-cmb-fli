# GALAXY Clustering × CMB Lensing Field-Level Inference

Field-level Bayesian inference of cosmology, primordial non-Gaussianity, galaxy bias, and the initial density field from galaxy clustering **jointly with** CMB-lensing convergence (κ) maps. The forward model evolves a
Gaussian initial field to a galaxy overdensity and a κ map; an MCLMC sampler conditions on the data
(closure or AbacusSummit) to constrain the parameters and the field.

---

## 1. Overview

![Field-Level Inference Architecture](figures/fli_architecture.png)

**Dataflow.** initial field δ_ini → gravitational evolution (LPT/N-body) → galaxy bias + RSD → galaxy
overdensity δ_g; the same evolved matter field is Born-integrated along the line of sight → κ map.
Both observables enter a joint likelihood compared against the data (illustrated here with future real DESI × Planck/ACT data).

**Modules.**
- `desi_cmb_fli.model` — `FieldLevelModel`: probabilistic model, forward pass (`evolve`), likelihood,
  reparametrization, preconditioning, config loader (`get_model_from_config`).
- `desi_cmb_fli.bricks` — bias, RSD, PNG, painting, coordinate/geometry helpers.
- `desi_cmb_fli.nbody` — growth/distance background (emulated), LPT/N-body integrators, Fourier utils.
- `desi_cmb_fli.cmb_lensing` — Born projector, HEALPix masks, κ/galaxy data loaders, theory spectra.
- `desi_cmb_fli.metrics` — power/angular spectra, MASTER decoupling, log-binning.
- `desi_cmb_fli.samplers` — MCLMC / NUTS, analytic scalar preconditioner.
- `desi_cmb_fli.validation`, `desi_cmb_fli.chains`, `desi_cmb_fli.plot` — diagnostics & plotting.

---

## 2. Forward model

All forward-model grids share the same physical box; only the cell resolution changes (see the
oversampling table in §2.5).

### 2.1 Initial conditions

A 3-D Gaussian random field on the **init grid** (`init_oversamp`, e.g. 48³ for a 32³ final grid).
The power spectrum is `jax_cosmo` (Eisenstein–Hu transfer). The field is the inference latent
(`white_mesh`), sampled in a Kaiser-whitened basis (see §5). Mesh dimensions are auto-adjusted so all
axes have an **even** number of cells (MCLMC requirement).

### 2.2 Primordial non-Gaussianity

Local PNG has two effects, both controlled by `model.png_type`:

- **Matter φ² term** (`bricks.add_png`, always uses `fNL`): in the primordial potential
  φ → φ + f_NL(φ² − ⟨φ²⟩), mapped back to the density field via the φ→δ transfer. Applied on the evol
  grid, then **re-band-limited** to the init Nyquist (`chreshape` to init_shape then back) to drop the
  spurious φ² power above the IC resolution ("PNG anti-aliasing").
- **Scale-dependent galaxy bias** amplitudes `fNL_bp = f_NL b_φ`, `fNL_bpd = f_NL b_{φδ}`, entering
  `lagrangian_weights` / `kaiser_boost` as Δb(k) ∝ 1/M(k), with M(k) the φ→δ conversion.

`png_type` values:
- `None` — Gaussian (no PNG).
- `'fNL'` — **universality**: `b_φ = 2δ_c(b₁+1−p)`, `b_{φδ} = 2(δ_c b₂ − b₁)`, so `f_NL` is the only
  free PNG parameter.
- `'fNL_bias'` — infers `fNL`, `fNL_bp`, `fNL_bpd` as **free** latents (group `png`), not derived from
  b₁,b₂. `add_png` still uses `fNL` for the matter channel.

### 2.3 Gravitational evolution

The init field is `chreshape`d up to the **evol grid** (`evol_oversamp`) and evolved with:
- **LPT** (1st/2nd order, default `lpt_order: 2`), or
- **N-body** (BullFrog integrator, `diffrax`) — snapshot only, no lightcone.

**Lightcone** (`lightcone: true`, `a_obs: null`): each particle evolves at its local scale factor
`a(χ)` from its comoving radius; snapshot mode uses a scalar `a_obs`.

**Background emulator (`nbody.BackgroundEmulator`).** Growth `D`, `D₂`, rates `f`, `f₂`, and the
distance maps `χ(a)`/`a(χ)` are served by a **bilinear emulator** rather than the `jax_cosmo` ODE
solvers, whose in-graph CPU callbacks otherwise exhaust LLVM memory on large MCMC runs. Tables are
precomputed once on CPU over `n_Om=100` values of `Ω_m ∈ [0.05, 0.7]` (`χ` on `logspace(-4,0,512)`),
then bilinearly interpolated in `(Ω_m, a)`. Activated automatically when the fixed background matches
the Abacus fiducial (`_is_abacus_background`: `Ω_b, h, n_s, w0=−1, wa=0, Ω_k=0`) — i.e. the Abacus
runs where only `Ω_m` (and `σ8`, which does not enter the background) vary. Otherwise the exact
`jax_cosmo` background is used. Exact in `σ8`; accurate to the `Ω_m` grid spacing (~0.007).

### 2.4 Galaxy bias & redshift-space distortions

**Lagrangian bias expansion** (Modi+2020), weights read at the initial particle positions from the
evol-grid **Gaussian** field:
- linear `b₁`, quadratic `b₂`, tidal shear `b_{s²}`, higher-derivative `b_{∇²}` (`bn2`).
- `b₂` uses the montecosmo convention `weights += b₂·(δ²−⟨δ²⟩)/2`.

**Finger-of-God higher-derivative LOS bias `bnpar` (b_∇∥).** Implemented as a **velocity** term, not a
painting weight. `bricks.lagrangian_fog_velocity` returns `dvel = bnpar·∇δ_L·D(a)` as a
full 3-vector at the initial positions. `bricks.rsd` adds it to the peculiar velocity **before** the
LOS projection `(v·n̂)n̂`, so the projection turns `∇δ` into the effective `∇²_∥δ` FoG term. The
projection LOS `n̂` is the **observer-dependent per-particle line of sight** (`los_part` from
`tophysical_pos`, using `box_center = box_shape/2 − observer_position`), so `bnpar` respects the
observer geometry automatically. It is only constrained with RSD on (`los` set); for snapshot/no-RSD
runs the FoG FFTs are skipped and `bnpar` must stay in `mcmc.fixed_params`.

**RSD** (`bricks.rsd`): peculiar velocity from the growth-time integrator (`v ∝ D·f`), projected onto
the local LOS; on the lightcone the LOS and `a` are per-particle. `los: null` disables RSD.

### 2.5 Painting & anti-aliasing (oversampling)

Particles are painted to the galaxy/matter mesh with **interlaced order-2 CIC + Fourier deconvolution**
at the oversampled `paint` resolution, then Fourier-cropped (`chreshape`) to the final grid
(`model.paint_and_deconv`, `bricks.interlace_paint_deconv`). The galaxy mesh is divided by
`prod(ptcl_shape)/prod(final_shape)` to recover `1 + δ_g` (robust to particle count).

| Grid | Factor (montecosmo) | Carries |
|------|---------------------|---------|
| init  | `init_oversamp` (3/2) | the inferred linear field; preconditioner & `kaiser_post` build on it |
| evol  | `evol_oversamp` (7/4) | LPT/N-body evolution, all bias products (δ², s², ∇²δ, φδ), `add_png` |
| ptcl  | `ptcl_oversamp` (7/4) | particle cloud density (`regular_pos(evol_shape, ptcl_shape)`) |
| paint | `paint_oversamp` (7/4)| interlaced+deconvolved CIC paint grid, cropped to final |

Flow in `FieldLevelModel.evolve`: init field → chreshape→evol → `lagrangian_weights` on the evol
Gaussian field → `add_png` + PNG re-band-limit → LPT displacement → paint+crop to final.

### 2.6 CMB lensing convergence

`cmb_lensing.convergence_Born_spherical` Born-integrates the evolved matter field over radial shells
**directly on HEALPix pixels** (curved-sky). Per shell, particles are painted to the shell support,
converted to δ, and accumulated with the lensing kernel
$$W_\kappa(\chi,a) = \tfrac{3}{2}\,\Omega_m\left(\tfrac{H_0}{c}\right)^2 \frac{\chi}{a}\,\frac{\chi_s-\chi}{\chi_s}.$$
Ray–box intersection intervals (`t_enter`, `t_exit`) restrict the sum to physically supported
shell/pixel contributions. The mean density `n̄` is derived from the actual particle count
(`pos.shape[0]`), so the normalisation is correct under particle oversampling.

**Line-of-sight depth.** `kappa_pred` integrates matter only to `box_shape[2]`; the residual depth
`χ_box → χ_high_z_max` is handled analytically in the likelihood covariance (§3.2). For AbacusSummit
base, `chi_high_z_max: 3990` Mpc/h (matter simulated to z≈2.45), not `χ_CMB`.

---

## 3. Likelihood

### 3.1 Galaxy likelihood

A per-cell Gaussian on the overdensity, `obs ~ Normal(gxy_mesh, √(s_e²/n̄))`, restricted to the survey
mask via `dist.mask`. Variance is the shot-noise `s_e²/n̄` with `s_e=1` ⇒ pure Poisson.

**Free shot-noise amplitude `s_e` (`model.gxy_stoch_noise`, default off).** An optional positive latent
(group `stoch`, truncated-normal `loc=1, scale=1, [0,10]`) rescaling the shot-noise std. It
marginalises non-Poisson model/stochastic scatter (which the 2-LPT + Lagrangian-bias model cannot match
exactly at the field level) so it does not leak into `f_NL`. Following montecosmo we keep `s_e = 1` fixed in
the reference runs.

**Survey mask & radial selection from randoms.** For the AbacusLensing LRG mocks (equal-weight
galaxies and ~20× randoms), the mesh is `1+δ = count/n̄_{3D}`, with `n̄_{3D} = n_eff·S(x)` and the
selection `S` = the **painted random density** `rand_mesh_plain` (normalised to unit mean on the mask;
no smoothing needed — randoms are densely sampled, ~2400/cell at 32³). The mask is the random occupancy
with a **completeness cut `S > 0.8`**: partial boundary cells (octant
faces + radial shell edges) are only fractionally filled, biasing `count/n̄` low and leaking into the
large-scale modes / `f_NL`; dropping them removes the obs–completeness correlation and
restores ⟨obs⟩→1, keeping ~80% of cells. The observed field is painted with the **same**
interlaced+deconvolved scheme as the model.

**Free per-radial-bin mean density `ngbars` (`model.gxy_ngbar_free`, integral constraint).** On the
lightcone the radial selection makes the survey mean density a per-bin unknown; fixing it exactly
imposes an integral constraint whose violation leaks low-k power into `f_NL`. Following montecosmo
(`set_radial_count`), one free **relative** amplitude `ngbar_<b>` (1 = fiducial) is added per **fine
uniform radial bin**:
- `get_model_from_config` sets `n_rbins = round((χ_hi−χ_lo)/dr)`, `dr = √3·cell_size` (χ-range from the
  catalog z-band extent under the fiducial cosmology), and injects `ngbar_0 … ngbar_{n−1}`
  (group `syst`, prior `loc=1, scale=1, scale_fid=1e-2, low=0`). Off by default.
- The loader builds `model.gxy_shell_id` by digitising the survey comoving radius into `n_rbins`
  uniform bins over `[r_min, r_max]` (−1 outside the survey).
- In the likelihood, `α = ngbar[bin_id]` rescales prediction **and** shot-noise per bin:
  `obs ~ Normal(α·gxy_mesh, √(α·s_e²/n̄))` — the overdensity-space equivalent of `set_radial_count`
  (which multiplies both the count and the selection). At `α=1` it reduces to the previous likelihood.

Mathematically, if the true per-bin density is `α_b ×` fiducial, then `E[obs] = α_b·gxy_mesh`,
`Var[obs] = α_b/n̄`; marginalising `α_b` over **fine** bins removes the entire radial-monopole subspace,
so `f_NL` is measured from transverse/3-D modes only and is insensitive to a mis-traced `n̄(χ)`.

### 3.2 CMB lensing likelihood

A single numpyro `Normal` site over the observed footprint (so `sample == log_prob` for closure). The
total per-mode variance is `C_ℓ = N_ℓ + C_ℓ^{high-z}` (reconstruction noise + the unmodeled high-z
contribution treated as structured Gaussian variance). Two modes (`cmb_lensing.likelihood_mode`):

- **`diagonal`** (default): the masked pseudo-$a_{\ell m}$ vector with diagonal covariance
  `var_ℓ = M_{ℓℓ'}(N_{ℓ'}+C_{ℓ'}^{high-z})`, `M` the MASTER mode-coupling matrix (NaMaster,
  `model.cmb_M_ll`). **Exact on the full sky** ($W=1$: pseudo-$a_{\ell m}$ = true $a_{\ell m}$). On a
  cut sky it drops the mask-induced (ℓ,ℓ')/(m,m') coupling, so it is approximate (can mis-size error
  bars) but fast, and the only option for full-sky / high-nside runs. Includes the parameter-dependent
  log-determinant `ln C_ℓ` (needed in `taylor`/`exact` high-z modes where `C_ℓ^{high-z}` depends on
  Ω_m,σ8).

- **`pixel_exact`**: the **exact cut-sky** field-level likelihood via signal-eigenmode (KL)
  compression. The band-limited noise field's pixel covariance depends only on angular separation,
  $$\mathrm{Cov\_pix}[i,j] = \sum_\ell \tfrac{2\ell+1}{4\pi}\,(N_\ell+C_\ell^{high-z})\,P_\ell(\cos\theta_{ij})$$
  (**bare spectrum, no pixel window** — this makes `diagonal` and `pixel_exact` coincide at
  $f_{sky}=1$). Because the field is band-limited to $\ell_{max}=2\,$nside, a footprint of area
  $f_{sky}$ supports only $\sim f_{sky}(\ell_{max}+1)^2$ modes (Slepian) while HEALPix gives $\sim3\times$
  more pixels, so `Cov_pix` is **rank-deficient**. We diagonalise it **once** (constant: fixed
  cosmo+mask+noise), keep the $k$ eigenmodes with $\lambda > \mathrm{kl\_rcond}\cdot\lambda_{max}$ (the
  supported Slepian subspace), and use the eigenmode amplitudes $a = U_k^\top\kappa[\mathrm{obs}]$: in
  that basis the covariance is exactly diagonal ($\Lambda_k$), so the site is a single
  `Normal(U_k^\top\kappa_{pred}, \sqrt{\Lambda_k T})`. The full mask coupling is retained exactly within
  the supported subspace; only numerically-null modes are dropped (lossless). This supersedes
  `diagonal` on a cut sky (which is over-confident there). Requires **fixed cosmology and
  `high_z_mode: fixed`** (constant covariance). `kl_rcond` (default `1e-6`) sets the cutoff: it keeps
  cond ≈ `1/rcond` — safe in float32 down to ~`1e-6`; larger (e.g. `1e-2`, conditioning ~10²) is more
  conservative/robust (drops the least-concentrated Slepian modes) without biasing the calibration.

**High-z correction (`full_los_correction`, `high_z_mode`).** `C_ℓ^{high-z}` for the missing depth
`χ_box → χ_high_z_max` is the Limber convergence power (`jax_cosmo`), added to the variance. Modes:
- `fixed` — cached at the fiducial cosmology (required by `pixel_exact`, exact if `Ω_m,σ8` fixed).
- `taylor` (default) — first-order expansion `C_ℓ(θ) ≈ C_ℓ(θ_fid) + ∇C_ℓ·Δθ`, gradients precomputed.
- `exact_linear` — recompute the Limber integral each step with the **linear** P(k) (slow).
- `exact` — full recompute each step (very slow).

Noise: ACT DR6 `N_ℓ^{κκ}` (`data/N_L_kk_act_dr6_lensing_v1_baseline.txt`, columns ℓ, N_ℓ) or Planck
PR4; `cmb_noise_scaling` (default 1.0) multiplies it to test sensitivity. Noise realizations:
`sample_healpix_gaussian`; map RMS: `compute_sigma_hp`.

### 3.3 Geometry & observer

- **Observer** (`cmb_lensing.observer_mode`: `center` / `face` / `corner`, or explicit
  `observer_position`): sets `box_center = box_shape/2 − observer_position`. `corner` places the
  AbacusSummit base-box octant footprint inside `[0,L]³` at full depth. The 3-D galaxy mesh and the 2-D
  κ map share this observer (galaxies at their true (RA,DEC,Z), Born ray-cast from the same point), so
  both probes cover the same lightcone volume — no box rotation (the box is axis-aligned).
- **Curved-sky scope** (`curved_sky`): galaxy LOS/RSD and CMB projection; CMB lensing is always
  curved-sky HEALPix.
- **Even mesh**: all axes forced even (MCLMC).

---

## 4. Data & observation modes

`observation_mode` (`config.yaml`, `utils.ObservationMode`) supports **exactly two values**,
`closure` and `abacus`. **Real observational data (DESI-LRG × Planck/ACT κ) is not yet supported** —
it is a roadmap item (§8), not a runnable mode; the only external data the pipeline ingests today is
AbacusSummit.

- **`closure`** — synthetic `obs`/`kappa_obs` generated from `truth_params` via `model.predict`
  (validation; data-generation and likelihood share the same distribution object).
- **`abacus`** — external AbacusSummit data. Optional deps: `pip install desi-cmb-fli[abacus]`
  (`healpy`, `asdf`, `abacusutils`). Three sub-modes:

| Mode | `galaxies_enabled` | `cmb_lensing.enabled` | Data |
|------|:--:|:--:|------|
| CMB-only | false | true | AbacusSummit κ map |
| Galaxy-only | true | false | galaxy catalog(s) |
| **Joint** | **true** | **true** | **both** |

**Primary mode: joint analysis on the AbacusSummit base box (`c000_ph000`).** The default `config.yaml`
targets the AbacusLensing **base** simulation, which is the only one carrying **both probes** on the
same lightcone volume — the LRG galaxy catalog over the octant footprint and the `kappa_00047.asdf` κ
map, whose footprint is **two small patches** (f_sky ≈ 4.5%) sitting inside the galaxy octant (see the
known limitations in §8). This joint galaxy × κ run (`galaxies_enabled: true`,
`cmb_lensing.enabled: true`, `observer_mode: corner`, `chi_high_z_max: 3990`) is the current headline
configuration; the galaxy-only and CMB-only sub-modes are used to cross-check the two probes
independently. The two special-geometry setups below are validation configurations, not the main
analysis.

**Two special-geometry AbacusSummit configurations** (both are `abacus` mode, they only differ in which
probe is enabled and which geometry flags are set):

- **CubicBox snapshot (galaxy-only, no lightcone).** A single-file periodic-box snapshot
  (`AbacusSummit_*_c000_ph000` cubic box at fixed z, auto-detected by `_is_cubic_box_catalog`). Here the
  observer/lightcone/curved-sky machinery is meaningless, so this mode **requires**
  `curved_sky: false`, `lightcone: false`, `los: null` (RSD off), `cmb_lensing.enabled: false`
  (galaxies only), and `a_obs` set to the snapshot scale factor. This is the montecosmo-parity
  validation setup (§8).
- **HUGE full-sky κ (CMB-only).** The `AbacusSummit_huge_c000_ph201/kappa_00045.asdf` map is a full-sky
  (`f_sky = 1`) CMB-lensing convergence map. Running it **requires disabling the galaxies**
  (`galaxies_enabled: false`, CMB-only), with `observer_mode: center` (full sky), `chi_high_z_max: 3942`,
  and `likelihood_mode: diagonal` (exact on the full sky — `pixel_exact` is a cut-sky construction and
  is not needed here). It is the clean full-sky CMB-only validation of the Born projector and the
  high-z covariance.

**Abacus κ ingestion** (`prepare_abacus_kappa_hp`, `load_abacus_kappa_observation`): the HEALPix map is
`ud_grade`d to the model nside and masked with the effective footprint (sim & external mask) so
`kappa_obs` and `kappa_pred` share support. We use `kappa_00047.asdf` (base `c000_ph000`,
`SourceRedshift = 1089.3`, CMB); its matter shells are simulated only to z≈2.45 (χ≈3990), hence
`chi_high_z_max: 3990`.

**Abacus galaxy loading** (`load_abacus_galaxy_observation`): a list of per-z-shell ASDF files
(`RA, DEC, Z_COSMO/Z_RSD`, `RAND_*`) via `bricks.catalog2positions`/`randoms2positions`, deduplicated
across overlapping shells by `Z_COSMO`, accumulated and painted once. A single-file **CubicBox
snapshot** (`x,y,z`, no `RA`) is auto-detected (`_is_cubic_box_catalog`) and routed to
`_load_abacus_cubic_box`: full periodic box, `selec=1`, uniform n̄, no randoms/observer (config
`lightcone: false`, `a_obs` at the snapshot z, `curved_sky: false`, `los: null`). `abacus_galaxy.file`
is a single path (snapshot) or a list of shell paths (lightcone).

---

## 5. Inference & sampling

**Sampler.** MCLMC (default) or NUTS (`desi_cmb_fli.samplers`), over the field latent + scalar
latents, in a reparametrized (whitened) space.

**Cosmology: inferable but fixed by default.** The cosmological parameters `Omega_m` and `sigma8` are
ordinary latents and **can be inferred** jointly with the field and biases (they carry priors in
`latents`, and the background emulator §2.3 covers the varying-`Ω_m` case). However, **the current
scientific direction is to fix them** and concentrate the constraining power on the primordial
non-Gaussianity parameters (§2.2). In practice this means: put `Omega_m, sigma8` in
`mcmc.fixed_params` (the default `config.yaml` ships `fixed_params: [Omega_m, sigma8]`) **and** run the
high-z correction in the fixed-cosmology mode (`cmb_lensing.high_z_mode: fixed`, which caches
`C_ℓ^{high-z}` at the fiducial cosmology — this is also what `pixel_exact` requires, §3.2). The
`taylor`/`exact`/`exact_linear` high-z modes exist precisely for the case where cosmology *is* varied,
so that `C_ℓ^{high-z}` tracks `Ω_m, σ8`; they are unnecessary once cosmology is fixed.

**Kaiser preconditioning** (`precond: kaiser`/`kaiser_dyn`). The field is sampled in a whitened basis
with `scale = √(1 + n_gal^{eff}·b_E²·P(k))`, `transfer = √P(k)/scale`. In galaxy/joint modes the warmup
initial field is a "reverse-Kaiser" estimate from the galaxy data; in **CMB-only** mode
(`n_gal^{eff}=0`) `scale=1`, `transfer=√P(k)` (pure prior — the field is constrained only through κ),
the initial field is a random Gaussian draw, and the biases are fixed to fiducial (galaxy calculations
skipped).

**Warmup steps.** STEP 1 warms the mesh with cosmo/bias fixed; STEP 2 frees the scalar latents; a
median-collapse then homogenises per-chain configs for STEP 3 sampling.

**Joint scalar preconditioning (`mcmc.scalar_precond`).** In the joint run, the bias scalars have
likelihood curvature ~1e6–1e7 while the whitened field is ~unit (ratio ~1e3–1e4), and the CMB κ
sharpens the bias geometry further. A diagonal-mass sampler cannot bootstrap this (it must estimate the
mass from chain variance, but the chain cannot step to gather it → `step_size → 0`). `scalar_precond`
breaks the deadlock by supplying the scalar mass **analytically**: `samplers.scalar_precond_mass`
computes each scalar's curvature (1-D second derivative via double autodiff) and builds
`inverse_mass_matrix` with `field = 1` (already whitened), `scalar = 1/|curv|`, passed as the STEP-2
initial config; blackjax then refines it per-chain. Pair with starting biases at their prior mode
(`loc`/`loc_fid`). Default off.

**Energy-variance tuning.** MCLMC performance depends on `mcmc.mclmc.desired_energy_var`. After the
first sampling batch, `run_inference.py` reports per-chain `MSE/dim` and its ratio to the target
(ratio < 2× acceptable). `5e-8`–`5e-7` works for realistic noise; use `5e-9` or lower for reduced-noise
runs (`cmb_noise_scaling: 0.01`).

---

## 6. Diagnostics & tools

- **`quick_pk_spectra.py`** — 3-D `P(k)` diagnostic. In **closure** mode: measured matter `P(k)` from
  the model's `matter_mesh` vs nonlinear `jax_cosmo` `P_mm` at `a_fid`. In **abacus** mode: galaxy
  `P_gg(k)` as seen by the likelihood, Abacus observed catalog vs LPT-simulated galaxies at the config
  cosmology (same survey mask, Poisson shot-noise subtracted).
- **`quick_cl_spectra.py`** — angular spectra `C_ℓ^{κκ}, C_ℓ^{gg}, C_ℓ^{κg}` vs theory, for closure (N LPT realizations, mean±std) and abacus (loaded maps + N noise draws) modes.
  Uses **resolution-aware Limber** (`compute_theoretical_cl_gg/kg`, `k_nyq` cut: integrate only shells
  with `k_⊥=(ℓ+0.5)/χ < k_Nyq = π/Δx`) and overlays the Poisson shot-noise floor
  `N_ℓ^{shot} = Ω_sky/N_gal` on the `C_ℓ^{gg}` panel; spectra are log-binned (`metrics.bin_cl_log`).
- **`analyze_run.py`** — merge batches, R-hat/ESS, corner/trace plots (abacus mode uses
  `abacus_truth_params` markers), custom burn-in / chain exclusion.
- **`compare_runs.py`** — GetDist triangle comparison of multiple runs (per-run burn-in, labels).
- **`plot_2D_maps.py`** — κ (and galaxy-projection) maps for one forward realization.
- **`benchmark_highz_cl_modes.py`** — precision/speed of the high-z modes; **`plot_lensing_fraction.py`**
  — box-captured lensing fraction vs depth; **`make_abacus_kappa_mask.py`** — degrade the AbacusLensing
  footprint to a model-nside `.npy`; **`plot_highz_correction_vs_depth.py`** — mean high-z κ correction
  amplitude vs box depth (`χ_min`); **`plot_linear_vs_nbody.py`** — linear (Kaiser) vs N-body evolved
  matter field from the same IC; **`plot_cmb_noise_comparison.py`** — ACT DR6 vs Planck PR4 κ noise
  spectra.
- **Freeze diagnostic** (`DIAGNOSE_FREEZE=1`): logpdf + per-group gradient + 1-D scans of each free
  scalar at the STEP-2 start; localises curvature pathologies. (Used to derive `scalar_precond`.)

---

## 8. Validation status, known limitations & roadmap

**Validated.** On the Abacus **CubicBox snapshot** (periodic full box, z=0.8), the
galaxy-only run reproduces montecosmo result — unbiased `f_NL` (≈ −111 ± 500).

**Current results (Abacus lightcone).**
- **Galaxy-only** is **unbiased**: with fine-bin `ngbars` + `png_type: fNL` (universality) +
  oversampling, `f_NL = 1.5 ± 33` (consistent with 0). Under `png_type: fNL_bias` the scale-dependent
  bias amplitude `fNL_bp ≈ 0` (the meaningful observable), while the decoupled matter-φ² `fNL` channel
  is weakly identified and can wander.
- **Joint (full galaxies + κ)** converges cleanly and is consistent with galaxy-only, but the CMB gain
  on `f_NL` and `b_1`is small: the galaxy footprint (~13% sky octant) already saturates the
  large-scale information that κ could add over its small footprint.

**Known limitations.**
- The AbacusSummit base κ footprint is small (two lobes, f_sky ≈ 4.5%) and sits inside the galaxy
  octant. Restricting galaxies to that footprint (to expose the κ gain) under-fills the box (~15% of
  cells) → the large-scale modes where `f_NL` lives become unconstrained and the chains do not
  converge. A tighter box is not available: the two-lobe geometry keeps any enclosing box ~50% empty,
  and the pipeline uses an axis-aligned box (no rotation).
- κ source z=1089 vs galaxies z=0.4–0.8 sets an intrinsic κ×g cross-correlation ceiling (r≈0.5–0.64);
  the high-z part is handled as covariance, not signal.

**Roadmap.**
1. Compare to a standard power-spectrum analysis (with A. de Mattia).
2. Full CMB-aware joint preconditioner (beyond the analytic scalar mass of `scalar_precond`).
3. Find another external simulation with a **larger κ footprint** and galaxies to expose the κ gain on `f_NL`.
4. Application to real DESI-LRG × Planck/ACT κ data.
