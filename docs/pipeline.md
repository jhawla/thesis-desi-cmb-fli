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
- `desi_cmb_fli.samplers` — MCLMC / NUTS warmup and sampling loops.
- `desi_cmb_fli.validation`, `desi_cmb_fli.chains`, `desi_cmb_fli.plot` — diagnostics & plotting.

---

## 2. Forward model

All forward-model grids share the same physical box; only the cell resolution changes (see the
oversampling table in §2.5).

### 2.1 Initial conditions

A 3-D Gaussian random field on the **init grid** (`init_oversamp`, e.g. 48³ for a 32³ final grid).
The power spectrum is `jax_cosmo` (Eisenstein–Hu transfer). The field is the inference latent
`init_mesh` (sampled as `init_mesh_` in the Kaiser-whitened basis, see §5). Mesh dimensions are
auto-adjusted (`get_model_from_config`) so all axes have an **even** number of cells: the real↔complex
Gaussian repacking `utils._rg2cgh`/`cgh2rg` asserts `all(shape % 2 == 0)`, because the
hermitian-symmetry bookkeeping (self-conjugate modes at 0 and Nyquist) assumes an exact Nyquist plane.

### 2.2 Primordial non-Gaussianity — theory

**Local-type PNG.** The primordial (matter-era) Bardeen potential is made non-Gaussian by

$$\Phi(\mathbf{x}) = \phi(\mathbf{x}) + f_\mathrm{NL}\left[\phi^2(\mathbf{x}) - \langle\phi^2\rangle\right],$$

with $\phi$ Gaussian. Note the normalisation of $\phi$ **in the matter era** — this is the
**CMB convention** of $f_\mathrm{NL}^\mathrm{loc}$; the LSS convention (normalising at $z=0$)
differs by $g(a\!\to\!0)/g(a\!=\!1) \simeq 1.28$. Any comparison to an external simulation or to
published constraints must state which convention is used.

**Potential → density transfer** (`bricks.trans_phi2delta_interp`):

$$\delta_L(k,a) = M(k,a)\,\Phi(k), \qquad
M(k,a) = \frac{2\,r_h^2\,k^2\,T(k)\,D_\mathrm{norm}(a)}{3\,\Omega_m},$$

with $r_h = c/H_0 = 2997.92\ \mathrm{Mpc}/h$, $T(k)$ the linear transfer function normalised to
1 as $k\to0$ (obtained as $T(k)=\sqrt{P_\mathrm{lin}(k)/k^{n_s}}$, renormalised at the lowest
tabulated $k$), and $D_\mathrm{norm}(a) = D(a)/D(a_\mathrm{norm})\times a_\mathrm{norm}$ with
$a_\mathrm{norm}$ in matter domination (`z_norm = 10`).

**Two physical effects.**

1. *Matter channel* (`bricks.add_png`): $\phi\to\Phi$ above, mapped back through $M(k)$. This is a
   second-order effect on the matter field, only captured by the LPT/N-body path (not by `kaiser`).

2. *Scale-dependent galaxy bias.* The Lagrangian bias expansion (`bricks.lagrangian_weights`,
   after [Modi+2020](http://arxiv.org/abs/1910.07097)) gains two PNG terms:

$$w(\mathbf{q}) = 1 + b_1\delta_L + \tfrac{b_2}{2}\left(\delta_L^2-\langle\delta_L^2\rangle\right)
 + b_{s^2}\left(s^2-\langle s^2\rangle\right) + b_{\nabla^2}\nabla^2\delta_L
 + f_\mathrm{NL} b_\phi\,\phi
 + f_\mathrm{NL} b_{\phi\delta}\left(\phi\delta_L-\langle\phi\delta_L\rangle\right).$$

   In Eulerian Fourier language the $b_\phi$ term is the familiar $1/k^2$ scale-dependent bias

$$\Delta b(k,a) = \frac{f_\mathrm{NL}\,b_\phi}{M(k,a)} \;\propto\; \frac{f_\mathrm{NL} b_\phi}{k^2\,T(k)\,D(a)} .$$

**Universality relations** (`bricks.b_phi`, `bricks.b_phi_delta`), used when `png_type: 'fNL'`:

$$b_\phi = 2\delta_c\,(b_1 + 1 - p) = 2\delta_c\,(b_1^E - p), \qquad
  b_{\phi\delta} = 2\,(\delta_c b_2 - b_1),$$

with $\delta_c = 1.686$, $b_1,b_2$ **Lagrangian** ($b_1^E = 1+b_1$, $b_2$ in the
$\tfrac{b_2}{2}\delta^2$ normalisation) and $p=1$ for a halo-mass-selected sample
($p\simeq1.6$ recent mergers, $p\simeq0.55$ stellar-mass selected).

**What is actually measurable.** From clustering alone only the *products*
$f_\mathrm{NL}b_\phi$ and $f_\mathrm{NL}b_{\phi\delta}$ enter the scale-dependent bias; $f_\mathrm{NL}$
by itself is fixed only through the (much weaker) matter $\phi^2$ channel. This is why
`png_type: 'fNL_bias'` samples the products directly, and why under that option `fNL` alone is
weakly identified.

**Caveat — super-box modes.** $P_\phi(k)\propto k^{n_s-4}$ makes $\langle\phi^2\rangle$
logarithmically divergent: modes with $k < k_\mathrm{fund}$ are absent from the box, so
$f_\mathrm{NL}$ is defined *within the simulated volume*. The missing long-wavelength modulation is
absorbed into the constant bias parameters.

### 2.2 Primordial non-Gaussianity — implementation

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
distance maps `χ(a)`/`a(χ)` are served — **in `nbody` only** — by a **bilinear emulator** rather than
the `jax_cosmo` ODE solvers, whose in-graph CPU callbacks otherwise exhaust LLVM memory. It is not a
global replacement: `cmb_lensing` still calls `jax_cosmo` inside the graph with the *sampled*
cosmology, notably `radial_comoving_distance` for `χ_s` in `convergence_Born_spherical`
(`cmb_lensing.py:301`, once per likelihood evaluation) and in `compute_cl_high_z` /
`compute_theoretical_cl_*` under the `exact`/`exact_linear` high-z modes. Tables are
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
`prod(ptcl_shape)/prod(paint_shape)` to recover `1 + δ_g` (robust to particle count). The crop is
deferred: the painted field stays on the paint grid (`paint_and_deconv(..., crop=False)`) so that
the survey selection multiplies it there, and `band_limit` averages the product onto the final grid
(§3.1). With no selection the two orders are identical.

| Grid | Factor (montecosmo) | Carries |
|------|---------------------|---------|
| init  | `init_oversamp` (3/2) | the inferred linear field; preconditioner & `kaiser_post` build on it |
| evol  | `evol_oversamp` (7/4) | LPT/N-body evolution, all bias products (δ², s², ∇²δ, φδ), `add_png` |
| ptcl  | `ptcl_oversamp` (7/4) | particle cloud density (`regular_pos(evol_shape, ptcl_shape)`) |
| paint | `paint_oversamp` (7/4)| interlaced+deconvolved CIC paint grid, cropped to final |

Under `evolution: lpt`, `ptcl_oversamp` **must equal** `evol_oversamp` (`__post_init__` raises
otherwise): jaxpm's `cic_read` reshapes the particle array to the force-grid shape, so LPT needs
exactly one particle per evol cell.

Flow in `FieldLevelModel.evolve`: init field → chreshape→evol → `add_png` + PNG re-band-limit
(`model.py:1212-1221`) → LPT displacement → paint+crop to final; `lagrangian_weights`
(`model.py:1275`) comes **after** and is handed `init_mesh_evol_grid`, i.e. the **pre-`add_png`**
Gaussian field, which is why the bias products read the Gaussian field and why `phi` is recovered
correctly there.

### 2.6 CMB lensing convergence

`cmb_lensing.convergence_Born_spherical` Born-integrates the evolved matter field over radial shells
**directly on HEALPix pixels** (curved-sky). Particles are painted to their shell support, converted
to δ, and accumulated with the lensing kernel
$$W_\kappa(\chi,a) = \tfrac{3}{2}\,\Omega_m\left(\tfrac{H_0}{c}\right)^2 \frac{\chi}{a}\,\frac{\chi_s-\chi}{\chi_s}.$$
Ray–box intersection intervals (`t_enter`, `t_exit`) restrict the sum to physically supported
shell/pixel contributions. The mean density `n̄` is derived from the actual particle count
(`pos.shape[0]`), so the normalisation is correct under particle oversampling.

**Shell assignment (single pass).** A particle's HEALPix pixel and bilinear angular weights
(`jax_healpy.vec2ang`, `get_interp_weights`) do not depend on which shell it falls in, so they are
computed once per particle; the shell index follows from its radius, and pixel, angular weight and
shell index are scattered together into one `(n_shells, npix)` array, which the per-shell volume,
mask and kernel then reduce to κ. Painting shell by shell would redo the angular work `n_shells`
times, and differentiating through that repetition is what makes the κ channel expensive.

Because each particle lands in exactly one radial bin, the shells must **tile
`[chi_min, chi_boundary]` exactly**: an overlap would count the mass inside it twice, a gap would
drop it. `convergence_Born_spherical` raises rather than accept either, and bin membership is
half-open, so a particle sitting on an interior edge is counted once
(`tests/test_born_shell_scatter.py`).

**Inner cut-off (`cmb_lensing.chi_min`, default 0).** The integration starts at `chi_min` rather than
at the observer. Close to the observer a shell subtends a small volume per pixel, so it holds far
fewer particles than there are pixels, and one particle contributes `W_κ/(n̄ Ω_pix χ²)` to the pixels
it touches: with `W_κ ∝ χ`, the sensitivity of κ to a single particle grows as 1/χ and its curvature
as 1/χ². A handful of particles within a few hundred Mpc/h of the observer therefore dominates the
curvature of the entire posterior — in a region the galaxy survey does not cover at all — and caps
the MCLMC step size in a joint run. Starting past them removes the pathology at its source;
preconditioning around it does not work, because the stiff direction rotates as those few particles
move between pixels.

The dropped range is not discarded: `C_ℓ(1 → chi_min)` is evaluated at the fiducial cosmology and
added to the cached line-of-sight correction, exactly like the high-z tail beyond the box (§3.2).
`chi_min > 0` therefore requires `full_los_correction` with `high_z_mode` `fixed` or `taylor`. Little
signal is given up, since `W_κ ∝ χ` vanishes at the observer.

**Radial shell weights (`cmb_lensing.shell_weights`).** `nearest` assigns each particle entirely to
the shell containing it; `linear` splits it between the two shells whose centres bracket its radius,
with the tent weight `max(0, 1 − |r − c_i| / d_r)`.

`linear` exists because the sampler differentiates through κ. Under `nearest`, a particle crossing a
shell edge moves its whole weight from one kernel to the next, so the log-density is discontinuous in
the particle radii everywhere in the box; MCLMC's energy error scales as ε⁶ only for a smooth target,
and a jumping one degrades it towards ε¹, capping the step size however the mass matrix is tuned. The
projector already interpolates bilinearly in angle over four HEALPix pixels — `linear` makes the
radial direction cloud-in-cell too, putting both directions in the same smoothness class as the
galaxy painting. Both ends of the radial range matter, since a particle switches on at `chi_min` and
off at `chi_boundary`, and the outer end dominates: the number of particles crossing a given radius
per unit radial distance grows as χ².

The `linear` tents therefore span the range apex to apex —
`d_r = (chi_boundary − chi_min)/(n_shells + 1)`, centres from `chi_min + d_r` to
`chi_boundary − d_r` — so each end tent's support stops exactly on a range boundary and a particle
enters and leaves the integration continuously at both, at the cost of a taper of one `d_r` of path
length at each end. Each shell is normalised by the volume its tent actually sees,
`Ω_pix ∫ w_i(r) r² dr = Ω_pix (r_i² d_r + d_r³/6)` (`linear_shell_volumes`), and its in-box mask uses
the support `r_i ± d_r`. Uniform shells only. `nearest` is the default and keeps its own layout,
`n_shells` bins of width `(chi_boundary − chi_min)/n_shells` tiling the range. Known gap:
`compute_shell_support_fractions`, used only for the closure-mode theory curve, still assumes
`r_i ± d_r/2` supports.

**Projection anti-aliasing (`cmb_lensing.proj_oversamp`, default 1).** The κ observable is the packed
pseudo-a_lm with ℓ ≤ 2·nside, obtained by `map2alm` of the masked map — a transform that assumes the
map is band-limited to that ℓ. The model's map is not: the Born projection scatters a finite number
of particles, so its κ carries shot noise down to the pixel scale, and `map2alm` folds that sub-pixel
power into the very modes the likelihood compares. The sampler then fits it, by building structure in
the field that the data does not contain.

Filtering the a_lm cannot fix this — the band cut is already there, and it acts after the fold.
Anti-aliasing has to precede the transform, exactly as it already does for the 3-D mesh (§2.5: paint
fine, then Fourier-crop). `proj_oversamp` is the spherical analogue: the particles are scattered onto
`nside × proj_oversamp` (a power of two), the transform runs at that sphere's own
`lmax = 2·nside·proj_oversamp` — jax_healpy's `map2alm` supports no other ratio — and only the (ℓ, m)
pairs of the observable band are selected, so the extra modes hold the sub-pixel power instead of
folding it in. The mask and the MASTER coupling matrix stay on the observable sphere; the shell
in-box mask and the ray entry/exit distances are recomputed on the refined one. The refined counts
array is small beside the evolution grids and the spherical transform is not where a log-density
gradient spends its time, so a modest refinement costs a few percent of the gradient — but the
transform's own cost climbs steeply with `nside`, so that balance is worth re-checking before
refining much further. Not available with `likelihood_mode: pixel_exact`, which is defined on the
observable pixels.

**Line-of-sight depth.** `kappa_pred` integrates matter only to `chi_boundary`, which is
`box_shape[2]` for `observer_mode: face`/`corner` but **`box_shape[2]/2` for `observer_mode: center`**
(the observer sits mid-box, so only half the depth lies ahead) — that is the full-sky/huge
configuration. The radial shells (`cmb_r_shells`) span exactly that range. The residual depth
`χ_box → χ_high_z_max` is handled analytically in the likelihood covariance (§3.2). For AbacusSummit
base, `chi_high_z_max: 3990` Mpc/h (matter simulated to z≈2.45), not `χ_CMB`.

---

## 3. Likelihood

### 3.1 Galaxy likelihood

A per-cell Gaussian on the galaxy **counts**, restricted to the survey mask via `dist.mask`:

```
obs ~ Normal( n̄·α·⟨(1+δ_g)·S⟩_cell ,  √(s_e²·n̄·α·S) )
```

with `n̄ = model.gxy_count` the mean count per cell at full completeness, `S` the selection, and
`α` the per-radial-bin amplitude (below). `s_e=1` ⇒ pure Poisson. The observable
`truth["obs"]` is the painted count mesh; `model.obs_to_delta` recovers `δ_g` for the warm start
and the diagnostics.

**Why counts and not `1+δ`.** The two are the same likelihood — dividing the data by `n̄S` and the
residual by `√(n̄S)` leaves `χ²` unchanged, so the posterior is identical up to a
parameter-independent Jacobian (`tests/test_counts_likelihood.py`). What differs is low
completeness: `obs = N/(n̄S)` and its variance `1/(n̄S)` both diverge as `S → 0`, so a partly covered
cell has to be discarded, whereas multiplying the prediction by `S` stays finite however small `S`
gets. That is what makes the completeness threshold a modelling choice rather than a numerical
necessity. This follows montecosmo (`intens_mesh = gxy_mesh * selec_mesh`, then a count likelihood);
we keep a Gaussian with a prediction-independent variance rather than its Poisson, since the cell
occupancies here are far inside the Gaussian regime and the Gaussian has smoother gradients. Its
`posit_fn = |·|` on the intensity is not used either: nothing here divides by the intensity.

**The selection multiplies at paint resolution.** `S` is carried on the paint grid
(`model.selec_paint`) as well as the final one (`model.selec_mesh`), and the predicted intensity is
`band_limit(gxy_paint · selec_paint)` — the cell average of the *product*, not the product of the
cell averages. The two differ by the sub-cell correlation between galaxy density and coverage, which
is exactly what a partly covered edge cell consists of: the coarse product predicts the whole-cell
mean density times the mean coverage, when the count actually observed comes from the covered
sub-region alone. `band_limit` preserves the mean, so the predicted total count is right by
construction. This is why the Fourier crop is deferred in `paint_and_deconv` (§2.5): `crop=False`
returns the field on the paint grid, and `gxy_mesh` is exactly `band_limit` of it
(`tests/test_counts_likelihood.py`).

**Free shot-noise amplitude `s_e` (`model.gxy_stoch_noise`, default off).** An optional positive latent
(group `stoch`, truncated-normal `loc=1, scale=1, [0,10]`) rescaling the shot-noise std. It
marginalises non-Poisson model/stochastic scatter (which the 2-LPT + Lagrangian-bias model cannot match
exactly at the field level) so it does not leak into `f_NL`. Following montecosmo we keep `s_e = 1` fixed in
the reference runs.

**Survey mask & radial selection from randoms.** For the AbacusLensing LRG mocks (equal-weight
galaxies and ~20× randoms), `S` is the **painted random density**, normalised to unit mean over the
occupied cells. The randoms are painted with **the same scheme as the galaxies and as the model** —
interlaced, CIC-deconvolved, at `paint_shape` — so `S` and the counts carry the same window. The
catalogue is far too large to hold, so the two interlacing shifts are accumulated chunk by chunk
(`interlace_accumulate`) and combined once at the end (`interlace_combine` / `interlace_finalize`);
painting is linear in the positions, so this is identical to one call on the whole catalogue
(`tests/test_interlace_accumulate.py`). The occupancy mask stays on the plain CIC paint, where
`> 0` still means "a random landed here" — the deconvolved mesh rings. The reduction is cached on
disk, keyed on everything that changes it, including the paint grid.

**Completeness cut (`abacus_galaxy.completeness_min`, default 0.8).** Cells on the survey boundary —
octant faces and radial shell edges — are only fractionally covered, and are dropped below this
threshold on `S`. It is a choice about how far to trust the model at the boundary, not a numerical
guard: the two things that otherwise make a partly covered cell unusable are handled in the forward
model. The randoms carry the same interlaced, deconvolved window as the galaxies and as the model, so
numerator and denominator of a density estimate share a window; and the band-limited product predicts
the covered sub-region directly, rather than a whole-cell average compared against a sub-region
count. Lower it in steps, watching the edge `ngbar` bins and `b1`, which respond first.

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
- In the likelihood, `α = ngbar[bin_id]` multiplies the predicted counts **and** their shot-noise
  per bin, exactly as `set_radial_count` multiplies both the intensity and the selection. At `α=1`
  it reduces to the plain likelihood.

Mathematically, if the true per-bin density is `α_b ×` fiducial, then `E[obs] = α_b·n̄·S·(1+δ_g)`
and `Var[obs] = α_b·n̄·S`; marginalising `α_b` over **fine** bins removes the entire radial-monopole
subspace, so `f_NL` is measured from transverse/3-D modes only and is insensitive to a mis-traced
`n̄(χ)`.

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

**Line-of-sight correction (`full_los_correction`, `high_z_mode`, `chi_min`).** `C_ℓ^{high-z}` for the
missing depth `χ_box → χ_high_z_max` is the Limber convergence power (`jax_cosmo`), added to the
variance. When `cmb_lensing.chi_min > 0` (§2.6) the near end `1 → χ_min` is added to the same cached
term at the fiducial cosmology, so `C_ℓ^{high-z}` in the formulas above is really the correction for
both unmodelled ends of the line of sight; only the high-z part carries `taylor` gradients. Modes:
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

**Primary mode: joint analysis on the AbacusSummit HUGE box (`c000_ph201`).** The default
`config.yaml` targets the HUGE simulation, which carries **both probes over the full sky** (below);
this is the headline configuration, and the galaxy-only and CMB-only sub-modes cross-check the two
probes independently. The AbacusLensing **base** box (`c000_ph000`) also carries both on the same
lightcone — the LRG catalog over the octant footprint and `kappa_00047.asdf`, whose footprint is two
small patches (f_sky ≈ 4.5 %) inside that octant (`observer_mode: corner`, `chi_high_z_max: 3990`) —
but its κ footprint is too small to expose any gain, which is why the analysis moved to HUGE.

**Two special-geometry AbacusSummit configurations** (both are `abacus` mode, they only differ in which
probe is enabled and which geometry flags are set):

- **CubicBox snapshot (galaxy-only, no lightcone).** A single-file periodic-box snapshot
  (`AbacusSummit_*_c000_ph000` cubic box at fixed z, auto-detected by `_is_cubic_box_catalog`). Here the
  observer/lightcone/curved-sky machinery is meaningless, so this mode **requires**
  `curved_sky: false`, `lightcone: false`, `los: null` (RSD off), `cmb_lensing.enabled: false`
  (galaxies only), and `a_obs` set to the snapshot scale factor. This is the montecosmo-parity
  validation setup (§8).

- **HUGE full-sky κ × LRG.** `AbacusSummit_huge_c000_ph201/kappa_00045.asdf` is a full-sky
  (`f_sky = 1`) κ map, and the **same phase** carries a full-sky LRG lightcone (below), so this box
  supports both a CMB-only run and the joint analysis at `f_sky = 1` on **both** probes. Geometry:
  `box_shape: [7500, 7500, 7500]` (the simulation box, which the published IC grid also covers),
  `cell_size: 93.75` → mesh 80³, `observer_mode: center` (`LightConeOrigins = [0,0,0]`) giving
  `chi_boundary = 3750`, `chi_high_z_max: 3942`, `likelihood_mode: diagonal` (exact on the full sky).
  With the galaxies disabled it is the clean full-sky CMB-only validation of the Born projector and
  the high-z covariance.

- **HUGE full-sky galaxies (`lrg_huge`).** `DR2/mock_catalogs/lrg_huge/` holds an LRG lightcone for
  `AbacusSummit_huge_c000_ph201/ph202`: 22.2 M galaxies over the **full sky**, `z ∈ [0.405, 1.095]`
  (χ ∈ [1093, 2447] Mpc/h), one FITS file (`RA, DEC, Z, MASS, ID, IS_CENT, WEIGHT_FKP`) with randoms
  in a **separate** 100× FITS file (`RA, DEC, Z, WEIGHT_FKP`) whose n(z) tracks the data to 0.1 % and
  whose angular distribution is Poisson — unlike the base-box AbacusLensing randoms, these carry no
  radial selection artefact. `Z` **includes RSD** (published DR2 multipoles give P₂/P₀ ≈ 0.42 at low
  k), so keep `los` enabled and `bnpar` free; `WEIGHT_FKP` is deliberately **not** applied, the
  likelihood works on raw counts. Configure with `abacus_galaxy.randoms`, `z_range: [0.4, 1.1]` and
  `randoms_chunk_rows` (2.22 × 10⁹ rows, streamed, never held in memory). `randoms_paint_chunk`
  bounds the peak memory of each scatter pass independently of the read chunk.

  **Randoms reduction cache.** However many rows the catalogue has, it only ever produces two meshes
  and one count, so that reduction is cached to `$SCRATCH/desi_cmb_fli_cache/randoms_<digest>.npz`
  and reused; re-streaming the FITS file otherwise dominates start-up. The digest covers everything
  that changes the result: randoms file identity (path, size, mtime), `z_range`, `box_shape`,
  `mesh_shape`, paint grid, observer position, `randoms_max_rows`, `randoms_chunk_rows`, and the
  background cosmology taken as the leaves of the `Cosmology` pytree — so a nuisance `loc_fid` such
  as `b1` does **not** invalidate it while `Omega_m` does. `abacus_galaxy.randoms_cache_dir`
  relocates it, `null` disables it (`tests/test_randoms_cache.py`).

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
is a single path (snapshot) or a list of shell paths (lightcone). On a lightcone the loader also
resets `model.a_fid` to the growth-weighted mean scale factor over the **survey cells** (their radii
from `radius_mesh`), so the Kaiser preconditioner (§5) is built at the redshift the data actually
covers.

Randoms come from one of two places, selected by whether `abacus_galaxy.randoms` is set; the two
producers package them differently and are not interchangeable. Unset (base-box AbacusLensing): the
`RAND_*` columns of each galaxy shell file, with a per-shell α = n_gal/n_rand. Set (`lrg_huge`):
standalone FITS file(s) read in `randoms_chunk_rows` slices and painted incrementally, with a single
global α. Redshift bounds likewise come from `abacus_galaxy.z_range` when set, otherwise from the
`/z0.400/`-style path token used for shell de-duplication; the same bounds drive the `ngbars` radial
binning.

**Abacus initial conditions** (`load_abacus_ic_truth`, `abacus_ic.file`): the published
`ic_dens_N*.asdf` (e.g. `ic/AbacusSummit_huge_c000_ph201/ic_dens_N576.asdf`) is the true linear δ over
the simulation box at `InitialRedshift` (z=99) — *not* at `CLASS_Redshift`, which is only the redshift
of the CLASS spectrum the ICs were drawn from. It is grown to a=1 with the header `GrowthTable` and
Fourier-resampled onto `model.init_shape` by `utils.chreshape` (power-preserving, so no extra
normalisation is needed; its spectrum matches `P_lin` to the Eisenstein-Hu-vs-CLASS difference).
**Validation reference only**: it populates `truth['init_mesh']`, read by
`plot_warmup_diagnostics` and by `analyze_run`'s reconstruction figure (§6). The sampler is
warm-started from the *observed* galaxy field via `model.kaiser_post` and never sees it.

---

## 5. Inference & sampling

**Sampler.** MCLMC, over the field latent + scalar latents, in a reparametrized (whitened) space.
`run_inference.py` wires **only** `get_mclmc_warmup`/`get_mclmc_run`; the NUTS-within-Gibbs
(`nutswg_*`) and MAMS (`get_mams_*`) paths exist in `desi_cmb_fli.samplers` but are not reachable from
the script and are not exposed by any config switch.

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
with `scale = √(1 + B(k,μ)²·P(k)/noise_eff)`, `transfer = √P(k)/scale`
(`model._precond_scale_and_transfer`). `B` is the **full** Eulerian Kaiser boost
`B = D(a_fid)·(b_E + f(a_fid)·μ²)` (`bricks.kaiser_boost`), not `b_E` alone — it reduces to `D·b_E`
only when `los: null` disables RSD. The effective noise is `noise_eff = 1/(n̄·⟨S²⟩)` whenever a
selection mesh is present (§3.1), falling back to `1/n̄` otherwise. `kaiser` evaluates the boost and
`P` at the **fiducial** cosmology (fixed preconditioner), `kaiser_dyn` at the sampled one. In
galaxy/joint modes the warmup initial field is a "reverse-Kaiser" estimate from the galaxy data; in
**CMB-only** mode (`n_gal^{eff}=0` ⇒ `noise_eff=∞`) `scale=1`, `transfer=√P(k)` (pure prior — the field is constrained only through κ),
the initial field is a random Gaussian draw, and the biases are fixed to fiducial (galaxy calculations
skipped).

**Warmup steps.** STEP 1 warms the mesh with cosmo/bias fixed (`mcmc.mesh_desired_energy_var`, no
diagonal preconditioning — the whitened field is already unit-scale); STEP 2 frees the scalar latents
at `mcmc.mclmc.desired_energy_var`; a median-collapse then homogenises per-chain configs for STEP 3
sampling.

**Joint runs.** The conditioning of the joint galaxy+κ posterior is set by how the Born integral
treats the shells nearest the observer (§2.6, `chi_min` and `shell_weights`). With those two set, the
joint problem warms up and samples like the galaxy-only one and needs no preconditioner of its own
beyond blackjax's diagonal adaptation.

**Energy-variance tuning.** MCLMC performance depends on `mcmc.mclmc.desired_energy_var`. After the
first sampling batch, `run_inference.py` reports per-chain `MSE/dim` and its ratio to the target
and flags it in both directions: above 2× the step size is too large, and below 0.1× the adaptation
has failed and compute is being wasted (since `Var[E] = O(ε⁶)`, the step is short by `ratio^(-1/6)`).
`5e-8`–`5e-7` works for realistic noise; use `5e-9` or lower for reduced-noise runs
(`cmb_noise_scaling: 0.01`).

---

## 6. Diagnostics & tools

- **`quick_pk_spectra.py`** — 3-D `P(k)` diagnostic. In **closure** mode: measured matter `P(k)` from
  the model's `matter_mesh` vs nonlinear `jax_cosmo` `P_mm` at `a_fid`. In **abacus** mode: galaxy
  `P_gg(k)` as seen by the likelihood, Abacus observed catalog vs LPT-simulated galaxies at the config
  cosmology (same survey mask, Poisson shot-noise subtracted). Binning runs past `k_Nyq` out to the
  k-space cube diagonal `√3 k_Nyq`: the sphere `|k| < k_Nyq` holds only `π/6` of the modes, and the
  corner modes enter the cell-wise likelihood at full weight, so they must be checked too.
- **`quick_cl_spectra.py`** — angular spectra `C_ℓ^{κκ}, C_ℓ^{gg}, C_ℓ^{κg}` vs theory, for closure (N LPT realizations, mean±std) and abacus (loaded maps + N noise draws) modes.
  Uses **resolution-aware Limber** (`compute_theoretical_cl_gg/kg`, `k_nyq` cut: integrate only shells
  with `k_⊥=(ℓ+0.5)/χ < k_Nyq = π/Δx`) and overlays the Poisson shot-noise floor
  `N_ℓ^{shot} = Ω_sky/N_gal` on the `C_ℓ^{gg}` panel; spectra are log-binned (`metrics.bin_cl_log`).
  In abacus mode both `Ω_sky` and `N_gal` are taken from the loaded survey (solid angle as
  `V_survey / ∫χ²dχ` over the occupied radial range, so it is geometry-agnostic) rather than from the
  closure-mode `gxy_density`. Curved-sky galaxy spectra are also available with the CMB disabled, via
  a HEALPix proxy whose nside matches the cell size at mid-survey distance — a flat-sky fallback would
  average along the box z-axis, which for a centred observer merges opposite sides of the sky. Both
  scripts condition on **every** latent (`validation.conditioning_params`, missing ones held at
  `loc_fid`), since `predict` would otherwise draw them from their priors.
- **`analyze_run.py`** — merge batches, R-hat/ESS, corner/trace plots (abacus mode uses
  `abacus_truth_params` markers), custom burn-in / chain exclusion, and two **field-level figures**.
  Runs automatically at the end of every job.
  - `initial_conditions.png` — true field, reconstruction, difference, then `T(k)`/`r(k)` and a
    radial profile (rms ratio, correlation with the truth, agreement between chains). The slice is
    the plane through the **observer** in observer-centred coordinates, right for every
    `observer_mode`; dashed circles mark the galaxy radial range.
  - `kappa_reconstruction.png` — observed κ, model κ, model error and their spectra against the
    covariance the likelihood assumes, all band-limited to the likelihood's own band at `cmb_nside`
    (a `proj_oversamp > 1` model map included): the figure shows the reconstruction the inference
    uses, not a finer one.
  - Both draw **one posterior sample, never a mean over chains** — averaging independent samples
    suppresses the amplitude wherever the data does not constrain the field, which reads as lost
    power when every sample carries the right power. Read from `sampler_state.pkl`, the only place
    the field survives (`samples_batch_*.npz` hold scalars only). One forward model each;
    `--no_field_plots` skips them.
- **`compare_reconstruction.py`** — correlates the true IC with the sampled one in radial shells
  around the observer, per chain, overlaying several runs (joint vs galaxy-only); this is what
  measures where κ extends the reconstruction (§7.1).
- **`compare_runs.py`** — GetDist triangle comparison of multiple runs (per-run burn-in, labels).
- **`plot_2D_maps.py`** — κ (and galaxy-projection) maps for one forward realization.
- **`benchmark_highz_cl_modes.py`** — precision/speed of the high-z modes; **`plot_lensing_fraction.py`**
  — box-captured lensing fraction vs depth; **`make_abacus_kappa_mask.py`** — degrade the AbacusLensing
  footprint to a model-nside `.npy`; **`plot_highz_correction_vs_depth.py`** — mean high-z κ correction
  amplitude vs box depth (`χ_min`); **`plot_linear_vs_nbody.py`** — linear (Kaiser) vs N-body evolved
  matter field from the same IC; **`plot_cmb_noise_comparison.py`** — ACT DR6 vs Planck PR4 κ noise
  spectra.
- **Freeze diagnostic** (`DIAGNOSE_FREEZE=1`): logpdf + per-group gradient + 1-D scans of each free
  scalar at the STEP-2 start; localises curvature pathologies.

---

## 7. Validation status & roadmap

**Validated.** On the Abacus **CubicBox snapshot** (periodic full box, z=0.8), the
galaxy-only run reproduces montecosmo result — unbiased `f_NL` (≈ −111 ± 500).

**Current results (Abacus lightcone).**
- **Galaxy-only** is **unbiased**: with fine-bin `ngbars` + `png_type: fNL` (universality) +
  oversampling, `f_NL = 1.5 ± 33` (consistent with 0). Under `png_type: fNL_bias` the scale-dependent
  bias amplitude `fNL_bp ≈ 0` (the meaningful observable), while the decoupled matter-φ² `fNL` channel
  is weakly identified and can wander.
- **Joint (full galaxies + κ)** converges cleanly and is consistent with galaxy-only; the measured
  κ contribution is in §7.1.

### 7.1 Full-sky κ × LRG on the HUGE box — measured

Reference pair, identical configuration apart from `cmb_lensing.enabled`: completeness cut 0.3,
`gxy_ngbar_free: false`, `png_type: fNL`, counts likelihood, ACT DR6 `N_ℓ` at ×1, `nside 32`,
`chi_min 350`, `shell_weights: linear`, `proj_oversamp: 2`.

| | run | batches |
|---|---|---|
| joint | `run_20260910_070354_58161527` | 92 |
| galaxy-only | `run_20260910_033019_58153868` | 55 |

**The joint problem converges like galaxy-only.** `step_size` 85.18 vs 85.88 (0.8 % apart), energy
variance 0.88–1.10 × target on all four chains, R-hat ≤ 1.02 — i.e. adding κ costs nothing in
sampling efficiency.

**κ moves no parameter.** `f_NL` 8.94 ± 5.70 (joint) vs 8.87 ± 5.77 (galaxy-only); `b1` 1.1910 vs
1.1915; `b2`, `bs2`, `bn2`, `bnpar` all within 0.1σ.

**κ adds nothing to any σ.** The two runs share `seed` and the warm start, and their chains correlate
at 0.73–0.89, so per-chain σ ratios are a *paired* comparison. At matched 55 batches, burn-in 50 %:
`f_NL` **0.992 ± 0.014**, `b1` 1.015 ± 0.013, `b2` 0.982 ± 0.009, `bs2` 0.969 ± 0.030,
`bn2` 1.019 ± 0.009, `bnpar` 0.998 ± 0.032 — scattered about 1, with any gain on `σ(f_NL)` below
3.5 % at 95 %. These ratios need the full chains: over the first few tens of batches they sit
coherently a few percent below 1 out of Monte-Carlo noise alone.

**What κ does add: the field outside the galaxy volume.** `scripts/compare_reconstruction.py`
correlates the true Abacus IC with the sampled one in radial shells around the observer, per chain.
Smoothed to 250 Mpc/h — the transverse scale κ's band `ℓ ≤ 64` actually reaches:

| shell χ [Mpc/h] | joint | galaxy-only |
|---|---|---|
| 361–722 | **0.327 ± 0.068** | −0.154 ± 0.175 |
| 722–1083 | 0.250 ± 0.228 | 0.051 ± 0.176 |
| 1443–1804 | 0.951 ± 0.003 | 0.950 ± 0.007 |
| 1804–2165 | 0.908 ± 0.007 | 0.892 ± 0.011 |
| 2165–2526 | **0.509 ± 0.034** | 0.419 ± 0.074 |
| 2526–2887 | **0.146 ± 0.009** | 0.034 ± 0.079 |
| 2887–3248 | **0.123 ± 0.025** | 0.000 ± 0.011 |
| > 3969 | ≈ 0 | ≈ 0 |

The LRG shell is χ ∈ [1094, 2447]. **Inside it κ adds nothing** — the galaxies already reach 0.95,
there is no room left. **Outside it κ is the only information**, and the inner cone goes from zero to
r ≈ 0.33. As a control that this is not a global artefact, beyond χ = 3969 both curves return to
zero, as they must, since the Born integral stops at χ = 3750.

**The variance moves much less than `r` does.** `r` explains `r²` of the variance, so `r = 0.33` in
the inner cone is 11 % and `r ≈ 0.12` over the much larger outer volume is 1.4 %. The rms residual
against the true IC therefore barely moves: inside the shell 0.002807 ± 0.00008 (joint) vs
0.003005 ± 0.0002 (galaxy-only), outside 0.005840 ± 0.00007 vs 0.006037 ± 0.00007 — a 3.3 % reduction
at ~2σ. Both statements hold: κ takes the reconstruction from nothing to something where the galaxies
see nothing, a qualitative change, but most of the field there is still prior-driven.

Two caveats on that table. **Run both smoothings**: at 100 Mpc/h the inner cone only reaches 0.16,
diluted by small-scale power κ cannot touch. And the inner/outer asymmetry is expected — `ℓ = 64` is
69 Mpc/h transverse at χ = 700 against 295 Mpc/h at χ = 3000, so κ's ~4200 modes cover far more of the
smaller inner volume. The error bars are the **spread over the four chains**, not the sampling
variance of `r` within a shell, which is non-negligible since the smoothed field is correlated.

**Roadmap.**
1. Compare to a standard power-spectrum analysis (with A. de Mattia).
2. **HalfDome** (`/global/cfs/cdirs/cmb/gsharing/halfdome`, arXiv:2407.17462) as a second external
   simulation, and the only one on hand with a **matched f_NL pair**: 11 Gaussian realisations plus
   `seed_100_fnl_20` (local f_NL = 20, sharing seed 100 with a Gaussian twin), 6144³ particles in a
   3.75 Gpc/h box, with exact linear ICs (`lineark`, `Nmesh = 12288`) for both. Cost: the release
   ships *"downsampled particles, halo catalogs, mass sheets, velocity sheets"* — there is **no
   CMB-source κ map**; the ready-made `lensing/` planes stop at z_s = 2.5 and cover only the Gaussian
   seeds. Using it requires a `bigfile` reader, Born-integrating κ_CMB from the `usmesh` mass sheets
   (HEALPix nside 8192, 80 shells over a = 0.2–1), and an HOD or mass cut on the RFOF lightcone halos.
3. Application to real DESI-LRG × Planck/ACT κ data.
