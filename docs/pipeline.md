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
The field is the inference latent
`init_mesh` (sampled as `init_mesh_` in the Kaiser-whitened basis, see §5). Mesh dimensions are
auto-adjusted (`get_model_from_config`) so all axes have an **even** number of cells: the real↔complex
Gaussian repacking `utils._rg2cgh`/`cgh2rg` asserts `all(shape % 2 == 0)`, because the
hermitian-symmetry bookkeeping (self-conjugate modes at 0 and Nyquist) assumes an exact Nyquist plane.

**Linear power spectrum (`model.lin_pk_table`).** Without a table the spectrum is jax_cosmo's
Eisenstein–Hu. With a table (two columns, k and P, at any redshift), `bricks.lin_power_interp`
multiplies Eisenstein–Hu at the sampled cosmology by the fixed shape ratio `R(k) = P_tab / P_EH`
taken at the fiducial cosmology (`lin_pk_ratio`; the latents' `loc_fid` must be the table's
cosmology) and renormalises the product to the sampled σ8 with its own top-hat integral over the
interpolation nodes: jax_cosmo's `sigmasqr` integrates in ln k between the base-10 logarithms of its
limits, i.e. over k ∈ [e⁻⁴, e³], which leaves its σ8 normalisation slightly off. The Eisenstein–Hu
spectrum itself (`bricks._eh_power`) is jax_cosmo's `linear_matter_power` without its growth factor
at a = 1, where it is 1 by definition: the jax_cosmo fork computes the growth through a host callback
that re-solves the ODE for every new cosmology, and with `Omega_m` sampled those callbacks exhaust
the compiler's memory and threads (§2.3). Without a table the result is identical to
`linear_matter_power`.
At the fiducial the spectrum is the table; away from it, the change of shape with `Omega_m` is
Eisenstein–Hu's, so the error is only how the Boltzmann/EH ratio itself moves with `Omega_m`. Holding
the fiducial shape fixed instead, as montecosmo's tabulated mode does, is exact only at fixed
`Omega_m`: at fixed σ8 the large-scale shape moves strongly with `Omega_m`. The ratio rather than a
Boltzmann emulator because no differentiable JAX emulator of the linear P(k) is available: the
`jaxace` emulators return background quantities, σ8(z) and growth only. Everything built on the
linear spectrum takes the same ratio: the prior and the Kaiser preconditioner (`lin_power_mesh`), the
φ→δ transfer of `add_png` and of the scale-dependent bias (`trans_phi2delta_interp`), the Kaiser model
and the Wiener start. The Limber C_ℓ (line-of-sight term, diagnostics) and `desi_cmb_fli.fisher` keep
Eisenstein–Hu. The Abacus configs carry the CLASS spectrum of the AbacusSummit ICs
(`data/abacus_cosm000_CLASS_power.txt`); closure configs have no table, and are self-consistent with
either since data and model share the spectrum. Measured accuracy in §7.11;
`tests/test_initial_conditions.py` pins the table at its cosmology, the exact σ8 elsewhere and the
gradient in `Omega_m`.

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
the `jax_cosmo` ODE solvers: the jax_cosmo fork serves growth and distances through host callbacks
that re-solve for every new cosmology, and with `Omega_m` sampled they exhaust the compiler's memory
and threads. Every background quantity in the sampled graph comes from the emulator — including `χ_s`
in `convergence_Born_spherical` (`nbody.a2chi`) — or is evaluated where it is known exactly (the
growth at a = 1 in the linear spectrum, §2.1); `test_sampling_omega_m_puts_no_jax_cosmo_callback_in_the_graph`
checks the full lightcone model with galaxies, κ and PNG. The exception is `compute_cl_high_z` /
`compute_theoretical_cl_*` under the `exact`/`exact_linear` high-z modes, which call jax_cosmo with
the sampled cosmology. Tables are
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

Flow in `FieldLevelModel.evolve`: init field → chreshape→evol → `add_png` + PNG re-band-limit → LPT
displacement → paint+crop to final; `lagrangian_weights` comes **after** and is handed `init_mesh_evol_grid`, i.e. the **pre-`add_png`**
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

**Per-pixel normalisation.** Counts become a density by dividing each pixel by the volume it
collects: the shell volume per pixel (`Ω_pix ∫ w_shell(r) r² dr`) times `bilinear_weight_norm`, the
bilinear weight that pixel receives from directions spread uniformly over the sphere, relative to
the mean. HEALPix's bilinear weights sum to one per particle but not to the same total per pixel:
the four pixels of each polar cap collect 1.167 × the mean at every nside, the others stay within
0.4 % at nside 64 (3 % at nside 8). A uniform solid angle per pixel — what jaxpm's
`paint_particles_spherical_bilinear` divides by — would read uniform matter as δ = +0.167 in those
eight pixels in every shell, a fixed convergence at the two poles that no field produces: it cancels
in closure, where data and model share it, but not against a map made otherwise (AbacusLensing,
real data), and the sampler would have to dig the field along the polar axis to hide it. The map is
computed once per nside by quadrature over the pixel centres of a grid 16 × finer (error ∝
oversampling⁻², 6·10⁻⁴ at the poles) and cached; it is a constant, so the gradient is unchanged.
The Abacus loader brings the data map to the projection sphere through the same kernel with the
same per-pixel division (`bilinear_resample_healpix`, §4), and the diagnostic galaxy map from
particles divides by it too (§6), so all three read a uniform field as uniform.
`tests/test_born_shell_scatter.py` checks that isotropic matter gives the same κ in every pixel,
polar caps included.

`jax_healpy.get_interp_weights` agrees with healpy on generic directions (2·10⁶ random directions,
identical weights) but not on a direction whose longitude equals a pixel centre's to machine
precision, where it gives a neighbour's weight to the wrong pixel. Particles never land there; a
test grid aligned with the HEALPix axes does, hence the rotated grid in that test.

Because each particle lands in exactly one radial bin, the shells must **tile
`[chi_matter_min, chi_boundary]` exactly**: an overlap would count the mass inside it twice, a gap
would drop it. `convergence_Born_spherical` raises rather than accept either, and bin membership is
half-open, so a particle sitting on an interior edge is counted once
(`tests/test_born_shell_scatter.py`).

**Radial range (`cmb_lensing.chi_matter_min`, default 0; `chi_high_z_max`).** The shells start
where the matter of the observed map starts and stop at the box edge. On real data that is the
observer (0). A simulated light cone may stop before z = 0: the AbacusSummit HUGE one stops at
z = 0.1 (`FinalRedshift` in `abacus.par`), so its κ holds no matter below χ = 292.6 Mpc/h and the
Abacus configs set `chi_matter_min: 292.6`. It is the counterpart of `chi_high_z_max`, where the
map's matter stops: the CMB for real data, 3942 Mpc/h (z = 2.4) for the AbacusLensing map. The
matter between the box edge and `chi_high_z_max` is the line-of-sight term of the covariance (§3.2);
below `chi_matter_min` the map holds nothing, so no term. The closure configs copy the AbacusLensing
range at both ends (292.6 and 3942), so that closure and Abacus runs share their κ geometry.
`FieldLevelModel.low_z_matter_start` is `chi_matter_min`, floored at 1 Mpc/h for the Limber
integrands that start there.

**The shells near the observer.** Two problems come from them. *Stiffness*: a near shell subtends a
small volume per pixel, so it holds far fewer particles than there are pixels, and one particle
contributes `W_κ/(n̄ Ω_pix χ²)` to the pixels it touches; with `W_κ ∝ χ`, the sensitivity of κ to a
single particle grows as 1/χ and its curvature as 1/χ². A handful of particles within a few hundred
Mpc/h of the observer would dominate the curvature of the whole posterior — in a region the galaxy
survey does not cover — and cap the MCLMC step size (§7.7, §7.12); preconditioning around it does not work, because the stiff direction
rotates as those few particles move between pixels. *Resolution*: a multipole ℓ of a shell at
distance χ is a transverse scale of about πχ/ℓ; where that is smaller than what the inferred field
resolves, the shell's κ at that ℓ is the particles' discreteness rather than the matter's structure,
and a nearer shell reaches it at a lower ℓ (excess model power at the top of the band, §7.6).

**Lattice discreteness.** The particles start on a regular lattice, and at the inference resolution
the LPT displacements are small against the particle spacing `d`, so the lattice survives evolution
— the same small-displacement regime that keeps 2LPT accurate. The 3-D painting does not see it (one
particle per cell of a grid aligned with the lattice paints to a constant), but the HEALPix pixels
are not aligned with it: seen from a centred observer, rays running along the lattice planes give a
moiré on the three coordinate great circles, and the shell at distance χ adds spurious power around
ℓ ≈ 2πχ/d. Near shells put it inside the band, far ones above it — but the coordinate planes cross
the whole box, and a thin line on the sky has power at every ℓ, so far shells keep a smaller in-band
part. In closure data and model share it; against AbacusLensing or a real map it is model error.

**Per-shell multipole cut (`cmb_lensing.shell_kmax`; W. Kabalan's `resolution_cut` in jax-fli).**
Both problems are handled per multipole: each shell is low-passed above the multipole the model
resolves at its distance, ℓ_res = `shell_kmax`·r_eff, r_eff the mass-weighted radius of the shell
under its radial profile (top hat or tent). `shell_kmax` defaults to the Nyquist frequency of the
inferred (init) grid, π/62.5 at the inference setting, π/31.25 at cell 46.875; the final-grid
Nyquist sends structure the model resolves to the covariance (§7.12). `shell_kmax: 0` switches the
cut off, and it is off without `full_los_correction`, which carries the removed power. A shell with
ℓ_res below the band edge 2·nside gets the cosine taper of jax-fli: 1 up to ℓ_cut − ℓ_cut//4, 0 from
ℓ_cut = ⌊ℓ_res⌋; a shell that resolves no multipole is removed whole (jax-fli floors ℓ_cut at 4
instead). At the inference setting the shells below r_eff ≈ 1270 Mpc/h are cut. The Born projector
returns its shells separately (`convergence_Born_spherical(per_shell=True)`), `cut_shells` transforms
each cut shell (`map2alm`, `iter=0`, batched), applies its taper, sums the a_ℓm and returns to the
map, to which the untouched shells are added, so `kappa_pred` is the cut map everywhere downstream
(closure data, `pixel_exact`, figures). The `iter=0` round trip of a cut shell costs 4·10⁻⁴ (rms,
relative) on the observable a_ℓm of a band-limited shell, its worst pixels at the poles. The power
removed — the model's discreteness, and on AbacusLensing or a real map real structure the model does
not resolve — enters the covariance as the Limber C_ℓ of the removed kernel Σ_s (1 − w_s(ℓ)) q_s(χ),
q_s = d_r W(r_s) p_s(χ) χ²/V_s the radial kernel of shell s in the projector (neighbouring tents with
their cross terms; `compute_cl_shell_cut`), added to the line-of-sight term, so it follows
`high_z_mode` like it (§3.2). Inside the taper's roll-off the removed part of a mode is proportional
to its kept part, and the likelihood treats it as independent noise: an approximation over a quarter
of each cut shell's ℓ_res. The cut removes the near-shell discreteness and the near-observer
stiffness, so that where the shells start no longer matters, at the price of a costlier κ gradient
(§7.12). It leaves the far shells' in-band discreteness at the top of the band, which shrinks with
the cell and which the paper states (§7.12). The covariance term carries the bilinear window of
the projection sphere, like the data and model maps (§3.2). On a cut sky the filter acts on the
zero-filled shell maps.

**Abandoned: an inner cut-off `chi_min`.** Before the per-shell cut, the shells started at a
`chi_min` beyond where the map's matter starts (350, then 700 Mpc/h), and the Limber C_ℓ of the
dropped range, from `chi_low_z_min` (the former name of `chi_matter_min`) to `chi_min`, entered the
line-of-sight covariance as a low-z term. It removed the stiffness and the near-shell discreteness by
dropping whole shells, but it gave up their field information at every ℓ, it replaced them by a Limber
term that is inaccurate at low ℓ for so near a kernel (§7.12), and its value was a resolution tuning
(§7.6). The cut does the same per multipole. `chi_min`, `chi_low_z_min` and the low-z term are removed; a κ
config that still holds either key is rejected with an explicit error, and the runs made with them
(§7.1–7.7, and every run before the cut) need the code before the change to be re-analysed.
Galaxy-only configs are not affected.

**Radial shell weights (`cmb_lensing.shell_weights`).** `nearest` assigns each particle entirely to
the shell containing it; `linear` splits it between the two shells whose centres bracket its radius,
with the tent weight `max(0, 1 − |r − c_i| / d_r)`.

`linear` exists because the sampler differentiates through κ. Under `nearest`, a particle crossing a
shell edge moves its whole weight from one kernel to the next, so the log-density is discontinuous in
the particle radii everywhere in the box; MCLMC's energy error scales as ε⁶ only for a smooth target,
and a jumping one degrades it towards ε¹, capping the step size however the mass matrix is tuned. The
projector already interpolates bilinearly in angle over four HEALPix pixels — `linear` makes the
radial direction cloud-in-cell too, putting both directions in the same smoothness class as the
galaxy painting. Both ends of the radial range matter, since a particle switches on at
`chi_matter_min` and off at `chi_boundary`, and the outer end dominates: the number of particles crossing a given radius
per unit radial distance grows as χ².

The `linear` tents therefore span the range apex to apex —
`d_r = (chi_boundary − chi_matter_min)/(n_shells + 1)`, centres from `chi_matter_min + d_r` to
`chi_boundary − d_r` — so each end tent's support stops exactly on a range boundary and a particle
enters and leaves the integration continuously at both, at the cost of a taper of one `d_r` of path
length at each end. Each shell is normalised by the volume its tent actually sees,
`Ω_pix ∫ w_i(r) r² dr = Ω_pix (r_i² d_r + d_r³/6)` (`linear_shell_volumes`), and its in-box mask uses
the support `r_i ± d_r`. Uniform shells only. `nearest` is the default and keeps its own layout,
`n_shells` bins of width `(chi_boundary − chi_matter_min)/n_shells` tiling the range. Known gap:
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

The return to the observable is a sharp harmonic cut, not a pixel average: `map2alm` at the refined
sphere's `lmax`, then only ℓ ≤ 2·`cmb_nside` is kept, so it adds no window inside the band. The one
smoothing the observable carries is that of the bilinear scatter itself, whose window
(`cmb_lensing.bilinear_window`, measured on a band-limited Gaussian field) falls with the pixel
size: in amplitude at ℓ = 30 / 50 / 64 it is 0.906 / 0.778 / 0.668 at nside 32, 0.977 / 0.936 / 0.898
at nside 64 and 0.994 / 0.984 / 0.974 at nside 128. `proj_oversamp: 2` therefore also halves the
in-band smoothing of scattering straight onto the observable sphere. The likelihood stays
consistent under it as long as data and model carry the same window: in closure by construction,
on Abacus because the loader brings the map through the same kernel (§4). A real κ map does not
carry this window.

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
the predicted intensity `gxy_count · S` and the observed count are on one window; and the
band-limited product predicts the covered sub-region directly, rather than a whole-cell average
compared against a sub-region count. The code default stays 0.8; the HUGE reference runs use **0.3**
(the scan that chose it is in §7.1). Below that, `b1` is the first parameter to move.

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

The nuisance is only needed when the randoms mis-trace `n̄(χ)`. The base-box AbacusLensing randoms
do (their `RAND_Z` is close to uniform in z and under-traces the outer comoving volume), so the
base-box runs free them. The HUGE `lrg_huge` randoms track the data `n(z)` to 0.1 % (§4), so the
HUGE reference runs keep `gxy_ngbar_free: false`; freeing them there changes `f_NL` by 0.15σ and
widens it (§7.1).

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
  (**the spectrum of `diagonal`, no HEALPix pixel window**; `C_ℓ^{high-z}` carries the bilinear
  window of the projection sphere, below — this makes `diagonal` and `pixel_exact` coincide at
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

**Line-of-sight correction (`full_los_correction`, `high_z_mode`).** `C_ℓ^{high-z}` for the
missing depth `χ_box → χ_high_z_max` is the Limber convergence power (`jax_cosmo`), added to the
variance. With the per-shell multipole cut (§2.6) the power the cut removes from the shells is added
to the same term, so `C_ℓ^{high-z}` in the formulas above is really the covariance of everything the
data map holds and the model does not, and every mode below treats both parts alike (one cache, one
pair of `taylor` gradients, one recomputation in the `exact` modes;
`test_every_high_z_mode_carries_the_cut_term_alike`).

**The window of the term.** The term is multiplied by `w_ℓ²`, the squared bilinear window of the
projection sphere (`bilinear_window` at `proj_nside`, `model.cmb_los_window2`; 0.88 at ℓ = 50 and
0.81 at ℓ = 64 at `proj_nside` 64), in every mode: the `exact` computation carries it, and the
`fixed` cache and the `taylor` gradients are built from that computation. The data map reaches the
observable through that kernel, the line-of-sight matter and the power the cut removes included —
on Abacus through the loader's `bilinear_resample_healpix` (§4), on a real map by the same
resampling — while the model map gets it from the Born scatter. The residual data − model is
therefore `w_ℓ (κ_LOS + Σ_s (1 − w_s(ℓ)) κ_s) + noise`, of variance `w_ℓ² C_ℓ^{high-z} + N_ℓ`; the
noise is added after the resampling and carries no window. In closure the data are drawn from the
likelihood itself, so they follow whatever the term holds. Without the window the assumed variance
of that part would be `1/w_ℓ²` too large, 1.14 × at ℓ = 50 and 1.24 × at ℓ = 64: small while the
term held only the matter beyond the box (2–6 % of the Abacus power over the band), not once it
holds what the cut removes (17–25 % of it at ℓ ≥ 48, §7.12). Modes:
- `fixed` — cached at the fiducial cosmology (required by `pixel_exact`, exact if `Ω_m,σ8` fixed).
- `taylor` (default) — first-order expansion `C_ℓ(θ) ≈ C_ℓ(θ_fid) + ∇C_ℓ·Δθ`, gradients precomputed.
- `exact_linear` — recompute the Limber integral each step with the **linear** P(k) (slow).
- `exact` — full recompute each step (very slow).

**Depth of the data map.** The model shells hold the matter from `chi_matter_min` to `chi_boundary`;
the line-of-sight term holds the matter the data map contains beyond the box, `chi_boundary →
chi_high_z_max`, plus what the per-shell cut removes from the shells (§2.6). On a real map the map's
matter runs from the observer to `χ_s` (9388 Mpc/h at the fiducial cosmology). Every run here,
closure and Abacus, has `chi_matter_min: 292.6` (z = 0.10, where the AbacusSummit HUGE light cone
stops) and `chi_high_z_max: 3942` (z = 2.40, the last lens plane of `kappa_00045`) — the closure
configs copy the Abacus range. The matter below 292.6 and beyond 3942 Mpc/h is therefore in neither
the data nor the covariance: data and likelihood agree, but the map is that of an experiment without
that matter. Limber (non-linear P, Abacus cosmology, `scripts/plot_lensing_fraction.py`), fraction
of the power from the observer to `χ_s`, ℓ bins 2–4 / 5–10 / 11–20 / 21–36 / 37–64 weighted by
2ℓ+1, before the per-shell cut: model shells (292.6 → 3750) 0.76 / 0.82 / 0.81 / 0.76 / 0.69; data
map 0.77 / 0.83 / 0.83 / 0.79 / 0.72; beyond the box 0.011 / 0.015 / 0.020 / 0.026 / 0.033; in
neither 0.23 / 0.17 / 0.17 / 0.21 / 0.28. With ACT DR6 `N_ℓ` ×1, `N_ℓ` plus the beyond-the-box term
is 0.81 / 0.79 / 0.74 / 0.68 / 0.62 of `N_ℓ` plus all the unmodelled matter of a real map: the κ
constraints and gains are those of a map with that smaller variance. Figures
`figures/lensing_fraction/lensing_fraction_vs_z.png` and `lensing_spectra_comparison.png`.

Noise: ACT DR6 `N_ℓ^{κκ}` (`data/N_L_kk_act_dr6_lensing_v1_baseline.txt`, columns ℓ, N_ℓ) or Planck
PR4; `cmb_noise_scaling` (default 1.0) multiplies it to test sensitivity. The ACT file is constant
(8.766·10⁻⁸) below L = 40 and varies from L = 40 on, so most of the band ℓ ≤ 64 sees a flat noise.
The band is signal-dominated: the full `C_ℓ^κκ` (observer to `χ_s`, Limber) over this `N_ℓ` is
1.8, 2.4, 2.2 at ℓ = 10, 30, 64, and crosses 1 at ℓ ≈ 150 (0.96). Noise realizations:
`sample_healpix_gaussian`; map RMS: `compute_sigma_hp`.

### 3.3 Geometry & observer

- **Observer** (`cmb_lensing.observer_mode`: `center` / `face` / `corner`, or explicit
  `observer_position`): sets `box_center = box_shape/2 − observer_position`. `corner` places the
  AbacusSummit base-box octant footprint inside `[0,L]³` at full depth. The 3-D galaxy mesh and the 2-D
  κ map share this observer (galaxies at their true (RA,DEC,Z), Born ray-cast from the same point), so
  both probes cover the same lightcone volume — no box rotation (the box is axis-aligned).
- **Curved-sky scope** (`curved_sky`): galaxy LOS/RSD and CMB projection; CMB lensing is always
  curved-sky HEALPix.
- **Even mesh**: all axes forced even (the real↔complex Gaussian repacking, §2.1).

---

## 4. Data & observation modes

`observation_mode` (`config.yaml`, `utils.ObservationMode`) supports **exactly two values**,
`closure` and `abacus`. **Real observational data (DESI-LRG × Planck/ACT κ) is not yet supported** —
it is a roadmap item, not a runnable mode; the only external data the pipeline ingests today is
AbacusSummit.

- **`closure`** — synthetic `obs`/`kappa_obs` generated from `truth_params` via `model.predict`
  (validation; data-generation and likelihood share the same distribution object). `truth_params`
  must give every scalar latent the prior samples (`FieldLevelModel.sampled_scalar_latents`):
  `run_inference.py` stops otherwise, since `predict` would draw the missing ones from their prior.
  The `model` and `cmb_lensing` sections of the config are checked against the keys
  `get_model_from_config` reads (`MODEL_CONFIG_KEYS`, `CMB_CONFIG_KEYS`): an unknown key is an error,
  not a silent fallback on the default.
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
  validation setup (§7).

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

**Abacus κ ingestion** (`load_abacus_kappa_observation`): the HEALPix map reaches the observable
through the Born projector's own angular kernel, so that data and model carry the same window. It is
averaged to `8 × cmb_proj_nside` (window 1 in the band), then each of those pixels is spread over its
four nearest `cmb_proj_nside` pixels with the projector's bilinear weights and each output pixel is
divided by the weight it received (`bilinear_resample_healpix`). A plain `ud_grade` to `cmb_nside`
carries the nside-32 pixel window instead, 0.94 × the model's in amplitude at ℓ = 64 (measured on a
band-limited random field). The noise is a Gaussian realisation band-limited to `cmb_lmax`, drawn on
the projection sphere; both maps are packed through the refined branch of `pack_kappa_map`. Both maps have a CMB source (`SourceRedshift = 1089.3`)
but matter only up to the last lens plane, which sets `chi_high_z_max`: `kappa_00045.asdf` (HUGE
`c000_ph201`, full sky) stops at χ≈3942, `kappa_00047.asdf` (base `c000_ph000`, two patches) at
χ≈3990.

**Cosmology of the Abacus maps.** AbacusSummit `c000` has `Ω_M` = 0.315192 in the background, of
which `Omega_Smooth` = 0.00142 is a smooth, non-clustering neutrino component (IC header). The
particles are CDM + baryons: the ICs are drawn from the CLASS CDM+baryon power spectrum at z = 1
and scaled back with a growth function holding the neutrinos as smooth matter (AbacusSummit
documentation, *Cosmologies*). Their amplitude is therefore `σ8_cb` = 0.811355, not the
total-matter `σ8_m` = 0.807952 (AbacusSummit `cosmologies.csv`, row `abacus_cosm000`); the configs
and the background emulator use 0.811355. `bricks.AbacusSummit0` (from cosmoprimo) carries
0.80764, which is neither; nothing reads it. AbacusLensing Born-integrates those particles with the
prefactor `3/2 Ω_cb H_0²/c²`, `Ω_cb = Ω_b + Ω_c` = 0.313770 (Hadzhiyska et al. 2023, §3.1, eq. 3).
The model has no neutrinos: its `lensing_kernel` uses `Omega_m` = 0.315192 with the same δ, so its κ
is 0.315192/0.313770 = 1.0045 × that of AbacusLensing in amplitude for the same field.

**Abacus galaxy loading** (`load_abacus_galaxy_observation`): either one FITS lightcone with a
separate randoms file (HUGE `lrg_huge`, above), or a list of per-z-shell ASDF files
(`RA, DEC, Z_COSMO/Z_RSD`, `RAND_*`, base-box AbacusLensing) via `bricks.catalog2positions`/`randoms2positions`, deduplicated
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
normalisation is needed; its spectrum matches `P_lin` to the Eisenstein-Hu-vs-CLASS difference,
3–4 % at k ≤ 0.03 h/Mpc at the same σ8, §7.11).
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
non-Gaussianity parameters (§2.2); the one exception is the κ-only comparison with a two-point
analysis (§8 step 4b), which frees them and fixes `fNL`. In practice this means: put `Omega_m, sigma8` in
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
treats the shells nearest the observer (§2.6, the per-shell multipole cut and `shell_weights`). With
those two set, the
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
  the model's `matter_mesh` against theory — on a lightcone, `P_lin(k, a=1)` times the volume average
  of `D(a(χ))²` over the cells, since a lightcone box mixes epochs and matches no single-`a`
  spectrum; in snapshot mode, nonlinear `P_mm` at `a_obs`. In **abacus** mode: galaxy `P_gg(k)` as
  seen by the likelihood, Abacus observed catalog vs LPT-simulated galaxies at the config cosmology
  (same survey mask, Poisson shot-noise subtracted). Binning runs past `k_Nyq` out to the k-space
  cube diagonal `√3 k_Nyq`: the sphere `|k| < k_Nyq` holds only `π/6` of the modes, and the corner
  modes enter the cell-wise likelihood at full weight, so they must be checked too.
- **`quick_cl_spectra.py`** — angular spectra `C_ℓ^{κκ}, C_ℓ^{gg}, C_ℓ^{κg}` against Limber, closure
  (N realisations, mean and error of the mean) or abacus (the simulation maps). One figure,
  `validation.plot_cl_figure`, shared with the startup check of `run_inference.py`, with **one
  Limber curve per spectrum** (`validation.compute_cl_theory`), because the measured maps are made
  comparable to it rather than the theory decorated with noise terms:
  - **κ** is the noiseless map (the `N_ℓ` realisation is not shown). Closure theory: the model's
    Born shells, `chi_matter_min` to `chi_boundary`, keeping only `k_⊥` below the **init-grid
    Nyquist** — the inferred linear field has no power above it, while the final-mesh Nyquist would
    cut modes the particles do carry — each shell times its squared multipole taper when the
    per-shell cut is on. Abacus theory: the line of sight from `chi_matter_min` to `chi_high_z_max`.
    Both are multiplied by the window of the bilinear kernel at `cmb_proj_nside`
    (`cmb_lensing.bilinear_window`, measured on a band-limited Gaussian field: 0.936 at ℓ = 50 and
    0.898 at ℓ = 64 for nside 64), which model and data κ both carry. The galaxy map is divided by
    its own window: HEALPix's pixel window for a catalogue histogram, the bilinear window for the
    model's particles.
  - **κg** in closure multiplies the lensing kernel by the radial window of the Born shells
    (`cmb_lensing.kappa_radial_window`: the sum of the tents, or top hats, times the angular support
    of each shell). Where the shells start or stop inside the galaxy range, the model κ does not hold
    the matter those galaxies trace, and the theory must not either.
  - **Galaxies** are projected from the galaxies, never from the mesh: the catalogue count map the
    Abacus loader stores (`gxy_hp_counts`, nside 256, galaxies whose nearest final node is in the
    survey), or the model's particles (`rsd_pos`) weighted by their bias weight (`gxy_weights`),
    the selection and the survey mask, and spread bilinearly over the four nearest pixels as the
    Born projector spreads them for κ, then divided by `bilinear_weight_norm` as the projector
    divides (§2.6). This map only enters the diagnostic spectra, never the likelihood, which works
    on the 3-D count mesh (§3.1). The particles sit on a displaced lattice; wherever a pixel
    is smaller than the lattice spacing (at nside 64, everywhere closer than χ ≈ 3400 Mpc/h for a
    54 Mpc/h spacing), nearest-pixel counts carry a moiré pattern at every ℓ, uncorrelated with κ,
    that inflates `C_ℓ^{gg}` by 10–60 % and not `C_ℓ^{κg}`. Catalogue galaxies are Poisson points
    and take the plain histogram. Either is divided by the unclustered expectation — the
    selection ray-integrated with the `r²` volume weight, which also gives the `dN/dχ` kernel of
    the theory. The HEALPix pixel window is removed and, for the Poisson catalogue, the shot noise
    subtracted; the particles are a displaced lattice and carry none. Mesh interpolation would put
    a low-pass window in the map that no Limber curve contains (≈20 % of `C_ℓ^{gg}` at 0.4 `k_Nyq`
    for trilinear sampling), which is why it is not used. The galaxy map is built at twice
    `cmb_nside`; κ, band-limited to `lmax`, is resynthesised there for the cross. Truths without
    particles or catalogue map (`evolution: kaiser`, old runs) fall back to the mesh nodes as points,
    cell window left in, with a warning.
  Both scripts condition on **every** latent (`validation.conditioning_params`, missing ones held
  at `loc_fid`), since `predict` would otherwise draw them from their priors.
- **`analyze_run.py`** — merge batches, R-hat/ESS, corner/trace plots (abacus mode uses
  `abacus_truth_params` markers), custom burn-in / chain exclusion, and two **field-level figures**.
  Runs automatically at the end of every job.
  - `initial_conditions.png` — true field, reconstruction, difference, then `T(k)`/`r(k)` and a
    radial profile (rms ratio, correlation with the truth, agreement between chains). The slice is
    the plane through the **observer** in observer-centred coordinates, right for every
    `observer_mode`; dashed circles mark the galaxy radial range.
  - `kappa_reconstruction.png` (joint and CMB-only runs) — observed κ, model κ, model error and their spectra against the
    covariance the likelihood assumes, all band-limited to the likelihood's own band at `cmb_nside`
    (a `proj_oversamp > 1` model map included): the figure shows the reconstruction the inference
    uses, not a finer one. The last panel is the harmonic coherence
    `r_ℓ = C_ℓ^{tm}/√(C_ℓ^{tt} C_ℓ^{mm})` in bins of Δℓ = 4: each chain's sample against the truth,
    the posterior mean against the truth, and the agreement between chains, each with its
    expectation for a correct Gaussian posterior (§7.4: `W·√(S_t/S_m)`, `√(W·S_t/S_m)` and `W`,
    with `S_t = S_m + L` on an external map and `S_t = S_m` in closure). A model error pulls the
    coherence with the truth below its expectation; a structure the chains share but the data do
    not impose pushes the agreement between chains above its own. These expectations are κ's own
    Wiener filter, i.e. they hold for a κ-only run. In a joint run the galaxies also inform the field,
    so these curves are not its expectation.
  - `kappa_posterior_mean.png` (runs that record κ per batch) — truth, posterior mean, per-pixel
    posterior standard deviation and `(truth − mean)/std`, ℓ = 2 … `lmax`. `run_inference.py`
    writes, after every batch, the `kappa_obs` observable (packed `a_ℓm`, or KL amplitudes) of each
    chain's current state to `kappa_batch_*.npz`, via `FieldLevelModel.kappa_observable` (one
    forward model per chain and batch, with the fixed latents); `analyze_run.py` keeps the batches
    after burn-in. Without these files the mean is over the final states of the chains only.
  - The maps of both figures are **one posterior sample, not a mean over chains** — averaging independent samples
    suppresses the amplitude wherever the data does not constrain the field, which reads as lost
    power when every sample carries the right power. Read from `sampler_state.pkl`, the only place
    the field survives (`samples_batch_*.npz` hold scalars only). One forward model each (one per
    chain for the κ figure); `--no_field_plots` skips them.
  - The scalars held by `mcmc.fixed_params` are not in the sampler state; both figures add them
    at the values `run_inference.py` conditioned on (`truth_params` in closure,
    `abacus_truth_params` on Abacus). Without them `predict` draws those latents from their priors.
- **`compare_reconstruction.py`** — correlates the true IC with the sampled one in radial shells
  around the observer, per chain, overlaying several runs (joint vs galaxy-only); this is what
  measures where κ extends the reconstruction (§7.1). Like `analyze_run.py`, it adds the fixed
  latents before `reparam`, so the field is built at the run's own cosmology.
- **`validate_kappa_from_ic.py`** — the κ model against AbacusLensing with no sampling: the HUGE
  ICs (`abacus_ic`) go through the forward model at the Abacus cosmology, galaxies off, once per
  `--shell_kmax` (default the config's; 0 = no cut), and the κ is compared with the map in the
  likelihood's band. Per ℓ bin: coherence, transfer, error power against `N_ℓ` and against `C_ℓ^LOS`
  (the beyond-the-box term plus what the cut removes), the coherence an exact model would reach given
  what it puts in the covariance, `√(1 − C^LOS/C^tt)`, and `(C^mm + C^LOS)/C^tt`, which is 1 for a
  model consistent with the map. The first call loads the
  nside-16384 map (compute node) and caches it on the projection sphere; `--abacus_map` takes a
  map already on disk. The figure goes to
  `figures/spectra_diagnostic/kappa_from_abacus_ic_cell<cell>.png` (or `--fig`), the spectra next to it
  as the same name `.npz` (`--replot` redraws from it); the Abacus map on the projection sphere is
  cached in `data/cache/` (git-ignored).
  `--maps` also saves the model maps and plots Abacus, model and their difference around both
  poles, at full resolution and at ℓ ≤ `cmb_lmax`
  (`figures/maps/kappa_from_abacus_ic_poles_cell<cell>_kmax<shell_kmax>.png`), printing the difference in the
  eight polar-cap pixels against the rest of the sphere (§7.9).
- **`kappa_stiffness.py`** — the largest eigenvalue of the Hessian of −log p with respect to the
  field (`init_mesh_`, the sampled basis), by power iteration on exact reverse-over-reverse
  Hessian-vector products (the model holds `custom_vjp` functions, which forward mode cannot
  enter), per `shell_weights:shell_kmax[:cmb_noise_scaling]` variant (`shell_kmax` in h/Mpc, 0 = no
  cut, `auto` = the config's), galaxies off, at a closure draw; the fraction of the top eigenvector's
  variance near the observer; the time of a Hessian-vector product and the device's peak memory. A κ
  noise scaled by 1e8 gives the prior alone, λ = 1. The MCLMC step size is capped by that eigenvalue.
- **`compare_runs.py`** — GetDist triangle comparison of multiple runs (per-run burn-in, labels).
- **`density_scan.py`** — the density-scan figure (`figures/results/density_scan.png`) and table:
  for each density of `configs/inference/scan/density_scan_runs.yaml` (paired closure runs, `joint:
  null` until run), the measured galaxy-only and joint σ(f_NL) and the paired gain; against them,
  on a density grid, the Fisher forecasts below (galaxy-only width from the 3-D Fisher, κ gain from
  the Limber increment added to it). The κ geometry, cell and fiducial biases come from the
  registry's `fisher_config`. `--noise_table RUN --config CFG [--density d]` prints `σ_κ` and the
  Fisher gain per `N_ℓ` scaling on the galaxy Fisher measured from `RUN` (§7.2). Numbers also in
  `figures/results/density_scan.json`. Runs are looked up on scratch, then among the runs kept on CFS
  (`docs/hpc.md`).

**Fisher forecasts** (`desi_cmb_fli.fisher`, parameters `(f_NL, b1, b∇²)`, fiducial `f_NL` = 0; tested
against closed forms in `tests/test_fisher.py`).

- *Galaxies alone, 3-D* (`galaxy_fisher_3d`). At linear order the field level of a Gaussian field
  holds the information of its power spectrum, so
  `F_ab = ∫ dV ∫ k² dk/(2(2π)²) ∫₋₁¹ dμ ∂_aP ∂_bP / (P + 1/n̄)²` with
  `P(k, μ, z) = (b(k, z) + f μ²)² P_lin(k, z)`, `b = 1 + b1 − b∇² k² + b_φ f_NL / M(k, z)`,
  `b_φ = 2 δ_c b1` (universality, the model's `png_type: fNL`, §2.2) and `M(k, z) = M(k, 0) D(z)`
  the φ → δ transfer. Volume: the full-sky light-cone shell of the LRG redshift range; `n̄(χ)` the
  catalogue `n(z)` histogram (piecewise constant over its bins) times the density scale. Band:
  `k_max = π / cell`, the Nyquist frequency of the likelihood mesh; `k_min = 2π / V^(1/3)`. The
  result depends strongly on `k_min`, which a full-sky shell does not define sharply: halving or
  doubling it changes σ(f_NL) by ×0.7 and ×1.6 at density 1; `√3 k_max` (the mesh corners) by ×0.9.
  Left out: `b2`, `b_{s²}` (they do not enter the linear spectrum; their measured correlations with
  `f_NL` are ≤ 0.09, §7.3), `b∇∥` (fixed at 0), the survey mask and window.
- *What κ adds* (`kappa_fisher_increment`). A tomographic Limber Fisher over κ's band
  (`ℓ_min = k_F χ_eff` to `ℓ_max = 2·nside`), `ΔF = F[galaxy shells + κ] − F[galaxy shells]` with
  `F = Σ_ℓ (2ℓ+1)/2 tr(C⁻¹ ∂_aC C⁻¹ ∂_bC)`, ten equal-number galaxy shells, galaxy bias
  `b_E − b∇² k² + b_φ f_NL / M` with no Kaiser term (it does not enter angular Limber spectra of
  broad shells, and never the κ–galaxy cross), κ from the matter between `chi_matter_min` and the box
  edge under the per-shell cut as a continuous weight, the taper at ℓ_res = `shell_kmax`·χ
  (`shell_cut_taper`, `kappa_spectra`: signal and κ–galaxy kernel times it, the removed power as
  independent noise, like the likelihood), and as κ noise `N_ℓ` plus the line of sight beyond the box
  (§2.6). `σ_κ = √(ΔF⁻¹)_{f_NL f_NL}` is κ's *incremental* information through the
  galaxies, not a κ-only constraint: κ alone sees `f_NL` only through the weak matter φ² channel
  (§2.2). The joint forecast adds `ΔF` to a galaxy Fisher, either the 3-D one above (the curve) or
  the one measured from a galaxy-only chain, `inv(Cov[f_NL, b1, b∇²])` (the table); `ΔF` is the
  increment over *angular* galaxy information, so adding it to the 3-D one is an approximation.
- *Cosmology* (`galaxy_fisher_3d_cosmo`, `angular_fisher_cosmo`, `fisher_cosmo.py`). The same two
  Fishers on (`Omega_m`, `sigma8`, `b1`, `b∇²`, optionally `f_NL`), derivatives by central
  differences over one `Background` per cosmology. As at the field level the galaxies sit at their
  fiducial distances (the data are painted once): a cosmology moves z(χ), hence growth, growth rate
  and linear power, and in the κ kernel `Omega_m` and χ_s, but no Alcock–Paczynski information.
  Galaxies: the 3-D Fisher with RSD; κ alone: the Limber Fisher of κ, ℓ = 2 … 2·nside; joint: the
  3-D galaxies plus `F[shells + κ] − F[shells]`. The line of sight is noise at the fiducial
  (`high_z_mode: fixed`) or signal following the cosmology. The config's Gaussian priors are added.
  Tests: the σ8 information of κ alone and of the galaxies against closed forms, the exact b1–σ8
  degeneracy of angular galaxies and its breaking by κ, and `kappa_fisher_increment` recovered at
  fixed cosmology.
- *Measured paired gain* (`paired_sigma_ratio`). The galaxy-only and joint runs of a density share
  `seed` and warm start, so chain `i` of one pairs with chain `i` of the other: the ratio is the mean
  over chains of σ_joint/σ_gxy (second half, at the number of batches both runs have), its error the
  spread over chains / √n_chains. The σ ratio of the chains pooled together is a different estimator
  and is not used.
- **`fisher_cosmo.py`** — the Fisher forecast on (`Omega_m`, `sigma8`, `b1`, `b∇²`) of galaxies alone,
  κ alone and joint, at a run config's geometry, band, noise, truth biases and priors (default the
  closure at the Abacus geometry), line of sight as noise or as signal; table,
  `figures/fisher_diagnostic/fisher_cosmo_contours.png` (68 % contours) and
  `figures/fisher_diagnostic/fisher_cosmo.json` (§7.10); `--resolution_scan` repeats it
  over box size and cell, the cut at each cell's init-grid Nyquist (§7.10); `--mismatch FILE` gives
  the first-order shift F⁻¹b of the parameters from a κ power mismatch (data/model ratio per ℓ: the
  `kappa_from_ic.npz` of `validate_kappa_from_ic.py` at the config's `shell_kmax`, or two columns ℓ,
  ratio), with b_a = f_sky Σ (2ℓ+1)/2 tr(C⁻¹ ∂_aC C⁻¹ ΔC) (§7.10).
- **`compare_linear_power.py --omega_m_scan`** (needs `camb`, not in the environment) — at fixed σ8,
  each linear-power option against CAMB as `Omega_m` moves off the fiducial: Eisenstein–Hu times the
  fiducial Boltzmann/EH ratio (`lin_pk_table`), Eisenstein–Hu alone, and the fiducial shape held
  fixed; `figures/spectra_diagnostic/linear_power_omega_m_accuracy.png` and its `.npz` (§7.11).
- **`compare_linear_power.py`** — the model's linear P(k) (jax_cosmo Eisenstein–Hu) against the CLASS
  spectrum the AbacusSummit ICs were drawn from (`data/abacus_cosm000_CLASS_power.txt`, AbacusSummit
  `Cosmologies/abacus_cosm000/CLASS_power`, the `ZD_Pk_filename` of the HUGE IC header), both
  normalised to σ8 = 1, table and `figures/spectra_diagnostic/linear_power_eh_vs_class.png` (§7.11).
- **`plot_2D_maps.py`** — κ (and galaxy-projection) maps for one forward realization. The galaxy panel
  ray-casts the mesh (`project_mesh_to_healpix`, node `i` at `i·dx` as in the painting) — display only.
- **`benchmark_highz_cl_modes.py`** — precision/speed of the high-z modes; **`plot_lensing_fraction.py`**
  — Limber κ power split along the line of sight at a run config's geometry (`--config`, default
  the Abacus joint): the fraction captured vs depth from the observer and from `chi_matter_min`,
  and the spectra of the model shells (before the per-shell cut), the data map, the covariance's
  beyond-the-box term, the part in neither and `N_ℓ` (§3.2); **`make_abacus_kappa_mask.py`** — degrade the AbacusLensing
  footprint to a model-nside `.npy`; **`plot_highz_correction_vs_depth.py`** — mean high-z κ correction
  amplitude vs box depth (`χ_min`); **`plot_linear_vs_nbody.py`** — linear (Kaiser) vs N-body evolved
  matter field from the same IC; **`plot_cmb_noise_comparison.py`** — ACT DR6 vs Planck PR4 κ noise
  spectra.
- **Freeze diagnostic** (`DIAGNOSE_FREEZE=1`): logpdf + per-group gradient + 1-D scans of each free
  scalar at the STEP-2 start; localises curvature pathologies.

---

## 7. Validation status & roadmap

**Validated.** On the Abacus **CubicBox snapshot** (periodic full box, z=0.8), the
galaxy-only run under `png_type: fNL_bias` reproduces the montecosmo result — unbiased `f_NL`
(≈ −111 ± 500).

**Earlier configuration (base-box lightcone, octant).** Galaxy-only with fine-bin `ngbars`,
`png_type: fNL` and oversampling gives `f_NL = 1.5 ± 33`. Under `png_type: fNL_bias` the
scale-dependent bias amplitude `fNL_bp ≈ 0` (the meaningful observable) while the decoupled
matter-φ² `fNL` wanders; `fNL_bias` is not identifiable and is not used. These runs predate the
number-count likelihood and are not comparable with anything below.

**Reference configuration: the HUGE box, §7.1–7.7.** Every current result uses it, in abacus mode
(§7.1–7.2, 7.4–7.6) and in closure at the same geometry (§7.3, 7.7).

**κ model of these results.** The κ and joint runs and computations of §7.1–7.11 predate the
per-shell multipole cut: their shells start at the `chi_min` of their config (350 or 700 Mpc/h)
with the dropped range as a low-z covariance term, the abandoned set-up of §2.6, and their configs
and commands use `chi_min` and `chi_low_z_min`, which the current code rejects. §7.12 measures the
current κ model against them. Galaxy-only results do not depend on it.

### 7.1 Full-sky κ × LRG on the HUGE box — measured

Reference pair, identical configuration apart from `cmb_lensing.enabled`: completeness cut 0.3,
`gxy_ngbar_free: false`, `png_type: fNL`, counts likelihood, ACT DR6 `N_ℓ` at ×1, `nside 32`,
`chi_min 350`, `shell_weights: linear`, `proj_oversamp: 2`. Both runs, and those of §7.2, use a
uniform solid angle per pixel in the projector and read the κ map through `ud_grade` (§7.9).

| | run | batches |
|---|---|---|
| joint | `run_20260910_070354_58161527` | 92 |
| galaxy-only | `run_20260910_033019_58153868` | 55 |

**How the survey treatment was chosen.** Galaxy-only, `png_type: fNL`, counts likelihood, second
half of each chain:

| completeness cut | `ngbars` | survey cells | `f_NL` | `b1` | run |
|---|---|---|---|---|---|
| 0.8 | free | 49 954 | 6.6 ± 6.7 | 1.173 ± 0.023 | `58118827` |
| 0.3 | free | 63 614 | 9.8 ± 6.2 | 1.191 ± 0.022 | `58153728` |
| **0.3** | **fixed** | 63 614 | **8.9 ± 5.8** | 1.192 ± 0.019 | `58153868` |
| 0.05 | free | 70 911 | 10.2 ± 5.8 | 1.205 ± 0.020 | `58152051` |

Lowering the cut from 0.8 to 0.3 adds 27 % of cells and tightens `f_NL` by 8 % with `b1` stable;
going to 0.05 adds little more and starts to pull `b1` up. Fixing `ngbars` at 0.3 moves `f_NL` by
0.15σ and tightens it by 6 %, as expected from randoms that trace `n(z)` (§3.1).

**The central value.** `f_NL = 8.9 ± 5.8` on a Gaussian simulation is +1.5σ from zero, and the
offset is common to every row above, so it belongs to this data set rather than to a survey-treatment
choice. A 1.5σ fluctuation is not by itself a detection of bias (p ≈ 0.13). The closure at this
geometry (§7.3) is unbiased, but it draws its own initial field, so it does not test this particular
realisation; only a closure on the Abacus ICs would.

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

### 7.2 Dependence on the CMB lensing noise level — measured

Same configuration as §7.1 with `cmb_noise_scaling: 0.1` (ACT DR6 `N_ℓ` divided by ten). Config-diffed:
across the triplet only `cmb_lensing.enabled` and `cmb_noise_scaling` differ.

| | run | batches | f_NL | σ | ESS |
|---|---|---|---|---|---|
| galaxy-only | `run_20260910_033019_58153868` | 55 | 8.88 | 5.78 | 442 |
| joint, `N_ℓ` ×1 | `run_20260910_070354_58161527` | 92 | 8.97 | 5.69 | 727 |
| joint, `N_ℓ` ×0.1 | `run_20260922_010054_58741397` | 140 | 8.71 | 5.77 | 990 |

All three converge (×0.1: step size 84.0 on every chain, energy variance 0.88 × target,
R-hat ≤ 1.008).

**Dividing the CMB noise by ten does not tighten `f_NL`.** The two joint runs, which differ only in
the κ noise, give σ(×0.1)/σ(×1) = **1.007 ± 0.062** at matched 92 batches. Ratios against the
55-batch galaxy-only run are not quoted: they move with where the chains are cut (1.014, 1.039,
0.956 at 27, 41, 55 batches).

**So the null is structural, not noise-limited.** Effective κ modes `Σ (2ℓ+1)[S/(S+N)]²`, ℓ ≤ 64:
1599 at ×1, 3737 at ×0.1, 4221 at zero noise. ×0.1 already holds 89 % of what a noiseless experiment
could give, and σ(f_NL) does not move. For the same reason a finer cell is not worth its cost:
ℓ_max = 128 needs a 26.9 Mpc/h cell, ×8 the cost, for ×2.5 the modes.

**Precision.** `density_scan.py --noise_table run_20260910_033019_58153868 --config
archive/abacus/abacus_joint_Nl1p0_chimin350.yaml` (§6, the geometry of these runs) expects 4.7 % at ×1 and 8.5 % at ×0.1 (`σ_κ` = 18.9 and 13.6): the
measurement is consistent with that as well as with zero, so the claim
is "at most a few percent". The 55-batch galaxy-only run caps the precision; resuming it is the
cheapest improvement.

**Not a forecast.** Lowering `N_ℓ` on Abacus lowers the assumed covariance but not the model error,
so the ×0.1 run is over-confident on κ; only closure makes the noise axis a forecast (§7.3).

The pre-refactor ×0.1 run `58027570` must not be reused: pairing it across the counts refactor
(§3.1) gave a spurious +13.6 %.

### 7.3 Closure at the HUGE configuration and the tracer-density scan — in progress

**Why closure at this exact geometry.** Closure isolates information from model error. Run at the
§7.1 configuration, the closure pair and the abacus pair differ by the model error alone: same box,
cell, `nside`, survey selection, completeness cut, density, priors and bias values. It also makes the
`N_ℓ` axis a valid forecast, which §7.2 notes it is not on Abacus (lowering `N_ℓ` drops the assumed
covariance but not the model error).

**Wiring.** `closure_geometry_from_abacus: true` calls the Abacus galaxy loader for its side effects
only (`selec_mesh`, `selec_paint`, `gxy_occ_mask3d`, `chi_range_gxy`), discards the catalogue counts,
and lets `model.predict` generate the synthetic counts *through* that selection, so the galaxies
occupy the LRG shell χ ∈ [1094, 2447] exactly as in abacus mode and the galaxy/κ overlap is the same.
`closure_gxy_density_scale` multiplies the catalogue n̄, so 1.0 reproduces the abacus-mode density
exactly (`gxy_count` 266.920 in both); `model.gxy_density` is not the knob, the loader overwrites it.
`truth_params` pin every sampled latent to the §7.1 galaxy-only posterior means (`b1 1.19`,
`b2 0.68`, `bs2 −0.48`, `bn2 78`, `bnpar −45`, `f_NL 0`). The initial field is a fresh Gaussian draw
(`seed`), not the Abacus ICs. Configs: `configs/inference/scan/closure_d<scale>_gxyonly.yaml` and, for the joints measured
with the abandoned inner cut-off, `configs/inference/archive/scan/closure_d<scale>_joint_chimin{350,700}.yaml`,
which differ from each other only in `job_name`, `closure_gxy_density_scale`,
`cmb_lensing.enabled` and `cmb_lensing.chi_min`.

**Density 1.0, galaxy-only** (`run_20260922_071921_58748954`, 31 batches before its resume,
second half): `f_NL` −2.7 ± 4.3 (pull −0.62), R-hat ≤ 1.05, every bias within 0.5σ of its truth. The field reconstruction
behaves as on Abacus: `r ≈ 0.96` inside the galaxy shell, zero outside, `T(k) ≈ 1`. Against the
abacus-mode twin `58153868` at the same geometry, σ ratios closure/abacus are 0.75 (`f_NL`), 0.91
(`b1`), 0.90 (`b2`), 0.69 (`bs2`), 0.93 (`bn2`), 0.93 (`bnpar`). Not yet interpretable as model
error: at field level σ(f_NL) depends on the realised large-scale potential, and the two runs do not
share their initial field. Its joint twin is in the table below.

**The density scan.** Densities 1.0, 0.2, 0.1, 0.03 × catalogue n̄, one galaxy-only and one joint run
each, `N_ℓ` ×1. The measured quantity is the paired gain `1 − σ_joint/σ_gxy` at each density.

| density | galaxy-only | joint | batches (matched) | σ(f_NL) gxy → joint | paired ratio | two-point Fisher gain |
|---|---|---|---|---|---|---|
| 1.0 | `run_20260922_071921_58748954` | `run_20260923_024316_58785136` | 124 | 4.43 → 4.02 | **0.910 ± 0.034** | 2.8 % |
| 0.03 | `run_20260923_054507_58787249` | `run_20260923_054540_58787260` | 103 | 14.1 → 11.6 | **0.829 ± 0.027** | 3.4 % |

The Fisher column is `density_scan.py --noise_table <galaxy-only run> --config
archive/scan/closure_d1p00_joint_chimin350.yaml` (the `chi_min` 350 geometry of these runs): the Limber increment on
the galaxy Fisher measured from the galaxy-only chain of the same density (§6, Fisher forecasts).

**Density 0.1, galaxy-only** (`run_20260924_071600_58822464`, `scan/closure_d0p10_gxyonly.yaml`,
140 batches, second half): `f_NL` 4.3 ± 7.5 (pull +0.58), R-hat ≤ 1.003, every bias within 1σ of
its truth (`b1` −0.90σ), step size 172. With the other two densities (140 batches each, same
estimator): σ(f_NL) = 4.47, 7.46, 13.92 at densities 1.0, 0.1, 0.03, and `f_NL` −3.1, +4.3, +16.9
(pulls −0.70, +0.58, +1.22) on the shared truth field (`seed` 77). Its joint twin, with the current κ
model, is `scan/closure_d0p10_joint.yaml` (§8).

At density 0.03 the gain is **17 % ± 3 %** and does not move with where the chains are cut (0.84,
0.87, 0.85, 0.83 at 40, 60, 80, 103 batches); both runs converge (R-hat ≤ 1.014, energy variance
0.96 and 1.03 × target, step size 243 vs 232, i.e. κ costs 5 %). The two-point Fisher, fed the
measured galaxy-only chain of the same density, predicts 3.4 %: **the field level extracts about five
times more from κ than the two-point proxy**. At density 1.0 the ratio is 0.910 ± 0.034 at 124
matched batches (0.92, 0.94 at 60, 90), i.e. a **9 % ± 3 %** gain against a 2.8 % Fisher; both runs
converge (R-hat ≤ 1.001 on `f_NL`), `f_NL` −3.1 ± 4.4 (galaxy-only) and −2.5 ± 4.0 (joint), truth 0.
The joint run stops at 124 of its 140 batches. Both d0.03 runs share the truth field and
sit at f_NL ≈ +17 (pull +1.2 and +1.6), the same realisation read twice; `b1` is 1.2σ low in both.
Field reconstruction at d0.03: `r` ≈ 0.5 inside the LRG shell in both runs (sparser galaxies than
the 0.96 of density 1.0); κ adds `r` ≈ 0.2 in the inner cone (200–600 Mpc/h) and 0.05–0.09 beyond
the shell, where the galaxy-only run has none.

**Where the gain comes from** (`compare_runs.py`, second half of each run;
`figures/results/closure_d0p03_gxyVSjoint_fNL_run_20260923_054507_58787249_run_20260923_054540_58787260.png`,
`figures/results/closure_d1p00_gxyVSjoint_fNL_run_20260922_071921_58748954_run_20260923_024316_58785136.png`).
Split the variance of `f_NL` into the part at fixed biases, `σ_cond² = 1/(Σ⁻¹)_{f_NL f_NL}`, and the
part the bias degeneracies add, `σ² − σ_cond²`:

| density | run | σ(f_NL) | σ at fixed biases | bias share of the variance | ρ(f_NL, b1) | ρ(f_NL, b∇²) |
|---|---|---|---|---|---|---|
| 0.03 | galaxy-only | 13.92 | 9.26 | 0.56 | −0.68 | −0.41 |
| 0.03 | joint | 11.57 | 8.55 | 0.45 | −0.59 | −0.33 |
| 1.0 | galaxy-only | 4.47 | 3.38 | 0.43 | −0.57 | −0.34 |
| 1.0 | joint | 4.02 | 3.16 | 0.38 | −0.52 | −0.29 |

At density 0.03, κ removes 60 of the 194 units of `f_NL` variance; 47 of them (79 %) come out of the
bias part and 13 (21 %) out of the part at fixed biases: most of what κ removes is the
**degeneracy of `f_NL` with `b1` and `b∇²`**, whose correlations with `f_NL` drop from −0.680 ± 0.012
to −0.593 ± 0.017 (`b1`) and from −0.413 ± 0.012 to −0.330 ± 0.030 (`b∇²`); errors are the spread
of the four chains over √4. At density 1.0 the split is 62 % / 38 % of a smaller reduction
(20.0 → 16.2); ρ(f_NL, b1) −0.572 ± 0.013 → −0.523 ± 0.019, ρ(f_NL, b∇²) −0.342 ± 0.008 →
−0.288 ± 0.035.
`b2`, `bs2` and `bnpar` are uncorrelated with `f_NL` (|ρ| ≤ 0.09) and stay so. The marginal widths of
the biases barely move (σ(b1) 0.047 → 0.044 at 0.03). The d0.03 joint run has 103 batches, its
galaxy-only twin 140.

What "five times" compares: the measured field-level gain against the Limber increment of §6, a
two-point proxy (Limber, κ band ℓ ≤ 64, ten galaxy shells, linear P, marginal over `b1`, `b∇²` only).
It is a statement about that proxy, not about every two-point analysis, and it rests on one density
until 0.1 and 0.2 are in. A κ-only run cannot show it: κ alone sees `f_NL` only through the matter φ²
channel; the gain is the κ–galaxy combination, which is exactly what the scan figure plots against
the Fisher curve.

**The x-axis is the density, not `σ_gxy/σ_κ`.** κ's information on `f_NL` passes through its
cross-correlation with the galaxies (κ measures δ_m, the galaxies `b(k)δ_m`; comparing the two
measures `b(k)` free of cosmic variance), so it degrades with the galaxy shot noise even though the
κ map does not change: `density_scan.py --noise_table run_20260922_071921_58748954 --config
archive/scan/closure_d1p00_joint_chimin350.yaml --density d` (biases of the density-1.0 chain, `chi_min` 350
geometry) gives `σ_κ` = 18.9, 40.6, 53.7, 53.1 at densities 1.0, 0.2, 0.1, 0.03. There is no fixed `σ_κ` to divide by, and the independent-probe
line `1 − [1 + (σ_gxy/σ_κ)²]^(−1/2)` does not apply. The figure plots the measured paired gain
against `n̄/n̄_LRG`, with the two-point Fisher gain as the reference curve, computed on the 3-D galaxy
Fisher so that it exists at every density.

**The density-scan figure** (`scripts/density_scan.py`, `configs/inference/scan/density_scan_runs.yaml`,
`figures/results/density_scan.png`; §6, Fisher forecasts; joints at `chi_min` 700 only, curves on the
geometry of `archive/scan/closure_d1p00_joint_chimin700.yaml` with the closure truth biases). The 3-D galaxy
Fisher gives σ(f_NL) = 4.31, 7.86, 15.71 at densities 1.0, 0.1, 0.03 (biases of each chain) against
4.47, 7.46, 13.92 measured by the galaxy-only runs: within 4 % at density 1, and the field level
6 % and 11 % below the forecast at 0.1 and 0.03. That agreement is conditional on `k_min` (×0.71 and
×1.58 at density 1 for `k_min` halved and doubled, §6). The Limber gain on the 3-D galaxy Fisher and
on the galaxy Fisher measured from the chain: 2.66 % and 2.81 % at density 1, 1.07 % and 0.98 % at
0.1, 4.12 % and 3.40 % at 0.03 (biases of each chain) — within 0.7 point, so the curve depends
little on how the galaxy information is modelled. Measured so far at `chi_min` 700: density 0.03, paired ratio
0.854 ± 0.016 at 140 batches (gain 14.6 % ± 1.6 %).

On an earlier, pre-refactor box-2000 closure (galaxies at z ≲ 0.37) the same kind of
proxy predicted under 2 % where the field level measured 18 %: the curve bounds the design, the
runs decide.

**Roadmap beyond the first paper.**
1. **Coarse-resolution pair** — the degeneracy-breaking mechanism as a corner plot: coarsening the
   cell narrows the k-band until constant `b1`, `+k²` from `b∇²` and `−1/k²` from the PNG response
   stop being separable. Not planned for the first paper.
2. Compare to a standard power-spectrum analysis (with A. de Mattia).
3. **HalfDome** (`/global/cfs/cdirs/cmb/gsharing/halfdome`, arXiv:2407.17462) as a second external
   simulation, and the only one on hand with a **matched f_NL pair**: 11 Gaussian realisations plus
   `seed_100_fnl_20` (local f_NL = 20, sharing seed 100 with a Gaussian twin), 6144³ particles in a
   3.75 Gpc/h box, with exact linear ICs (`lineark`, `Nmesh = 12288`) for both. Cost: the release
   ships *"downsampled particles, halo catalogs, mass sheets, velocity sheets"* — there is **no
   CMB-source κ map**; the ready-made `lensing/` planes stop at z_s = 2.5 and cover only the Gaussian
   seeds. Using it requires a `bigfile` reader, Born-integrating κ_CMB from the `usmesh` mass sheets
   (HEALPix nside 8192, 80 shells over a = 0.2–1), and an HOD or mass cut on the RFOF lightcone halos.
4. Application to real DESI-LRG × Planck/ACT κ data.
5. *Moved into the first paper:* κ on `Omega_m` and `sigma8` (§7.10, §8 step 5).

### 7.4 κ-only on Abacus HUGE: validation of the κ model — measured

The κ model on its own against the real Abacus κ map (`kappa_00045`, full sky), with no galaxies:
`galaxies_enabled: false`, `Omega_m`, `sigma8` and `f_NL` fixed at the Abacus values, ACT DR6
`N_ℓ` ×1, `nside 32`, `shell_weights: linear`, `proj_oversamp: 2`. Only the field is sampled. The
two runs differ only in `chi_min`. They, and the ×0.1 run below, use a uniform solid angle
per pixel in the projector and read the κ map through `ud_grade` (§7.9); the two runs at 700 are
to be rerun (§8 step 5d):

| `chi_min` | run | config | batches (second half used) |
|---|---|---|---|
| 350 | `run_20260924_071449_58822478` | `archive/abacus/abacus_kappaonly_Nl1p0_fnlfixed_chimin350.yaml` | 65 |
| 700 | `run_20260924_081242_58822478` | `archive/abacus/abacus_kappaonly_Nl1p0_fnlfixed_chimin700.yaml` | 140 |

**What a correct result looks like.** The model generates the κ of the shells, of power `S_m`; the
likelihood treats `N_a` = noise + line of sight (`C_ℓ^{low-z}` below `chi_min`, `C_ℓ^{high-z}`
beyond the box) as noise. For a correct Gaussian posterior, `W = S_m/(S_m+N_a)`; two independent
samples then agree at coherence `W`. The Abacus map is `S_t = S_m + L`: it also holds the
line-of-sight matter `L`, which the model does not generate but which the data show, so a sample
reaches `W·√(S_t/S_m)` with the truth — above the agreement between chains. In closure `L = 0` in
the truth and both expectations are `S/(S+N_a)`. The derivation: the posterior mean is `m̂ = W·d`
with `d = t + n` and `t = m + u`, so `cov(m̂, t) = W·S_t` while `var(m̂) = W²(S_t + N) = W·S_m`
(because `S_t + N = S_m + N_a`); a sample is `m̂` plus an independent fluctuation of variance
`W·N_a`, so it carries `S_m`, two samples share `var(m̂)`, and a sample shares `W·S_t` with the
truth. The same algebra gives the expected error power `S_t + S_m − 2W·S_t` (compared with `N_a`
in the spectrum panel) and the coherence of the posterior mean with the truth, `√(W·S_t/S_m)`, the
limit of the mean-over-chains curve as the number of chains grows. At `N_ℓ` ×1 (S/N ≈ 1.2, `L/S_t` ≈ 0.1) both are
≈ 0.5–0.6, error/assumed covariance ≈ 1, and the error rms ≈ 0.95 × the truth rms: a single-sample
error map as large as the κ map is the expected outcome at this noise level, not a failure. The
coherence panel of `kappa_reconstruction.png` draws both expectations and is the test.

| | `chi_min` 350 | `chi_min` 700 | correct model |
|---|---|---|---|
| rms model / truth (σ_hp units) | 1.24 / 1.19 | 1.14 / 1.19 | ≈ 1 |
| model C_ℓ vs truth at ℓ ≳ 35 | above | equal | equal |
| error / assumed covariance, ℓ 6–11 / 11–20 / 20–36 / 36–65 | 1.14 / 1.04 / 1.27 / 1.18 | 0.81 / 0.87 / 1.09 / 1.05 | ≈ 2W ≈ 1.0–1.1 |
| coherence sample–truth minus `S_t/(S_t+N_a)`, ℓ < 20 / 20–48 / ≥ 48 | +0.01 / +0.01 / +0.02 | +0.03 / +0.02 / +0.01 | ≈ +0.02–0.03 (the √(S_t/S_m) factor) |
| coherence between chains vs sample–truth, ℓ ≥ 48 | 0.58 vs 0.54 | 0.52 vs 0.52 | between ≤ sample–truth |
| IC, 0–600 Mpc/h: rms rec/true, r(chains), r(truth) | 1.5, 0.04–0.40, ≈ 0 | 0.96–1.16, ≈ 0, ≈ 0 | 1, 0, small |

**At `chi_min` 700 the κ model passes.** The model κ carries the Abacus power, every chain's sample
has the expected coherence with the Abacus map in every ℓ range, the chains agree with each other
no more than with the truth, and the model error sits at the level a perfect model gives.
Nothing in the band says the projector or the line-of-sight covariance is wrong. **At `chi_min` 350
it fails in the way §2.6 predicts:** excess model power at the top of the band, model error
above the covariance at ℓ ≥ 20, chains agreeing with each other more than with the truth at ℓ ≥ 48
(a shared component absent from the data), and a spurious near-observer structure in the
reconstructed field (1.5× the true rms at 0–600 Mpc/h, correlated between chains, uncorrelated
with the truth). The two runs differ in `chi_min` only, so all four come from the shells between
350 and 700 Mpc/h, where item 8b places the lattice pattern (ℓ ≈ 41–70).

The 3-D field reconstruction from κ alone is weak, as expected from a projection: r(truth) 0.03–0.15
from 800 to 3600 Mpc/h at 100 Mpc/h smoothing, `T(k)` ≈ 1 at every k.

**Stress test at `N_ℓ` ×0.1** (`run_20260925_020623_58859550`,
`archive/abacus/abacus_kappaonly_Nl0p1_fnlfixed_chimin700.yaml`, 140 batches, final state). With the noise
10× lower, an error of the κ model that hides under the noise at ×1 would pull the coherence below
its expectation. It does not: per-sample coherence with the Abacus map is 0.85–0.90 for ℓ ≥ 15 (0.93–0.96
for the mean of the four chains), the single-sample error rms is 0.50 × the truth rms (error/signal
0.25–0.27 for ℓ ≥ 11, error/assumed covariance 0.95–1.26 against `(S_t + S_m − 2W·S_t)/N_a` ≈ 1.25). No model error is detected down to a tenth of the ACT noise.

Both expectations hold bin by bin. With `S_m = S_t − L` and the covariance as assumed, the
expected coherence with the truth is 0.86–0.89 for ℓ ≥ 19 (mean 0.871), measured 0.86–0.90 (mean
over chains, 0.875); between chains 0.812 expected, 0.820 measured. Every bin from ℓ = 11 is within
0.03 (truth) and 0.04 (between chains); below ℓ = 11 the bins hold a few tens of modes. The coherence with the truth exceeds the agreement
between chains, as it must when the truth holds line-of-sight matter the model does not generate:
the posterior mean follows the data, which contain it. The model power is 0.94 × the truth rms for
the same reason. The line of sight dominates the assumed covariance at this noise level
(`L/S_t` ≈ 0.11–0.17 for ℓ ≥ 19 against a noise of 0.06–0.09), and it is sized right.

**Where the Abacus κ matter starts.** The AbacusLensing maps are the Born integral of the on-the-fly
particle light cone of the simulation (AbacusLensing `README.md`), and the HUGE runs stop at
`FinalRedshift = 0.1` (`abacus.par`, `status.log`): the light cone, hence the κ map, holds no matter
below z = 0.1, χ = 292.6 Mpc/h. The runs above integrate `C_ℓ^{low-z}` from χ = 1, whose 1–293 part
is not in the data: 17 %, 7 %, 2.9 %, 1.3 %, 0.7 % of the Limber κ power in ℓ 2–4, 5–10, 11–20,
21–36, 37–64; removing it from the expectation of this run moves the expected coherence with the
truth from 0.871 to 0.868 (mean over ℓ ≥ 20). `cmb_lensing.chi_low_z_min` (§2.6) now sets this lower
limit for Abacus runs.

No Born shell lies below `chi_min`, so κ does not see the field there. At 0–600 Mpc/h the final
state gives rms rec/true 1.50, 1.09, 0.99, r(truth) 0.39, 0.24, 0.10 and r(chains) −0.10, −0.02,
0.10 in the 200 Mpc/h shells 0–200, 200–400, 400–600 (≈ 137, 960 and 2610 init-grid cells by volume).

**What `chi_min` 700 gives up** (Limber at the fiducial cosmology, the Abacus configuration
`archive/abacus/abacus_joint_Nl1p0_chimin700.yaml`). The matter at 350–700 Mpc/h, modelled at `chi_min` 350,
moves into the covariance at 700: 24 %, 19 %, 12 %, 7 %, 4 % of the κ power (from 293 to 3942 Mpc/h)
in ℓ 2–4, 5–10, 11–20, 21–36, 37–64, i.e. 19 %, 25 %, 21 %, 13 %, 7 % of the ACT DR6 `N_ℓ` (×1)
in the same bands. No Born shell lies below 700 Mpc/h, so κ carries no direct information on the
field there. The inner-cone reconstruction at 250 Mpc/h smoothing does not drop for all that; it
is measured on the Abacus joint at 700 in §7.5.

**Consequence for the closure scan (§7.3).** The closure joint runs of the §7.3 table use `chi_min`
350; the joints of §8 step 5 run at 700. At 350 the
lattice is in the truth and in the model alike: the cross shows in both maps, and the truth C_ℓ
spikes at ℓ > 50 raise the Wiener coherence at ℓ ≥ 48 to 0.68, against 0.51 on Abacus. The
closure κ therefore carries top-of-band power that the Abacus κ does not have. The galaxy-only runs
do not depend on `chi_min`; whether the measured joint gains depend on it is measured by a joint
closure run at `chi_min` 700 paired with the existing galaxy-only run of the same density. Measured
at density 0.03 (`run_20260925_020200_58859522`, `archive/scan/closure_d0p03_joint_chimin700.yaml`, 140
batches, R-hat 1.000, step size 245 vs 232 at 350): at the same 103 batches the paired ratio
(mean of the per-chain ratios, as in §7.3) is 0.859 ± 0.012 against 0.829 ± 0.027 at `chi_min` 350
(0.854 ± 0.016 at 140), a gain of 14 % ± 1 % instead of 17 % ± 3 % (the two joints share the
galaxy-only run, so the errors of the ratios are correlated). The decomposition at `chi_min` 700: σ(f_NL) at fixed biases 9.26 → 8.52, bias share of
the variance 0.56 → 0.48, ρ(f_NL, b1) −0.680 ± 0.012 → −0.629 ± 0.011, ρ(f_NL, b∇²) −0.413 ± 0.012
→ −0.373 ± 0.013.

### 7.5 Abacus joint at `chi_min` 700 — measured

`run_20260927_060952_58951736` (`archive/abacus/abacus_joint_Nl1p0_chimin700.yaml`, 140 batches). Its config
differs from the §7.1 joint `58161527` only in `chi_min` (700) and `chi_low_z_min` (292.6), and
shares `seed` 77 with it and with the galaxy-only `58153868`. It predates the current κ loader
(§7.9): its κ data went through `ud_grade` to nside 32.

**Convergence.** R-hat ≤ 1.004 on every parameter, energy variance 0.87–0.99 × target on the four
chains, step size 83.5 against 85.2 (joint at 350) and 85.9 (galaxy-only).

**Parameters.** Second half: `f_NL` 8.85 ± 5.57, `b1` 1.191 ± 0.019, `b2` 0.683 ± 0.123, `bs2`
−0.474 ± 0.136, `bn2` 78.5 ± 13.6, `bnpar` −44.9 ± 21.5, the same as the joint at 350 and the
galaxy-only run (§7.1). Paired σ(f_NL) ratio joint-700 / galaxy-only (per chain, error = spread of
the four / 2): 0.953 ± 0.049, 0.919 ± 0.026, 0.944 ± 0.016 at 27, 41, 55 matched batches;
joint-700 / joint-350: 1.001 ± 0.026, 0.905 ± 0.011, 0.957 ± 0.013, 0.989 ± 0.048 at 27, 41, 55,
92. The ratio moves with the cut by more than its error, so the four-chain spread understates the
uncertainty; the gain is "a few percent at most", as at 350. The 55-batch galaxy-only run caps it
(§8 step 1a).

**Field reconstruction** (`compare_reconstruction.py`, final states, per chain; joint 700 /
joint 350 / galaxy-only):

| shell χ [Mpc/h] | 250 Mpc/h smoothing | 100 Mpc/h smoothing |
|---|---|---|
| 0–361 | 0.469 ± 0.165 / 0.306 ± 0.225 / −0.319 ± 0.276 | 0.164 ± 0.180 / 0.084 ± 0.210 / −0.018 ± 0.211 |
| 361–722 | **0.395 ± 0.105** / 0.327 ± 0.068 / −0.154 ± 0.175 | 0.021 ± 0.083 / 0.086 ± 0.070 / 0.048 ± 0.064 |
| 722–1083 | 0.254 ± 0.164 / 0.250 ± 0.228 / 0.051 ± 0.176 | 0.131 ± 0.055 / 0.157 ± 0.038 / 0.090 ± 0.076 |
| 1443–1804 | 0.952 ± 0.004 / 0.951 ± 0.003 / 0.950 ± 0.007 | 0.968 / 0.970 / 0.969 |
| 2165–2526 | 0.491 ± 0.024 / 0.509 ± 0.034 / 0.419 ± 0.074 | 0.553 / 0.555 / 0.516 |
| 2526–2887 | 0.115 ± 0.040 / 0.146 ± 0.009 / 0.034 ± 0.079 | 0.082 / 0.079 / −0.001 |
| 2887–3248 | 0.052 ± 0.014 / 0.123 ± 0.025 / 0.000 ± 0.011 | 0.059 / 0.066 / 0.004 |

With no Born shell below 700 Mpc/h, the inner cone at 250 Mpc/h smoothing is as well reconstructed
as at 350; why is not established. The rms residual against the true IC (250 Mpc/h) is 0.002847 /
0.002806 / 0.003003 inside the galaxy shell and 0.005885 / 0.005835 / 0.006032 outside.

**κ posterior** (`kappa_batch_*.npz`, second half, 70 batches × 4 chains, observable at nside 32).
Coherence in ℓ bins 2–11, 12–23, 24–35, 36–47, 48–55, 56–64: sample vs truth 0.574, 0.666, 0.684,
0.682, 0.613, 0.579; between chains 0.564, 0.658, 0.675, 0.676, 0.628, 0.620; posterior mean vs
truth 0.76, 0.82, 0.83, 0.83, 0.77, 0.73; model/truth power 0.75, 0.90, 0.89, 0.90, 1.06, 0.97.
Below ℓ = 48 the chains agree with each other as they agree with the truth; above, they agree
with each other more (0.620 vs 0.579 at ℓ 56–64); the κ-only run at 700 had 0.52 vs 0.52 there.
The cause is not established. Cutting |b| > 80° changes no coherence by more than
0.003, so it is not the polar caps. The polar-cap excess of the uniform solid angle this run uses (§7.9) is
nonetheless visible in it: the posterior mean of the eight polar pixels (nside 32) is +0.83
to +2.01 σ_hp, where the truth ranges from −4.1 to +3.2 σ_hp (mean −0.3), with a posterior std of
0.64–0.72 σ_hp.

### 7.6 κ from the Abacus ICs through the forward model — measured

`scripts/validate_kappa_from_ic.py` (§6), `archive/abacus/abacus_joint_Nl1p0_chimin700.yaml` with galaxies
off and `chi_min` overridden, Abacus cosmology, current κ loader (the AbacusLensing map brought to
nside 64 through the projector's bilinear kernel, so data and model share their window). No
sampling: the HUGE ICs go through the forward model once per `chi_min`. Two resolutions:

```bash
python scripts/validate_kappa_from_ic.py --config configs/inference/archive/abacus/abacus_joint_Nl1p0_chimin700.yaml \
  --chi_min 292.6 350 500 700 --out_dir $SCRATCH/outputs/kappa_from_ic/cell93
XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async python scripts/validate_kappa_from_ic.py \
  --config configs/inference/archive/abacus/abacus_joint_Nl1p0_chimin700.yaml --chi_min 292.6 350 500 700 \
  --cell_size 46.875 --abacus_map $SCRATCH/outputs/kappa_from_ic/cell93/abacus_kappa_nside64.npy \
  --out_dir $SCRATCH/outputs/kappa_from_ic/cell47
```

Figures `figures/spectra_diagnostic/kappa_from_abacus_ic_cell93p75.png` (cell 93.75, the inference
resolution: init 120³, particles 140³) and `figures/spectra_diagnostic/kappa_from_abacus_ic_cell46p875.png`
(cell 46.875: init 240³, particles 280³), made with the code at the tag `pre-shell-cut` (commands as
run); their spectra files were overwritten by the §7.12 bench and are not kept.
Same band in both, ℓ ≤ 64 (`cmb_lensing.nside` 32). In each ℓ bin, "max" is the coherence of a model
exact on the matter it holds, `√(1 − C^LOS/C^tt)`, and the error beyond LOS is
`(C^err − C^LOS)/N_ℓ`, the model error the likelihood's covariance does not hold.

Cell 93.75, ℓ bins 2–11 / 12–23 / 24–35 / 36–47 / 48–55 / 56–64:

| `chi_min` | coherence | max | transfer `√(C^mm/C^tt)` | error beyond LOS / `N_ℓ` |
|---|---|---|---|---|
| 292.6 | 0.985 / 0.991 / 0.967 / 0.885 / 0.698 / 0.595 | 0.991 / 0.985 / 0.982 / 0.978 / 0.968 / 0.969 | 0.967 / 0.989 / 0.990 / 1.085 / 1.320 / 1.482 | 0.013 / −0.016 / 0.049 / 0.358 / 1.011 / 1.759 |
| 350 | 0.967 / 0.985 / 0.971 / 0.932 / 0.745 / 0.643 | 0.974 / 0.979 / 0.978 / 0.976 / 0.965 / 0.967 | 0.946 / 0.989 / 0.986 / 1.020 / 1.229 / 1.384 | 0.017 / −0.021 / 0.023 / 0.154 / 0.737 / 1.376 |
| 500 | 0.911 / 0.961 / 0.968 / 0.947 / 0.884 / 0.822 | 0.927 / 0.956 / 0.965 / 0.967 / 0.956 / 0.960 | 0.945 / 0.965 / 0.976 / 1.006 / 1.020 / 1.054 | 0.034 / −0.017 / −0.012 / 0.070 / 0.181 / 0.385 |
| 700 | 0.861 / 0.926 / 0.948 / 0.944 / 0.902 / 0.858 | 0.863 / 0.914 / 0.940 / 0.948 / 0.935 / 0.945 | 0.922 / 0.935 / 0.951 / 0.996 / 1.002 / 0.999 | 0.008 / −0.034 / −0.024 / 0.016 / 0.086 / 0.225 |

Cell 46.875:

| `chi_min` | coherence | max | transfer | error beyond LOS / `N_ℓ` |
|---|---|---|---|---|
| 292.6 | 0.986 / 0.994 / 0.995 / 0.995 / 0.988 / 0.985 | 0.991 / 0.985 / 0.982 / 0.978 / 0.968 / 0.969 | 0.965 / 0.991 / 0.993 / 0.991 / 0.993 / 0.986 | 0.011 / −0.026 / −0.044 / −0.056 / −0.049 / −0.041 |
| 350 | 0.967 / 0.987 / 0.991 / 0.994 / 0.988 / 0.987 | 0.974 / 0.979 / 0.978 / 0.976 / 0.965 / 0.967 | 0.945 / 0.989 / 0.991 / 0.988 / 0.991 / 0.985 | 0.016 / −0.026 / −0.044 / −0.060 / −0.054 / −0.048 |
| 500 | 0.913 / 0.963 / 0.978 / 0.986 / 0.982 / 0.985 | 0.927 / 0.956 / 0.965 / 0.967 / 0.956 / 0.960 | 0.941 / 0.963 / 0.974 / 0.981 / 0.981 / 0.981 | 0.031 / −0.022 / −0.044 / −0.062 / −0.061 / −0.061 |
| 700 | 0.862 / 0.927 / 0.956 / 0.968 / 0.965 / 0.973 | 0.863 / 0.914 / 0.940 / 0.948 / 0.935 / 0.945 | 0.917 / 0.937 / 0.947 / 0.971 / 0.967 / 0.965 | 0.004 / −0.035 / −0.052 / −0.063 / −0.069 / −0.069 |

At the inference resolution the model κ carries excess power at the top of the band, incoherent with
the Abacus map, which grows as `chi_min` decreases: 1.76 × `N_ℓ` of error outside the covariance at
ℓ 56–64 for `chi_min` 292.6, 0.39 at 500, and still 0.23 at 700 (0.09 at ℓ 48–55). Below ℓ ≈ 36
every `chi_min` reaches its maximum coherence, and the lowest reaches the highest (0.985 at ℓ 2–11
for 292.6 against 0.861 for 700). With the cell halved, same band, the error beyond LOS is at most
0.03 × `N_ℓ` in every bin for every `chi_min` down to 292.6, and negative from ℓ = 12 on: the model
error is below the line-of-sight term the covariance assumes.

**Why the spectra agree while the coherence drops.** The Abacus map holds the line of sight the
model does not (below `chi_min`, beyond the box): 10–13 % of its power at ℓ ≥ 36, 26 % at ℓ 2–11 for
`chi_min` 700. A model exact on its own range would sit *below* the Abacus spectrum by that much —
the dotted curves of the spectrum panel, `C^tt − C^LOS` (`validate_kappa_from_ic.py --replot` redraws
the figures from the npz). Writing the model as `a·t_in + e`, with `t_in` the Abacus matter in the
model's range (power `C^tt − C^LOS`) and `e` independent of the map (`a = C^tm/(C^tt − C^LOS)`,
`E = C^mm − a²(C^tt − C^LOS)`), cell 93.75, ℓ bins 2–11 /
12–23 / 24–35 / 36–47 / 48–55 / 56–64:

| `chi_min` | `a` | `E / (C^tt − C^LOS)` |
|---|---|---|
| 700 | 1.066 / 1.037 / 1.020 / 1.047 / 1.034 / 0.961 | +0.005 / −0.028 / −0.016 / +0.008 / +0.080 / +0.195 |
| 350 | 0.963 / 1.018 / 1.001 / 0.998 / 0.983 / 0.951 | +0.015 / −0.014 / +0.014 / +0.097 / +0.655 / +1.144 |

The independent power is the particle discreteness (§2.6): nothing below ℓ ≈ 48, then 8 % and
20 % of the target at 700 against 66 % and 114 % at 350. At 700 it roughly offsets the missing line of
sight at ℓ ≥ 36 (model/Abacus total 0.99–1.00), which is why the spectrum panel alone shows no
problem. The amplitude `a` = 1.02–1.07 at 700 below ℓ = 48, against ≈ 1.00 at 350, rests on the
Limber estimate of the 293–700 Mpc/h line of sight in `C^LOS`. A check: at 350 the model follows
the Abacus matter (a ≈ 1), so `C^tm(350) − C^tm(700)` measures the power of the 350–700 Mpc/h
Abacus matter the 350 model holds. Against the Limber power of that slice in `C^LOS`, in the same
bins: 0.120 vs 0.204, 0.108 vs 0.122, 0.056 vs 0.072 × `C^tt` (Limber/measured 1.70, 1.13, 1.30),
then 5.7 and 5.0 at ℓ 36–55. At ℓ 2–11 Limber overestimates the slice, which lowers the target
and accounts for the apparent amplitude excess there; from ℓ ≈ 12 the check is confounded by the
model's resolution in the near shells (πχ/ℓ below the cell, the resolution cut above), which
removes coherent power from the 350 model itself, so the 2–5 % between ℓ 12 and 47 stays
unattributed. The same Limber term is the line-of-sight covariance of the likelihood: too large at
low ℓ, it errs on the conservative side.

**What the likelihood expects against what the Abacus map holds.** The likelihood takes the data
power to be the model's plus `C^LOS` plus `N_ℓ`; the Abacus map holds `C^tt` plus the noise. From the
cell-93.75 npz computed with the per-pixel normalisation of §2.6 (`poles_fixed/kappa_from_ic.npz`),
`(C^mm + C^LOS) / C^tt` in the same ℓ bins: 1.105 / 1.039 / 1.020 / 1.094 / 1.131 / 1.106 at
`chi_min` 700 and 0.945 / 1.021 / 1.015 / 1.088 / 1.579 / 1.981 at 350. At 700 the discreteness
fills the place of the missing line of sight in the model's own power (transfer ≈ 1 at ℓ ≥ 36), and
the covariance adds that line of sight on top, so the top two bins expect 11–13 % more power than the
map holds (its effect on a free `sigma8`: §7.10). The 4–10 % below
ℓ = 12 and at ℓ 36–47 comes with a line-of-sight term that the slice check above finds too large at
low ℓ; it is not attributed further.

### 7.7 Stiffness of the κ log-density vs `chi_min` and shell weights — measured

`scripts/kappa_stiffness.py` (§6), `archive/abacus/abacus_joint_Nl1p0_chimin700.yaml` at cell 93.75, galaxies
off, scalars fixed, one closure draw (`seed` 77), 60 power iterations of exact reverse-over-reverse
Hessian-vector products; `figures/conditioning/kappa_stiffness.json`.
λ_max of −log p with respect to `init_mesh_`, and the fraction of the variance of the top
eigenvector (real-space initial field) within 350 / 700 / 1100 Mpc/h of the observer (volume
fractions 0.0004 / 0.0034 / 0.0132):

| `shell_weights` | `chi_min` | λ_max | variance < 350 / < 700 / < 1100 |
|---|---|---|---|
| nearest | 0 | 17 311 | 0.98 / 1.00 / 1.00 |
| linear | 0 | 471.7 | 0.92 / 0.99 / 1.00 |
| nearest | 350 | 15.67 | 0.07 / 0.84 / 0.89 |
| linear | 350 | 10.64 | 0.02 / 0.79 / 0.85 |
| linear | 700 | 2.38 (still rising by 0.001 per iteration) | 0.00 / 0.00 / 0.11 |
| linear, κ noise × 10⁸ (prior alone) | 700 | 1.00 | 0.0004 / 0.0034 / 0.0131 |

The tents divide λ_max by 37 at `chi_min` 0; the stiffest direction then still lies in the first
350 Mpc/h and λ_max is 200 × its value at 700. κ alone, one draw, cell 93.75 only.

**Tuned step sizes of the runs** (`warmup_config.yaml`, median over chains; the MCLMC step size
scales as λ_max^(−1/2), and as `desired_energy_var`^(1/6) since Var[E] ∝ ε⁶). Without `chi_min` and
the tents — closure, box 5000, cell 39.1, `nside` 128, observer at the centre, pre-counts code — the galaxy-only and joint runs differ only in `cmb_lensing.enabled` and
`desired_energy_var` (5·10⁻⁷ and 5·10⁻⁸): step size 138 (`run_20260706_030446_55565548`) against
3.20 (`run_20260707_001039_55621246`), 4.7 at the galaxy-only target, i.e. κ divides the step size
by ≈ 30. With `chi_min` 350 and the tents, on the Abacus HUGE configuration, same target 10⁻⁷: 85.9
(galaxy-only `58153868`) against 85.2 (joint `58161527`), and 83.5 at `chi_min` 700 (`58951736`);
in closure 85.3 against 86.9 at density 1.0 (`58748954`, `58785136`), 243 against 232 at 0.03
(`58787249`, `58787260`), 245 at 0.03 and `chi_min` 700 (`58859522`). The two regimes differ in
geometry as well, so the isolated effect of `chi_min` and the tents is the λ_max column above: a
factor 7 300 in λ_max from (nearest, 0) to (linear, 700), 85 in step size.

### 7.8 Spectra of the forward model against theory — measured

No sampling: closure draws of the model and the Abacus maps against Limber.

**Angular spectra** (`scripts/quick_cl_spectra.py --n_realizations 20 --seed 77`, seeds 77–96). Galaxy
map from the particles, divided by `bilinear_weight_norm`; Limber theory with the survey's dN/dχ as
galaxy kernel and the bilinear window of the projection sphere on κ. Bin centres from `bin_cl_log`,
20 bins from ℓ = 2.2 to 59.4; below ℓ ≈ 8 Limber itself is inaccurate for kernels this broad.

At `chi_min` 700 (`configs/inference/archive/validation/closure_chimin700.yaml`, identical to
`archive/scan/closure_d1p00_joint_chimin350.yaml` but for `chi_min`; `figures/spectra_diagnostic/cl_closure_20real_chimin700.png`),
measured/Limber in the 20 bins (| at ℓ = 10 and 48): κκ 1.06 1.00 1.08 0.88 0.86 0.90 0.92 0.95 |
1.06 1.01 0.96 0.94 1.00 1.00 1.05 1.01 1.03 1.08 | 1.11 1.23; gg 1.60 1.21 1.48 1.10 1.02 1.04 1.12
1.07 | 1.15 1.09 1.03 1.00 1.06 1.02 1.05 1.02 1.02 1.02 | 1.01 1.04; κg 1.26 1.01 1.20 0.93 0.90 0.93
0.93 0.94 | 1.08 1.02 0.97 0.94 1.02 0.99 1.03 1.00 1.00 1.02 | 1.01 1.05. κκ at ℓ = 51 and 59 is the
particle discreteness (§2.6), 11 % and 23 % in power above the windowed Limber (w = 0.935 and 0.916
there), the same excess as in §7.6. The galaxy kernel matters for gg and κg: galaxies uniform in χ
over the LRG shell instead of the catalogue dN/dχ raise the Limber κg by 10 % at ℓ = 5–10, 5–8 % at
ℓ = 10–25 and 2 % at ℓ = 50–60, and the Limber gg by 20–25 % at ℓ < 10 (computed with the catalogue
n(z)); a theory with the uniform kernel reads as a κg deficit of that size. The bilinear
normalisation moves κg by ≤ 0.01 (same realisations with it set to one).

At `chi_min` 350 (`archive/scan/closure_d1p00_joint_chimin350.yaml`, `figures/spectra_diagnostic/cl_closure_20real_chimin350.png`;
this figure's theory has no window on κ and galaxies uniform in χ): κκ 1.38 and 1.79 at ℓ = 51 and
59, i.e. 0.45 and 0.8 × the ACT DR6 `N_ℓ` — the lattice of the 350–700 Mpc/h shells (§2.6); gg
identical to 700 bin by bin; κg(350) − κg(700) with that same theory +0.18 +0.06 +0.19 +0.06 +0.10
+0.09 +0.08 +0.06 (ℓ < 10), +0.00 to +0.07 above. The kernel cancels in that difference, which stays
unexplained; excluded by computation are the lattice (an undisplaced-lattice toy gives at most ±3 %
of the Limber κg) and the correlation Limber drops (the exact linear cross of the 350–700 κ window
with the galaxies: −0.022 of the Limber κg at ℓ = 2.2, below 0.01 from ℓ = 3).

**The projector to ℓ = 256** (`configs/inference/archive/validation/closure_nside128_cell47_chimin1100.yaml`:
identical to `archive/scan/closure_d1p00_joint_chimin350.yaml` but for cell 46.875, nside 128 and `chi_min` 1100, run
with `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`; `figures/spectra_diagnostic/cl_closure_20real_nside128_cell47_chimin1100.png`).
`chi_min` 1100 puts the lattice of the shells (particle spacing 7500/280 = 26.8 Mpc/h) above
ℓ = 256; it is a setting of this diagnostic only. 29 bins from ℓ = 2.2 to 237.3; measured/Limber
(| at ℓ = 11.4, 70.6 and 175.2): κκ 1.09 0.82 0.98 1.06 1.09 1.15 0.97 1.12 | 1.04 1.04 1.02 1.01 0.95
1.00 0.99 0.99 1.00 0.99 0.97 1.01 | 0.99 0.99 0.99 0.99 1.04 1.05 | 1.10 1.26 1.62; gg 1.90 1.22 1.19
1.30 1.16 1.19 1.06 1.25 | 1.12 1.09 1.06 1.05 0.99 1.01 1.00 1.00 1.00 0.98 0.95 0.97 | 0.95 0.93 0.90
0.87 0.85 0.82 | 0.79 0.78 0.82; κg 1.35 0.90 1.06 1.18 1.11 1.16 1.00 1.17 | 1.07 1.07 1.04 1.04 0.97
1.00 0.99 0.99 1.00 0.99 0.97 0.99 | 0.97 0.96 0.94 0.91 0.92 0.90 | 0.89 0.91 0.97. κg is 0.97–1.07 for
ℓ = 11–61, against 0.82–0.91 with a theory with galaxies uniform in χ and no κ radial window: the
deficit is that theory's, as at nside 32. κκ is 0.95–1.04 for ℓ = 11–61 and 0.99–1.05 for ℓ = 71–151;
at ℓ = 175 / 204 / 237 it is 1.10 / 1.26 / 1.62 against the Limber curve windowed by the bilinear
kernel at nside 256 (`bilinear_window`: w² = 0.907 / 0.877 / 0.843 there), i.e. the ratio to the
unwindowed curve, 1.00 / 1.10 / 1.35, divided by w²: the particle discreteness, as at the top of the
nside-32 band. gg falls from 0.95 at ℓ = 71 to 0.78–0.82 at ℓ = 175–237, and κg to 0.89–0.92 over
ℓ = 111–204; the cause of this high-ℓ galaxy-side loss is not measured. It lies outside the
likelihood band (ℓ ≤ 64), and the likelihood uses the 3-D galaxy counts, not this HEALPix map.

**Abacus** (`configs/inference/archive/abacus/abacus_joint_Nl1p0_chimin700.yaml`, on a compute node — the
10.8 GB κ map exceeds the login-node memory; `figures/spectra_diagnostic/cl_abacus_huge_chimin700.png`).
One realisation (the simulation), same bins. Measured/Limber: κκ 0.25 0.77 1.25 1.40 0.87 0.94 0.80
0.87 | 1.05 0.87 0.72 1.05 0.91 0.84 0.90 1.00 1.02 1.00 | 0.81 1.01; gg 2.18 1.86 1.61 2.67 1.67 1.21
1.28 1.06 | 0.81 0.95 0.88 1.13 1.17 0.89 0.91 0.97 1.04 0.91 | 0.94 1.05; κg 0.42 0.50 1.93 2.86 0.35
0.35 0.53 0.98 | 1.06 0.68 0.74 1.26 1.00 0.69 0.85 1.05 1.14 0.99 | 0.84 1.14. The Abacus κκ follows the
windowed Limber up to ℓ = 59 (1.00, 1.02, 0.81, 1.01 in the last four bins) where the model sits
11–23 % above: the window is right and the model excess is its discreteness. Simple mean of κg over
ℓ ≥ 10: 0.95 (Abacus, one realisation, 0.68–1.26 per bin) against 1.01 (closure model).

**Where the lattice lands** (§2.6). An undisplaced-lattice toy (numpy, the HUGE geometry, bilinear
at nside 64, ℓ ≤ 64 kept; its spurious C_ℓ over the Abacus κ C_ℓ, a worst case since the LPT
displacements damp it) splits the discreteness by distance: the shells at 350–550 Mpc/h alone give
0.75 / 1.57 / 1.18 at ℓ 45–55 / 55–60 / 60–65; everything beyond 700 gives 0.19 / 0.22 / 0.41, most
of it from 1500–3750. Moving the observer off the lattice node leaves the near shells unchanged and
halves the far part. Without galaxies the discreteness drags a free `f_NL`: the κ-only Abacus run with
`f_NL` free and `chi_min` 350 (`run_20260924_060836_58822478`, `archive/abacus/abacus_kappaonly_Nl1p0_chimin350.yaml`)
parks its chains at |f_NL| ~ 4000–9000, where `f_NL` bends the model spectrum down at ℓ 36–65.

**Galaxy power spectrum** (`scripts/quick_pk_spectra.py`). With the config biases (`b1` 1.15, the
others 0): Abacus LRG against model `P_gg` within ±5 % for k ≳ 0.015 h/Mpc including the corner
modes. With the posterior means of the Eisenstein–Hu galaxy-only reference (`--bias_run $SCRATCH/outputs/run_20260910_033019_58153868`, a run directory path:
`b1` 1.19, `b2` 0.71, `b_{s²}` −0.49, `b∇²` 77.3, `b∇∥` −46.4; `figures/spectra_diagnostic/pk_abacus_vs_model.png`):
Abacus/model 1.008 at k = 0.015–0.025, 1.054 at 0.025–0.0335, 1.083 at 0.0335–0.042 and 1.130 at
0.042–0.053 h/Mpc (corner modes), an excess of 0.17, 0.91, 1.10 and 1.35 × `P_shot` (3087 (Mpc/h)³);
below k = 0.015 the ratio is 0.86–1.37 at the few-mode points. The difference between the two bias
sets is `b∇²` (`--set_bias`): with `bn2=0` the ratio is 0.99 above 0.5 k_Nyq and 0.96 in the corner
modes, with `bnpar=0` 1.03 and 1.07 (all biases fitted: 1.04 and 1.10). The field level fits
`b∇²` = 77 ± 14, whose −b∇² k² lowers the model spectrum by 5–13 % at k > 0.025 h/Mpc, where the
Abacus spectrum does not want it; the excess it leaves is ≈ 1 `P_shot`, roughly flat in k. Whether
the field level sets `b∇²` from the phases and treats the missing power as noise, or the LRG
stochasticity exceeds Poisson, is not established; a galaxy-only reference run with
`gxy_stoch_noise: true` would tell (s_e and `b∇²`). In closure the matter `P(k)` is 0.98 × the
lightcone-averaged linear theory, flat in k (`figures/spectra_diagnostic/pk_closure_matter.png`).
At the posterior means of the `lin_pk_table` reference (`--bias_run
$SCRATCH/outputs/run_20261002_021703_59194575`, 136 batches, second half: `b1` 1.149, `b2` 0.662,
`b_{s²}` −0.469, `b∇²` 56.2, `b∇∥` −49.5; `figures/spectra_diagnostic/pk_abacus_vs_model_lin_pk_table.png`),
mean Abacus/model 1.035 above 0.5 k_Nyq and 1.100 in the corner modes, against 1.04 and 1.10 with
the Eisenstein–Hu reference: the lower `b1` and `b∇²` leave the high-k excess unchanged, so it does not
come from the shape of the linear spectrum.
Maps: `scripts/plot_2D_maps.py` on the Abacus config, `figures/maps/maps2d_abacus_huge_chimin700.png`
(observed and noiseless AbacusLensing κ, projected LRG counts).

### 7.9 Polar caps of the bilinear projector — measured

The per-pixel normalisation of §2.6 measured on the Abacus ICs through the forward model
(`scripts/validate_kappa_from_ic.py --maps`, cell 93.75, `archive/abacus/abacus_joint_Nl1p0_chimin700.yaml`;
spectra in `figures/maps/kappa_from_abacus_ic_poles_cell93p75_chimin700.npz`, model maps in the `.npy` of
the same name; maps
`figures/maps/kappa_from_abacus_ic_poles_cell93p75_chimin700.png` with `bilinear_weight_norm`,
`figures/maps/kappa_from_abacus_ic_poles_cell93p75_chimin700_before_fix.png` with a uniform solid
angle per pixel). The poles lie on the z axis, where two lattice planes cross (§2.6), so the
reference is the other four coordinate axes, equivalent for the lattice but where the bilinear
weights are normal. Model − Abacus at ℓ ≤ 64, interpolated at each axis:

| `chi_min` | ±x, ±y | poles N / S, uniform solid angle | poles N / S, `bilinear_weight_norm` |
|---|---|---|---|
| 350 | +0.020, +0.016, +0.012, +0.011 | +0.046 / +0.037 | +0.013 / +0.005 |
| 700 | −0.001, −0.003, −0.004, −0.005 | +0.024 / +0.021 | −0.004 / −0.007 |

(rms over random directions 0.004 at 350, 0.0025 at 700). The normalisation leaves the four
equatorial axes unchanged to 2·10⁻⁴ and brings the poles into their range: what remains at the poles
is the lattice, common to all six axes. With the uniform solid angle, at the Abacus configuration
(Σ_i d_r W(χ_i) = 0.549 over the shells), uniform matter gives κ = 0.091 in the eight polar-cap
pixels of the projection sphere and up to 0.020 in the observable (ℓ ≤ 64 at nside 32), against an
Abacus κ rms of 0.0093 and a noise σ_hp of 0.0054; 1–3 % of the κ C_ℓ at ℓ = 10–60. Model −
AbacusLensing in the eight polar-cap pixels, band-limited to ℓ ≤ 64: +0.020 to +0.025 at `chi_min`
700 (mean +0.022, 8.2 × the rms of the difference elsewhere, 4 × σ_hp) and +0.035 to +0.047 at 350
(mean +0.042, 9.6 ×); all sixteen values positive. On the projection sphere, before band-limiting,
the mean is +0.104 at 700 against the 0.091 expected, with a difference rms of 0.055 elsewhere. The
runs of §7.4–7.5 use the uniform solid angle: the posterior mean of the Abacus joint at 700 (§7.5) has
all eight polar pixels at +0.8 to +2.0 σ_hp, harmonically negligible there (a polar cut moves the
coherences by ≤ 0.003). Those runs also read the Abacus map through `ud_grade` to nside 32 (§4), whose
window is 0.998, 0.98 and 0.94 × the projector's at ℓ = 50, 60 and 64 in amplitude.

### 7.10 Fisher forecasts — computed

**`f_NL` from κ at two-point level.** The Limber increment of `desi_cmb_fli.fisher`
(`kappa_fisher_increment` on the 3-D galaxy Fisher, Abacus geometry, closure truth biases) is 2.84 %
at density 1 and 4.15 % at 0.03 for every `chi_min` from 292.6 to 700 (§2.6 for why it does not
depend on `chi_min`). The field-level gains at density 0.03, 17 % ± 3 % at 350 and 14 % ± 1 % at 700
(§7.4), are within their errors of each other.

**`Omega_m` and `sigma8`** (`python scripts/fisher_cosmo.py --densities 1.0 0.1 0.03`, §6: closure
config at the Abacus geometry, `scan/closure_d1p00_joint.yaml`, its truth biases and priors, `f_NL`
fixed, the current κ model — shells from 292.6, per-shell cut at π/62.5, bilinear window of the
projection sphere on the κ signal and the covariance term —, ACT DR6 ×1;
`figures/fisher_diagnostic/fisher_cosmo.json`). Marginal σ of `Omega_m` / `sigma8`: galaxies
0.0183 / 0.0503, κ alone 0.0608 / 0.0461, joint 0.0131 / 0.0188 at density 1, a κ gain of 29 % and
63 %; 31 % and 68 % at 0.1, 31 % and 72 % at 0.03; with the line of sight as signal, κ alone 0.0444 /
0.0273 and the gains 30 % and 68 % at density 1. The two probes' degeneracy directions in the
(`Omega_m`, `sigma8`) plane cross (`figures/fisher_diagnostic/fisher_cosmo_contours.png`). With the
abandoned `chi_min` 700 and no window (`archive/scan/closure_d1p00_joint_chimin700.yaml`,
`fisher_cosmo_chimin700.json`): κ alone 0.0424 / 0.0279, joint 0.0123 / 0.0166, gains 33 % and 67 %
at density 1, 35 % and 73 % at 0.1, 38 % and 78 % at 0.03.

**Resolution** (`fisher_cosmo.py --resolution_scan`, density 1, line of sight as noise; the box
enters as κ's box edge, box/2, the cell as the galaxies' k_max = π/cell and as the per-shell cut at the
init-grid Nyquist; cost ∝ mesh³; `figures/fisher_diagnostic/fisher_cosmo_resolution.json`):
σ(`Omega_m`) / σ(`sigma8`) joint and κ gain on them — 7500 / cell 93.75 (the inference setting, cost
1): 0.0131 / 0.0188, 29 % / 63 %; 7500 / 62.5 (3.4): 0.0071 / 0.0132, 6 % / 38 %; 7500 / 46.875 (8):
0.0040 / 0.0097, 2 % / 23 %; 5000 / 62.5 (1): 0.0070 / 0.0147, 7 % / 31 %; 5000 / 41.67 (3.4):
0.0035 / 0.0091, 2 % / 13 %; 5000 / 31.25 (8): 0.0030 / 0.0061, 1 % / 6 %. A finer cell gives the
galaxies alone far more (σ(`sigma8`) 0.050 → 0.013 at cell 46.875) and leaves κ a smaller share: κ
matters where the galaxies stop at large scales. The galaxy Fisher is linear, with `b1` and `b∇²` only
(no `b2`, `b_{s²}`, `b∇∥`, no non-linearity), so it flatters the galaxies more the higher k_max goes; a
5000 box does not match the published IC grid (the test of §7.6). The same scan with `chi_min` (700 at
cell 93.75, 500 at 62.5, 292.6 below) and no window: `fisher_cosmo_resolution_chimin700.json`.

**Shift from a κ power mismatch** (`python scripts/fisher_cosmo.py --mismatch
figures/maps/kappa_from_abacus_ic_poles_cell93p75_chimin700.npz`, closure config at the Abacus geometry,
the per-ℓ data/model ratio `C^tt / (C^mm + C^LOS)` of the §7.6 test at `chi_min` 700, first order:
F⁻¹ b with b_a = f_sky Σ (2ℓ+1)/2 tr(C⁻¹ ∂_a C C⁻¹ ΔC); `fisher_cosmo_mismatch_kappa_from_ic.json`):
κ alone Δ`Omega_m` −0.024 (−0.57σ), Δ`sigma8` −0.047 (−1.67σ); joint −0.002 (−0.16σ) and −0.026
(−1.56σ). The ratio is that of one realisation: the Abacus map holds a draw of the line of sight
where the model holds its variance, so the per-ℓ ratio scatters (0.27 at ℓ = 2) and the shift is
bias plus that scatter, not a bias alone.

### 7.11 Linear power spectrum — measured

`scripts/compare_linear_power.py` (§6). At the same σ8, Eisenstein–Hu against the CLASS spectrum of
the Abacus ICs (`data/abacus_cosm000_CLASS_power.txt`): 0.966 at k = 0.001, 0.962–0.975 over
k = 0.005–0.03, 0.99–1.00 at 0.04–0.06, 1.03 at 0.1 h/Mpc — 3–4 % low over the likelihood's band
(k ≤ 0.034 h/Mpc, corners to 0.058); `figures/spectra_diagnostic/linear_power_eh_vs_class.png`.

When `Omega_m` moves (`compare_linear_power.py --omega_m_scan`, CAMB 2.0.4 as the reference; CAMB
matches the Abacus CLASS table to 0.9994–1.0026 at σ8 = 1 over k = 0.001–0.3;
`figures/spectra_diagnostic/linear_power_omega_m_accuracy.png` and its `.npz`), max |option/CAMB − 1|
at fixed σ8 over k = 0.001–0.034 h/Mpc (corners to 0.058 in brackets), at `Omega_m` = 0.315 / 0.28 /
0.35 / 0.25 / 0.40: `lin_pk_table` 0 / 1.06 (1.21) / 0.81 (1.00) / 2.24 (2.42) / 1.69 (2.17) %;
Eisenstein–Hu alone 3.89 / 3.38 / 4.32 / 3.00 / 4.84 %; the fiducial shape held fixed 0 / 30 / 36 /
52 / 100 %. Below k = 0.02 the `lin_pk_table` error is a near-constant offset (+0.7 % at 0.25, −0.8 %
at 0.40).

**Effect on the galaxy-only posterior.** The Abacus galaxy-only reference with Eisenstein–Hu
(`run_20260910_033019_58153868`, 55 batches) and with `lin_pk_table`
(`run_20261002_021703_59194575`, 140 batches, `abacus/abacus_gxyonly.yaml`) differ in config only by
`lin_pk_table`; second half of the chains: `f_NL` 8.88 ± 5.79 and 8.92 ± 5.84, `b1` 1.192 ± 0.019 and
1.148 ± 0.020, `b2` 0.708 and 0.661 (± 0.12), `b_{s²}` −0.489 ± 0.153 and −0.469 ± 0.137, `b∇²`
77.3 ± 13.7 and 56.2 ± 13.9, `b∇∥` −46.4 and −49.5 (± 22). `f_NL` does not move; `b1` moves down by
2σ, the direction of a stronger prior spectrum at fixed galaxy power, and `b∇²` by 1.5σ.

---

### 7.12 Per-shell multipole cut — measured

`cmb_lensing.shell_kmax` (§2.6) against no cut, at k = π/62.5 = 0.0503 (init-grid Nyquist at cell
93.75), π/93.75 = 0.0335 (final grid) and, at cell 46.875, π/31.25 = 0.1005. Figures in
`figures/spectra_diagnostic/`. Measured with the bench version of the code, where `chi_min` still
existed and `chi_min` 292.6 (0 in closure) meant the shells starting where the map's matter starts,
today's set-up; the commands are given as run (today `--chi_min` is gone and `--shell_kmax` takes a
list in `validate_kappa_from_ic.py` and a variant field in `kappa_stiffness.py`, §6). This bench
decided the current κ model (§2.6).

**IC test** (§7.6 set-up, `validate_kappa_from_ic.py --chi_min 292.6 700 --shell_kmax K`, outputs
the `.npz` next to each figure in `figures/spectra_diagnostic/`;
figures `kappa_from_abacus_ic_cell93p75_{nocut,kinit,kfinal}.png`,
`kappa_from_abacus_ic_cell46p875_{nocut,kinit}.png`). `(C^mm + C^LOS)/C^tt` in ℓ 2–11 / 12–23 /
24–35 / 36–43 / 44–47 / 48–51 / 52–55 / 56–64:

| cell, cut | `chi_min` 292.6 | `chi_min` 700 |
| --- | --- | --- |
| 93.75, none | 0.95 1.01 1.02 1.15 1.38 1.77 1.84 2.26 | 1.10 1.04 1.02 1.08 1.13 1.11 1.15 1.11 |
| 93.75, 0.0503 | 0.95 1.00 1.02 1.06 1.09 1.14 1.18 1.12 | 1.10 1.04 1.02 1.06 1.09 1.14 1.18 1.12 |
| 93.75, 0.0335 | 0.96 1.02 1.02 1.06 1.08 1.26 1.20 1.13 | 1.10 1.04 1.02 1.05 1.07 1.24 1.19 1.12 |
| 46.875, none | 0.95 1.01 1.02 1.03 1.02 1.05 1.06 1.03 | 1.10 1.04 1.01 1.05 1.02 1.04 1.08 1.04 |
| 46.875, 0.1005 | 0.95 1.01 1.02 1.02 1.01 1.03 1.06 1.04 | as without cut (no shell cut) |

At cell 93.75 the cut at the init-grid Nyquist removes the near-shell discreteness (292.6: top-band
coherence 0.62 → 0.85, the value of a model exact but for its covariance terms; error power equal to
those terms over the band) and makes `chi_min` 292.6 and 700 identical above ℓ ≈ 12; 292.6 is better
below (0.95 against 1.10: no low-z Limber term). The final-grid Nyquist sends resolved structure to the
covariance (coherence 0.73 at the top, 1.25 at ℓ 48–51). The remaining +8–15 % at ℓ 44–64 is common
to every cut at cell 93.75 and falls to +2–6 % at cell 46.875: resolution, from the shells beyond
r_eff = 1272 Mpc/h that the cut leaves. The table's covariance term carries no window; with the
bilinear window w_ℓ² that the code now applies to it (§3.2; the saved spectra of `cell47_kinit` with
`bilinear_window(64, 64)²` on their `C^LOS`) the ratio at cell 46.875 becomes 1.01 1.02 1.04 1.02 in
ℓ 44–47 / 48–51 / 52–55 / 56–64.

**IC test with the current code** (window on the covariance term, shells from 292.6;
`validate_kappa_from_ic.py --config configs/inference/abacus/abacus_joint_Nl1p0.yaml --shell_kmax 0
0.0503 --maps --fig figures/spectra_diagnostic/kappa_from_abacus_ic_cell93p75_maps_cut.png`, the
spectra in the `.npz` next to the figure), cell 93.75, same bins as the
table: with the cut 0.95 0.99 1.01 1.05 1.08 1.11 1.15 1.08, without 0.95 1.01 1.02 1.15 1.37 1.77
1.83 2.25; coherence at ℓ 48–51 / 52–55 / 56–64 0.87 0.87 0.85 with the cut, against 0.89 0.89 0.89
for a model exact but for its covariance term, and 0.69 0.71 0.60 without. The error power beyond
the covariance term, (C^err − w_ℓ² C^LOS)/N_ℓ in ℓ 2–11 / 12–23 / 24–35 / 36–47 / 48–55 / 56–64:
0.01 −0.02 −0.01 0.02 0.05 0.12 with the cut, 0.01 −0.02 0.05 0.36 1.02 1.77 without. In the eight
polar-cap pixels at ℓ ≤ 64, model − Abacus is +0.015 (+3.1 × the rms elsewhere) without the cut and
−0.002 (−0.8 ×) with it. The maps around the poles
(`figures/maps/kappa_from_abacus_ic_poles_cell93p75_kmax{0,0.0503}.png`, ±25°, Abacus, model,
model − Abacus): at full resolution the cut removes the pixel-scale speckle of the near shells and
leaves the lines of the coordinate planes (the meridians of the x = 0 and y = 0 planes and the
diagonals), which come from the far shells it does not cut; at ℓ ≤ 64 the model − Abacus residual
becomes a faint cross along those meridians, and the star centred on the pole of the uncut model
map is gone.

**Closure spectra** (§7.8 set-up, `quick_cl_spectra.py --config configs/inference/archive/validation/closure_chimin700.yaml
--n_realizations 20 --seed 77 --chi_min C --shell_kmax K`; figures
`cl_closure_20real_chimin{0,700}_kmax{0,0.0503}.png`, `cl_closure_10real_cell46p875_chimin0_kmax0.1005.png`
with 10 realisations). κκ measured/Limber, last four of the 20 bins (ℓ ≈ 40–60): `chi_min` 0 without cut
2.72 2.45 3.18 4.39, with cut 1.03 1.06 1.05 1.13; `chi_min` 700 without cut 1.03 1.08 1.11 1.23, with
cut 1.03 1.06 1.05 1.13; cell 46.875, `chi_min` 0, cut 0.1005: 1.00 0.99 0.98 1.01. κg in the last two
bins at `chi_min` 0: 1.23 1.40 without cut, 0.97 1.00 with it (the discreteness correlates with the
galaxy map).

**Stiffness and cost** (§7.7 set-up, `kappa_stiffness.py --variants linear:0 linear:350 linear:700
--shell_kmax K`; `figures/conditioning/kappa_stiffness_kmax{0,0.0503}.json`). λ_max without
cut 486 / 10.9 / 2.43 at `chi_min` 0 / 350 / 700, with the cut 2.61 / 2.67 / 2.40: the cut removes the
near-observer stiffness, its top eigenvector no longer near the observer (variance within 350 / 700 /
1100 Mpc/h at `chi_min` 0: 0.005 / 0.11 / 0.30). A κ-only Hessian-vector product costs 0.057–0.060 s
without cut and 0.093–0.099 s with it whatever `chi_min` (6 to 11 shells cut: the batched transforms
cost about the same), device peak 3.16 → 3.39 GB.

**First-order shift of the remaining mismatch** (`fisher_cosmo.py --mismatch` on the closure config
at the current κ model — shells from 292.6, cut at π/62.5 — with the per-ℓ ratio `C^tt / (C^mm +
C^LOS)` of the cut IC test at 292.6, `figures/spectra_diagnostic/kappa_from_abacus_ic_cell93p75_kinit.npz`,
as a two-column file): κ alone Δ`Omega_m` −0.074 (−1.25σ), Δ`sigma8` −0.082 (−1.85σ); joint +0.001
(+0.05σ) and −0.020 (−1.06σ). With the ratio set to 1 above ℓ = 47: κ alone −0.08σ and −0.25σ, joint
+0.05σ and −0.25σ; with it set to 1 below ℓ = 48: κ alone −1.17σ and −1.60σ, joint 0.00σ and −0.81σ.
The shift comes from the top of the band, the far-shell residual. One realisation, as in §7.10: bias
plus line-of-sight scatter. The ratio used holds the unwindowed covariance term, 2–4 points too high
at ℓ ≥ 44 (above). With the windowed term (the ratio of the IC test with the current code,
`fisher_cosmo.py --mismatch figures/spectra_diagnostic/kappa_from_abacus_ic_cell93p75_maps_cut.npz`;
`figures/fisher_diagnostic/fisher_cosmo_mismatch_kappa_from_abacus_ic_cell93p75_maps_cut*.json`, the
Fisher σ being those of the current κ model, §7.10): κ alone Δ`Omega_m` −0.055 (−0.91σ), Δ`sigma8`
−0.061 (−1.32σ); joint +0.000 (+0.03σ) and −0.014 (−0.72σ). The mismatch at ℓ ≤ 47 alone: κ alone
−0.11σ and −0.23σ, joint +0.03σ and −0.18σ; at ℓ ≥ 48 alone: κ alone −0.80σ and −1.09σ, joint 0.00σ
and −0.54σ.

## 8. Remaining steps before the paper

In order. Runs go in a 4 h interactive `salloc` with a bare `run_inference.py` (`docs/hpc.md`). The κ
model now starts its shells where the map's matter starts and applies the per-shell multipole cut
(§2.6, §7.12); every κ and joint run is made with it.

1. **Galaxy-only runs.**
   (a) `Omega_m`–`sigma8` on Abacus, `abacus/abacus_gxyonly_cosmo.yaml`: is `sigma8` unbiased from the
   galaxies alone? A first attempt (`run_20261002_070145_59203212`) diverged in its first sampling
   batch: the warmup ended at `Omega_m` 0.25, `sigma8` 0.61 with a step size of 133 (85 for the
   galaxy-only `f_NL` run `59194575`), and the first batch's energy variance was 3·10⁻² in the one
   finite chain against a target of 10⁻⁷. The priors are truncated Normals, `Omega_m` on [0.1, 0.6]
   and `sigma8` on [0.5, 1.2] (the run's `model.yaml`). In that batch the latents of both reached
   ≈ 37 σ_fid, which puts both parameters on their upper bound, the bias latents left too (`b1_` up
   to 168, `bn2_` to 784), and three of the four chains are NaN. Galaxy-only runs of montecosmo on Abacus
   did not reach an unbiased `sigma8` either (H. Simon-Onfroy, private communication). The closure member (c),
   `run_20261005_021418_59363213`, same priors and target, ends its warmup at `Omega_m` 0.293–0.316
   and `sigma8` 0.78–0.83 (truth 0.315, 0.811) with the same step size, 134, and its first batch
   holds the energy variance at 0.47–0.84 × the target in all four chains: the sampler handles
   `Omega_m` and `sigma8` free at that step, and what differs on Abacus is where its warmup goes.
   The relaunch targets an energy variance of 10⁻⁸, a step near the 85 of `59194575`.
   (b) `scan/closure_d0p20_gxyonly.yaml`.
   (c) `scan/closure_d1p00_gxyonly_cosmo.yaml`, the galaxy member of the closure
   `Omega_m`–`sigma8` triplet (step 3).

2. **Code before the κ runs.**
   (a) **The Fisher of §7.10 at the current κ model** (`fisher_cosmo.py`: shells from
   `chi_matter_min`, the cut at the init-grid Nyquist, the bilinear window on the κ signal and the
   covariance term as in the likelihood): the κ gains the runs of step 3 are compared with.
   (b) **`cmb_lensing.ell_max`** below 2·nside (a_ℓm above it dropped from the observable), for a
   κ-only `Omega_m`–`sigma8` robustness run on Abacus at ℓ ≤ 47: the remaining top-of-band excess
   shifts `sigma8` by −1.1σ in κ alone and −0.5σ joint at first order with the windowed covariance
   term, the band below by −0.2σ (§7.12). First the expected scatter of that shift from the
   line-of-sight realisation; the Abacus κ-only `Omega_m`–`sigma8` run at the full band (step 3b)
   measures the actual shift.
   (c) **Two-point comparison material** (paper §6, `sec:res_twopt`; an independent analysis by
   C. Payerne, C_ℓ^κκ and an MCMC with a CCL theory, on exactly the map our field level analyses).
   A script to write, `scripts/export_kappa_map.py`, exports from the `truth.npz` of the κ-only
   closure `Omega_m`–`sigma8` run of step 3: the observed map (`unpack_to_map`, nside 32, ℓ ≤ 64, full
   sky, FITS), the noise spectrum the likelihood uses, the bilinear window w_ℓ (`bilinear_window`), the
   radial window of the shells (`kappa_radial_window`), the per-shell taper (`cmb_shell_taper` with
   the shell radii), the source redshift, the band and a note of the conventions. In closure,
   `kappa_obs` is the packed a_ℓm of the cut Born κ from `chi_matter_min` to the box edge plus a
   Gaussian draw of variance N_ℓ + w_ℓ² C_ℓ^LOS (the matter from the box edge to `chi_high_z_max` and the
   power the cut removes). The two-point signal theory is therefore a Limber with the shells' radial
   window and an ℓ-dependent radial cut, times w_ℓ², and its noise N_ℓ + w_ℓ² C_ℓ^LOS as exported. The
   protocol: the line of sight as noise (`high_z_mode: fixed`), ACT DR6 ×1, the same priors on
   `Omega_m` and `sigma8`, the same linear spectrum, not blind on the truth (the fiducial), every
   choice fixed before either analysis is run, the two-point contour produced before ours is shown.
   The closure map's remaining top-of-band discreteness (§7.12) against that theory: its first-order
   shift with `fisher_cosmo.py --mismatch` and a two-column file of the measured/Limber ratio.

3. **Runs with κ.**
   (a) The closure `Omega_m`–`sigma8` triplet, `scan/closure_d1p00_{kappaonly,joint}_cosmo.yaml`
   with the galaxy-only member of step 1c: the Fisher κ gain (step 2a) without model error. The
   κ-only one is the run of the two-point comparison.
   (b) `Omega_m`–`sigma8` on Abacus: `abacus/abacus_kappaonly_Nl1p0_cosmo.yaml`,
   `abacus/abacus_joint_Nl1p0_cosmo.yaml`, paired with step 1a.
   (c) The Abacus joint for `f_NL`, `abacus/abacus_joint_Nl1p0.yaml`, paired with the
   galaxy-only reference `run_20261002_021703_59194575` (§7.11).
   (d) The κ-only Abacus validations, `abacus/abacus_kappaonly_Nl1p0_fnlfixed.yaml` and
   `abacus/abacus_kappaonly_Nl0p1_fnlfixed.yaml` (§7.4 with the current κ model); the first
   is the convergence proof of the cut in sampling.
   (e) The closure joints `scan/closure_d{1p00,0p10,0p20}_joint.yaml`.
   (f) The κ-only `Omega_m`–`sigma8` robustness run on Abacus at ℓ ≤ 47 (step 2b).

4. **Analyses.** Paired gains against the Fisher of step 2a (`Omega_m`, `sigma8`) and the tables of §7
   (`f_NL`); the two-point comparison; the share of the κ gain taken from the `f_NL`–`b1`/`b∇²`
   degeneracy at densities 0.1 and 0.2 (§7.3); the density-scan figure (`density_scan.py`, filling
   `scan/density_scan_runs.yaml`); the field and κ reconstructions of the new runs; the stiffness at
   cell 46.875 (`kappa_stiffness.py` needs a `--cell_size`); the cost of the cut in a joint gradient.
   **To understand before any conclusion: the closure widths against the Fisher.** Second half of
   140 batches, R-hat ≤ 1.008: galaxy-only `run_20261005_021418_59363213`
   (`scan/closure_d1p00_gxyonly_cosmo.yaml`) σ(`Omega_m`) 0.0093, σ(`sigma8`) 0.048; κ-only
   `run_20261005_030253_59364089` (`scan/closure_d1p00_kappaonly_cosmo.yaml`) 0.026 and 0.031.
   The Fisher of §7.10 at the same configuration: galaxies 0.0183 and 0.0503, κ alone 0.0608 and
   0.0461. `Omega_m` is about 2 × narrower than the Fisher in both runs; `sigma8` agrees for the
   galaxies and is 1.5 × narrower for κ, a nearly Gaussian field at ℓ ≤ 64 on the full sky, where a
   field-level posterior is expected to carry about the information of its power spectrum. One
   realisation each. Checks: the bias of the unadjusted sampler (the κ-only run again at a smaller
   `desired_energy_var`); the `Omega_m` dependences the Fisher has and the model has, or not
   (linear spectrum shape, growth, distances of the light cone, lensing kernel); the two-point
   analysis of C. Payerne on the κ-only map (step 2c, paper `sec:res_twopt`).

Open, without a run planned: the remaining +8–15 % of `(C^mm + C^LOS)/C^tt` at ℓ 44–64 at cell 93.75,
from the shells the cut leaves (§7.12); the lensing prefactor, which uses `Omega_m` where AbacusLensing
uses `Ω_cb` (§4, Abacus only); the high-ℓ galaxy-side loss of §7.8, outside the band; the ≈ 1 `P_shot`
excess of the Abacus `P_gg` (§7.8), which a galaxy-only reference with `gxy_stoch_noise: true` would
settle. Optional: a closure on the Abacus ICs (needs a `closure_init_from_abacus_ic` knob; the only
way to tell model error from realisation in the closure/Abacus comparison of §7.3); `bn2` fixed at the
current configuration; a projection of the evolved Abacus particles through the Born projector, to
separate the projector from LPT in §7.6; the CLASS/EH ratio in the line-of-sight term and in
`desi_cmb_fli.fisher`.

Configurations: `configs/inference/abacus/` (Abacus runs), `configs/inference/scan/` (closure at the
same configuration), `configs/inference/validation/` (forward model vs theory), all with the current κ
model. `configs/inference/archive/` holds the κ configs of the runs and measurements made with the
abandoned inner cut-off (suffix `_chimin<value>`, §7.1–7.12): the current code rejects them, and the
code they ran with is the git tag `pre-shell-cut`.
