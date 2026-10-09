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

A 3-D Gaussian random field on the **init grid** (`init_oversamp` × the final grid, 1.5 in every config).
The field is the inference latent
`init_mesh` (sampled as `init_mesh_` in the Kaiser-whitened basis, see §5). Mesh dimensions are
auto-adjusted (`get_model_from_config`) so all axes have an **even** number of cells: the real↔complex
Gaussian repacking `utils._rg2cgh`/`cgh2rg` asserts `all(shape % 2 == 0)`, because the
hermitian-symmetry bookkeeping (self-conjugate modes at 0 and Nyquist) assumes an exact Nyquist plane.

**Linear power spectrum (ACE emulator).** `bricks.lin_power_interp` is the linear CDM+baryon
spectrum of the ACE emulator (CosmologicalEmulators' jaxmapse, network `mnuw0wacdm_class`, trained on
CLASS; weights from Zenodo record 21328528),
normalised to the sampled σ8 with its own top-hat integral at a = 1 and grown with the background
emulator's D(a) (§2.3). The network takes (z, H0, ω_b, ω_cdm, M_ν, w0, wa) and returns, through a
PCA basis on its own k grid (7·10⁻⁶–148 h/Mpc, wider than the 10⁻⁴–10 of the interpolation), the
ratio to an analytic transfer function; the primordial A_s (k/0.05)^{n_s−1} is applied analytically,
so A_s drops out of the σ8 normalisation. The inputs follow AbacusSummit c000: the model has no
neutrinos and `Omega_m` includes one 0.06 eV species, so ω_cdm = (Ω_m − Ω_b) h² − M_ν/93.14 at the
fixed h, Ω_b, n_s, w0, wa of `get_cosmology`; the spectrum is P_cb at z = 1, the CLASS spectrum the
Abacus ICs were drawn from (`data/abacus_cosm000_CLASS_power.txt`, §4), which the ICs scale back
with a scale-independent growth, as the model does. The network is evaluated in `bricks` itself
(`_ace_power`: the weights of `data/ace_pk_lin_cb/`, a five-layer tanh MLP, the min-max scalings, the
PCA and jaxmapse's analytic transfer), without jaxmapse, jaxace or flax: importing jaxmapse turns on
`jax_enable_x64` globally, which would run the float32 production in float64, and their pins
conflict with ours; the result is jaxmapse's to 10⁻¹⁴ (`test_ace_evaluator_reproduces_jaxmapse`).
The primordial factor is taken in logarithms because (k c)⁴ overflows float32. The emulator is
trained for ω_cdm ∈ [0.08, 0.18], i.e. `Omega_m` 0.227–0.447 here; below `Omega_m` ≈ 0.15 it
returns a negative ratio. The `Omega_m` prior bounds must lie within `ACE_OMEGA_M_RANGE` = [0.2, 0.55],
where the emulator is more accurate than Eisenstein–Hu throughout and than the former
Eisenstein–Hu × fiducial CLASS/EH ratio everywhere but at the fiducial itself (§7.11); a model with
wider bounds is an error. Everything built on the linear spectrum takes it: the prior and the Kaiser
preconditioner (`lin_power_mesh`), the φ→δ transfer of `add_png` and of the scale-dependent bias
(`trans_phi2delta_interp`), the Kaiser model, the Wiener start and `desi_cmb_fli.fisher`. Closures
use the same spectrum as the Abacus runs. The Limber C_ℓ (line-of-sight term, diagnostics) keep
jax_cosmo's Eisenstein–Hu halofit. Measured accuracy in §7.11.

### 2.2 Primordial non-Gaussianity

**Theory.**

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

**Implementation.** Local PNG has two effects, both controlled by `model.png_type`:

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
`jax_cosmo` background is used, through its callbacks: a sampled `Omega_m` with another background
(real data, another simulation) needs the emulator rebuilt for it. Exact in `σ8`; its accuracy,
measured against the exact background in §7.11, is far below what the inference resolves.

### 2.4 Galaxy bias & redshift-space distortions

**Lagrangian bias expansion** (Modi+2020), weights read at the initial particle positions from the
evol-grid **Gaussian** field:
- linear `b₁`, quadratic `b₂`, tidal shear `b_{s²}`, higher-derivative `b_{∇²}` (`bn2`).
- `b₂` uses the montecosmo convention `weights += b₂·(δ²−⟨δ²⟩)/2`.

**Redshift evolution of the linear bias `b1_alpha` (α).** On the light cone the `b₁` term is
`b₁(a)·D(a)·δ_L` with `1 + b₁(a) = (1 + b₁)(D(a_fid)/D(a))^α`, `a_fid` the growth-weighted mean scale
factor of the survey cells (the loader's, the redshift where the Kaiser preconditioner is built):
`b₁` is the linear bias at `a_fid`. α = 0 is a z-independent Eulerian bias `1 + b₁`, α = 1 a constant
clustering amplitude `b_E·D`. α is a bias latent, prior N(1, 1), `scale_fid` 0.05 (`bricks.lagrangian_weights`
takes `b1_alpha` and `a_b1`; the Kaiser evolution applies the same factor to its `bE` mesh). Why:
across a broad shell a fixed `1 + b₁` makes the galaxy amplitude fall as `D(χ)`; the Abacus LRG
amplitude stays flat, and with α fixed at 0 `Omega_m`, the only parameter that changes `D(χ)` across
the shell, absorbs the difference (§7.14). DESI analyses avoid the question with narrow redshift bins
and free biases per bin; one free α is the equivalent for a single broad shell, with the
z-independent bias as a special case. Only the linear term evolves; `b₂`, `b_{s²}`, `b∇²`, `bnpar`
and the PNG terms (`b_φ` from `b₁`, i.e. at `a_fid`) do not. In snapshot mode `a_fid = a_obs` and α
has no effect: keep it in `mcmc.fixed_params` there, like `bnpar`.

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

The configs run **200³ particles** (paint 1.75, init 1.5): an evol/ptcl factor of 2.5 on the 80³
final grid of the 7500 Mpc/h box, 200/54 ≈ 3.70 on the 54³ grid of the 5000 Mpc/h box, above
montecosmo's 7/4. At 7/4 the displaced particle lattice and the bias products leave structure in the
predicted band that grows with the displacement amplitude: on Abacus the galaxy likelihood at the
true field then prefers a smaller field (`sigma8` biased low) and the κ model holds too much power at
the top of the band; at 2.5 both go (§7.12, §7.14). The inferred field (init grid) is unchanged; the
cost is that of the particles. What the κ lattice template depends on is the particle spacing
box / 200, hence the 5000 box (§2.6, §7.15).

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
map's matter stops: the CMB for real data, 3750 Mpc/h for the AbacusLensing HUGE map, whose
light cone is full-sky only to the half box (z = 2.18; §3.2). The matter the shells leave out of
that range is the line-of-sight term of the covariance (§3.2); below `chi_matter_min` the map holds
nothing, so no term. The closure configs copy the AbacusLensing range at both ends (292.6 and 3750),
so that closure and Abacus runs share their κ geometry.
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

**Lattice discreteness and the lattice template.** The particles start on a regular lattice of
spacing d = box / N_ptcl, and the LPT displacements are small against d, so the lattice survives
evolution — the same small-displacement regime that keeps 2LPT accurate. The 3-D painting does not
see it (one particle per cell of a grid aligned with the lattice paints to a constant); the HEALPix
pixels are not aligned with it. Seen from a centred observer, a shell at distance χ adds spurious
power around ℓ ≈ 2πχ/d — inside the band for near shells, above it for far ones — and rays running
along the lattice planes draw lines on the three coordinate great circles and their diagonals, which
have power at every ℓ, so far shells keep an in-band part. Averaged over the field it does not
vanish: the model's κ carries a deterministic mean E_z[κ](θ), the lattice damped by the
displacements, that a real map lacks. A power spectrum sees it at second order (power small against
the cosmic variance of a band); the field-level likelihood sees it at first order, coherently over
every mode of the band, as a mean the data must contain. Its amplitude grows with `Omega_m` (the
lensing kernel and the mass per particle) and depends little on `sigma8`, so data without it pull
`Omega_m` down along the κ degeneracy. Its size is set by d, not by the cell; the configs keep 200³
particles in a 5000 Mpc/h box (d = 25 Mpc/h) at the inference cell (§7.15). In closure data and model
share it. E_z[κ] is measured with antithetic pairs, (κ(z) + κ(−z))/2, which cancel the part of κ odd
in the field.

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
not resolve — is part of the line-of-sight term of the covariance, the matter the map holds and the
shells do not (§3.2), so it follows `high_z_mode` like it. Inside the taper's roll-off the removed part of a mode is proportional
to its kept part, and the likelihood treats it as independent noise: an approximation over a quarter
of each cut shell's ℓ_res. The cut removes the near-shell discreteness and the near-observer
stiffness, so that where the shells start no longer matters, at the price of a costlier κ gradient
(§7.12). It leaves the far shells' in-band discreteness, the lattice template, which shrinks with the
particle spacing (above, §7.15). The covariance term carries the bilinear window of
the projection sphere, like the data and model maps (§3.2). On a cut sky the filter acts on the
zero-filled shell maps.

**Abandoned: an inner cut-off `chi_min`.** Before the per-shell cut the shells started at a
`chi_min` (350, then 700 Mpc/h) and the dropped range entered the covariance as a low-z Limber term.
It worked — with the tents it made the joint sample like the galaxies alone (§7.7) and passed the
κ-only validation at 700 (§7.4) — but it gave up the near shells at every ℓ, its Limber term was
inaccurate at low ℓ, and its value was a resolution tuning (§7.6). W. Kabalan's per-shell cut does
the same per multipole and replaced it; a config that still holds `chi_min` or `chi_low_z_min` is
rejected, and the code of those runs is the git tag `pre-shell-cut`.

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
configuration (3750 Mpc/h in the 7500 box, 2500 in the 5000 box). The radial shells (`cmb_r_shells`)
span exactly that range. The residual depth `χ_box → χ_high_z_max` is in the likelihood covariance
(§3.2): 2500 → 3750 Mpc/h for the HUGE map in the 5000 box. For AbacusSummit base,
`chi_high_z_max: 3990` Mpc/h (matter simulated to z≈2.45), not `χ_CMB`.

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

**What the line-of-sight term holds.** The data map holds the matter with the lensing kernel
`W(χ)` from `chi_matter_min` to `chi_high_z_max`; shell s holds it with `q_s(χ) = d_r W(r_s) p_s(χ)
χ²/V_s` (its lensing weight at the centre times its radial profile normalised by its volume, as in
the projector), times its multipole taper `w_s(ℓ)` (§2.6). The term is the Limber C_ℓ of the one
kernel `B_ℓ(χ) = W(χ) − Σ_s w_s(ℓ) q_s(χ)` (`cmb_lensing.compute_cl_outside_model`): the matter beyond
the box, the end ramps of the linear tents (the tents sum to one only between the first and last
apex, so one `d_r` at each end is held partly) and what the cut removes, with their cross terms.
Taking the pieces apart and leaving out the ramps misstates it on both sides of the band (§7.12).

**Depth of the data map.** The model shells hold the matter from `chi_matter_min` to `chi_boundary`.
On a real map the map's matter runs from the observer to `χ_s` (9388 Mpc/h at the fiducial
cosmology). The AbacusSummit HUGE light cone is one copy of the box around a central observer,
full-sky to the half box, z = 2.18, and beyond that only toward the eight corners (AbacusSummit
documentation; Hadzhiyska et al. 2023, Fig. 3: the HUGE maps cover the full sky until z = 2.18); its
CMB map (`kappa_00045`, source at z = 1089.3) holds the matter only to there, so the Abacus configs set
`chi_high_z_max: 3750` (= `chi_boundary`): no matter beyond the box, the term is the ramps and the
cut. The source planes of the AbacusLensing maps run to z = 2.4, which is not where the matter
stops. Every run here, closure and Abacus, has `chi_matter_min: 292.6` (z = 0.10, where the
AbacusSummit HUGE light cone stops) and `chi_high_z_max: 3750` — the closure configs copy the
Abacus range. The matter below 292.6 and beyond 3750 Mpc/h is therefore in neither the data nor the
covariance: data and likelihood agree, but the map is that of an experiment without that matter. Limber (non-linear P, Abacus cosmology, `scripts/plot_lensing_fraction.py`), fraction
of the power from the observer to `χ_s`, ℓ bins 2–4 / 5–10 / 11–20 / 21–36 / 37–64 weighted by
2ℓ+1, before the per-shell cut: model shells and data map (292.6 → 3750) 0.76 / 0.82 / 0.81 / 0.76 /
0.69; in neither 0.24 / 0.18 / 0.19 / 0.24 / 0.31. The κ constraints and gains are those of a map
without that matter, i.e. with a smaller variance than a real map at the same `N_ℓ`. Figures
`figures/lensing_fraction/lensing_fraction_vs_z.png` and `lensing_spectra_comparison.png`.

Noise (`cmb_lensing.cmb_noise_nell`, a file of columns ℓ, N_ℓ; `cmb_noise_scaling`, default 1.0,
multiplies it to test sensitivity; the source is at `cmb_lensing.z_source`, 1089.28). The paper's runs
use the Simons Observatory baseline, `data/N_L_kk_so_v3_1_1_baseline_mv.txt`: the `N_lensing_MV`
column (minimum variance of TT, TE, EE, TB, EB, iterative reconstruction, no foreground
deprojection, baseline sensitivity, fsky 0.4) of `so_noise_models` v3.1.1
(`nlkk_v3_1_0_deproj0_SENS1_fsky0p4_it_lT30-3000_lP30-5000.dat`), given from L = 2. It is
0.32–0.34 × the ACT DR6 baseline over ℓ 2–64 (0.25 for the SO goal sensitivity, 0.37–0.52 with CIB or
tSZ deprojection), so no single scaling of the ACT file reproduces it, and the file is used as is.
Every config points to the SO file; `Nl<x>` in a config name is its `cmb_noise_scaling`. The ACT DR6
file (`data/N_L_kk_act_dr6_lensing_v1_baseline.txt`, every run before 2026-10-09, i.e. all of §7 but
the Fisher of §7.10, and the record config `abacus/abacus_kappaonly_Nl1p0_cosmo_box5000.yaml`) is
constant (8.766·10⁻⁸) below L = 40 and varies from L = 40 on. The band is signal-dominated: the full
`C_ℓ^κκ` (observer to `χ_s`, Limber) over `N_ℓ` is 5.6, 7.4, 6.5 at ℓ = 10, 30, 64 with SO (1.8, 2.4,
2.2 with ACT, crossing 1 at ℓ ≈ 150). Noise realizations:
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
  `chi_boundary = 3750`, `chi_high_z_max: 3750` (the map's matter stops at the half box, §3.2),
  `likelihood_mode: diagonal` (exact on the full sky). The `_box5000` configs take the central
  5000 Mpc/h cube of the same box at the same cell (`cell_size` 5000/54, 200³ particles): galaxies
  (χ ≤ 2447) inside, `chi_boundary` 2500, `chi_high_z_max` still 3750 (§2.6, §7.15).
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
  and one count, so that reduction is cached to `data/cache/randoms_<digest>.npz` (git-ignored) and
  reused; re-streaming the FITS file otherwise dominates start-up (≈ 30–40 min for a new geometry). The digest covers everything
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
but matter only as deep as the light cone, which sets `chi_high_z_max`: `kappa_00045.asdf` (HUGE
`c000_ph201`, full sky) to the half box, χ = 3750 (§3.2, measured in §7.12); `kappa_00047.asdf` (base
`c000_ph000`, two patches, three box copies) to its last lens plane, χ≈3990.

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
normalisation is needed; its spectrum is the CLASS spectrum `P_lin` emulates, §2.1, §7.11). A model box smaller than the simulation box gets the
central cube of the IC grid, the observer at the centre of both (an integer number of IC cells, 384 of
576 for 5000 Mpc/h); the model treats that cube as periodic, as an analysis of real data treats its box.
**Validation reference only**: it populates `truth['init_mesh']`, read by
`plot_warmup_diagnostics` and by `analyze_run`'s reconstruction figure (§6). The sampler is
warm-started from the *observed* galaxy field via `model.kaiser_post` and never sees it.

---

## 5. Inference & sampling

**Sampler.** MCLMC, over the field latent + scalar latents, in a reparametrized (whitened) space.
`run_inference.py` wires **only** `get_mclmc_warmup`/`get_mclmc_run`; the NUTS-within-Gibbs
(`nutswg_*`) and MAMS (`get_mams_*`) paths exist in `desi_cmb_fli.samplers` but are not reachable from
the script and are not exposed by any config switch.

**Cosmology: inferred or fixed, per config.** `Omega_m` and `sigma8` are ordinary latents with
priors in `latents`, and the background emulator (§2.3) covers a varying `Ω_m`. Two families of
runs: the `Omega_m`–`sigma8` runs (`*_cosmo.yaml`, `fixed_params: [fNL]`) and the `f_NL` runs
(`fixed_params: [Omega_m, sigma8]`, as the default `config.yaml`). Every config keeps
`cmb_lensing.high_z_mode: fixed`, the line-of-sight term cached at the fiducial cosmology (also what
`pixel_exact` requires, §3.2): in the `Omega_m`–`sigma8` runs that term does not follow the cosmology.
The `taylor`/`exact`/`exact_linear` modes make it follow; no run uses them.

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

**Run length** (`mcmc`): `num_chains` (one per GPU, lowered to the devices available),
`num_warmup`, `num_samples` per batch (after `thinning`) and `num_batches`; every batch is written
to `samples_batch_*.npz` (scalars) so a run can be resumed (`--resume`) or cut. `save_large_fields`
(default false) also keeps the field per sample, which is what makes those files large; without it
the field survives only in `sampler_state.pkl`. The `slurm` block is read by
`configs/inference/submit.py` only (runs here go through an interactive `salloc`, `docs/hpc.md`).

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

Every script in `scripts/` documents its options in its docstring; this section says what each one
measures and the conventions that make its output comparable with the model.

| script | what it gives | output |
|---|---|---|
| `analyze_run.py` | merged batches, R̂/ESS, corner and traces, the field and κ figures below; runs at the end of every job | `<run>/analysis_burn*/` |
| `compare_runs.py` | GetDist triangle of several runs (per-run burn-in, labels) | `--output` |
| `compare_reconstruction.py` | correlation of the true and sampled ICs in radial shells, per chain, several runs overlaid (§7.1); adds the fixed latents before `reparam` | `figures/results/` |
| `quick_pk_spectra.py` | 3-D spectra: closure matter `P(k)` against the light-cone-averaged linear theory (`P_lin(k, a=1)` × ⟨`D²`⟩ over the cells), or the Abacus galaxy `P_gg` against the model's at a run's biases (`--bias_run`), shot noise subtracted; bins out to `√3 k_Nyq`, since the corner modes enter the cell likelihood at full weight | `figures/spectra_diagnostic/pk_*` |
| `quick_cl_spectra.py` | `C_ℓ^{κκ}`, `C_ℓ^{gg}`, `C_ℓ^{κg}` against Limber, closure (N realisations) or Abacus; conventions below | `figures/spectra_diagnostic/cl_*` |
| `validate_kappa_from_ic.py` | the κ model at the true Abacus ICs against the AbacusLensing map, no sampling: per ℓ bin coherence, transfer, error power against `N_ℓ` and `C_ℓ^LOS`, the coherence `√(1 − C^LOS/C^tt)` of a model exact but for its covariance term, and `(C^mm + C^LOS)/C^tt` (1 for a consistent model); `--shell_kmax` list, `--maps` for the pole maps (§7.9); the nside-16384 map is read once on a compute node and cached in `data/cache/` | `figures/spectra_diagnostic/kappa_from_abacus_ic_*.{png,npz}`, `figures/maps/` |
| `kappa_stiffness.py` | largest Hessian eigenvalue of −log p in the field (power iteration on reverse-over-reverse Hessian-vector products), its eigenvector's share near the observer, cost and peak memory, per `shell_weights:shell_kmax[:noise]` variant; the MCLMC step size is capped by it | `figures/conditioning/` |
| `kappa_lattice_template.py` | `measure`: E_z[κ] by antithetic pairs `(κ(z) + κ(−z))/2` on an (`Omega_m`, `sigma8`) list, packed like the likelihood, with the field power, the undisplaced-lattice κ, `N_ℓ` and the line-of-sight term; `analyse`: S/N, projection of a run's data, Gaussian posterior with and without the template (§7.15) | `data/cache/template/`, `figures/kappa_lattice_template/` |
| `plot_lattice_template.py` | the two figures of §7.15 | `figures/kappa_lattice_template/` |
| `galaxy_likelihood_profile.py` | counts likelihood at the true ICs (cropped to the model box), biases profiled by L-BFGS, against the field amplitude A or `Omega_m`; grid overrides `--box --mesh --particles` (§7.14–7.16) | `figures/galaxy_likelihood_profile/` |
| `galaxy_bias_regression.py` | linear bias and RSD amplitude of a run's counts against its true ICs per k band and shell (§7.14) | `figures/galaxy_bias_evolution/` |
| `bench_gradient.py` | time and peak memory of one log-density gradient, particle grid or geometry overridden | stdout |
| `fisher_cosmo.py` | Fisher on (`Omega_m`, `sigma8`, `b1`, `b∇²`[, `b1_alpha`, `f_NL`]) of galaxies, κ and joint at a config's geometry, band, noise and priors; `--resolution_scan` over box and cell; `--mismatch FILE` the first-order shift F⁻¹b from a per-ℓ data/model κ power ratio, `b_a = f_sky Σ (2ℓ+1)/2 tr(C⁻¹∂_aC C⁻¹ΔC)` (§7.10) | `figures/fisher_diagnostic/` |
| `density_scan.py` | measured paired gain on σ(`f_NL`) against the Fisher, per density of `scan/density_scan_runs.yaml`; `--noise_table` the gain per `N_ℓ` scaling (§7.2, §7.3) | `figures/results/density_scan.{png,json}` |
| `compare_linear_power.py` | the model's linear P(k) (and Eisenstein–Hu) against the CLASS spectrum of the Abacus ICs; `--omega_m_scan` the emulator, the former EH × fiducial ratio and EH against CLASS as `Omega_m` moves, from the reference written by `linear_power_class_reference.py` (cosmodesi env, cosmoprimo) (§7.11) | `figures/spectra_diagnostic/linear_power_*` |
| `plot_cmb_noise_comparison.py` | Planck PR4, ACT DR6, SO baseline `N_ℓ` and their ratio to ACT, likelihood band shaded | `figures/spectra_diagnostic/cmb_noise_comparison.png` |
| `plot_lensing_fraction.py` | Limber κ power along the line of sight at a config's geometry: fraction captured against depth, model shells, map, covariance term, `N_ℓ` (§3.2) | `figures/lensing_fraction/` |
| `plot_2D_maps.py` | κ and galaxy-projection maps of one forward realisation (the galaxy panel ray-casts the mesh, display only) | `figures/maps/` |
| `benchmark_highz_cl_modes.py`, `plot_highz_correction_vs_depth.py` | precision and speed of the line-of-sight modes; its amplitude against depth | `figures/high_z_modes_comparaison/` |
| `make_abacus_kappa_mask.py`, `plot_linear_vs_nbody.py` | AbacusLensing footprint at the model nside; linear against N-body field from one IC | — |

`DIAGNOSE_FREEZE=1` in `run_inference.py` prints the log-density, per-group gradients and 1-D scans of
each free scalar at the start of step 2, to localise curvature pathologies.

**Conditioning.** Every diagnostic that runs `predict` conditions on every latent
(`validation.conditioning_params`, the missing ones at `loc_fid`); otherwise `predict` draws them from
their priors. `analyze_run.py` and `compare_reconstruction.py` add the `mcmc.fixed_params` at the values
the run conditioned on (`truth_params` in closure, `abacus_truth_params` on Abacus).

**Angular spectra** (`quick_cl_spectra.py` and the startup check of `run_inference.py`, one figure
`validation.plot_cl_figure`). One Limber curve per spectrum (`validation.compute_cl_theory`); the
measured maps are made comparable with it rather than the theory decorated with noise terms.
- κ is the noiseless map. Theory in closure: the Born shells from `chi_matter_min` to `chi_boundary`
  with `k_⊥` below the init-grid Nyquist (the inferred field has no power above it), each shell times
  its squared taper under the per-shell cut; on Abacus: the line of sight from `chi_matter_min` to
  `chi_high_z_max`. Both times the bilinear window at `cmb_proj_nside` (`bilinear_window`: 0.936 at
  ℓ = 50, 0.898 at ℓ = 64 for nside 64), which model and data carry alike.
- κg in closure multiplies the lensing kernel by the shells' radial window (`kappa_radial_window`):
  where the shells start or stop inside the galaxy range, the model κ does not hold the matter those
  galaxies trace.
- Galaxies are projected from galaxies, never from the mesh (mesh interpolation adds a low-pass window
  no Limber curve holds, ≈ 20 % of `C_ℓ^{gg}` at 0.4 `k_Nyq`): the catalogue count map of the loader
  (`gxy_hp_counts`, nside 256), pixel window removed and shot noise subtracted; or the model particles
  (`rsd_pos`) weighted by `gxy_weights`, the selection and the mask, spread bilinearly like the Born
  projector and divided by `bilinear_weight_norm` — nearest-pixel counts of a displaced lattice carry a
  moiré that inflates `C_ℓ^{gg}` by 10–60 %. Either is divided by the unclustered expectation (the
  selection ray-integrated with `r²`, which also gives the theory's `dN/dχ`). This map enters only the
  diagnostic, never the likelihood (§3.1).

**Field-level figures of `analyze_run.py`.** Their maps are one posterior sample, not a mean over
chains: averaging independent samples suppresses the amplitude wherever the data leave the field free.
The field is read from `sampler_state.pkl` (the batches hold scalars only); `--no_field_plots` skips them.
- `initial_conditions.png`: true field, one sample, difference in the plane through the observer, then
  `T(k)`, `r(k)` and a radial profile (rms ratio, correlation with the truth, agreement between
  chains); dashed circles = the galaxy range.
- `kappa_reconstruction.png` (runs with κ): observed κ, model κ, error, and their spectra against the
  assumed covariance, band-limited to the likelihood's band; the harmonic coherence
  `r_ℓ = C^{tm}/√(C^{tt}C^{mm})` (Δℓ = 4) of each chain's sample with the truth, of the posterior mean
  with the truth and between chains, each against its expectation for a correct Gaussian posterior
  (§7.4). A model error pulls the coherence with the truth below its expectation; a structure the
  chains share but the data do not impose pushes the agreement between chains above its own. These
  expectations are κ's own Wiener filter: they hold for κ-only runs, not for joints.
- `kappa_posterior_mean.png`: truth, posterior mean, per-pixel std and `(truth − mean)/std` from
  `kappa_batch_*.npz` (each chain's `kappa_obs` after every batch, `FieldLevelModel.kappa_observable`);
  without them, from the final states only.

**Fisher forecasts** (`desi_cmb_fli.fisher`, tested against closed forms in `tests/test_fisher.py`).
- *Galaxies, 3-D* (`galaxy_fisher_3d`, `galaxy_fisher_3d_cosmo`): `F_ab = ∫dV ∫k²dk/(2(2π)²) ∫dμ
  ∂_aP ∂_bP/(P + 1/n̄)²`, `P = (b(k, z) + fμ²)² P_lin(k, z)`, `b = 1 + b1 − b∇²k² + b_φ f_NL/M(k, z)`,
  `b_φ = 2δ_c b1` (universality, §2.2); volume the full-sky LRG shell, `n̄(χ)` the catalogue n(z);
  `k_max = π/cell`, `k_min = 2π/V^{1/3}` (halving or doubling `k_min` changes σ(`f_NL`) by ×0.7 and
  ×1.6). With `b1_alpha`, `(1 + b1)(D(z_b1)/D(z))^α`. Left out: `b2`, `b_{s²}` (not in the linear
  spectrum; |ρ| with `f_NL` ≤ 0.09, §7.3), `b∇∥`, the mask.
- *κ* (`angular_fisher_cosmo`, `kappa_fisher_increment`): tomographic Limber over ℓ 2 … 2·nside,
  `F = Σ_ℓ (2ℓ+1)/2 tr(C⁻¹∂_aC C⁻¹∂_bC)`, ten equal-number galaxy shells without Kaiser term, κ from
  the shells under the per-shell taper (the removed power as noise), noise `N_ℓ` plus the line of
  sight (as noise at the fiducial, or as signal). Joint = 3-D galaxies + `F[shells + κ] − F[shells]`,
  the increment over *angular* galaxy information (an approximation). For `f_NL`, `σ_κ = √(ΔF⁻¹)` is
  κ's incremental information through the galaxies, not a κ-only constraint: κ alone sees `f_NL` only
  through the weak matter φ² channel (§2.2). Cosmology moves z(χ), growth and
  linear power at fixed galaxy distances (no Alcock–Paczynski information, as at the field level).
- *Measured paired gain* (`paired_sigma_ratio`): runs sharing `seed` and warm start pair chain `i`
  with chain `i`; the ratio is the mean over chains of σ_joint/σ_gxy on the second halves at matched
  batches, its error the spread over chains/√n.

---

## 7. Validation status & measured results

| question | answer | where |
|---|---|---|
| galaxies alone, `Omega_m`–`sigma8`, Abacus | unbiased with `b1_alpha` free and 200³ particles: box 7500 −0.47σ / −0.19σ; box 5000 (non-periodic) −0.25σ / +1.16σ | §7.14 |
| κ model at the true ICs | error inside the covariance at ℓ ≤ 64 (box 7500); at box 5000 from ℓ = 12, excess at ℓ 2–11 | §7.12, §7.15 |
| κ alone, `Omega_m`–`sigma8`, Abacus | box 7500: biased along the degeneracy by the lattice template; box 5000: Mahalanobis distance of the truth 2.2 | §7.15 |
| closure `Omega_m`–`sigma8` triplet | unbiased; κ gain on σ(`sigma8`) 62 % (Fisher 63 %), on σ(`Omega_m`) 13 % (Fisher 29 %) | §7.13 |
| joint `Omega_m`–`sigma8`, Abacus | box 7500 (40 batches): −0.4σ of the galaxies on `Omega_m`, σ(`sigma8`) −65 %; box 5000 to run | §7.15, §8 |
| galaxy likelihood against the grids | depends on the ratio of the particle and paint grids; cause not established | §7.16 |
| κ gain on σ(`f_NL`) | Abacus none (0.992 ± 0.014); closure 9 % ± 3 % (density 1) to 17 % ± 3 % (0.03) | §7.1–7.3, to redo |
| field outside the galaxy shell | reconstructed by κ alone (inner cone r ≈ 0.3–0.4 at 250 Mpc/h) | §7.1, §7.5, to redo |

§7.1–7.11 are records: their κ and joint runs predate the per-shell multipole cut (shells from the
`chi_min` of their config, 350 or 700 Mpc/h, configs in the git history, §8) and, before §7.9, the
projector's per-pixel normalisation; their galaxy-only results do not depend on it. §7.12–7.16 use the
current κ model. Runs before the counts likelihood (on the Abacus CubicBox and base-box octant, where
`png_type: fNL` with fine-bin `ngbars` and oversampling gave `f_NL` = 1.5 ± 33 and `fNL_bias` proved
not identifiable) are not comparable with anything below.

### 7.1 Full-sky κ × LRG on the HUGE box, `f_NL` — measured (`chi_min` 350, to redo)

Pair identical but for `cmb_lensing.enabled` (completeness cut 0.3, `png_type: fNL`, ACT DR6 ×1,
nside 32, tents, `proj_oversamp: 2`, cosmology fixed): joint `run_20260910_070354_58161527` (92
batches), galaxies `run_20260910_033019_58153868` (55). The survey treatment (cut 0.8 / 0.3 / 0.05,
`ngbars` free or fixed) moves `f_NL` by < 0.2σ; its +1.5σ is common to every variant (p ≈ 0.13).

**κ moves no parameter and adds nothing to any σ.** `f_NL` 8.94 ± 5.70 (joint) and 8.87 ± 5.77;
paired σ ratios (55 batches, second half): `f_NL` 0.992 ± 0.014, `b1` 1.015 ± 0.013, `b2` 0.982 ±
0.009, `bs2` 0.969 ± 0.030, `bn2` 1.019 ± 0.009, `bnpar` 0.998 ± 0.032 — a gain on σ(`f_NL`) below
3.5 % at 95 %. Early in the chains these ratios sit a few percent below 1 from Monte-Carlo noise alone:
compare full chains only. The joint samples like the galaxies (step size 85.2 against 85.9, R̂ ≤ 1.02).

**What κ adds: the field outside the galaxy volume** (`compare_reconstruction.py`, 250 Mpc/h
smoothing, correlation of true and sampled ICs, spread over the chains):

| shell χ [Mpc/h] | 361–722 | 722–1083 | 1443–1804 | 1804–2165 | 2165–2526 | 2526–2887 | 2887–3248 | > 3969 |
|---|---|---|---|---|---|---|---|---|
| joint | **0.327 ± 0.068** | 0.250 ± 0.228 | 0.951 ± 0.003 | 0.908 ± 0.007 | **0.509 ± 0.034** | **0.146 ± 0.009** | **0.123 ± 0.025** | ≈ 0 |
| galaxy-only | −0.154 ± 0.175 | 0.051 ± 0.176 | 0.950 ± 0.007 | 0.892 ± 0.011 | 0.419 ± 0.074 | 0.034 ± 0.079 | 0.000 ± 0.011 | ≈ 0 |

Inside the LRG shell (χ 1094–2447) the galaxies already reach 0.95; outside it κ is the only
information; beyond the Born range both return to zero. At 100 Mpc/h the inner cone reaches 0.16.

### 7.2 Dependence on the CMB lensing noise level — measured (`chi_min` 350, to redo)

§7.1 at `cmb_noise_scaling: 0.1` (`run_20260922_010054_58741397`): σ(×0.1)/σ(×1) = 1.007 ± 0.062,
although ×0.1 holds 89 % of a noiseless experiment's effective κ modes (`Σ(2ℓ+1)[S/(S+N)]²` = 1599,
3737, 4221 at ×1, ×0.1, noiseless): the null on `f_NL` is structural, not noise-limited. On Abacus a
lower `N_ℓ` lowers the assumed covariance, not the model error; only closure makes the noise axis a
forecast.

### 7.3 Closure at the HUGE configuration and the tracer-density scan — measured (to redo)

`closure_geometry_from_abacus: true` builds the selection from the Abacus loader so the closure
galaxies fill the LRG shell exactly; `closure_gxy_density_scale` multiplies n̄; `truth_params` are the
§7.1 galaxy means (`b1` 1.19, `b2` 0.68, `bs2` −0.48, `bn2` 78, `bnpar` −45, `f_NL` 0) on a fresh
field (`seed` 77). Galaxies alone: σ(`f_NL`) 4.47, 7.46, 13.92 at densities 1.0, 0.1, 0.03 (3-D
Fisher 4.31, 7.86, 15.71).

| density | galaxy-only | joint (`chi_min` 350) | σ(`f_NL`) gxy → joint | paired ratio | two-point Fisher gain |
|---|---|---|---|---|---|
| 1.0 | `run_20260922_071921_58748954` | `run_20260923_024316_58785136` | 4.43 → 4.02 | 0.910 ± 0.034 | 2.8 % |
| 0.03 | `run_20260923_054507_58787249` | `run_20260923_054540_58787260` | 14.1 → 11.6 | 0.829 ± 0.027 | 3.4 % |

At density 0.03 the field level extracts about five times the two-point gain (14 % ± 1 % at `chi_min`
700). The gain comes out of the `f_NL`–`b1`/`b∇²` degeneracy (79 % of the variance κ removes at 0.03,
62 % at 1.0; ρ(`f_NL`, `b1`) −0.680 → −0.593 at 0.03): κ measures δ_m and the galaxies `b(k)δ_m`, so
its information on `f_NL` passes through the cross-correlation and degrades with the galaxy shot noise.
The density scan's x-axis is therefore the density, not σ_gxy/σ_κ (`density_scan.py`,
`figures/results/density_scan.png`). Figures: `figures/results/closure_d{0p03,1p00}_gxyVSjoint_fNL_run_…png`.

**Beyond the first paper.** A power-spectrum analysis for comparison (A. de Mattia); HalfDome
(`/global/cfs/cdirs/cmb/gsharing/halfdome`, arXiv:2407.17462: Gaussian and matched `f_NL` = 20 ICs, no
CMB κ — Born-integrate its mass sheets); DESI LRG × Planck/ACT/SO κ.

### 7.4 κ-only on Abacus HUGE: what a correct κ posterior looks like — measured

**Expectation** (`Omega_m`, `sigma8`, `f_NL` fixed, field sampled). The model generates power `S_m`;
the likelihood treats `N_a` = noise + line of sight as noise, so `W = S_m/(S_m + N_a)`; the Abacus map
is `S_t = S_m + L` (`L` = the line of sight the model does not generate, 0 in closure). The posterior
mean is `W·d` and a sample adds an independent fluctuation of variance `W·N_a`: two samples agree at
coherence `W`, a sample reaches `W·√(S_t/S_m)` with the truth, the posterior mean `√(W·S_t/S_m)`, and
the expected error power is `S_t + S_m − 2W·S_t`. `analyze_run.py` draws these curves (§6).

**Measured** (`chi_min` 700, `run_20260924_081242_58822478`): every chain at the expected coherence in
every ℓ range, chains agreeing no more than with the truth, error/assumed covariance 0.81–1.09. At
`chi_min` 350 the near-shell discreteness showed (chains agreeing with each other more than with the
truth at ℓ ≥ 48, a spurious near-observer structure) — what the per-shell cut now handles (§2.6). At
`N_ℓ` ×0.1 (`run_20260925_020623_58859550`): coherence with the map 0.86–0.90 against 0.871 expected.
The 3-D field from κ alone is weak (r 0.03–0.15 at 100 Mpc/h).

### 7.5 Abacus joint at `chi_min` 700 — measured (to redo)

`run_20260927_060952_58951736`: `f_NL` 8.85 ± 5.57, paired σ(`f_NL`) ratio 0.92–0.95. Inner cone at
250 Mpc/h (361–722 Mpc/h) 0.395 ± 0.105 against 0.327 ± 0.068 at `chi_min` 350, gone at 100 Mpc/h;
why it survives without shells below 700 is not established.

### 7.6 κ from the Abacus ICs through the forward model — measured (`chi_min` set-up)

`validate_kappa_from_ic.py` (§6) with shells from where the map's matter starts and no cut, cell 93.75:
excess model power at the top of the band, 1.76 × `N_ℓ` of error outside the covariance at ℓ 56–64
(0.39 from 500 Mpc/h, 0.23 from 700); at cell 46.875 ≤ 0.03 `N_ℓ` whatever the start. `chi_min` was a
resolution tuning, which led to the per-shell cut (§2.6, §7.12). A spectrum alone did not show it: the
discreteness power roughly filled the place of the line of sight the model lacks (10–13 % of the map's
power at ℓ ≥ 36).

### 7.7 Stiffness of the κ log-density and the radial tents — measured

`kappa_stiffness.py`, κ alone, cell 93.75, one closure draw (`figures/conditioning/kappa_stiffness.json`):

| `shell_weights` | shells from | λ_max | variance of the top eigenvector < 350 / < 700 / < 1100 Mpc/h |
|---|---|---|---|
| nearest | observer | 17 311 | 0.98 / 1.00 / 1.00 |
| linear (tents) | observer | 471.7 | 0.92 / 0.99 / 1.00 |
| nearest | 350 | 15.67 | 0.07 / 0.84 / 0.89 |
| linear | 350 | 10.64 | 0.02 / 0.79 / 0.85 |
| linear | 700 | 2.38 | 0.00 / 0.00 / 0.11 |
| prior alone (κ noise × 10⁸) | 700 | 1.00 | — |

The tents divide λ_max by 37 (with `nearest` a particle crossing a shell edge moves its whole weight
and the log-density jumps); the stiffest direction is the near observer. With the per-shell cut λ_max
is 2.4–2.7 wherever the shells start (§7.12). Step sizes scale as λ_max^(−1/2): before the tents κ
divided the joint step size by ≈ 30 (138 against 4.7); since, the joint samples like the galaxies.

### 7.8 Spectra of the forward model against theory — measured

Closure angular spectra (`quick_cl_spectra.py`, 20 realisations, `chi_min` 700, to regenerate, §8),
ℓ ≈ 10–48: κκ and κg 0.94–1.08 × Limber, gg 1.00–1.15; a galaxy kernel uniform in χ instead of the
survey dN/dχ reads as a 10 % κg deficit at ℓ 5–10. At nside 128 (cell 46.875) κκ is 0.95–1.05 ×
Limber up to ℓ ≈ 150, then rises with the discreteness. An undisplaced-lattice toy places the
discreteness: shells at 350–550 Mpc/h give 0.75–1.57 × the Abacus C_ℓ at ℓ 45–65, everything beyond
700 0.19–0.41. Abacus LRG `P_gg` against the model at the galaxy posterior biases (`quick_pk_spectra.py
--bias_run`): 1.01 at k 0.015–0.025 rising to 1.13 at 0.042–0.053 h/Mpc, ≈ 1 `P_shot` of excess
(`figures/spectra_diagnostic/pk_abacus_vs_model{,_lin_pk_table}.png`); closure matter `P(k)` 0.98 ×
the light-cone-averaged linear theory (`pk_closure_matter.png`).

### 7.9 Polar caps of the bilinear projector — measured, fixed

On the Abacus ICs (`validate_kappa_from_ic.py --maps`, cell 93.75, `chi_min` 700) model − Abacus at
ℓ ≤ 64 at the poles: +0.024 / +0.021 with a uniform solid angle per pixel, −0.004 / −0.007 with
`bilinear_weight_norm`, i.e. in the range of the four other axes (−0.001 to −0.005). Before the fix
the eight polar-cap pixels sat at 8.2 × the rms elsewhere (uniform matter gives κ = 0.091 there).
Maps: `figures/maps/kappa_from_abacus_ic_poles_cell93p75_chimin700{_before_fix,}.png`, kept as the
before/after record; the current model: `…_poles_cell{93p75,92p5926}_kmax0.05027.png` (§7.12).

### 7.10 Fisher forecasts — computed

**`Omega_m`, `sigma8`, box 5000, `b1_alpha` free** (`fisher_cosmo.py --b1_alpha` on
`abacus/abacus_joint_so_cosmo_box5000.yaml` and the same with the ACT file; line of sight as noise,
as in the likelihood; `figures/fisher_diagnostic/fisher_cosmo_box5000_{act,so}_b1alpha.{png,json}`):

| σ(`Omega_m`) / σ(`sigma8`) | ACT DR6 | SO baseline |
|---|---|---|
| galaxies | 0.0288 / 0.0662 | 0.0288 / 0.0662 |
| κ alone | 0.0646 / 0.0345 | 0.0488 / 0.0256 |
| joint | 0.0153 / 0.0210 | 0.0139 / 0.0169 |
| κ gain on the joint | 47 % / 68 % | 52 % / 75 % |

SO lowers κ alone by 24–26 % and the joint by 9 % (`Omega_m`) and 20 % (`sigma8`), less than its ×3
lower noise: at ℓ ≤ 64 κ is already signal-dominated with ACT (§3.2), and the line of sight stays in
the noise. With the line of sight as signal: κ alone 0.0434 / 0.0251 (ACT), 0.0321 / 0.0183 (SO).

**Box 7500, closure configuration, no bias evolution** (the reference of §7.13;
`fisher_cosmo.json`, `…_contours.png`): galaxies 0.0183 / 0.0503, κ alone 0.0608 / 0.0461, joint
0.0131 / 0.0188 (gains 29 % / 63 %); with `b1_alpha` free (`fisher_cosmo_b1alpha.json`) galaxies 0.0288
/ 0.0667, joint 0.0155 / 0.0201.

**Resolution** (`--resolution_scan`, ACT, density 1, `fisher_cosmo_resolution.json`), joint σ and κ gain:

| box / cell (cost) | σ(`Omega_m`) / σ(`sigma8`) | κ gain |
|---|---|---|
| 7500 / 93.75 (1) | 0.0131 / 0.0188 | 29 % / 63 % |
| 7500 / 62.5 (3.4) | 0.0071 / 0.0132 | 6 % / 38 % |
| 7500 / 46.875 (8) | 0.0040 / 0.0097 | 2 % / 23 % |
| 5000 / 92.6 (0.3) | 0.0127 / 0.0201 | — |
| 5000 / 62.5 (1) | 0.0070 / 0.0147 | 7 % / 31 % |
| 5000 / 41.67 (3.4) | 0.0035 / 0.0091 | 2 % / 13 % |

A finer cell gives the galaxies far more and κ a smaller share; the linear galaxy Fisher flatters the
galaxies more the higher `k_max`. **`f_NL`**: the two-point κ increment is 2.84 % at density 1 and
4.15 % at 0.03.

### 7.11 Linear power spectrum and background — measured

Shape at fixed σ8, max |P/P_CLASS − 1| over k 0.001–0.0335 h/Mpc (the band; to the corners 0.058 in
brackets), reference CLASS P_cb at z = 1 with `Omega_m` moved at fixed h, Ω_b, n_s
(`compare_linear_power.py`, `linear_power_vs_class.png`, `linear_power_omega_m_accuracy.{png,npz}`):

| `Omega_m` | 0.15 | 0.20 | 0.22 | 0.25 | 0.28 | 0.315 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ACE emulator (model) | 84 | 2.19 | 0.73 | 0.22 | 0.08 | 0.06 (0.09) | 0.05 | 0.05 | 0.04 | 0.38 | 1.95 | 5.1 |
| EH × [CLASS/EH] at the fiducial (former `lin_pk_table`) | 9.2 | 4.9 | 3.7 | 2.2 | 1.1 | 0 | 0.8 | 1.7 (2.2) | 2.3 | 2.9 | 3.3 | 3.6 |
| Eisenstein–Hu | 7.1 | 2.9 | 2.5 | 2.8 | 3.3 | 3.8 | 4.3 | 4.8 | 5.2 | 5.5 | 5.7 | 5.9 |

(in %; at `Omega_m` 0.12 the emulator ratio is negative.) Against the Abacus table itself at the
fiducial: emulator 0.9998–1.0011 over k 0.001–0.06, Eisenstein–Hu 0.962–0.997. The table is CLASS
P_cb at z = 1: CLASS on c000 matches it to 0.07 % in shape there, 0.32 % at z = 0, 0.86–1.08 % in
total matter. Time on CPU (jit, float32, `lin_power_interp` and `trans_phi2delta_interp` with their
`Omega_m` gradient): 3–6 ms per call for both; compiling the gradient 0.5 s against 32 s for the
Eisenstein–Hu path (jax_cosmo's σ8 integral).
The background emulator (§2.3) against jax_cosmo midway between `Omega_m` nodes, z ≤ 2.1: D, f, D₂, f₂
within 6·10⁻⁵, χ within 0.2 Mpc/h — two orders of magnitude below the spectrum's error, and
independent of it (the spectrum is taken at a = 1). On the galaxy-only `f_NL` run, the former
`lin_pk_table` against Eisenstein–Hu: `f_NL` 8.92 ± 5.84 against 8.88 ± 5.79, `b1` −2σ, `b∇²` −1.5σ.

### 7.12 Per-shell multipole cut — measured

The bench that chose the cut (§2.6), at k = π/62.5 = 0.0503 (init Nyquist at cell 93.75), against no
cut and the final-grid Nyquist 0.0335. IC test, `(C^mm + C^LOS)/C^tt` in ℓ 2–11 / 12–23 / 24–35 /
36–43 / 44–47 / 48–51 / 52–55 / 56–64, shells from 292.6 (`kappa_from_abacus_ic_cell93p75_{nocut,kinit,kfinal}`,
`…cell46p875_{nocut,kinit}`):

| cell, cut | ratio |
|---|---|
| 93.75, none | 0.95 1.01 1.02 1.15 1.38 1.77 1.84 2.26 |
| 93.75, 0.0503 | 0.95 1.00 1.02 1.06 1.09 1.14 1.18 1.12 |
| 93.75, 0.0335 | 0.96 1.02 1.02 1.06 1.08 1.26 1.20 1.13 |
| 46.875, none | 0.95 1.01 1.02 1.03 1.02 1.05 1.06 1.03 |

The cut at the init Nyquist removes the near-shell discreteness (top-band coherence 0.62 → 0.85, the
value of a model exact but for its covariance term) and makes the shell start irrelevant above ℓ ≈ 12;
the final-grid Nyquist sends resolved structure to the covariance. With the cut, the closure κκ at
ℓ ≈ 40–60 goes from 2.5–4.4 × Limber to 1.03–1.13, κg from 1.23–1.40 to 0.97–1.00; λ_max from 486 to
2.6 (shells from the observer) at +60 % per Hessian-vector product (`kappa_stiffness_kmax{0,0.0503}.json`).

**The current model at the true ICs** (cut 0.0503, windowed covariance term; the first two rows with
the earlier 3942 Mpc/h depth, the last at 3750, where with 3942 its ratio read 0.953 0.998 1.009 1.019
1.076 1.003), in ℓ 2–11 / 12–23 / 24–35 / 36–47 / 48–55 / 56–64:

| configuration | `(C^mm + C^LOS)/C^tt` | (error − covariance)/`N_ℓ` | coherence (exact model) |
|---|---|---|---|
| cell 93.75, 140³ particles (evolution grid 1.75) | 0.954 0.994 1.010 1.057 1.128 1.082 | +0.014 −0.016 −0.010 +0.017 +0.049 +0.118 | at ℓ 48–51 / 52–55 / 56–64: 0.87 0.87 0.85 (0.89) |
| cell 62.5 | 0.950 1.014 1.017 1.023 1.046 1.015 | +0.012 −0.024 −0.033 −0.042 −0.035 −0.025 | 0.986 0.994 0.988 0.978 0.957 0.954 |
| **cell 93.75, 200³ particles (2.5)** | 0.967 0.984 0.990 0.992 1.037 0.966 | +0.002 −0.003 +0.002 +0.010 +0.016 +0.032 | 0.986 0.987 0.965 0.944 0.903 0.900 (0.987 0.986 0.965 0.947 0.909 0.914) |

(`kappa_from_abacus_ic_cell93p75_maps_cut`, `…cell62p5`, `…cell93p75_evol2p5`.) More particles at the
same cell keep the model error inside the covariance at every ℓ, as a finer cell does, for the cost of
the particles alone; power and coherence do not see the lattice template (1 % of the power, §7.15).
The polar caps sit at +0.5 × the rms elsewhere (`figures/maps/kappa_from_abacus_ic_poles_cell93p75_kmax0.05027.png`).
First-order shift of the 140³ mismatch (`fisher_cosmo.py --mismatch …cell93p75_maps_cut.npz`): κ alone
−0.91σ / −1.32σ on `Omega_m` / `sigma8`, joint +0.03σ / −0.72σ; ℓ ≤ 47 alone: κ alone −0.11σ / −0.23σ,
joint +0.03σ / −0.18σ (one realisation: bias plus line-of-sight scatter).

**The depth of the AbacusLensing HUGE map.** The model at the true ICs recovers the map with a
transfer of 0.999 (`ph201`) and 1.000 (`ph202`) over ℓ 12–64. The map's power not correlated with the
model matches the non-Limber power of the kernel the shells leave out with the map's matter stopping
at 3750 Mpc/h (1.29 / 0.75 / 0.77 / 0.79 / 0.80 against the measured 1.23 / 0.68 / 0.79 / 0.85 / 0.90
in units of the term the earlier runs used, ℓ 2–11 / 12–23 / 24–35 / 36–47 / 48–64), not at 3942
(3.56 / 1.73 / …): the map holds no matter beyond the half box, where its light cone stops being
full-sky; 3942 was its last source plane. No deficit in the caps around the box axes, no corner
excess. Limber of that kernel (`compute_cl_outside_model`) is within 2.1 % of the Bessel power in
every band. Scripts in `~/claude_scratch/bias_evol/` (`los_exact_kernel_v2.py`, `powpost_*.py`).

### 7.13 Closure `Omega_m`–`sigma8` triplet — measured

Box 7500, current κ model, `f_NL` 0, `Omega_m`, `sigma8` and the five biases free:
`scan/closure_d1p00_{gxyonly,kappaonly,joint}_cosmo.yaml` (`run_20261005_021418_59363213`,
`run_20261005_030253_59364089`, `run_20261005_053406_59368973`), 140 batches, 4 paired chains, second
halves; figure `figures/results/closure_d1p00_cosmo_gxyVSkappaVSjoint_run_…png` (`compare_runs.py`).

| | `Omega_m` (truth 0.3152) | `sigma8` (truth 0.8114) | corr | Fisher σ (§7.10) |
|---|---|---|---|---|
| galaxies | 0.3083 ± 0.0093 (−0.74σ) | 0.825 ± 0.048 (+0.28σ) | 0.53 | 0.0183 / 0.0503 |
| κ only | 0.3488 ± 0.0255 (+1.31σ) | 0.836 ± 0.031 (+0.80σ) | 0.82 | 0.0608 / 0.0461 |
| joint | 0.3084 ± 0.0081 (−0.84σ) | 0.804 ± 0.018 (−0.40σ) | 0.59 | 0.0131 / 0.0188 |

Biases within 1.1σ. Paired ratios joint/galaxies: `sigma8` 0.379 ± 0.014 (gain 62 %, Fisher 63 %),
`Omega_m` 0.871 ± 0.022 (13 %, Fisher 29 %). Split R̂ ≤ 1.018, ESS ≥ 268; widths stable across halves
(4 %) and chains (8 %). `Omega_m` is 1.6–2.4 × narrower than the Fisher in all three runs, `sigma8`
only for κ alone. **κ alone: the lattice template** (§7.15), whose amplitude depends on `Omega_m` and
`sigma8`, carries information the Limber Fisher does not have; in closure it is in the data. **Galaxies:
open** — §7.14 gives a candidate channel (the growth ratio read without sample variance), untested (§8).

### 7.14 Redshift evolution of the galaxy bias — measured

**The Abacus LRG bias evolves; the model's did not.** The model's `b1` term is `b1·D(a)·δ_L` with one
`b1`. Regression of the Abacus counts on the true linear field (`galaxy_bias_regression.py`, k 0.01–0.034,
linear Kaiser with the radial line of sight): `B` = 1.833 ± 0.012, 2.102 ± 0.008, 2.264 ± 0.013 in three
shells of χ 1093–2447, so `B·D(χ)` is flat (outer/inner 1.235 against 1/D's 1.244); the closure
(non-evolving truth) 2.123, 2.149, 2.151. RSD amplitude 0.970 ± 0.018 of the fiducial `f`
(`figures/galaxy_bias_evolution/bias_evolution_true_ic.{png,json}`).

**It drives `Omega_m` down**, the only parameter changing the slope of `D(a(χ))` across the shell.
Galaxy likelihood at the true field, biases profiled, χ² − χ²(0.315) (63 614 cells), with the code's
bias (α = 0) and with `b_E(a) = (1 + b1)(D(a_fid)/D(a))^α`:

| `Omega_m` | 0.20 | 0.245 | 0.28 | 0.35 | α̂ |
|---|---|---|---|---|---|
| Abacus, α = 0 | −2522 | −1595 | −819 | +850 | — |
| Abacus, α free | +641 | +379 | +186 | −171 | 1.04 |
| closure, α = 0 | +527 | +175 | +38 | +43 | — |
| closure, α free | +113 | +46 | +15 | −3 | 0.01 |

Freeing α lowers χ² by 7305 at the fiducial and reverses the trend. `sigma8` follows `Omega_m`: at
k < 0.034 the field fixes the amplitude in the band, and `sigma8` is reached through the shape `Omega_m`
sets. The code's bias gave `Omega_m` 0.246 ± 0.006, `sigma8` 0.603 ± 0.036 on Abacus
(`run_20261005_050249_59363213`, after a first attempt that diverged at the same point); a closure whose
truth has α = 1, inferred at α = 0, reproduces that warmup and divergence (`run_20261006_051115_59425114`):
the bias evolution alone causes it. The field level reads the growth ratio across the shell without
sample variance when the evolution is known, which is why the closure `Omega_m` beats the Fisher
(§7.13); freeing α in the Fisher costs 58 % on σ(`Omega_m`) (§7.10).

**The remaining `sigma8` deficit was the particle grid.** With α = 1 fixed, `Omega_m` 0.3185 ± 0.0107
but `sigma8` 0.713 ± 0.047 (−2.1σ, `run_20261006_043937_59424345`). The galaxy likelihood at the true
phases with the field scaled by A (`sigma8` = A × 0.811 for it), biases profiled:

| A | 0.85 | 0.90 | 0.95 | 1.00 | 1.05 |
|---|---|---|---|---|---|
| 140³ particles (evolution grid 1.75) | 59 498 | 59 986 | 60 571 | 61 232 | 61 982 |
| 1LPT | — | 59 941 | — | 61 171 | — |
| **200³ particles (2.5)** | — | 59 702 | — | **59 621** | — |

At 1.75 the likelihood prefers a smaller field (−1246 from A = 1 to 0.9; same with `bnpar` fixed or
1LPT); at 2.5 χ² at the truth drops by 1611 and A = 1 is preferred. The evolution grid sets both the
LPT/bias-product grid and the particle number; this test does not separate the two.

**Both, on Abacus** (α free with prior N(1, 1), 200³ particles; `abacus/abacus_gxyonly_cosmo.yaml`):

| run | box | `Omega_m` | `sigma8` | Mahalanobis | α |
|---|---|---|---|---|---|
| `run_20261006_070354_59428033` (69 batches) | 7500 | 0.3095 ± 0.0121 (−0.47σ) | 0.802 ± 0.053 (−0.19σ) | 0.49 | 1.044 ± 0.043 |
| `run_20261008_050425_59542727` (140 batches) | 5000, non-periodic | 0.3123 ± 0.0119 (−0.25σ) | 0.873 ± 0.053 (+1.16σ) | 1.54 | 1.025 ± 0.043 |

Second halves (box 5000: last 90 batches), split R̂ ≤ 1.037 and ≤ 1.015. Box 5000, the first
inference on a cube cut out of the simulation and treated as periodic: `b1` 0.95 ± 0.12, `b2` 0.52,
`b_{s²}` −0.25, `b∇²` −13 ± 29, `bnpar` −84 ± 31; at 42 batches (`analysis_burn0_allchains/`) the
correlation with the true ICs is ≈ 0.95 in the shell (100 Mpc/h) and the transfer ≈ 1.08, the sampled
`sigma8` over the truth (0.87/0.81). Both reconstruct the field without
the radial tilt of the code's bias (rms ratio flat in radius). Corner of the 7500 run:
`figures/results/abacus_huge_gxyonly_cosmo_corner_run_20261006_070354_59428033.png`.

**`f_NL` with both** (`abacus/abacus_gxyonly.yaml`, cosmology fixed, `run_20261006_152727_59452749`,
137 batches): `f_NL` 2.9 ± 5.0 against 8.9 ± 5.8 at 140³ without α, `b∇²` 13.7 ± 14.2 against 56.2 ±
13.9; σ(`f_NL`) −15 %. The two changes enter together; these runs do not say which one moves `f_NL`.

### 7.15 Lattice template of the κ model — measured

**κ only on Abacus at box 7500 is biased along the degeneracy** (200³ particles, `chi_high_z_max`
3750, ACT; `abacus/abacus_kappaonly_Nl1p0_cosmo{,_ph202}.yaml`, 140 batches):

| run | sky | `Omega_m` | `sigma8` | Mahalanobis |
|---|---|---|---|---|
| `run_20261007_110809_59497681` | `ph201` | 0.236 ± 0.024 | 0.690 ± 0.033 | 3.69 |
| `run_20261008_001508_59533775` | `ph202` | 0.218 ± 0.022 | 0.694 ± 0.034 | 4.5 |

(Earlier: 140³ particles 0.162 ± 0.012 / 0.579 ± 0.025, Mahalanobis 13.4; cell 62.5 0.251 / 0.704,
3.9.) The combination κ constrains is within 1.1σ; the offset lies along the degeneracy. At the true
field the κ likelihood does not pull (the truth is the best point tested), and a first-order
power-spectrum shift predicts neither the Abacus nor the closure posterior: the field-level κ posterior
is not set by the data's power alone.

**Two components.** (1) The `ph201` sky: the true ICs hold 0.963 × the prior's power at k < 0.01 and
the data 0.950 × their expectation at the truth (`ph202` 0.990); the model at the true ICs plus noise
(closure, `run_20261007_093053_59492626`) has the same offset along the constrained combination (−1.8σ
against −1.6σ). (2) The lattice template: a deterministic κ the particle lattice leaves, which the field
level reads linearly.

**The template** (`kappa_lattice_template.py`): E_z[κ] from 8 antithetic pairs per point of a 7 × 6
grid (`Omega_m` 0.20–0.37, `sigma8` 0.65–0.85). At the truth: 1.3 % of the κ power (2.6 % at ℓ 48–64),
S/N = (Σ T²/C)^½ = 5.5 with C = S + N + LOS (4.9 from ℓ 48–64), correlation 0.98 with the κ of the
undisplaced lattice; amplitude ∝ `Omega_m`^1.3 roughly. Projection of the data on it: Abacus −0.04 ±
0.18, model at the true ICs 0.90 ± 0.18. A power spectrum sees T², below the cosmic variance of a band;
the field level sees T over 4221 coherent modes. The Gaussian likelihood with T(θ) as mean reproduces the
MCMC, and predicted `ph202` before it ran:

| data | with T | without T | MCMC |
|---|---|---|---|
| Abacus `ph201` | 0.244 ± 0.025, 0.705 ± 0.035 | 0.324 ± 0.030, 0.786 ± 0.033 | 0.236 ± 0.024, 0.690 ± 0.033 |
| Abacus `ph202` | 0.229 ± 0.020, 0.714 ± 0.030 | 0.289 ± 0.037, 0.781 ± 0.042 | 0.218 ± 0.022, 0.694 ± 0.034 |
| model at the true ICs + noise | 0.321 ± 0.026, 0.785 ± 0.028 | 0.320 ± 0.031, 0.789 ± 0.033 | — |

**Set by the particle spacing d** (`figures/kappa_lattice_template/kappa_template_map_and_spacing.{png,json}`):

| box / cell / d (Mpc/h) | 7500/93.75/37.5 | 7500/93.75/31.2 | 6000/75/30.0 | 5495/94.7/27.5 | 7500/93.75/26.8 | 5000/92.6/25.0 | 7500/93.75/23.4 | 5000/62.5/20.8 | 5000/62.5/17.9 |
|---|---|---|---|---|---|---|---|---|---|
| S/N, ℓ ≤ 64 | 5.46 | 3.59 | 2.72 | 1.84 | 2.27 | 1.31 | 1.64 | 0.99 | 0.74 |

The S/N falls smoothly with d, whatever the box or cell. These are with the ACT `N_ℓ` in C; with the SO
baseline substituted (same template grids) they rise by 18 %: 5.46 → 6.47 at d = 37.5, 1.30 → 1.54 at box
5000 — the band is signal-dominated, so the noise moves the template's weight little. Its ratio to the undisplaced lattice follows
exp[−(2π/d)²σ²/2] with σ = 4.8 Mpc/h, larger than the linear displacement rms at the shells (1.5–3.7
Mpc/h): the form fits, the size is not explained. Cost of a gradient on an A100 at box 7500: κ only
0.125 / 0.191 / 0.286 / 0.391 s at evolution grid 2.5 / 3.0 / 3.5 / 4.0, joint 0.28 s at 2.5 and
0.45 s at 3.0, out of memory beyond — hence the 5000 box: d = 25 at the inference cell with smaller
meshes (final 54³, init 80³) (`bench_gradient.py`).

**The template is information.** Its Fisher matrix, `F_T = Σ ∂T/∂θ_a ∂T/∂θ_b / C` (derivatives by
central differences on the template grid at the truth, ACT `N_ℓ`), added to the Limber κ Fisher with the
run's priors (`~/claude_scratch/bias_evol/fisher_template/ftemplate_combo.py`), σ(`Omega_m`) / σ(`sigma8`):

| | Limber κ | Limber κ + template | measured κ only |
|---|---|---|---|
| box 7500 | 0.0599 / 0.0453 | 0.0373 / 0.0312 | closure 0.0255 / 0.031, Abacus `ph201` 0.024 / 0.033 |
| box 5000 | 0.0646 / 0.0345 | 0.0591 / 0.0331 | Abacus 0.066 / 0.040 (ACT, below) |

At box 7500 the template accounts for the κ-only `sigma8` width and for most of the `Omega_m` one (the
rest within the linearisation of a template ∝ `Omega_m`^1.3 over the grid step; the Gaussian likelihood
with T(θ) above gives 0.025): the κ-only widths narrower than the Fisher of §7.13 are the lattice's
information, real in closure (the template is in the data) and spurious on a real sky. At box 5000 the
template adds little and the measured width is the Fisher's.

**Box 5000.** Prediction from the template grid at box 5000: `ph201` 0.315 ± 0.035, 0.764 ± 0.030
(with or without T alike), `ph202` 0.275 ± 0.038, 0.794 ± 0.028. κ only on `ph201`, ACT
(`abacus/abacus_kappaonly_Nl1p0_cosmo_box5000.yaml`, `run_20261008_050011_59542668`, 140 batches,
R̂ ≤ 1.004): `Omega_m` 0.351 ± 0.066, `sigma8` 0.769 ± 0.040, Mahalanobis 2.2; its κ reconstruction
follows the expectations of a correct model. IC test at box 5000 (`kappa_from_abacus_ic_box5000_cell92p59.{png,npz}`),
(error − covariance)/`N_ℓ` in the six bins +0.143 −0.019 +0.024 −0.005 −0.035 −0.032: inside the
covariance from ℓ = 12; at ℓ 2–11 an excess (0.08 at box 5495, 0 at 7500), cause not established
(§8).

**The joint at box 7500** (`abacus/abacus_joint_Nl1p0_cosmo.yaml`, `run_20261007_235050_59532519`, 40
batches): `Omega_m` 0.3052 ± 0.0095, `sigma8` 0.784 ± 0.018, −0.4σ of the galaxies on `Omega_m`,
σ(`sigma8`) −65 %.

Figures: `figures/kappa_lattice_template/kappa_only_template_contours.{png,json}` and
`kappa_template_map_and_spacing.{png,json}` (`plot_lattice_template.py`); grids
`data/cache/template/box{7500,5000}_ptcl200_grid.npz`, one `geometry_*.json` per row of the spacing
table, `box5000_ptcl200_grid_prediction.json`.

### 7.16 Galaxy likelihood against the particle and paint grids — measured

The amplitude profile of §7.14 (true phases, field × A, biases with `b1_alpha` profiled) at other
grids (`galaxy_likelihood_profile.py --box --mesh --particles`; paint grid 1.75 × the final mesh),
χ²(A) − χ²(1) at A = 0.90 and 1.05:

| final mesh | paint | particles | box | 0.90 | 1.05 |
|---|---|---|---|---|---|
| 80 | 140 | 200 | 7500 | +85 | +62 |
| 54 | 94 | 200 | 5000 | +63 | +58 |
| 54 | 94 | 200 | 5495 | +35 | +44 |
| 58 | 102 | 200 | 5000 | −165 | +207 |
| 58 | 102 | 200 | 5495 | −241 | +206 |
| 58 | 102 | 196 | 5000 | −137 | +201 |
| 64 | 112 | 200 | 5990 | +1048 | −435 |
| 58 | 102 | 204 | 5000 | −1217 | +763 |
| 54 | 94 | 188 | 5000 | −1105 | +670 |
| 64 | 112 | 224 | 5990 | −1029 | +632 |
| 80 | 140 | 280 | 7500 | −1029 | +629 |

(The 200-particle rows from `~/claude_scratch/bias_evol/boxscan/subbox.py`, the same estimator; the
others `figures/galaxy_likelihood_profile/abacus_gxyonly_cosmo_amplitude_m*p*.json`.) The result
follows the ratio of the particle and paint grids, not the box or the cell: 196 and 204 particles at
the same mesh differ by an order of magnitude, and a particle grid exactly twice the paint grid
prefers a field at least 10 % too small at every mesh. The production grids (meshes 54 and 80 with
200³ particles) put the minimum near A ≈ 0.97–0.98, and the box-5000 galaxy run is unbiased (§7.14).
Mechanism not established.

## 8. Remaining steps before the paper

Runs: a 4 h interactive `salloc` with a bare `run_inference.py` (`docs/hpc.md`), two at a time. The
paper's configuration: the central 5000 Mpc/h cube, cell 92.6 (mesh 54), 200³ particles (lattice
template S/N 1.3, §7.15), treated as periodic like real data (to state in the paper), SO baseline
noise (§3.2).

1. **Abacus `Omega_m`–`sigma8` triplet at box 5000.** Galaxies: done (§7.14,
   `abacus/abacus_gxyonly_cosmo_box5000.yaml`). κ only at SO noise:
   `abacus/abacus_kappaonly_so_cosmo_box5000.yaml` (the ACT run of §7.15 is the record), then `…_ph202`.
   Joint: `abacus/abacus_joint_so_cosmo_box5000.yaml` (watch the warmup: κ is three times more
   constraining at SO noise). Then `compare_runs.py` of the three, the paired gains against the
   Fisher of §7.10, the field and κ reconstructions.
2. **To understand.** (a) The galaxy likelihood against the particle/paint grid ratio (§7.16), and
   whether a glass Lagrangian start (no lattice, same particle count) removes it and the κ template
   (bench by monkeypatching `regular_pos`; it would replace the box reduction). (b) The κ excess at
   ℓ 2–11 at box 5000 (§7.15): non-periodic crop (fix: a buffer between the shells and the box faces)
   or the Limber line-of-sight term of the 2500–3750 slab at low ℓ (fix: a non-Limber term); a
   decomposition test is written (`~/claude_scratch/bias_evol/crop/crop_test.py`). (c) The galaxy `Omega_m`
   width narrower than the Fisher (§7.13; at box 5000 0.0119 against 0.0288 with `b1_alpha` free, §7.10):
   the closure with α free; whether the grid effect of §7.16 also adds spurious information. The κ-only
   widths are explained by the lattice template (§7.15).
3. **Other configs to box 5000 and SO noise.** The `f_NL` pair (`abacus/abacus_gxyonly.yaml`,
   `abacus/abacus_joint_Nl1p0.yaml`) and its field reconstruction (§7.1); the κ-only validations of
   §7.4 (`abacus_kappaonly_Nl{1p0,0p1}_fnlfixed`); the closure triplet (§7.13) and the density-scan
   closures (`scan/`, §7.3).
4. **Two-point comparison** (paper `sec:res_twopt`, C. Payerne: C_ℓ^κκ with a CCL theory on the same
   map). `scripts/export_kappa_map.py`, to write: from the κ-only closure `truth.npz`, the observed map
   (`unpack_to_map`, nside 32, ℓ ≤ 64, FITS), `N_ℓ`, the bilinear window w_ℓ, the shells' radial window
   (`kappa_radial_window`), the per-shell taper with the shell radii, the source redshift and the
   conventions. The closure `kappa_obs` is the cut Born κ from `chi_matter_min` to the box edge plus a
   Gaussian draw of variance `N_ℓ + w_ℓ² C_ℓ^LOS`; the two-point theory is a Limber with the shells'
   window and the ℓ-dependent radial cut, times w_ℓ². Protocol fixed before either analysis runs: line
   of sight as noise, same priors and linear spectrum, the two-point contour produced first.
5. **Code.** `fisher.kappa_spectra` still splits the far shell and the cut and has no tent ramps.
6. **Linear spectrum changed** (ACE emulator, §2.1; `Omega_m` bounds [0.2, 0.55]). No run before it
   resumes (its `config.yaml` carries `lin_pk_table` or the old bounds). At fixed cosmology the Abacus
   spectrum moves by ≤ 0.11 % against the former table, so the fixed-cosmology Abacus records stand;
   the `Omega_m`-free runs moved by up to 1–2 % in shape within the posterior, and every closure moved
   from Eisenstein–Hu (3–5 %). To rerun with the emulator: the galaxy-only box-5000 run of step 1
   (`abacus/abacus_gxyonly_cosmo_box5000.yaml`); the κ-only and joint box-5000 runs of step 1 and
   every config of step 3 get it by construction.
7. **Moriond abstract**: run numbers only (§7.13–7.15 and step 1).

Open, without a run planned: the size of the template's damping (§7.15); the lensing prefactor uses
`Omega_m` where AbacusLensing uses `Ω_cb` (§4); the ≈ 1 `P_shot` excess of the Abacus `P_gg` (§7.8,
`gxy_stoch_noise: true` would settle it); a closure on the Abacus ICs (a `closure_init_from_abacus_ic`
knob) to tell model error from realisation; the closure triplet at the depth of a real map
(`chi_matter_min: 0`, `chi_high_z_max` unset).

**To rerun at SO noise** (every κ number of §7 used ACT DR6). Runs: the box-5000 κ only (`ph201`,
`ph202`) and joint (step 1); the `f_NL` pair, the κ-only validations and the closures as they move to
box 5000 (step 3) — galaxy-only runs do not depend on the noise. Figures and numbers that carry
`N_ℓ`: the IC tests' error in units of `N_ℓ` (§7.12, §7.15: `validate_kappa_from_ic.py` on a box-5000
SO config), `plot_lensing_fraction.py`, the density-scan Fisher (`fisher.NELL_FILE` is now SO), the
§7.10 box-7500 Fisher figures (record), the two-point comparison material (step 4).

**Figures to regenerate** (made with the abandoned inner cut-off `chi_min`, or at box 7500 with ACT;
paper section in brackets):

| figure (`figures/…`) | made by | replacement |
|---|---|---|
| `results/closure_d{1p00,0p03}_gxyVSjoint_fNL_run_….png` (06) | `compare_runs.py` | step 3 closure joints |
| `results/density_scan.png` (06) | `density_scan.py` | after step 3 |
| `results/reconstruction_noise_ladder_250.png` (06) | script not found | the κ noise ladder again, or drop |
| `maps/maps2d_abacus_huge_chimin700.png` (04) | `plot_2D_maps.py` | on a box-5000 config |
| `fisher_diagnostic/fisher_cosmo{,_contours,_resolution,_b1alpha}.*` (§7.10) | `fisher_cosmo.py` | box-5000 SO figures exist; keep the 7500 ones as the record of §7.13 |
| `spectra_diagnostic/cl_closure_*` (05) | `quick_cl_spectra.py` | `validation/closure.yaml`, 20 realisations, seed 77 (GPU) |

Removed from the tree and kept in the git history: the κ configs and figures of the `chi_min` set-up
(`configs/inference/archive/`, suffix `_chimin<value>`) — `git log --diff-filter=D --name-only -- <path>`
gives the commit, `git show <commit>^:<path>` the file; their code is the tag `pre-shell-cut`.

---

## 9. Design history

What was tried and replaced, in order, with the commit that introduced the current way. Mechanisms
are in §2–§5, measurements in §7.

- **Starting point** (Oct.–Nov. 2025, `497533e`, `2f6b0d0`): H. Simon-Onfroy's benchmark-field-level
  / montecosmo forward model and MCLMC sampler on synthetic galaxies, then a joint κ channel.
- **Flat sky → curved sky.** The first κ model projected onto a flat-sky tangent plane with a field
  size and pixel count set from the box (`c5b45b6`); the galaxies went to a curved-sky light cone in
  `4f96ff5` (Feb. 2026) and κ to the curved-sky Born projector on HEALPix pixels in `dbc760e`
  (July 2026), which the full-sky HUGE map requires.
- **CMB noise**: Planck `N_ℓ` (`9c17510`), then ACT DR6 (`9f78efa`), then the Simons Observatory
  baseline for the paper (§3.2).
- **Line of sight**: the unmodelled depth as a Gaussian covariance term (`1206a30`), its
  cosmology-dependent modes (`94c6e0f`) and the `ln C` term of the likelihood (`1421d32`); later
  the power the per-shell cut removes and the tents' end ramps in one kernel, and the map's true depth
  3750 Mpc/h instead of its last source plane 3942 (§3.2).
- **Aliasing**: a k-filter at 0.85 × Nyquist (`9f6309a`) replaced by montecosmo's oversampled grids
  (init, evolution, particles, paint) with interlaced deconvolved painting (`c3ffe91`, `dbc760e`, §2.5).
- **Galaxy model**: PNG with or without universality (`png_type: fNL_bias` turned out not
  identifiable, `fNL` kept, §2.2), `bnpar` as a velocity term, free per-bin `ngbars` (`dbc760e`); the
  counts likelihood instead of 1+δ (`91fcdcd`: runs before it are not comparable); the redshift
  evolution of the linear bias, `b1_alpha` (`b40ee86`, §7.14).
- **Sampling the joint**: κ divided the joint step size by ≈ 30. A scalar preconditioner was tried
  and removed once `chi_min` and the radial tents (`91fcdcd`) made the joint sample like the galaxies
  (§7.7); `chi_min` was then replaced by the per-shell multipole cut (`96b1cfc`, §2.6).
- **Projector artefacts**: sub-pixel shot noise folded into the band, fixed by `proj_oversamp`
  (`91fcdcd`); the polar-cap excess of the bilinear weights, fixed by `bilinear_weight_norm`
  (`8a27acb`, §7.9); the particle-lattice template, fixed by the particle spacing (200³ particles in
  the 5000 Mpc/h box, §7.15).
- **Abacus data**: the observation mode (`c3ffe91`), the base-box octant then the full-sky HUGE box
  (`91fcdcd`), the map read through the projector's own kernel (§4), the true ICs cropped to the
  analysis box (§4).
- **Cosmology**: jax_cosmo ODE callbacks exhausted the compiler's memory with `Omega_m` sampled,
  replaced by the background emulator (`dbc760e`) and no callback in the sampled graph (`28d74cc`);
  the CLASS shape of the Abacus ICs through `lin_pk_table` (`2cc4bb5`), Eisenstein–Hu times a
  fiducial CLASS/EH ratio, replaced by the ACE emulator everywhere (§2.1, §7.11); 200³ particles against the
  galaxy `sigma8` deficit (`b40ee86`, §7.14).
