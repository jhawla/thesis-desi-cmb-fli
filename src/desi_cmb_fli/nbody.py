import jax_cosmo as jc
import numpy as np
from diffrax import Euler, ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve
from jax import debug, lax, tree
from jax import numpy as jnp
from jax_cosmo import Cosmology
from jaxpm.kernels import longrange_kernel
from jaxpm.painting import cic_paint, cic_read

from desi_cmb_fli.utils import ch2rshape, safe_div

ABACUS_OMEGA_B = 0.04930169
ABACUS_H = 0.6736
ABACUS_N_S = 0.9649
ABACUS_SIGMA8 = 0.811355
_EMULATOR = None

class BackgroundEmulator:
    """
    Bilinear emulator for cosmological background quantities (growth, distances).
    This completely bypasses the jax_cosmo ODE solvers and CPU callbacks during
    JAX compilation, preventing LLVM from running out of memory (OOM) on large MCMC runs.
    """
    def __init__(self, Om_min=0.05, Om_max=0.7, n_Om=100):
        # 1. Define the grid boundaries for Omega_m
        self.Om_grid = np.linspace(Om_min, Om_max, n_Om)

        # 2. Setup fiducial AbacusSummit cosmology to extract the standard scale factor array
        fid_cosmo = jc.Cosmology(
            Omega_c=0.315192 - ABACUS_OMEGA_B, Omega_b=ABACUS_OMEGA_B,
            Omega_k=0.0, h=ABACUS_H, n_s=ABACUS_N_S, sigma8=ABACUS_SIGMA8, w0=-1.0, wa=0.0
        )
        atab, _, _, _, _, _, _ = jc.background._compute_growth_tables(fid_cosmo)
        self.a_grid_growth = np.array(atab)
        self.a_grid_chi = np.logspace(-4, 0, 512)

        # 3. Initialize empty NumPy arrays for precomputation
        g_grid = np.zeros((n_Om, len(self.a_grid_growth)))
        f_grid = np.zeros((n_Om, len(self.a_grid_growth)))
        g2_grid = np.zeros((n_Om, len(self.a_grid_growth)))
        f2_grid = np.zeros((n_Om, len(self.a_grid_growth)))
        chi_grid = np.zeros((n_Om, len(self.a_grid_chi)))

        print("[Emulator] Precomputing background ODE grids...")

        # 4. Fill the grids using jax_cosmo (runs on CPU before any JIT compilation)
        for i, Om in enumerate(self.Om_grid):
            c = jc.Cosmology(
                Omega_c=float(Om) - ABACUS_OMEGA_B, Omega_b=ABACUS_OMEGA_B,
                Omega_k=0.0, h=ABACUS_H, n_s=ABACUS_N_S, sigma8=ABACUS_SIGMA8, w0=-1.0, wa=0.0
            )
            _, gtab, ftab, _, g2tab, f2tab, _ = jc.background._compute_growth_tables(c)
            g_grid[i, :] = gtab
            f_grid[i, :] = ftab
            g2_grid[i, :] = g2tab
            f2_grid[i, :] = f2tab
            chi_grid[i, :] = jc.background.radial_comoving_distance(c, self.a_grid_chi)

        # 5. Convert everything to JAX arrays for fast inference
        self.g_grid = jnp.array(g_grid)
        self.f_grid = jnp.array(f_grid)
        self.g2_grid = jnp.array(g2_grid)
        self.f2_grid = jnp.array(f2_grid)
        self.chi_grid = jnp.array(chi_grid)
        self.a_grid_growth_jnp = jnp.array(self.a_grid_growth)
        self.a_grid_chi_jnp = jnp.array(self.a_grid_chi)
        self.Om_grid_jnp = jnp.array(self.Om_grid)

    def _get_Om_blend(self, Om):
        """
        Finds the exact bounding indices and interpolation weights for a given Omega_m.
        We do this manually to avoid using jax.vmap, which creates ghost dimensions
        and crashes jnp.interp.
        """
        # Force Om to be a strict 0D scalar (removes any hidden JAX tracer dimensions)
        Om = jnp.atleast_1d(Om)[0]

        # Find the fractional index of Om inside the grid
        Om_idx = jnp.interp(Om, self.Om_grid_jnp, jnp.arange(len(self.Om_grid_jnp)))

        # Get the integer bounding indices
        idx0 = jnp.floor(Om_idx).astype(int)
        idx1 = jnp.clip(idx0 + 1, 0, len(self.Om_grid_jnp) - 1)

        # Calculate linear interpolation weights
        w1 = Om_idx - idx0
        w0 = 1.0 - w1
        return idx0, idx1, w0, w1

    def get_growth_tables(self, Om):
        # Get interpolation weights for the current Omega_m
        idx0, idx1, w0, w1 = self._get_Om_blend(Om)

        # Extract the 1D arrays from the 2D grid and blend them
        gtab = self.g_grid[idx0] * w0 + self.g_grid[idx1] * w1
        ftab = self.f_grid[idx0] * w0 + self.f_grid[idx1] * w1
        g2tab = self.g2_grid[idx0] * w0 + self.g2_grid[idx1] * w1
        f2tab = self.f2_grid[idx0] * w0 + self.f2_grid[idx1] * w1

        return self.a_grid_growth_jnp, gtab, ftab, g2tab, f2tab

    def a2chi(self, Om, a):
        # Get interpolation weights for the current Omega_m
        idx0, idx1, w0, w1 = self._get_Om_blend(Om)

        # Extract the 1D chi array
        chi_1d = self.chi_grid[idx0] * w0 + self.chi_grid[idx1] * w1

        # Interpolate along the scale factor axis
        return jnp.interp(a, self.a_grid_chi_jnp, chi_1d)

    def chi2a(self, Om, chi):
        # Get interpolation weights for the current Omega_m
        idx0, idx1, w0, w1 = self._get_Om_blend(Om)

        # Extract the 1D chi array
        chi_1d = self.chi_grid[idx0] * w0 + self.chi_grid[idx1] * w1

        return jnp.interp(chi, chi_1d[::-1], self.a_grid_chi_jnp[::-1])

def _get_background_emulator():
    global _EMULATOR
    if _EMULATOR is None:
        _EMULATOR = BackgroundEmulator()
    return _EMULATOR


def _is_abacus_background(cosmo):
    """Return True when the emulator grid matches all fixed background parameters."""
    checks = (
        ("Omega_b", ABACUS_OMEGA_B),
        ("Omega_k", 0.0),
        ("h", ABACUS_H),
        ("n_s", ABACUS_N_S),
        ("w0", -1.0),
        ("wa", 0.0),
    )
    try:
        return all(np.isclose(float(getattr(cosmo, name)), value, rtol=0.0, atol=1e-10)
                   for name, value in checks)
    except (TypeError, ValueError):
        return False



def rfftk(shape):
    """
    Return wavevectors in cell units for rfftn.
    """
    kx = np.fft.fftfreq(shape[0]) * 2 * np.pi
    ky = np.fft.fftfreq(shape[1]) * 2 * np.pi
    kz = np.fft.rfftfreq(shape[2]) * 2 * np.pi

    kx = kx.reshape([-1, 1, 1])
    ky = ky.reshape([1, -1, 1])
    kz = kz.reshape([1, 1, -1])

    return kx, ky, kz


def invlaplace_kernel(kvec, fd=False):
    """
    Compute the inverse Laplace kernel.

    Parameters
    -----------
    kvec: list
        List of wavevectors
    fd: bool
        Finite difference kernel

    Returns
    --------
    weights: array
        Complex kernel values
    """
    if fd:
        kk = sum((ki * np.sinc(ki / (2 * np.pi))) ** 2 for ki in kvec)
    else:
        kk = sum(ki**2 for ki in kvec)
    return -safe_div(1, kk)


def gradient_kernel(kvec, direction: int, fd=False):
    """
    Compute the gradient kernel in the given direction

    Parameters
    -----------
    kvec: list
        List of wavevectors
    direction: int
        Index of the direction in which to take the gradient
    fd: bool
        Finite difference kernel

    Returns
    --------
    weights: array
        Complex kernel values
    """
    ki = kvec[direction]
    if fd:
        ki = (8.0 * np.sin(ki) - np.sin(2.0 * ki)) / 6.0
    return 1j * ki


def paint_kernel(kvec, order: int = 2):
    """
    Compute painting kernel of given order.

    Parameters
    ----------
    kvec: list
        List of wavevectors
    order: int
        order of the kernel
        * 0: Dirac
        * 1: Nearest Grid Point (NGP)
        * 2: Cloud-In-Cell (CIC)
        * 3: Triangular-Shape Cloud (TSC)
        * 4: Piecewise-Cubic Spline (PCS)

        cf. [List and Hahn, 2024](https://arxiv.org/abs/2309.10865)

    Returns
    -------
    weights: array
        Complex kernel values
    """
    wts = [np.sinc(kvec[i] / (2 * np.pi)) for i in range(3)]
    wts = (wts[0] * wts[1] * wts[2]) ** order
    return wts


def pm_forces(pos, mesh_shape, mesh=None, grad_fd=False, lap_fd=False, r_split=0):
    """
    Compute gravitational forces on particles using a PM scheme
    """
    if mesh is None:
        delta_k = jnp.fft.rfftn(cic_paint(jnp.zeros(mesh_shape), pos))
    # elif jnp.isrealobj(mesh):
    #     delta_k = jnp.fft.rfftn(mesh)
    else:
        delta_k = mesh

    # Compute gravitational potential
    kvec = rfftk(mesh_shape)
    pot_k = delta_k * invlaplace_kernel(kvec, lap_fd) * longrange_kernel(kvec, r_split=r_split)

    # # If painted field, double deconvolution to account for both painting and reading
    # if mesh is None:
    #     print("deconv")
    #     pot_k /= paint_kernel(kvec, order=2)**2

    # Compute gravitational forces
    return jnp.stack(
        [
            cic_read(jnp.fft.irfftn(-gradient_kernel(kvec, i, grad_fd) * pot_k), pos)
            for i in range(3)
        ],
        axis=-1,
    )


def pm_forces2(delta_k, pos, mesh_shape, lap_fd=False, grad_fd=False):
    """
    Return 2LPT source term.
    """
    kvec = rfftk(mesh_shape)
    pot_k = delta_k * invlaplace_kernel(kvec, lap_fd)

    delta2 = 0
    shear_acc = 0
    for i in range(3):
        # Add products of diagonal terms = 0 + s11*s00 + s22*(s11+s00)...
        shear_ii = gradient_kernel(kvec, i, grad_fd) ** 2
        shear_ii = jnp.fft.irfftn(shear_ii * pot_k)
        delta2 += shear_ii * shear_acc
        shear_acc += shear_ii

        for j in range(i + 1, 3):
            # Substract squared strict-up-triangle terms
            hess_ij = gradient_kernel(kvec, i, grad_fd) * gradient_kernel(kvec, j, grad_fd)
            delta2 -= jnp.fft.irfftn(hess_ij * pot_k) ** 2

    force2 = pm_forces(pos, mesh_shape, mesh=jnp.fft.rfftn(delta2), grad_fd=grad_fd, lap_fd=lap_fd)
    return force2


def lpt(cosmo: Cosmology, init_mesh, pos, a, order=2, grad_fd=False, lap_fd=False):
    """
    Compute first and second order LPT displacement,
    e.g. Eq 3.5 and 3.7 [List and Hahn](https://arxiv.org/abs/2409.19049)
    or Eq. 2 and 3 [Jenkins2010](https://arxiv.org/pdf/0910.0258)
    """
    # if jnp.isrealobj(init_mesh):
    #     delta_k = jnp.fft.rfftn(init_mesh)
    #     mesh_shape = init_mesh.shape
    # else:
    delta_k = init_mesh
    mesh_shape = ch2rshape(init_mesh.shape)

    force1 = pm_forces(pos, mesh_shape, mesh=delta_k, grad_fd=grad_fd, lap_fd=lap_fd)
    growth = a2g(cosmo, a)
    if jnp.ndim(growth) == 1:
        growth = growth[:, None]
    dpos = growth * force1
    vel = force1

    if order == 2:
        force2 = pm_forces2(delta_k, pos, mesh_shape, grad_fd=grad_fd, lap_fd=lap_fd)
        growth2 = a2gg(cosmo, a)
        if jnp.ndim(growth2) == 1:
            growth2 = growth2[:, None]
        dpos -= growth2 * force2

        dggdg = a2dggdg(cosmo, a)
        if jnp.ndim(dggdg) == 1:
            dggdg = dggdg[:, None]
        vel -= dggdg * force2

    return dpos, vel


###########
# Growths #
###########

def _get_growth_tables(cosmo):
    """Return growth tables, using the emulator only for its calibrated background."""
    if _is_abacus_background(cosmo):
        Om = cosmo.Omega_c + cosmo.Omega_b
        return _get_background_emulator().get_growth_tables(Om)
    atab, gtab, ftab, _, g2tab, f2tab, _ = jc.background._compute_growth_tables(cosmo)
    return atab, gtab, ftab, g2tab, f2tab


# Growth from scale factor
def a2g(cosmo, a):
    atab, gtab, *_ = _get_growth_tables(cosmo)
    return jnp.interp(a, atab, gtab)


def a2gg(cosmo, a):
    atab, gtab, _, g2tab, _ = _get_growth_tables(cosmo)
    # gg = -3/7 * D2 (second-order growth for 2LPT)
    return jnp.interp(a, atab, g2tab) * -3 / 7


def a2f(cosmo, a):
    atab, _, ftab, _, _ = _get_growth_tables(cosmo)
    return jnp.interp(a, atab, ftab)


def a2ff(cosmo, a):
    atab, _, _, _, f2tab = _get_growth_tables(cosmo)
    return jnp.interp(a, atab, f2tab)


def a2dggdg(cosmo, a):
    g, gg, f, ff = a2g(cosmo, a), a2gg(cosmo, a), a2f(cosmo, a), a2ff(cosmo, a)
    return safe_div(gg * ff, g * f)  # NOTE: dggdg(0) = 0


# Growth from growth factor (inverse lookups via table)
def g2a(cosmo, g):
    atab, gtab, *_ = _get_growth_tables(cosmo)
    return jnp.interp(g, gtab, atab)


def g2gg(cosmo, g):
    atab, gtab, _, g2tab, _ = _get_growth_tables(cosmo)
    return jnp.interp(g, gtab, g2tab) * -3 / 7


def g2f(cosmo, g):
    atab, gtab, ftab, _, _ = _get_growth_tables(cosmo)
    return jnp.interp(g, gtab, ftab)


def g2ff(cosmo, g):
    atab, gtab, _, _, f2tab = _get_growth_tables(cosmo)
    return jnp.interp(g, gtab, f2tab)


def g2dggdg(cosmo, g):
    gg, f, ff = g2gg(cosmo, g), g2f(cosmo, g), g2ff(cosmo, g)
    return safe_div(gg * ff, g * f)  # NOTE: dggdg(0) = 0


#############
# Distances #
#############
def a2chi(cosmo, a):
    """Radial comoving distance in Mpc/h for a given scale factor."""
    a = jnp.asarray(a)
    if _is_abacus_background(cosmo):
        Om = cosmo.Omega_c + cosmo.Omega_b
        res = _get_background_emulator().a2chi(Om, a.reshape(-1))
        return res.reshape(a.shape)
    # General cosmology: fall back to the h-aware jax_cosmo background.
    chi = jc.background.radial_comoving_distance(cosmo, a.reshape(-1))
    return chi.reshape(a.shape)


def chi2a(cosmo, chi):
    """Scale factor for a given radial comoving distance in Mpc/h."""
    chi = jnp.asarray(chi)
    original_shape = chi.shape
    if _is_abacus_background(cosmo):
        Om = cosmo.Omega_c + cosmo.Omega_b
        res = _get_background_emulator().chi2a(Om, chi.reshape(-1))
        return res.reshape(original_shape)
    # General cosmology: fall back to the h-aware jax_cosmo background.
    a = jc.background.a_of_chi(cosmo, chi.reshape(-1))
    return a.reshape(original_shape)


###########
# Solvers #
###########
def bullfrog_vf(cosmo: Cosmology, dg, mesh_shape, grad_fd=False, lap_fd=False):
    """
    BullFrog vector field.
    """

    def alpha_bf(cosmo, g0, dg):
        """
        BullFrog growth-time integrator coefficient.

        See Eq. 2.3 in [List and Hahn, 2024](https://arxiv.org/abs/2106.00461)
        """
        g1 = g0 + dg / 2
        g2 = g0 + dg

        dggdg0, dggdg2 = g2dggdg(cosmo, g0), g2dggdg(cosmo, g2)
        lin_ratio = (g2gg(cosmo, g0) + dggdg0 * dg / 2) / g1 - g1
        # NOTE: linearization of ratio (gg - g^2)/g aroung g0, evaluated at g1
        return (dggdg2 - lin_ratio) / (dggdg0 - lin_ratio)

    def alpha_fpm(cosmo, g0, dg):
        """
        FastPM growth-time integrator coefficient.

        See Eq. 3.16 in [List and Hahn, 2024](https://arxiv.org/abs/2106.00461)
        """
        g2 = g0 + dg
        a0, a2 = g2a(cosmo, g0), g2a(cosmo, g2)
        coeff0 = jc.background.Esqr(cosmo, a0) ** 0.5 * g0 * g2f(cosmo, g0) * a0**2
        coeff2 = jc.background.Esqr(cosmo, a2) ** 0.5 * g2 * g2f(cosmo, g2) * a2**2
        return coeff0 / coeff2

    def kick(state, g0, cosmo, dg):
        pos, vel = state
        g1 = g0 + dg / 2
        forces = pm_forces(pos, mesh_shape, grad_fd=grad_fd, lap_fd=lap_fd)
        alpha = alpha_bf(cosmo, g0, dg)
        return pos, alpha * vel + (1 - alpha) * forces / g1
        # return pos, vel + (1 - alpha) * (forces / g1 - vel) # equivalent
        # return pos, vel + dg * forces

    def drift(state, dg):
        pos, vel = state
        return pos + vel * dg, vel

    def vector_field(g0, state, args):
        old = state
        state = drift(state, dg / 2)
        state = kick(state, g0, cosmo, dg)
        state = drift(state, dg / 2)
        return tree.map(lambda new, old: (new - old) / dg, state, old)

    # def step(state, g0):
    #     state = drift(state, dg / 2)
    #     state = kick(state, g0, cosmo, dg)
    #     state = drift(state, dg / 2)
    #     return state, None

    return vector_field
    # return step


def nbody_bf(
    cosmo: Cosmology,
    init_mesh,
    pos,
    a,
    n_steps=5,
    grad_fd=False,
    lap_fd=False,
    snapshots: int | list = None,
):
    """
    N-body simulation with BullFrog solver.
    """
    n_steps = int(n_steps)
    g = a2g(cosmo, a)
    dg = g / n_steps

    mesh_shape = ch2rshape(init_mesh.shape)
    terms = ODETerm(bullfrog_vf(cosmo, dg, mesh_shape, grad_fd=grad_fd, lap_fd=lap_fd))
    solver = Euler()

    vel = pm_forces(pos, mesh_shape, mesh=init_mesh, grad_fd=grad_fd, lap_fd=lap_fd)
    state = pos, vel

    if snapshots is None or (isinstance(snapshots, int) and snapshots < 2):
        saveat = SaveAt(t1=True)
    elif isinstance(snapshots, int):
        saveat = SaveAt(ts=a2g(cosmo, jnp.linspace(0, a, snapshots)))
    else:
        saveat = SaveAt(ts=a2g(cosmo, jnp.asarray(snapshots)))

    sol = diffeqsolve(
        terms, solver, 0.0, g, dt0=dg, y0=state, max_steps=n_steps, saveat=saveat
    )  # cosmo as args may leak
    states = sol.ys
    # debug.print("bullfrog n_steps: {n}", n=sol.stats['num_steps'])
    return states


def nbody_bf_scan(
    cosmo: Cosmology,
    init_mesh,
    pos,
    a,
    n_steps=5,
    grad_fd=False,
    lap_fd=False,
    snapshots: int | list = None,
):
    """
    No-diffrax version of N-body simulation with BullFrog solver.
    Simpler but does not optimize for memory usage with binomial checkpointing.
    """
    g = a2g(cosmo, a)
    dg = g / n_steps
    gs = jnp.arange(n_steps) * dg

    mesh_shape = ch2rshape(init_mesh.shape)
    # vector_field = bullfrog_vf(cosmo, dg, mesh_shape, grad_fd=grad_fd, lap_fd=lap_fd)

    # def step(state, g0):
    #     vf = vector_field(g0, state, None)
    #     state = tree.map(lambda x, y: x + dg * y, state, vf)
    #     return state, None

    step = bullfrog_vf(cosmo, dg, mesh_shape, grad_fd=grad_fd, lap_fd=lap_fd)

    vel = pm_forces(pos, mesh_shape, mesh=init_mesh, grad_fd=grad_fd, lap_fd=lap_fd)
    state = pos, vel

    state, _ = lax.scan(step, state, gs)
    return tree.map(lambda x: x[None], state)


def lpt_fpm(cosmo: Cosmology, init_mesh, pos, a, order=1, grad_fd=True, lap_fd=False):
    """
    Computes first and second order LPT displacement, e.g. Eq. 2 and 3 [Jenkins2010](https://arxiv.org/pdf/0910.0258)
    """
    a = jnp.atleast_1d(a)
    E = jc.background.Esqr(cosmo, a) ** 0.5
    if jnp.isrealobj(init_mesh):
        delta_k = jnp.fft.rfftn(init_mesh)
        mesh_shape = init_mesh.shape
    else:
        delta_k = init_mesh
        mesh_shape = ch2rshape(init_mesh.shape)

    init_force = pm_forces(pos, mesh_shape, mesh=delta_k, grad_fd=grad_fd, lap_fd=lap_fd)
    dq = a2g(cosmo, a) * init_force
    p = a**2 * a2f(cosmo, a) * E * dq

    if order == 2:
        kvec = rfftk(mesh_shape)
        pot_k = delta_k * invlaplace_kernel(kvec, lap_fd)

        delta2 = 0
        shear_acc = 0
        for i in range(3):
            # Add products of diagonal terms = 0 + s11*s00 + s22*(s11+s00)...
            shear_ii = gradient_kernel(kvec, i, grad_fd) ** 2
            shear_ii = jnp.fft.irfftn(shear_ii * pot_k)
            delta2 += shear_ii * shear_acc
            shear_acc += shear_ii

            for j in range(i + 1, 3):
                # Substract squared strict-up-triangle terms
                hess_ij = gradient_kernel(kvec, i, grad_fd) * gradient_kernel(kvec, j, grad_fd)
                delta2 -= jnp.fft.irfftn(hess_ij * pot_k) ** 2

        init_force2 = pm_forces(
            pos, mesh_shape, mesh=jnp.fft.rfftn(delta2), grad_fd=grad_fd, lap_fd=lap_fd
        )
        dq2 = a2gg(cosmo, a) * init_force2  # D2 is renormalized: - D2 = 3/7 * growth_factor_second
        p2 = (a**2 * a2ff(cosmo, a) * E) * dq2

        dq -= dq2
        p -= p2

    return dq, p


def diffrax_vf(cosmo: Cosmology, mesh_shape, grad_fd=True, lap_fd=False):
    """
    N-body ODE vector field for diffrax, e.g. Tsit5 or Dopri5

    vector field signature is (a, state, args) -> dstate, where state is a tuple (position, velocities)
    """

    def vector_field(a, state, args):
        pos, vel = state
        forces = pm_forces(pos, mesh_shape, grad_fd=grad_fd, lap_fd=lap_fd) * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1.0 / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel
        # Computes the update of velocity (kick)
        dvel = 1.0 / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces
        return dpos, dvel

    return vector_field


def jax_ode_vf(cosmo: Cosmology, mesh_shape, grad_fd=True, lap_fd=False):
    """
    Return N-body ODE vector field for jax.experimental.ode.odeint

    vector field signature is (state, a, *args) -> dstate, where state is a tuple (position, velocities)
    """
    vf = diffrax_vf(cosmo, mesh_shape, grad_fd, lap_fd)

    def vector_field(state, a, *args):
        return vf(a, state, args)

    return vector_field


def nbody_tsit5(
    cosmo: Cosmology,
    mesh_shape,
    particles,
    a_lpt,
    a_obs,
    tol=1e-2,
    grad_fd=True,
    lap_fd=False,
    snapshots: int | list = None,
):
    if a_lpt == a_obs:
        return tree.map(lambda x: x[None], particles)
    else:
        terms = ODETerm(diffrax_vf(cosmo, mesh_shape, grad_fd, lap_fd))
        solver = Tsit5()  # Tsit5 usually better than Dopri5
        controller = PIDController(rtol=tol, atol=tol, pcoeff=0.4, icoeff=1, dcoeff=0)

        if snapshots is None or (isinstance(snapshots, int) and snapshots < 2):
            saveat = SaveAt(t1=True)
        elif isinstance(snapshots, int):
            saveat = SaveAt(ts=jnp.linspace(a_lpt, a_obs, snapshots))
        else:
            saveat = SaveAt(ts=jnp.asarray(snapshots))

        sol = diffeqsolve(
            terms,
            solver,
            a_lpt,
            a_obs,
            dt0=None,
            y0=particles,
            stepsize_controller=controller,
            max_steps=1000,
            saveat=saveat,
        )
        # NOTE: if max_steps > 50 for dopri5/tsit5, just quit :')
        particles = sol.ys
        debug.print("tsit5 n_steps: {n}", n=sol.stats["num_steps"])
        return particles
