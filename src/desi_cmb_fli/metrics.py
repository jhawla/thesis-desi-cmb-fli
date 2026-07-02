from functools import partial

import jax.numpy as jnp
import numpy as np

# from blackjax.diagnostics import effective_sample_size
from jax_cosmo import Cosmology
from jaxpm.growth import growth_factor, growth_rate
from numpyro.diagnostics import effective_sample_size, gelman_rubin
from scipy.special import legendre

from desi_cmb_fli.nbody import paint_kernel, rfftk
from desi_cmb_fli.utils import ch2rshape, safe_div


############
# Spectrum #
############
def _waves(mesh_shape, box_shape, kedges, los):
    """
    Parameters
    ----------
    mesh_shape : tuple of int
        Shape of the mesh grid.
    box_shape : tuple of float
        Physical dimensions of the box.
    kedges : None, int, float, or list
        * If None, set dk to twice the minimum.
        * If int, specifies number of edges.
        * If float, specifies dk.
    los : array_like
        Line-of-sight vector.

    Returns
    -------
    kedges : ndarray
        Edges of the bins.
    kmesh : ndarray
        Wavenumber mesh.
    mumesh : ndarray
        Cosine mesh.
    rfftw : ndarray
        RFFT weights accounting for Hermitian symmetry.
    """
    kmax = np.pi * np.min(mesh_shape / box_shape)  # = knyquist

    if isinstance(kedges, type(None) | int | float):
        if kedges is None:
            dk = 2 * np.pi / np.min(box_shape) * 2  # twice the fundamental wavenumber
        if isinstance(kedges, int):
            dk = kmax / kedges  # final number of bins will be kedges-1
        elif isinstance(kedges, float):
            dk = kedges
        kedges = np.arange(0, kmax, dk) + dk / 2  # from dk/2 to kmax-dk/2

    kvec = rfftk(mesh_shape)  # cell units
    kvec = [
        ki * (m / b) for ki, m, b in zip(kvec, mesh_shape, box_shape, strict=False)
    ]  # h/Mpc physical units
    kmesh = sum(ki**2 for ki in kvec) ** 0.5

    if los is None:
        mumesh = 0.0
    else:
        mumesh = sum(ki * losi for ki, losi in zip(kvec, los, strict=False))
        mumesh = safe_div(mumesh, kmesh)

    rfftw = np.full_like(kmesh, 2)
    rfftw[..., 0] = 1
    if mesh_shape[-1] % 2 == 0:
        rfftw[..., -1] = 1

    return kedges, kmesh, mumesh, rfftw


def spectrum(
    mesh,
    mesh2=None,
    box_shape=None,
    kedges: int | float | list = None,
    comp=(0, 0),
    poles=0,
    los: np.ndarray = None,
):
    """
    Compute the auto and cross spectrum of 3D fields, with multipole.
    """
    # Initialize
    mesh_shape = np.array(mesh.shape)
    if box_shape is None:
        box_shape = mesh_shape
    else:
        box_shape = np.asarray(box_shape)

    if los is not None:
        los = np.asarray(los)
        los /= np.linalg.norm(los)
    pls = np.atleast_1d(poles)

    # FFTs and deconvolution
    if isinstance(comp, int):
        comp = (comp, comp)

    mesh = jnp.fft.rfftn(mesh, norm="ortho")
    kvec = rfftk(mesh_shape)  # cell units
    mesh /= paint_kernel(kvec, order=comp[0])

    if mesh2 is None:
        mmk = mesh.real**2 + mesh.imag**2
    else:
        mesh2 = jnp.fft.rfftn(mesh2, norm="ortho")
        mesh2 /= paint_kernel(kvec, order=comp[1])
        mmk = mesh * mesh2.conj()

    # Binning
    kedges, kmesh, mumesh, rfftw = _waves(mesh_shape, box_shape, kedges, los)
    n_bins = len(kedges) + 1
    dig = np.digitize(kmesh.reshape(-1), kedges)

    # Count wavenumber in bins
    kcount = np.bincount(dig, weights=rfftw.reshape(-1), minlength=n_bins)
    kcount = kcount[1:-1]

    # Average wavenumber values in bins
    # kavg = (kedges[1:] + kedges[:-1]) / 2
    kavg = np.bincount(dig, weights=(kmesh * rfftw).reshape(-1), minlength=n_bins)
    kavg = kavg[1:-1] / kcount

    # Average wavenumber power in bins
    pow = jnp.empty((len(pls), n_bins))
    for i_ell, ell in enumerate(pls):
        weights = (mmk * (2 * ell + 1) * legendre(ell)(mumesh) * rfftw).reshape(-1)
        if mesh2 is None:
            psum = jnp.bincount(dig, weights=weights, length=n_bins)
        else:
            # NOTE: bincount is really slow with complex numbers, so bincount the real part.
            psum = jnp.bincount(dig, weights=weights.real, length=n_bins)
        pow = pow.at[i_ell].set(psum)
    pow = pow[:, 1:-1] / kcount * (box_shape / mesh_shape).prod()  # from cell units to [Mpc/h]^3

    # kpow = jnp.concatenate([kavg[None], pk])
    if poles == 0:
        return kavg, pow[0]
    else:
        return kavg, pow


def get_cl_2d(map1, map2=None, field_size_deg=1.0, comp=(0, 0)):
    """
    Compute 2D angular power spectrum C_l directly from FFT modes (no binning).

    Parameters
    ----------
    map1 : array
        First map (Nx, Ny)
    map2 : array, optional
        Second map (Nx, Ny). If None, computes auto-spectrum of map1.
    field_size_deg : float
        Field size in degrees.
    comp : tuple
        Deconvolution order for paint_kernel. Defaults to (0, 0) (No deconvolution).

    Returns
    -------
    ell : array
        Angular wavenumbers (unique values)
    cl : array
        Power spectrum C_l (one value per unique ell)
    """
    field_size_rad = field_size_deg * np.pi / 180.0
    nx, ny = map1.shape

    # Compute FFT
    fft1 = np.fft.fft2(map1)
    if map2 is not None:
        fft2 = np.fft.fft2(map2)
        power_2d = np.real(fft1 * np.conj(fft2))
    else:
        power_2d = np.abs(fft1)**2

    # Compute 2D wavenumbers in terms of ell
    kx = np.fft.fftfreq(nx, d=field_size_rad / (2 * np.pi * nx))
    ky = np.fft.fftfreq(ny, d=field_size_rad / (2 * np.pi * ny))
    kx_2d, ky_2d = np.meshgrid(kx, ky, indexing='ij')
    ell_2d = np.sqrt(kx_2d**2 + ky_2d**2)

    # Normalize: convert to C_ell units
    # Power in Fourier space needs to be normalized by area
    area_sr = (field_size_rad)**2
    cl_2d = power_2d / (nx * ny)**2 * area_sr

    # Flatten and get unique ell values with their corresponding C_ell
    ell_flat = ell_2d.flatten()
    cl_flat = cl_2d.flatten()

    # Round ell to avoid floating point issues
    ell_rounded = np.round(ell_flat, decimals=2)

    # Get unique ell values and average C_ell for each
    unique_ell = np.unique(ell_rounded)
    cl_averaged = np.array([np.mean(cl_flat[ell_rounded == ell_val]) for ell_val in unique_ell])

    # Remove ell=0 (DC mode) and very low ell if needed
    valid_idx = unique_ell > 1.0

    return unique_ell[valid_idx], cl_averaged[valid_idx]


def masked_healpix_to_full(masked_map, mask, fill_value=0.0):
    """Expand a masked HEALPix vector to a full-sky map."""
    mask = np.asarray(mask, dtype=bool)
    full = np.full(mask.size, fill_value, dtype=float)
    full[mask] = np.asarray(masked_map, dtype=float)
    return full


def get_cl_healpix(
    masked_map1,
    mask1,
    masked_map2=None,
    mask2=None,
    lmax=None,
    decouple="fsky",
    coupling_matrix=None,
):
    """Compute masked HEALPix pseudo-C_ell with decoupling.

    Parameters
    ----------
    masked_map1 : array
        Values on the active pixels of ``mask1``.
    mask1 : array-like bool
        Angular mask for the first map.
    masked_map2 : array, optional
        Values on the active pixels of ``mask2``. If None, compute auto-spectrum.
    mask2 : array-like bool, optional
        Angular mask for the second map. Defaults to ``mask1``.
    lmax : int, optional
        Maximum multipole. Defaults to ``3*nside-1`` used by healpy.
    decouple : {"fsky", "master", "none"}
        ``"fsky"``   — divide pseudo-Cl by mean(mask1 * mask2) (diagonal MASTER approximation).
        ``"master"`` — full MASTER decoupling: solve M_ll @ x = Cl_pseudo via
                       least-squares (requires ``coupling_matrix``).
        ``"none"``   — return raw pseudo-Cl without correction.
    coupling_matrix : array (lmax+1, lmax+1), optional
        MASTER mode-coupling matrix M_ll. Required when ``decouple="master"``.
        Typically ``model.cmb_M_ll`` (precomputed by ``FieldLevelModel`` via NaMaster).

    Returns
    -------
    ell : ndarray
    cl : ndarray
    info : dict
        Contains ``cl_pseudo``, ``norm``, and (for ``decouple="master"``) ``lstsq_rank``.
    """
    import healpy as hp

    mask1 = np.asarray(mask1, dtype=bool)
    if mask2 is None:
        mask2 = mask1
    else:
        mask2 = np.asarray(mask2, dtype=bool)

    if mask1.size != mask2.size:
        raise ValueError("mask1 and mask2 must have the same full-sky length")

    masked_map1 = np.asarray(masked_map1, dtype=float)
    masked_map1 = masked_map1 - np.mean(masked_map1)
    full1 = masked_healpix_to_full(masked_map1, mask1, fill_value=0.0)
    if masked_map2 is None:
        full2 = full1
    else:
        masked_map2 = np.asarray(masked_map2, dtype=float)
        masked_map2 = masked_map2 - np.mean(masked_map2)
        full2 = masked_healpix_to_full(masked_map2, mask2, fill_value=0.0)

    cl_pseudo = hp.anafast(full1, full2, lmax=lmax)
    norm = float(np.mean(mask1.astype(float) * mask2.astype(float)))
    norm = max(norm, 1e-12)
    info_extra = {}

    if decouple == "fsky":
        cl = cl_pseudo / norm
    elif decouple == "master":
        if coupling_matrix is None:
            raise ValueError(
                "coupling_matrix (M_ll) must be provided when decouple='master'."
            )
        M = np.asarray(coupling_matrix, dtype=float)
        n = len(cl_pseudo)
        if M.shape[0] != n or M.shape[1] != n:
            # Truncate/restrict to the range covered by cl_pseudo
            n_use = min(n, M.shape[0], M.shape[1])
            cl_full = np.zeros(n)
            cl_decoupled, _, rank, _ = np.linalg.lstsq(
                M[:n_use, :n_use], cl_pseudo[:n_use], rcond=None
            )
            cl_full[:n_use] = cl_decoupled
            cl = cl_full
        else:
            cl, _, rank, _ = np.linalg.lstsq(M, cl_pseudo, rcond=None)
            info_extra["lstsq_rank"] = int(rank)
    elif decouple == "none":
        cl = cl_pseudo.copy()
    else:
        raise ValueError(f"Unknown decouple mode: {decouple!r}")

    ell = np.arange(len(cl))
    valid = ell >= 2
    return ell[valid], np.asarray(cl)[valid], {
        "cl_pseudo": np.asarray(cl_pseudo)[valid],
        "norm": norm,
        **info_extra,
    }


def bin_cl_log(ell, cl, n_bins_per_decade=15, ell_min=None, ell_max=None):
    """
    Logarithmically bin an angular power spectrum.

    Parameters
    ----------
    ell : array
        Angular wavenumbers (1-D, from get_cl_2d).
    cl : array
        Power spectrum values.
    n_bins_per_decade : int
        Number of bins per decade of ell (default 15).
    ell_min, ell_max : float or None
        Bin range.  Defaults to the data range.

    Returns
    -------
    ell_binned : array
        Geometric centre of each bin.
    cl_binned : array
        Mean Cl in each bin.
    n_modes : array (int)
        Number of modes averaged in each bin (for error bars).
    """
    ell = np.asarray(ell, dtype=float)
    cl  = np.asarray(cl,  dtype=float)

    mask = np.isfinite(ell) & np.isfinite(cl) & (ell > 0)
    ell, cl = ell[mask], cl[mask]

    if len(ell) == 0:
        return np.array([]), np.array([]), np.array([], dtype=int)

    if ell_min is None:
        ell_min = ell.min()
    if ell_max is None:
        ell_max = ell.max()

    log_min = np.log10(ell_min)
    log_max = np.log10(ell_max)
    n_bins = max(1, int(np.ceil((log_max - log_min) * n_bins_per_decade)))

    edges = np.logspace(log_min, log_max, n_bins + 1)
    bin_idx = np.digitize(ell, edges)

    ell_binned, cl_binned, n_modes = [], [], []
    for b in range(1, n_bins + 1):
        in_bin = bin_idx == b
        if not np.any(in_bin):
            continue
        ell_binned.append(np.sqrt(edges[b - 1] * edges[b]))  # geometric centre
        cl_binned.append(np.mean(cl[in_bin]))
        n_modes.append(int(np.sum(in_bin)))

    return np.array(ell_binned), np.array(cl_binned), np.array(n_modes, dtype=int)


def transfer(mesh0, mesh1, box_shape, kedges: int | float | list = None, comp=(False, False)):
    if isinstance(comp, int):
        comp = (comp, comp)
    pow_fn = partial(spectrum, box_shape=box_shape, kedges=kedges)
    ks, pow0 = pow_fn(mesh0, comp=comp[0])
    ks, pow1 = pow_fn(mesh1, comp=comp[1])
    return ks, (pow1 / pow0) ** 0.5


def coherence(mesh0, mesh1, box_shape, kedges: int | float | list = None, comp=(False, False)):
    if isinstance(comp, int):
        comp = (comp, comp)
    pow_fn = partial(spectrum, box_shape=box_shape, kedges=kedges)
    ks, pow01 = pow_fn(mesh0, mesh1, comp=comp)
    ks, pow0 = pow_fn(mesh0, comp=comp[0])
    ks, pow1 = pow_fn(mesh1, comp=comp[1])
    return ks, pow01 / (pow0 * pow1) ** 0.5


def powtranscoh(mesh0, mesh1, box_shape, kedges: int | float | list = None, comp=(False, False)):
    if isinstance(comp, int):
        comp = (comp, comp)
    pow_fn = partial(spectrum, box_shape=box_shape, kedges=kedges)
    ks, pow01 = pow_fn(mesh0, mesh1, comp=comp)
    ks, pow0 = pow_fn(mesh0, comp=comp[0])
    ks, pow1 = pow_fn(mesh1, comp=comp[1])
    return ks, pow1, (pow1 / pow0) ** 0.5, pow01 / (pow0 * pow1) ** 0.5


def deconv_paint(mesh, order=2):
    """
    Deconvolve the mesh by the paint kernel of given order.
    """
    if jnp.isrealobj(mesh):
        kvec = rfftk(mesh.shape)
        mesh = jnp.fft.rfftn(mesh)
        mesh /= paint_kernel(kvec, order)
        mesh = jnp.fft.irfftn(mesh)
    else:
        kvec = rfftk(ch2rshape(mesh.shape))
        mesh /= paint_kernel(kvec, order)
    return mesh


def kaiser_formula(cosmo: Cosmology, a, lin_kpow, bE, poles=0):
    """
    bE is the Eulerien linear bias
    """
    poles = jnp.atleast_1d(poles)
    beta = growth_rate(cosmo, a) / bE
    k, pow = lin_kpow
    pow *= growth_factor(cosmo, a) ** 2

    weights = np.ones(len(poles)) * bE**2
    for i_ell, ell in enumerate(poles):
        if ell == 0:
            weights[i_ell] *= 1 + beta * 2 / 3 + beta**2 / 5
        elif ell == 2:
            weights[i_ell] *= beta * 4 / 3 + beta**2 * 4 / 7
        elif ell == 4:
            weights[i_ell] *= beta**2 * 8 / 35
        else:
            raise NotImplementedError(
                "Handle only poles of order ell=0, 2 ,4. ell={ell} not implemented."
            )

    pow = jnp.moveaxis(pow[..., None] * weights, -1, -2)
    return k, pow


#################
# Chain Metrics #
#################
def geomean(x, axis=None):
    return jnp.exp(jnp.mean(jnp.log(x), axis=axis))


def harmean(x, axis=None):
    return 1 / jnp.mean(1 / x, axis=axis)


def multi_ess(x, axis=None):
    return harmean(effective_sample_size(x), axis=axis)


def multi_gr(x, axis=None):
    """
    In the order of (1+nc/mESS)^(1/2), with nc the number of chains.
    cf. https://arxiv.org/pdf/1812.09384 and mESS := HarMean(ESS)
    """
    return jnp.mean(gelman_rubin(x) ** 2, axis=axis) ** 0.5
