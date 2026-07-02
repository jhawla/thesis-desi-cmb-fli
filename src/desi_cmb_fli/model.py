from __future__ import annotations  # for Union typing with | in python<3.10

from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from pprint import pformat

import jax_cosmo as jc
import numpy as np
import numpyro.distributions as dist
from jax import grad, tree
from jax import numpy as jnp
from jax import random as jr
from jax_cosmo import Cosmology
from jaxpm.painting import cic_paint
from numpyro import deterministic, render_model, sample
from numpyro.handlers import block, condition, seed, trace
from numpyro.infer.util import log_density

from desi_cmb_fli.bricks import (
    add_png,
    fNL_bias,
    get_cosmology,
    interlace_paint_deconv,
    kaiser_boost,
    kaiser_model,
    kaiser_posterior,
    lagrangian_fog_velocity,
    lagrangian_weights,
    lin_power_mesh,
    regular_pos,
    rsd,
    samp2base,
    samp2base_mesh,
    tophysical_mesh,
    tophysical_pos,
)
from desi_cmb_fli.chains import Chains
from desi_cmb_fli.metrics import powtranscoh, spectrum
from desi_cmb_fli.nbody import a2g, g2a, lpt, nbody_bf
from desi_cmb_fli.utils import (
    DetruncTruncNorm,
    DetruncUnif,
    cgh2rg,
    ch2rshape,
    chreshape,
    get_scaled_shape,
    nvmap,
    r2chshape,
    rg2cgh,
    ysafe_dump,
    ysafe_load,
)

default_config = {
    # Mesh and box parameters
    "mesh_shape": 3 * (64,),  # int
    "box_shape": 3 * (320.0,),  # in Mpc/h
    # Evolution
    "a_obs": None,  # None => lightcone, float => snapshot
    "evolution": "lpt",  # kaiser, lpt, nbody
    "init_oversamp": 1.0,  # initial linear field 1D oversampling (montecosmo: 3/2)
    "evol_oversamp": 1.0,  # LPT evolution / bias-product mesh 1D oversampling (montecosmo: 7/4)
    "ptcl_oversamp": 1.0,  # particle cloud 1D oversampling (montecosmo: 7/4)
    "paint_oversamp": 1.0,  # CIC paint oversampling factor (1.0 = no oversampling, 1.5 = standard)
    "nbody_steps": 5,
    "nbody_snapshots": None,
    "lpt_order": 2,
    # Observables
    "gxy_density": 1e-3,  # in galaxy / (Mpc/h)^3
    "observable": "field",  # 'field', TODO: 'powspec' (with poles), 'bispec'
    "curved_sky": True,
    "los": (0.0, 0.0, 1.0),
    "poles": (0, 2, 4),
    "galaxies_enabled": True,  # Enable galaxy likelihood (set False for CMB-only runs)
    "gxy_stoch_noise": False,  # Free galaxy shot-noise amplitude s_e (montecosmo); s_e=1 => pure Poisson
    "gxy_ngbar_free": False,  # Free per-radial-shell mean density (montecosmo ngbars); marginalises the
    # integral constraint on the lightcone. Off => n̄ fixed (snapshot). One latent ngbar_<b> per z-shell.
    # CMB lensing parameters
    "cmb_enabled": False,  # Enable CMB lensing in forward model and likelihood
    "cmb_lensing_obs": None,  # Observed convergence map (set during conditioning)
    "cmb_noise_nell": None,  # Path to N_ell file or dictionary {'ell': ..., 'N_ell': ...}
    "cmb_noise_scaling": 1.0,  # Artificial noise scaling factor (for tests: 0.01 = divide by 100)
    "cmb_nside": 256,      # HEALPix nside for curved-sky convergence map
    "cmb_n_shells": 189,   # Number of radial shells for Born integration
    "cmb_observer_mode": "face",  # 'face' or 'center'
    "cmb_observer_position": None,  # Optional explicit [x, y, z] in Mpc/h
    "cmb_mask": None,  # Optional external survey mask (npy/FITS or array)

    "cmb_z_source": 1100.0,  # CMB last scattering surface
    "high_z_mode": "taylor",  # 'fixed', 'taylor', 'exact'
    "cmb_likelihood_mode": "diagonal",  # 'diagonal' (fast, exact on full sky) | 'pixel_exact' (exact cut-sky via KL eigenmodes)
    "cmb_kl_rcond": 1e-6,  # pixel_exact: keep eigenmodes with lambda > rcond*lambda_max (supported Slepian subspace; 1e-6 keeps cond~1e6, safe in float32; lower=more complete but more fragile)

    # Primordial non-Gaussianity
    "png_type": None,  # None (Gaussian), 'fNL' (derive b_phi/b_phi_delta from bias), 'fNL_bias'
    # Latents
    "precond": "kaiser_dyn",  # direct, fourier, kaiser, kaiser_dyn
    "latents": {
        "Omega_m": {
            "group": "cosmo",
            "label": "{\\Omega}_m",
            "loc": 0.3,
            "scale": 0.15,
            "scale_fid": 0.05,
            "loc_fid": 0.26,
            "low": 0.10,  # Physical minimum (must be > Ω_b ≈ 0.049)
            "high": 0.60,
        },
        "sigma8": {
            "group": "cosmo",
            "label": "{\\sigma}_8",
            "loc": 0.8,
            "scale": 0.15,
            "scale_fid": 0.04,
            "loc_fid": 0.74,
            "low": 0.50,
            "high": 1.10,
        },
        # Bias latents: UNBOUNDED (no low/high) as in montecosmo — wide prior `scale`
        # (data-constrained) with a small preconditioning `scale_fid`. Hard bounds caused
        # railing (bn2 at -60), so they are removed.
        "b1": {
            "group": "bias",
            "label": "{b}_1",
            "loc": 0.8,
            "scale": 1e2,
            "scale_fid": 1e-2,
            "loc_fid": 0.8,
        },
        "b2": {
            "group": "bias",
            "label": "{b}_2",
            "loc": 0.0,
            "scale": 1e2,
            "scale_fid": 3e-2,
            "loc_fid": 0.0,
        },
        "bs2": {
            "group": "bias",
            "label": "{b}_{s^2}",
            "loc": 0.0,
            "scale": 1e2,
            "scale_fid": 1e-1,
            "loc_fid": 0.0,
        },
        "bn2": {
            "group": "bias",
            "label": "{b}_{\\nabla^2}",
            "loc": 0.0,
            "scale": 1e3,
            "scale_fid": 1.0,
            "loc_fid": 0.0,
        },
        "bnpar": {
            "group": "bias",
            "label": "{b}_{\\nabla_\\parallel}",
            "loc": 0.0,
            "scale": 1e2,
            "scale_fid": 1.0,
            "loc_fid": 0.0,
        },
        "fNL": {
            "group": "png",
            "label": "{f}_\\mathrm{NL}",
            "loc": 0.0,
            "scale": 1e4,
            "scale_fid": 1e2,
        },
        "fNL_bp": {
            "group": "png",
            "label": "{f}_\\mathrm{NL} b_\\phi",
            "loc": 0.0,
            "scale": 1e4,
            "scale_fid": 3e1,
        },
        "fNL_bpd": {
            "group": "png",
            "label": "{f}_\\mathrm{NL} b_{\\phi\\delta}",
            "loc": 0.0,
            "scale": 1e4,
            "scale_fid": 3e2,
        },
        "s_e": {
            # Galaxy shot-noise amplitude (montecosmo convention): noise std = s_e * sqrt(1/n̄),
            # variance = s_e**2 / n̄. s_e=1 => pure Poisson. Only sampled when gxy_stoch_noise=True.
            "group": "stoch",
            "label": "{s}_\\epsilon",
            "loc": 1.0,
            "scale": 1.0,
            "scale_fid": 0.3,
            "low": 0.0,
            "high": 10.0,
        },
        "init_mesh": {
            "group": "init",
            "label": "{\\delta}_L",
        },
    },
}


def get_model_from_config(config_or_path):
    """
    Factory function to create a FieldLevelModel from a config dictionary or path.
    Handles loading, mesh calculation, and parameter setup (CMB, etc.).

    Args:
        config_or_path (dict or str or Path): Config dict or path to config.yaml.

    Returns:
        tuple: (field_level_model_instance, config_dict)
    """
    if isinstance(config_or_path, str | Path):
        from desi_cmb_fli import utils
        cfg = utils.yload(config_or_path)
    else:
        cfg = config_or_path

    # Build model config
    model_config = default_config.copy()
    model_config["box_shape"] = tuple(cfg["model"]["box_shape"])

    # Cell size & Mesh shape
    cell_size = float(cfg["model"]["cell_size"])
    mesh_shape = [int(round(L / cell_size)) for L in model_config["box_shape"]]
    mesh_shape = [n + 1 if n % 2 != 0 else n for n in mesh_shape] # Ensure even
    model_config["mesh_shape"] = tuple(mesh_shape)

    # Evolution params
    model_config["evolution"] = cfg["model"]["evolution"]
    model_config["lpt_order"] = cfg["model"]["lpt_order"]
    model_config["gxy_density"] = cfg["model"]["gxy_density"]
    model_cfg = cfg.get("model", {})
    lightcone = bool(model_cfg.get("lightcone", True))
    if lightcone:
        model_config["a_obs"] = None
    elif "a_obs" in model_cfg:
        model_config["a_obs"] = float(model_cfg["a_obs"])
    else:
        raise ValueError("model.a_obs must be provided when model.lightcone=false")

    if "precond" in model_cfg:
        model_config["precond"] = model_cfg["precond"]
    for _ov in ("init_oversamp", "evol_oversamp", "ptcl_oversamp"):
        if _ov in model_cfg:
            model_config[_ov] = float(model_cfg[_ov])
    if "paint_oversamp" in model_cfg:
        model_config["paint_oversamp"] = float(model_cfg["paint_oversamp"])
    if "curved_sky" in model_cfg:
        model_config["curved_sky"] = bool(model_cfg["curved_sky"])
    if "los" in model_cfg:
        model_config["los"] = None if model_cfg["los"] is None else tuple(model_cfg["los"])
    if "galaxies_enabled" in cfg["model"]:
        model_config["galaxies_enabled"] = cfg["model"]["galaxies_enabled"]
    if "gxy_stoch_noise" in cfg["model"]:
        model_config["gxy_stoch_noise"] = bool(cfg["model"]["gxy_stoch_noise"])
    if "gxy_ngbar_free" in cfg["model"]:
        model_config["gxy_ngbar_free"] = bool(cfg["model"]["gxy_ngbar_free"])
    if "png_type" in cfg["model"]:
        png_type = cfg["model"]["png_type"]
        model_config["png_type"] = None if png_type in (None, "None", "none") else str(png_type)

    # CMB lensing config
    cmb_cfg = cfg.get("cmb_lensing", {})
    model_config["cmb_observer_mode"] = str(cmb_cfg.get("observer_mode", "face"))
    model_config["cmb_observer_position"] = cmb_cfg.get("observer_position", None)
    if cmb_cfg.get("enabled", False):
        model_config["cmb_enabled"] = True
        # If enabled, setup specific CMB params
        model_config["cmb_nside"] = int(cmb_cfg.get("nside", 256))
        model_config["cmb_n_shells"] = int(cmb_cfg.get("n_shells", 189))
        if "likelihood" in cmb_cfg:
            raise ValueError(
                "cmb_lensing.likelihood has been removed; the HEALPix harmonic likelihood is always used."
            )
        model_config["cmb_likelihood_mode"] = str(cmb_cfg.get("likelihood_mode", "diagonal"))
        model_config["cmb_kl_rcond"] = float(cmb_cfg.get("kl_rcond", 1e-6))
        model_config["cmb_mask"] = cmb_cfg.get("mask", None)
        model_config["full_los_correction"] = cmb_cfg.get("full_los_correction", False)
        model_config["chi_high_z_max"] = cmb_cfg.get("chi_high_z_max", None)  # None = integrate to chi_CMB
        model_config["cmb_z_source"] = float(cmb_cfg.get("z_source", 1100.0))
        model_config["cmb_lensing_obs"] = None # Will be set via conditioning if needed

        # High-Z mode selection
        if "high_z_mode" in cmb_cfg:
            model_config["high_z_mode"] = cmb_cfg["high_z_mode"]
        else:
            # Default
            model_config["high_z_mode"] = "taylor"

        if "cmb_noise_nell" in cmb_cfg:
            model_config["cmb_noise_nell"] = cmb_cfg["cmb_noise_nell"]
        if "cmb_noise_scaling" in cmb_cfg:
            model_config["cmb_noise_scaling"] = cmb_cfg["cmb_noise_scaling"]

    # Load latents from config if provided, otherwise use defaults
    if "latents" in cfg:
        # Merge with defaults: config values override defaults
        merged_latents = {}
        for param_name, default_latent in default_config["latents"].items():
            merged_latents[param_name] = default_latent.copy()
            if param_name in cfg["latents"]:
                merged_latents[param_name].update(cfg["latents"][param_name])
        model_config["latents"] = merged_latents
    else:
        model_config["latents"] = {k: v.copy() for k, v in default_config["latents"].items()}

    if model_config.get("gxy_ngbar_free", False):
        import re as _re

        import jax_cosmo as _jc

        from desi_cmb_fli.bricks import get_cosmology as _get_cosmology

        raw = cfg.get("abacus_galaxy", {}).get("file")
        paths = raw if isinstance(raw, list | tuple) else [raw]
        zs = sorted({float(m.group(1)) for p in paths if p
                     for m in [_re.search(r"/z(\d+\.\d+)/", str(p))] if m})
        if len(zs) < 2:
            print(f"[ngbars] gxy_ngbar_free set but only {len(zs)} z-shell found; disabling.")
            model_config["gxy_ngbar_free"] = False
        else:
            # Effective survey z-range (catalog bands extend ~half a spacing beyond the nominal z's).
            dz_lo, dz_hi = zs[1] - zs[0], zs[-1] - zs[-2]
            z_lo, z_hi = max(zs[0] - 0.5 * dz_lo, 0.0), zs[-1] + 0.5 * dz_hi

            def _fid(name, dflt):
                c = model_config["latents"].get(name, {})
                return float(c.get("loc_fid", c.get("loc", dflt)))

            cosmo_fid = _get_cosmology(Omega_m=_fid("Omega_m", 0.315192),
                                       sigma8=_fid("sigma8", 0.811355))
            chi_lo = float(_jc.background.radial_comoving_distance(cosmo_fid, 1.0 / (1.0 + z_lo))[0])
            chi_hi = float(_jc.background.radial_comoving_distance(cosmo_fid, 1.0 / (1.0 + z_hi))[0])
            dr = 3 ** 0.5 * cell_size
            n_rbins = max(int(round((chi_hi - chi_lo) / dr)), 1)

            model_config["n_gxy_shells"] = n_rbins
            tmpl = {"group": "syst", "loc": 1.0, "scale": 1.0,
                    "scale_fid": 1e-2, "loc_fid": 1.0, "low": 0.0, "high": float("inf")}
            for b in range(n_rbins):
                lat = tmpl.copy()
                lat["label"] = r"{\bar{n}}_{g," + str(b) + "}"
                model_config["latents"][f"ngbar_{b}"] = lat
            print(f"[ngbars] {n_rbins} free fine radial bins (dr≈{dr:.0f} Mpc/h, "
                  f"chi≈[{chi_lo:.0f},{chi_hi:.0f}] Mpc/h), montecosmo-style; ngbar_0..ngbar_{n_rbins - 1}")

    model_instance = FieldLevelModel(**model_config)

    return model_instance, model_config


class Model:
    ###############
    # Model calls #
    ###############
    def _model(self, *args, **kwargs):
        raise NotImplementedError

    def model(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def reset(self):
        self.model = self._model

    def __call__(self):
        return self.model()

    def reparam(self, params, inv=False):
        return params

    def _block_det(self, model, hide_base=True, hide_det=True):
        base_name = self.latents.keys()
        if hide_base:
            if hide_det:

                def hide_fn(site):
                    return site["type"] == "deterministic"
            else:

                def hide_fn(site):
                    return site["type"] == "deterministic" and site["name"] in base_name
        else:
            if hide_det:

                def hide_fn(site):
                    return site["type"] == "deterministic" and site["name"] not in base_name
            else:

                def hide_fn(site):
                    return False

        return block(model, hide_fn=hide_fn)

    def predict(
        self,
        rng=42,
        samples=None,
        batch_ndim=0,
        hide_base=True,
        hide_det=True,
        hide_samp=True,
        frombase=False,
    ):
        """
        Run model conditioned on samples.
        * If samples is None, return a single prediction.
        * If samples is an int or tuple, return a prediction of such shape.
        * If samples is a dict, return a prediction for each sample, assuming batch_ndim batch dimensions.
        """
        if isinstance(rng, int):
            rng = jr.key(rng)

        def single_prediction(rng, sample=None):
            if sample is None:
                sample = {}
            # Optionally reparametrize base to sample params
            if frombase:
                sample = self.reparam(sample, inv=True)
                # NOTE: deterministic sites have no effects with handlers.condition, but do with handlers.subsitute

            # Condition then block
            model = condition(self.model, data=sample)
            if hide_samp:
                model = block(model, hide=sample.keys())
            model = self._block_det(model, hide_base=hide_base, hide_det=hide_det)

            # Trace and return values
            tr = trace(seed(model, rng_seed=rng)).get_trace()
            return {k: v["value"] for k, v in tr.items()}

        if samples is None:
            return single_prediction(rng)

        elif isinstance(samples, int | tuple):
            if isinstance(samples, int):
                samples = (samples,)
            rng = jr.split(rng, samples)
            return nvmap(single_prediction, len(samples))(rng)

        elif isinstance(samples, dict):
            shape = jnp.shape(next(iter(samples.values())))[:batch_ndim]
            if not shape:
                return single_prediction(jr.split(rng, 1)[0], samples)
            rng = jr.split(rng, shape)
            return nvmap(single_prediction, len(shape))(rng, samples)

    ############
    # Wrappers #
    ############
    def logpdf(self, params):
        """
        A log-density function of the model. In particular, it is the log-*probability*-density function
        with respect to the full set of variables, i.e. E[e^logpdf] = 1.

        For unnormalized log-densities in numpyro, see https://forum.pyro.ai/t/unnormalized-densities/3251/9
        """
        return log_density(self.model, (), {}, params)[0]

    def potential(self, params):
        return -self.logpdf(params)

    def force(self, params):
        return grad(self.logpdf)(params)  # force = - grad potential = grad logpdf

    def trace(self, rng):
        return trace(seed(self.model, rng_seed=rng)).get_trace()

    def seed(self, rng):
        self.model = seed(self.model, rng_seed=rng)

    def condition(self, data=None, frombase=False):
        if data is None:
            data = {}
        # Optionally reparametrize base to sample params
        if frombase:
            data = self.reparam(data, inv=True)
        self.model = condition(self.model, data=data)

    def block(
        self, hide_fn=None, hide=None, expose_types=None, expose=None, hide_base=True, hide_det=True
    ):
        """
        Selectively hides parameters in the model.

        Precedence is given according to the order: hide_fn, hide, expose_types, expose, (hide_base, hide_det).
        Only the set of parameters with the precedence is considered.
        The default call thus hides base and other deterministic sites, for sampling purposes.

        In CMB-only mode, unconstrained galaxy parameters (b1_, b2_, bs2_, bn2_)
        are automatically hidden since they are fixed to fiducial values and not sampled.
        """
        if all(x is None for x in (hide_fn, hide, expose_types, expose)):
            self.model = self._block_det(self.model, hide_base=hide_base, hide_det=hide_det)

            if self.cmb_enabled and not self.galaxies_enabled:
                bias_params_sample = [k + "_" for k in self.groups["bias"]]
                self.model = block(self.model, hide=bias_params_sample)
        else:
            self.model = block(
                self.model, hide_fn=hide_fn, hide=hide, expose_types=expose_types, expose=expose
            )

    def render(self, render_dist=False, render_params=False):
        from IPython.display import display

        display(
            render_model(self.model, render_distributions=render_dist, render_params=render_params)
        )

    def partial(self, *args, **kwargs):
        self.model = partial(self.model, *args, **kwargs)

    #################
    # Save and load #
    #################
    def save(self, path):  # with yaml because not array-like
        ysafe_dump(asdict(self), path)

    @classmethod
    def load(cls, path):
        return cls(**ysafe_load(path))






@dataclass
class FieldLevelModel(Model):
    """
    Field-level cosmological model,
    with LPT and PM displacements, Lagrangian bias, and RSD.
    The relevant variables can be traced.

    Parameters
    ----------
    mesh_shape : array_like of int
        Shape of the mesh.
    box_shape : array_like
        Shape of the box in Mpc/h. Typically such that cell lengths would be between 1 and 10 Mpc/h.
    evolution : str
        Evolution model: 'kaiser', 'lpt', 'nbody'.
    a_obs : float or None
        Scale factor of observations. If None, run in lightcone mode with local a(chi).
    nbody_steps : int
        Number of N-body steps.
        Only used for 'nbody' evolution.
    nbody_snapshots : int or list
        Number or list of N-body snapshots to save. If None, only save last.
        Only used for 'nbody' evolution.
    lpt_order : int
        Order of LPT displacement.
        Only used for 'lpt' evolution.
    observable : str
        Observable: 'field', 'powspec'.
    gxy_density : float
        Galaxy density in galaxy / (Mpc/h)^3
    los : array_like
        Line-of-sight direction. If None, no Redshift Space Distorsion is applied.
    curved_sky : bool
        If True, use radial LOS and distance for lightcone quantities.
    poles : array_like of int
        Power spectrum poles to compute.
        Only used for 'powspec' observable.
    cmb_lensing_obs : array_like or None
        Observed CMB convergence map. If None, no CMB constraint is applied.
    cmb_noise_nell : str or dict or None
        CMB noise power spectrum N_ell. If str, path to file with two columns (ell, N_ell).
    cmb_nside : int
        HEALPix nside for the curved-sky convergence map (default 256).
    cmb_n_shells : int
        Number of radial Born integration shells (default 189).
    cmb_z_source : float
        CMB source redshift (last scattering surface).
    precond : str
        Preconditioning method: 'direct', 'fourier', 'kaiser'.
    latents : dict
        Latent variables configuration.
    """

    # Mesh and box parameters
    mesh_shape: np.ndarray = field(default_factory=lambda: np.array(default_config["mesh_shape"]))
    box_shape: np.ndarray = field(default_factory=lambda: np.array(default_config["box_shape"]))
    # Evolution
    evolution: str = field(default=default_config["evolution"])
    a_obs: float | None = field(default=default_config["a_obs"])
    nbody_steps: int = field(default=default_config["nbody_steps"])
    nbody_snapshots: int | list = field(default=default_config["nbody_snapshots"])
    lpt_order: int = field(default=default_config["lpt_order"])
    init_oversamp: float = field(default=default_config["init_oversamp"])
    evol_oversamp: float = field(default=default_config["evol_oversamp"])
    ptcl_oversamp: float = field(default=default_config["ptcl_oversamp"])
    paint_oversamp: float = field(default=default_config["paint_oversamp"])
    # Observable
    observable: str = field(default=default_config["observable"])
    gxy_density: float = field(default=default_config["gxy_density"])
    galaxies_enabled: bool = field(default=default_config["galaxies_enabled"])
    gxy_stoch_noise: bool = field(default=default_config["gxy_stoch_noise"])
    gxy_ngbar_free: bool = field(default=default_config["gxy_ngbar_free"])
    n_gxy_shells: int = field(default=0)
    curved_sky: bool = field(default=default_config["curved_sky"])
    los: tuple | None = field(default=default_config["los"])
    poles: tuple = field(default=default_config["poles"])
    # CMB lensing (with defaults for backward compatibility)
    cmb_lensing_obs: np.ndarray | None = field(default=default_config["cmb_lensing_obs"])
    cmb_noise_nell: str | dict | None = field(default=default_config["cmb_noise_nell"])
    cmb_noise_scaling: float = field(default=default_config["cmb_noise_scaling"])
    cmb_nside: int = field(default=default_config["cmb_nside"])
    cmb_n_shells: int = field(default=default_config["cmb_n_shells"])
    cmb_observer_mode: str = field(default=default_config["cmb_observer_mode"])
    cmb_observer_position: tuple | None = field(default=default_config["cmb_observer_position"])
    cmb_mask: np.ndarray | str | None = field(default=default_config["cmb_mask"])
    cmb_z_source: float = field(default=default_config["cmb_z_source"])
    cmb_enabled: bool = field(default=False)  # Explicit flag to enable CMB lensing

    full_los_correction: bool = field(default=False)  # Enable high-z kappa correction
    chi_high_z_max: float | None = field(default=None)  # Upper chi limit for high-z correction (None = chi_CMB)
    high_z_mode: str = field(default="taylor")  # 'fixed', 'taylor', 'exact'
    cmb_likelihood_mode: str = field(default=default_config["cmb_likelihood_mode"])  # 'diagonal' | 'pixel_exact'
    cmb_kl_rcond: float = field(default=default_config["cmb_kl_rcond"])  # pixel_exact KL eigenmode cutoff
    # Primordial non-Gaussianity
    png_type: str | None = field(default=default_config["png_type"])
    # Latents (required, from default_config)
    precond: str = field(default=default_config["precond"])
    latents: dict = field(default_factory=lambda: default_config["latents"])

    def __post_init__(self):
        self.latents = self._validate_latents()
        self.groups = self._groups(base=True)
        self.groups_ = self._groups(base=False)
        self.labels = self._labels()
        self.loc_fid = self._loc_fid()

        self.mesh_shape = np.asarray(self.mesh_shape)
        # NOTE: if x32, cast mesh_shape into float32 to avoid int32 overflow when computing products
        self.box_shape = np.asarray(self.box_shape, dtype=float)
        self.cell_shape = self.box_shape / self.mesh_shape

        self.sim_mesh_shape = np.asarray(self.mesh_shape)
        self.sim_box_shape = np.asarray(self.box_shape)

        if self.cmb_observer_position is not None:
            self.observer_position = np.asarray(self.cmb_observer_position, dtype=float)
        elif self.cmb_observer_mode == "center":
            self.observer_position = 0.5 * np.asarray(self.box_shape, dtype=float)
        elif self.cmb_observer_mode == "face":
            self.observer_position = np.array(
                [self.box_shape[0] / 2.0, self.box_shape[1] / 2.0, 0.0], dtype=float
            )
        elif self.cmb_observer_mode == "corner":
            self.observer_position = np.zeros(3, dtype=float)
        else:
            raise ValueError(f"Unknown cmb_observer_mode: {self.cmb_observer_mode}")

        self.box_center = np.asarray(self.box_shape, dtype=float) / 2.0 - self.observer_position
        self.sim_box_center = self.box_center
        self._sim_center = self.box_center
        self._sim_shape = self.sim_box_shape
        self._sim_mesh = self.sim_mesh_shape

        self.init_shape = np.asarray(get_scaled_shape(tuple(self.sim_mesh_shape), self.init_oversamp))
        self.evol_shape = np.asarray(get_scaled_shape(tuple(self.sim_mesh_shape), self.evol_oversamp))
        self.ptcl_shape = np.asarray(get_scaled_shape(tuple(self.sim_mesh_shape), self.ptcl_oversamp))

        # Oversampled paint shape for CIC deconvolution
        if self.paint_oversamp != 1.0:
            self.paint_shape = np.asarray(get_scaled_shape(tuple(self.sim_mesh_shape), self.paint_oversamp))
        else:
            self.paint_shape = self.sim_mesh_shape
        if any(s != self.sim_mesh_shape[0] for s in
               (self.init_shape[0], self.evol_shape[0], self.ptcl_shape[0], self.paint_shape[0])):
            print(f"  Oversampling: init={tuple(self.init_shape)} evol={tuple(self.evol_shape)} "
                  f"ptcl={tuple(self.ptcl_shape)} paint={tuple(self.paint_shape)} (final={tuple(self.sim_mesh_shape)})")

        if self.los is not None:
            self.los = np.asarray(self.los)
            self.los = self.los / np.linalg.norm(self.los)

        cosmo_fid = get_cosmology(**self.loc_fid)
        self.lightcone = self.a_obs is None
        if self.lightcone:
            _, a_mesh = tophysical_mesh(
                self.box_center,
                self.box_shape,
                self.mesh_shape,
                cosmo_fid,
                a_obs=None,
                curved_sky=self.curved_sky,
                los=self.los,
            )
            self.a_fid = float(g2a(cosmo_fid, jnp.mean(a2g(cosmo_fid, a_mesh))))
            self.a_obs_center = float(
                jc.background.a_of_chi(cosmo_fid, float(np.linalg.norm(self.box_center)))[0]
            )
        else:
            self.a_obs = float(self.a_obs)
            self.a_fid = float(self.a_obs)
            self.a_obs_center = float(self.a_obs)
            a2g(cosmo_fid, self.a_obs)  # build the background emulator eagerly (snapshot skips the lightcone a2g)

        # Calculate z_max for logging
        if self.cmb_observer_mode == "center":
            self.chi_boundary = self.box_shape[2] / 2.0
        else:
            # face and corner observers: box spans the full depth along +z.
            self.chi_boundary = self.box_shape[2]

        a_max = float(jc.background.a_of_chi(cosmo_fid, self.chi_boundary)[0])
        z_max_box = 1/a_max - 1

        print(f"  Box size: {self.box_shape} Mpc/h")
        print(f"  Center chi: {self.box_center[2]:.2f} Mpc/h")
        if self.lightcone:
            print(f"  Lightcone mode: a_obs=None (a_center={self.a_obs_center:.4f}, z_center={1/self.a_obs_center - 1:.4f})")
        else:
            print(f"  Snapshot mode: a_obs={self.a_obs:.4f} (z_obs={1/self.a_obs - 1:.4f})")
        print(f"  Box extends to chi={self.chi_boundary:.2f} Mpc/h (z_max = {z_max_box:.4f})")

        self.k_funda = 2 * np.pi / np.min(self.box_shape)
        self.k_nyquist = np.pi * np.min(self.mesh_shape / self.box_shape)
        self.gxy_count = self.gxy_density * self.cell_shape.prod()

        # Validation: at least one observable must be enabled
        if not self.galaxies_enabled and not self.cmb_enabled:
            raise ValueError("At least one observable must be enabled (galaxies_enabled or cmb_enabled)")

        if self.cmb_enabled:
            import healpy as hp

            from desi_cmb_fli.cmb_lensing import (
                _box_ray_intervals,
                compute_sigma_hp,
                compute_theoretical_cl_kappa,
                load_healpix_mask,
            )

            # Load N_ell
            if self.cmb_noise_nell is None:
                raise ValueError("cmb_noise_nell is required when cmb_enabled=True")

            if isinstance(self.cmb_noise_nell, str):
                _data = np.loadtxt(self.cmb_noise_nell)
                if _data.ndim == 1:
                    ell_in, nell_in = np.arange(len(_data)), _data
                else:
                    ell_in, nell_in = _data[:, 0], _data[:, 1]
            elif isinstance(self.cmb_noise_nell, dict):
                ell_in, nell_in = self.cmb_noise_nell["ell"], self.cmb_noise_nell["N_ell"]
            else:
                ell_in, nell_in = self.cmb_noise_nell[0], self.cmb_noise_nell[1]

            # Apply artificial scaling
            nell_in = np.asarray(nell_in, dtype=float).copy()
            if self.cmb_noise_scaling != 1.0:
                nell_in *= self.cmb_noise_scaling
                print(f"\n[CMB] WARNING: ARTIFICIAL NOISE SCALING APPLIED: {self.cmb_noise_scaling:.4f}")

            print(f"[CMB] Computing HEALPix deep-field mask (nside={self.cmb_nside}) ...")

            t_enter, t_exit = _box_ray_intervals(self.observer_position, self.box_shape, self.cmb_nside)
            self.t_enter = jnp.asarray(t_enter)
            self.t_exit = jnp.asarray(t_exit)

            self.cmb_sim_mask = (t_exit >= self.chi_boundary - 1e-4) & (t_exit > t_enter)
            self.cmb_chi_max_model = self.chi_boundary

            self.cmb_external_mask = load_healpix_mask(self.cmb_mask, self.cmb_nside)
            self.cmb_mask = self.cmb_sim_mask & self.cmb_external_mask
            n_pix_mask = int(np.sum(self.cmb_mask))
            print(f"[CMB] Effective mask: {n_pix_mask} pixels / {len(self.cmb_mask)} ({n_pix_mask / len(self.cmb_mask):.2%})")

            # 4. Radial shells strictly limited to the exact geometric depth
            dr = self.cmb_chi_max_model / self.cmb_n_shells
            self.cmb_d_r = dr
            self.cmb_r_shells = np.linspace(
                dr / 2.0, self.cmb_chi_max_model - dr / 2.0, self.cmb_n_shells
            )
            self.cmb_a_shells = np.array(
                [float(jc.background.a_of_chi(cosmo_fid, r)[0]) for r in self.cmb_r_shells]
            )

            # 1-D ell array for high-z correction / harmonic covariance.
            self.cmb_lmax = 2 * self.cmb_nside
            self.ell_1d = np.arange(self.cmb_lmax + 1, dtype=float)

            # Interpolate N_ell onto the model ell grid (needed by both likelihood modes).
            valid_mask = (np.asarray(ell_in) > 0) & np.isfinite(nell_in) & (nell_in > 0)
            ell_valid = np.asarray(ell_in, dtype=float)[valid_mask]
            nell_valid = np.asarray(nell_in, dtype=float)[valid_mask]
            self.nell_1d = np.exp(
                np.interp(
                    np.log(np.maximum(self.ell_1d, 1e-5)),
                    np.log(ell_valid),
                    np.log(nell_valid),
                )
            )
            self.nell_1d[0] = 0.0

            # Scalar noise summary retained for logging and synthetic tests.
            self.sigma_hp_base = compute_sigma_hp(self.ell_1d, self.nell_1d, self.cmb_nside)
            self.sigma_hp = self.sigma_hp_base
            print(f"[CMB] sigma_hp = {self.sigma_hp:.6f}")

            if self.cmb_likelihood_mode not in ("diagonal", "pixel_exact"):
                raise ValueError(
                    f"cmb_lensing.likelihood_mode must be 'diagonal' or 'pixel_exact', "
                    f"got {self.cmb_likelihood_mode!r}"
                )
            print(f"[CMB] Likelihood mode: {self.cmb_likelihood_mode}")

            if self.cmb_likelihood_mode == "diagonal":
                # Precompute MASTER mode-coupling matrix (required for exact harmonics).
                try:
                    import pymaster as nmt
                except ImportError as err:
                    raise ImportError(
                        "[CMB] pymaster (namaster) is required for CMB harmonics but not installed. "
                        "Install via conda: conda install -c conda-forge namaster"
                    ) from err

                print("[CMB] Computing MASTER mode-coupling matrix (M_ll)...")
                # Use the full effective mask (sim & external) so M_ll matches the
                # mask applied during both measurement and likelihood evaluation.
                mask_np = np.asarray(self.cmb_mask, dtype=float)

                bins = nmt.NmtBin.from_lmax_linear(self.cmb_lmax, 1)
                workspace = nmt.NmtWorkspace()

                f0 = nmt.NmtField(mask_np, None, spin=0, lmax=self.cmb_lmax)
                workspace.compute_coupling_matrix(f0, f0, bins)

                self.cmb_M_ll = jnp.asarray(workspace.get_coupling_matrix())
                print("[CMB] M_ll shape:", self.cmb_M_ll.shape)
                self.cmb_alm_l = np.asarray(hp.Alm.getlm(self.cmb_lmax)[0], dtype=int)
                self.cmb_alm_m = np.asarray(hp.Alm.getlm(self.cmb_lmax)[1], dtype=int)

                _valid = self.cmb_alm_l >= 2
                _idx_re_m0 = np.where(_valid & (self.cmb_alm_m == 0))[0]
                _idx_mp = np.where(_valid & (self.cmb_alm_m > 0))[0]
                self.cmb_pack_re_idx = jnp.asarray(np.concatenate([_idx_re_m0, _idx_mp]))
                self.cmb_pack_im_idx = jnp.asarray(_idx_mp)
                self.cmb_u_dim = int(_idx_re_m0.size + 2 * _idx_mp.size)
                self.cmb_l_of_u = jnp.asarray(
                    np.concatenate(
                        [self.cmb_alm_l[_idx_re_m0], self.cmb_alm_l[_idx_mp], self.cmb_alm_l[_idx_mp]]
                    )
                )
                # Re and Im of an m>0 mode each carry half the modal variance var_l.
                self.cmb_u_half = jnp.asarray(
                    np.concatenate(
                        [
                            np.ones(_idx_re_m0.size),
                            np.full(2 * _idx_mp.size, 1.0 / np.sqrt(2.0)),
                        ]
                    )
                )
            else:
                if self.full_los_correction and self.high_z_mode != "fixed":
                    raise ValueError(
                        "cmb_likelihood_mode='pixel_exact' requires high_z_mode='fixed' when "
                        "full_los_correction=True (the pixel covariance is precomputed once and "
                        f"must stay constant), got high_z_mode={self.high_z_mode!r}."
                    )

            # Cache/Precompute High-Z Correction (1-D ell)
            self.cl_high_z_cached = None
            self.high_z_gradients = None

            if self.full_los_correction:
                chi_source_fid = float(
                    jc.background.radial_comoving_distance(cosmo_fid, 1.0 / (1.0 + self.cmb_z_source))[0]
                )
                chi_high_z_upper = float(self.chi_high_z_max) if self.chi_high_z_max is not None else chi_source_fid
                if self.chi_high_z_max is not None:
                    print(
                        f"  [high-z] chi_high_z_max={self.chi_high_z_max:.0f} Mpc/h "
                        f"(chi_CMB={chi_source_fid:.0f})"
                    )

                if self.high_z_mode in ["fixed", "taylor"]:
                    self.cl_high_z_cached = compute_theoretical_cl_kappa(
                        cosmo_fid,
                        self.ell_1d,
                        self.chi_boundary,
                        chi_high_z_upper,
                        self.cmb_z_source,
                    )
                    print(
                        f"  Cached C_l^{{high-z}} at fiducial "
                        f"(mode={self.high_z_mode}, chi={self.chi_boundary:.0f}->{chi_high_z_upper:.0f} Mpc/h)"
                    )

                if self.high_z_mode == "taylor":
                    print("  Computing gradients for High-Z Taylor correction ...")

                    def _cl_wrapper(theta):
                        c_ = get_cosmology(Omega_m=theta[0], sigma8=theta[1])
                        chi_src_ = jc.background.radial_comoving_distance(c_, 1.0 / (1.0 + self.cmb_z_source))[0]
                        chi_up_ = jnp.minimum(jnp.array(chi_high_z_upper), chi_src_)
                        return compute_theoretical_cl_kappa(
                            c_, self.ell_1d, self.chi_boundary, chi_up_, self.cmb_z_source
                        )

                    om_fid = self.loc_fid["Omega_m"]
                    s8_fid = self.loc_fid["sigma8"]
                    eps_om = 1e-3
                    eps_s8 = 1e-3

                    cl_om_up = _cl_wrapper(jnp.array([om_fid + eps_om, s8_fid]))
                    cl_om_dn = _cl_wrapper(jnp.array([om_fid - eps_om, s8_fid]))
                    cl_s8_up = _cl_wrapper(jnp.array([om_fid, s8_fid + eps_s8]))
                    cl_s8_dn = _cl_wrapper(jnp.array([om_fid, s8_fid - eps_s8]))

                    self.high_z_gradients = {
                        "dCl_dOm": (cl_om_up - cl_om_dn) / (2.0 * eps_om),
                        "dCl_ds8": (cl_s8_up - cl_s8_dn) / (2.0 * eps_s8),
                    }

                    print(
                        f"  Gradients computed: dCl/dOm shape={self.high_z_gradients['dCl_dOm'].shape}"
                    )

                    # Update sigma_hp to include high-z variance
                    self.sigma_hp = compute_sigma_hp(
                        ell_in, nell_in, self.cmb_nside, cl_extra_1d=self.cl_high_z_cached
                    )
                    print(f"[CMB] sigma_hp (with high-z) = {self.sigma_hp:.6f}")

            if self.cmb_likelihood_mode == "pixel_exact":
                self._build_cmb_pixel_cov()


    def _build_cmb_pixel_cov(self):
        """Build the exact cut-sky CMB-lensing likelihood via signal-eigenmode (KL) compression.

        The masked-field stochastic covariance (noise N_l + fixed high-z C_l) of a band-limited
        HEALPix field over the observed pixels depends only on angular separation:

            Cov_pix[i, j] = sum_l (2l+1)/(4pi) * (N_l + Cl_high_z_l) * P_l(cos theta_ij)
        """
        import healpy as hp

        nside = int(self.cmb_nside)
        lmax = int(self.cmb_lmax)
        n_modes = (lmax + 1) ** 2

        # Constant total stochastic spectrum: instrument noise + (optional) fixed high-z.
        cl = np.asarray(self.nell_1d, dtype=np.float64).copy()
        if self.full_los_correction and self.cl_high_z_cached is not None:
            cl = cl + np.asarray(self.cl_high_z_cached, dtype=np.float64)
        ell = np.arange(lmax + 1, dtype=np.float64)
        coef = (2.0 * ell + 1.0) / (4.0 * np.pi) * cl

        # Observed pixels and their unit vectors.
        obs_pix = np.where(np.asarray(self.cmb_mask))[0].astype(np.int64)
        npix = obs_pix.size
        f_sky = npix / float(hp.nside2npix(nside))
        vecs = np.asarray(hp.pix2vec(nside, obs_pix), dtype=np.float64).T  # (npix, 3)
        print(f"[CMB] pixel_exact (KL): building {npix}x{npix} pixel covariance "
              f"(~{npix * npix * 8 / 1e9:.1f} GB f64) ...")

        # Tabulate xi(mu) = sum_l coef_l P_l(mu) on a fine grid, then interpolate per pair.
        mu_grid = np.linspace(-1.0, 1.0, 200001)
        xi_grid = np.polynomial.legendre.legval(mu_grid, coef)
        cov = np.empty((npix, npix), dtype=np.float64)
        chunk = 2048
        for i0 in range(0, npix, chunk):
            i1 = min(i0 + chunk, npix)
            mu = vecs[i0:i1] @ vecs.T
            np.clip(mu, -1.0, 1.0, out=mu)
            cov[i0:i1] = np.interp(mu, mu_grid, xi_grid)

        # Diagonalise and keep the supported (Slepian) subspace.
        print("[CMB] pixel_exact (KL): eigendecomposition ...")
        evals, evecs = np.linalg.eigh(cov)  # ascending eigenvalues, orthonormal eigenvectors
        del cov
        lam_max = float(evals[-1])
        rcond = float(getattr(self, "cmb_kl_rcond", 1e-8))
        keep = evals > rcond * lam_max
        k = int(np.count_nonzero(keep))
        if k == 0:
            raise np.linalg.LinAlgError(
                "[CMB] pixel_exact (KL): no eigenmodes above rcond*lambda_max; check the mask/spectrum."
            )
        U_k = np.ascontiguousarray(evecs[:, keep])          # (npix, k)
        lam_k = np.ascontiguousarray(evals[keep])           # (k,)

        self.cmb_obs_pix = jnp.asarray(obs_pix)
        self.cmb_kl_U = jnp.asarray(U_k)                    # eigenmode projection
        self.cmb_kl_var = jnp.asarray(lam_k)                # per-mode noise variance
        self.cmb_u_dim = int(k)
        print(
            f"[CMB] pixel_exact (KL): ready. npix_obs={npix} (f_sky={f_sky:.3f}), "
            f"kept k={k} modes (rcond={rcond:.0e}); supported~{int(f_sky * n_modes)}; "
            f"cond={lam_max / float(lam_k.min()):.1e}; lambda in [{lam_k.min():.2e}, {lam_max:.2e}]."
        )


    def __str__(self):
        out = ""
        out += "# CONFIG\n"
        out += pformat(asdict(self), width=1)
        out += "\n\n# INFOS\n"
        out += f"cell_shape:     {list(self.cell_shape)} Mpc/h\n"
        out += f"k_funda:        {self.k_funda:.5f} h/Mpc\n"
        out += f"k_nyquist:      {self.k_nyquist:.5f} h/Mpc\n"
        out += f"mean_gxy_count: {self.gxy_count:.3f} gxy/cell\n"
        out += f"lightcone:      {self.lightcone}\n"
        out += f"a_fid:          {self.a_fid:.4f}\n"

        if self.full_los_correction:
            out += "full_los_correction: True\n"
        return out

    def _model(self, temp_prior=1.0, temp_lik=1.0):
        cosmo, bias, init, png, s_e, ngbars = self.prior(temp=temp_prior)
        fields = self.evolve((cosmo, bias, init, png))
        return self.likelihood(
            cosmology=cosmo,
            bias=bias,
            temp=temp_lik,
            s_e=s_e,
            ngbars=ngbars,
            **fields,
        )

    def prior(self, temp=1.0):
        """
        A prior for cosmological model.

        Return base parameters, as reparametrization of sample parameters.
        """
        # Sample, reparametrize, and register cosmology
        cosmo_ = self._sample(self.groups["cosmo"])
        cosmo_ = samp2base(cosmo_, self.latents, inv=False, temp=temp)
        cosmo = {k: deterministic(k, v) for k, v in cosmo_.items()}

        # Bias parameters: fix to fiducial in CMB-only mode (unconstrained by data)
        if self.cmb_enabled and not self.galaxies_enabled:
            bias = {
                k: deterministic(k, jnp.asarray(self.loc_fid[k]))
                for k in self.groups["bias"]
            }
        else:
            bias_ = self._sample(self.groups["bias"])
            bias_ = samp2base(bias_, self.latents, inv=False, temp=temp)
            bias = {k: deterministic(k, v) for k, v in bias_.items()}

        cosmology = get_cosmology(**cosmo)

        if self.png_type is not None:
            png_ = self._sample(self.groups["png"])
            png_ = samp2base(png_, self.latents, inv=False, temp=temp)
            png = {k: deterministic(k, v) for k, v in png_.items()}
        else:
            png = {}

        # Stochastic noise nuisance: free galaxy shot-noise amplitude s_e.
        # s_e=1 => pure Poisson. Only sampled when enabled and galaxies are present.
        if self.gxy_stoch_noise and self.galaxies_enabled:
            stoch_ = self._sample(self.groups["stoch"])
            stoch_ = samp2base(stoch_, self.latents, inv=False, temp=temp)
            stoch = {k: deterministic(k, v) for k, v in stoch_.items()}
            s_e = stoch["s_e"]
        else:
            s_e = 1.0

        # Free per-shell mean-density nuisance (montecosmo ngbars). Vector of relative amplitudes
        # ngbar_<b> (1 = fiducial), one per radial z-shell; applied per radial bin in the likelihood.
        if self.gxy_ngbar_free and self.galaxies_enabled:
            syst_ = self._sample(self.groups["syst"])
            syst_ = samp2base(syst_, self.latents, inv=False, temp=temp)
            syst = {k: deterministic(k, v) for k, v in syst_.items()}
            ngbars = jnp.stack([syst[f"ngbar_{b}"] for b in range(self.n_gxy_shells)])
        else:
            ngbars = None

        # Sample, reparametrize, and register initial conditions
        init = {}
        name_ = self.groups["init"][0] + "_"

        bE = 1 + bias["b1"]
        scale, transfer = self._precond_scale_and_transfer(cosmology, bE)
        init[name_] = sample(name_, dist.Normal(0.0, scale))  # sample

        init = samp2base_mesh(
            init, self.precond, transfer=transfer, inv=False, temp=temp
        )  # reparametrize
        init = {k: deterministic(k, v) for k, v in init.items()}  # register base params

        return cosmology, bias, init, png, s_e, ngbars

    def _ngbar_alpha_field(self, ngbars):
        """Per-cell relative density amplitude from the per-shell ``ngbars`` vector.

        ``self.gxy_shell_id`` maps each cell to a radial z-shell index (0..n-1), or -1 for cells
        outside any galaxy shell. Appending 1.0 lets index -1 select unity (no rescaling there).
        """
        sid = jnp.asarray(self.gxy_shell_id)
        ngbars_padded = jnp.concatenate([jnp.asarray(ngbars), jnp.ones((1,), ngbars.dtype)])
        return ngbars_padded[sid]

    def paint_and_deconv(self, pos, weights=None, from_shape=None):
        """CIC paint + interlace + deconvolve, with optional oversampled Fourier crop.

        Follows montecosmo's approach:
        - Interlaced CIC painting (order 2) to cancel leading-order aliases
        - Deconvolution in Fourier space at paint resolution
        - Fourier crop via chreshape to final resolution (when oversampled)

        ``pos`` is in ``from_shape`` cell units (default: the final mesh grid; with
        evolution oversampling, the caller passes ``evol_shape``). The output always has
        shape ``mesh_shape`` (final grid).
        """
        if from_shape is None:
            from_shape = self._sim_mesh
        from_shape = tuple(int(s) for s in from_shape)
        paint_shape = tuple(int(s) for s in self.paint_shape)
        final_shape = tuple(int(s) for s in self._sim_mesh)

        if paint_shape == final_shape == from_shape:
            # No oversampling anywhere: plain CIC paint without deconvolution
            if weights is None:
                return cic_paint(jnp.zeros(final_shape), pos)
            return cic_paint(jnp.zeros(final_shape), pos, weights)

        # Rescale positions from `from_shape` grid units to paint_shape grid units, paint at
        # paint resolution (interlaced + deconvolved), then Fourier-crop to the final grid.
        scale_pos = jnp.asarray(np.asarray(paint_shape) / np.asarray(from_shape), dtype=pos.dtype)
        pos_paint = pos * scale_pos

        return interlace_paint_deconv(pos_paint, paint_shape, final_shape, weights=weights)

    def evolve(self, params: tuple):
        cosmology, bias, init, png = params

        fNL = png.get("fNL", 0.0)
        fNL_bp_lat = png.get("fNL_bp", 0.0)
        fNL_bpd_lat = png.get("fNL_bpd", 0.0)

        init_mesh = next(iter(init.values()))  # inferred linear field on the init grid
        _center = self._sim_center
        _bshape = self._sim_shape  # physical box, shared by all (oversampled) grids

        if self.evolution == "kaiser":
            # Linear evolution needs no oversampling; collapse the init field onto the final grid.
            init_mesh = chreshape(init_mesh, r2chshape(tuple(self._sim_mesh)))
            _mshape = self._sim_mesh
            if self.lightcone:
                los_mesh, a_evol = tophysical_mesh(
                    _center,
                    _bshape,
                    _mshape,
                    cosmology,
                    a_obs=None,
                    curved_sky=self.curved_sky,
                    los=self.los,
                )
                los_evol = None if self.los is None else los_mesh
            else:
                los_evol = self.los
                a_evol = self.a_obs

            # Local PNG scale-dependent galaxy bias (fNL_bp / M(k)). The matter field stays
            # Gaussian in linear Kaiser: the f_NL matter signal (phi^2 term) is 2nd-order and
            # only captured by the LPT/N-body add_png path.
            fNL_bp, _ = fNL_bias(
                fNL, bias["b1"], bias["b2"], p=1.0, png_type=self.png_type,
                fNL_bp=fNL_bp_lat, fNL_bpd=fNL_bpd_lat,
            )

            gxy_mesh = kaiser_model(
                cosmology, a_evol, bE=1 + bias["b1"], init_mesh=init_mesh, los=los_evol,
                fNL_bp=fNL_bp, png_type=self.png_type, box_shape=_bshape,
            )
            gxy_mesh = deterministic("gxy_mesh", gxy_mesh)

            # Matter mesh (linear growth, no bias, no RSD)
            matter_mesh = kaiser_model(cosmology, a_evol, bE=1.0, init_mesh=init_mesh, los=None)
            matter_mesh = deterministic("matter_mesh", matter_mesh)

            return {"gxy_mesh": gxy_mesh, "matter_mesh": matter_mesh}

        # LPT / N-body run on the (oversampled) evolution grid; particles on the ptcl grid.
        # The inferred init field is Fourier-padded up to the evolution grid (high-k modes
        # above the init Nyquist start at zero) so LPT mode-coupling and the bias products
        # (delta^2, s^2, nabla^2 delta, phi*delta, add_png's phi^2) are computed without
        # aliasing into the final band, then cropped back.
        _mshape = self.evol_shape
        init_mesh_evol_grid = chreshape(init_mesh, r2chshape(tuple(self.evol_shape)))

        # Create regular grid of particles in Lagrangian space and get local scale factors.
        pos_initial = regular_pos(tuple(self.evol_shape), tuple(self.ptcl_shape))
        _, _, _, a_initial = tophysical_pos(
            pos_initial,
            _center,
            _bshape,
            _mshape,
            cosmology,
            a_obs=self.a_obs,
            curved_sky=self.curved_sky,
            los=self.los,
        )

        # Primordial non-Gaussianity: modify the matter field that gets displaced
        # (affects matter_mesh and CMB-lensing kappa). The original Gaussian field
        # is kept for the scale-dependent galaxy-bias terms in lagrangian_weights.
        fNL_bp, fNL_bpd = 0.0, 0.0
        init_mesh_evol = init_mesh_evol_grid
        if self.png_type is not None:
            fNL_bp, fNL_bpd = fNL_bias(
                fNL, bias["b1"], bias["b2"], p=1.0, png_type=self.png_type,
                fNL_bp=fNL_bp_lat, fNL_bpd=fNL_bpd_lat,
            )
            init_mesh_evol = add_png(cosmology, fNL, init_mesh_evol_grid, _bshape)
            init_mesh_evol = chreshape(
                chreshape(init_mesh_evol, r2chshape(tuple(self.init_shape))),
                r2chshape(tuple(self.evol_shape)),
            )

        if self.evolution == "lpt":
            cosmology._workspace = {}  # HACK: temporary fix
            dpos, vel = lpt(
                cosmology,
                init_mesh=init_mesh_evol,
                pos=pos_initial,
                a=a_initial,
                order=self.lpt_order,
                grad_fd=False,
                lap_fd=False,
            )
            pos = pos_initial + dpos
            pos, vel = deterministic("lpt_pos", pos), vel

        elif self.evolution == "nbody":
            if self.lightcone:
                raise NotImplementedError("N-body lightcone is not implemented yet.")
            cosmology._workspace = {}  # HACK: temporary fix
            pos, vel = nbody_bf(
                cosmology,
                init_mesh=init_mesh_evol,
                pos=pos_initial,
                a=self.a_obs,
                n_steps=self.nbody_steps,
                grad_fd=False,
                lap_fd=False,
                snapshots=self.nbody_snapshots,
            )
            part = deterministic("nbody_pos", pos), vel
            pos, vel = tree.map(lambda x: x[-1], part)
        else:
            raise ValueError(f"Unknown evolution mode: {self.evolution}")

        # Preserve real-space positions for matter field painting (always needed for CMB)
        pos_real = pos

        # Paint unbiased matter field in real space (always needed for CMB lensing).
        # Particles are in evol-grid cell units; mean-normalise to 1 + delta (robust to
        # particle count, so unaffected by ptcl oversampling).
        matter_mesh = self.paint_and_deconv(pos_real, from_shape=self.evol_shape)
        matter_mesh = matter_mesh / jnp.mean(matter_mesh)
        matter_mesh = deterministic("matter_mesh", matter_mesh)

        # Galaxy-specific calculations (skip if galaxies disabled for efficiency)
        if self.galaxies_enabled:
            # bnpar (Finger-of-God) is not a lagrangian_weights term: it enters as a velocity
            # contribution projected onto the LOS in rsd().
            bnpar = bias.get("bnpar", 0.0)
            bias_w = {k: v for k, v in bias.items() if k != "bnpar"}

            # Lagrangian bias expansion weights, computed from the evol-grid Gaussian field
            # (anti-aliased products) and read at the initial particle positions.
            lbe_weights = lagrangian_weights(
                cosmology, a_initial, pos_initial, _bshape, **bias_w, init_mesh=init_mesh_evol_grid,
                fNL_bp=fNL_bp, fNL_bpd=fNL_bpd, png_type=self.png_type,
            )

            # Finger-of-God velocity term (b_nabla_parallel): full grad(delta) 3-vector at the
            # initial positions; projected onto the observer-dependent per-particle LOS in rsd().
            # Only meaningful with RSD (los set); skip the FFTs otherwise.
            fog_dvel = 0.0
            if self.los is not None:
                fog_dvel = lagrangian_fog_velocity(
                    cosmology, a_initial, pos_initial, _bshape, bnpar, init_mesh_evol_grid,
                )

            # Local LOS and scale factors at displaced positions.
            _, _, los_part, a_part = tophysical_pos(
                pos,
                _center,
                _bshape,
                _mshape,
                cosmology,
                a_obs=self.a_obs,
                curved_sky=self.curved_sky,
                los=self.los,
            )
            rsd_los = None if self.los is None else los_part

            # RSD displacement with local a and observer-dependent LOS; FoG dvel projected too.
            pos_rsd = pos + rsd(cosmology, vel, rsd_los, a_part, _bshape, _mshape, dvel=fog_dvel)
            pos_rsd = deterministic("rsd_pos", pos_rsd)

            # CIC paint weighted by Lagrangian bias expansion weights. The painted field is
            # counts-per-final-cell (mean = ptcl/final cells); divide by that fixed factor to
            # recover the 1 + delta_g overdensity (identity when ptcl_oversamp=1).
            ptcl_per_cell = float(np.prod(self.ptcl_shape) / np.prod(self._sim_mesh))
            gxy_mesh = self.paint_and_deconv(
                pos_rsd, weights=lbe_weights, from_shape=self.evol_shape
            ) / ptcl_per_cell
            gxy_mesh = deterministic("gxy_mesh", gxy_mesh)
        else:
            # CMB-only: create dummy gxy_mesh for API consistency
            gxy_mesh = jnp.zeros(self.mesh_shape)

        return {"gxy_mesh": gxy_mesh, "matter_mesh": matter_mesh, "pos_real": pos_real}

    def likelihood(self, gxy_mesh, matter_mesh=None, pos_real=None, cosmology=None, bias=None, temp=1.0, s_e=1.0, ngbars=None):
        """
        Gaussian field-level likelihood with shot-noise variance.

        Variance per cell = s_e**2 / n̄  (s_e rescales the
        shot-noise amplitude, s_e=1 => pure Poisson).
        Per-z n̄(z) is used when available (Abacus mode); otherwise the
        global gxy_count (closure mode).  Cells outside the survey mask
        are excluded via ``dist.mask``.
        """

        if self.observable == "field":
            # Galaxy likelihood (if enabled)
            if self.galaxies_enabled:
                selec = getattr(self, "selec_mesh", None)
                if selec is not None:
                    nbar = jnp.maximum(jnp.asarray(selec) * self.gxy_count, 1e-10)
                else:
                    nbar = self.gxy_count

                # Free per-shell mean density: the data overdensity was built
                # with a fixed n̄, so a per-shell relative amplitude alpha rescales the predicted
                # field (mean) and the shot-noise (var ∝ alpha) per radial bin. alpha=1 outside bins.
                mean_field = gxy_mesh
                if ngbars is not None and getattr(self, "gxy_shell_id", None) is not None:
                    alpha = self._ngbar_alpha_field(ngbars)
                    mean_field = alpha * gxy_mesh
                    nbar = nbar / jnp.maximum(alpha, 1e-6)

                variance = temp * (s_e**2 / nbar)
                gxy_dist = dist.Normal(mean_field, variance**0.5)

                if getattr(self, "gxy_occ_mask3d", None) is not None:
                    gxy_dist = gxy_dist.mask(jnp.asarray(self.gxy_occ_mask3d))

                obs_mesh = sample("obs", gxy_dist)
            else:
                # CMB-only mode: still return something for API compatibility
                obs_mesh = deterministic("obs_disabled", gxy_mesh)

            # CMB lensing likelihood (if enabled)
            if self.cmb_enabled and cosmology is not None and bias is not None:
                from desi_cmb_fli.cmb_lensing import (
                    compute_cl_high_z,
                    convergence_Born_spherical,
                )

                # pos_real is in evol-grid cell units; the Born integrator expects final-grid
                # cell units (0 ... mesh_shape-1). Rescale (identity when evol_oversamp=1).
                pos_cmb = pos_real * jnp.asarray(
                    np.asarray(self.mesh_shape) / np.asarray(self.evol_shape), dtype=pos_real.dtype
                )
                kappa_pred = convergence_Born_spherical(
                    cosmology,
                    pos_cmb,
                    self.box_shape,
                    self.mesh_shape,
                    self.observer_position,
                    self.cmb_r_shells,
                    self.cmb_a_shells,
                    self.cmb_d_r,
                    self.cmb_nside,
                    self.cmb_sim_mask,
                    self.cmb_z_source,
                    self.t_enter,
                    self.t_exit,
                    return_full=True,
                )

                kappa_pred = deterministic("kappa_pred", kappa_pred)

                if self.cmb_likelihood_mode == "pixel_exact":
                    loc_k = self.cmb_kl_U.T @ kappa_pred[self.cmb_obs_pix]
                    scale_k = jnp.sqrt(self.cmb_kl_var * temp)
                    sample("kappa_obs", dist.Normal(loc_k, scale_k))
                else:
                    # High-z correction to the CMB covariance.
                    if self.full_los_correction:
                        cl_high_z_1d = compute_cl_high_z(
                            cosmology,
                            self.ell_1d,
                            self.chi_boundary,
                            self.chi_high_z_max,
                            self.cmb_z_source,
                            mode=self.high_z_mode,
                            cl_cached=self.cl_high_z_cached,
                            gradients=self.high_z_gradients,
                            loc_fid=self.loc_fid,
                        )

                    total_cl_1d = jnp.asarray(self.nell_1d)
                    if self.full_los_correction:
                        total_cl_1d = total_cl_1d + cl_high_z_1d

                    loc_u = self.pack_kappa_map(kappa_pred)
                    var_l = jnp.matmul(self.cmb_M_ll, total_cl_1d * temp)
                    scale_u = jnp.sqrt(jnp.maximum(var_l[self.cmb_l_of_u], 1e-30)) * self.cmb_u_half
                    sample("kappa_obs", dist.Normal(loc_u, scale_u))

            return obs_mesh  # NOTE: mesh is 1+delta_obs

    def pack_alm(self, alm):
        """Pack complex a_lm into the flat real observable vector (l>=2 modes)."""
        return jnp.concatenate(
            [jnp.real(alm)[self.cmb_pack_re_idx], jnp.imag(alm)[self.cmb_pack_im_idx]]
        )

    def pack_kappa_map(self, kmap):
        """Mask a HEALPix kappa map, transform to a_lm, and pack to the real observable vector."""
        import jax_healpy as jhp

        W = jnp.asarray(self.cmb_mask, dtype=kmap.dtype)
        alm = jhp.map2alm(W * kmap, lmax=self.cmb_lmax, pol=False, iter=0, healpy_ordering=True)
        return self.pack_alm(alm)

    def pack_kappa_obs(self, kmap):
        """Project a HEALPix kappa map onto the ``kappa_obs`` observable of the active mode.

        diagonal -> packed masked pseudo-a_lm vector;
        pixel_exact (KL) -> amplitudes on the k supported eigenmodes, a = U_k^T kappa[obs_pix].
        """
        if self.cmb_likelihood_mode == "pixel_exact":
            return self.cmb_kl_U.T @ jnp.asarray(kmap)[self.cmb_obs_pix]
        return self.pack_kappa_map(kmap)

    def unpack_kappa_obs_to_map(self, vec):
        """Reconstruct a HEALPix map from the ``kappa_obs`` observable (display only)."""
        if self.cmb_likelihood_mode == "pixel_exact":
            # Project the eigenmode amplitudes back to observed pixels (U_k a), then scatter.
            npix = 12 * self.cmb_nside**2
            vec = jnp.asarray(vec)
            pix_vals = self.cmb_kl_U @ vec
            return jnp.zeros(npix, dtype=pix_vals.dtype).at[self.cmb_obs_pix].set(pix_vals)
        return self.unpack_to_map(vec)

    def unpack_u(self, u):
        """Inverse of pack_alm: scatter the real vector back to a complex a_lm array.

        Modes with l<2 stay zero and the m=0 imaginary part stays zero. For tests / mock use.
        """
        n_alm = self.cmb_alm_l.shape[0]
        alm = jnp.zeros(n_alm, dtype=jnp.complex128)
        n_re = self.cmb_pack_re_idx.shape[0]
        alm = alm.at[self.cmb_pack_re_idx].add(u[:n_re].astype(jnp.complex128))
        alm = alm.at[self.cmb_pack_im_idx].add(1j * u[n_re:])
        return alm

    def unpack_to_map(self, u):
        """Reconstruct a real HEALPix map from the packed observable (for display only)."""
        import jax_healpy as jhp

        alm = self.unpack_u(jnp.asarray(u))
        return jnp.real(
            jhp.alm2map(alm, nside=self.cmb_nside, lmax=self.cmb_lmax, pol=False, healpy_ordering=True)
        )

    def reparam(self, params: dict, fourier=True, inv=False, temp=1.0):
        """
        Transform sample params into base params (inv=False) or vice versa (inv=True).
        """
        # Extract groups from params
        groups = ["cosmo", "bias", "png", "stoch", "syst", "init"]
        key = tuple([k if inv else k + "_"] for k in groups) + (
            ["*"] + ["~" + k if inv else "~" + k + "_" for k in groups],
        )
        params = Chains(params, self.groups | self.groups_).get(key)  # use chain querying
        cosmo_, bias_, png_, stoch_, syst_, init, rest = (q.data for q in params)

        # Cosmology
        cosmo = samp2base(cosmo_, self.latents, inv=inv, temp=temp)

        # Primordial non-Gaussianity (empty unless png_type is set)
        png = samp2base(png_, self.latents, inv=inv, temp=temp) if len(png_) > 0 else {}

        # Stochastic noise (empty unless gxy_stoch_noise is enabled)
        stoch = samp2base(stoch_, self.latents, inv=inv, temp=temp) if len(stoch_) > 0 else {}

        # Free per-shell mean density (empty unless gxy_ngbar_free is enabled)
        syst = samp2base(syst_, self.latents, inv=inv, temp=temp) if len(syst_) > 0 else {}

        # Biases
        if inv and self.cmb_enabled and not self.galaxies_enabled:
            bias = {}
        elif len(bias_) > 0:
            bias = samp2base(bias_, self.latents, inv=inv, temp=temp)
        else:
            bias = {k: self.loc_fid[k] for k in self.groups["bias"]}

        # Initial conditions
        if len(init) > 0:
            cosmology = get_cosmology(**(cosmo_ if inv else cosmo))

            if self.cmb_enabled and not self.galaxies_enabled:
                bE = 1 + self.loc_fid["b1"]
            elif inv:
                bE = 1 + bias_["b1"]
            else:
                bE = 1 + bias["b1"]

            _, transfer = self._precond_scale_and_transfer(cosmology, bE)

            if not fourier and inv:
                init = tree.map(lambda x: jnp.fft.rfftn(x), init)

            init = samp2base_mesh(init, self.precond, transfer=transfer, inv=inv, temp=temp)

            if not fourier and not inv:
                init = tree.map(lambda x: jnp.fft.irfftn(x), init)

        return rest | cosmo | bias | png | stoch | syst | init  # possibly update rest

    ###########
    # Getters #
    ###########
    def _validate_latents(self):
        """
        Return a validated latents config.
        """
        new = {}
        for name, conf in self.latents.items():
            new[name] = conf.copy()
            loc, scale = conf.get("loc"), conf.get("scale")
            low, high = conf.get("low"), conf.get("high")
            loc_fid, scale_fid = conf.get("loc_fid"), conf.get("scale_fid")

            assert not (loc is None) ^ (scale is None), (
                f"latent '{name}' not valid: loc and scale must be both provided or both not provided"
            )
            assert not (low is None) ^ (high is None), (
                f"latent '{name}' not valid: low and high must be both provided or both not provided"
            )

            if loc is not None:  # Normal or Truncated Normal prior
                if loc_fid is None:
                    new[name]["loc_fid"] = loc
                if scale_fid is None:
                    new[name]["scale_fid"] = scale

            elif low is not None:  # Uniform prior
                assert low <= high, f"latent '{name}' not valid: low must be lower than high"
                assert low != -jnp.inf and high != jnp.inf, (
                    f"latent '{name}' not valid: low and high must be finite for uniform distribution"
                )
                if loc_fid is None:
                    new[name]["loc_fid"] = (low + high) / 2
                if scale_fid is None:
                    new[name]["scale_fid"] = (high - low) / 12**0.5
        return new

    def _sample(self, names: str | list):
        """
        Sample latent parameters from latents config.
        """
        dic = {}
        names = np.atleast_1d(names)
        for name in names:
            conf = self.latents[name]
            loc, scale = conf.get("loc", None), conf.get("scale", None)
            low, high = conf.get("low", -jnp.inf), conf.get("high", jnp.inf)
            loc_fid, scale_fid = conf["loc_fid"], conf["scale_fid"]

            if loc is not None:
                if low == -jnp.inf and high == jnp.inf:
                    dic[name + "_"] = sample(
                        name + "_", dist.Normal((loc - loc_fid) / scale_fid, scale / scale_fid)
                    )
                else:
                    dic[name + "_"] = sample(
                        name + "_", DetruncTruncNorm(loc, scale, low, high, loc_fid, scale_fid)
                    )
            else:
                dic[name + "_"] = sample(name + "_", DetruncUnif(low, high, loc_fid, scale_fid))
        return dic

    def _precond_scale_and_transfer(self, cosmo: Cosmology, bE):
        """
        Return scale and transfer fields for linear matter field preconditioning.

        When a selection function is available, the effective noise uses
        ``selec_rms² = ⟨selec²⟩`` which accounts
        for both survey coverage and density variation along the line of sight.
        Otherwise falls back to ``1/n̄``.
        """
        # The inferred linear field lives on the (possibly oversampled) init grid.
        pmeshk = lin_power_mesh(cosmo, self.init_shape, self._sim_shape)

        # Effective galaxy count: 0 if galaxies disabled, actual count otherwise
        gxy_count_eff = self.gxy_count if self.galaxies_enabled else 0.0

        # Selection-based effective noise:
        # noise = 1 / (n̄ · selec_rms²) where selec_rms² = ⟨W²⟩ over the
        # full mesh, capturing both f_sky and n(z) variation.
        selec = getattr(self, "selec_mesh", None)
        if selec is not None and gxy_count_eff > 0:
            selec_rms2 = jnp.mean(jnp.asarray(selec) ** 2)
            noise_eff = 1.0 / (gxy_count_eff * jnp.maximum(selec_rms2, 1e-30))
        else:
            noise_eff = 1.0 / gxy_count_eff if gxy_count_eff > 0 else float("inf")

        if self.precond in ["direct", "fourier"]:
            scale = jnp.ones(self.init_shape)
            transfer = pmeshk**0.5

        elif self.precond == "kaiser":
            cosmo_fid, bE_fid = get_cosmology(**self.loc_fid), 1 + self.loc_fid["b1"]
            boost_fid = kaiser_boost(cosmo_fid, self.a_fid, bE_fid, self.init_shape, self.los)
            pmeshk_fid = lin_power_mesh(cosmo_fid, self.init_shape, self._sim_shape)

            scale = (1 + boost_fid**2 * pmeshk_fid / noise_eff) ** 0.5
            transfer = pmeshk**0.5 / scale
            scale = cgh2rg(scale, norm="amp")

        elif self.precond == "kaiser_dyn":
            boost = kaiser_boost(cosmo, self.a_fid, bE, self.init_shape, self.los)

            scale = (1 + boost**2 * pmeshk / noise_eff) ** 0.5
            transfer = pmeshk**0.5 / scale
            scale = cgh2rg(scale, norm="amp")

        return scale, transfer

    def _groups(self, base=True):
        """
        Return groups from latents config.

        The 'png' group (primordial non-Gaussianity) is omitted when
        ``png_type is None`` so Gaussian runs are unaffected by the fNL latent.
        """
        groups = {}
        for name, val in self.latents.items():
            group = val["group"]
            if group == "png" and self.png_type is None:
                continue
            if name in ("fNL_bp", "fNL_bpd") and self.png_type != "fNL_bias":
                continue
            if group == "stoch" and not self.gxy_stoch_noise:
                continue
            if group == "syst" and not self.gxy_ngbar_free:
                continue
            group = group if base else group + "_"
            if group not in groups:
                groups[group] = []
            groups[group].append(name if base else name + "_")
        return groups

    def _labels(self):
        """
        Return labels from latents config
        """
        labs = {}
        for name, val in self.latents.items():
            if val["group"] == "png" and self.png_type is None:
                continue
            if name in ("fNL_bp", "fNL_bpd") and self.png_type != "fNL_bias":
                continue
            if val["group"] == "stoch" and not self.gxy_stoch_noise:
                continue
            lab = val["label"]
            labs[name] = lab
            labs[name + "_"] = "\\tilde" + lab
        return labs

    def _loc_fid(self):
        """
        Return fiducial location values from latents config.
        """
        return {
            k: v["loc_fid"]
            for k, v in self.latents.items()
            if "loc_fid" in v
            and not (v["group"] == "png" and self.png_type is None)
            and not (k in ("fNL_bp", "fNL_bpd") and self.png_type != "fNL_bias")
            and not (v["group"] == "stoch" and not self.gxy_stoch_noise)
        }

    ###########
    # Metrics #
    ###########
    def spectrum(self, mesh, mesh2=None, kedges: int | float | list = None, comp=(0, 0), poles=0):
        return spectrum(
            mesh,
            mesh2=mesh2,
            box_shape=self.box_shape,
            kedges=kedges,
            comp=comp,
            poles=poles,
            los=self.los,
        )

    def powtranscoh(self, mesh0, mesh1, kedges: int | float | list = None, comp=(0, 0)):
        return powtranscoh(mesh0, mesh1, box_shape=self.box_shape, kedges=kedges, comp=comp)

    ########################
    # Chains init and load #
    ########################
    def load_runs(self, path: str, start: int, end: int, transforms=None, batch_ndim=2) -> Chains:
        return Chains.load_runs(
            path,
            start,
            end,
            transforms,
            groups=self.groups | self.groups_,
            labels=self.labels,
            batch_ndim=batch_ndim,
        )

    def reparam_chains(self, chains: Chains, fourier=False, batch_ndim=2):
        chains = chains.copy()
        chains.data = nvmap(partial(self.reparam, fourier=fourier), batch_ndim)(chains.data)
        return chains

    def powtranscoh_chains(
        self,
        chains: Chains,
        mesh0,
        name: str = "init_mesh",
        kedges: int | float | list = None,
        comp=(0, 0),
        batch_ndim=2,
    ) -> Chains:
        chains = chains.copy()
        fn = nvmap(lambda x: self.powtranscoh(mesh0, x, kedges=kedges, comp=comp), batch_ndim)
        chains.data["kptc"] = fn(chains.data[name])
        return chains

    def kaiser_post(self, rng, delta_obs, base=False, temp=1.0, scale_field=1.0):
        _mask3d = getattr(self, "gxy_occ_mask3d", None)
        if _mask3d is not None:
            _m = jnp.asarray(_mask3d)
            delta_obs = jnp.where(_m, delta_obs, 0.0)
            n_signal_cells = jnp.sum(_m)
        else:
            n_signal_cells = None

        # Compute f_sky on the FFT grid.
        if n_signal_cells is not None:
            f_sky = n_signal_cells / delta_obs.size
            f_sky = jnp.maximum(f_sky, 0.01)  # safety floor
        else:
            f_sky = 1.0

        if jnp.isrealobj(delta_obs):
            delta_obs = jnp.fft.rfftn(delta_obs)

        # Pseudo-Cℓ correction: zeroing (1 - f_sky) of cells dilutes the
        # FFT signal by f_sky.  Divide by f_sky to unbias.
        delta_obs = delta_obs / f_sky

        cosmo_fid, bE_fid = get_cosmology(**self.loc_fid), 1 + self.loc_fid["b1"]
        # Effective noise: after dividing the FFT by f_sky, noise is amplified
        # by 1/f_sky.  Use selec_rms² when available (accounts for both f_sky
        # and n(z) variation); fall back to binary f_sky otherwise.
        _selec = getattr(self, "selec_mesh", None)
        if _selec is not None:
            selec_rms2 = jnp.mean(jnp.asarray(_selec) ** 2)
            gxy_count_eff = self.gxy_count * selec_rms2
        else:
            gxy_count_eff = self.gxy_count * f_sky

        means, stds = kaiser_posterior(
            delta_obs, cosmo_fid, bE_fid, self.a_fid, self._sim_shape, gxy_count_eff, self.los
        )
        post_mesh = rg2cgh(jr.normal(rng, ch2rshape(means.shape)))
        post_mesh = temp**0.5 * stds * post_mesh + means

        post_mesh *= scale_field

        # The Wiener init is built at the data (final) grid; the inferred field lives on the
        # init grid. Pad with zeros above the final Nyquist (those modes start prior-driven).
        if tuple(self.init_shape) != tuple(self._sim_mesh):
            post_mesh = chreshape(post_mesh, r2chshape(tuple(self.init_shape)))

        init_params = self.loc_fid | {"init_mesh": post_mesh}
        if base:
            return init_params
        else:
            return self.reparam(init_params, inv=True)
