#!/usr/bin/env python
"""Forward-model the AbacusSummit initial conditions and compare C_κg with Abacus.

The joint counterpart of `validate_kappa_from_ic.py` and `validate_galaxy_from_ic.py`: one prediction
of the joint model at the true ICs (Abacus cosmology, the biases of `--profile`, a Newton or amplitude
profile JSON of `galaxy_likelihood_profile.py`) gives a κ map (the likelihood's mean map, with the
out-of-model gain) and a galaxy map; the AbacusLensing κ map and the LRG catalogue give the same two
maps for the simulation. With the initial conditions shared, cosmic variance cancels: per ℓ band,
C_κg of Abacus over C_κg of the model is 1 when κ and the galaxies of the model are consistent with
each other as in the simulation (same matter, same distances, same light-cone time). Error bars:
the Gaussian scatter of C_κg^Abacus − C_κg^model when the κ error (Abacus − model κ) and the galaxy
error (Abacus − model galaxies, shot noise included) are independent of the model maps, as the joint
likelihood assumes, with f_sky the overlap of the two masks. The mixed crosses (model κ × Abacus
galaxies, Abacus κ × model galaxies), which say on which side a departure sits, go to the JSON.

Galaxy maps from the galaxies themselves (`validation.galaxy_healpix_delta`: catalogue counts for
Abacus with their pixel window and shot noise removed, bias-weighted particles spread bilinearly for
the model), κ maps on the projection sphere (both carry its bilinear window, which cancels in the
ratios), all band-limited to ℓ ≤ 2·cmb_nside on the κ mask; and with the galaxy maps the likelihood
holds: the final-grid count meshes of the data and of the model projected along each ray (the modes of
the final cube only). The figure shows the second, the spectra and their ratio; the JSON holds both. Above
ℓ ≈ 160 the per-shell cut of κ removes matter that galaxies beyond the cube still trace, which only the
catalogue-projected version sees.

    python scripts/validate_joint_from_ic.py \\
        --config configs/inference/abacus/abacus_joint_so_cosmo_box5000_m100_ph202.yaml \\
        --profile figures/galaxy_likelihood_profile/abacus_gxyonly_cosmo_box5000_m100_ph202_fog_newton.json

Writes figures/joint_ic_test/<config stem>.{png,json}. CPU node (JAX_PLATFORMS=cpu) or GPU.
"""

import argparse
import json
from pathlib import Path

import healpy as hp
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import numpy as np
import yaml
from scipy.ndimage import map_coordinates

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from desi_cmb_fli.bricks import band_limit, get_cosmology  # noqa: E402
from desi_cmb_fli.cmb_lensing import (  # noqa: E402
    bilinear_window,
    load_abacus_galaxy_observation,
    load_abacus_ic_truth,
    load_abacus_kappa_observation,
)
from desi_cmb_fli.metrics import get_cl_healpix  # noqa: E402
from desi_cmb_fli.model import get_model_from_config  # noqa: E402
from desi_cmb_fli.validation import conditioning_params, galaxy_healpix_delta  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BLUE, INK = "#2a78d6", "#1f1f1e"


def biases_from(path):
    src = json.loads(Path(path).read_text())
    if src.get("mode") == "newton":
        return {k: v for k, v in src["iterations"][-1]["theta"].items() if k != "A"}
    return next(p for p in src["profile"] if p["value"] == 1.0)["biases"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--profile", required=True, type=Path)
    ap.add_argument("--edges", type=float, nargs="+", default=[2, 10, 20, 40, 70, 110, 160, 257])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["galaxies_enabled"] = True
    model, _ = get_model_from_config(cfg)
    tr = load_abacus_galaxy_observation(abacus_gxy_cfg=cfg["abacus_galaxy"], model=model)
    nside_p, lmax = int(model.cmb_proj_nside), int(model.cmb_lmax)
    sky = Path(cfg["abacus_kappa"]["file"]).parent.name
    cache = REPO / "data" / "cache" / f"abacus_kappa_{sky}_nside{nside_p}.npy"
    if cache.exists():
        k_data = np.load(cache)
    else:
        k_data = np.asarray(load_abacus_kappa_observation(cfg["abacus_kappa"], model)["kappa_pred"])
        np.save(cache, k_data)

    biases = biases_from(args.profile)
    params = conditioning_params(model, cfg.get("truth_params", {}),
                                 cfg.get("abacus_truth_params", {}), biases)
    ic = load_abacus_ic_truth(cfg["abacus_ic"], model)["init_mesh"]
    model.reset()
    pred = model.predict(samples={**params, "init_mesh": jnp.asarray(ic)}, hide_base=False,
                         hide_samp=False, hide_det=False, frombase=True, rng=jr.key(0))
    k_model = np.asarray(pred["kappa_mean"], dtype=float)
    print(f"biases {biases}; kappa maps {k_model.size} / {k_data.size} pixels", flush=True)

    # Galaxy maps on the projection sphere, kappa band-limited onto it on the kappa mask.
    chi_max = float(getattr(model, "chi_boundary", np.min(model.box_shape) / 2))
    g_model = galaxy_healpix_delta({"rsd_pos": pred["rsd_pos"], "gxy_weights": pred["gxy_weights"]},
                                   model, nside_p, chi_max)
    g_data = galaxy_healpix_delta({"gxy_hp_counts": tr["gxy_hp_counts"]}, model, nside_p, chi_max)
    kmask = np.asarray(model.cmb_proj_mask, dtype=bool)

    def kappa_on_mask(m):
        return hp.alm2map(hp.map2alm(m * kmask, lmax=lmax), nside_p, lmax=lmax)[kmask]

    km, kd = kappa_on_mask(k_model), kappa_on_mask(k_data)
    w_bil, w_pix = bilinear_window(nside_p, lmax), hp.pixwin(nside_p, lmax=lmax)
    gwin = {"model": w_bil, "data": w_pix}
    gmap = {"model": g_model, "data": g_data}
    kmap = {"model": km, "data": kd}

    def cl(a, ma, b, mb):
        ell, c, _ = get_cl_healpix(a, ma, b, mb, lmax=lmax)
        return np.asarray(ell), np.asarray(c)

    ell, _ = cl(km, kmask, None, None)
    res = {}
    for kk in ("model", "data"):
        for gg in ("model", "data"):
            _, c = cl(kmap[kk], kmask, gmap[gg]["delta_masked"], gmap[gg]["mask_full"])
            res[f"kg_{kk}_{gg}"] = c / gwin[gg][: c.size]
    _, kk_mm = cl(km, kmask, None, None)
    _, kk_dd = cl(kd, kmask, None, None)
    _, kk_md = cl(km, kmask, kd, kmask)
    _, gg_mm = cl(g_model["delta_masked"], g_model["mask_full"], None, None)
    _, gg_dd = cl(g_data["delta_masked"], g_data["mask_full"], None, None)
    _, gg_md = cl(g_model["delta_masked"], g_model["mask_full"], g_data["delta_masked"], g_data["mask_full"])
    gg_mm, gg_dd = gg_mm / w_bil[: gg_mm.size] ** 2, (gg_dd - g_data["shot"]) / w_pix[: gg_dd.size] ** 2
    gg_md = gg_md / (w_bil * w_pix)[: gg_md.size]

    # Gaussian variance of C_kg^data - C_kg^model per ell, the kappa and galaxy errors independent
    # of the model maps; the galaxy error carries the shot noise the data spectrum had removed.
    e_k = kk_dd - 2 * kk_md + kk_mm
    e_g = gg_dd + g_data["shot"] / w_pix[: gg_dd.size] ** 2 - 2 * gg_md + gg_mm
    fsky = float(np.mean(kmask & np.asarray(g_data["mask_full"], dtype=bool)))
    var_l = (e_k * gg_mm + kk_mm * e_g + e_k * e_g) / ((2 * ell + 1) * fsky)

    edges = np.asarray(args.edges)
    centre = 0.5 * (edges[1:] + edges[:-1])
    sel = [(ell >= lo) & (ell < hi) for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    w = 2 * ell + 1

    def band(x):
        return np.array([np.sum((w * x)[m]) / np.sum(w[m]) for m in sel])

    ref = band(res["kg_model_model"])
    sigma = np.array([np.sqrt(np.sum((w**2 * var_l)[m])) / np.sum(w[m]) for m in sel]) / ref

    # The galaxies as the likelihood holds them: the final-grid count meshes (data, and the model's
    # at the same biases) projected along each ray of the projection sphere, trilinear on the cell
    # nodes, survey cells only; they hold only the modes of the final cube.
    obs3, mask3 = np.asarray(tr["obs"], float), np.asarray(tr["gxy_occ_mask3d"], bool)
    nbar3 = np.asarray(model.gxy_count * jnp.asarray(model.selec_mesh), float)
    truth = {**cfg.get("truth_params", {}), **cfg.get("abacus_truth_params", {})}
    cosmo = get_cosmology(Omega_m=float(truth["Omega_m"]), sigma8=float(truth["sigma8"]))
    f = model.evolve((cosmo, biases, {"init_mesh": jnp.asarray(ic)}, {"fNL": 0.0}))
    m3 = np.asarray(model.gxy_count * band_limit(f["gxy_paint"] * jnp.asarray(model.selec_paint),
                                                 model._sim_mesh), float)
    cell = float(model.box_shape[0]) / obs3.shape[0]
    obsv = np.asarray(model.observer_position, float)
    chi_s = np.linspace(1080.0, 2460.0, 139)
    vec = np.array(hp.pix2vec(nside_p, np.arange(hp.nside2npix(nside_p)))).T

    def project(field):
        num, den = np.zeros(len(vec)), np.zeros(len(vec))
        fz, nz = np.where(mask3, field - nbar3, 0.0), np.where(mask3, nbar3, 0.0)
        for s0 in range(0, len(vec), 32768):
            v = vec[s0 : s0 + 32768]
            pts = ((obsv + chi_s[None, :, None] * v[:, None, :]) / cell).reshape(-1, 3).T
            num[s0 : s0 + 32768] = map_coordinates(fz, pts, order=1, mode="grid-wrap").reshape(len(v), -1).sum(1)
            den[s0 : s0 + 32768] = map_coordinates(nz, pts, order=1, mode="grid-wrap").reshape(len(v), -1).sum(1)
        return num / np.maximum(den, 1e-12)

    full = np.ones(hp.nside2npix(nside_p), bool)
    gl = {"model": project(m3), "data": project(obs3)}
    _, lp_dd = cl(kd, kmask, gl["data"], full)
    _, lp_mm = cl(km, kmask, gl["model"], full)
    _, lg_mm = cl(gl["model"], full, None, None)
    _, lg_dd = cl(gl["data"], full, None, None)
    _, lg_md = cl(gl["model"], full, gl["data"], full)
    e_gl = lg_dd - 2 * lg_md + lg_mm  # the data map keeps its shot noise: it is part of the error
    var_lp = (e_k * lg_mm + kk_mm * e_gl + e_k * e_gl) / ((2 * ell + 1) * float(kmask.mean()))
    ref_lp = band(lp_mm)
    sigma_lp = np.array([np.sqrt(np.sum((w**2 * var_lp)[m])) / np.sum(w[m]) for m in sel]) / ref_lp
    out = {
        "config": args.config, "profile": str(args.profile), "ell_edges": edges.tolist(), "fsky": fsky,
        "kg_data_over_model": (band(res["kg_data_data"]) / ref).tolist(),
        "sigma": sigma.tolist(),
        "kg_modelkappa_datagal_over_model": (band(res["kg_model_data"]) / ref).tolist(),
        "kg_datakappa_modelgal_over_model": (band(res["kg_data_model"]) / ref).tolist(),
        "kg_model": ref.tolist(),
        "kg_data": band(res["kg_data_data"]).tolist(),
        "kg_data_over_model_final_cube": (band(lp_dd) / ref_lp).tolist(),
        "kg_model_final_cube": ref_lp.tolist(),
        "kg_data_final_cube": band(lp_dd).tolist(),
        "sigma_final_cube": sigma_lp.tolist(),
    }
    for k, v in out.items():
        if isinstance(v, list) and k != "ell_edges":
            print(f"{k:36s} " + " ".join(f"{x:6.3f}" for x in v), flush=True)

    xerr = [centre - edges[:-1], edges[1:] - centre]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(5.5, 5.2), sharex=True,
                                  gridspec_kw={"height_ratios": [1.3, 1]})
    # the figure shows the galaxies the joint likelihood holds; the catalogue-projected version is in the JSON
    ax.errorbar(centre, centre * np.asarray(out["kg_data_final_cube"]), xerr=xerr, fmt="o", color=BLUE, ms=5,
                lw=1.2, label="AbacusLensing × LRG")
    ax.errorbar(centre, centre * np.asarray(out["kg_model_final_cube"]), xerr=xerr, fmt="s", color=INK, ms=4,
                lw=1.0, mfc="none", label="model from the Abacus ICs")
    ax.set(ylabel=r"$\ell\,C_\ell^{\kappa g}$")
    ax.legend(frameon=False, fontsize=8)
    axr.axhline(1, color=INK, lw=0.8)
    axr.errorbar(centre, out["kg_data_over_model_final_cube"], yerr=sigma_lp, xerr=xerr, fmt="o", color=BLUE,
                 ms=5, lw=1.2, capsize=0)
    axr.set(xscale="log", xlabel=r"$\ell$", ylabel="Abacus / model")
    for a_ in (ax, axr):
        a_.grid(alpha=0.2)
    fig.tight_layout()
    odir = REPO / "figures" / "joint_ic_test"
    odir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.config).stem
    fig.savefig(odir / f"{stem}.png", dpi=200)
    (odir / f"{stem}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"Saved {odir / stem}.png and .json")


if __name__ == "__main__":
    main()
