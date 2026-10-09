#!/usr/bin/env python
"""The particle-lattice template of the κ model (pipeline §2.6, §7.15).

Averaged over the field, the model's κ is not zero: the particle lattice leaves a deterministic mean
E_z[κ](θ). It is measured with antithetic pairs, (κ(z) + κ(−z))/2, which cancel the part of κ odd in
the field, in the packed a_ℓm of the likelihood divided by `cmb_u_half`.

    # E_z[κ] and the field power on an (Omega_m, sigma8) grid (CPU: one process per subset of points)
    python scripts/kappa_lattice_template.py measure --config CFG --om 0.20 0.27 0.315192 \\
        --s8 0.65 0.811355 --pairs 8 --out data/cache/template/CFGSTEM_part0.npz
    # another geometry: --box 5000 --mesh 54 --particles 200 (central cube, same cell)
    # S/N, projection of a run's data on it, and the Gaussian κ posterior with and without it
    python scripts/kappa_lattice_template.py analyse --grid 'data/cache/template/CFGSTEM_*.npz' \\
        --data RUN_DIR [--data RUN_DIR2] --json figures/kappa_lattice_template/CFGSTEM.json

`measure` saves, per point, the pair means (n_pairs, n_u), the mean field power ⟨u²⟩, the κ of the
undisplaced lattice (zero field) and the likelihood's N_ℓ and line-of-sight term. `analyse` reports
S/N = (Σ T²/C)^½ with C = S_ℓ + N_ℓ + LOS_ℓ (estimation noise subtracted in quadrature), the ratio of T
to the undisplaced lattice, the projection α = Σ yT/C / Σ T²/C of each data set, and, when the points
form a grid, the posterior of −2 ln L = Σ (y − T)²/C + ln C (diagonal, the run's priors) with T as mean
and without it.
"""

import argparse
import copy
import glob
import json
from pathlib import Path

import numpy as np
import yaml

OM_FID, S8_FID = 0.315192, 0.811355


def build_model(cfg_path, box=None, mesh=None, particles=None):
    from desi_cmb_fli.model import get_model_from_config

    cfg = copy.deepcopy(yaml.safe_load(open(cfg_path)))
    m = cfg["model"]
    if box is not None:
        m["box_shape"] = [float(box)] * 3
        m["cell_size"] = float(box) / int(mesh)
    if particles is not None:
        n_final = (
            int(mesh)
            if mesh is not None
            else round(float(m["box_shape"][0]) / float(m["cell_size"]))
        )
        m["evol_oversamp"] = m["ptcl_oversamp"] = float(particles) / n_final
    model, _ = get_model_from_config(cfg)
    return model, cfg


def measure(args):
    import jax.numpy as jnp
    import jax.random as jr

    model, _ = build_model(args.config, args.box, args.mesh, args.particles)
    u_half = np.asarray(model.cmb_u_half)
    lmax = int(np.asarray(model.cmb_l_of_u).max())
    spacing = float(model.box_shape[0]) / float(model.evol_shape[0])
    print(
        f"box {float(model.box_shape[0]):.1f}, final {tuple(int(x) for x in model.mesh_shape)}, "
        f"init {tuple(int(x) for x in model.init_shape)}, particle spacing {spacing:.2f} Mpc/h",
        flush=True,
    )

    def kappa(om, s8, init_mesh=None, rng=0):
        s = {"Omega_m": om, "sigma8": s8, "fNL": 0.0}
        if init_mesh is not None:
            s["init_mesh"] = jnp.asarray(init_mesh, jnp.complex64)
        p = model.predict(
            samples=s,
            hide_base=False,
            hide_samp=False,
            hide_det=False,
            frombase=True,
            rng=jr.key(rng),
        )
        return np.asarray(model.pack_kappa_map(jnp.asarray(p["kappa_pred"]))) / u_half, np.asarray(
            p["init_mesh"]
        )

    points, pairs, power, zero = [], [], [], []
    for om in args.om:
        for s8 in args.s8:
            T, P = [], []
            for i in range(args.pairs):
                u1, z = kappa(om, s8, rng=args.seed + i)
                u2, _ = kappa(om, s8, init_mesh=-z)
                T.append(0.5 * (u1 + u2))
                P.append(0.5 * (u1**2 + u2**2))
            points.append((om, s8))
            pairs.append(np.array(T, np.float32))
            power.append(np.mean(P, 0))
            zero.append(kappa(om, s8, init_mesh=np.zeros_like(z))[0])
            print(f"({om:.4f}, {s8:.4f}) done", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        points=np.array(points),
        pairs=np.array(pairs),
        power=np.array(power),
        zero=np.array(zero),
        l_of_u=np.asarray(model.cmb_l_of_u),
        N=np.asarray(model.cmb_M_ll @ jnp.asarray(model.nell_1d))[: lmax + 1],
        LOS=np.asarray(model.cmb_M_ll @ jnp.asarray(model.cl_high_z_cached))[: lmax + 1],
        u_half=u_half,
        spacing=spacing,
        box=float(model.box_shape[0]),
    )
    print(f"Saved {args.out}")


def load_grid(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    d0 = np.load(files[0])
    pts, pairs, power, zero = [], [], [], []
    for f in files:
        d = np.load(f)
        pts += [tuple(np.round(p, 6)) for p in d["points"]]
        pairs += list(d["pairs"])
        power += list(d["power"])
        zero += list(d["zero"])
    return {
        "points": pts,
        "pairs": pairs,
        "power": power,
        "zero": zero,
        "l": d0["l_of_u"],
        "N": d0["N"],
        "LOS": d0["LOS"],
        "u_half": d0["u_half"],
        "spacing": float(d0["spacing"]),
        "box": float(d0["box"]),
    }


def covariance(g, i):
    """Diagonal covariance per packed mode: field power minus the template's, smoothed in ℓ, + N + LOS."""
    ell_u = g["l"]
    ells = np.arange(2, ell_u.max() + 1)
    T = g["pairs"][i].mean(0)
    S = np.array([g["power"][i][ell_u == e].mean() - (T[ell_u == e] ** 2).mean() for e in ells])
    S = np.exp(np.polyval(np.polyfit(np.log(ells), np.log(S), 6), np.log(ells)))
    return np.concatenate([[np.inf, np.inf], S + (g["N"] + g["LOS"])[2:]])[ell_u]


def data_vector(run_dir, u_half):
    t = np.load(Path(run_dir) / "config" / "truth.npz")
    return np.asarray(t["kappa_obs_packed"]) / u_half


def analyse(args):
    from scipy.interpolate import RectBivariateSpline

    g = load_grid(args.grid)
    ell_u, sel = g["l"], g["l"] >= 2
    fid = min(
        range(len(g["points"])),
        key=lambda i: abs(g["points"][i][0] - OM_FID) + abs(g["points"][i][1] - S8_FID),
    )
    T = g["pairs"][fid].mean(0)
    C = covariance(g, fid)
    e2 = g["pairs"][fid].std(0) ** 2 / len(g["pairs"][fid])
    out = {
        "grid": args.grid,
        "box": g["box"],
        "particle_spacing": g["spacing"],
        "fiducial_point": g["points"][fid],
    }

    def sn(m):
        return float(np.sqrt(max(np.sum(T[m] ** 2 / C[m]) - np.sum(e2[m] / C[m]), 0.0)))

    out["S/N"] = {"l<=64": sn(sel), "l<=47": sn(sel & (ell_u <= 47))}
    z = g["zero"][fid]
    out["template/undisplaced"] = float(
        np.sum(T[sel] * z[sel] / C[sel]) / np.sum(z[sel] ** 2 / C[sel])
    )
    F = np.sum(T[sel] ** 2 / C[sel])
    out["projection"] = {}
    for run in args.data or []:
        y = data_vector(run, g["u_half"])
        out["projection"][run] = [float(np.sum(y[sel] * T[sel] / C[sel]) / F), float(F**-0.5)]
    print(json.dumps({k: v for k, v in out.items() if k != "grid"}, indent=1))

    om_g = np.unique([p[0] for p in g["points"]])
    s8_g = np.unique([p[1] for p in g["points"]])
    if len(om_g) > 3 and len(s8_g) > 3 and args.data:
        idx = {p: i for i, p in enumerate(g["points"])}
        Ts = {p: g["pairs"][i].mean(0) for p, i in idx.items()}
        Cs = {p: covariance(g, i) for p, i in idx.items()}
        omf, s8f = (
            np.linspace(om_g.min(), om_g.max(), 240),
            np.linspace(s8_g.min(), s8_g.max(), 240),
        )
        Om, S8 = np.meshgrid(omf, s8f, indexing="ij")
        lp = ((Om - OM_FID) / args.prior_scale) ** 2 + ((S8 - S8_FID) / args.prior_scale) ** 2
        out["posterior"] = {}
        for run in args.data:
            y = data_vector(run, g["u_half"])
            out["posterior"][run] = {}
            for with_t in (True, False):
                L = np.array(
                    [
                        [
                            np.sum(
                                ((y - Ts[(o, s)]) if with_t else y)[sel] ** 2 / Cs[(o, s)][sel]
                                + np.log(Cs[(o, s)][sel])
                            )
                            for s in s8_g
                        ]
                        for o in om_g
                    ]
                )
                M = RectBivariateSpline(om_g, s8_g, L)(omf, s8f) + lp
                w = np.exp(-0.5 * (M - M.min()))
                w /= w.sum()
                mo, ms = float((w * Om).sum()), float((w * S8).sum())
                key = "with template" if with_t else "without template"
                out["posterior"][run][key] = {
                    "Omega_m": [mo, float(np.sqrt((w * (Om - mo) ** 2).sum()))],
                    "sigma8": [ms, float(np.sqrt((w * (S8 - ms) ** 2).sum()))],
                }
                print(
                    f"{Path(run).name} {key:16s}: Omega_m {mo:.3f} ± {out['posterior'][run][key]['Omega_m'][1]:.3f}, "
                    f"sigma8 {ms:.3f} ± {out['posterior'][run][key]['sigma8'][1]:.3f}"
                )
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
        print(f"Saved {args.json}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="E_z[κ] by antithetic pairs on a list of (Omega_m, sigma8)")
    m.add_argument("--config", required=True)
    m.add_argument("--om", type=float, nargs="+", default=[OM_FID])
    m.add_argument("--s8", type=float, nargs="+", default=[S8_FID])
    m.add_argument("--pairs", type=int, default=8)
    m.add_argument("--seed", type=int, default=5000)
    m.add_argument("--box", type=float, default=None, help="cubic box side (Mpc/h), with --mesh")
    m.add_argument("--mesh", type=int, default=None, help="final mesh cells per side, with --box")
    m.add_argument(
        "--particles", type=int, default=None, help="particles per side (evol = ptcl grid)"
    )
    m.add_argument("--out", required=True)
    a = sub.add_parser(
        "analyse", help="S/N, projections and the Gaussian posterior with and without T"
    )
    a.add_argument("--grid", required=True, help="glob of measure outputs")
    a.add_argument(
        "--data", nargs="*", help="run directories whose truth.npz holds kappa_obs_packed"
    )
    a.add_argument("--prior_scale", type=float, default=0.15, help="Gaussian prior width on both")
    a.add_argument("--json", default=None)
    args = ap.parse_args()
    {"measure": measure, "analyse": analyse}[args.cmd](args)


if __name__ == "__main__":
    main()
