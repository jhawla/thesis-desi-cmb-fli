"""Omega_m-sigma8 triplet (galaxies, kappa alone, joint): means, pulls, Mahalanobis, split R-hat, paired kappa gain.

Numbers of docs/pipeline.md 7.17. Paired gain: chain i of the joint against chain i of the galaxies,
at the joint's length.

    python scripts/triplet_numbers.py OUT.json GALAXY_RUN KAPPA_RUN JOINT_RUN
"""

import json
import sys
from pathlib import Path

import numpy as np
from analyze_run import load_and_process_run

OUT = Path(sys.argv[1])
RUNS = {"galaxies": sys.argv[2], "kappa": sys.argv[3], "joint": sys.argv[4]}
TRUTH = np.array([0.315192, 0.811355])
P = ["Omega_m", "sigma8"]


def split_rhat(x):
    n = x.shape[1] // 2
    y = np.concatenate([x[:, :n], x[:, n : 2 * n]])
    w = y.var(1, ddof=1).mean()
    b = n * y.mean(1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


chains, res = {}, {}
for name, run in RUNS.items():
    s = load_and_process_run(run)["physical_samples"]
    x = {p: np.asarray(s[p]) for p in P}
    chains[name] = x
    flat = np.stack([x[p].ravel() for p in P], 1)
    mu, cov = flat.mean(0), np.cov(flat.T)
    sd = np.sqrt(np.diag(cov))
    d = mu - TRUTH
    res[name] = {
        "run": Path(run).name,
        "n_chains": x[P[0]].shape[0],
        "n_samples": x[P[0]].shape[1],
        "mean": mu.tolist(),
        "std": sd.tolist(),
        "pull": (d / sd).tolist(),
        "corr": float(cov[0, 1] / sd[0] / sd[1]),
        "mahalanobis": float(np.sqrt(d @ np.linalg.solve(cov, d))),
        "split_rhat": [split_rhat(x[p]) for p in P],
        "std_second_half": [float(x[p][:, x[p].shape[1] // 2 :].std()) for p in P],
    }
    print(name, json.dumps(res[name]))

# paired gain at matched length: chain i of the joint against chain i of the galaxies
n = min(chains["galaxies"][P[0]].shape[1], chains["joint"][P[0]].shape[1])
gain = {}
for p in P:
    r = np.array(
        [
            chains["joint"][p][c, :n].std() / chains["galaxies"][p][c, :n].std()
            for c in range(chains["joint"][p].shape[0])
        ]
    )
    gain[p] = {
        "ratio_mean": float(r.mean()),
        "ratio_err": float(r.std(ddof=1) / np.sqrt(len(r))),
        "gain": float(1 - r.mean()),
        "per_chain": r.tolist(),
        "n_samples": n,
    }
    print(p, gain[p])
shift = (np.array(res["joint"]["mean"]) - np.array(res["galaxies"]["mean"])) / np.array(
    res["galaxies"]["std"]
)
print("joint - galaxies in sigma_gal:", shift)
json.dump(
    {
        "truth": TRUTH.tolist(),
        "params": P,
        "runs": res,
        "paired_gain": gain,
        "joint_minus_galaxies_in_sigma_gal": shift.tolist(),
    },
    open(OUT, "w"),
    indent=1,
)
print("saved", OUT)
