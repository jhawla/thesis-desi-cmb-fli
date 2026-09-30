"""The model's linear P(k) against the CLASS spectrum the AbacusSummit ICs were drawn from.

data/abacus_cosm000_CLASS_power.txt is AbacusSummit's Cosmologies/abacus_cosm000/CLASS_power (the
`ZD_Pk_filename` of the HUGE c000 IC header, z = 1). Both spectra are normalised to sigma8 = 1
(top hat, R = 8 Mpc/h) so the ratio is a pure shape difference, the part a sigma8 rescaling cannot
absorb; the model's is bricks.lin_power_interp (jax_cosmo, Eisenstein-Hu) at the Abacus cosmology.

Usage: python scripts/compare_linear_power.py
Figure: figures/spectra_diagnostic/linear_power_eh_vs_class.png (spectra and their ratio, the
likelihood's band shaded: k below the final-mesh Nyquist of the Abacus runs, then the corner modes).
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from desi_cmb_fli.bricks import get_cosmology, lin_power_interp

ROOT = Path(__file__).resolve().parents[1]
K_NYQ = np.pi / 93.75  # final mesh of the Abacus runs (cell 93.75 Mpc/h)
BLUE = "#2a78d6"
K_PRINT = [1e-3, 3e-3, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2]


def sigma8_squared(k, p):
    kf = np.logspace(-5, 1.5, 20000)
    pf = np.exp(np.interp(np.log(kf), np.log(k), np.log(p)))
    x = kf * 8.0
    w = 3 * (np.sin(x) - x * np.cos(x)) / x**3
    return np.trapezoid(kf**3 * pf * w**2 / (2 * np.pi**2), np.log(kf))


def main():
    kc, pc = np.loadtxt(ROOT / "data/abacus_cosm000_CLASS_power.txt", unpack=True)
    cosmo = get_cosmology(Omega_m=0.315192, sigma8=0.811355)
    ke = np.logspace(-4, 1, 2000)
    pe = np.asarray(lin_power_interp(cosmo, a=1.0, n_interp=2048)(jnp.asarray(ke)))
    ks = np.array(K_PRINT)
    shape_model = np.interp(ks, ke, pe) / sigma8_squared(ke, pe)
    shape_class = np.exp(np.interp(np.log(ks), np.log(kc), np.log(pc))) / sigma8_squared(kc, pc)
    print("P_model / P_CLASS, both normalised to sigma8 = 1:")
    for k, r in zip(ks, shape_model / shape_class, strict=True):
        print(f"  k = {k:<6g} h/Mpc  {r:.3f}")

    kk = np.logspace(-3.5, 0, 600)
    pm = np.interp(kk, ke, pe) / sigma8_squared(ke, pe)
    pcl = np.exp(np.interp(np.log(kk), np.log(kc), np.log(pc))) / sigma8_squared(kc, pc)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for ax in (a1, a2):
        ax.axvspan(kk[0], K_NYQ, color="0.85", lw=0)
        ax.axvspan(K_NYQ, np.sqrt(3) * K_NYQ, color="0.93", lw=0)
        ax.grid(alpha=0.25, which="both")
    a1.loglog(kk, pcl, color="k", lw=2, label="CLASS, AbacusSummit cosm000 (the ICs' spectrum)")
    a1.loglog(kk, pm, color=BLUE, ls="--", lw=2, label="model: jax_cosmo Eisenstein–Hu")
    a1.set_ylabel(r"$P_\mathrm{lin}(k)\ /\ \sigma_8^2$  [$(h^{-1}\mathrm{Mpc})^3$]")
    a1.legend(fontsize=9, loc="lower left")
    a1.set_title("Linear power at the same $\\sigma_8$ (shape only)")
    a2.semilogx(kk, pm / pcl, color=BLUE, lw=2)
    a2.axhline(1, color="k", lw=1)
    a2.set_ylim(0.9, 1.1)
    a2.set_xlabel(r"$k$  [$h\,\mathrm{Mpc}^{-1}$]")
    a2.set_ylabel("model / CLASS")
    a2.text(1.2 * kk[0], 1.07, "likelihood band ($k < k_\\mathrm{Nyq}$)", fontsize=8, color="0.3")
    a2.text(1.03 * K_NYQ, 1.07, "corners", fontsize=8, color="0.3")
    fig.tight_layout()
    path = ROOT / "figures/spectra_diagnostic/linear_power_eh_vs_class.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
