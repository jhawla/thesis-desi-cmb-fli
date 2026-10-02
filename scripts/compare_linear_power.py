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


OM_FID, S8_FID = 0.315192, 0.811355
OM_SCAN = [0.25, 0.28, 0.35, 0.40]
# diverging around the fiducial: blue below, red above, the far value darker
OM_COLORS = {0.25: "#1c5cab", 0.28: "#6da7ec", 0.35: "#ee8a89", 0.40: "#b8302f"}


def camb_shape(omega_m, k):
    """CAMB linear P_cb(k) at z = 0, normalised to sigma8 = 1, varying Omega_m as the model does:
    Omega_c = Omega_m - Omega_b at fixed h, Omega_b, n_s (AbacusSummit c000, one 0.06 eV neutrino
    inside Omega_m)."""
    import camb

    h, omb, mnu = 0.6736, 0.04930169, 0.06
    omnuh2 = mnu / 93.14
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * h, ombh2=omb * h**2, omch2=omega_m * h**2 - omb * h**2 - omnuh2,
                       mnu=mnu, num_massive_neutrinos=1, omk=0.0)
    pars.InitPower.set_params(ns=0.9649, As=2.1e-9)
    pars.set_matter_power(redshifts=[0.0], kmax=40.0)
    res = camb.get_results(pars)
    pk = res.get_matter_power_interpolator(nonlinear=False, var1="delta_nonu", var2="delta_nonu",
                                           hubble_units=True, k_hunit=True, extrap_kmax=100.0)
    p = pk.P(0.0, k)
    return p / sigma8_squared(k, p)


def omega_m_scan():
    """How well each linear-power option follows a Boltzmann code when Omega_m moves away from the
    fiducial at fixed sigma8: EH times the fiducial shape ratio (lin_pk_table), plain EH, and the
    fiducial shape held fixed (montecosmo's tabulated mode). CAMB is the reference here, so the
    ratio is CAMB/EH at the fiducial, which isolates the approximation from code differences."""
    from desi_cmb_fli.bricks import lin_power_interp as lpi

    k = np.logspace(-4, 1, 3000)  # the range of lin_power_interp
    ref_fid = camb_shape(OM_FID, k)
    kc, pc = np.loadtxt(ROOT / "data/abacus_cosm000_CLASS_power.txt", unpack=True)
    tab = np.exp(np.interp(np.log(k), np.log(kc), np.log(pc)))
    tab = tab / sigma8_squared(k, tab)
    sel = (k > 1e-3) & (k < 0.3)
    print(f"CAMB / Abacus CLASS table at the fiducial, sigma8 = 1: "
          f"{(ref_fid / tab)[sel].min():.4f}-{(ref_fid / tab)[sel].max():.4f} (k = 0.001-0.3)")

    def eh_shape(om):
        p = np.asarray(lpi(get_cosmology(Omega_m=om, sigma8=S8_FID), n_interp=4096)(jnp.asarray(k)))
        return p / sigma8_squared(k, p)

    ratio = ref_fid / eh_shape(OM_FID)
    out = {"k": k, "omega_m": np.array(OM_SCAN)}
    for om in [OM_FID, *OM_SCAN]:
        ref, eh = camb_shape(om, k), eh_shape(om)
        ours = eh * ratio
        ours = ours / sigma8_squared(k, ours)
        out[f"ours_{om}"], out[f"eh_{om}"], out[f"fixed_{om}"] = ours / ref, eh / ref, ref_fid / ref

    band, corner = K_NYQ, np.sqrt(3) * K_NYQ
    print("max |option / CAMB - 1| in % over k <= k_Nyq (and up to the corners sqrt(3) k_Nyq):")
    for om in [OM_FID, *OM_SCAN]:
        line = []
        for name in ("ours", "eh", "fixed"):
            r = out[f"{name}_{om}"]
            line.append(f"{name} {100 * np.abs(r[(k >= 1e-3) & (k <= band)] - 1).max():.2f}"
                        f" ({100 * np.abs(r[(k >= 1e-3) & (k <= corner)] - 1).max():.2f})")
        print(f"  Omega_m = {om:<8g} " + "   ".join(line))
    np.savez(ROOT / "figures/spectra_diagnostic/linear_power_omega_m_accuracy.npz", **out)

    titles = {"ours": "EH($\\Omega_m$) × [Boltzmann / EH] at the fiducial  (lin_pk_table)",
              "eh": "Eisenstein–Hu alone  (current runs)",
              "fixed": "Fiducial Boltzmann shape held fixed  (montecosmo tabulated mode)"}
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.5), sharex=True)
    kk = (k >= 1e-3) & (k <= 0.5)
    for ax, name in zip(axes, ("ours", "eh", "fixed"), strict=True):
        ax.axvspan(1e-3, band, color="#e1e0d9", lw=0, alpha=0.6)
        ax.axvspan(band, corner, color="#e1e0d9", lw=0, alpha=0.25)
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.semilogx(k[kk], 100 * (out[f"{name}_{OM_FID}"][kk] - 1), color="#898781", lw=2,
                    label=f"$\\Omega_m$ = {OM_FID} (fiducial)")
        for om in OM_SCAN:
            ax.semilogx(k[kk], 100 * (out[f"{name}_{om}"][kk] - 1), color=OM_COLORS[om], lw=2,
                        label=f"$\\Omega_m$ = {om}")
        ax.set_title(titles[name], fontsize=10, loc="left", color="#0b0b0b")
        ax.set_ylabel("option / CAMB − 1  [%]", color="#52514e")
        ax.grid(alpha=0.3, which="major", color="#e1e0d9")
        ax.tick_params(colors="#52514e")
        for s in ax.spines.values():
            s.set_color("#c3c2b7")
    axes[0].set_ylim(-6, 6)
    axes[1].set_ylim(-6, 6)
    axes[2].set_ylim(-110, 110)  # its own scale: tens of per cent
    axes[0].text(1.1e-3, 4.6, "likelihood band", fontsize=8, color="#52514e")
    axes[0].text(1.04 * band, 4.6, "corners", fontsize=8, color="#52514e")
    axes[0].legend(fontsize=8, ncol=3, loc="lower left", frameon=False)
    axes[-1].set_xlabel(r"$k$  [$h\,\mathrm{Mpc}^{-1}$]", color="#52514e")
    fig.suptitle("Linear power at fixed $\\sigma_8$ when $\\Omega_m$ leaves the fiducial", fontsize=11)
    fig.tight_layout()
    path = ROOT / "figures/spectra_diagnostic/linear_power_omega_m_accuracy.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    import sys

    if "--omega_m_scan" in sys.argv:  # needs camb
        omega_m_scan()
    else:
        main()
