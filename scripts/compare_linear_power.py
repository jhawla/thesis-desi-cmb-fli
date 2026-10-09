"""The model's linear P(k) (ACE emulator, bricks.lin_power_interp) against CLASS.

Default: at the Abacus cosmology, against data/abacus_cosm000_CLASS_power.txt, the CLASS P_cb at
z = 1 the AbacusSummit ICs were drawn from (`ZD_Pk_filename` of the HUGE c000 IC header), with
jax_cosmo's Eisenstein-Hu for comparison. Every spectrum is normalised to sigma8 = 1 (top hat,
R = 8 Mpc/h), so the ratios are pure shape differences, the part a sigma8 rescaling cannot absorb.

--omega_m_scan: the same at fixed sigma8 as Omega_m moves, against the CLASS spectra of
scripts/linear_power_class_reference.py (run it first, in the cosmodesi env), for the emulator, the
former EH x [CLASS / EH at the fiducial] and EH alone; max error over the likelihood band against
Omega_m, the emulator's validated range shaded.

Usage: python scripts/compare_linear_power.py [--omega_m_scan]
Figures: figures/spectra_diagnostic/linear_power_vs_class.png,
         figures/spectra_diagnostic/linear_power_omega_m_accuracy.{png,npz}
"""

from pathlib import Path

import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np

from desi_cmb_fli.bricks import ACE_OMEGA_M_RANGE, get_cosmology, lin_power_interp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/spectra_diagnostic"
K_NYQ = np.pi / 93.75  # final mesh of the Abacus runs (cell 93.75 Mpc/h)
OM_FID, S8_FID = 0.315192, 0.811355
BLUE, ORANGE, GREY = "#2a78d6", "#e8862a", "#898781"
K_PRINT = [1e-3, 3e-3, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2]


def sigma8_squared(k, p):
    kf = np.logspace(-5, 1.5, 20000)
    pf = np.exp(np.interp(np.log(kf), np.log(k), np.log(p)))
    x = kf * 8.0
    w = 3 * (np.sin(x) - x * np.cos(x)) / x**3
    return np.trapezoid(kf**3 * pf * w**2 / (2 * np.pi**2), np.log(kf))


def shape(k, p):
    return p / sigma8_squared(k, p)


def model_shape(om, k):
    p = np.asarray(lin_power_interp(get_cosmology(Omega_m=om, sigma8=S8_FID), n_interp=4096)(jnp.asarray(k)))
    return shape(k, p)


def eh_shape(om, k):
    p = np.asarray(jc.power.linear_matter_power(get_cosmology(Omega_m=om, sigma8=S8_FID), jnp.asarray(k), a=1.0))
    return shape(k, p)


def main():
    kc, pc = np.loadtxt(ROOT / "data/abacus_cosm000_CLASS_power.txt", unpack=True)
    kk = np.logspace(-4, 1, 3000)
    pcl = shape(kk, np.exp(np.interp(np.log(kk), np.log(kc), np.log(pc))))
    pm, pe = model_shape(OM_FID, kk), eh_shape(OM_FID, kk)
    ks = np.array(K_PRINT)
    print("P / P_CLASS(Abacus ICs), all normalised to sigma8 = 1:   model (ACE)   Eisenstein-Hu")
    for k, rm, re in zip(ks, np.interp(ks, kk, pm / pcl), np.interp(ks, kk, pe / pcl), strict=True):
        print(f"  k = {k:<6g} h/Mpc   {rm:.4f}   {re:.4f}")

    sel = (kk >= 10**-3.5) & (kk <= 1.0)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for ax in (a1, a2):
        ax.axvspan(kk[sel][0], K_NYQ, color="0.85", lw=0)
        ax.axvspan(K_NYQ, np.sqrt(3) * K_NYQ, color="0.93", lw=0)
        ax.grid(alpha=0.25, which="both")
    a1.loglog(kk[sel], pcl[sel], color="k", lw=2, label="CLASS, AbacusSummit c000 (the ICs' spectrum)")
    a1.loglog(kk[sel], pm[sel], color=BLUE, ls="--", lw=2, label="model: ACE emulator")
    a1.loglog(kk[sel], pe[sel], color=GREY, ls=":", lw=2, label="jax_cosmo Eisenstein–Hu")
    a1.set_ylabel(r"$P_\mathrm{lin}(k)\ /\ \sigma_8^2$  [$(h^{-1}\mathrm{Mpc})^3$]")
    a1.legend(fontsize=9, loc="lower left")
    a1.set_title("Linear power at the same $\\sigma_8$ (shape only)")
    a2.semilogx(kk[sel], (pm / pcl)[sel], color=BLUE, lw=2)
    a2.semilogx(kk[sel], (pe / pcl)[sel], color=GREY, lw=2, ls=":")
    a2.axhline(1, color="k", lw=1)
    a2.set_ylim(0.9, 1.1)
    a2.set_xlabel(r"$k$  [$h\,\mathrm{Mpc}^{-1}$]")
    a2.set_ylabel("/ CLASS")
    a2.text(1.2 * kk[sel][0], 1.07, "likelihood band ($k < k_\\mathrm{Nyq}$)", fontsize=8, color="0.3")
    a2.text(1.03 * K_NYQ, 1.07, "corners", fontsize=8, color="0.3")
    fig.tight_layout()
    path = OUT / "linear_power_vs_class.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def omega_m_scan():
    ref = np.load(OUT / "linear_power_class_omega_m.npz")
    k, oms = ref["k"], [float(om) for om in ref["omega_m"]]
    classes = {om: shape(k, ref[f"class_{i}"]) for i, om in enumerate(oms)}
    ratio_fid = classes[OM_FID] / eh_shape(OM_FID, k)  # the former lin_pk_table, CLASS at the fiducial
    band, corner = (k >= 1e-3) & (k <= K_NYQ), (k >= 1e-3) & (k <= np.sqrt(3) * K_NYQ)
    out = {"k": k, "omega_m": np.array(oms)}
    names = {"model": "ACE emulator (model)", "former": "EH × [CLASS / EH] at the fiducial (former)",
             "eh": "Eisenstein–Hu"}
    print("max |option / CLASS - 1| in % over k <= k_Nyq (and up to the corners sqrt(3) k_Nyq):")
    for om in oms:
        eh = eh_shape(om, k)
        opts = {"model": model_shape(om, k), "former": shape(k, eh * ratio_fid), "eh": eh}
        line = []
        for name, p in opts.items():
            r = p / classes[om]
            out[f"{name}_{om}"] = r
            line.append(f"{name} {100 * np.nanmax(np.abs(r[band] - 1)):.3f}"
                        f" ({100 * np.nanmax(np.abs(r[corner] - 1)):.3f})")
        print(f"  Omega_m = {om:<9g} " + "   ".join(line))
    np.savez(OUT / "linear_power_omega_m_accuracy.npz", **out)

    lo, hi = ACE_OMEGA_M_RANGE
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 8))
    inside = [om for om in oms if lo <= om <= hi]
    cmap = plt.get_cmap("coolwarm")
    kk = (k >= 1e-3) & (k <= 0.1)
    a1.axvspan(1e-3, K_NYQ, color="#e1e0d9", lw=0, alpha=0.6)
    a1.axvspan(K_NYQ, np.sqrt(3) * K_NYQ, color="#e1e0d9", lw=0, alpha=0.25)
    a1.axhline(0, color="#c3c2b7", lw=1)
    for om in inside:
        c = cmap((om - lo) / (hi - lo))
        a1.semilogx(k[kk], 100 * (out[f"model_{om}"][kk] - 1), color=c, lw=2, label=f"$\\Omega_m$ = {om:g}")
        a1.semilogx(k[kk], 100 * (out[f"former_{om}"][kk] - 1), color=c, lw=1, ls="--")
    a1.set_xlabel(r"$k$  [$h\,\mathrm{Mpc}^{-1}$]")
    a1.set_ylabel("option / CLASS − 1  [%]")
    a1.set_title("ACE emulator (solid) and former EH × ratio (dashed), validated range", fontsize=10, loc="left")
    a1.legend(fontsize=7, ncol=4, loc="lower left", frameon=False)
    a1.grid(alpha=0.3)
    a2.axvspan(lo, hi, color="#e1e0d9", lw=0, alpha=0.6)
    for name, color in (("model", BLUE), ("former", ORANGE), ("eh", GREY)):
        err = [100 * np.nanmax(np.abs(out[f"{name}_{om}"][band] - 1)) for om in oms]
        a2.semilogy(oms, err, "o-", color=color, lw=2, label=names[name])
    a2.set_ylim(1e-2, 200)  # the former is exact at the fiducial by construction (off the axis)
    a2.set_xlabel(r"$\Omega_m$  (fixed $\sigma_8$, $h$, $\Omega_b$, $n_s$)")
    a2.set_ylabel(r"max error over $k \leq k_\mathrm{Nyq}$  [%]")
    a2.text(lo + 0.005, 50, "emulator prior range", fontsize=8, color="#52514e")
    a2.legend(fontsize=8, frameon=False)
    a2.grid(alpha=0.3, which="both")
    fig.suptitle("Linear power at fixed $\\sigma_8$ against CLASS ($P_{cb}$, z = 1) as $\\Omega_m$ moves",
                 fontsize=11)
    fig.tight_layout()
    path = OUT / "linear_power_omega_m_accuracy.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    import sys

    if "--omega_m_scan" in sys.argv:
        omega_m_scan()
    else:
        main()
