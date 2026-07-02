#!/usr/bin/env python
"""
Plot mean amplitude of the high-z kappa correction at fiducial cosmology
as a function of box depth (chi_min = lower boundary of the high-z integral).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np

from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import compute_theoretical_cl_kappa

jax.config.update("jax_enable_x64", True)

# ── Fiducial cosmology ──────────────────────────────────────────────────────
Om_fid, s8_fid = 0.3, 0.8
cosmo = get_cosmology(Omega_m=Om_fid, sigma8=s8_fid)

# ── Source plane ─────────────────────────────────────────────────────────────
z_cmb = 1100.0
chi_cmb = float(jc.background.radial_comoving_distance(cosmo, 1.0 / (1.0 + z_cmb))[0])
print(f"chi_CMB = {chi_cmb:.0f} Mpc/h")

# ── ell grid (representative of a CMB-lensing reconstruction) ───────────────
ell_1d = jnp.array(np.logspace(1, 3.5, 80))

# ── Depth grid: chi_min from 100 to ~14000 Mpc/h ────────────────────────────
chi_depths = np.linspace(100.0, 0.97 * chi_cmb, 60)

# ── Corresponding redshifts for secondary x-axis ────────────────────────────
a_depths = jax.vmap(lambda chi: jc.background.a_of_chi(cosmo, jnp.array([chi]))[0])(
    jnp.array(chi_depths)
)
z_depths = 1.0 / np.array(a_depths) - 1.0

# ── Compute C_ell^{kappa kappa} for each depth ───────────────────────────────
print("Computing C_ell^{kappa kappa} for each box depth ...")

# Total C_l from 0 to chi_CMB (normalisation reference)
cl_total = compute_theoretical_cl_kappa(cosmo, ell_1d, 1.0, chi_cmb, z_cmb, n_steps=200)
mean_cl_total = float(jnp.mean(cl_total))

cl_means = []
cl_peak = []  # amplitude at ell=100

for chi_min in chi_depths:
    cl = compute_theoretical_cl_kappa(
        cosmo, ell_1d, chi_min, chi_cmb, z_cmb, n_steps=200
    )
    cl_means.append(float(jnp.mean(cl)))
    # Interpolate at ell=100
    cl_peak.append(float(jnp.interp(jnp.array(100.0), ell_1d, cl)))

cl_means = np.array(cl_means)
cl_peak = np.array(cl_peak)
frac_mean = cl_means / mean_cl_total
frac_peak = cl_peak / float(jnp.interp(jnp.array(100.0), ell_1d, cl_total))

print(f"Done. Total mean C_ell = {mean_cl_total:.4e}")

# ── Plot ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.0,
})

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# ── Top panel: absolute amplitude ────────────────────────────────────────────
ax0 = axes[0]
ax0.semilogy(chi_depths, cl_means, color="#1f77b4", label=r"Mean over $\ell \in [10,\,3200]$")
ax0.semilogy(chi_depths, cl_peak, color="#ff7f0e", linestyle="--", label=r"$\ell = 100$")
ax0.axhline(mean_cl_total, color="#1f77b4", linestyle=":", alpha=0.5, linewidth=1,
            label="Full LOS (mean)")
ax0.set_ylabel(r"$\langle C_\ell^{\kappa\kappa}\rangle$ (high-z correction)")
ax0.legend(frameon=True, framealpha=0.92)
ax0.grid(alpha=0.2, linestyle="--", linewidth=0.5)
ax0.set_title(
    rf"High-z $\kappa$ correction at fiducial cosmology "
    rf"($\Omega_m={Om_fid}$, $\sigma_8={s8_fid}$)",
    fontsize=11, fontweight="bold",
)

# ── Bottom panel: fraction of total ──────────────────────────────────────────
ax1 = axes[1]
ax1.semilogy(chi_depths, frac_mean * 100, color="#1f77b4",
             label=r"Mean over $\ell$")
ax1.semilogy(chi_depths, frac_peak * 100, color="#ff7f0e", linestyle="--",
             label=r"$\ell = 100$")
ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.6, linewidth=1, label="1 %")
ax1.set_ylabel(r"Fraction of total $C_\ell^{\kappa\kappa}$ (\%)")
ax1.set_xlabel(r"Box depth $\chi_{\rm min}$ (Mpc/$h$)")
ax1.legend(frameon=True, framealpha=0.92)
ax1.grid(alpha=0.2, linestyle="--", linewidth=0.5)

# ── Secondary x-axis: redshift ────────────────────────────────────────────────
ax0_top = ax0.twiny()
ax0_top.set_xlim(ax0.get_xlim())
z_ticks = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
chi_ticks = np.array([
    float(jc.background.radial_comoving_distance(cosmo, 1.0 / (1.0 + z))[0])
    for z in z_ticks
])
mask = (chi_ticks > chi_depths[0]) & (chi_ticks < chi_depths[-1])
ax0_top.set_xticks(chi_ticks[mask])
ax0_top.set_xticklabels([f"$z={z:.0f}$" if z >= 1 else f"$z={z:.1f}$"
                          for z in z_ticks[mask]], fontsize=9)
ax0_top.set_xlabel("Redshift", fontsize=11)

plt.tight_layout()

output_dir = Path(__file__).parent.parent / "figures"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "highz_correction_vs_depth.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.close()
