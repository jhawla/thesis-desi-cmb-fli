
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt

from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import compute_theoretical_cl_kappa


def plot_fraction_vs_depth():
    print("Computing Lensing Signal Fraction...")

    # 1. Define Cosmology
    cosmo = get_cosmology(Omega_m=0.3, sigma8=0.8)

    # 2. Parameters
    z_source = 1100.0
    # Use a range of scales for averaging
    ell_arr = jnp.linspace(100, 2000, 20)

    z_max_list = jnp.linspace(0.1, 5.0, 50)

    # Pre-calculate chi for z_max list
    chi_max_list = jc.background.radial_comoving_distance(cosmo, 1.0/(1.0+z_max_list))

    # 3. Compute Total Cl (0 -> z_source) for reference
    print("Computing Total Cl...")
    cl_tot = compute_theoretical_cl_kappa(cosmo, ell_arr, chi_min=1.0, chi_max=14000.0, z_source=z_source, n_steps=512)
    cl_tot = jnp.where(cl_tot <= 0, 1e-30, cl_tot)

    # 4. Loop over depths
    ratios = []

    print("Computing Fractions vs Box Depth...")
    for i, chi_max in enumerate(chi_max_list):
        cl_box = compute_theoretical_cl_kappa(cosmo, ell_arr, chi_min=1.0, chi_max=chi_max, z_source=z_source, n_steps=256)

        # Ratio of amplitudes
        ratio = jnp.sqrt(cl_box / cl_tot)
        ratios.append(ratio)

        if i % 10 == 0:
            print(f"  z={z_max_list[i]:.2f}, Mean Fraction: {jnp.mean(ratio):.2%}")

    ratios = jnp.array(ratios) # Shape [N_z, N_ell]

    # Average over ell
    mean_ratio = jnp.mean(ratios, axis=1)
    min_ratio = jnp.min(ratios, axis=1)
    max_ratio = jnp.max(ratios, axis=1)

    # 5. Plotting
    plt.figure(figsize=(10, 6))

    plt.plot(z_max_list, mean_ratio, label=r'Mean Fraction ($\ell \in [100, 2000]$)', color='blue', lw=3)
    plt.fill_between(z_max_list, min_ratio, max_ratio, color='blue', alpha=0.2, label='Variation with $\ell$')

    plt.axhline(1.0, color='k', linestyle='--', alpha=0.5)

    # Vertical Lines
    plt.axvline(0.83, color='red', linestyle='--', label='Box Depth ($z \sim 0.83, \chi=2000$)')

    plt.xlabel(r'Box Depth $z_{max}$')
    plt.ylabel(r'Amplitude Ratio $\sqrt{C_\ell^{box} / C_\ell^{total}}$')
    plt.title('CMB Lensing Signal Captured by Box (Depth 2000 Mpc/h)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.ylim(0, 1.1)

    output_path = 'figures/lensing_fraction/lensing_fraction_vs_z.png'
    plt.savefig(output_path, dpi=150)
    print(f"✓ Plot saved to {output_path}")

def plot_spectra_comparison():
    print("\nComputing Lensing Spectra Comparison (Box vs Total)...")

    # 1. Define Cosmology
    cosmo = get_cosmology(Omega_m=0.3, sigma8=0.8)
    z_source = 1100.0

    # High resolution ell for smooth plot
    ell_plot = jnp.geomspace(10, 3000, 100)

    # 2. Define Depths
    chi_box = 2000.0
    chi_total = 14000.0

    # Calculate Z for labeling
    def get_z(chi_target):
        z_grid = jnp.linspace(0, 1100, 10000)
        chi_grid = jc.background.radial_comoving_distance(cosmo, 1.0/(1.0+z_grid))
        return jnp.interp(chi_target, chi_grid, z_grid)

    z_box = get_z(chi_box)
    print(f"  Box Depth: chi={chi_box} => z={z_box:.2f}")

    # 3. Compute Spectra
    print("  Computing C_l (Box)...")
    cl_box = compute_theoretical_cl_kappa(cosmo, ell_plot, chi_min=1.0, chi_max=chi_box, z_source=z_source, n_steps=512)

    print("  Computing C_l (Total)...")
    cl_total = compute_theoretical_cl_kappa(cosmo, ell_plot, chi_min=1.0, chi_max=chi_total, z_source=z_source, n_steps=1024)

    # Compute Ratio
    ratio_ampl = jnp.sqrt(cl_box / cl_total)

    # 4. Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Upper: Spectra
    ax1.loglog(ell_plot, cl_total, 'k-', lw=2, label='Total Signal ($z < 1100$)')
    ax1.loglog(ell_plot, cl_box, 'r--', lw=2, label=f'Box Signal ($z < {z_box:.2f}$)')

    ax1.set_ylabel(r'$C_\ell^{\kappa\kappa}$')
    ax1.set_title('Comparison of Lensing Power Spectra (Theoretical)')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.2)

    # Lower: Ratio
    ax2.semilogx(ell_plot, ratio_ampl, 'b-', lw=2, label=r'Amplitude Ratio $\sqrt{C_\ell^{box}/C_\ell^{tot}}$')
    ax2.axhline(1.0, color='k', ls=':', alpha=0.5)
    ax2.set_xlabel(r'Multipole $\ell$')
    ax2.set_ylabel('Amplitude Fraction')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, which='both', alpha=0.2)
    ax2.legend()

    plt.tight_layout()
    output_path = 'figures/lensing_fraction/lensing_spectra_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"✓ Spectra comparison saved to {output_path}")

if __name__ == "__main__":
    plot_fraction_vs_depth()
    plot_spectra_comparison()
