#!/usr/bin/env python
"""
Comprehensive benchmark: High-z correction modes (fixed, taylor, exact_linear)
Tests precision and performance across different grid sizes.
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np
from jax import jacfwd

from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import compute_cl_high_z, compute_theoretical_cl_kappa

jax.config.update("jax_enable_x64", True)

def benchmark_precision():
    """
    Test precision: compare fixed and taylor against exact_linear mode
    for various cosmologies.
    """
    print("=" * 70)
    print("BENCHMARK 1: PRECISION")
    print("=" * 70)

    # Fiducial cosmology
    Om_fid, s8_fid = 0.3, 0.8
    cosmo_fid = get_cosmology(Omega_m=Om_fid, sigma8=s8_fid)

    # Geometry
    box_shape = np.array([320.0, 320.0, 320.0])
    chi_max_box = box_shape[2]
    cmb_z_source = 1100.0

    # Create realistic 2D ell grid (like FFT)
    npix = 64
    pixel_scale_deg = 2.0  # degrees
    freq = np.fft.fftfreq(npix, d=pixel_scale_deg) * 360.0
    lx, ly = np.meshgrid(freq, freq, indexing='ij')
    ell_eval = jnp.array(np.sqrt(lx**2 + ly**2))

    # Also compute on 1D array for plotting
    ell_1d = np.logspace(1, 3.5, 100)

    print(f"\nGrid size: {npix}×{npix}, ell range: [{ell_1d.min():.0f}, {ell_1d.max():.0f}]")

    # Precompute fiducial and gradients
    print("\nPrecomputing fiducial C_l and Taylor gradients...")
    chi_source_fid = float(jc.background.radial_comoving_distance(
        cosmo_fid, 1.0 / (1.0 + cmb_z_source))[0])

    cl_cached = compute_theoretical_cl_kappa(
        cosmo_fid, ell_eval, chi_max_box, chi_source_fid, cmb_z_source, n_steps=150
    )

    # Taylor gradients
    def get_cl_wrapper(theta):
        Om, s8 = theta
        c = get_cosmology(Omega_m=Om, sigma8=s8)
        chi_s = jc.background.radial_comoving_distance(c, 1.0 / (1.0 + cmb_z_source))[0]
        return compute_theoretical_cl_kappa(c, ell_eval, chi_max_box, chi_s, cmb_z_source, n_steps=150)

    theta_fid = jnp.array([Om_fid, s8_fid])
    jac_fn = jacfwd(get_cl_wrapper)
    print("  Computing Jacobian (this takes a while)...")
    grads = jac_fn(theta_fid)

    gradients = {
        'dCl_dOm': grads[..., 0],
        'dCl_ds8': grads[..., 1]
    }
    loc_fid = {'Omega_m': Om_fid, 'sigma8': s8_fid}

    # Test cosmologies
    test_params = [
        (r"$\Omega_m = 0.26$", 0.26, 0.80, '#1f77b4'),
        (r"$\Omega_m = 0.34$", 0.34, 0.80, '#ff7f0e'),
        (r"$\sigma_8 = 0.72$", 0.30, 0.72, '#2ca02c'),
        (r"$\sigma_8 = 0.88$", 0.30, 0.88, '#d62728'),
    ]

    # Setup plot
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'lines.linewidth': 2,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Line styles for each mode
    styles = {
        'fixed': ('--', 0.5, 1.5),
        'taylor': ('-', 0.9, 2.2),
        'exact_linear': (':', 0.8, 2.0),
    }

    for label, Om, s8, color in test_params:
        print(f"\n  Testing {label}...")
        cosmo_test = get_cosmology(Omega_m=Om, sigma8=s8)

        # Exact (reference)
        print("    Computing exact (reference)...")
        cl_exact = compute_cl_high_z(
            cosmo_test, ell_eval, chi_max_box, None, cmb_z_source,
            mode='exact', n_steps=150
        )

        # Interpolate to 1D for plotting
        ell_flat = ell_eval.flatten()
        cl_exact_flat = np.array(cl_exact).flatten()
        sort_idx = np.argsort(ell_flat)
        cl_exact_1d = np.interp(ell_1d, ell_flat[sort_idx], cl_exact_flat[sort_idx])

        # Test each mode
        for mode_name, (linestyle, alpha, lw) in styles.items():
            print(f"    Computing {mode_name}...")

            if mode_name == 'fixed':
                cl_mode = cl_cached
            elif mode_name == 'taylor':
                cl_mode = compute_cl_high_z(
                    cosmo_test, ell_eval, chi_max_box, None, cmb_z_source,
                    mode='taylor', cl_cached=cl_cached, gradients=gradients, loc_fid=loc_fid
                )
            elif mode_name == 'exact_linear':
                cl_mode = compute_cl_high_z(
                    cosmo_test, ell_eval, chi_max_box, None, cmb_z_source,
                    mode='exact_linear', n_steps=150
                )

            # Interpolate to 1D
            cl_mode_flat = np.array(cl_mode).flatten()
            cl_mode_1d = np.interp(ell_1d, ell_flat[sort_idx], cl_mode_flat[sort_idx])

            # Relative error in percent
            rel_error = (cl_mode_1d - cl_exact_1d) / cl_exact_1d * 100

            # Plot
            mode_label = {'fixed': 'Fixed', 'taylor': 'Taylor', 'exact_linear': 'Exact (Linear)'}[mode_name]
            ax.plot(ell_1d, rel_error, linestyle=linestyle, color=color, alpha=alpha,
                   linewidth=lw, label=f"{label} ({mode_label})")

    # Formatting
    ax.axhline(0, color='black', linestyle=':', alpha=0.3, linewidth=1, zorder=0)
    ax.set_xlabel(r'Multipole $\ell$', fontsize=12)
    ax.set_ylabel(r'Relative Error (\%)', fontsize=12)
    ax.set_xscale('log')
    ax.set_xlim(ell_1d.min(), ell_1d.max())
    ax.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', frameon=True, shadow=False, ncol=2,
              columnspacing=0.8, handlelength=2.2, framealpha=0.95)

    # Add text box with info
    textstr = f'Grid: {npix}×{npix}\nReference: exact mode'
    props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.15}
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    # Save in figures directory
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'highz_modes_precision.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Precision plot saved to: {output_path}")
    plt.close()


def benchmark_performance():
    """
    Test performance: measure execution time for each mode across different grid sizes.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 2: PERFORMANCE")
    print("=" * 70)

    # Fiducial cosmology
    Om_fid, s8_fid = 0.3, 0.8
    cosmo_fid = get_cosmology(Omega_m=Om_fid, sigma8=s8_fid)

    # Test cosmology (slightly different)
    cosmo_test = get_cosmology(Omega_m=0.31, sigma8=0.82)

    # Geometry
    chi_max_box = 320.0
    cmb_z_source = 1100.0
    pixel_scale_deg = 2.0

    # Grid sizes to test
    grid_sizes = [16, 32, 64, 128, 256]

    print(f"\nTesting grid sizes: {grid_sizes}")
    print("n_steps: 50 (reduced for speed)")

    # Results storage
    results = {
        'fixed': {'times': [], 'n_calls': []},
        'taylor': {'times': [], 'n_calls': []},
        'exact_linear': {'times': [], 'n_calls': []},
        'exact': {'times': [], 'n_calls': []},
    }

    for npix in grid_sizes:
        print(f"\n  Grid {npix}×{npix}:")

        # Create grid
        freq = np.fft.fftfreq(npix, d=pixel_scale_deg) * 360.0
        lx, ly = np.meshgrid(freq, freq, indexing='ij')
        ell_eval = jnp.array(np.sqrt(lx**2 + ly**2))

        n_total = npix * npix
        results['fixed']['n_calls'].append(1)  # Cached
        results['taylor']['n_calls'].append(1)  # Cached + gradient
        results['exact_linear']['n_calls'].append(n_total)
        results['exact']['n_calls'].append(n_total)

        print(f"    Total ell values: {n_total}")

        # Precompute cache for fixed/taylor
        cl_cached = compute_theoretical_cl_kappa(
            cosmo_fid, ell_eval, chi_max_box,
            float(jc.background.radial_comoving_distance(cosmo_fid, 1.0/(1.0+cmb_z_source))[0]),
            cmb_z_source, n_steps=50
        )

        # Mock gradients (for timing, not computing jacobian each time)
        gradients = {
            'dCl_dOm': jnp.ones_like(ell_eval) * 0.1,
            'dCl_ds8': jnp.ones_like(ell_eval) * 0.2
        }
        loc_fid = {'Omega_m': Om_fid, 'sigma8': s8_fid}

        # Benchmark each mode
        modes_to_test = [
            ('fixed', {'mode': 'fixed', 'cl_cached': cl_cached}),
            ('taylor', {'mode': 'taylor', 'cl_cached': cl_cached, 'gradients': gradients, 'loc_fid': loc_fid}),
            ('exact_linear', {'mode': 'exact_linear', 'n_steps': 50}),
            ('exact', {'mode': 'exact', 'n_steps': 50}),
        ]

        for mode_name, kwargs in modes_to_test:
            @jax.jit
            def bench_fn(c, e, kwargs=kwargs):
                return compute_cl_high_z(c, e, chi_max_box, None, cmb_z_source, **kwargs)

            # Warmup
            _ = bench_fn(cosmo_test, ell_eval).block_until_ready()

            # Measure
            t0 = time.time()
            _ = bench_fn(cosmo_test, ell_eval).block_until_ready()
            elapsed = time.time() - t0

            results[mode_name]['times'].append(elapsed)
            print(f"    {mode_name:12s}: {elapsed:6.3f}s")

    # Create performance plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        'fixed': '#2ca02c',
        'taylor': '#ff7f0e',
        'exact_linear': '#9467bd',
        'exact': '#d62728'
    }

    markers = {'fixed': 'o', 'taylor': 's', 'exact_linear': '^', 'exact': 'D'}

    # Left plot: Execution time
    for mode_name in ['fixed', 'taylor', 'exact_linear', 'exact']:
        times = results[mode_name]['times']
        label_map = {
            'fixed': 'Fixed (cached)',
            'taylor': 'Taylor (1st order)',
            'exact_linear': 'Exact (Linear Pk)',
            'exact': 'Exact (full)'
        }
        ax1.plot(grid_sizes, times, marker=markers[mode_name],
                color=colors[mode_name], linewidth=2, markersize=8,
                label=label_map[mode_name])

    ax1.set_xlabel('Grid Size (pixels per side)', fontsize=12)
    ax1.set_ylabel('Execution Time (s)', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=10)
    ax1.set_title('Computation Time vs Grid Size', fontsize=12, fontweight='bold')

    # Right plot: Speedup relative to exact
    for mode_name in ['fixed', 'taylor', 'exact_linear']:
        speedups = [results['exact']['times'][i] / results[mode_name]['times'][i]
                   for i in range(len(grid_sizes))]
        label_map = {
            'fixed': 'Fixed',
            'taylor': 'Taylor',
            'exact_linear': 'Exact (Linear Pk)'
        }
        ax2.plot(grid_sizes, speedups, marker=markers[mode_name],
                color=colors[mode_name], linewidth=2, markersize=8,
                label=label_map[mode_name])

    ax2.axhline(1, color='gray', linestyle=':', alpha=0.5, linewidth=1.5, label='Exact (baseline)')
    ax2.set_xlabel('Grid Size (pixels per side)', fontsize=12)
    ax2.set_ylabel('Speedup (relative to exact mode)', fontsize=12)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    ax2.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=10)
    ax2.set_title('Speedup Factor', fontsize=12, fontweight='bold')

    # Add annotations
    textstr = 'n_steps = 50\nCPU mode'
    props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.15}
    ax2.text(0.97, 0.03, textstr, transform=ax2.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'highz_modes_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Performance plot saved to: {output_path}")
    plt.close()

    # Print summary table
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY (256×256 grid)")
    print("=" * 70)
    idx = -1  # Last grid size
    for mode_name in ['fixed', 'taylor', 'exact_linear', 'exact']:
        t = results[mode_name]['times'][idx]
        speedup = results['exact']['times'][idx] / t
        print(f"  {mode_name:12s}: {t:6.3f}s  (speedup: {speedup:5.1f}×)")


def main():
    """Run both benchmarks."""
    print("\n" + "=" * 70)
    print("HIGH-Z CORRECTION MODES: COMPREHENSIVE BENCHMARK")
    print("=" * 70)

    # Run benchmarks
    benchmark_precision()
    benchmark_performance()

    print("\n" + "=" * 70)
    print("✓ All benchmarks completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
