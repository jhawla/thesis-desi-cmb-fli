"""CLASS linear P_cb(k, z = 1) of AbacusSummit c000 with Omega_m moved, the reference of
compare_linear_power.py --omega_m_scan.

Omega_m moves as the model moves it: omega_cdm = (Omega_m - Omega_b) h^2 - omega_nu at fixed h,
Omega_b, n_s, A_s, one 0.06 eV neutrino (N_ur = 2.0328), as in c000. z = 1 is the CLASS redshift of
the Abacus ICs. Needs cosmoprimo with CLASS, not this package; at NERSC:

    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
    python scripts/linear_power_class_reference.py

Output: figures/spectra_diagnostic/linear_power_class_omega_m.npz (k [h/Mpc], omega_m, class_<i>).
"""

from pathlib import Path

import numpy as np
from cosmoprimo import Cosmology

ROOT = Path(__file__).resolve().parents[1]
H, OMEGA_B, N_S, A_S, M_NU, Z = 0.6736, 0.04930169, 0.9649, 2.083e-9, 0.06, 1.0
OMEGA_M = [0.12, 0.15, 0.2, 0.22, 0.25, 0.28, 0.315192, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]


def main():
    k = np.logspace(-4, 1, 3000)
    out = {"k": k, "omega_m": np.array(OMEGA_M), "z": Z}
    for i, om in enumerate(OMEGA_M):
        omch2 = om * H**2 - OMEGA_B * H**2 - M_NU / 93.14
        cosmo = Cosmology(h=H, omega_b=OMEGA_B * H**2, omega_cdm=omch2, m_ncdm=[M_NU], N_ur=2.0328,
                          n_s=N_S, A_s=A_S, engine="class")
        pk = cosmo.get_fourier().pk_interpolator(of="delta_cb", extrap_kmin=1e-6)
        out[f"class_{i}"] = pk(k, z=Z)
        print(f"Omega_m = {om}: omega_cdm = {omch2:.5f}")
    path = ROOT / "figures/spectra_diagnostic/linear_power_class_omega_m.npz"
    np.savez(path, **out)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
