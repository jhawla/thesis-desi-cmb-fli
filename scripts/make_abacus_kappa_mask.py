"""Convert an AbacusLensing footprint mask (ASDF, nside=16384) to a HEALPix .npy.

The official AbacusLensing products ship a binary footprint mask per source shell
(``mask_XXXXX.asdf``) at nside=16384, marking pixels where the weak-lensing maps
are available.  ``load_healpix_mask`` (in cmb_lensing.py) reads ``.npy``/FITS masks
and ud_grades them to the model nside, so we degrade the huge ASDF mask once here
and store a compact .npy that the inference config can point to via
``cmb_lensing.mask``.

The full nside=16384 map (3.2e9 pixels) is too large to ud_grade in one shot, so we
stream it in chunks: for each block of high-res pixels we take the non-zero ones,
convert to angles and bin them into the target (lower) nside via ``ang2pix``.

Usage
-----
    python scripts/make_abacus_kappa_mask.py \
        --in  /global/cfs/cdirs/desi/cosmosim/AbacusLensing/v1/AbacusSummit_base_c000_ph000/mask_00047.asdf \
        --out data/abacus_kappa_mask_00047.npy \
        --nside 1024
"""

import argparse

import asdf
import healpy as hp
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in",
        dest="infile",
        default=(
            "/global/cfs/cdirs/desi/cosmosim/AbacusLensing/v1/"
            "AbacusSummit_base_c000_ph000/mask_00047.asdf"
        ),
        help="Input AbacusLensing mask_XXXXX.asdf file.",
    )
    ap.add_argument(
        "--out",
        dest="outfile",
        default="data/abacus_kappa_mask_00047.npy",
        help="Output .npy HEALPix boolean mask (RING ordering).",
    )
    ap.add_argument(
        "--nside",
        type=int,
        default=1024,
        help="Target HEALPix nside for the stored mask (default 1024).",
    )
    ap.add_argument(
        "--step",
        type=int,
        default=50_000_000,
        help="High-res pixels processed per chunk (memory control).",
    )
    args = ap.parse_args()

    out_nside = int(args.nside)
    out = np.zeros(hp.nside2npix(out_nside), dtype=bool)

    with asdf.open(args.infile, lazy_load=True) as a:
        hdr = dict(a.tree.get("header", {}))
        ns_hi = int(hdr.get("HEALPix_nside", 0))
        print(f"[mask] input {args.infile}")
        print(f"[mask] header: {hdr}")
        m = a["data"]["mask"]
        n_hi = m.shape[0]
        if ns_hi == 0:
            ns_hi = hp.npix2nside(n_hi)
        n_set_hi = 0
        for i in range(0, n_hi, args.step):
            block = np.asarray(m[i : i + args.step])
            idx = np.flatnonzero(block != 0) + i
            if idx.size == 0:
                continue
            n_set_hi += idx.size
            theta, phi = hp.pix2ang(ns_hi, idx, nest=False)
            lo = hp.ang2pix(out_nside, theta, phi, nest=False)
            out[np.unique(lo)] = True
            del block, idx, theta, phi, lo

    f_in = n_set_hi / n_hi
    f_out = float(out.mean())
    print(f"[mask] input  nside={ns_hi}  f_sky={f_in:.4f}")
    print(f"[mask] output nside={out_nside}  f_sky={f_out:.4f}  npix_set={int(out.sum())}")

    np.save(args.outfile, out)
    print(f"[mask] saved -> {args.outfile}")


if __name__ == "__main__":
    main()
