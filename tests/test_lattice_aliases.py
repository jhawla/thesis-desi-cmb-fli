"""The particle lattice must not alias into the galaxy band (docs/pipeline.md 2.5).

Harmonic m of the particle lattice, folded by n paint periods, lands at index m * ptcl - n * paint;
order-2 interlacing cancels the odd folds and the crop keeps |index| <= final / 2. These tests pin
the rule on the model's own interlaced paint and on the grids the pipeline has used.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from desi_cmb_fli.bricks import interlace_paint_deconv, lattice_aliases

jax.config.update("jax_enable_x64", True)

PAINT, FINAL = 20, 12


def _in_band_lattice(ptcl):
    """Largest |delta_k| of the undisplaced lattice painted as the model paints it."""
    q = np.indices((ptcl,) * 3).reshape(3, -1).T * (PAINT / ptcl)
    out = np.asarray(interlace_paint_deconv(jnp.asarray(q), (PAINT,) * 3, (FINAL,) * 3))
    d = out / out.mean() - 1.0
    return np.abs(np.fft.fftn(d)).max() / d.size


@pytest.mark.parametrize("ptcl", [42, 44, 46])
def test_flagged_counts_leave_a_lattice_pattern(ptcl):
    assert lattice_aliases(ptcl, PAINT, FINAL)
    assert _in_band_lattice(ptcl) > 1e-3


@pytest.mark.parametrize("ptcl", [48, 50])
def test_clean_counts_leave_none(ptcl):
    assert not lattice_aliases(ptcl, PAINT, FINAL)
    assert _in_band_lattice(ptcl) < 1e-4


def test_odd_folds_cancel_with_interlacing():
    # 50 on 20: harmonic 2 lands at index 0 with fold 5, cancelled by order-2 interlacing
    assert (2, 5, 0) not in lattice_aliases(50, PAINT, FINAL)
    assert (2, 5, 0) in lattice_aliases(50, PAINT, FINAL, interlace_order=1)


def test_pipeline_grids():
    # box 5000 production: 200 particles, paint 94, final 54 -> first harmonic at index 12
    assert (1, 2, 12) in lattice_aliases(200, 94, 54)
    # box 7500, 140 particles on paint 140: harmonic 2 commensurate at index 0
    assert (2, 2, 0) in lattice_aliases(140, 140, 80)
    # box 7500, 200 particles: harmonic 1 at 60 is cropped, harmonic 2 (fold 3) cancelled,
    # harmonic 3 on the Nyquist plane
    assert lattice_aliases(200, 140, 80) == [(3, 4, 40)]
    assert not lattice_aliases(324, 140, 80) and lattice_aliases(300, 140, 80)
