"""Streaming the interlaced paint over chunks must equal painting the catalogue at once.

The randoms catalogue is far too large to hold, so the selection function is built by
accumulating the interlacing shifts chunk by chunk and combining them once at the end. That is
only legitimate because painting is linear in the positions; these tests pin it.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from desi_cmb_fli.bricks import (
    INTERLACE_ORDER,
    interlace_accumulate,
    interlace_finalize,
    interlace_paint_deconv,
)

jax.config.update("jax_enable_x64", True)

PAINT_SHAPE = (14, 14, 14)
FINAL_SHAPE = (8, 8, 8)


def _positions(n, seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(0.0, PAINT_SHAPE[0], size=(n, 3)))


@pytest.mark.parametrize("n_chunks", [1, 2, 5])
def test_chunked_accumulation_equals_one_call(n_chunks):
    pos = _positions(400, 0)
    one = interlace_paint_deconv(pos, PAINT_SHAPE, FINAL_SHAPE)

    shifted = [jnp.zeros(PAINT_SHAPE) for _ in range(INTERLACE_ORDER)]
    for part in np.array_split(np.asarray(pos), n_chunks):
        shifted = interlace_accumulate(shifted, jnp.asarray(part))
    streamed = interlace_finalize(shifted, PAINT_SHAPE, FINAL_SHAPE)

    np.testing.assert_allclose(np.asarray(streamed), np.asarray(one), rtol=0, atol=1e-12)


def test_an_empty_chunk_changes_nothing():
    pos = _positions(200, 1)
    shifted = [jnp.zeros(PAINT_SHAPE) for _ in range(INTERLACE_ORDER)]
    shifted = interlace_accumulate(shifted, pos)
    ref = interlace_finalize(shifted, PAINT_SHAPE, FINAL_SHAPE)
    shifted = interlace_accumulate(shifted, jnp.zeros((0, 3)))
    np.testing.assert_allclose(
        np.asarray(interlace_finalize(shifted, PAINT_SHAPE, FINAL_SHAPE)),
        np.asarray(ref), rtol=0, atol=1e-12)


def test_the_total_weight_is_conserved():
    """Interlacing, deconvolution and the Fourier crop are all mean-preserving up to the cell
    volume ratio, so the summed mesh must still count every object."""
    pos = _positions(300, 2)
    mesh = interlace_paint_deconv(pos, PAINT_SHAPE, FINAL_SHAPE)
    np.testing.assert_allclose(float(jnp.sum(mesh)), 300.0, rtol=1e-10)


def test_chunking_matches_at_the_paint_grid_too():
    """The accumulator itself, before any Fourier work, must be chunk-invariant."""
    pos = _positions(250, 3)
    whole = interlace_accumulate([jnp.zeros(PAINT_SHAPE)] * INTERLACE_ORDER, pos)
    parts = [jnp.zeros(PAINT_SHAPE)] * INTERLACE_ORDER
    for chunk in np.array_split(np.asarray(pos), 4):
        parts = interlace_accumulate(parts, jnp.asarray(chunk))
    for a, b in zip(whole, parts, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12)
