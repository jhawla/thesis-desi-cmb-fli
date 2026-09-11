"""Cache-key tests for the reduced randoms meshes.

A stale randoms cache would silently feed a wrong selection function into the
inference, so the key must change whenever anything that changes the reduction
changes -- and stay put otherwise.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import _randoms_cache_path


@pytest.fixture
def base(tmp_path):
    randoms = tmp_path / "randoms.fits"
    randoms.write_bytes(b"not a real catalogue, only stat() is read")
    return {
        "cache_dir": tmp_path / "cache",
        "rand_paths": [randoms],
        "z_range": (0.4, 1.1),
        "box_shape": [7500.0, 7500.0, 7500.0],
        "mesh_shape": (80, 80, 80),
        "observer_pos": np.array([3750.0, 3750.0, 3750.0]),
        "cosmo": get_cosmology(Omega_m=0.315192, sigma8=0.811355),
        "max_rows": None,
        "chunk_rows": 50_000_000,
        "paint_shape": (140, 140, 140),
    }


def key(**kw):
    return _randoms_cache_path(**kw).name


def test_key_is_deterministic(base):
    assert key(**base) == key(**base)


@pytest.mark.parametrize(
    "field, value",
    [
        ("z_range", (0.4, 1.0)),
        ("box_shape", [5000.0, 5000.0, 5000.0]),
        ("mesh_shape", (128, 128, 128)),
        ("observer_pos", np.array([0.0, 0.0, 0.0])),
        ("max_rows", 1_000_000),
        ("chunk_rows", 10_000_000),
        ("paint_shape", (160, 160, 160)),
        ("paint_shape", None),
    ],
)
def test_key_changes_when_the_reduction_changes(base, field, value):
    assert key(**{**base, field: value}) != key(**base)


def test_key_changes_with_background_cosmology(base):
    # z -> comoving distance depends on Omega_m, so the reduction does too.
    other = get_cosmology(Omega_m=0.40, sigma8=0.811355)
    assert key(**{**base, "cosmo": other}) != key(**base)


def test_key_ignores_nuisance_latents(base):
    # b1 & friends never touch the randoms geometry: they must not invalidate a
    # 30-minute cache. get_cosmology drops them, so they are not pytree leaves.
    same = get_cosmology(Omega_m=0.315192, sigma8=0.811355, b1=9.99, bn2=-140.0)
    assert key(**{**base, "cosmo": same}) == key(**base)


def test_key_changes_when_the_catalogue_file_changes(base):
    before = key(**base)
    path = base["rand_paths"][0]
    path.write_bytes(b"a catalogue of a different size entirely, so stat differs")
    assert key(**base) != before


def test_cache_dir_does_not_enter_the_key(base, tmp_path):
    moved = key(**{**base, "cache_dir": tmp_path / "elsewhere"})
    assert moved == key(**base)
