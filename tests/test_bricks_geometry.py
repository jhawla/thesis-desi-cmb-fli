"""Geometry regression tests for the cell <-> physical coordinate helpers."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from desi_cmb_fli.bricks import cell2phys_pos, phys2cell_pos


def test_phys_cell_roundtrip_is_consistent():
    box_shape = np.array([200.0, 220.0, 260.0])
    mesh_shape = np.array([20, 22, 26])
    box_center = np.array([10.0, -5.0, 3.0])

    rng = np.random.default_rng(0)
    # Draw points safely inside the volume to avoid edge clipping effects.
    phys = (box_center - box_shape / 2.0) + box_shape * rng.uniform(0.05, 0.95, size=(128, 3))

    cell = phys2cell_pos(phys, box_center, box_shape, mesh_shape)
    back = cell2phys_pos(cell, box_center, box_shape, mesh_shape)

    assert np.all(np.isfinite(cell))
    assert np.all(np.isfinite(back))
    assert np.allclose(back, phys, atol=1e-9)


def test_box_center_maps_to_mesh_center():
    box_shape = np.array([200.0, 220.0, 260.0])
    mesh_shape = np.array([20, 22, 26])
    box_center = np.array([10.0, -5.0, 3.0])

    cell = phys2cell_pos(box_center, box_center, box_shape, mesh_shape)
    assert np.allclose(cell, mesh_shape / 2.0, atol=1e-9)
