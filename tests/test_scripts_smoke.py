"""Smoke tests: the diagnostic scripts run end to end on a tiny abacus-mode configuration.

The real kappa-only Abacus config, shrunk (box 1000, 8^3 cells, nside 4) and pointed at a fake
AbacusLensing map. The Abacus loader returns maps on the projection sphere (proj_oversamp 2), which
both scripts must bring to the observable sphere; they failed on it once each.
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tiny_abacus_config(tmp_path):
    import asdf
    import healpy as hp

    cfg = yaml.safe_load(open(ROOT / "configs/inference/abacus/abacus_kappaonly_Nl1p0_fnlfixed_chimin700.yaml"))
    cfg["model"].update(box_shape=[1000.0] * 3, cell_size=125.0, lpt_order=1, init_oversamp=1.0,
                        evol_oversamp=1.0, ptcl_oversamp=1.0, paint_oversamp=1.0)
    cfg["cmb_lensing"].update(nside=4, n_shells=4, chi_matter_min=50.0, chi_high_z_max=600.0,
                              cmb_noise_nell=str(ROOT / "data/N_L_kk_act_dr6_lensing_v1_baseline.txt"))
    kappa = np.random.default_rng(0).normal(scale=1e-2, size=hp.nside2npix(32))
    asdf.AsdfFile({"data": {"kappa": kappa}, "header": {}}).write_to(tmp_path / "kappa.asdf")
    cfg["abacus_kappa"] = {"file": str(tmp_path / "kappa.asdf"), "noise_seed": 5}
    cfg["observation_mode"] = "abacus"
    path = tmp_path / "config.yaml"
    yaml.safe_dump(cfg, open(path, "w"))
    return path


@pytest.mark.parametrize("script, argv", [
    ("plot_2D_maps", ["--output-dir"]),
    ("quick_cl_spectra", ["--output_dir"]),
])
def test_diagnostic_script_runs_in_abacus_mode(tiny_abacus_config, tmp_path, monkeypatch, script, argv):
    out = tmp_path / script
    monkeypatch.setattr(sys, "argv", [script, "--config", str(tiny_abacus_config), argv[0], str(out)])
    _load_script(script).main()
    assert list(out.glob("*.png")), f"{script} wrote no figure"
