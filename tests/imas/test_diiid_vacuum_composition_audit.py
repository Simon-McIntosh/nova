import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[2] / "benchmarks" / "diiid_vacuum_composition_audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diiid_vacuum_composition_audit", MODULE_PATH
)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_magnitude_reports_total_flux_without_fitting():
    values = np.array([-1.0, 0.0, 1.0]) / (2.0 * np.pi)
    result = audit._magnitude(values)
    np.testing.assert_allclose(result["centered_rms_wb"], np.sqrt(2.0 / 3.0))
    np.testing.assert_allclose(result["span_wb"], 2.0)


def test_analytic_patch_has_declared_total_current():
    radius = np.linspace(1.0, 2.0, 33)
    height = np.linspace(-0.8, 0.8, 33)
    _, _, current = audit._plasma_patch(radius, height, 1.5, 0.0, total_current_a=7.5e5)
    np.testing.assert_allclose(np.sum(current), 7.5e5, rtol=1.0e-14)


def test_unit_receipt_identifies_kiloampere_scale(tmp_path, monkeypatch):
    loader = tmp_path / "loader.py"
    loader.write_text(
        "return np.asarray(table[name][0].as_py(), dtype=dtype)\n"
        "values=_array(table, name)\n"
    )
    monkeypatch.setattr(audit, "AMBIX_LOADER", loader)

    class Schema:
        metadata = {b"fusion_coil_units": b"kA.turn/v1"}

        @staticmethod
        def field(name):
            return f"{name}: list<float>"

    frames = [
        {"raw_plasma_current": 500.0, "extracted_plasma_current_a": -5.1e5},
        {"raw_plasma_current": 600.0, "extracted_plasma_current_a": -6.0e5},
        {"raw_plasma_current": 700.0, "extracted_plasma_current_a": -6.9e5},
    ]
    result = audit._unit_receipt(frames, Schema())
    assert result["verdict"] == "kA"
    assert result["ampere_per_raw_unit"]["median"] == 1000.0
