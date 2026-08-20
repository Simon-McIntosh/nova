from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "diiid_netcdf_census.py"
SPEC = importlib.util.spec_from_file_location("diiid_netcdf_census", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_classification_firewalls_equilibrium_and_magnetics_signals():
    category, _ = MODULE.classify_path("equilibrium", "time_slice/profiles_2d/psi")
    assert category == MODULE.FIREWALLED

    for path in (
        "flux_loop/flux/data",
        "b_field_pol_probe/field/time",
        "diamagnetic_flux/data_error_upper",
        "ip/data",
    ):
        category, _ = MODULE.classify_path("magnetics", path)
        assert category == MODULE.NOT_ADMISSIBLE


def test_classification_keeps_geometry_and_actuator_description_admissible():
    paths = (
        ("magnetics", "flux_loop/position/r"),
        ("pf_active", "coil/element/geometry/outline/r"),
        ("pf_active", "coil/element/turns_with_sign"),
        ("tf", "b_field_tor_vacuum_r/data"),
        ("wall", "description_2d/limiter/unit/outline/z"),
    )
    assert all(
        MODULE.classify_path(ids_name, path)[0] == MODULE.MACHINE_DESCRIPTION
        for ids_name, path in paths
    )


def test_summary_counts_every_field_once():
    census = {
        "ids": {
            "equilibrium": {
                "dd_version": "3.41.0",
                "homogeneous_time": 1,
                "filled_leaf_count": 2,
                "fields": [
                    {"admissibility": MODULE.FIREWALLED},
                    {"admissibility": MODULE.FIREWALLED},
                ],
            },
            "wall": {
                "dd_version": "3.41.0",
                "homogeneous_time": 2,
                "filled_leaf_count": 1,
                "fields": [{"admissibility": MODULE.MACHINE_DESCRIPTION}],
            },
        }
    }
    rendered = MODULE.summary_html(census)
    assert "<table>" in rendered
    assert "equilibrium" in rendered
    assert "wall" in rendered
    assert rendered.count("<tr>") == 3


def test_declared_ids_and_reasons_are_exhaustive():
    assert MODULE.IDS_NAMES == ("equilibrium", "magnetics", "pf_active", "tf", "wall")
    assert set(MODULE.CLASS_REASONS) == {
        MODULE.MACHINE_DESCRIPTION,
        MODULE.NOT_ADMISSIBLE,
        MODULE.FIREWALLED,
    }
