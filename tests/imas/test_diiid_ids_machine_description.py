"""Tests for the DIII-D IMAS machine-description receipt."""

from pathlib import Path

from benchmarks.diiid_ids_machine_description import build_receipt, write_figures


def _entry() -> dict:
    return {
        "shot": 1000,
        "run": 0,
        "database": "DIII-D",
        "user": "hoeneno",
        "dd_versions": {
            "wall": "3.28.0",
            "pf_active": "3.28.0",
            "thomson_scattering": "3.28.0",
        },
        "contour": {"kind": "limiter", "r": [1.0, 2.0, 1.0], "z": [-1.0, 0.0, 1.0]},
        "pf_active": [
            {
                "name": "rect",
                "geometry_type": 2,
                "r": 1.2,
                "z": 0.3,
                "width": 0.1,
                "height": 0.2,
            },
            {
                "name": "skew",
                "geometry_type": 1,
                "r": [1.4, 1.5, 1.6, 1.5],
                "z": [0.0, 0.1, 0.1, 0.0],
            },
        ],
        "pf_passive_loop_count": 0,
        "tf_coil_count": 0,
        "thomson_scattering": [
            {"name": "ts", "position": [1.8, 0.0, 6.5], "start": None, "end": None}
        ],
    }


def test_receipt_routes_sections_and_preserves_unreachable_quantities():
    receipt, machine = build_receipt([_entry()])

    quantities = receipt["quantities"]
    assert quantities["wall_or_limiter"]["vertex_count"] == 3
    assert quantities["pf_active"]["coil_count"] == 2
    assert [coil["geometry_class"] for coil in quantities["pf_active"]["coils"]] == [
        "rectangle",
        "outline",
    ]
    assert quantities["pf_passive"]["status"] == "cannot_reach"
    assert quantities["tf"]["status"] == "cannot_reach"
    assert quantities["thomson_scattering"]["endpoint_pair_count"] == 0
    assert machine.sightlines[0].start is None


def test_receipt_pins_ids_cocos_transform_separately_from_the_corpus():
    receipt, _ = build_receipt([_entry()])

    assert receipt["cocos"]["source_index"] == 11
    assert receipt["cocos"]["target_index"] == 17
    assert receipt["cocos"]["factors"]["psi_like"] == -1.0
    assert receipt["cocos"]["factors"]["ip_like"] == 1.0


def test_three_figures_show_geometry_and_absence(tmp_path: Path):
    _, machine = build_receipt([_entry()])

    figures = write_figures(machine, tmp_path)

    assert len(figures) == 3
    assert all(path.stat().st_size > 0 for path in figures)
