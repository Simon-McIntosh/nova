from __future__ import annotations

import numpy as np
import pytest

from benchmarks.diiid_machine_description import build_receipt, write_figures
from nova.imas import diiid_description as diiid
from nova.imas.machine import StaticMachineDescription


def _dataset_row() -> dict[str, object]:
    names = list(diiid.POLOIDAL_CONDUCTORS)
    return {
        "coil_name": names,
        "coil_input_column": [f"magnetics_{name}" for name in names],
        "coil_R": [
            0.8608,
            0.8614,
            0.8628,
            0.8611,
            1.0041,
            2.6124,
            2.3733,
            1.2518,
            1.6890,
            0.8608,
            0.8607,
            0.8611,
            0.8630,
            1.0025,
            2.6124,
            2.3834,
            1.2524,
            1.6889,
            0.7225,
        ],
        "coil_Z": [
            0.1683,
            0.5081,
            0.8491,
            1.1899,
            1.5169,
            0.4376,
            1.1171,
            1.6019,
            1.5874,
            -0.1737,
            -0.5135,
            -0.8543,
            -1.1957,
            -1.5169,
            -0.4376,
            -1.1171,
            -1.6027,
            -1.5780,
            0.0,
        ],
        "coil_width": [
            0.0508,
            0.0508,
            0.0508,
            0.0508,
            0.1392,
            0.1732,
            0.1880,
            0.2349,
            0.1694,
            0.0508,
            0.0508,
            0.0508,
            0.0508,
            0.1392,
            0.1732,
            0.1880,
            0.2349,
            0.1694,
            0.12875,
        ],
        "coil_height": [
            0.32106,
            0.32106,
            0.32106,
            0.32106,
            0.1194,
            0.1946,
            0.1692,
            0.0851,
            0.1331,
            0.32106,
            0.32106,
            0.32106,
            0.32106,
            0.1194,
            0.1946,
            0.1692,
            0.0851,
            0.1331,
            3.5058,
        ],
        "coil_angle1": [
            0.0,
            0.0,
            0.0,
            0.0,
            45.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -45.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "coil_angle2": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            92.4,
            108.06,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -92.4,
            -108.06,
            0.0,
            0.0,
            0.0,
        ],
        "thomson_chord_name": [f"TS_{index:02d}" for index in range(70)],
        "thomson_chord_R": np.linspace(1.485, 2.0613, 70),
        "thomson_chord_Z": np.linspace(-1.0713, 1.2155, 70),
        "efit_grid_R": np.linspace(0.84, 2.54, 65),
        "efit_grid_Z": np.linspace(-1.6, 1.6, 65),
    }


def test_dataset_geometry_routes_all_sections_through_static_machine() -> None:
    row = _dataset_row()
    description = diiid.dataset_machine_description(
        row, source_row="d3d_shot_example.parquet"
    )

    assert isinstance(description.machine, StaticMachineDescription)
    assert description.provenance_complete
    assert len(description.machine.active_sections) == 19
    assert len(description.machine.sightlines) == 70
    assert description.machine.contour is None
    assert description.machine.passive_loop_count == 0
    assert (len(description.grid_z), len(description.grid_r)) == (65, 65)

    sections = {
        section.name: section for section in description.machine.active_sections
    }
    for conductor in description.physical.conductors:
        if conductor.vertices is None:
            continue
        routed = np.column_stack(
            [
                sections[conductor.name].section.data["r"],
                sections[conductor.name].section.data["z"],
            ]
        )
        np.testing.assert_allclose(routed, conductor.vertices, rtol=0.0, atol=1.0e-12)


def test_receipt_quantifies_complete_geometry_and_explicit_absence() -> None:
    receipt, _ = build_receipt(_dataset_row(), source_row="d3d_shot_example.parquet")
    quantities = receipt["quantities"]

    assert receipt["provenance_complete"]
    assert len(receipt["physical_geometry_digest"]) == 64
    assert quantities["poloidal_conductors"]["count"] == 19
    assert quantities["poloidal_conductors"]["skewed_count"] == 6
    assert quantities["thomson_chords"]["count"] == 70
    assert quantities["efit_grid"]["shape"] == [65, 65]
    route = receipt["machine_dataclass_route"]
    assert route["maximum_vertex_route_difference_m"] == 0.0
    assert quantities["bcoil"]["axisymmetric_poloidal_section"] is None
    for name in ("wall_contour", "passive_structure"):
        assert quantities[name]["acceptance"] == "absent"
        assert quantities[name]["external_source"] is None
        assert quantities[name]["value"] is None


def test_receipt_preserves_representative_shipped_section_values() -> None:
    receipt, _ = build_receipt(_dataset_row(), source_row="d3d_shot_example.parquet")
    conductors = {
        item["name"]: item
        for item in receipt["quantities"]["poloidal_conductors"]["conductors"]
    }

    assert conductors["F5A"]["centre_m"] == [1.0041, 1.5169]
    assert conductors["F5A"]["width_m"] == 0.1392
    assert conductors["F5A"]["height_m"] == 0.1194
    assert conductors["F5A"]["skew"]["effective_deg"] == 45.0
    assert conductors["F6B"]["skew"]["effective_deg"] == -92.4
    assert len(conductors["ECOILA"]["section_vertices_m"]) == 4


def test_geometry_figures_cover_overview_diagnostics_and_skew_detail(tmp_path) -> None:
    _, description = build_receipt(
        _dataset_row(), source_row="d3d_shot_example.parquet"
    )

    paths = write_figures(description, tmp_path)

    assert {path.name for path in paths} == {
        "conductor_sections.png",
        "skewed_conductor_sections.png",
        "thomson_grid_extent.png",
    }
    assert all(path.stat().st_size > 10_000 for path in paths)


def test_dataset_route_rejects_mismatched_chord_columns() -> None:
    row = _dataset_row()
    row["thomson_chord_Z"] = row["thomson_chord_Z"][:-1]

    with pytest.raises(
        diiid.DiiidDescriptionError, match="geometry columns have different lengths"
    ):
        diiid.dataset_machine_description(row, source_row="d3d_shot_example.parquet")
