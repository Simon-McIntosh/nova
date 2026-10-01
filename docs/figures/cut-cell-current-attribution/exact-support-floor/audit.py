"""Recompute attribution closure and require complete paired controls."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def validate(report):
    expected = {(case, rung) for case in report["cases"] for rung in report["rungs"]}
    assert report["completed"]
    assert {(r["case"], r["requested_cells"]) for r in report["rows"]} == expected
    assert len(report["rows"]) == len(expected)
    for row in report["rows"]:
        for mode in ("exact", "chord"):
            value = row["modes"][mode]
            cells = value["cells"]
            assert len(cells) == row["realised_cells"]
            target = value["archived_target_current_a"]
            booked = sum(c["booked_moments_a_am_am"][0] for c in cells)
            np.testing.assert_allclose(booked, value["booked_current_a"], rtol=1e-12)
            np.testing.assert_allclose(
                booked / target, value["archived_support_fraction"], rtol=1e-9
            )
            fields = (
                "self_intersection_loss_a",
                "simple_moment_integration_error_a",
                "support_geometry_error_a",
                "non_simple_integration_residual_a",
            )
            for field in fields:
                cell_sum = sum(
                    c["support_geometry_error_a_am_am"][0]
                    if field == "support_geometry_error_a"
                    else c[field]
                    for c in cells
                )
                np.testing.assert_allclose(
                    cell_sum, value[field], rtol=1e-10, atol=1e-7
                )
                np.testing.assert_allclose(
                    sum(c[field] for c in value["classes"].values()),
                    value[field],
                    rtol=1e-10,
                    atol=1e-7,
                )
            np.testing.assert_allclose(
                sum(value[f] for f in fields),
                -value["missing_current_against_true_region_a"],
                rtol=1e-10,
                atol=1e-7,
            )
            non_simple = json.loads((HERE / value["non_simple_artifact"]).read_text())
            assert non_simple["count"] == value["non_simple_count"]
            assert non_simple["count"] == len(non_simple["cells"])
            for cell in non_simple["cells"]:
                assert cell["vertices_rz_m"] and cell["atomic_cell_vertices_rz_m"]
                assert "branch_support_pieces_rz_m" in cell
                np.testing.assert_allclose(
                    cell["production_formula_analytic_density_current_a"]
                    - cell["geometric_lobe_union_current_a"],
                    cell["self_intersection_loss_a"],
                    rtol=1e-10,
                    atol=1e-8,
                )
        e, c = row["modes"]["exact"], row["modes"]["chord"]
        control = abs(1 - c["support_fraction_of_archived_target"]) < 0.1 * abs(
            1 - e["support_fraction_of_archived_target"]
        )
        assert control
        assert c["negative_control"]["absolute_deficit_below_tenth_of_exact"] == control
        if row["case"] == "diverted-single-null" and row["requested_cells"] == 110:
            assert abs(e["centroid_displacement_from_fixture_mm"][1] + 35.4) < 1
    assert len(report["panels"]) == 2
    for panel in report["panels"]:
        assert (HERE / panel["path"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


if __name__ == "__main__":
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    print(f"revision={revision} tree={ROOT} command={sys.argv!r}")
    payload = json.loads((HERE / "report.json").read_text())
    validate(payload)
    print(
        "PASS: ten paired rows; cell/class totals, four-term closure, controls, panels"
    )
    shortened = payload | {"rows": payload["rows"][:-1]}
    try:
        validate(shortened)
    except AssertionError:
        print("REFUSED: one missing case/rung row")
    else:
        raise AssertionError("incomplete receipt was accepted")
    control = payload["rows"][0]["modes"]["chord"]["negative_control"]
    control["absolute_deficit_below_tenth_of_exact"] = False
    try:
        validate(payload)
    except AssertionError:
        print("REFUSED: false chord control verdict")
    else:
        raise AssertionError("false control verdict was accepted")
