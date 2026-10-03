"""Retain the published certificate receipt without accepting failed rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT / "docs/figures/gs-absolute-accuracy/solovev-certificate-production-route.json"
)


def _receipt() -> dict[str, object]:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def _unqualified_rows(receipt: dict[str, object]) -> list[str]:
    rows = []
    for case_name, case in receipt["cases"].items():
        for row in case["rows"]:
            if row["solver"]["qualification"] != "qualified":
                rows.append(f"{case_name}:{row['requested_cells']}")
    return rows


def test_pinned_certificate_retains_its_census_and_render_integrity() -> None:
    """Keep the published receipt internally consistent while it remains data."""

    receipt = _receipt()
    expected_convergence = {
        "diverted-single-null": [True, True, True, False],
        "moderate-rotation-conventional-static": [False, False, False, False],
        "strong-rotation-compact-static": [False, False, False, False],
        "weak-rotation-reactor-static": [False, False, False, False],
    }
    cases = receipt["cases"]
    assert set(cases) == set(expected_convergence)
    for case_name, expected_flags in expected_convergence.items():
        rows = cases[case_name]["rows"]
        assert [row["solver"]["converged"] for row in rows] == expected_flags
        for row in rows:
            render_data = row["render_data"]
            assert render_data["schema"] == "nova.solovev-certificate-render-data"
            assert render_data["version"] == 1
            coordinate_count = len(render_data["coordinates_rz_m"])
            assert coordinate_count == len(render_data["terminal_flux_wb"])
            assert coordinate_count == len(render_data["analytic_flux_wb"])
            figure = row["figure"]
            assert figure["project_absolute_src"].startswith(
                "/nova/figures/gs-absolute-accuracy/solovev/"
            )
            figure_path = ROOT / figure["filesystem_path"]
            assert (
                hashlib.sha256(figure_path.read_bytes()).hexdigest()
                == figure["sha256"]
            )


def test_pinned_certificate_refuses_unqualified_production_rows() -> None:
    """A stored receipt is evidence, never an exemption from qualification."""

    receipt = _receipt()
    verdict = receipt["verdict"]
    assert verdict["case_count"] == 4
    assert verdict["resolution_rows"] == 16
    assert verdict["qualified_rows"] == 1
    assert verdict["unqualified_rows"] == 15
    assert receipt["production_run"]["measurement_scheduler"]["job_id"] == "1268058"

    unqualified = _unqualified_rows(receipt)
    assert len(unqualified) == verdict["unqualified_rows"]
    assert not unqualified, (
        "the pinned production receipt contains unqualified rows and cannot be "
        f"accepted as a certificate: {', '.join(unqualified)}"
    )


def test_certificate_figure_renderer_uses_line_contours_without_scatter() -> None:
    """Keep the certificate figures readable as geometric fields, not dots."""

    source = (ROOT / "benchmarks/solovev_certificate.py").read_text(encoding="utf-8")
    assert "contourf" not in source
    assert "scatter(" not in source
    assert "poloidal.draw_flux_contours" in source
    assert "poloidal.draw_nulls" in source
    assert "poloidal.draw_wall" in source
