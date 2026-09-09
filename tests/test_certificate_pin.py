"""Pinned census facts for the published production-route certificate."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT / "docs/figures/gs-absolute-accuracy/solovev-certificate-production-route.json"
)


def _receipt() -> dict[str, object]:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_production_route_census_and_execution_configuration_are_pinned() -> None:
    """Keep every terminal qualification and its declared H200 lane visible."""

    receipt = _receipt()
    assert receipt["verdict"] == {
        "all_locked_recovery_bounds_reproduced": True,
        "all_rows_retained": True,
        "case_count": 4,
        "qualified_rows": 3,
        "resolution_rows": 16,
        "schema_valid": True,
        "unqualified_rows": 13,
    }

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
        assert [
            row["solver"]["qualification"] == "qualified" for row in rows
        ] == expected_flags
        for row in rows:
            lane = row["lane"]
            assert lane["jax_platforms"] == "cuda"
            assert lane["precision"] == "float64"
            assert lane["cpu_count"] == 4
            assert lane["threaded_settings"] == {
                "mkl_num_threads": "4",
                "numexpr_num_threads": "4",
                "omp_num_threads": "4",
                "openblas_num_threads": "4",
                "xla_flags": None,
            }


def test_certificate_figure_renderer_uses_line_contours_without_scatter() -> None:
    """Keep the certificate figures readable as geometric fields, not dots."""

    source = (ROOT / "benchmarks/solovev_certificate.py").read_text(encoding="utf-8")
    assert "contourf" not in source
    assert "scatter(" not in source
    assert "poloidal.draw_flux_contours" in source
    assert "poloidal.draw_nulls" in source
    assert "poloidal.draw_wall" in source
