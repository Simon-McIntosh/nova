"""Retain the published certificate receipt without accepting failed rows."""

from __future__ import annotations

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
