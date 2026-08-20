"""Validate the forcing-residual decomposition artifact."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Check controls, closure, response solves, and figure integrity."""
    report = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    controls = report["controls"]
    assert (
        abs(
            controls["forcing_grid_sup_fraction_of_span"]
            - controls["expected_forcing_grid_sup_fraction_of_span"]
        )
        < 1.0e-14
    )
    assert (
        abs(
            controls["comparator_residual_projection_on_total_forcing"]
            - controls["expected_comparator_residual_projection"]
        )
        < 1.0e-12
    )
    assert report["additive_closure"]["relative_sup"] < 1.0e-12
    assert abs(report["additive_closure"]["projection_fraction_sum"] - 1.0) < 1.0e-12
    for component in report["components"].values():
        assert component["root_response"]["solve"]["gmres_info"] == 0
        assert component["root_response"]["solve"]["residual_relative_sup"] < 1.0e-10
        assert all(
            math.isfinite(value)
            for value in component["forcing"].values()
            if isinstance(value, float)
        )
    assert report["verdict"]["dominant_component"] == "source_density"
    assert (ROOT / "decomposition.png").stat().st_size > 50_000
    print("VERIFY_RESULTS_EXIT=0")


if __name__ == "__main__":
    main()
