"""Verify the banked moment-route scorecard is complete and mechanical."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path(__file__).resolve().parent


def main() -> None:
    report = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
    assert report["oracle"]["held_worktree_dependency"] is False
    assert set(report["functionals"]) == {"smooth", "steep_pedestal"}
    assert set(report["population_definitions"]["counts"]) == {
        "interior",
        "boundary_clipped",
        "ring",
    }
    for functional in report["functionals"].values():
        assert functional["truth"]["unconverged_cells"] == []
        assert set(functional["errors"]) == {
            "degree_nine_fit",
            "faithful_duffy",
        }
    decision = report["decision"]
    expected = (
        decision["fidelity_improvement_factor"] >= decision["fidelity_threshold"]
        and decision["cost_ratio"] <= decision["cost_ratio_ceiling"]
    )
    assert decision["faithful_qualifies"] is expected
    assert report["bounds_moved_or_applied"] is False
    for artifact in report["artifacts"]:
        assert (OUTPUT / artifact).is_file()
    print("scorecard verification: passed")


if __name__ == "__main__":
    main()
