"""Verify the faithful-route integration receipt."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path(__file__).resolve().parent


def main() -> None:
    report = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
    assert report["faithful"]["round_off_class"]
    assert (
        report["faithful"]["forcing_sup_wb"]
        <= report["faithful"]["round_off_ceiling_wb"]
    )
    assert report["topology_qualification"]["excluded_current_sup_a"] == 0.0
    assert report["topology_qualification"]["excluded_first_sup_a_m"] == 0.0
    assert report["adjudication"]["steep_ring_fidelity_improvement_factor"] >= 3.0
    assert report["adjudication"]["faithful_to_fit_cpu_cost_ratio"] <= 2.0
    assert report["bounds_moved_or_applied"] is False
    print("faithful moment integration receipt: passed")


if __name__ == "__main__":
    main()
