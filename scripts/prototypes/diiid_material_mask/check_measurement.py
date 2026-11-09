"""Verify the paired solver cohort against its saved states and job log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from nova.equilibrium.wall_mask import inside_polygon

PROBE = Path(__file__).resolve().parent.parent / "diiid_forward_convergence"
sys.path.insert(0, str(PROBE))
from boundary_competition import _wall_distance  # noqa: E402


def check(root: Path) -> list[dict]:
    rows = []
    for frame in (179, 137, 146):
        for branch in ("limited", "diverted"):
            for arm in ("pseudo", "physical"):
                stem = root / "measurement" / arm / f"frame-{frame}-{branch}"
                row = json.loads(stem.with_suffix(".json").read_text())
                assert row["status"] == "solved", row
                assert row["storage_order"] == "radius_height"
                with np.load(stem.with_suffix(".npz")) as state:
                    flux = state["flux"]
                    assert flux.shape == (len(state["radius"]), len(state["height"]))
                    np.testing.assert_array_equal(
                        state["state"][: flux.size].reshape(flux.shape), flux
                    )
                    assert int(state["connected"].sum()) == row["support_cells"]
                    assert int(state["inside_material"].sum()) == row["material_cells"]
                    assert row["material_cells"] == (527 if arm == "physical" else 1089)
                    rr, zz = np.meshgrid(
                        state["radius"], state["height"], indexing="ij"
                    )
                    limiter = state["limiter"]
                    contact_control, _ = _wall_distance(limiter, limiter)
                    assert float(np.min(contact_control)) == 0.0
                    reference_gap, _ = _wall_distance(
                        state["reference_boundary"], limiter
                    )
                    row["efit_boundary_min_wall_gap_m"] = float(np.min(reference_gap))
                    inside = inside_polygon(
                        rr.ravel(), zz.ravel(), limiter[:, 0], limiter[:, 1]
                    )
                    assert inside.any() and (~inside).any()
                    assert (
                        int((state["connected"] & ~inside).sum())
                        == row["support_outside_limiter"]
                    )
                    if arm == "physical":
                        np.testing.assert_array_equal(inside, state["inside_material"])
                        assert row["support_outside_limiter"] == 0
                        assert row["axis_inside_limiter"]
                    axis = state["axis"]
                    np.testing.assert_array_equal(axis, row["axis_rz_m"])
                    assert (
                        bool(
                            inside_polygon(
                                axis[0:1], axis[1:2], limiter[:, 0], limiter[:, 1]
                            )[0]
                        )
                        == row["axis_inside_limiter"]
                    )
                rows.append(row)
    assert len(rows) == 12
    log = (root / "measurement-final-lane.log").read_text()
    exits = [line for line in log.splitlines() if line.startswith("FINISH ")]
    expected = {
        f"FINISH {arm} {frame} EXIT=0"
        for arm in ("physical", "pseudo")
        for frame in (179, 137, 146)
    }
    assert len(exits) == 6 and set(exits) == expected, exits
    assert "RUNG=h200" in log and "NOVA_LANE_EXIT_STATUS=0" in log
    assert any(row["support_cells"] > 0 for row in rows)
    print(
        "VERIFIED: 12 solved rows, 6 subprocess exits 0; saved states and counts "
        "agree; all physical masks equal the governed 527-cell material domain."
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = check(args.input)
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
