"""Check complete physical arms and refuse incomplete or hidden-panel evidence."""

import json
import math
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print(f"revision={revision} tree={Path.cwd()} command=python {__file__}", flush=True)
r = json.loads((root / "trip-panels.json").read_text())
assert r["status"] == "complete", "physical measurement is incomplete"
assert r["slurm_job_id"]
assert r["sacct_state"] == "COMPLETED"
assert r["sacct_exit_code"] == "0:0"
assert len(r["arms"]) == 3
before, after, control = r["arms"]
assert before["revision"] == r["pre_repair_revision"]
assert after["revision"] == control["revision"] == r["dispatch_base"]
assert [a["clip_mode"] for a in r["arms"]] == ["exact", "exact", "chord"]
assert control["trip_count"] == 1
panels = 0
for arm in r["arms"]:
    assert arm["status"] == "measured" and arm["process_exit_status"] == 0
    assert arm["capture_terminal_and_history_equal"]
    assert arm["trip_count"] == len(arm["trips"]) > 0
    assert arm["construction"]["policy"]["route"] == "newton_krylov"
    assert arm["realised_cells"] == 132
    assert arm["certificate_residual_bound"] == 1e-12
    assert math.isfinite(arm["terminal_residual"])
    assert arm["terminal_residual"] == arm["trips"][-1]["residual"]
    loss = None
    for index, row in enumerate(arm["trips"], 1):
        assert row["trip"] == index
        distance = row["saddle_distance_m"]
        outside = distance is None or distance > 0.15
        assert row["saddle_outside_bound"] == outside
        if outside and loss is None:
            loss = index
        assert row["moment_element_count"] > 0
        assert row["nonfinite_counter_positive_control"] > 0
        assert len(row["cell_current_a"]) == len(row["flood_labels"]) == 132
        panel = row["panel"]
        assert panel["axis_off"] and min(panel["contour_segments"]) > 0
        assert panel["null_markers"][0]["x_points_drawn"] == 1
        if row["nulls"]["x_point_rz_m"] is not None:
            assert panel["null_markers"][1]["x_points_drawn"] == 1
        assert all(
            x["x_points_dropped_outside_wall"] == 0 for x in panel["null_markers"]
        )
        for kind in ["png", "svg"]:
            assert (root / panel[kind]).stat().st_size > 1000
        panels += 1
    assert loss == arm["saddle_loss_trip"]
loss = before["saddle_loss_trip"]
verdict = "neither"
if loss is not None and after["saddle_loss_trip"] is None:
    if before["trips"][loss - 1]["nonfinite_moment_count"] > 0:
        verdict = "support-overflow"
if loss is not None and after["saddle_loss_trip"] is not None:
    if all(
        t["nonfinite_moment_count"] == 0 for a in [before, after] for t in a["trips"]
    ):
        verdict = "census-read"
assert r["verdict"] == verdict
print(f"PASS: 3 measured arms, {panels} PNG/SVG panel pairs; verdict={verdict}")
