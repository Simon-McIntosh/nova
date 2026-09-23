"""Independently check serialized map metrics, support and threshold claims."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
receipt = json.loads((ROOT / "map-fidelity.json").read_text())
print(
    f"revision={receipt['source_revision']} tree={receipt['worktree']} "
    f"command={sys.argv!r}"
)
assert receipt["completed"]
expected = {
    (case, rung, mode)
    for case in receipt["cases"]
    for rung in receipt["requested_rungs"]
    for mode in receipt["clip_modes"]
}
seen = {
    (row["case"], row["requested_cells"], row["clip_mode"]) for row in receipt["rows"]
}
assert expected == seen and len(expected) == len(receipt["rows"])
controls = receipt["instrument_controls"]
assert controls and all(controls.values())
assert (ROOT / "negative-control.log").read_text().splitlines()[0] == receipt[
    "negative_control_declaration"
]
for row in receipt["rows"]:
    assert row["status"] == "measured"
    assert row["source_revision"] == receipt["source_revision"]
    assert row["jax_backend"] == "gpu" and row["x64"]
    with np.load(ROOT / row["state_archive"]) as arrays:
        reference = arrays["analytic"]
        mapped = arrays["mapped"]
        assert reference.dtype == mapped.dtype == np.dtype("float64")
        assert arrays["moments"].dtype == np.dtype("float64")
        error = mapped - reference
        sup = np.max(np.abs(error)) / np.max(np.abs(reference))
        rms = np.linalg.norm(error) / np.linalg.norm(reference)
        np.testing.assert_allclose(
            [sup, rms],
            [row["mismatch"]["sup_relative"], row["mismatch"]["rms_relative"]],
            rtol=1e-13,
        )
        assert row["passes"] == (max(sup, rms) < 1e-2)
        assert row["nonfinite_support_moments"] == np.count_nonzero(
            ~np.isfinite(arrays["moments"])
        )
        assert row["nonzero_support_moments"] == np.count_nonzero(arrays["moments"])
        assert row["core_cell_count"] == np.count_nonzero(arrays["core"])
        np.testing.assert_allclose(
            row["lambda"] * arrays["moments"][0].sum(),
            row["analytic_plasma_current_a"],
            rtol=1e-13,
        )
        closure = np.where(
            arrays["shadow"], reference, arrays["external"] + arrays["plasma"]
        )
        np.testing.assert_allclose(closure, mapped, rtol=0, atol=1e-11)
        shifted_error = arrays["shifted_map"] - reference
        control_sup = np.max(np.abs(shifted_error)) / np.max(np.abs(reference))
        control_rms = np.linalg.norm(shifted_error) / np.linalg.norm(reference)
        np.testing.assert_allclose(
            [control_sup, control_rms],
            [
                row["negative_control"]["mismatch"]["sup_relative"],
                row["negative_control"]["mismatch"]["rms_relative"],
            ],
            rtol=1e-13,
        )
        detected = bool(control_sup > sup and control_rms > rms)
        assert row["negative_control"]["detected_in_both_norms"] == detected
    for panel in row.get("panels", []):
        path = ROOT / panel
        assert path.stat().st_size > 1000
        if path.suffix == ".png":
            assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        else:
            assert "<svg" in path.read_text()
for case in receipt["cases"]:
    for mode in receipt["clip_modes"]:
        rows = [
            r for r in receipt["rows"] if r["case"] == case and r["clip_mode"] == mode
        ]
        passing = [abs(r["requested_cells"]) for r in rows if r["passes"]]
        first = min(passing) if passing else None
        assert receipt["first_passing_rung"][case][mode] == first
        for row in rows:
            if abs(row["requested_cells"]) in {
                abs(receipt["requested_rungs"][0]),
                first,
            }:
                assert len(row["panels"]) == 2
print(
    f"PASS: {len(expected)} rows; independently recomputed errors, "
    "current normalization, composition, controls and panel coverage"
)
