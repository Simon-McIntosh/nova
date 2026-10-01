"""Project the first accepted native Newton displacement on saved seed spectra."""

import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from benchmarks.plasma_cell_first_step_directions import clean, spectrum, write

root = Path("docs/figures/plasma-cell-read-fidelity/first-step")
print(
    "revision="
    + subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    + " tree="
    + str(Path.cwd())
    + " command=python derive-first-promotion.py",
    flush=True,
)
deadline = time.monotonic() + 300
while not (root / "jacobians.npz").exists():
    if time.monotonic() >= deadline:
        raise TimeoutError("seed Jacobians did not arrive in the allocated window")
    time.sleep(2)
time.sleep(1)
with np.load(root / "jacobians.npz") as archive:
    normalized = archive["normalized"]
    fixed = archive["fixed_lambda"]
    active = archive["active"]
    gradient = archive["current_gradient"]
events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
for row in events:
    if row["kind"] == "promotion":
        with np.load(root / f"event-{row['event']:05d}.npz") as archive:
            if bool(archive["value_5"]):
                origin = archive["value_0"]
                accepted = archive["value_1"]
                before, predicted, actual = [
                    float(archive[f"value_{i}"]) for i in (2, 3, 4)
                ]
                distrusted = bool(archive["value_6"])
                break
else:
    raise AssertionError("no accepted native Newton promotion")
step = (accepted - origin)[active]
shrink = -gradient / np.linalg.norm(gradient)
result = {
    "job": os.environ["SLURM_JOB_ID"],
    "event": row["event"],
    "norm_wb": np.linalg.norm(step),
    "support_shrink_projection_wb": np.dot(step, shrink),
    "support_shrink_cosine": np.dot(step, shrink) / np.linalg.norm(step),
    "predicted_merit_decrease": before - predicted,
    "actual_merit_decrease": before - actual,
    "model_distrusted": distrusted,
    "spectra": {},
}
for name, matrix in (
    ("normalized_map", normalized),
    ("fixed_lambda_map", fixed),
    ("normalized_newton_residual", np.eye(len(active)) - normalized),
    ("fixed_lambda_newton_residual", np.eye(len(active)) - fixed),
):
    result["spectra"][name] = spectrum(matrix, step, shrink, active, name)
write(root / "first-promotion-directions.json", result)
print(
    json.dumps(
        clean({key: value for key, value in result.items() if key != "spectra"})
    ),
    flush=True,
)
print(
    "PASS: first accepted Newton displacement projected on all four saved seed spectra",
    flush=True,
)
