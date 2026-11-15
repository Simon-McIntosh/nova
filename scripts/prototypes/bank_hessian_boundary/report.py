"""Report retained-route errors and finite-distance one-sided Hessian behaviour."""

import argparse
import hashlib
import json
import subprocess

import numpy as np

from scripts.prototypes.bank_hessian_boundary.inspect import (
    ROOT,
    STATES,
    load,
    relative,
    route_residuals,
    comparison_control,
)
from scripts.prototypes.bank_hessian_boundary.corners import coefficients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutate", action="store_true")
    args = parser.parse_args()
    polygons, archives, ref, _, _, interior = load()
    split = json.loads((ROOT / "split.json").read_text())
    records = [
        json.loads((ROOT / "nodes" / f"node-{i}.json").read_text())
        for i in ref["indices"][~interior]
    ]
    assert len(records) == int((~interior).sum())
    assert all(
        s["contained"] and s["step_m"] < s["clearance_m"]
        for r in records
        for s in r["stencils"]
    )
    values = np.array([r["values"] for r in records])
    errors = np.array([r["error"] for r in records])
    trace = np.array([r["trace"] for r in records], dtype=float)
    forcing = np.array([r["forcing"] for r in records])
    active_corner = np.array([r["corner"] and not r["full_field"] for r in records])
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(errors))
    predicted = coefficients(polygons, archives, records)
    assert np.max(predicted["relative"]) < 5e-4
    increment = np.diff(values, axis=1)
    slope = increment[:, -1]
    # A persistent increment per decade identifies a logarithmic approach;
    # a finite smooth limit instead shrinks the increment by a factor of ten.
    logarithmic = (np.abs(slope - increment[:, -2]) < 0.05 * np.abs(slope)) & (
        np.abs(slope)
        > np.maximum(100 * errors[:, -1], 1e-4 * np.maximum(1, np.abs(values[:, -1])))
    )
    output = dict(
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        states=[],
        nodes=[],
        corner_prediction=predicted,
        retained_corner_revision="3ea364d9bb6703ccc9d084938932bb9661d17a58",
        interior_count=int(interior.sum()),
    )
    for state, state_id in enumerate(STATES):
        truth = ref["plasma"][state] + ref["coils"][state]
        baseline = split["states"][state]["interior"]
        measured = route_residuals(
            archives[state], ref["indices"], interior, truth, mutate=args.mutate
        )
        row = dict(
            state=state_id,
            interior={
                **{
                    name: relative(r, truth[interior]).tolist()
                    for name, r in measured.items()
                },
                "reference_bound": baseline["reference_bound"],
            },
            boundary={},
        )
        final = values[:, -1, :, state]
        choices = dict(lower=final[:, 0], upper=final[:, 1], average=final.mean(axis=1))
        finite = ~np.any(logarithmic[:, :, state], axis=(1, 2))
        row["logarithmic_corner_nodes"] = int((~finite & active_corner).sum())
        smooth = np.array(
            [
                r["full_field"]
                or not np.any(archives[state]["moments"][r["source_cells"]])
                for r in records
            ]
        )
        limit = (
            1000 * values[:, -1, :, state]
            - 110 * values[:, -2, :, state]
            + values[:, -3, :, state]
        ) / 891
        limit_error = 4 * np.maximum(
            np.abs(
                limit - (10 * values[:, -1, :, state] - values[:, -2, :, state]) / 9
            ),
            errors[:, -1, :, state],
        )
        row["smooth_boundary"] = dict(
            count=int(smooth.sum()),
            side_difference=relative(
                limit[smooth, 1] - limit[smooth, 0], limit[smooth].mean(axis=1)
            ).tolist(),
            envelope=relative(
                limit_error[smooth].max(axis=1), limit[smooth].mean(axis=1)
            ).tolist(),
        )
        for name, key in (("bank", "actual"), ("direct", "expected")):
            route = archives[state][key][ref["indices"][~interior], 4:6]
            row["boundary"][name] = {
                label: relative(route - choice, choice).tolist()
                for label, choice in choices.items()
            }
            scale = np.linalg.norm(final.reshape(-1, 2), axis=0)
            per_node = np.stack(
                [
                    np.linalg.norm((route - choice) / scale, axis=1)
                    for choice in choices.values()
                ],
                axis=-1,
            )
            best = per_node.argmin(axis=-1)
            row["boundary"][name]["nearest_finite_distance_counts"] = {
                label: int((best == i).sum()) for i, label in enumerate(choices)
            }
            row["boundary"][name]["within_budget_count"] = int(
                (per_node.min(axis=-1) <= 1e-8).sum()
            )
            row["smooth_boundary"][name] = {
                label: relative(route[smooth] - choice[smooth], choice[smooth]).tolist()
                for label, choice in dict(
                    lower=limit[:, 0], upper=limit[:, 1], average=limit.mean(axis=1)
                ).items()
            }
        jump = trace[:, :, 1, state] - trace[:, :, 0, state]
        source_jump = forcing[:, :, 1, state] - forcing[:, :, 0, state]
        zz_jump = values[:, :, 1, state, 1] - values[:, :, 0, state, 1]
        row["active_corner_jump"] = dict(
            trace_relative=[
                float(
                    np.linalg.norm((jump - source_jump)[active_corner, j])
                    / np.linalg.norm(source_jump[active_corner, j])
                )
                for j in range(values.shape[1])
            ],
            zz_relative=[
                float(
                    np.linalg.norm((zz_jump - source_jump)[active_corner, j])
                    / np.linalg.norm(source_jump[active_corner, j])
                )
                for j in range(values.shape[1])
            ],
        )
        threshold = max(1e-8, 4 * float(baseline["reference_bound"][1]))
        row["control"] = comparison_control(
            split["states"][state]["interior_residuals"],
            measured,
            truth[interior],
            threshold,
        )
        current = route_residuals(archives[state], ref["indices"], interior, truth)
        perturbed = route_residuals(
            archives[state], ref["indices"], interior, truth, mutate=True
        )
        row["sensitivity"] = comparison_control(
            current, perturbed, truth[interior], threshold
        )
        output["states"].append(row)
        print("RESULT", json.dumps(row, default=float), flush=True)
    for i, record in enumerate(records):
        output["nodes"].append(
            dict(
                node=record["node"],
                corner=record["corner"],
                active_corner=bool(active_corner[i]),
                logarithmic=logarithmic[i].tolist(),
                increment_per_decade=slope[i].tolist(),
                record=str(ROOT / "nodes" / f"node-{record['node']}.json"),
                revision=record.get("revision") or output["retained_corner_revision"],
            )
        )
    control = json.loads((ROOT / "nodes/node--1.json").read_text())
    measured = (
        np.array(control["values"])[-1, 1, :, 1]
        - np.array(control["values"])[-1, 0, :, 1]
    )
    expected = np.array(control["forcing"])[-1, 1] - np.array(control["forcing"])[-1, 0]
    control_error = float(np.max(np.abs(measured - expected) / np.abs(expected)))
    output["uniform_edge_control"] = dict(
        measured=measured.tolist(), expected=expected.tolist(), relative=control_error
    )
    jumps = (
        np.array(control["values"])[:, 1, :, 1]
        - np.array(control["values"])[:, 0, :, 1]
    )
    limit_jump = (1000 * jumps[-1] - 110 * jumps[-2] + jumps[-3]) / 891
    output["uniform_edge_control"]["extrapolated_jump"] = limit_jump.tolist()
    output["uniform_edge_control"]["extrapolated_relative"] = float(
        np.max(np.abs(limit_jump - expected) / np.abs(expected))
    )
    output["inputs"] = json.loads((ROOT / "retained-inputs.json").read_text())
    for item in output["inputs"]:
        assert (
            hashlib.file_digest(open(item["copy"], "rb"), "sha256").hexdigest()
            == item["sha256"]
        )
    output["verdict"] = (
        "Boundary-only explanation refuted by interior nodes; corner limits "
        "must be interpreted separately from smooth-edge jumps."
    )
    (ROOT / ("mutation.json" if args.mutate else "receipt.json")).write_text(
        json.dumps(output, indent=2, default=float)
    )
    print("EDGE_CONTROL", json.dumps(output["uniform_edge_control"]), flush=True)
    assert control_error < 1e-5, "known horizontal edge density jump not recovered"
    assert all(r["sensitivity"]["detected"] for r in output["states"]), (
        "measured psi_ZZ perturbation is inert"
    )
    assert all(r["control"]["accepted"] for r in output["states"]), (
        "interior psi_ZZ measurement or perturbation refused"
    )
    print(
        "BOUNDARY_MEASURE_COMPLETE",
        len(records),
        f"boundary nodes; {int(interior.sum())} interior nodes; three states",
        flush=True,
    )


if __name__ == "__main__":
    main()
