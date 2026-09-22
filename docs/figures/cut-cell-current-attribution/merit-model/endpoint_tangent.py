"""Retain both finite-difference endpoint states of the local tangent check.

The local merit model compares a forward-mode tangent with a central finite
difference at a fixed increment.  That difference alone cannot say whether the
tangent or the increment is at fault, because a support or classification
branch change inside the increment produces a discrepancy of order one while a
wrong local coefficient leaves a discrepancy that persists as the increment
shrinks.  This driver re-runs the same instrument on one terminal state and
keeps the two endpoint states, their classifications, their nonzero-current
supports and their current moments for every measured increment, so the two
roots are separable from the receipt alone.

Run one process per state; every increment and direction is measured inside
that process.
"""
# ruff: noqa: E501 -- Receipt path literals and printed receipt lines retain their wording.

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure as instrument  # noqa: E402

from nova.equilibrium import forward_operator as forward  # noqa: E402

configure_dtypes = instrument.configure_dtypes
jax = instrument.jax
jnp = instrument.jnp

INCREMENTS = (1e-6, 1e-7, 1e-8)
CONTROL_INCREMENT = 1e-4
NAMED_CELLS = 12


def support_class(included, area, full_area, tolerance=1e-12):
    """Classify each cell: wholly outside, cut, or wholly inside."""
    included = np.asarray(included, dtype=bool)
    area = np.asarray(area, dtype=np.float64)
    full_area = np.asarray(full_area, dtype=np.float64)
    whole = included & (np.abs(area - full_area) <= tolerance * np.abs(full_area))
    return {
        "outside": np.flatnonzero(~included),
        "cut": np.flatnonzero(included & ~whole),
        "inside": np.flatnonzero(whole),
    }


def class_of(cells, classification):
    lookup = {}
    for name in ("cut", "inside", "outside"):
        for index in classification[name]:
            lookup[int(index)] = name
    return [lookup.get(int(index), "unknown") for index in cells]


def leading(values, count):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-np.abs(values))[:count]
    return [[int(i), float(values[i])] for i in order]


def index_list(cells):
    return [int(v) for v in cells]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    assert jax.config.jax_enable_x64 and jax.default_backend() == "cpu"
    instrument.fixture._cache_lock = lambda store: nullcontext(0.0)

    def refuse_store(*_args, **_kwargs):
        raise RuntimeError("Fixture cache miss requires an out-of-scope cache write")

    instrument.fixture.ZarrStore.store = refuse_store
    (
        machine,
        operator,
        _profile,
        _exact,
        _seed,
        target,
        _seed_receipt,
        _coordinates,
        analytic,
    ) = instrument._build("weak-rotation-reactor-static", -args.cells)
    root = Path(__file__).resolve().parents[4]
    compact_path = (
        root
        / "docs/figures/cut-cell-current-attribution/continuous-support/rows-titan-adjoint/record"
        / f"cells-{args.cells}/receipt.json"
    )
    compact = json.loads(compact_path.read_text())
    receipt_path = Path(compact["raw_receipt"])
    assert (
        hashlib.sha256(receipt_path.read_bytes()).hexdigest() == compact["raw_sha256"]
    )
    receipt = json.loads(receipt_path.read_text())
    state = jnp.asarray(receipt["trips"][-1]["flux"], dtype=jnp.float64)
    assert instrument.digest(state) == compact["terminal"]["state_digest"]
    np.testing.assert_array_equal(analytic, receipt["analytic_state"])
    external = operator.external()
    target = jnp.asarray(target, dtype=jnp.float64)
    program = jax.jit(operator.traced_flux_map(target_current=target))

    def mapped(value):
        return program(value, external, operator, target)

    @jax.jit
    def map_jvp(value, vector):
        return jax.jvp(mapped, (value,), (vector,))[1]

    @jax.jit
    def moment_program(value, op):
        raw = jnp.stack(op.cell_current_moments(value))
        amplitude = op.current_normalisation_amplitude(target, raw[0].sum())
        return raw, raw * amplitude

    @jax.jit
    def moment_jvp(value, vector, op):
        return jax.jvp(lambda v: moment_program(v, op), (value,), (vector,))[1]

    @jax.jit
    def diagnostic_program(value, op):
        masks, topology, _sample, support = op._support_partition(value)
        return {
            "confined": masks.confined_profile,
            "open": masks.open_field_line,
            "labels": masks.label,
            "included": support.included,
            "area": support.area,
            "full_area": support.full_area,
            "branch_area": support.branch_area,
            "shadow": op.residual_shadow_mask(value),
            "axis": topology.axis,
            "x_point": topology.x_point,
        }

    @jax.jit
    def newton_program(value, op, ext, target_value):
        image, tangent = jax.linearize(
            lambda v: program(v, ext, op, target_value), value
        )
        return instrument.fixed._qualified_krylov_step(
            lambda v: v - tangent(v),
            image - value,
            instrument.fixed._relative_residual(image, value),
            gmres_iterations=30,
            condition_ratio_limit=math.e,
            preceding_condition_baseline=jnp.asarray(jnp.nan),
        )

    def snapshot(value):
        raw, scaled = moment_program(value, operator)
        diag = diagnostic_program(value, operator)
        classification = support_class(
            diag["included"], diag["area"], diag["full_area"]
        )
        return {
            "raw": np.asarray(raw),
            "scaled": np.asarray(scaled),
            "confined": np.asarray(diag["confined"]),
            "open": np.asarray(diag["open"]),
            "labels": np.asarray(diag["labels"]),
            "shadow": np.asarray(diag["shadow"]),
            "class": classification,
            "included": np.asarray(diag["included"]),
            "area": np.asarray(diag["area"]),
            "full_area": np.asarray(diag["full_area"]),
        }

    image = mapped(state)
    base = snapshot(state)
    incumbent = instrument.norm_terms(image, state)
    qualified = newton_program(state, operator, external, target)
    jax.block_until_ready(qualified)
    directions = {
        "analytic": jnp.asarray(analytic) - state,
        "newton": qualified.step,
        "map_defect": image - state,
    }
    data = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "worktree": str(root),
        "backend": jax.default_backend(),
        "devices": str(jax.devices()),
        "support_clip_mode": forward.support_clip_mode(),
        "realised_cells": len(machine.node),
        "requested_cells": args.cells,
        "input_receipt": str(receipt_path),
        "input_sha256": compact["raw_sha256"],
        "state_digest": instrument.digest(state),
        "state": state,
        "incumbent": incumbent,
        "increments": list(INCREMENTS),
        "control_increment": CONTROL_INCREMENT,
        "newton_receipt": {k: getattr(qualified, k) for k in qualified._fields},
        "base": {
            "nonzero_current_ids": index_list(np.flatnonzero(base["raw"][0] != 0)),
            "raw_current_total_a": float(np.sum(base["raw"][0])),
            "confined_count": int(np.count_nonzero(base["confined"])),
            "shadow_ids": index_list(np.flatnonzero(base["shadow"])),
            "class_counts": {k: len(v) for k, v in base["class"].items()},
        },
        "directions": {},
        "completed": False,
    }
    path = output / f"endpoint-tangent-cells-{len(machine.node)}.json"
    instrument.write(path, data)
    print(
        f"BASE merit={incumbent['merit']:.12g} nonzero={len(np.flatnonzero(base['raw'][0] != 0))} "
        f"classes={data['base']['class_counts']} clip={data['support_clip_mode']}",
        flush=True,
    )
    for name, direction in directions.items():
        derivative = map_jvp(state, direction)
        raw_dot, scaled_dot = moment_jvp(state, direction, operator)
        group = {
            "map_jvp_norm": float(jnp.linalg.norm(derivative)),
            "raw_current_jvp": raw_dot[0],
            "increments": {},
        }
        for increment in (
            (*INCREMENTS, CONTROL_INCREMENT) if name == "analytic" else INCREMENTS
        ):
            plus_value, minus_value = (
                state + increment * direction,
                (state - increment * direction),
            )
            plus_image, minus_image = mapped(plus_value), mapped(minus_value)
            plus, minus = snapshot(plus_value), snapshot(minus_value)
            fd = (plus_image - minus_image) / (2 * increment)
            discrepancy = fd - derivative
            current_fd = (plus["raw"][0] - minus["raw"][0]) / (2 * increment)
            current_gap = current_fd - raw_dot[0]
            nonzero_plus = np.flatnonzero(plus["raw"][0] != 0)
            nonzero_minus = np.flatnonzero(minus["raw"][0] != 0)
            support_plus_only = np.setdiff1d(nonzero_plus, nonzero_minus)
            support_minus_only = np.setdiff1d(nonzero_minus, nonzero_plus)
            confined_flip = np.flatnonzero(plus["confined"] != minus["confined"])
            open_flip = np.flatnonzero(plus["open"] != minus["open"])
            label_flip = np.flatnonzero(plus["labels"] != minus["labels"])
            shadow_flip = np.flatnonzero(plus["shadow"] != minus["shadow"])
            named = np.argsort(-np.abs(current_gap))[:NAMED_CELLS]
            named_rows = []
            for index in named:
                cell = int(index)
                named_rows.append(
                    {
                        "cell": cell,
                        "class_base": class_of([cell], base["class"])[0],
                        "class_plus": class_of([cell], plus["class"])[0],
                        "class_minus": class_of([cell], minus["class"])[0],
                        "area_base_m2": float(base["area"][cell]),
                        "full_area_m2": float(base["full_area"][cell]),
                        "area_plus_m2": float(plus["area"][cell]),
                        "area_minus_m2": float(minus["area"][cell]),
                        "raw_current_base_a": float(base["raw"][0][cell]),
                        "raw_current_plus_a": float(plus["raw"][0][cell]),
                        "raw_current_minus_a": float(minus["raw"][0][cell]),
                        "raw_current_fd_a": float(current_fd[cell]),
                        "raw_current_jvp_a": float(raw_dot[0][cell]),
                        "raw_current_gap_a": float(current_gap[cell]),
                    }
                )
            group["increments"][f"{increment:g}"] = {
                "increment": increment,
                "plus_state": plus_value,
                "minus_state": minus_value,
                "plus_state_digest": instrument.digest(plus_value),
                "minus_state_digest": instrument.digest(minus_value),
                "fd_norm": float(jnp.linalg.norm(fd)),
                "jvp_norm": float(jnp.linalg.norm(derivative)),
                "relative_error_over_fd": float(
                    jnp.linalg.norm(discrepancy)
                    / jnp.maximum(jnp.linalg.norm(fd), 1e-30)
                ),
                "relative_error_over_jvp": float(
                    jnp.linalg.norm(discrepancy)
                    / jnp.maximum(jnp.linalg.norm(derivative), 1e-30)
                ),
                "map_discrepancy_max_wb": instrument.maximum(discrepancy),
                "leading_map_discrepancy_nodes": leading(discrepancy, NAMED_CELLS),
                "fd_over_jvp_norm": float(
                    jnp.linalg.norm(fd)
                    / jnp.maximum(jnp.linalg.norm(derivative), 1e-30)
                ),
                "confined_flip_count": int(confined_flip.size),
                "confined_flip_ids": index_list(confined_flip),
                "open_flip_count": int(open_flip.size),
                "open_flip_ids": index_list(open_flip),
                "label_flip_count": int(label_flip.size),
                "label_flip_ids": index_list(label_flip),
                "shadow_flip_count": int(shadow_flip.size),
                "shadow_flip_ids": index_list(shadow_flip),
                "support_plus_only_count": int(support_plus_only.size),
                "support_plus_only_ids": index_list(support_plus_only),
                "support_minus_only_count": int(support_minus_only.size),
                "support_minus_only_ids": index_list(support_minus_only),
                "support_flip_vs_base_ids": index_list(
                    np.setdiff1d(
                        np.union1d(nonzero_plus, nonzero_minus),
                        np.flatnonzero(base["raw"][0] != 0),
                    )
                ),
                "current_fd_vs_jvp_max_a": instrument.maximum(current_gap),
                "current_fd_vs_jvp_norm_ratio": float(
                    jnp.linalg.norm(current_gap)
                    / jnp.maximum(jnp.linalg.norm(raw_dot[0]), 1e-30)
                ),
                "named_cells": named_rows,
                "class_counts_plus": {k: len(v) for k, v in plus["class"].items()},
                "class_counts_minus": {k: len(v) for k, v in minus["class"].items()},
                "raw_current_total_plus_a": float(np.sum(plus["raw"][0])),
                "raw_current_total_minus_a": float(np.sum(minus["raw"][0])),
            }
            instrument.write(path, data)
            print(
                f"ROW {name} h={increment:g} fd={float(jnp.linalg.norm(fd)):.6g} "
                f"jvp={float(jnp.linalg.norm(derivative)):.6g} "
                f"rel={float(jnp.linalg.norm(discrepancy) / jnp.maximum(jnp.linalg.norm(fd), 1e-30)):.8g} "
                f"confined_flips={int(confined_flip.size)} open_flips={int(open_flip.size)} "
                f"support_pm={int(support_plus_only.size)}/{int(support_minus_only.size)} "
                f"nominal_error={instrument.maximum(current_gap):.6g}",
                flush=True,
            )
        data["directions"][name] = group
        instrument.write(path, data)

    # Reproduction of the recorded verdict with the same denominator as the
    # recorded one, then the detector's positive control: a seeded single-bit
    # flip and a moved state must both be seen.
    recorded = {
        "analytic": 0.999565182706,
        "newton": 0.995715816449,
        "map_defect": 1.00009238076,
    }
    data["reproduction"] = {}
    for name, group in data["directions"].items():
        measured = group["increments"]["1e-06"]["relative_error_over_fd"]
        data["reproduction"][name] = {
            "measured": measured,
            "recorded": recorded[name],
            "absolute_delta": abs(measured - recorded[name]),
        }
    changed = np.array(np.asarray(base["confined"]), copy=True)
    changed[0] = ~changed[0]
    data["positive_controls"] = {
        "classification_detector_single_flip": int(
            np.count_nonzero(changed != base["confined"])
        ),
        "support_detector_nonzero_cells": int(np.count_nonzero(base["raw"][0] != 0)),
        "control_increment_flip_count": data["directions"]["analytic"]["increments"][
            f"{CONTROL_INCREMENT:g}"
        ]["label_flip_count"],
        "control_increment_support_change": data["directions"]["analytic"][
            "increments"
        ][f"{CONTROL_INCREMENT:g}"]["support_plus_only_count"]
        + data["directions"]["analytic"]["increments"][f"{CONTROL_INCREMENT:g}"][
            "support_minus_only_count"
        ],
    }
    data["wall_seconds"] = perf_counter() - start
    data["completed"] = True
    instrument.write(path, data)
    print(
        f"COMPLETED cells={len(machine.node)} seconds={perf_counter() - start:.2f} "
        f"controls={json.dumps(data['positive_controls'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
