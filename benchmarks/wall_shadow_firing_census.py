#!/usr/bin/env python3
"""Census private-wall anchor exclusions on persisted labeller frames."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
import uuid
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import zarr

from benchmarks.forward_labeller_throughput import (
    NEWTON_STEPS,
    SHOT_STORE,
    _centroid_pair,
    _circuit_names,
    _requested_class,
    _slices_seed,
)
from nova.equilibrium import reduced_newton
from nova.equilibrium.connectivity_boundary import wall_height_shadow_mask
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes
from scripts.labeller_batch import shard


DEFAULT_INPUT = Path(
    "/work/projects/imas_gpu/sophelio/labeller_sessions/"
    "boundary-repair-validation-20260907T1124Z"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/playable-forward-solve/wall-shadow-firing/"
    "wall-shadow-firing-census.json"
)
DEFAULT_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/playable/"
    "wall-shadow-firing-census.md"
)
EXPECTED_SHOT_COUNT = 7
BRANCH_NAMES = {
    0: "retain_previous",
    1: "qualified_height_band",
    2: "connectivity_private_fallback",
}


@dataclass(frozen=True)
class ReplayPrograms:
    """Compiled free and conditioned routes for one current polarity."""

    free: reduced_newton.ReducedProgram | None = None
    conditioned: reduced_newton.ReducedProgram | None = None


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """Atomically persist a receipt so completed shots survive interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git_revision() -> str:
    """Return the revision supplying the census implementation."""
    import subprocess

    return subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _companion_rows(path: Path) -> dict[int, dict[str, Any]]:
    """Return the replay decisions stored beside one labeller session."""
    with np.load(path, allow_pickle=False) as companion:
        return {
            int(row): {
                "conditioned": bool(conditioned),
                "time": float(frame_time),
            }
            for row, conditioned, frame_time in zip(
                companion["row"],
                companion["conditioned"],
                companion["time"],
                strict=True,
            )
        }


def _host_scalar(value: Any) -> int | float | bool:
    """Convert one device scalar to its JSON-native value."""
    item = np.asarray(jax.device_get(value)).item()
    if isinstance(item, (bool, np.bool_)):
        return bool(item)
    if isinstance(item, (int, np.integer)):
        return int(item)
    return float(item)


def _wall_report(operator: Any, state: Any, requested: Any) -> dict[str, Any]:
    """Read the settled wall mask and whether it covers the unmasked winner."""
    physical = jnp.asarray(state)[: operator.physical_node_number]
    wall_start = operator.grid.node_number
    wall_flux = physical[wall_start : operator.physical_node_number]
    signed_flux = jnp.asarray(operator.polarity, dtype=wall_flux.dtype) * wall_flux
    finite_wall = jnp.isfinite(signed_flux)
    winner_index = int(
        np.asarray(jnp.argmax(jnp.where(finite_wall, signed_flux, -jnp.inf)))
    )

    previous = jnp.zeros(operator.wall.node_number, dtype=bool)
    report = None
    iterations = 0
    for iterations in range(1, 17):
        masks, topology, _connected, _admitted = operator._fixed_design_read(  # noqa: SLF001
            physical,
            requested,
            private_wall_node_mask=previous,
        )
        reading = operator._carrier_shadow_read(physical, masks)  # noqa: SLF001
        current, report = wall_height_shadow_mask(
            operator.wall.coordinate[:, 1],
            topology.axis[1],
            topology.x_point,
            reading["xset"],
            reading["private_wall_node_mask"],
            previous,
            operator._wall_height_hysteresis,  # noqa: SLF001
            operator._x_qualification_distance,  # noqa: SLF001
            return_report=True,
        )
        current = jnp.asarray(current, dtype=bool)
        current.block_until_ready()
        if np.array_equal(np.asarray(current), np.asarray(previous)):
            break
        previous = current
    else:
        raise RuntimeError("wall shadow did not settle within 16 active-set reads")
    assert report is not None
    values = {name: _host_scalar(value) for name, value in report.items()}
    lower_code = int(values.pop("lower_branch"))
    upper_code = int(values.pop("upper_branch"))
    return {
        **values,
        "lower_eligibility_branch": BRANCH_NAMES[lower_code],
        "upper_eligibility_branch": BRANCH_NAMES[upper_code],
        "winning_anchor_node_index": winner_index,
        "winning_anchor_node_r_m": float(operator.wall.coordinate[winner_index, 0]),
        "winning_anchor_node_z_m": float(operator.wall.coordinate[winner_index, 1]),
        "winning_anchor_node_was_masked": bool(np.asarray(current)[winner_index]),
        "shadow_settling_reads": iterations,
    }


def _replay_shot(
    prepared: shard.PreparedLabeller,
    manifest_path: Path,
    programs: ReplayPrograms,
) -> tuple[ReplayPrograms, list[dict[str, Any]], dict[str, Any]]:
    """Replay one relabelled shot and census each written limited frame."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"input manifest is not complete: {manifest_path}")
    shot = int(manifest["shot"])
    companion = _companion_rows(manifest_path.with_name(f"{shot}.npz"))
    group = zarr.open_group(str(SHOT_STORE / f"{shot}.zarr"), mode="r")["efm"]
    full_r = np.asarray(group["gridr"], dtype=np.float64)
    full_z = np.asarray(group["gridz"], dtype=np.float64)
    circuit_names = _circuit_names(prepared.policy_evidence)
    state = None
    frames: list[dict[str, Any]] = []
    replayed = 0
    for record in manifest["slices"]:
        row = int(record["row"])
        if not record.get("written"):
            state = None
            continue
        replayed += 1
        inputs = shard._slice_inputs(group, row)  # noqa: SLF001
        if inputs is None:
            raise ValueError(
                f"written row has no reconstruction: shot={shot} row={row}"
            )
        seed = _slices_seed(group, row, full_r, full_z)
        initial = jnp.asarray(seed) if state is None else jnp.asarray(state)
        requested_value = _requested_class(group, row)
        requested = jnp.asarray(requested_value, dtype=jnp.int8)
        target_current = abs(inputs["reference_plasma_current"])
        current = jnp.asarray(inputs["current"])
        free = reduced_newton.solve_reduced_newton(
            prepared.profile.operator,
            initial,
            requested_class=requested,
            target_current=target_current,
            prescribed_current=current,
            tolerance=shard.FIXED_POINT_CRITERION,
            newton_steps=NEWTON_STEPS,
            program=programs.free,
            stream=False,
        )
        free_program = free.program
        conditioned_program = programs.conditioned
        selected = free
        decision = companion.get(row)
        if decision is None:
            raise ValueError(f"companion omits written row: shot={shot} row={row}")
        if decision["conditioned"]:
            pair, _selection = _centroid_pair(
                prepared.profile,
                initial,
                target=inputs["target_centroid_z"],
                unknown=None,
                target_current=target_current,
                requested=requested,
                names=circuit_names,
            )
            selected = reduced_newton.solve_constrained_reduced_newton(
                prepared.profile,
                initial,
                constraint_pairs=(pair,),
                requested_class=requested,
                target_current=target_current,
                prescribed_current=current,
                tolerance=shard.FIXED_POINT_CRITERION,
                newton_steps=NEWTON_STEPS,
                program=programs.conditioned,
                stream=False,
            )
            conditioned_program = selected.program
        programs = ReplayPrograms(free_program, conditioned_program)
        if requested_value == int(TopologyClass.LIMITED):
            frames.append(
                {
                    "shot": shot,
                    "row": row,
                    "time_s": float(inputs["time"]),
                    "conditioned": bool(decision["conditioned"]),
                    "replay_terminal_residual": float(selected.terminal_residual),
                    "source_terminal_residual": record.get("terminal_residual"),
                    **_wall_report(
                        prepared.profile.operator, selected.state, requested
                    ),
                }
            )
        state = None if record.get("geometry_masked") else selected.state
    summary = {
        "shot": shot,
        "written_frames_replayed": replayed,
        "limited_frames": len(frames),
        "masked_wall_nodes": sum(
            int(frame["masked_wall_node_count"]) for frame in frames
        ),
        "winning_anchor_masked_frames": sum(
            int(frame["winning_anchor_node_was_masked"]) for frame in frames
        ),
    }
    return programs, frames, summary


def _write_report(payload: dict[str, Any], path: Path) -> None:
    """Write the concise human verdict beside the machine-readable census."""
    summary = payload["summary"]
    lines = [
        "# Private-wall shadow firing census",
        "",
        (
            f"The seven-shot replay found **{summary['masked_wall_nodes']} masked "
            "wall nodes over all limited frames**, against the stated bound of "
            f"**zero**. This supports **{summary['verdict']}**."
        ),
        "",
        (
            f"Coverage: {summary['limited_frames']} limited frames across "
            f"{summary['shot_count']} shots; every row records its wall-node "
            "denominator, qualified-saddle count, both eligibility branches, and "
            "whether the unmasked winning anchor node was masked."
        ),
        "",
        "| Shot | Limited frames | Masked wall nodes | Winning anchors masked |",
        "|---:|---:|---:|---:|",
    ]
    for shot in payload["shots"]:
        lines.append(
            f"| {shot['shot']} | {shot['limited_frames']} | "
            f"{shot['masked_wall_nodes']} | {shot['winning_anchor_masked_frames']} |"
        )
    lines.extend(
        [
            "",
            f"Machine-readable table: `{payload['output']}`.",
            f"Input sessions: `{payload['input_root']}`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(input_root: Path, output: Path, report: Path) -> dict[str, Any]:
    """Run the complete seven-shot census and persist each completed shot."""
    configure_dtypes()
    manifests = sorted(input_root.glob("*.manifest.json"))
    if len(manifests) != EXPECTED_SHOT_COUNT:
        raise ValueError(
            f"expected {EXPECTED_SHOT_COUNT} shot manifests, found {len(manifests)}"
        )
    started = time.perf_counter()
    prepared = shard.prepare_labeller()
    payload: dict[str, Any] = {
        "schema": "nova-wall-shadow-firing-census",
        "status": "working",
        "nova_revision": _git_revision(),
        "input_root": str(input_root.resolve()),
        "output": str(output.resolve()),
        "report": str(report.resolve()),
        "compilation_cache": prepared.cache_directory,
        "eligibility_branch_codes": BRANCH_NAMES,
        "shots": [],
        "frames": [],
    }
    _write_json(payload, output)
    programs_by_polarity: dict[int, ReplayPrograms] = {}
    prepared_by_polarity: dict[int, shard.PreparedLabeller] = {}
    for manifest_path in manifests:
        shot = int(json.loads(manifest_path.read_text(encoding="utf-8"))["shot"])
        polarity = shard._shot_polarity(shot)  # noqa: SLF001
        shot_prepared = prepared_by_polarity.setdefault(
            polarity,
            shard._prepared_with_polarity(prepared, polarity),  # noqa: SLF001
        )
        programs, frames, summary = _replay_shot(
            shot_prepared,
            manifest_path,
            programs_by_polarity.get(polarity, ReplayPrograms()),
        )
        programs_by_polarity[polarity] = programs
        payload["shots"].append(summary)
        payload["frames"].extend(frames)
        _write_json(payload, output)
        print("CENSUS_SHOT " + json.dumps(summary, sort_keys=True), flush=True)
    masked = sum(int(frame["masked_wall_node_count"]) for frame in payload["frames"])
    winner_masked = sum(
        int(frame["winning_anchor_node_was_masked"]) for frame in payload["frames"]
    )
    if masked == 0:
        verdict = "the predicate never classifies wall nodes on these limited frames"
    elif winner_masked == 0:
        verdict = (
            "the mask fires away from the winning anchor, so selection is downstream "
            "of where the mask applies"
        )
    else:
        verdict = (
            "neither predeclared outcome: the mask fires and covers at least one "
            "winning anchor"
        )
    payload["summary"] = {
        "shot_count": len(payload["shots"]),
        "limited_frames": len(payload["frames"]),
        "masked_wall_nodes": masked,
        "masked_wall_node_bound": 0,
        "winning_anchor_masked_frames": winner_masked,
        "verdict": verdict,
        "wall_seconds": time.perf_counter() - started,
    }
    payload["status"] = "complete"
    _write_json(payload, output)
    _write_report(payload, report)
    print("CENSUS_COMPLETE " + json.dumps(payload["summary"], sort_keys=True))
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run(args.input_root, args.output, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
