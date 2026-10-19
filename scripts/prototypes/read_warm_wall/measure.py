"""Time the existing topology read with ordered device-to-host stage markers."""

# Precision precedes array-valued imports.
# ruff: noqa: E402
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import resource
import subprocess
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import contour_tree, topology
from nova.equilibrium.solve_request import TopologyPolicy
from tests.equilibrium import test_topology_read as oracle_fixture

assert jax.config.jax_enable_x64 is True

ROOT = Path(os.environ["NOVA_WORKTREE"]).resolve()
EVENTS: list[tuple[str, str, float]] = []


def _marker(stage: str, edge: str, value) -> None:
    leaves = [leaf for leaf in jax.tree.leaves(value) if isinstance(leaf, jax.Array)]
    token = jnp.ravel(leaves[0])[0] if leaves else jnp.asarray(0)
    jax.debug.callback(
        lambda _token: EVENTS.append((stage, edge, perf_counter())),
        token,
        ordered=True,
    )


def _timed(stage: str, function):
    def call(*args, **kwargs):
        _marker(stage, "begin", ())
        result = function(*args, **kwargs)
        _marker(stage, "end", result)
        return result

    return call


def _install_markers() -> None:
    for name, stage in (
        ("_sampled_flux", "sampled field"),
        ("_null_census", "null census"),
        ("_support_at_level", "support clip"),
        ("_connected_fragment_labels", "component labels"),
        ("_wall_events", "wall events"),
    ):
        setattr(topology, name, _timed(stage, getattr(topology, name)))
    contour_tree.build_contour_tree = _timed(
        "contour tree", contour_tree.build_contour_tree
    )


def _stage_walls(events: list[tuple[str, str, float]]) -> dict[str, float]:
    active: list[tuple[str, float, float]] = []
    exclusive = defaultdict(float)
    for stage, edge, instant in events:
        if edge == "begin":
            active.append((stage, instant, 0.0))
        else:
            opened, start, child_wall = active.pop()
            if opened != stage:
                raise RuntimeError(f"stage marker order changed: {opened} then {stage}")
            duration = instant - start
            exclusive[stage] += duration - child_wall
            if active:
                parent, parent_start, parent_children = active[-1]
                active[-1] = (parent, parent_start, parent_children + duration)
    if active:
        raise RuntimeError(f"unclosed stage markers: {active}")
    return dict(exclusive)


def measure(kind: str, cells: int) -> dict:
    _oracle, field, wall, _axis, saddle = oracle_fixture._analytic_inputs(kind)
    geometry = oracle_fixture._realised_hex_geometry(wall, cells)
    policy = TopologyPolicy(
        normal_form_radius=0.054806712567833996 if saddle is not None else 0.0,
        normal_form_pitch_floor=1.5,
    )
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    _install_markers()

    def marked_read(field, geometry, convention, policy):
        _marker("whole read", "begin", ())
        result = topology.read(field, geometry, convention, policy)
        _marker("whole read", "end", result)
        return result

    evaluate = jax.jit(marked_read)
    started = perf_counter()
    executable = evaluate.lower(field, geometry, convention, policy).compile()
    compile_wall = perf_counter() - started
    result = executable(field, geometry, convention, policy)
    jax.block_until_ready(result)
    EVENTS.clear()
    started = perf_counter()
    result = executable(field, geometry, convention, policy)
    jax.block_until_ready(result)
    warm_wall = perf_counter() - started
    events = list(EVENTS)
    stages = _stage_walls(events)
    marker_wall = next(
        end - begin
        for begin_stage, begin_edge, begin in events
        if begin_stage == "whole read" and begin_edge == "begin"
        for end_stage, end_edge, end in events
        if end_stage == "whole read" and end_edge == "end"
    )
    stages.pop("whole read", None)
    stages["other read work"] = max(0.0, marker_wall - sum(stages.values()))
    labels = np.asarray(result.fragment_component)
    return {
        "revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "cwd": str(Path.cwd().resolve()),
        "module": str(Path(topology.__file__).resolve()),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "device": str(jax.devices()[0]),
        "case": kind,
        "cells": cells,
        "realised_cells": int(geometry.centre.shape[0]),
        "valid": bool(result.valid),
        "reason": int(result.reason),
        "qualified": bool(result.qualified),
        "compile_seconds": compile_wall,
        "warm_total_seconds": warm_wall,
        "marker_total_seconds": marker_wall,
        "stage_seconds": stages,
        "marker_count": len(events),
        "component_count": int(len(np.unique(labels))),
        "host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    row = measure(args.kind, args.cells)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print("PROFILE_ROW " + json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
