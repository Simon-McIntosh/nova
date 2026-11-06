"""Assign exclusive kernel ownership from compiler source frames, not timings."""

import ast
from functools import lru_cache
from pathlib import Path


GROUPS = (
    "read census",
    "read cut-cell membership",
    "read normal form + set",
    "read curvature stencil",
    "read wall events",
    "read candidate loop",
    "read fragment selection",
    "read continuous geometry",
    "read qualification",
    "step_residual",
    "step_jacobian",
    "equilibrate + lu_factor",
    "lu_solve + update",
    "damped trial selection",
    "solve_batch (other)",
    "unbound compiler/library",
)


@lru_cache(None)
def source_lines():
    root = Path(__file__).resolve().parents[3]
    return (root / "nova/equilibrium/topology.py").read_text().splitlines()


@lru_cache(None)
def normal_set_lines():
    tree = ast.parse("\n".join(source_lines()))
    record = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TopologyRead"
    )
    fields = [node.target.id for node in record.body if isinstance(node, ast.AnnAssign)]
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_read_at_nulls"
    )
    returned = next(node for node in function.body if isinstance(node, ast.Return))
    argument = returned.value.args[fields.index("normal_form_cells")]
    return range(argument.lineno, argument.end_lineno + 1)


def group(item):
    frames = item["stack"]
    names = {frame["function"].split(".<locals>")[0] for frame in frames}
    if names & {
        "step_jacobian",
        "_normal_geometry_booking_tangent",
        "_normal_form_rate",
        "_curvature_rates",
        "_local_booking_sensitivities",
    }:
        return "step_jacobian"
    if "_equilibrate" in names or "lu_factor" in item["op_name"]:
        return "equilibrate + lu_factor"
    if "step_residual" in names:
        return "step_residual"
    if "newton_step" in names:
        return "lu_solve + update"
    if "stationary_stencil" in names:
        return "read census"
    if "_curvature_derivatives" in names:
        return "read curvature stencil"
    if "saddle_normal_form" in names:
        return "read normal form + set"
    if any(
        f["function"] == "_read_at_nulls" and f["line"] in normal_set_lines()
        for f in frames
    ):
        return "read normal form + set"
    if "_wall_events" in names:
        return "read wall events"
    if "_connected_fragment_labels" in names:
        return "read fragment selection"
    if any(".inspect" in frame["function"] for frame in frames):
        return "read candidate loop"
    if "_support_at_level" in names:
        source = source_lines()
        caller = next(
            (
                frame
                for frame in reversed(frames)
                if frame["function"] == "_read_at_nulls"
            ),
            None,
        )
        if caller and "terminal =" in source[caller["line"] - 1]:
            return "read continuous geometry"
        return "read cut-cell membership"
    if names & {"nodal_jets", "field_from_jets", "sampled_coefficient"}:
        return "read continuous geometry"
    if names & {"read_at_stationary_points", "_read_at_nulls"}:
        caller_lines = [
            source_lines()[f["line"] - 1]
            for f in frames
            if f["function"] == "_read_at_nulls"
        ]
        if any("fori_loop" in line for line in caller_lines):
            return "read candidate loop"
        if any(
            any(word in line for word in ("wall_cell", "on_core", "wall_score"))
            for line in caller_lines
        ):
            return "read fragment selection"
        return "read qualification"
    if "_damped_scale" in names:
        return "damped trial selection"
    if "solve_batch" in names:
        return "solve_batch (other)"
    return "unbound compiler/library"
