"""Measure booked current against an independently integrated analytic support."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTROL_PART = (
    "docs/figures/uniform-cell-clip-and-coupling/exact-participation/"
    "discriminator/parts/control/weak-rotation-reactor-static-production-route-reduced.json"
)
ARCHIVE_PATHS = ("nova", "tests", "benchmarks", "scripts")
PINNED_REVISION = "247e5aa4a"
RESPONSIBLE_REVISION = "38b441dad1dea2b10b684fa612b8802ef0973b86"
TRIANGLE_CONVERGENCE_RELATIVE_TOLERANCE = 1.0e-8


def _git_text(revision: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{revision}:{path}"], text=True
    )


def _archive_tree(revision: str, scratch: Path) -> Path:
    destination = scratch / revision
    destination.mkdir()
    archive = subprocess.run(
        ["git", "-C", str(ROOT), "archive", "--format=tar", revision, *ARCHIVE_PATHS],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(destination, filter="data")
    control = destination / CONTROL_PART
    control.parent.mkdir(parents=True, exist_ok=True)
    control.write_text(_git_text(revision, CONTROL_PART), encoding="utf-8")
    return destination


_TRIANGLE_RULE = np.asarray(
    (
        (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
    ),
    dtype=np.float64,
)


def _triangle_area(triangle: np.ndarray) -> float:
    first, second, third = triangle
    first_edge = second - first
    second_edge = third - first
    cross = first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
    return abs(float(cross)) / 2.0


def _refined_triangles(triangle: np.ndarray, refinement: int) -> np.ndarray:
    """Uniformly split one triangle into a conforming triangular lattice."""

    count = 2**refinement
    first, second, third = triangle

    def point(row: int, column: int) -> np.ndarray:
        return first + (row * (second - first) + column * (third - first)) / count

    triangles: list[np.ndarray] = []
    for row in range(count):
        for column in range(count - row):
            lower = point(row, column)
            right = point(row + 1, column)
            upper = point(row, column + 1)
            triangles.append(np.stack((lower, right, upper)))
            if column < count - row - 1:
                diagonal = point(row + 1, column + 1)
                triangles.append(np.stack((right, diagonal, upper)))
    return np.asarray(triangles, dtype=np.float64)


def _triangle_reference(
    polygon: np.ndarray, density_sampler: Any, cell: int
) -> tuple[float, list[dict[str, float]]]:
    """Refine a symmetric triangle rule until adjacent totals agree."""

    fan = [
        np.stack((polygon[0], polygon[index], polygon[index + 1]))
        for index in range(1, len(polygon) - 1)
    ]
    ladder: list[dict[str, float]] = []
    previous = None
    for refinement in range(8):
        triangles = np.concatenate(
            [_refined_triangles(triangle, refinement) for triangle in fan]
        )
        points = np.einsum("qi,tij->tqj", _TRIANGLE_RULE, triangles).reshape(-1, 2)
        density = np.asarray(density_sampler(points, np.int32(cell)), dtype=np.float64)
        weighted = density.reshape(len(triangles), len(_TRIANGLE_RULE)).mean(axis=1)
        areas = np.asarray([_triangle_area(item) for item in triangles])
        total = float(np.sum(weighted * areas))
        relative = (
            None if previous is None else abs(total - previous) / max(abs(total), 1.0)
        )
        ladder.append(
            {"level": refinement, "current_a": total, "relative_change": relative}
        )
        if relative is not None and relative <= TRIANGLE_CONVERGENCE_RELATIVE_TOLERANCE:
            return total, ladder
        previous = total
    raise RuntimeError(f"triangle reference did not converge for cell {cell}: {ladder}")


def _refinement_error_bound(ladder: list[dict[str, float | None]]) -> float | None:
    """Return the observed final refinement change for one polygon integral."""

    if len(ladder) < 2:
        return None
    return abs(float(ladder[-1]["current_a"]) - float(ladder[-2]["current_a"]))


def _install_absent_saddle_bridge(forward_operator: Any) -> None:
    target = forward_operator.ForwardFluxOperator._profile_support

    def bridged(self, masks, topology, physical, sample_psi_norm, **kwargs):
        if not hasattr(topology, "x_point"):
            topology = SimpleNamespace(
                **vars(topology),
                x_point=np.full(2, np.nan),
                x_point_flux=np.asarray(0.0),
            )
        return target(self, masks, topology, physical, sample_psi_norm, **kwargs)

    forward_operator.ForwardFluxOperator._profile_support = bridged


def _profile_density_sampler(jax: Any, field: Any, profile: Any):
    """Compile terminal-profile samples once for the adaptive reference."""

    @jax.jit
    def sample(points, cell):
        psi_norm, _radial, _vertical = field.sample(points[None, ...], cell[None])
        return profile.current_density(points[:, 0], psi_norm[0])

    return sample


def _full_state_partition(operator: Any, state: np.ndarray, jnp: Any) -> dict[str, Any]:
    """Read every support operand from the persisted, unsliced terminal state."""

    physical = jnp.asarray(state)
    base_masks, topology, _connected, _admitted = operator._fixed_design_read(physical)
    sample_flux = operator.sample_node_flux(physical)
    sample_psi_norm = (sample_flux - topology.axis_flux) / topology.flux_span
    profile_support = operator._profile_support(
        base_masks, topology, physical, sample_psi_norm
    )
    return {
        "base_masks": base_masks,
        "topology": topology,
        "sample_psi_norm": sample_psi_norm,
        "profile_support": profile_support,
        "moment_masks": operator._moment_support_masks(base_masks, profile_support),
    }


@contextmanager
def _operator_clip_mode(forward_operator: Any, operator: Any, mode: str):
    """Yield an operator configured by the interface present in its revision."""

    if hasattr(operator, "with_clip_mode"):
        selected = operator.with_clip_mode(mode)
        yield selected, "with_clip_mode"
        return
    previous = forward_operator.support_clip_mode()
    forward_operator.set_support_clip_mode(mode)
    try:
        yield operator, "set_support_clip_mode"
    finally:
        forward_operator.set_support_clip_mode(previous)


def _measure(
    revision: str,
    scratch: Path,
    component: str,
    reference_cells: set[int] | None,
    generating_commit: str,
) -> dict[str, Any]:
    tree = _archive_tree(revision, scratch)
    previous = list(sys.path)
    roots = ("nova", "benchmarks", "scripts")
    prefixes = tuple(f"{root}." for root in roots)
    modules = {
        name: module
        for name, module in sys.modules.items()
        if name in roots or name.startswith(prefixes)
    }
    try:
        for name in list(modules):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(tree))
        import jax
        import jax.numpy as jnp
        import nova.equilibrium.forward_operator as forward_operator
        from benchmarks import unit_amplitude_current_census as census
        from nova.jax.config import configure_dtypes
        from nova.equilibrium.source import _FluxSelectedProfile

        configure_dtypes()
        _install_absent_saddle_bridge(forward_operator)
        control = census._read_control_row()
        context = census._build_context(control)
        state = np.asarray(control["terminal_flux_wb"], dtype=np.float64)
        operator = context["operator"]
        legacy_mode = (
            None
            if hasattr(operator, "with_clip_mode")
            else forward_operator.support_clip_mode()
        )
        mode_strategy = (
            "with_clip_mode"
            if hasattr(operator, "with_clip_mode")
            else "set_support_clip_mode"
        )
        booked = None
        exact_cell_current = None
        if component != "reference":
            try:
                mode_current = {}
                for mode in ("chord", "exact"):
                    with _operator_clip_mode(forward_operator, operator, mode) as (
                        selected_operator,
                        _selection_strategy,
                    ):
                        mode_current[mode] = np.asarray(
                            selected_operator.cell_current_moments(
                                jnp.asarray(state)
                            ).cell_current,
                            dtype=np.float64,
                        )
                booked = {
                    mode: float(np.sum(current))
                    for mode, current in mode_current.items()
                }
                exact_cell_current = mode_current["exact"].tolist()
            finally:
                if legacy_mode is not None:
                    forward_operator.set_support_clip_mode(legacy_mode)
        reference = None
        error_bound = None
        error_bound_reason = None
        reference_cell_rows = None
        if component != "booking":
            with _operator_clip_mode(forward_operator, operator, "exact") as (
                exact_operator,
                _reference_mode_strategy,
            ):
                exact_probe = _full_state_partition(exact_operator, state, jnp)
                exact_field = forward_operator.flux_field_polynomial(
                    exact_operator._support_moment_stencils,
                    exact_probe["moment_masks"].psi_norm,
                    exact_probe["sample_psi_norm"],
                )
                profile = _FluxSelectedProfile(
                    exact_operator.source.core, exact_operator.source.common_sol
                )
                density_sampler = _profile_density_sampler(jax, exact_field, profile)
                support_vertices = np.asarray(
                    exact_probe["profile_support"].support_vertices
                )
                support_count = np.asarray(exact_probe["profile_support"].vertex_count)
                selected = np.asarray(exact_field.active) & np.asarray(
                    exact_probe["moment_masks"].profile_participation
                )
                cells = (
                    range(len(support_vertices))
                    if reference_cells is None
                    else sorted(reference_cells)
                )
                reference_cell_rows = []
                for cell in cells:
                    if selected[cell] and support_count[cell] >= 3:
                        value, ladder = _triangle_reference(
                            support_vertices[cell, : support_count[cell]],
                            density_sampler,
                            cell,
                        )
                    else:
                        value, ladder = 0.0, []
                    cell_error_bound = _refinement_error_bound(ladder)
                    reference_cell_rows.append(
                        {
                            "cell": cell,
                            "current_a": value,
                            "quadrature_error_bound_a": cell_error_bound,
                            "refinement": ladder,
                        }
                    )
            reference = float(sum(row["current_a"] for row in reference_cell_rows))
            bounds = [row["quadrature_error_bound_a"] for row in reference_cell_rows]
            if all(bound is not None for bound in bounds):
                error_bound = float(sum(float(bound) for bound in bounds))
            else:
                error_bound_reason = (
                    "At least one selected cell has no pair of refinement levels, so a "
                    "summed adjacent-level difference is unavailable."
                )
        exact = None if booked is None else float(booked["exact"])
        selected_exact = (
            None
            if exact_cell_current is None or reference_cells is None
            else float(sum(exact_cell_current[cell] for cell in reference_cells))
        )
        return {
            "revision": revision,
            "generating_commit": generating_commit,
            "archive_tree": str(tree),
            "module": str(Path(census.__file__).resolve()),
            "forward_operator_module": str(Path(forward_operator.__file__).resolve()),
            "cwd": str(Path.cwd().resolve()),
            "mode_strategy": mode_strategy,
            "solving_operator_mode": "exact",
            "component": component,
            "jax_backend": jax.default_backend(),
            "chord_booked_a": None if booked is None else float(booked["chord"]),
            "exact_booked_a": exact,
            "exact_cell_current_a": exact_cell_current,
            "selected_exact_booked_a": selected_exact,
            "quadrature_reference_a": reference,
            "quadrature_error_bound_a": error_bound,
            "quadrature_error_bound_reason": error_bound_reason,
            "quadrature_cells": reference_cell_rows,
            "exact_relative_difference": (
                None
                if selected_exact is None or reference is None
                else abs(selected_exact - reference) / abs(reference)
            ),
        }
    finally:
        sys.path[:] = previous
        for name in list(sys.modules):
            if name in roots or name.startswith(prefixes):
                sys.modules.pop(name, None)
        sys.modules.update(modules)


def _render_terminal_panel(path: Path) -> dict[str, Any]:
    """Render the persisted terminal state, reference state and changed supports."""

    import jax.numpy as jnp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from benchmarks import solovev_certificate as certificate
    from benchmarks import unit_amplitude_current_census as census
    from nova.media import poloidal
    from nova.media.ink import DEFAULT_INK, poloidal_axes

    control = census._read_control_row()
    context = census._build_context(control)
    state = np.asarray(control["terminal_flux_wb"], dtype=np.float64)
    operator = context["operator"].with_clip_mode("exact")
    partition = _full_state_partition(operator, state, jnp)
    coordinates = np.asarray(control["coordinates_rz_m"], dtype=np.float64)
    wall = np.asarray(context["machine"].wall_node, dtype=np.float64)
    analytic_state = certificate._exact_state(
        census.CASE_NAME, context["exact"], coordinates
    )
    radial, height, terminal = certificate._raster_field(coordinates, state, wall)
    _radial, _height, analytic = certificate._raster_field(
        coordinates, analytic_state, wall
    )
    levels = poloidal.contour_levels(
        np.concatenate((terminal.ravel(), analytic.ravel())), count=12
    )
    figure, axis = plt.subplots(figsize=(14, 9), dpi=100)
    wall_units = (wall,)
    poloidal.draw_flux_contours(
        axis, radial, height, analytic, levels, color="#31759f", wall=wall_units
    )
    poloidal.draw_flux_contours(
        axis, radial, height, terminal, levels, color="#a85d30", wall=wall_units
    )
    poloidal.draw_wall(axis, units=wall_units)
    poloidal.draw_nulls(
        axis,
        magnetic_axis=certificate.AXIS_M,
        x_points=np.asarray(certificate.X_POINT_M, dtype=np.float64)[None, :],
        style=DEFAULT_INK.variant(axis_color="#31759f", xpoint_color="#31759f"),
        contain=wall_units,
    )
    topology = partition["topology"]
    poloidal.draw_nulls(
        axis,
        magnetic_axis=np.asarray(topology.axis, dtype=np.float64),
        x_points=np.asarray(topology.x_point, dtype=np.float64)[None, :],
        style=DEFAULT_INK.variant(axis_color="#a85d30", xpoint_color="#a85d30"),
        contain=wall_units,
    )
    changed_cells = (5, 7, 30, 65, 74, 78, 87, 110, 119, 124)
    support = partition["profile_support"]
    vertices = np.asarray(support.support_vertices, dtype=np.float64)
    count = np.asarray(support.vertex_count, dtype=np.int32)
    for cell in changed_cells:
        polygon = vertices[cell, : count[cell]]
        axis.plot(
            *np.vstack((polygon, polygon[0])).T,
            color="#b33939",
            linewidth=1.4,
            zorder=10,
        )
        centre = polygon.mean(axis=0)
        axis.text(centre[0], centre[1], str(cell), color="#b33939", fontsize=11)
    poloidal_axes(axis)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    plt.close(figure)
    return {
        "figure": str(path),
        "terminal_residual": float(
            json.loads(census.CONTROL_PART.read_text(encoding="utf-8"))["solver"][
                "terminal_fixed_point_residual"
            ]
        ),
        "terminal_status": "unqualified active-set-settled",
        "changed_cells": list(changed_cells),
    }


def _reference_comparison(row: dict[str, Any] | None) -> dict[str, Any]:
    """Compare the booking and reference over the same selected support cells."""

    if row is None:
        return {
            "available": False,
            "reason": "No receipt was generated for this revision.",
        }
    booking = row["selected_exact_booked_a"]
    reference = row["quadrature_reference_a"]
    bound = row["quadrature_error_bound_a"]
    if booking is None or reference is None or bound is None:
        return {
            "available": False,
            "booking_a": booking,
            "reference_a": reference,
            "tolerance_a": bound,
            "tolerance_basis": row["quadrature_error_bound_reason"],
            "reason": (
                "A booking, reference, or measured refinement bound is unavailable."
            ),
        }
    difference = abs(float(booking) - float(reference))
    return {
        "available": True,
        "booking_a": float(booking),
        "reference_a": float(reference),
        "absolute_difference_a": difference,
        "relative_difference": difference / max(abs(float(reference)), 1.0),
        "tolerance_a": float(bound),
        "tolerance_basis": (
            "Sum of the absolute differences between the last two triangle-rule "
            "refinement levels for every selected clipped polygon."
        ),
        "within_tolerance": difference <= float(bound),
    }


def _recommendation(current: dict[str, Any], pinned: dict[str, Any]) -> str:
    """Select the disposition from the paired measurements, not a preset verdict."""

    current_matches = current.get("within_tolerance")
    pinned_matches = pinned.get("within_tolerance")
    if current_matches is True and pinned_matches is False:
        return "re-pin"
    if current_matches is False and pinned_matches is True:
        return "repair"
    return "undetermined"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head", default="main")
    parser.add_argument("--revision", action="append")
    parser.add_argument(
        "--component", choices=("full", "booking", "reference"), default="full"
    )
    parser.add_argument("--reference-cells")
    parser.add_argument("--reference-revision", action="append")
    parser.add_argument("--terminal-panel", type=Path)
    arguments = parser.parse_args()
    scratch = Path(
        tempfile.mkdtemp(prefix="outboard-census-", dir=os.environ["TMPDIR"])
    )
    try:
        if arguments.terminal_panel is not None:
            payload = _render_terminal_panel(arguments.terminal_panel)
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return
        head = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", arguments.head], text=True
        ).strip()
        generating_commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        parent = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", f"{RESPONSIBLE_REVISION}^"],
            text=True,
        ).strip()
        revisions = arguments.revision or (
            PINNED_REVISION,
            parent,
            RESPONSIBLE_REVISION,
            head,
        )
        reference_cells = (
            None
            if arguments.reference_cells is None
            else {int(cell) for cell in arguments.reference_cells.split(",") if cell}
        )
        reference_revisions = set(arguments.reference_revision or revisions)
        rows = []
        for revision in revisions:
            component = arguments.component
            if component == "full" and revision not in reference_revisions:
                component = "booking"
            rows.append(
                _measure(
                    revision,
                    scratch,
                    component,
                    reference_cells,
                    generating_commit,
                )
            )
        by_revision = {row["revision"]: row for row in rows}
        current = by_revision.get(head)
        pinned = by_revision.get(PINNED_REVISION)
        current_comparison = _reference_comparison(current)
        pinned_comparison = _reference_comparison(pinned)
        payload = {
            "generating_commit": generating_commit,
            "revisions": rows,
            "responsible_commit": RESPONSIBLE_REVISION,
            "responsible_parent": parent,
            "current_comparison": current_comparison,
            "pinned_comparison": pinned_comparison,
            "current_matches_reference": current_comparison.get("within_tolerance"),
            "pinned_matches_reference": pinned_comparison.get("within_tolerance"),
            "reference_a": (
                None if current is None else current["quadrature_reference_a"]
            ),
            "recommendation": _recommendation(current_comparison, pinned_comparison),
            "recommendation_basis": (
                "Derived from current_comparison.within_tolerance and "
                "pinned_comparison.within_tolerance over the same selected "
                "support cells."
            ),
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
