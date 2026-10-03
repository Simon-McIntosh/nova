"""Measure booked current against an independently integrated analytic support."""

from __future__ import annotations

import argparse
import io
import json
import math
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
from scipy.integrate import quad


ROOT = Path(__file__).resolve().parents[1]
CONTROL_PART = (
    "docs/figures/uniform-cell-clip-and-coupling/exact-participation/"
    "discriminator/parts/control/weak-rotation-reactor-static-production-route-reduced.json"
)
ARCHIVE_PATHS = ("nova", "tests", "benchmarks", "scripts")
PINNED_REVISION = "247e5aa4a"
RESPONSIBLE_REVISION = "38b441dad1dea2b10b684fa612b8802ef0973b86"
RELATIVE_TOLERANCE = 5.0e-13


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


def _vertical_bounds(vertices: np.ndarray, radius: float) -> tuple[float, float] | None:
    """Return a convex cell's vertical intersection at one radius."""

    scale = max(float(np.max(abs(vertices))), 1.0)
    tolerance = 256.0 * np.finfo(np.float64).eps * scale
    heights: list[float] = []
    for first, second in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        radial_delta = second[0] - first[0]
        if abs(radial_delta) <= tolerance:
            if abs(radius - first[0]) <= tolerance:
                heights.extend((float(first[1]), float(second[1])))
            continue
        fraction = (radius - first[0]) / radial_delta
        if -tolerance <= fraction <= 1.0 + tolerance:
            heights.append(float(first[1] + fraction * (second[1] - first[1])))
    return None if len(heights) < 2 else (min(heights), max(heights))


def _exact_support_current(case: Any, polygon: np.ndarray) -> tuple[float, float]:
    """Integrate one analytic cell support without production clip moments."""

    plasma_lower, plasma_upper = case.boundary_midplane_radii()
    lower = max(float(polygon[:, 0].min()), float(plasma_lower))
    upper = min(float(polygon[:, 0].max()), float(plasma_upper))
    if upper <= lower:
        return 0.0, 0.0

    def density(radius: float) -> float:
        remaining = float(case.axis_flux - case._flux_offset(case._flux_label(radius)))
        if remaining <= 0.0:
            return 0.0
        half_height = math.sqrt(remaining / float(case.field_coefficient))
        bounds = _vertical_bounds(polygon, radius)
        if bounds is None:
            return 0.0
        lower_height = max(bounds[0], -half_height)
        upper_height = min(bounds[1], half_height)
        if upper_height <= lower_height:
            return 0.0
        return float(case.toroidal_current_density(radius, 0.0)) * (
            upper_height - lower_height
        )

    breaks = [lower, upper]
    breaks.extend(float(value) for value in polygon[:, 0] if lower < value < upper)
    axis = float(case.major_radius)
    if lower < axis < upper:
        breaks.append(axis)
    total = 0.0
    error = 0.0
    intervals = sorted(set(breaks))
    for start, stop in zip(intervals, intervals[1:]):
        value, estimate = quad(
            density,
            start,
            stop,
            epsabs=1.0e-8,
            epsrel=RELATIVE_TOLERANCE,
            limit=300,
        )
        total += value
        error += estimate
    return float(total), float(error)


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


def _measure(revision: str, scratch: Path) -> dict[str, Any]:
    tree = _archive_tree(revision, scratch)
    previous = list(sys.path)
    prefixes = ("nova.", "benchmarks.", "scripts.")
    modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "nova" or name.startswith(prefixes)
    }
    try:
        for name in list(modules):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(tree))
        import jax
        import nova.equilibrium.forward_operator as forward_operator
        from benchmarks import unit_amplitude_current_census as census
        from nova.jax.config import configure_dtypes

        configure_dtypes()
        _install_absent_saddle_bridge(forward_operator)
        control = census._read_control_row()
        context = census._build_context(control)
        state = np.asarray(control["terminal_flux_wb"], dtype=np.float64)
        operator = context["operator"]
        probe = census._partition_probe(operator, state, "exact")
        curve = census._curve_probe(
            operator,
            probe["base_masks"],
            probe["topology"],
            probe["sample_psi_norm"],
        )
        analytic = census._cell_analysis(
            context["machine"], context["exact"], context["target_current"]
        )
        booked = census._state_census(
            operator,
            context["machine"],
            state,
            "terminal",
            context["target_current"],
            analytic,
            curve,
        )["unit_amplitude_totals_a"]
        reference_values = [
            _exact_support_current(
                context["exact"], np.asarray(polygon, dtype=np.float64)
            )
            for polygon in context["machine"].cell_polygons
        ]
        reference = float(sum(value for value, _error in reference_values))
        error_bound = float(sum(error for _value, error in reference_values))
        exact = float(booked["exact"])
        return {
            "revision": revision,
            "archive_tree": str(tree),
            "module": str(Path(census.__file__).resolve()),
            "cwd": str(Path.cwd().resolve()),
            "jax_backend": jax.default_backend(),
            "chord_booked_a": float(booked["chord"]),
            "exact_booked_a": exact,
            "quadrature_reference_a": reference,
            "quadrature_error_bound_a": error_bound,
            "exact_relative_difference": abs(exact - reference) / abs(reference),
        }
    finally:
        sys.path[:] = previous
        for name in list(sys.modules):
            if name == "nova" or name.startswith(("nova.", "benchmarks.", "scripts.")):
                sys.modules.pop(name, None)
        sys.modules.update(modules)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head", default="main")
    parser.add_argument("--revision", action="append")
    arguments = parser.parse_args()
    scratch = Path(
        tempfile.mkdtemp(prefix="outboard-census-", dir=os.environ["TMPDIR"])
    )
    try:
        head = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", arguments.head], text=True
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
        rows = [_measure(revision, scratch) for revision in revisions]
        by_revision = {row["revision"]: row for row in rows}
        current = by_revision.get(head)
        pinned = by_revision.get(PINNED_REVISION)
        payload = {
            "revisions": rows,
            "responsible_commit": RESPONSIBLE_REVISION,
            "responsible_parent": parent,
            "current_matches_reference": (
                None
                if current is None
                else current["exact_relative_difference"] <= 1.0e-9
            ),
            "pinned_matches_reference": (
                None
                if pinned is None
                else pinned["exact_relative_difference"] <= 1.0e-9
            ),
            "reference_a": (
                None if current is None else current["quadrature_reference_a"]
            ),
            "recommendation": (
                "re-pin"
                if current is not None
                and current["exact_relative_difference"] <= 1.0e-9
                else "repair"
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
