"""Measure exact-section design builds against the pre-excision default."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import types
from types import SimpleNamespace

import numpy as np

from nova.biot.polysection import PolySection
from nova.biot.cylinder import Cylinder
from nova.biot.polybow import section_corners
from nova.biot.solve import Solve
from nova.frame.coil import Coil


REFERENCE_PATH = Path("tests/test_equilibrium_forward_reference.py")
FIXTURES = {"coarse": (1, 566), "fine": (2, 1069)}
MATRIX_FIELDS = (
    "source_to_grid",
    "plasma_to_grid",
    "plasma_to_grid_r",
    "plasma_to_grid_z",
    "source_to_sample",
    "plasma_to_sample",
    "plasma_to_sample_r",
    "plasma_to_sample_z",
    "source_to_wall",
    "plasma_to_wall",
    "plasma_to_wall_r",
    "plasma_to_wall_z",
)


def load_module(path: Path, name: str):
    """Load a Python module without collecting its tests."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def committed_module(path: str, name: str):
    """Load one committed production module beside the revised worktree module."""
    source = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = types.ModuleType(name)
    module.__file__ = f"<committed {path}>"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def baseline_elements():
    """Return the committed polygon and rectangle production elements."""
    module = committed_module(
        "nova/biot/polysection.py", "nova.biot._pre_excision_polysection"
    )

    class CompatiblePolySection(module.PolySection):
        """Accept the revised immutable policy at the benchmark seam."""

        def __post_init__(self):
            if hasattr(self.policy, "key"):
                self.policy = self.policy.key
            super().__post_init__()

    rectangle = committed_module(
        "nova/biot/cylinder.py", "nova.biot._pre_fidelity_cylinder"
    ).Cylinder
    return CompatiblePolySection, rectangle


def build(
    reference,
    case,
    multiplier: int,
    polygon_element,
    rectangle_element,
    *,
    qualify_authored_sections: bool,
) -> tuple[object, float]:
    """Build one complete design-matrix fixture through one section element."""
    reference.WALL_NODES = 3 * multiplier
    requested = reference.SUITE_CELLS * multiplier
    original_polygon = Solve.generator["polysection"]
    original_rectangle = Solve.generator["cylinder"]
    original_route = Coil._route_authored_sections
    Solve.generator["polysection"] = polygon_element
    Solve.generator["cylinder"] = rectangle_element
    if not qualify_authored_sections:
        Coil._route_authored_sections = lambda self, index: None
    started = time.perf_counter()
    try:
        machine = reference.build_machine(case, requested, passive=True)
    finally:
        elapsed = time.perf_counter() - started
        Solve.generator["polysection"] = original_polygon
        Solve.generator["cylinder"] = original_rectangle
        Coil._route_authored_sections = original_route
    return machine, elapsed


def compare(before, after) -> dict[str, object]:
    """Account every matrix row changed by removal of unreachable branches."""
    fields = {}
    maximum = 0.0
    changed = 0
    for name in MATRIX_FIELDS:
        first = np.asarray(getattr(before, name))
        second = np.asarray(getattr(after, name))
        difference = np.abs(first - second)
        field_maximum = float(np.max(difference, initial=0.0))
        field_changed = int(np.count_nonzero(first != second))
        fields[name] = {
            "shape": list(first.shape),
            "bitwise_equal": bool(np.array_equal(first, second)),
            "changed_entries": field_changed,
            "maximum_absolute_change": field_maximum,
        }
        maximum = max(maximum, field_maximum)
        changed += field_changed
    for prefix, first_block, second_block in (
        ("radial_field", before.radial_field, after.radial_field),
        ("vertical_field", before.vertical_field, after.vertical_field),
    ):
        for index, (first, second) in enumerate(
            zip(first_block, second_block, strict=True)
        ):
            name = f"{prefix}_{index}"
            first = np.asarray(first)
            second = np.asarray(second)
            difference = np.abs(first - second)
            field_maximum = float(np.max(difference, initial=0.0))
            field_changed = int(np.count_nonzero(first != second))
            fields[name] = {
                "shape": list(first.shape),
                "bitwise_equal": bool(np.array_equal(first, second)),
                "changed_entries": field_changed,
                "maximum_absolute_change": field_maximum,
            }
            maximum = max(maximum, field_maximum)
            changed += field_changed
    return {
        "fields": fields,
        "all_bitwise_equal": changed == 0,
        "changed_entries": changed,
        "maximum_absolute_change": maximum,
        "accounting": (
            "Every changed entry is generated by replacing the rectangle-kernel "
            "substitution on a non-axis-aligned authored source with the exact "
            "polygon-section kernel; rows already using an exact shape-matched "
            "kernel are checked separately for bit identity."
        ),
    }


def driven_field_correction(before, after) -> dict[str, float | int]:
    """Return the external-current field change caused by authored-shape routing."""
    current = np.asarray(after.source_current)
    flux = (after.source_to_grid - before.source_to_grid) @ current
    radial = (after.radial_field[0] - before.radial_field[0]) @ current
    vertical = (after.vertical_field[0] - before.vertical_field[0]) @ current
    wall_flux = (after.source_to_wall - before.source_to_wall) @ current
    magnitude = np.hypot(radial, vertical)
    grid_flux_index = int(np.argmax(np.abs(flux)))
    wall_flux_index = int(np.argmax(np.abs(wall_flux)))
    field_index = int(np.argmax(magnitude))
    return {
        "grid_flux_sup_wb": float(np.max(np.abs(flux))),
        "grid_flux_argmax_cell": grid_flux_index,
        "grid_flux_argmax_rz_m": np.asarray(after.node)[grid_flux_index].tolist(),
        "wall_flux_sup_wb": float(np.max(np.abs(wall_flux))),
        "wall_flux_argmax_row": wall_flux_index,
        "wall_flux_argmax_rz_m": np.asarray(after.wall_node)[wall_flux_index].tolist(),
        "grid_radial_field_sup_t": float(np.max(np.abs(radial))),
        "grid_vertical_field_sup_t": float(np.max(np.abs(vertical))),
        "grid_poloidal_field_vector_sup_t": float(np.max(magnitude)),
        "grid_poloidal_field_argmax_cell": field_index,
        "grid_poloidal_field_argmax_rz_m": np.asarray(after.node)[field_index].tolist(),
    }


def source_column_accounting(before, after, labels) -> list[dict[str, object]]:
    """Attribute every externally driven column change to its conductor."""
    blocks = {
        "grid_flux_wb_per_a": (before.source_to_grid, after.source_to_grid),
        "sample_flux_wb_per_a": (before.source_to_sample, after.source_to_sample),
        "wall_flux_wb_per_a": (before.source_to_wall, after.source_to_wall),
        "grid_radial_field_t_per_a": (
            before.radial_field[0],
            after.radial_field[0],
        ),
        "grid_vertical_field_t_per_a": (
            before.vertical_field[0],
            after.vertical_field[0],
        ),
    }
    rows = []
    for column, label in enumerate(labels):
        maxima = {
            name: float(
                np.max(
                    np.abs(np.asarray(second)[:, column] - np.asarray(first)[:, column])
                )
            )
            for name, (first, second) in blocks.items()
        }
        rows.append(
            {
                "column": column,
                "conductor": label,
                "bitwise_equal": all(value == 0.0 for value in maxima.values()),
                "maximum_absolute_change": maxima,
            }
        )
    return rows


SNAPSHOT_FIELDS = (
    "source_current",
    "node",
    "wall_node",
    *MATRIX_FIELDS,
)


def _hash_array(digest, name: str, value) -> None:
    """Add one named array to a stable shape-, dtype-, and value-aware hash."""
    array = np.asarray(value)
    if array.dtype.kind in "OU":
        array = np.asarray(array, dtype=str)
    array = np.ascontiguousarray(array)
    digest.update(name.encode())
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())


def _source_geometry_arrays(machine) -> dict[str, np.ndarray]:
    """Return route-independent authored source geometry and ordering arrays."""
    frame = machine.coilset.subframe
    conductor = ~np.asarray(frame["plasma"], dtype=bool)
    arrays = {
        name: np.asarray(frame[name])[conductor]
        for name in (
            "frame",
            "part",
            "section",
            "polysection_policy",
            "x",
            "z",
            "dl",
            "dt",
            "area",
            "nturn",
        )
    }
    digest = hashlib.sha256()
    for position, polygon in enumerate(
        np.asarray(frame["poly"], dtype=object)[conductor]
    ):
        _hash_array(digest, f"polygon-{position}", section_corners(polygon))
    arrays["polygon_digest"] = np.asarray(digest.hexdigest())
    return arrays


def fixture_receipt(machine, fixture: str, route: str, seconds: float) -> dict:
    """Return a common-input receipt and route-specific construction metadata."""
    digest = hashlib.sha256()
    common = {
        "source_current": machine.source_current,
        "node": machine.node,
        "area": machine.area,
        "hexagon": machine.hexagon,
        "stencil": machine.stencil,
        "wall_node": machine.wall_node,
        "sampling_vertices": machine.sampling_vertices,
        "sample_coordinates": machine.sample_coordinates,
        **_source_geometry_arrays(machine),
    }
    for name, value in common.items():
        _hash_array(digest, name, value)
    for position, polygon in enumerate(machine.cell_polygons):
        _hash_array(digest, f"cell-polygon-{position}", polygon)
    frame = machine.coilset.subframe
    conductor = ~np.asarray(frame["plasma"], dtype=bool)
    labels, counts = np.unique(
        np.asarray(frame["segment"], dtype=str)[conductor], return_counts=True
    )
    return {
        "fixture": fixture,
        "route": route,
        "fixture_sha256": digest.hexdigest(),
        "cells": len(machine.node),
        "wall_rows": len(machine.wall_node),
        "sample_rows": len(machine.sample_coordinates),
        "source_columns": len(machine.source_current),
        "source_elements": int(np.count_nonzero(conductor)),
        "route_counts": {
            str(label): int(count) for label, count in zip(labels, counts)
        },
        "cpu_design_matrix_wall_seconds": seconds,
    }


def write_snapshot(machine, labels, output: Path) -> str:
    """Write one immutable numeric route snapshot and return its content hash."""
    if output.exists():
        raise FileExistsError(f"refusing to replace immutable snapshot {output}")
    arrays = {name: np.asarray(getattr(machine, name)) for name in SNAPSHOT_FIELDS}
    arrays.update(
        {
            f"radial_field_{index}": np.asarray(value)
            for index, value in enumerate(machine.radial_field)
        }
    )
    arrays.update(
        {
            f"vertical_field_{index}": np.asarray(value)
            for index, value in enumerate(machine.vertical_field)
        }
    )
    arrays["source_labels"] = np.asarray(labels, dtype=str)
    digest = hashlib.sha256()
    for name in sorted(arrays):
        _hash_array(digest, name, arrays[name])
    np.savez(output, **arrays)
    return digest.hexdigest()


def snapshot_hash(path: Path) -> str:
    """Return the content hash of a banked numeric route snapshot."""
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as stored:
        for name in sorted(stored.files):
            _hash_array(digest, name, stored[name])
    return digest.hexdigest()


def load_snapshot(path: Path) -> SimpleNamespace:
    """Load a numeric route snapshot into the comparison interface."""
    with np.load(path, allow_pickle=False) as stored:
        values = {name: np.asarray(stored[name]) for name in SNAPSHOT_FIELDS}
        values["source_labels"] = np.asarray(stored["source_labels"], dtype=str)
        values["radial_field"] = tuple(
            np.asarray(stored[f"radial_field_{index}"]) for index in range(4)
        )
        values["vertical_field"] = tuple(
            np.asarray(stored[f"vertical_field_{index}"]) for index in range(4)
        )
    return SimpleNamespace(**values)


def stage(args) -> None:
    """Build and bank one scheduler-bounded route with its common-input receipt."""
    if args.receipt.exists():
        raise FileExistsError(f"refusing to replace immutable receipt {args.receipt}")
    multiplier, expected_cells = FIXTURES[args.fixture]
    reference = load_module(REFERENCE_PATH, "exact_section_reference_fixture")
    reference.configure_dtypes()
    case = reference.require_reference()
    if args.route == "before":
        polygon_element, rectangle_element = baseline_elements()
        qualify = False
    else:
        polygon_element, rectangle_element = PolySection, Cylinder
        qualify = True
    machine, seconds = build(
        reference,
        case,
        multiplier,
        polygon_element,
        rectangle_element,
        qualify_authored_sections=qualify,
    )
    if len(machine.node) != expected_cells:
        raise AssertionError(
            f"{args.fixture} fixture cell count is not {expected_cells}"
        )
    labels = [conductor.name for conductor in case.drive(True)]
    receipt = fixture_receipt(machine, args.fixture, args.route, seconds)
    receipt["snapshot_sha256"] = write_snapshot(machine, labels, args.snapshot)
    receipt["snapshot_path"] = str(args.snapshot)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


def compare_snapshots(args) -> None:
    """Verify shared receipts and compare two immutable route snapshots."""
    before_receipt = json.loads(args.before_receipt.read_text())
    after_receipt = json.loads(args.after_receipt.read_text())
    for name in ("fixture", "fixture_sha256", "cells", "source_columns"):
        if before_receipt[name] != after_receipt[name]:
            raise AssertionError(f"route receipt mismatch for {name}")
    if before_receipt["fixture"] != args.fixture:
        raise AssertionError("comparison fixture differs from route receipts")
    for path, receipt in (
        (args.before_snapshot, before_receipt),
        (args.after_snapshot, after_receipt),
    ):
        if snapshot_hash(path) != receipt["snapshot_sha256"]:
            raise AssertionError(f"snapshot content hash differs for {path}")
    before = load_snapshot(args.before_snapshot)
    after = load_snapshot(args.after_snapshot)
    if not np.array_equal(before.source_labels, after.source_labels):
        raise AssertionError("source ordering differs between route snapshots")
    if not np.array_equal(before.source_current, after.source_current):
        raise AssertionError("source currents differ between route snapshots")
    before_seconds = before_receipt["cpu_design_matrix_wall_seconds"]
    after_seconds = after_receipt["cpu_design_matrix_wall_seconds"]
    report = {
        "fixture": args.fixture,
        "cells": before_receipt["cells"],
        "receipt_verification": {
            "shared_fixture_sha256": before_receipt["fixture_sha256"],
            "fixture_inputs_identical": True,
            "source_order_bitwise_equal": True,
            "source_current_bitwise_equal": True,
            "before": before_receipt,
            "after": after_receipt,
        },
        "cpu_design_matrix_wall_seconds": {
            "before_reachable_route_excision": before_seconds,
            "after_exact_only_policy": after_seconds,
            "after_over_before": after_seconds / before_seconds,
        },
        "matrix_comparison": compare(before, after),
        "driven_field_correction": driven_field_correction(before, after),
        "source_column_accounting": source_column_accounting(
            before, after, before.source_labels.tolist()
        ),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["cpu_design_matrix_wall_seconds"], sort_keys=True))


def main() -> None:
    """Stage scheduler-bounded route snapshots or compare a verified pair."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    stage_parser = commands.add_parser("stage")
    stage_parser.add_argument("fixture", choices=tuple(FIXTURES))
    stage_parser.add_argument("route", choices=("before", "after"))
    stage_parser.add_argument("snapshot", type=Path)
    stage_parser.add_argument("receipt", type=Path)
    stage_parser.set_defaults(function=stage)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("fixture", choices=tuple(FIXTURES))
    compare_parser.add_argument("before_snapshot", type=Path)
    compare_parser.add_argument("before_receipt", type=Path)
    compare_parser.add_argument("after_snapshot", type=Path)
    compare_parser.add_argument("after_receipt", type=Path)
    compare_parser.add_argument("output", type=Path)
    compare_parser.set_defaults(function=compare_snapshots)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
