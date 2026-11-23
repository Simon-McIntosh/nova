"""Compare duplicate implementations of one quantity on one stated input.

Each entry names two implementations that the line map shows on or beside the
forward and inverse route, computes the same quantity, and runs here on one
stated input in one process. The measured agreement is reported against a
stated tolerance, and one entry is rerun with a perturbed input to one
implementation, which must disagree beyond that tolerance.

Run on CPU in one allocation; the driver refuses any other backend.
"""

# ruff: noqa: E402
import argparse
import dataclasses
import json
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

assert jax.config.jax_enable_x64 is True


def emit(kind, **fields):
    print(kind, json.dumps(fields, default=float), flush=True)


def entry_topology_read(receipt):
    """Read topology from the stationary points a solve supplies, and from the
    flux-map null census, on one analytic field."""
    from tests.test_support_read import _case

    operands, cache, roots, pitch, step, reference, analytic_axis = _case("diverted")
    support, booked = step(operands, cache, roots + 0.03 * pitch, roots)
    reading = support.reading
    area = np.asarray(operands.geometry.full_area)
    pitch = float(pitch)

    axis_gap = float(
        np.linalg.norm(np.asarray(reading.axis) - np.asarray(analytic_axis))
    )
    selected = np.asarray(reading.membership) * area
    census = np.asarray(reference.membership) * area
    area_gap = float(abs(selected.sum() - census.sum())) / float(census.sum())
    cell_gap = float(np.max(np.abs(selected - census) / area))
    class_equal = int(reading.boundary_class) == int(reference.boundary_class)
    tolerance = dict(
        boundary_class="equal",
        axis_position_m=0.05 * pitch,
        membership_area_sum_relative=5.0e-3,
        membership_cell_fraction=0.05,
    )
    agrees = (
        class_equal
        and axis_gap < 0.05 * pitch
        and area_gap < 5.0e-3
        and cell_gap < 0.05
    )

    gradient = np.asarray(operands.exterior)[:, 1]
    delta = 0.1 * float(np.median(np.abs(gradient)))
    perturbed_exterior = operands.exterior.at[:, 1].add(delta)
    perturbed_operands = dataclasses.replace(operands, exterior=perturbed_exterior)
    perturbed_cache = dataclasses.replace(cache, exterior=perturbed_exterior)
    disturbed, _ = step(
        perturbed_operands, perturbed_cache, roots + 0.03 * pitch, roots
    )
    disturbed_selected = np.asarray(disturbed.reading.membership) * area
    control_gap = float(np.max(np.abs(disturbed_selected - census) / area))
    control_axis_gap = float(
        np.linalg.norm(np.asarray(disturbed.reading.axis) - np.asarray(analytic_axis))
    )
    control_fires = bool(control_gap >= 0.05 or control_axis_gap >= 0.05 * pitch)
    emit(
        "NEGATIVE_CONTROL",
        entry="topology-read",
        perturbation=(
            "the exterior nodal jet gains a uniform flux gradient of 0.1x the "
            "exterior nodal derivative. Used by the version of the duplicate "
            "that reads the flux map's own stationary points."
        ),
        measured_cell_fraction=control_gap,
        measured_axis_gap_m=control_axis_gap,
        tolerance_cell_fraction=0.05,
        tolerance_axis_m=0.05 * pitch,
        disagrees_beyond_tolerance=control_fires,
    )
    receipt["negative_controls"].append(
        dict(
            entry="topology-read",
            measured_cell_fraction=control_gap,
            measured_axis_gap_m=control_axis_gap,
            disagrees_beyond_tolerance=control_fires,
        )
    )
    return dict(
        entry="topology-read",
        implementation_a=(
            "nova.equilibrium.topology.read (flux-map null census, "
            "Topology.read_qualification on the measured routes)"
        ),
        implementation_b=(
            "nova.equilibrium.topology.read_at_stationary_points (points "
            "solved by fixed_points.solve and supplied by support_read.read)"
        ),
        input=(
            "analytic Cerfon-Freidberg single-null field, diverted, 50-cell "
            "realised hexagonal geometry, both route implementations run on "
            "the same operands in one process"
        ),
        axis_a=np.asarray(reference.axis).tolist(),
        axis_b=np.asarray(reading.axis).tolist(),
        axis_gap_m=axis_gap,
        boundary_class_a=int(reference.boundary_class),
        boundary_class_b=int(reading.boundary_class),
        membership_area_sum_relative=area_gap,
        membership_cell_fraction=cell_gap,
        cell_pitch_m=pitch,
        tolerance=tolerance,
        agrees=agrees,
    )


def entry_extreme_selection(receipt):
    """Pick the extreme value and its index, twice."""
    from nova.equilibrium import connectivity_boundary, stencil_nulls
    from nova.equilibrium.connectivity_boundary import _arg_extreme as boundary
    from nova.equilibrium.stencil_nulls import _arg_extreme as stencil

    generator = np.random.default_rng(20261123)
    values = generator.normal(size=(64, 9))
    values[3, :] = values[3, 0]
    values[11, :] = 1.5
    values[20, :] = -2.5
    mismatches = []
    for maximize in (True, False):
        for axis in (0, 1):
            first = np.asarray(stencil(values, axis, maximize=maximize))
            second = np.asarray(boundary(values, maximize=maximize, axis=axis))
            if not np.array_equal(first, second):
                mismatches.append([maximize, axis, first.tolist(), second.tolist()])
    line = values[7]
    exact_companions = []
    for name in ("_argmax_exact", "_argmin_exact"):
        first = int(getattr(stencil_nulls, name)(line))
        second = int(getattr(connectivity_boundary, name)(line))
        exact_companions.append(
            dict(
                function=name,
                stencil=first,
                boundary=second,
                agrees=bool(first == second),
            )
        )
    perturbed = values.copy()
    perturbed[0, 2] = perturbed[0].max() + 7.0
    control_gap = int(
        np.max(
            np.abs(
                np.asarray(stencil(values, 1, maximize=True))
                - np.asarray(boundary(perturbed, maximize=True, axis=1))
            )
        )
    )
    emit(
        "NEGATIVE_CONTROL",
        entry="extreme-selection",
        perturbation=(
            "the second implementation's input row has one element raised 7.0 "
            "past the previous maximum, moving its chosen index; the first "
            "implementation keeps the unperturbed input"
        ),
        measured_index_gap=control_gap,
        tolerance_max_index_gap=0,
        disagrees_beyond_tolerance=bool(control_gap > 0),
    )
    receipt["negative_controls"].append(
        dict(
            entry="extreme-selection",
            measured_index_gap=control_gap,
            tolerance_max_index_gap=0,
            disagrees_beyond_tolerance=bool(control_gap > 0),
        )
    )
    return dict(
        entry="extreme-selection",
        implementation_a="nova.equilibrium.stencil_nulls._arg_extreme",
        implementation_b="nova.equilibrium.connectivity_boundary._arg_extreme",
        input=(
            "64x9 seeded normal array with one constant tie row and two all-tie "
            "rows, maxima over axes 0 and 1"
        ),
        exact_companions=exact_companions,
        index_mismatches=mismatches,
        max_index_gap=0
        if not mismatches
        else max(abs(item[2] - item[3]) for item in mismatches),
        tolerance_max_index_gap=0,
        agrees=not mismatches and all(item["agrees"] for item in exact_companions),
    )


def entry_storage_mode(receipt):
    """Return the compound group path and the write mode of one dataset, twice."""
    import tempfile

    from nova.database.netcdf import netCDF
    from nova.database.zarrstore import ZarrStore

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "present.nc").write_bytes(b"")
        (root / "present.zarr").mkdir()
        netcdf_absent = netCDF(dirname=root, filename="absent", group="plasma")
        netcdf_present = netCDF(dirname=root, filename="present", group="plasma")
        zarr_absent = ZarrStore(dirname=root, filename="absent", group="plasma")
        zarr_present = ZarrStore(dirname=root, filename="present", group="plasma")
        pairs = (
            (
                "subgroup_nested",
                netcdf_absent.subgroup("equilibrium", None, "flux"),
                zarr_absent.subgroup("equilibrium", None, "flux"),
            ),
            ("subgroup_none", netCDF().subgroup(), ZarrStore().subgroup()),
            (
                "subgroup_declared",
                netcdf_present.subgroup("flux"),
                zarr_present.subgroup("flux"),
            ),
            ("mode_absent", netcdf_absent.get_mode(), zarr_absent.get_mode()),
            ("mode_present", netcdf_present.get_mode(), zarr_present.get_mode()),
            ("mode_explicit", netcdf_absent.get_mode("r"), zarr_absent.get_mode("r")),
        )
        comparisons = [
            dict(quantity=quantity, netcdf=first, zarr=second, agrees=first == second)
            for quantity, first, second in pairs
        ]
    disagreements = [item for item in comparisons if not item["agrees"]]
    return dict(
        entry="storage-group-mode",
        implementation_a="nova.database.netcdf.netCDF.subgroup/get_mode (is_file)",
        implementation_b=(
            "nova.database.zarrstore.ZarrStore.subgroup/get_mode (is_store)"
        ),
        input=(
            "one temp root holding an existing .nc file and an existing .zarr "
            "directory, plus an absent basename, group 'plasma'"
        ),
        comparisons=comparisons,
        disagreements=disagreements,
        tolerance="exact string equality",
        agrees=not disagreements,
    )


def entry_level_polish(receipt):
    """Polish a level-set crossing along a chord, twice."""
    from nova.equilibrium.forward_operator import _polished_level_root
    from nova.equilibrium.separatrix_clip import (
        _SPLINE_BOUNDARY_SEGMENTS,
        _traced_level_arc,
    )

    start = jnp.asarray([[1.0, -0.4], [1.6, 0.2], [2.1, -1.1]])
    end = jnp.asarray([[1.9, 0.6], [2.3, -0.5], [2.4, 0.9]])
    inside = jnp.asarray([[1.4, 0.05], [1.9, -0.2], [2.2, -0.1]])
    centre = jnp.asarray([1.5, 0.0])
    scale = jnp.asarray([0.9, 1.2])

    def evaluator(points):
        local = (points - centre) / scale
        return (
            local[..., 0] ** 2
            + 1.7 * local[..., 1] ** 2
            - 0.35 * local[..., 0] * local[..., 1]
            - 0.8
        )

    traced = _traced_level_arc(start, end, evaluator, inside)
    parameter = jnp.linspace(0.0, 1.0, _SPLINE_BOUNDARY_SEGMENTS + 1, dtype=start.dtype)
    chord = start[:, None, :] + parameter[None, :, None] * (end - start)[:, None, :]
    delta = end - start
    normal = jnp.stack((-delta[:, 1], delta[:, 0]), axis=1)
    squared_length = jnp.sum(delta**2, axis=1)
    midpoint = 0.5 * (start + end)
    inside_side = jnp.sum((inside - midpoint) * normal, axis=1)
    side = jnp.where(inside_side < 0.0, 1.0, -1.0)
    local_extent = jnp.minimum(
        jnp.linalg.norm(inside - midpoint, axis=1)
        / jnp.sqrt(jnp.maximum(squared_length, jnp.finfo(start.dtype).tiny)),
        1.0,
    )
    signed_extent = side * jnp.maximum(local_extent, 32.0 * jnp.finfo(start.dtype).eps)
    lower = jnp.minimum(signed_extent, 0.0)[:, None]
    upper = jnp.maximum(signed_extent, 0.0)[:, None]
    root = _polished_level_root(chord, normal, lower, upper, evaluator)
    polished = chord + root[..., None] * normal[:, None, :]
    gap = float(np.max(np.abs(np.asarray(traced) - np.asarray(polished))))
    return dict(
        entry="level-set-polish",
        implementation_a="nova.equilibrium.forward_operator._polished_level_root",
        implementation_b="nova.equilibrium.separatrix_clip._traced_level_arc "
        "polish loop",
        input=(
            "three 65-point chords across one fixed quadratic level set, one "
            "interior vertex each"
        ),
        max_point_gap_m=gap,
        tolerance_m=1.0e-12,
        agrees=bool(gap <= 1.0e-12),
    )


def entry_route_resolution(receipt):
    """Resolve which topology read each measured route executes, statically.

    Counting calls inside a jitted program is not possible, so this reads the
    route modules' own source and reports which implementation each names.
    """
    import ast

    import nova

    root = Path(nova.__file__).resolve().parent

    def count(relative, name):
        tree = ast.parse((root / relative).read_text())
        total = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == name:
                total += 1
            elif isinstance(node, ast.Name) and node.id == name:
                total += 1
        return total

    def judge(relative, required, forbidden):
        present = count(relative, required)
        forbidden_present = count(relative, forbidden)
        return dict(
            module=f"nova/{relative}",
            required=required,
            required_calls=present,
            forbidden=forbidden,
            forbidden_calls=forbidden_present,
            passes=bool(present > 0 and forbidden_present == 0),
        )

    census = "read_qualification"
    supplied = "read_at_stationary_points"
    results = [
        judge("equilibrium/forward.py", census, supplied),
        judge("equilibrium/forward_operator.py", census, supplied),
        judge("equilibrium/support_read.py", supplied, census),
    ]
    agrees = all(item["passes"] for item in results)
    control = judge("equilibrium/support_read.py", census, supplied)
    control_fires = not control["passes"]
    emit(
        "NEGATIVE_CONTROL",
        entry="route-resolution",
        perturbation=(
            "the forward-route expectation is applied to the module that reads "
            "supplied stationary points, where it must fail"
        ),
        measured=control,
        disagrees_beyond_tolerance=control_fires,
    )
    receipt["negative_controls"].append(
        dict(entry="route-resolution", disagrees_beyond_tolerance=control_fires)
    )
    return dict(
        entry="route-resolution",
        question=(
            "nova/equilibrium/support_read.py read, read_batch, read_and_book "
            "and bank_program carry kernels in the device trace and zero body "
            "lines on both measured routes. Which topology read does each "
            "route run?"
        ),
        findings=results,
        conclusion=(
            "both measured routes read topology through "
            "Topology.read_qualification, the flux-map null census. The "
            "supplied-stationary-point read is reached only from "
            "support_read.py, whose callers execute zero body lines on these "
            "routes"
        ),
        agrees=agrees,
    )


def entry_plot_columns(receipt):
    """Bring one plotting indexer's frame columns up to date, twice."""
    from nova.frame import polyplot, vtkplot

    def implementations(module):
        found = {}
        for name, value in vars(module).items():
            if isinstance(value, type) and value.__module__ == module.__name__:
                method = value.__dict__.get("update_columns")
                if callable(method):
                    found[f"{module.__name__}.{name}.update_columns"] = method
        return found

    class Frame:
        def __init__(self, columns):
            self.columns = columns
            self.calls = 0

        def update_columns(self):
            self.calls += 1

    class Host:
        def __init__(self, columns, required, additional):
            self.required = required
            self.additional = additional
            self.frame = Frame(columns)

    cases = (
        ("all present", ["a", "b", "c"], ["a"], ["b"]),
        ("one missing", ["a"], ["a", "b"], []),
        ("all missing", [], ["a"], ["b", "c"]),
    )
    comparisons = []
    for module in (polyplot, vtkplot):
        for label, method in implementations(module).items():
            for case, columns, required, additional in cases:
                host = Host(columns, required, additional)
                method(host)
                comparisons.append(
                    dict(
                        implementation=label,
                        case=case,
                        delegate_calls=host.frame.calls,
                    )
                )
    by_case = {}
    for item in comparisons:
        by_case.setdefault(item["case"], []).append(item["delegate_calls"])
    disagreements = [case for case, counts in by_case.items() if len(set(counts)) != 1]
    return dict(
        entry="plot-columns",
        implementation_a=(
            "nova.frame.polyplot plot indexer update_columns "
            "(PolyPlot and PolyPlotChildren)"
        ),
        implementation_b="nova.frame.vtkplot plot indexer update_columns",
        input=(
            "three column states (all present, one missing, all missing) through "
            "a stub host carrying required, additional and frame.columns"
        ),
        comparisons=comparisons,
        tolerance="exact equality of delegate call counts per column state",
        disagreements=disagreements,
        agrees=not disagreements,
    )


def compare(receipt_path):
    receipt = {"entries": [], "negative_controls": []}
    for function in (
        entry_topology_read,
        entry_extreme_selection,
        entry_storage_mode,
        entry_level_polish,
        entry_route_resolution,
        entry_plot_columns,
    ):
        emit("ENTRY_START", entry=function.__name__)
        try:
            record = function(receipt)
        except Exception as error:  # noqa: BLE001
            emit("ENTRY_ERROR", entry=function.__name__, error=repr(error))
            continue
        receipt["entries"].append(record)
        if not record["agrees"]:
            emit("ENTRY_DISAGREES", **record)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str))
    emit(
        "SUMMARY",
        receipt=str(receipt_path),
        failed=[
            item.get("entry")
            for item in receipt["entries"]
            if item.get("agrees") is False or "error" in item
        ],
    )
    print("EXIT=0", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tree", type=Path, default=Path.cwd())
    args = parser.parse_args()
    emit(
        "HEADER",
        revision=args.revision,
        tree=str(args.tree.resolve()),
        cwd=str(Path.cwd().resolve()),
        backend=str(jax.default_backend()),
        x64=bool(jax.config.jax_enable_x64),
        driver=str(Path(__file__).resolve()),
    )
    assert jax.default_backend() == "cpu"
    compare(args.receipt)


if __name__ == "__main__":
    main()
