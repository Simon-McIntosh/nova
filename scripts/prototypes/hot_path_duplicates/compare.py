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
        np.linalg.norm(np.asarray(reading.axis) - np.asarray(reference.axis))
    )
    axis_gap_to_analytic = float(
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
        np.linalg.norm(np.asarray(disturbed.reading.axis) - np.asarray(reference.axis))
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
        axis_gap_to_analytic_m=axis_gap_to_analytic,
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
                mismatches.append(
                    dict(
                        maximize=maximize,
                        axis=axis,
                        stencil=first.tolist(),
                        boundary=second.tolist(),
                        index_gap=int(np.max(np.abs(first - second))),
                    )
                )
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
        max_index_gap=max((item["index_gap"] for item in mismatches), default=0),
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
    modules = (polyplot, vtkplot)
    found = {module.__name__: implementations(module) for module in modules}
    comparisons = []
    by_case = {case: {module.__name__: [] for module in modules} for case, *_ in cases}
    for module in modules:
        for label, method in found[module.__name__].items():
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
                by_case[case][module.__name__].append(host.frame.calls)
    missing = [name for name, methods in found.items() if not methods]
    vacuous = [
        case
        for case, counts in by_case.items()
        if any(not side for side in counts.values())
    ]
    disagreements = [
        case
        for case, counts in by_case.items()
        if case not in vacuous
        and len({tuple(sorted(side)) for side in counts.values()}) != 1
    ]
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
        implementations_found={
            name: sorted(methods) for name, methods in found.items()
        },
        tolerance="exact equality of delegate call counts per column state",
        missing_implementations=missing,
        vacuous_cases=vacuous,
        disagreements=disagreements,
        agrees=not missing and not vacuous and not disagreements,
    )


def _function_bodies(tree, modules):
    """Every function the route modules define, with its body's source form."""
    import ast
    import hashlib

    rows = []
    for relative in modules:
        path = tree / relative
        if not path.exists():
            continue
        parsed = ast.parse(path.read_text())
        for node in ast.walk(parsed):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            body = list(node.body)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            source = ast.unparse(ast.Module(body=body, type_ignores=[]))
            rows.append(
                dict(
                    module=relative,
                    function=node.name,
                    statements=len(body),
                    asynchronous=isinstance(node, ast.AsyncFunctionDef),
                    source=source,
                    digest=hashlib.sha256(source.encode()).hexdigest()[:12],
                )
            )
    return rows


def entry_candidate_agreements(receipt):
    """Measure the four candidates the scan flags but no entry covered.

    Each pair runs on one stated input in one process, and each carries the
    keeper and the removal the register records.
    """
    import types

    pairs = []

    from nova.equilibrium.flux_surface_extraction import (
        _tensor_bernstein as extraction_bernstein,
    )
    from nova.linalg.tensor_spline import _tensor_bernstein as spline_bernstein

    coefficient = jnp.asarray(np.cos(np.arange(16, dtype=float)).reshape(4, 4))
    radial = jnp.asarray(np.linspace(0.0, 1.0, 5))
    vertical = jnp.asarray(np.linspace(1.0, 0.0, 5))
    first = np.asarray(extraction_bernstein(coefficient, radial, vertical, 3, 3))
    second = np.asarray(spline_bernstein(coefficient, radial, vertical, 3, 3))
    gap = (
        float(np.max(np.abs(first - second)))
        if first.shape == second.shape
        else float("inf")
    )
    pairs.append(
        dict(
            candidate="tensor-bernstein",
            implementation_a="nova.equilibrium.flux_surface_extraction._tensor_bernstein",
            implementation_b="nova.linalg.tensor_spline._tensor_bernstein",
            input="one 4x4 coefficient block at five paired coordinates, order 3",
            tolerance=0.0,
            max_abs_gap=gap,
            agrees=gap == 0.0,
            keeper="nova.linalg.tensor_spline._tensor_bernstein",
            lines_removable=4,
        )
    )

    from nova.biot.cylinder import Cylinder
    from nova.biot.polysection import PolySection

    class _Stub:
        Psi = np.arange(1.0, 9.0).reshape(4, 2)
        mu_0 = 4.0e-7 * np.pi

        def target(self, key):
            assert key == "r"
            return np.asarray([[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])

    first = np.asarray(Cylinder.__dict__["Aphi"].func(_Stub()))
    second = np.asarray(PolySection.__dict__["Aphi"].func(_Stub()))
    gap = float(np.max(np.abs(first - second)))
    pairs.append(
        dict(
            candidate="aphi-axis-zero",
            implementation_a="nova.biot.cylinder.Cylinder.Aphi",
            implementation_b="nova.biot.polysection.PolySection.Aphi",
            input="one 4x2 target grid with a zero-radius cell, mu_0 = 4 pi 1e-7",
            tolerance=0.0,
            max_abs_gap=gap,
            agrees=gap == 0.0,
            keeper="one copy in the shared Matrix base",
            lines_removable=9,
        )
    )

    from nova.biot.polygonanalytic import _Vertex as AnalyticVertex
    from nova.biot.polygonarc import _Vertex as ArcVertex

    class _Corner:
        xp = np
        radius = np.asarray([0.4, 0.7, 1.1])
        level = np.asarray([0.05, 0.2, 0.05])
        offset = np.asarray([0.0, 0.0, 0.0])
        radius_sum = np.asarray([1.1, 1.4, 1.8])
        panels = ((0.0, 0.5 * np.pi), (0.5 * np.pi, np.pi))

    first = np.asarray(ArcVertex._first_residual(_Corner(), 8))
    second = np.asarray(AnalyticVertex._first_residual(_Corner(), 8))
    gap = (
        float(np.max(np.abs(first - second)))
        if first.shape == second.shape
        else float("inf")
    )
    pairs.append(
        dict(
            candidate="first-residual",
            implementation_a="nova.biot.polygonarc._Vertex._first_residual",
            implementation_b="nova.biot.polygonanalytic._first_residual",
            input="one three-row corner at eight graded nodes",
            tolerance=0.0,
            max_abs_gap=gap,
            agrees=gap == 0.0,
            keeper="the polygonanalytic form (a superset: optional weight)",
            lines_removable=1,
        )
    )

    from nova.transport.ensemble import EnsembleTransportReceipt
    from nova.transport.window_batch import WindowBatchReceipt

    members = tuple(
        types.SimpleNamespace(member_id=name, transport=name)
        for name in ("alpha", "beta", "gamma")
    )
    ensemble = types.SimpleNamespace(members=members)
    window = types.SimpleNamespace(members=members)

    def identity(value):
        return getattr(value, "member_id", value)

    selected_a = identity(EnsembleTransportReceipt.for_member(ensemble, "beta"))
    selected_b = identity(WindowBatchReceipt.for_member(window, "beta"))
    missing = []
    for implementation, host in (
        (EnsembleTransportReceipt.for_member, ensemble),
        (WindowBatchReceipt.for_member, window),
    ):
        try:
            implementation(host, "absent")
        except KeyError:
            missing.append(True)
        else:
            missing.append(False)
    pairs.append(
        dict(
            candidate="for-member",
            implementation_a="nova.transport.ensemble.EnsembleTransportReceipt.for_member",
            implementation_b="nova.transport.window_batch.WindowBatchReceipt.for_member",
            input="one three-member list addressed by identity, present and absent",
            tolerance="same member selected, KeyError for an absent identity",
            selected_same_member=selected_a == selected_b == "beta",
            keyerror_both=all(missing),
            agrees=selected_a == selected_b == "beta" and all(missing),
            keeper="one loop with a per-class receipt accessor",
            lines_removable=7,
        )
    )
    return dict(
        entry="candidate-agreements",
        pairs=pairs,
        tolerance="each pair states its own; the record agrees when every pair does",
        agrees=all(pair["agrees"] for pair in pairs),
    )


def _reuse_check(tree):
    """Load the existing reuse scanner from the tree being scanned."""
    import importlib.util

    path = tree / "scripts/iron_operator_bound/reuse_check.py"
    spec = importlib.util.spec_from_file_location("reuse_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def definitions_cross_check(tree, modules, rows):
    """Discover function names with the existing scanner, and require the walk
    to have seen exactly the same names in every route module."""
    reuse = _reuse_check(tree)
    mismatches = []
    for relative in modules:
        path = tree / relative
        if not path.exists():
            continue
        discovered = reuse.definitions(path)
        walked = {
            row["function"]
            for row in rows
            if row["module"] == relative and not row["asynchronous"]
        }
        if discovered != walked:
            mismatches.append(
                dict(
                    module=relative,
                    missing=sorted(discovered - walked),
                    unexpected=sorted(walked - discovered),
                )
            )
    return dict(
        source="scripts/iron_operator_bound/reuse_check.py:definitions",
        agrees=not mismatches,
        mismatches=mismatches,
    )


def ast_scan(tree, map_path):
    """The static duplicate scan the register's candidate list comes from.

    Function names shared by two route modules, exact body digests across
    modules, and a near-identity pass over the shared names. The route modules
    are the ones the line map records as executing at least one line.
    """
    import difflib

    line_map = json.loads(map_path.read_text())
    modules = [
        item["module"]
        for item in line_map["modules"]
        if item.get("union", {}).get("executed", 0) > 0
    ]
    rows = _function_bodies(tree, modules)

    names = {}
    for row in rows:
        names.setdefault(row["function"], set()).add(row["module"])
    shared = {name: sorted(found) for name, found in names.items() if len(found) > 1}

    groups = {}
    for row in rows:
        groups.setdefault(row["digest"], []).append(row)
    identical = []
    for digest, entries in sorted(groups.items()):
        modules_in_group = {entry["module"] for entry in entries}
        if len(entries) < 2 or len(modules_in_group) < 2:
            continue
        identical.append(
            dict(
                digest=digest,
                same_name=len({entry["function"] for entry in entries}) == 1,
                entries=[
                    {key: entry[key] for key in ("module", "function", "statements")}
                    for entry in entries
                ],
            )
        )

    near = []
    for name in sorted(shared):
        per_module = {}
        for row in rows:
            if row["function"] == name:
                per_module.setdefault(row["module"], []).append(row)
        ordered = sorted(per_module)
        for first in range(len(ordered)):
            for second in range(first + 1, len(ordered)):
                for left in per_module[ordered[first]]:
                    for right in per_module[ordered[second]]:
                        ratio = difflib.SequenceMatcher(
                            None, left["source"], right["source"]
                        ).ratio()
                        if ratio >= 0.85:
                            near.append(
                                dict(
                                    function=name,
                                    module_a=ordered[first],
                                    module_b=ordered[second],
                                    statements_a=left["statements"],
                                    statements_b=right["statements"],
                                    ratio=round(ratio, 4),
                                )
                            )
    near.sort(key=lambda item: (-item["ratio"], item["function"]))
    return dict(
        line_map=str(map_path),
        line_map_revision=line_map.get("revision"),
        definitions=definitions_cross_check(tree, modules, rows),
        route_modules=modules,
        route_module_count=len(modules),
        shared_names=shared,
        shared_name_count=len(shared),
        identical_bodies=identical,
        same_named_multi_statement=[
            group
            for group in identical
            if group["same_name"]
            and min(entry["statements"] for entry in group["entries"]) >= 2
        ],
        near_identity_threshold=0.85,
        near_identity=near,
    )


def write_ast_scan(path, tree, revision):
    map_path = (
        tree / "docs/evidence/fragments/code-reuse-and-hot-path/hot-path-map.json"
    )
    scan = ast_scan(tree, map_path)
    scan["revision"] = revision
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scan, indent=2))
    emit(
        "AST_SCAN",
        path=str(path),
        revision=revision,
        route_module_count=scan["route_module_count"],
        shared_name_count=scan["shared_name_count"],
        identical_groups=len(scan["identical_bodies"]),
        near_identity_pairs=len(scan["near_identity"]),
        definitions_agree=scan["definitions"]["agrees"],
    )
    if not scan["definitions"]["agrees"]:
        emit("DEFINITIONS_DISAGREE", mismatches=scan["definitions"]["mismatches"])
        raise SystemExit(1)


def compare(receipt_path):
    receipt = {"entries": [], "negative_controls": []}
    for function in (
        entry_topology_read,
        entry_extreme_selection,
        entry_storage_mode,
        entry_level_polish,
        entry_route_resolution,
        entry_plot_columns,
        entry_candidate_agreements,
    ):
        emit("ENTRY_START", entry=function.__name__)
        try:
            record = function(receipt)
        except Exception as error:  # noqa: BLE001
            emit("ENTRY_ERROR", entry=function.__name__, error=repr(error))
            receipt.setdefault("errors", []).append(function.__name__)
            continue
        receipt["entries"].append(record)
        if not record["agrees"]:
            emit("ENTRY_DISAGREES", **record)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str))
    disagreeing = [
        item.get("entry") for item in receipt["entries"] if item.get("agrees") is False
    ]
    required_controls = ("topology-read", "extreme-selection", "route-resolution")
    fired = {
        item["entry"]
        for item in receipt["negative_controls"]
        if item.get("disagrees_beyond_tolerance")
    }
    silent = [entry for entry in required_controls if entry not in fired]
    for entry in silent:
        emit("NEGATIVE_CONTROL_SILENT", entry=entry, required=True)
    failed = disagreeing + list(receipt.get("errors", [])) + silent
    emit(
        "SUMMARY",
        receipt=str(receipt_path),
        failed=failed,
        controls_fired=sorted(fired),
        controls_required=list(required_controls),
    )
    print(
        f"{len(receipt['entries']) - len(disagreeing)} entries passed, "
        f"{len(failed)} failed, {len(fired)} negative controls fired",
        flush=True,
    )
    return len(failed)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--scan-ast", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tree", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.receipt is None and args.scan_ast is None:
        parser.error("one of --receipt or --scan-ast is required")
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
    if args.scan_ast is not None:
        write_ast_scan(args.scan_ast, args.tree, args.revision)
    raise SystemExit(1 if args.receipt is not None and compare(args.receipt) else 0)


if __name__ == "__main__":
    main()
