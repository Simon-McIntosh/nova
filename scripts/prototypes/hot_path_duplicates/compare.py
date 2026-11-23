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
import math
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
    """Read topology from a flux-map null census, and from supplied stationary
    points, on one analytic field."""
    from nova.equilibrium.solve_request import TopologyPolicy
    from nova.equilibrium.topology import (
        TopologyConvention,
        read,
        read_at_stationary_points,
        stationary_read,
    )
    from tests.equilibrium.test_topology_read import (
        _analytic_inputs,
        _realised_hex_geometry,
    )

    oracle, field, wall, axis, saddle = _analytic_inputs("diverted")
    geometry = _realised_hex_geometry(wall, 50)
    convention = TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    pitch = float(np.sqrt(np.median(np.asarray(geometry.full_area))))
    seeds = jnp.asarray(np.asarray([axis, saddle]))

    def supplied_points(evaluator):
        def polish(point):
            return stationary_read(evaluator, point, pitch, policy)

        return jax.vmap(polish)(seeds)

    def supplied_read(evaluator):
        return jax.jit(
            lambda inner, solved: read_at_stationary_points(
                inner, geometry, convention, policy, solved
            )
        )(evaluator, supplied_points(evaluator))

    census_read = jax.jit(read)(field, geometry, convention, policy)
    supplied = supplied_read(field)

    axis_gap = float(
        np.max(np.abs(np.asarray(census_read.axis) - np.asarray(supplied.axis)))
    )
    census_x = np.asarray(census_read.x_points)[
        np.asarray(census_read.x_point_valid).astype(bool)
    ]
    supplied_x = np.asarray(supplied.x_points)[
        np.asarray(supplied.x_point_valid).astype(bool)
    ]
    if census_x.shape == supplied_x.shape and census_x.size:
        x_gap = float(np.max(np.abs(census_x - supplied_x)))
    else:
        x_gap = math.nan
    membership_gap = float(
        np.max(
            np.abs(np.asarray(census_read.membership) - np.asarray(supplied.membership))
        )
    )
    labels_equal = bool(
        np.array_equal(
            np.asarray(census_read.domain_labels), np.asarray(supplied.domain_labels)
        )
    )
    agrees = (
        axis_gap <= pitch
        and (math.isnan(x_gap) or x_gap <= pitch)
        and membership_gap <= 1.0e-12
        and labels_equal
    )

    shifted = dataclasses.replace(field, shift=jnp.asarray([6.0 * pitch, 3.0 * pitch]))
    census_shifted = jax.jit(read)(shifted, geometry, convention, policy)
    control_gap = float(
        np.max(np.abs(np.asarray(census_shifted.axis) - np.asarray(supplied.axis)))
    )
    emit(
        "NEGATIVE_CONTROL",
        entry="topology-read",
        perturbation=(
            "the field passed to read_at_stationary_points is shifted by "
            "(6, 3) cell pitches"
        ),
        measured_gap_m=control_gap,
        tolerance_m=pitch,
        disagrees_beyond_tolerance=bool(control_gap > pitch),
    )
    receipt["negative_controls"].append(
        dict(
            entry="topology-read",
            measured_gap_m=control_gap,
            tolerance_m=pitch,
            disagrees_beyond_tolerance=bool(control_gap > pitch),
        )
    )
    return dict(
        entry="topology-read",
        implementation_a="nova.equilibrium.topology.read (flux-map null census)",
        implementation_b=(
            "nova.equilibrium.topology.read_at_stationary_points (points "
            "supplied by topology.stationary_read)"
        ),
        input=(
            "analytic Cerfon-Freidberg single-null field, diverted, 50-cell "
            "realised hexagonal geometry"
        ),
        axis_a=np.asarray(census_read.axis).tolist(),
        axis_b=np.asarray(supplied.axis).tolist(),
        axis_gap_m=axis_gap,
        x_gap_m=x_gap,
        membership_gap=membership_gap,
        domain_labels_equal=labels_equal,
        cell_pitch_m=pitch,
        tolerance=dict(
            axis_position_m=pitch, x_point_position_m=pitch, membership=1.0e-12
        ),
        agrees=agrees,
    )


def entry_extreme_selection(receipt):
    """Pick the extreme value and its index, twice."""
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
        index_mismatches=mismatches,
        max_index_gap=0
        if not mismatches
        else max(abs(item[2] - item[3]) for item in mismatches),
        tolerance_max_index_gap=0,
        agrees=not mismatches,
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


def entry_route_probe(receipt):
    """Count which support read the public forward solve executes."""
    import nova.equilibrium.topology as topology_module
    from nova.equilibrium.topology import Topology
    from tests.test_equilibrium_magnetics_fit import _physical_problem

    counts = {"read_qualification": 0, "read_at_stationary_points": 0}
    original_read_qualification = Topology.read_qualification
    original_stationary = topology_module.read_at_stationary_points

    def counted_read_qualification(self, *args, **kwargs):
        counts["read_qualification"] += 1
        return original_read_qualification(self, *args, **kwargs)

    def counted_stationary(field, geometry, convention, policy, solved, **options):
        counts["read_at_stationary_points"] += 1
        return original_stationary(
            field, geometry, convention, policy, solved, **options
        )

    Topology.read_qualification = counted_read_qualification
    topology_module.read_at_stationary_points = counted_stationary
    try:
        profile, request, *_ = _physical_problem("mast")
        profile.solve(request)
    finally:
        Topology.read_qualification = original_read_qualification
        topology_module.read_at_stationary_points = original_stationary
    emit(
        "ROUTE_PROBE",
        route="public-forward-solve",
        **counts,
        positive_control=counts["read_qualification"] > 0,
        support_read_receipt=(
            "nova/equilibrium/support_read.py read/read_batch carry 0 body "
            "lines on both measured routes"
        ),
    )
    return dict(
        entry="route-probe",
        counts=counts,
        positive_control=counts["read_qualification"] > 0,
        agrees=counts["read_at_stationary_points"] == 0,
    )


def compare(receipt_path):
    receipt = {"entries": [], "negative_controls": []}
    for function in (
        entry_topology_read,
        entry_extreme_selection,
        entry_storage_mode,
        entry_level_polish,
        entry_route_probe,
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
