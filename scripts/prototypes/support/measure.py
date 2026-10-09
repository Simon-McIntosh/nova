"""Compare connected support authorities in the certificate map."""

# Precision precedes every array-valued import.
# ruff: noqa: E402
from __future__ import annotations

import argparse
from copy import copy
import json
import os
from pathlib import Path
import resource
import subprocess
import traceback
from time import perf_counter
from types import MethodType, SimpleNamespace

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.database.zarrstore import ZarrStore
from nova.equilibrium import topology
from nova.equilibrium.domain import DomainMasks, PlasmaDomain
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from tests.equilibrium.test_topology_read import _analytic_inputs
from scripts.prototypes.support.geometry import (
    certificate_field,
    pack,
    analytic_polygons,
    read_polygons,
    analytic_moments,
)

assert jax.config.jax_enable_x64 is True
ROOT = Path(__file__).resolve().parents[3]
STAGE_LABEL = "STAGE"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def stage(name, function):
    print(f"{STAGE_LABEL}_START {name}", flush=True)
    start = perf_counter()
    result = function()
    jax.block_until_ready(result)
    wall = perf_counter() - start
    print(f"{STAGE_LABEL}_DONE {name} seconds={wall:.9g}", flush=True)
    return result, wall


def machine_from_cache(carrier, requested, wall, inherited):
    identity = fixture.cache_identity(
        carrier,
        requested_cells=requested,
        wall_nodes=fixture.WALL_POINT_COUNT,
        wall=wall,
    )
    store = ZarrStore(
        filename=f"{fixture.CACHE_FILENAME}_{abs(requested)}", dirname=inherited
    )
    store.group = store.hash_attrs(identity)
    try:
        store.load()
        machine = fixture._from_dataset(store.data, identity, store.group)
    except FileNotFoundError, KeyError, OSError, ValueError:
        return fixture.cached_machine(
            carrier, requested, wall_nodes=fixture.WALL_POINT_COUNT, wall=wall
        )
    machine.cache.update(
        store=str(store.filepath), semantic_key=store.group, hit=True, readonly=True
    )
    return machine


def exterior_from_cache(source, exact, machine, empty, analytic, inherited):
    identity = fixture._exterior_cache_identity(source, exact, machine, analytic)
    store = ZarrStore(filename=fixture.EXTERIOR_CACHE_FILENAME, dirname=inherited)
    store.group = store.hash_attrs(identity)
    try:
        store.load()
        assert store.data.attrs["semantic_identity"] == json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        )
        moments = fixture.CellCurrentMoments(
            *(
                np.asarray(store.data[name])
                for name in ("cell_current", "radial_moment", "vertical_moment")
            )
        )
        exterior = np.asarray(store.data["exterior"])
        return (
            moments,
            exterior,
            dict(hit=True, readonly=True, store=str(store.filepath)),
        )
    except FileNotFoundError, KeyError, OSError, ValueError:
        return fixture.cached_fixture_exterior(source, exact, machine, empty, analytic)


def partition(operator, state, axis_flux, boundary_flux, saddle, support):
    span = boundary_flux - axis_flux
    grid = state[: operator.grid.node_number]
    norm = (grid - axis_flux) / span
    sample = (operator.sample_node_flux(state) - axis_flux) / span
    labels = jnp.where(
        support.included, int(PlasmaDomain.CORE), int(PlasmaDomain.EXCLUDED_MATERIAL)
    ).astype(jnp.int32)
    return (
        DomainMasks(labels, norm),
        SimpleNamespace(
            axis_flux=axis_flux,
            boundary_flux=boundary_flux,
            flux_span=span,
            x_point=saddle,
        ),
        sample,
        support,
    )


def override(operator, fixed):
    """Override only the private partition on a shallow prototype instance."""
    active = copy(operator)

    def support_partition(self, psi, requested_class=None):
        del requested_class
        masks, state, _sample, support = fixed
        return partition(
            self, psi, state.axis_flux, state.boundary_flux, state.x_point, support
        )

    active._support_partition = MethodType(support_partition, active)
    return active


def read_support(operator, analytic, machine, kind, template):
    _, field, _, _, _ = _analytic_inputs(kind)
    field = certificate_field(field, kind)
    probes = jax.vmap(field.value)(jnp.asarray(machine.node[:16]))
    np.testing.assert_allclose(probes, analytic[:16], rtol=1e-11, atol=1e-13)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    reading = jax.jit(topology.read)(field, geometry, convention, TopologyPolicy())
    jax.block_until_ready(reading)
    if not bool(reading.valid) or not bool(reading.qualified):
        raise AssertionError(
            f"read refused certificate carrier: reason={int(reading.reason)}"
        )
    polygons = read_polygons(geometry, reading, convention.sigma)
    support = pack(polygons, template)
    gap = float(
        np.max(
            np.abs(
                np.asarray(support.area / support.full_area)
                - np.asarray(reading.membership)
            )
        )
    )
    if gap > 2e-5:
        raise AssertionError(f"packed support differs from read membership: {gap}")
    return reading, support


def measure(args):
    source_revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        f"SOURCE_REVISION={source_revision} "
        f"HARNESS_REVISION={os.environ.get('NOVA_MEASUREMENT_REVISION')} "
        f"CASE={args.kind} CELLS={args.cells} ARMS={args.arms}",
        flush=True,
    )
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false":
        raise RuntimeError("CUDA preallocation must be disabled")
    if jax.default_backend() != "gpu" or len(jax.devices("gpu")) != 1:
        raise RuntimeError("one GPU is required")
    jax.config.update("jax_enable_compilation_cache", False)
    print(
        f"MODULE={Path(certificate.__file__).resolve()} CWD={Path.cwd().resolve()}",
        flush=True,
    )
    fixture.ZarrStore = lambda *, filename, dirname, group=None: ZarrStore(
        filename=filename, dirname=args.out / "cache", group=group
    )
    name = (
        "weak-rotation-reactor-static"
        if args.kind == "limited"
        else "diverted-single-null"
    )
    (carrier, source, exact), case_wall = stage(
        "case", lambda: certificate._case(name, clip_mode="exact")
    )
    wall = certificate._diverted_wall(exact) if args.kind == "diverted" else None
    requested = -500 if args.cells == 550 else -args.cells
    platforms = os.environ.get("JAX_PLATFORMS", "cuda,cpu")
    os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        machine, machine_wall = stage(
            "machine", lambda: machine_from_cache(carrier, requested, wall, args.cache)
        )
    finally:
        os.environ["JAX_PLATFORMS"] = platforms
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = jnp.asarray(
        certificate._exact_state(name, exact, coordinates), dtype=jnp.float64
    )
    assert np.ptp(np.asarray(analytic[: len(machine.node)])) > 1e-10
    empty = fixture.forward_operator(source, machine)
    (physical, exterior, exterior_cache), exterior_wall = stage(
        "exterior",
        lambda: exterior_from_cache(
            source, exact, machine, empty, analytic, args.cache
        ),
    )
    operator, carrier_wall = stage(
        "carrier",
        lambda: fixture.forward_operator(source, machine, exterior).with_clip_mode(
            "exact"
        ),
    )
    target, _, target_receipt = certificate._closed_form_current_target(
        name, source, operator, physical
    )
    span = abs(float(fixture._analytic_axis_flux(exact)))
    branch = (
        topology.TopologyClass.LIMITED
        if args.kind == "limited"
        else topology.TopologyClass.DIVERTED
    )
    identity = dict(
        revision=source_revision,
        harness_revision=os.environ["NOVA_MEASUREMENT_REVISION"],
        module=str(Path(certificate.__file__).resolve()),
        cwd=str(Path.cwd().resolve()),
        job_id=os.environ.get("SLURM_JOB_ID"),
        device=str(jax.devices()[0]),
        case=args.kind,
        requested_cells=requested,
        realised_cells=len(machine.node),
        pitch_m=float(np.sqrt(np.median(machine.area))),
        fixture_walls=dict(
            case=case_wall,
            machine=machine_wall,
            exterior=exterior_wall,
            carrier=carrier_wall,
        ),
        machine_cache=machine.cache,
        exterior_cache=exterior_cache,
    )
    write(args.out / f"{args.kind}-{args.cells}-fixture.json", identity)
    legacy_partition, legacy_wall = stage(
        "legacy-support",
        lambda: jax.jit(lambda state: operator._support_partition(state, branch))(
            analytic
        ),
    )
    # The topology object is a pytree only in the production partition. The
    # prototype adapter is closed over after its measured support construction.
    exact_polygons, polygon_wall = stage(
        "analytic-polygons", lambda: analytic_polygons(exact, machine.cell_polygons)
    )
    coarse_polygons, refinement_wall = stage(
        "analytic-polygon-refinement",
        lambda: analytic_polygons(exact, machine.cell_polygons, points=4097),
    )
    point_count = 8193

    def fraction_change(first, second):
        return float(
            np.max(
                np.abs(
                    np.asarray([p.area for p in first])
                    - np.asarray([p.area for p in second])
                )
                / np.asarray(legacy_partition[3].full_area)
            )
        )

    geometry_uncertainty = fraction_change(exact_polygons, coarse_polygons)
    subdivisions = 0
    while geometry_uncertainty > args.oracle_fraction_bound and point_count < 65537:
        coarse_polygons = exact_polygons
        if args.kind == "diverted":
            subdivisions += 1
            point_count = 8193 * 2**subdivisions
        else:
            point_count = 2 * point_count - 1
        exact_polygons, extra_wall = stage(
            "analytic-polygon-refinement",
            lambda: analytic_polygons(
                exact,
                machine.cell_polygons,
                points=8193 if args.kind == "diverted" else point_count,
                subdivisions=subdivisions,
            ),
        )
        refinement_wall += extra_wall
        geometry_uncertainty = fraction_change(exact_polygons, coarse_polygons)
    if geometry_uncertainty > args.oracle_fraction_bound:
        raise AssertionError(
            f"analytic polygon refinement unresolved: {geometry_uncertainty}"
        )
    exact_support, exact_wall = stage(
        "analytic-support", lambda: pack(exact_polygons, legacy_partition[3])
    )
    exact_wall += polygon_wall + refinement_wall
    identity["oracle_fraction_bound"] = args.oracle_fraction_bound
    identity["analytic_boundary_points"] = point_count
    identity["analytic_polygon_fraction_uncertainty"] = geometry_uncertainty
    identity["fixture_walls"]["analytic_polygon_refinement"] = refinement_wall
    true_physical, truth_wall = stage(
        "analytic-density",
        lambda: analytic_moments(
            source,
            exact_polygons,
            np.asarray(operator.moment_geometry.atomic_mesh.centroids),
        ),
    )
    true_current = float(jnp.sum(true_physical.cell_current))
    identity["fixture_walls"]["analytic_density"] = truth_wall
    exact_support.assert_no_refusal()
    exact_fraction = np.asarray(exact_support.area / exact_support.full_area)
    axis_flux = jnp.asarray(fixture._analytic_axis_flux(exact))
    exact_partition = partition(
        operator,
        analytic,
        axis_flux,
        jnp.asarray(0.0),
        fixture._analytic_saddle(exact),
        exact_support,
    )
    arms = [
        ("legacy", operator, legacy_partition, legacy_wall, None),
        (
            "exact",
            override(operator, exact_partition),
            exact_partition,
            exact_wall,
            None,
        ),
    ]
    shadow = np.asarray(
        jax.jit(lambda state: operator.residual_shadow_mask(state, branch))(analytic)
    )[: len(machine.node)]
    rows = []

    def run_arm(arm, active, fixed, support_wall, reading=None):
        support = fixed[3]
        support.assert_no_refusal()
        mapped = active.flux_map(requested_class=branch, target_current=target)
        executable, compile_wall = stage(
            f"{arm}-compile", lambda: jax.jit(mapped).lower(analytic).compile()
        )
        value, _ = stage(f"{arm}-first-execute", lambda: executable(analytic))
        value, warm_wall = stage(f"{arm}-warm-execute", lambda: executable(analytic))
        delta = np.asarray(value[: len(machine.node)] - analytic[: len(machine.node)])
        memory = executable.memory_analysis()
        raw, booking_wall = stage(
            f"{arm}-booking",
            lambda: jax.jit(lambda state: active.cell_current_moments(state, branch))(
                analytic
            ),
        )
        booked = float(jnp.sum(raw.cell_current))
        scaled = active.scaled_current_moments(
            raw, active.current_normalisation_amplitude(target, booked)
        )
        reference_image = operator.current_moment_image(
            operator.coupling_current_moments(true_physical)
        )
        booking_delta = np.asarray(
            operator.current_moment_image(scaled) - reference_image
        )[: len(machine.node)]
        exterior_closure = np.asarray(exterior + reference_image - analytic)[
            : len(machine.node)
        ]
        booking_delta = np.where(shadow, 0.0, booking_delta)
        exterior_closure = np.where(shadow, 0.0, exterior_closure)
        reconstruction_error = float(
            np.max(np.abs(booking_delta + exterior_closure - delta)) / span
        )
        if reconstruction_error > 1e-10:
            raise AssertionError(
                "residual decomposition does not reconstruct map: "
                f"{reconstruction_error}"
            )
        fraction = np.asarray(support.area / support.full_area)
        row = dict(
            identity,
            arm=arm,
            map_relative_sup=float(np.max(np.abs(delta)) / span),
            map_relative_rms=float(np.sqrt(np.mean(delta**2)) / span),
            membership_error_sup=float(np.max(np.abs(fraction - exact_fraction))),
            membership_error_rms=float(
                np.sqrt(np.mean((fraction - exact_fraction) ** 2))
            ),
            read_membership_error_sup=None
            if reading is None
            else float(np.max(np.abs(np.asarray(reading.membership) - exact_fraction))),
            read_polygon_fraction_gap=None
            if reading is None
            else float(np.max(np.abs(np.asarray(reading.membership) - fraction))),
            booked_current_raw_a=booked,
            booked_current_normalised_a=float(jnp.sum(scaled.cell_current)),
            analytic_current_a=true_current,
            certificate_target_current_a=target,
            certificate_current_receipt=target_receipt,
            raw_current_relative_error=booked / true_current - 1,
            normalised_current_relative_error=float(jnp.sum(scaled.cell_current))
            / true_current
            - 1,
            boundary_flux=float(fixed[1].boundary_flux),
            axis_flux=float(fixed[1].axis_flux),
            x_point=np.asarray(fixed[1].x_point).tolist(),
            support_seconds=support_wall,
            booking_seconds=booking_wall,
            cold_compile_seconds=compile_wall,
            warm_execute_seconds=warm_wall,
            serialized_executable_bytes=None,
            serialization_status="pending",
            device_temp_bytes=memory.temp_size_in_bytes,
            host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            decomposition_reconstruction_relative_sup=reconstruction_error,
            booking_image_relative_sup=float(np.max(np.abs(booking_delta)) / span),
            exterior_closure_relative_sup=float(
                np.max(np.abs(exterior_closure)) / span
            ),
            completed=True,
        )
        if arm == "legacy" and args.kind == "diverted" and args.cells == 550:
            control = json.loads(args.control.read_text())["map_relative_sup"]
            row["positive_control_expected"] = control
            row["positive_control_relative_delta"] = abs(
                row["map_relative_sup"] / control - 1
            )
            write(args.out / f"{args.kind}-{args.cells}-{arm}.json", row)
            assert row["positive_control_relative_delta"] <= 1e-9, row
            print(
                "POSITIVE_CONTROL "
                + json.dumps(
                    {k: v for k, v in row.items() if k.startswith("positive_control")}
                ),
                flush=True,
            )
        write(args.out / f"{args.kind}-{args.cells}-{arm}.json", row)
        print(
            "NUMERICAL_ROW_PERSISTED "
            + str(args.out / f"{args.kind}-{args.cells}-{arm}.json"),
            flush=True,
        )
        try:
            row["serialized_executable_bytes"] = len(
                executable.runtime_executable().serialize()
            )
            row["serialization_status"] = "complete"
        except jax.errors.JaxRuntimeError as error:
            if "size must be smaller than 2GiB" not in str(error):
                raise
            row["serialization_status"] = "unavailable"
            row["serialization_refusal"] = str(error)
            print(f"SERIALIZATION_REFUSAL arm={arm} {error}", flush=True)
        row["host_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write(args.out / f"{args.kind}-{args.cells}-{arm}.json", row)
        print("SUPPORT_ROW " + json.dumps(row, sort_keys=True, default=str), flush=True)
        rows.append(row)
        return row

    arm_failures = []
    for arm in arms:
        if arm[0] in args.arms:
            try:
                run_arm(*arm)
            except Exception as error:
                refusal = dict(
                    case=args.kind,
                    requested_cells=args.cells,
                    arm=arm[0],
                    error_type=type(error).__name__,
                    error=str(error),
                    traceback=traceback.format_exc(),
                    completed=False,
                )
                write(
                    args.out / f"{args.kind}-{args.cells}-{arm[0]}-refusal.json",
                    refusal,
                )
                print("ARM_REFUSAL " + json.dumps(refusal), flush=True)
                arm_failures.append(arm[0])
    if arm_failures:
        raise RuntimeError("support arms refused: " + ", ".join(arm_failures))
    if "read" in args.arms:
        (reading, support), read_wall = stage(
            "read-support",
            lambda: read_support(
                operator, analytic, machine, args.kind, legacy_partition[3]
            ),
        )
        read_partition = partition(
            operator,
            analytic,
            reading.axis_flux,
            reading.boundary_flux,
            jnp.where(
                jnp.any(reading.x_point_valid),
                reading.x_points[jnp.argmax(reading.x_point_valid)],
                jnp.full(2, jnp.nan),
            ),
            support,
        )
        run_arm(
            "read",
            override(operator, read_partition),
            read_partition,
            read_wall,
            reading,
        )
    if "shifted" in args.arms:
        if args.kind != "diverted" or args.cells != 550:
            raise ValueError("shifted support control requires diverted 550 cells")
        pitch = identity["pitch_m"]
        wrong, wrong_wall = stage(
            "shifted-support",
            lambda: pack(
                analytic_polygons(exact, machine.cell_polygons, (pitch, 0.0)),
                legacy_partition[3],
            ),
        )
        wrong_partition = partition(
            operator,
            analytic,
            axis_flux,
            jnp.asarray(0.0),
            fixture._analytic_saddle(exact) + jnp.asarray((pitch, 0.0)),
            wrong,
        )
        row = run_arm(
            "shifted", override(operator, wrong_partition), wrong_partition, wrong_wall
        )
        floor_row = next((r for r in rows if r["arm"] == "exact"), None)
        if floor_row is None:
            floor_row = json.loads((args.out / "diverted-550-exact.json").read_text())
        floor = floor_row["map_relative_sup"]
        assert row["map_relative_sup"] > floor, (row["map_relative_sup"], floor)
        print(
            f"NEGATIVE_CONTROL shifted_sup={row['map_relative_sup']:.12g} "
            f"exact_sup={floor:.12g} sensitivity=True",
            flush=True,
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("legacy", "exact", "shifted", "read"),
        default=("legacy", "exact", "read"),
    )
    parser.add_argument("--oracle-fraction-bound", type=float, default=2e-5)
    args = parser.parse_args()
    if not 0 < args.oracle_fraction_bound <= 1e-4:
        raise ValueError("oracle fraction bound must be positive and at most 1e-4")
    try:
        measure(args)
    except Exception as error:
        refusal = {
            "case": args.kind,
            "requested_cells": args.cells,
            "arms": args.arms,
            "revision": os.environ.get("NOVA_MEASUREMENT_REVISION"),
            "error_type": type(error).__name__,
            "error": str(error),
            "signature": type(error).__name__ + ":" + str(error).split(":")[0],
            "traceback": traceback.format_exc(),
            "completed": False,
        }
        write(
            args.out / f"{args.kind}-{args.cells}-{'-'.join(args.arms)}-refusal.json",
            refusal,
        )
        raise


if __name__ == "__main__":
    main()
