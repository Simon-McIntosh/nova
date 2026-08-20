"""Profile closed-form oracle build and fixed-point accuracy costs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import json
from pathlib import Path
import re
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import LineString

from nova.biot.plasmagrid import PlasmaGrid
from nova.equilibrium import fixed_point
from nova.equilibrium.stencil_mesh import MomentGeometry, StencilMesh, ring_condition
from nova.frame.coilset import CoilSet
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.oracle_rebaseline import measure as recovery


OUTPUT = Path(__file__).resolve().parent
REPOSITORY_ROOT = OUTPUT.parents[1]
GATE_RESULTS = REPOSITORY_ROOT / "scripts/oracle_rebaseline/results.json"
BATCH_SIZES = (1, 4, 16)
TIMING_SAMPLES = 7


def _write(path: Path, payload: dict[str, object]) -> None:
    """Write strict, stable JSON."""

    def strict(value):
        if isinstance(value, dict):
            return {key: strict(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [strict(item) for item in value]
        if isinstance(value, np.generic):
            return strict(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _array_digest(values: np.ndarray) -> str:
    import hashlib

    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _machine_identity(machine: fixture.OracleMachine) -> dict[str, str]:
    return {
        name: _array_digest(value)
        for name, value in fixture._machine_arrays(machine).items()
    }


def _profiled_build(name: str) -> tuple[fixture.OracleMachine, dict[str, object]]:
    """Reproduce one production carrier with additive stage timers."""
    case = fixture.analytic_case()
    requested_cells = fixture.FIXTURE_REQUESTS[name]
    stage: dict[str, float] = {}
    family: dict[str, float] = {}

    started = perf_counter()
    section_started = perf_counter()
    wall = fixture.limiter_contour(case, points=fixture.WALL_POINT_COUNT)
    coilset = CoilSet(dplasma=requested_cells, tplasma="hex")
    coilset.firstwall.insert(wall, turn="hex")
    plasma = np.asarray(coilset.subframe.loc[:, "plasma"], dtype=bool)
    material = np.asarray(coilset.subframe.loc[:, "poly"], dtype=object)[plasma]
    centres = np.c_[
        np.asarray(coilset.subframe.loc[plasma, "x"], dtype=np.float64),
        np.asarray(coilset.subframe.loc[plasma, "z"], dtype=np.float64),
    ]
    polygons = tuple(
        fixture._clean_vertices(np.asarray(item.poly.exterior.coords)[:-1, :2])
        for item in material
    )
    area = np.asarray([item.poly.area for item in material], dtype=np.float64)
    stage["section_shape_seconds"] = perf_counter() - section_started

    assembly_started = perf_counter()
    triangulation = Delaunay(centres)
    boundary = LineString(wall)
    boundary_cells = np.asarray(
        [
            position
            for position, item in enumerate(material)
            if item.poly.intersects(boundary)
        ]
    )
    stencil, _ = PlasmaGrid.loop_neighbour_vertices(
        centres, triangulation.vertex_neighbor_vertices, boundary_cells
    )
    mesh = StencilMesh(centres, stencil, area)
    sections = np.asarray(coilset.aloc["plasma", "section"], dtype=object).astype(str)
    full = np.flatnonzero(sections == "hexagon")
    if len(full) == 0:
        raise ValueError("the oracle mesh has no complete hex generator")
    dimensions = np.c_[
        np.asarray(coilset.aloc["plasma", "dl"], dtype=float)[full],
        np.asarray(coilset.aloc["plasma", "dt"], dtype=float)[full],
    ]
    width, height = dimensions[0]
    radius = min(width / 2.0, height / np.sqrt(3.0))
    angles = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
    offsets = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    sampling = centres[:, None, :] + offsets[None, :, :]
    condition = ring_condition(centres, stencil)
    regular = np.asarray([len(polygon) == 6 for polygon in polygons])
    interior_stencil = stencil[(condition < 1.0e3) & regular[stencil].all(axis=1)]
    assembly_before_moment = perf_counter() - assembly_started

    moment_started = perf_counter()
    geometry = MomentGeometry.from_cells(mesh, polygons, sampling_vertices=sampling)
    stage["moment_order_seconds"] = perf_counter() - moment_started
    sample = geometry.sample_node_coordinates

    kernel_started = perf_counter()
    for target_name, targets in (
        ("grid", centres),
        ("wall", wall),
        ("sample", sample),
    ):
        target_started = perf_counter()
        blocks = fixture._flux_blocks(targets, polygons, geometry.atomic_mesh.centroids)
        family[target_name] = perf_counter() - target_started
        if target_name == "grid":
            grid_blocks = blocks
        elif target_name == "wall":
            wall_blocks = blocks
        else:
            sample_blocks = blocks
    stage["kernel_family_seconds"] = perf_counter() - kernel_started

    final_started = perf_counter()
    machine = fixture.OracleMachine(
        node=centres,
        area=area,
        cell_polygons=polygons,
        stencil=stencil,
        interior_stencil=interior_stencil,
        wall_node=wall,
        sampling_vertices=sampling,
        sample_coordinates=sample,
        plasma_to_grid=grid_blocks[0],
        plasma_to_grid_r=grid_blocks[1],
        plasma_to_grid_z=grid_blocks[2],
        plasma_to_wall=wall_blocks[0],
        plasma_to_wall_r=wall_blocks[1],
        plasma_to_wall_z=wall_blocks[2],
        plasma_to_sample=sample_blocks[0],
        plasma_to_sample_r=sample_blocks[1],
        plasma_to_sample_z=sample_blocks[2],
    )
    stage["assembly_seconds"] = assembly_before_moment + (
        perf_counter() - final_started
    )
    total = perf_counter() - started
    stage_sum = sum(stage.values())

    warm = fixture.cached_machine(
        case, requested_cells, wall_nodes=fixture.WALL_POINT_COUNT
    )
    built_identity = _machine_identity(machine)
    warm_identity = _machine_identity(warm)
    mismatches = sorted(
        name for name in built_identity if built_identity[name] != warm_identity[name]
    )
    if mismatches:
        raise AssertionError(
            f"profiled build differs from production cache: {mismatches}"
        )
    receipt = {
        "fixture": name,
        "requested_cells": requested_cells,
        "realised_cells": len(machine.node),
        "cold_direct_build": True,
        "cache_used_for_timed_build": False,
        "production_cache_identity_verified": True,
        "arrays_verified": len(built_identity),
        "stage_seconds": stage,
        "kernel_target_family_seconds": family,
        "total_seconds": total,
        "stage_sum_seconds": stage_sum,
        "additive_closure_seconds": total - stage_sum,
        "additive_closure_fraction": (total - stage_sum) / total,
        "stage_definitions": {
            "section_shape": "authored wall, hex tiling, polygons and areas",
            "moment_order": "fixed atomic support and first-moment geometry",
            "kernel_family": "exact authored-polygon zeroth and first flux blocks",
            "assembly": "triangulation, stencils, sampling layout and carrier packing",
        },
    }
    return machine, receipt


def profile_build(name: str, output: Path) -> None:
    configure_dtypes()
    _, receipt = _profiled_build(name)
    _write(output, receipt)
    print(
        f"BUILD_PROFILE fixture={name} cells={receipt['realised_cells']} "
        f"total_s={receipt['total_seconds']:.9g} "
        f"kernel_s={receipt['stage_seconds']['kernel_family_seconds']:.9g}",
        flush=True,
    )


def _prepare_operator(name: str):
    case = fixture.analytic_case()
    machine = fixture.cached_machine(
        case, fixture.FIXTURE_REQUESTS[name], wall_nodes=fixture.WALL_POINT_COUNT
    )
    coordinates = np.vstack(
        [machine.node, machine.wall_node, machine.sample_coordinates]
    )
    oracle = fixture.exact_state(case, coordinates)
    empty = fixture.forward_operator(case, machine)
    exact_physical = fixture.exact_current_moments(case, empty, oracle)
    exact_coefficients = empty.coupling_current_moments(exact_physical)
    exterior = oracle - fixture._internal_flux_image(empty, exact_coefficients)
    operator = fixture.forward_operator(case, machine, exterior)
    seed, _image, seed_receipt = recovery._moment_seed(case, machine, operator)
    return case, machine, oracle, operator, seed, seed_receipt


def _timed(compiled, *arguments, samples: int = TIMING_SAMPLES) -> dict[str, object]:
    jax.block_until_ready(compiled(*arguments))
    elapsed = []
    for _ in range(samples):
        started = perf_counter()
        jax.block_until_ready(compiled(*arguments))
        elapsed.append(perf_counter() - started)
    values = np.asarray(elapsed)
    return {
        "samples_seconds": elapsed,
        "median_seconds": float(np.median(values)),
        "minimum_seconds": float(np.min(values)),
        "maximum_seconds": float(np.max(values)),
    }


def _trace_call(path: Path, label: str, compiled, *arguments) -> dict[str, object]:
    """Trace one already-compiled device phase into an isolated profile."""
    path.mkdir(parents=True, exist_ok=True)
    options = jax.profiler.ProfileOptions()
    options.enable_hlo_proto = False
    options.host_tracer_level = 2
    options.python_tracer_level = 0
    with jax.profiler.trace(
        str(path), create_perfetto_trace=True, profiler_options=options
    ):
        with jax.profiler.TraceAnnotation(label):
            jax.block_until_ready(compiled(*arguments))
    for carrier in path.rglob("*.xplane.pb"):
        carrier.unlink()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    return {
        "label": label,
        "directory": str(path.relative_to(REPOSITORY_ROOT)),
        "files": [
            {
                "path": str(item.relative_to(REPOSITORY_ROOT)),
                "bytes": item.stat().st_size,
            }
            for item in files
        ],
        "total_bytes": sum(item.stat().st_size for item in files),
    }


def _full_solve(map_fn, state):
    return fixed_point.newton_krylov(
        map_fn,
        state,
        newton_steps=recovery.NEWTON_STEPS,
        gmres_iterations=recovery.KRYLOV_ITERATIONS,
        warmup=0,
    )


def _one_newton(map_fn, state):
    mapped, tangent = jax.linearize(map_fn, state)
    forcing = mapped - state
    step, _info = jax.scipy.sparse.linalg.gmres(
        lambda vector: vector - tangent(vector),
        forcing,
        maxiter=recovery.KRYLOV_ITERATIONS,
        restart=recovery.KRYLOV_ITERATIONS,
        solve_method="batched",
    )
    return state + step


def _hlo_loops(text: str) -> dict[str, object]:
    """Extract StableHLO loop-bound evidence without source-code inference."""
    lines = text.splitlines()
    loops = []
    for index, line in enumerate(lines):
        if "stablehlo.while" not in line:
            continue
        context = "\n".join(lines[index : min(index + 160, len(lines))])
        constants = sorted(
            {int(value) for value in re.findall(r"dense<([0-9]+)>", context)}
        )
        loops.append(
            {
                "line": index + 1,
                "integer_constants_in_context": constants,
                "context": context[:12000],
            }
        )
    return {
        "stablehlo_while_count": len(loops),
        "loops": loops,
        "required_newton_trip_count_observed": any(
            recovery.NEWTON_STEPS in row["integer_constants_in_context"]
            for row in loops
        ),
        "required_krylov_trip_count_observed": any(
            recovery.KRYLOV_ITERATIONS in row["integer_constants_in_context"]
            for row in loops
        ),
    }


def _accuracy(case, machine, oracle, operator, result) -> dict[str, object]:
    state = np.asarray(result.state)
    mapped = np.asarray(operator.flux_map()(result.state))
    grid_count = operator.grid.node_number
    span = fixture.TOTAL_FLUX_FACTOR * case.axis_flux
    difference = state[:grid_count] - oracle[:grid_count]
    masks, topology = operator.read(result.state)
    current = float(jnp.sum(operator.cell_current(result.state)))
    return {
        "fixed_point_residual": float(result.residual),
        "map_residual_sup_wb": float(np.max(np.abs(mapped - state))),
        "flux_sup_fraction_of_span": float(np.max(np.abs(difference)) / span),
        "flux_rms_fraction_of_span": float(np.sqrt(np.mean(difference**2)) / span),
        "axis_position_m": float(
            np.linalg.norm(np.asarray(topology.axis) - np.asarray(case.magnetic_axis))
        ),
        "plasma_current_fraction": float(
            abs(current - case.plasma_current()) / abs(case.plasma_current())
        ),
        "plasma_current_a": current,
        "topology_class": "diverted" if bool(topology.diverted) else "limited",
        "core_cell_count": int(np.count_nonzero(np.asarray(masks.core))),
    }


def _gate_margins(accuracy: dict[str, object]) -> dict[str, object]:
    registry = json.loads(GATE_RESULTS.read_text(encoding="utf-8"))["gate_registry"]
    margins = {}
    for name in (
        "fixed_point_residual",
        "flux_sup_fraction_of_span",
        "flux_rms_fraction_of_span",
        "axis_position_m",
        "plasma_current_fraction",
    ):
        value = float(accuracy[name])
        bound = float(registry[name]["proposed_bound"])
        margins[name] = {
            "value": value,
            "proposed_bound": bound,
            "owner_locked": False,
            "pass": value <= bound,
            "signed_margin": bound - value,
            "bound_over_value": bound / max(value, np.finfo(float).tiny),
        }
    expected = registry["topology_class"]["proposed_bound"]
    margins["topology_class"] = {
        "value": accuracy["topology_class"],
        "proposed_bound": expected,
        "owner_locked": False,
        "pass": accuracy["topology_class"] == expected,
    }
    return margins


def _compile_solve(map_fn, seed: jax.Array, batch_size: int):
    batched_seed = jnp.broadcast_to(seed, (batch_size, seed.shape[0]))
    solve = jax.vmap(lambda state: _full_solve(map_fn, state))
    lowered = jax.jit(solve).lower(batched_seed)
    return lowered, lowered.compile(), batched_seed


def profile_gpu(name: str, output: Path, batches: tuple[int, ...]) -> None:
    configure_dtypes()
    case, machine, oracle, production, seed_host, seed_receipt = _prepare_operator(name)
    seed = jax.device_put(jnp.asarray(seed_host))
    map_fn = production.flux_map()
    device = jax.devices()[0]
    if device.platform != "gpu":
        raise RuntimeError(f"GPU profile requires a GPU device, got {device}")

    map_compiled = jax.jit(map_fn).lower(seed).compile()
    vector = jnp.ones_like(seed)

    def tangent_fn(state, direction):
        return jax.jvp(map_fn, (state,), (direction,))[1]

    tangent_compiled = jax.jit(tangent_fn).lower(seed, vector).compile()
    krylov_compiled = (
        jax.jit(lambda state: _one_newton(map_fn, state)).lower(seed).compile()
    )
    krylov_matvec_compiled = (
        jax.jit(lambda state, direction: direction - tangent_fn(state, direction))
        .lower(seed, vector)
        .compile()
    )
    identity_compiled = jax.jit(lambda value: value).lower(seed).compile()

    phases = {
        "map_evaluation": _timed(map_compiled, seed, samples=20),
        "tangent_action": _timed(tangent_compiled, seed, vector, samples=20),
        "krylov_one_newton_inclusive": _timed(krylov_compiled, seed, samples=10),
        "launch_identity": _timed(identity_compiled, seed, samples=30),
    }
    transfer_samples = []
    host = np.asarray(seed_host)
    for _ in range(20):
        started = perf_counter()
        moved = jax.device_put(host)
        moved.block_until_ready()
        to_device = perf_counter() - started
        started = perf_counter()
        np.asarray(moved)
        to_host = perf_counter() - started
        transfer_samples.append((to_device, to_host))
    phases["transfer"] = {
        "host_to_device_median_seconds": float(
            np.median([row[0] for row in transfer_samples])
        ),
        "device_to_host_median_seconds": float(
            np.median([row[1] for row in transfer_samples])
        ),
        "samples_seconds": transfer_samples,
    }

    timings = {}
    hlo_receipt = None
    production_result = None
    batch_one_compiled = None
    batch_one_seed = None
    for batch_size in batches:
        compile_started = perf_counter()
        lowered, compiled, batched_seed = _compile_solve(map_fn, seed, batch_size)
        compile_seconds = perf_counter() - compile_started
        timing = _timed(compiled, batched_seed, samples=TIMING_SAMPLES)
        result = compiled(batched_seed)
        result.state.block_until_ready()
        if batch_size == 1:
            production_result = jax.tree.map(lambda value: value[0], result)
            batch_one_compiled = compiled
            batch_one_seed = batched_seed
            stablehlo = lowered.compiler_ir(dialect="stablehlo")
            hlo_text = stablehlo.operation.get_asm(
                enable_debug_info=False, large_elements_limit=16
            )
            hlo_path = OUTPUT / f"solve-{name}-stablehlo.txt.gz"
            with gzip.open(hlo_path, "wt", encoding="utf-8") as stream:
                stream.write(hlo_text)
            hlo_receipt = {
                "artifact": str(hlo_path.relative_to(REPOSITORY_ROOT)),
                "compressed_bytes": hlo_path.stat().st_size,
                **_hlo_loops(hlo_text),
            }
        timings[str(batch_size)] = {
            **timing,
            "compile_seconds": compile_seconds,
            "per_state_median_seconds": timing["median_seconds"] / batch_size,
            "states_per_second": batch_size / timing["median_seconds"],
            "target_seconds_per_state": 1.0e-3,
            "multiple_of_target": timing["median_seconds"] / batch_size / 1.0e-3,
        }

    if any(
        item is None
        for item in (
            production_result,
            hlo_receipt,
            batch_one_compiled,
            batch_one_seed,
        )
    ):
        raise ValueError("batch sizes must include one")

    traces = {
        "map_evaluation": _trace_call(
            OUTPUT / f"trace-{name}-map", "map_evaluation", map_compiled, seed
        ),
        "tangent_action": _trace_call(
            OUTPUT / f"trace-{name}-tangent",
            "tangent_action",
            tangent_compiled,
            seed,
            vector,
        ),
        "krylov_inner_matvec": _trace_call(
            OUTPUT / f"trace-{name}-krylov",
            "krylov_inner_matvec",
            krylov_matvec_compiled,
            seed,
            vector,
        ),
    }

    production_accuracy = _accuracy(
        case, machine, oracle, production, production_result
    )
    production_forcing = float(
        np.max(np.abs(np.asarray(map_fn(jnp.asarray(oracle))) - oracle))
    )
    production_accuracy["standing_forcing_sup_wb"] = production_forcing

    rungs = {
        f"{name}_first_moment": {
            "fixture": name,
            "realised_cells": len(machine.node),
            "moment_order": 1,
            "density_degree": 9,
            "accuracy": production_accuracy,
            "proposed_gate_margins": _gate_margins(production_accuracy),
            "batch_one_solve": timings["1"],
        }
    }
    if name == "coarse":
        centroid = replace(production, use_linear_moments=False)
        centroid_map = centroid.flux_map()
        lowered, compiled, batched_seed = _compile_solve(centroid_map, seed, 1)
        centroid_timing = _timed(compiled, batched_seed, samples=TIMING_SAMPLES)
        centroid_result = compiled(batched_seed)
        centroid_result.state.block_until_ready()
        centroid_result = jax.tree.map(lambda value: value[0], centroid_result)
        centroid_accuracy = _accuracy(case, machine, oracle, centroid, centroid_result)
        centroid_accuracy["standing_forcing_sup_wb"] = float(
            np.max(np.abs(np.asarray(centroid_map(jnp.asarray(oracle))) - oracle))
        )
        rungs["coarse_centroid"] = {
            "fixture": name,
            "realised_cells": len(machine.node),
            "moment_order": 0,
            "density_degree": 9,
            "accuracy": centroid_accuracy,
            "proposed_gate_margins": _gate_margins(centroid_accuracy),
            "batch_one_solve": {
                **centroid_timing,
                "per_state_median_seconds": centroid_timing["median_seconds"],
                "states_per_second": 1.0 / centroid_timing["median_seconds"],
                "target_seconds_per_state": 1.0e-3,
                "multiple_of_target": centroid_timing["median_seconds"] / 1.0e-3,
            },
        }

    receipt = {
        "fixture": name,
        "device": {
            "platform": device.platform,
            "device_kind": device.device_kind,
            "jax_backend": jax.default_backend(),
            "jax_version": jax.__version__,
        },
        "solver": {
            "route": "production moment-seeded fixed-shape newton-krylov",
            "newton_steps": recovery.NEWTON_STEPS,
            "gmres_iterations": recovery.KRYLOV_ITERATIONS,
            "warmup": 0,
            "criterion": recovery.SOLVER_CRITERION,
            "seed": seed_receipt,
        },
        "batch_timings": timings,
        "phase_probes": phases,
        "phase_accounting": (
            "independent inclusive device probes; Krylov includes tangent actions, "
            "so phase probes are diagnostic and are not summed"
        ),
        "compiled_hlo": hlo_receipt,
        "trace_artifacts": traces,
        "accuracy_rungs": rungs,
    }
    _write(output, receipt)
    print(
        f"GPU_PROFILE fixture={name} device={device.device_kind} "
        f"batch1_ms={1e3 * timings['1']['per_state_median_seconds']:.9g} "
        f"residual={production_accuracy['fixed_point_residual']:.9g}",
        flush=True,
    )


def profile_trace(name: str, output: Path) -> None:
    """Bank isolated phase traces without recompiling the full fixed-point solve."""
    configure_dtypes()
    _case, _machine, _oracle, operator, seed_host, _seed_receipt = _prepare_operator(
        name
    )
    seed = jax.device_put(jnp.asarray(seed_host))
    vector = jnp.ones_like(seed)
    map_fn = operator.flux_map()

    def tangent_fn(state, direction):
        return jax.jvp(map_fn, (state,), (direction,))[1]

    map_compiled = jax.jit(map_fn).lower(seed).compile()
    tangent_compiled = jax.jit(tangent_fn).lower(seed, vector).compile()
    krylov_matvec_compiled = (
        jax.jit(lambda state, direction: direction - tangent_fn(state, direction))
        .lower(seed, vector)
        .compile()
    )
    traces = {
        "map_evaluation": _trace_call(
            OUTPUT / f"trace-{name}-map", "map_evaluation", map_compiled, seed
        ),
        "tangent_action": _trace_call(
            OUTPUT / f"trace-{name}-tangent",
            "tangent_action",
            tangent_compiled,
            seed,
            vector,
        ),
        "krylov_inner_matvec": _trace_call(
            OUTPUT / f"trace-{name}-krylov",
            "krylov_inner_matvec",
            krylov_matvec_compiled,
            seed,
            vector,
        ),
    }
    receipt = {
        "fixture": name,
        "device": jax.devices()[0].device_kind,
        "traces": traces,
        "krylov_total_trip_count_source": "compiled StableHLO receipt",
    }
    _write(output, receipt)
    print(
        f"GPU_TRACE fixture={name} total_bytes="
        f"{sum(row['total_bytes'] for row in traces.values())}",
        flush=True,
    )


def _device_feasibility(
    builds: dict[str, object], gpu: dict[str, object]
) -> dict[str, object]:
    rows = {}
    for name, receipt in builds.items():
        stages = receipt["stage_seconds"]
        total = receipt["total_seconds"]
        eligible = stages["kernel_family_seconds"] + stages["moment_order_seconds"]
        map_seconds = gpu[name]["phase_probes"]["map_evaluation"]["median_seconds"]
        full_blocks_per_map = 9
        rows[name] = {
            "projected_device_side_seconds": eligible,
            "projected_device_side_share": eligible / total,
            "host_only_shape_and_assembly_seconds": total - eligible,
            "h200_one_block_throughput_estimate": {
                "full_moment_blocks_per_second": full_blocks_per_map / map_seconds,
                "seconds_per_full_moment_block": map_seconds / full_blocks_per_map,
                "block_definition": (
                    "one dense source-to-target matrix for one of three target "
                    "families and one of three moment orders"
                ),
                "basis": (
                    "the measured H200 map contracts nine full blocks and also pays "
                    "topology, clipping and density work; dividing its inclusive map "
                    "time by nine is a conservative contraction-side feasibility "
                    "estimate, not a measured device build kernel"
                ),
            },
            "basis": (
                "exact kernels and fixed moment geometry are array-block work; "
                "authored shape construction and host carrier assembly remain host work"
            ),
        }
    coarse = builds["coarse"]
    pairs = coarse["realised_cells"] * (
        coarse["realised_cells"] + fixture.WALL_POINT_COUNT
    )
    kernel_seconds = coarse["stage_seconds"]["kernel_family_seconds"]
    return {
        "status": "estimate_only_no_port_attempted",
        "by_fixture": rows,
        "one_block_throughput_estimate": {
            "fixture": "coarse",
            "conservative_grid_plus_wall_pairs": pairs,
            "host_measured_pairs_per_second": pairs / kernel_seconds,
            "definition": (
                "source-target polygon moment blocks per measured CPU kernel second; "
                "sample-family work omitted, making this a conservative host "
                "denominator "
                "rather than a GPU claim"
            ),
            "device_projection": (
                "requires a device-native exact polygon kernel; no speedup factor is "
                "assumed"
            ),
        },
    }


def finalize(output: Path) -> None:
    builds = {
        name: json.loads((OUTPUT / f"build-{name}.json").read_text(encoding="utf-8"))
        for name in ("coarse", "fine")
    }
    gpu = {
        name: json.loads((OUTPUT / f"gpu-{name}.json").read_text(encoding="utf-8"))
        for name in ("coarse", "fine")
    }
    for name, receipt in gpu.items():
        trace_receipt = OUTPUT / f"trace-{name}.json"
        if trace_receipt.is_file():
            receipt.pop("trace_artifact", None)
            receipt["trace_artifacts"] = json.loads(
                trace_receipt.read_text(encoding="utf-8")
            )["traces"]
        phases = receipt["phase_probes"]
        full_solve = receipt["batch_timings"]["1"]["per_state_median_seconds"]
        map_seconds = phases["map_evaluation"]["median_seconds"]
        newton_seconds = phases["krylov_one_newton_inclusive"]["median_seconds"]
        receipt["phase_decomposition"] = {
            "map_evaluation_seconds": map_seconds,
            "single_tangent_action_seconds": phases["tangent_action"]["median_seconds"],
            "krylov_inner_loop_seconds_per_newton_inclusive_of_tangents": (
                max(newton_seconds - map_seconds, 0.0)
            ),
            "one_newton_seconds_inclusive": newton_seconds,
            "ten_newton_projection_seconds": recovery.NEWTON_STEPS * newton_seconds,
            "measured_full_solve_seconds": full_solve,
            "projection_over_measured_full_solve": (
                recovery.NEWTON_STEPS * newton_seconds / full_solve
            ),
            "launch_seconds": phases["launch_identity"]["median_seconds"],
            "host_to_device_seconds": phases["transfer"][
                "host_to_device_median_seconds"
            ],
            "device_to_host_seconds": phases["transfer"][
                "device_to_host_median_seconds"
            ],
            "accounting": (
                "map is the Newton primal; the Krylov inner row subtracts that "
                "primal from the inclusive one-Newton probe and retains all tangent "
                "actions; launch and transfers are separate probes"
            ),
        }
    rungs = {}
    for receipt in gpu.values():
        rungs.update(receipt["accuracy_rungs"])
    for rung in rungs.values():
        build = builds[rung["fixture"]]
        if rung["moment_order"] == 0:
            rung["cpu_build_seconds"] = (
                build["stage_seconds"]["section_shape_seconds"]
                + build["stage_seconds"]["assembly_seconds"]
                + build["stage_seconds"]["kernel_family_seconds"] / 3.0
            )
            rung["cpu_build_cost_status"] = (
                "projected from measured production stages with one of three moment "
                "blocks"
            )
        else:
            rung["cpu_build_seconds"] = build["total_seconds"]
            rung["cpu_build_cost_status"] = "measured cold"
        margins = rung["proposed_gate_margins"]
        rung["proposed_gate_summary"] = {
            "passed": sum(bool(row["pass"]) for row in margins.values()),
            "total": len(margins),
            "all_pass": all(bool(row["pass"]) for row in margins.values()),
            "bounds_owner_locked": False,
        }
        if rung["accuracy"]["fixed_point_residual"] is None:
            rung["accuracy_verdict"] = (
                "rejected_nonfinite_root; cheaper centroid solve did not converge"
            )
        elif rung["proposed_gate_summary"]["all_pass"]:
            rung["accuracy_verdict"] = "passes_all_proposed_gates"
        else:
            rung["accuracy_verdict"] = (
                "scientific_hold; fixed-point residual passes but alternate-root "
                "recovery gates fail"
            )

    ordered = ["coarse_centroid", "coarse_first_moment", "fine_first_moment"]
    figure_path = OUTPUT / "accuracy-cost-ladder.png"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for key in ordered:
        rung = rungs[key]
        label = key.replace("_", " ")
        axes[0].scatter(
            rung["cpu_build_seconds"],
            rung["accuracy"]["flux_sup_fraction_of_span"],
            s=55,
            label=label,
        )
        axes[1].scatter(
            1e3 * rung["batch_one_solve"]["per_state_median_seconds"],
            rung["accuracy"]["flux_sup_fraction_of_span"],
            s=55,
            label=label,
        )
    axes[0].set(xlabel="cold CPU build [s]", ylabel="root flux sup / analytic span")
    axes[1].set(xlabel="H200 solve per state [ms]")
    axes[1].axvline(1.0, color="0.25", linestyle="--", linewidth=1, label="1 ms target")
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(alpha=0.2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    phase_figure_path = OUTPUT / "h200-phase-profile.png"
    labels = ("map", "tangent", "Krylov inner", "full solve")
    locations = np.arange(len(labels))
    width = 0.34
    fig, axis = plt.subplots(figsize=(7.8, 4.2))
    for offset, name in zip((-width / 2, width / 2), ("coarse", "fine"), strict=True):
        phase = gpu[name]["phase_decomposition"]
        values = 1e3 * np.asarray(
            [
                phase["map_evaluation_seconds"],
                phase["single_tangent_action_seconds"],
                phase["krylov_inner_loop_seconds_per_newton_inclusive_of_tangents"],
                phase["measured_full_solve_seconds"],
            ]
        )
        axis.bar(locations + offset, values, width=width, label=name)
    axis.axhline(1.0, color="0.25", linestyle="--", linewidth=1)
    axis.set(
        xticks=locations,
        xticklabels=labels,
        ylabel="H200 wall time [ms]",
        yscale="log",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(phase_figure_path, dpi=180)
    plt.close(fig)

    report = {
        "schema": "accuracy-cost-ladder",
        "oracle": "closed-form moderate-rotation-conventional",
        "bounds": {
            "status": "proposed_owner_lock_required",
            "moved_applied_or_widened": False,
            "source": "scripts/oracle_rebaseline/results.json",
        },
        "cpu_cold_builds": builds,
        "device_kernel_build_feasibility": _device_feasibility(builds, gpu),
        "h200": gpu,
        "accuracy_cost_rungs": {key: rungs[key] for key in ordered},
        "runtime_target": {
            "seconds_per_state": 1.0e-3,
            "coarse_batch_one_measured_seconds": gpu["coarse"]["batch_timings"]["1"][
                "per_state_median_seconds"
            ],
            "coarse_batch_one_multiple": gpu["coarse"]["batch_timings"]["1"][
                "multiple_of_target"
            ],
            "verdict": "measured_against_target_not_locked",
        },
        "recommendation": {
            "status": "owner_decision_required",
            "operating_point": "coarse_first_moment",
            "reason": (
                "production first moments preserve the exact-state round-off forcing; "
                "the alternate-root gate failures remain a basin-selection HOLD rather "
                "than an accuracy-cost trade"
            ),
        },
        "artifacts": {
            "figures": [
                str(figure_path.relative_to(REPOSITORY_ROOT)),
                str(phase_figure_path.relative_to(REPOSITORY_ROOT)),
            ],
            "run_logs": [
                "scripts/accuracy_cost_ladder/gpu-device-probe-resumed.log",
                "scripts/accuracy_cost_ladder/build-coarse-resumed.log",
                "scripts/accuracy_cost_ladder/build-fine-resumed.log",
                "scripts/accuracy_cost_ladder/gpu-coarse-profile-resumed.log",
                "scripts/accuracy_cost_ladder/gpu-coarse-trace-pruned.log",
                "scripts/accuracy_cost_ladder/gpu-fine-resumed.log",
                "scripts/accuracy_cost_ladder/finalize.log",
            ],
        },
    }
    _write(output, report)
    print(f"FINALIZED output={output} figure={figure_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("profile-build")
    build.add_argument("--fixture", choices=fixture.FIXTURE_REQUESTS, required=True)
    build.add_argument("--output", type=Path, required=True)
    gpu = subparsers.add_parser("profile-gpu")
    gpu.add_argument("--fixture", choices=fixture.FIXTURE_REQUESTS, required=True)
    gpu.add_argument("--output", type=Path, required=True)
    gpu.add_argument("--batches", default="1,4,16")
    trace = subparsers.add_parser("profile-trace")
    trace.add_argument("--fixture", choices=fixture.FIXTURE_REQUESTS, required=True)
    trace.add_argument("--output", type=Path, required=True)
    finish = subparsers.add_parser("finalize")
    finish.add_argument("--output", type=Path, default=OUTPUT / "results.json")
    args = parser.parse_args()
    if args.command == "profile-build":
        profile_build(args.fixture, args.output)
    elif args.command == "profile-gpu":
        batches = tuple(int(item) for item in args.batches.split(","))
        profile_gpu(args.fixture, args.output, batches)
    elif args.command == "profile-trace":
        profile_trace(args.fixture, args.output)
    else:
        finalize(args.output)


if __name__ == "__main__":
    main()
