"""Measure full-read Picard maps from certified analytic moment rows."""

# ruff: noqa: E402
import os
import sys

sys.path.insert(0, os.environ["NOVA_HELD_ROOT"])

from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import nova
from scipy.spatial import cKDTree

from benchmarks import solovev_certificate as certificate
from nova.database.zarrstore import ZarrStore
from nova.equilibrium import current, fixed_points, gs, support_read, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def _stored(cache, filename, key):
    store = ZarrStore(filename=filename, dirname=cache, group=key)
    store.load()
    return store


def _point_booking(psi, geometry, profile, reading, sigma, target, smooth):
    coefficients = jnp.einsum("nij,nj->ni", geometry.fit_inverse, psi)
    value = coefficients[:, 0]
    signed = sigma * (value - reading.boundary_flux)
    width = jnp.maximum(jnp.linalg.norm(coefficients[:, 1:3], axis=-1), 1e-14)
    membership = (
        jax.nn.sigmoid(signed / width) if smooth else (signed > 0).astype(value.dtype)
    )
    density = profile.current_density(
        geometry.centre[:, 0],
        (value - reading.axis_flux) / (reading.boundary_flux - reading.axis_flux),
    )
    raw = geometry.full_area * density * membership
    booked = raw * (target / jnp.sum(raw))
    return jnp.stack((booked, jnp.zeros_like(booked), jnp.zeros_like(booked)), axis=-1)


def _setup(kind, cells, cache, bias_row, newton_row):
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, exact = certificate._case(name, clip_mode="exact")
    booking_receipt = json.loads(bias_row.read_text())
    newton_receipt = json.loads(newton_row.read_text())
    machine_key = booking_receipt["machine_key"]
    machine_store = _stored(cache, f"{fixture.CACHE_FILENAME}_{cells}", machine_key)
    identity = json.loads(machine_store.data.attrs["semantic_identity"])
    assert identity["analytic_case"] == carrier.name
    machine = fixture._from_dataset(machine_store.data, identity, machine_key)
    exterior_store = _stored(
        cache, fixture.EXTERIOR_CACHE_FILENAME, booking_receipt["exterior_key"]
    )
    physical = jnp.stack(
        tuple(
            jnp.asarray(np.asarray(exterior_store.data[name]))
            for name in ("cell_current", "radial_moment", "vertical_moment")
        ),
        axis=-1,
    )
    exterior_flux = np.asarray(exterior_store.data["exterior"])
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    samples = np.asarray(geometry.sample_points)
    nodes, inverse = np.unique(
        np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
    )
    row_path = Path(newton_receipt["exterior"]["row_cache_path"])
    if not row_path.is_file():
        raise FileNotFoundError(f"certified jet rows absent: {row_path}")
    rows = np.load(row_path, mmap_mode="r", allow_pickle=False)
    assert rows.shape == (len(nodes), len(machine.node), 3, 6)
    _, field, _, axis, saddle = _analytic_inputs(kind)
    field = certificate_field(field, kind)
    roots = jnp.asarray([axis] if saddle is None else [axis, saddle])
    closed = jax.jit(
        lambda field, nodes: fixed_points.stack_jet(jax.vmap(field.evaluate)(nodes))
    )(field, jnp.asarray(nodes))
    rows = jnp.asarray(rows)
    exterior_jets = closed - jnp.einsum("ncmq,cm->nq", rows, physical)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    distance, indices = cKDTree(coordinates).query(nodes)
    shared = distance < 2e-11
    if not np.any(shared):
        raise ValueError("exterior value check found no shared target")
    actual = np.asarray(exterior_jets)[shared, 0]
    expected = exterior_flux[indices[shared]]
    relative_value_error = float(np.max(np.abs(actual - expected))) / max(
        float(np.max(np.abs(expected))), 1e-30
    )
    if relative_value_error > 1e-11:
        raise ValueError(f"exterior jet/value mismatch: {relative_value_error}")
    operands = support_read.SupportOperands.from_rows(
        geometry, rows, exterior_jets, nodes, inverse.reshape(samples.shape[:-1])
    )
    cache_window = support_read.row_window(operands, roots)
    target = jnp.sum(physical[:, 0])
    convention = topology.TopologyConvention.from_cocos(17, target)
    policy = TopologyPolicy()
    profile = fixture.analytic_profile(source)
    quadrature = current.Quadrature(sigma=convention.sigma, target_current=target)
    return (
        physical,
        operands,
        cache_window,
        roots,
        convention,
        policy,
        profile,
        quadrature,
        dict(
            machine_key=machine_key,
            row_path=str(row_path),
            row_shape=list(rows.shape),
            shared_targets=int(np.sum(shared)),
            relative_exterior_value_error=relative_value_error,
        ),
    )


def run(kind, cells, cache, bias_row, newton_row, out):
    if jax.default_backend() != "gpu" or jax.config.jax_enable_x64 is not True:
        raise RuntimeError("live-map measurement requires H200 double precision")
    (
        initial,
        operands,
        cache_window,
        roots,
        convention,
        policy,
        profile,
        quadrature,
        provenance,
    ) = _setup(kind, cells, cache, bias_row, newton_row)
    pitch = operands.geometry.pitch
    current_scale = jnp.maximum(jnp.max(jnp.abs(initial[:, 0])), 1e-30)
    scale = current_scale * jnp.stack((jnp.ones_like(pitch), pitch, pitch), axis=-1)

    def exact(state, operands, cache_window):
        residual, (support, booked) = gs.moment_residual(
            state,
            operands,
            cache_window,
            roots,
            convention,
            policy,
            profile,
            quadrature,
        )
        return state - residual, support.valid & booked.valid, support.stationary.status

    def point(state, operands, cache_window, smooth):
        moments = tuple(state[:, index] for index in range(3))
        jets = support_read.nodal_jets(operands, moments)
        support = support_read.read(
            operands,
            cache_window,
            moments,
            roots,
            roots,
            convention,
            policy,
            axis_sign=-convention.sigma,
            jets=jets,
        )
        psi = jets[operands.sample_index, 0]
        booked = _point_booking(
            psi,
            operands.geometry,
            profile,
            support.reading,
            convention.sigma,
            quadrature.target_current,
            smooth,
        )
        return booked, support.valid, support.stationary.status

    functions = {
        "exact": jax.jit(exact),
        "hard": jax.jit(lambda state, ops, window: point(state, ops, window, False)),
        "smooth": jax.jit(lambda state, ops, window: point(state, ops, window, True)),
    }
    results = {}
    for method, program in functions.items():
        for seed_name, seed in (
            ("analytic_perturbed", initial * 1.001),
            ("zero_current", jnp.zeros_like(initial)),
        ):
            state = seed
            history = []
            walls = []
            validity = []
            statuses = []
            started = perf_counter()
            for _ in range(20):
                before = perf_counter()
                next_state, valid, status = program(state, operands, cache_window)
                jax.block_until_ready((next_state, valid, status))
                walls.append((perf_counter() - before) * 1000)
                error = float(jnp.max(jnp.abs((next_state - state) / scale)))
                history.append(error)
                validity.append(bool(valid))
                statuses.append(np.asarray(status).tolist())
                state = next_state
                if not np.isfinite(error) or not bool(valid) or error < 1e-12:
                    break
            results[f"{method}_{seed_name}"] = dict(
                iterations=len(history),
                residual_history=history,
                valid_history=validity,
                stationary_status=statuses,
                converged=bool(validity[-1] and np.isfinite(error) and error < 1e-12),
                iteration_wall_ms=walls,
                total_wall_ms=(perf_counter() - started) * 1000,
            )
            print(
                f"ROW {method} {seed_name} {results[f'{method}_{seed_name}']}",
                flush=True,
            )
    result = dict(
        kind=kind,
        requested_cells=cells,
        realised_cells=len(initial),
        provenance=provenance,
        seed_basis="analytic moments scaled by 1.001 or zero current",
        rows=results,
        gpu_memory=jax.devices()[0].memory_stats(),
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{kind}-{cells}.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--bias-row", type=Path, required=True)
    parser.add_argument("--newton-row", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"revision={revision} tree={Path.cwd().resolve()}", flush=True)
    print(f"NOVA_MODULE={nova.__file__} GS_MODULE={gs.__file__}", flush=True)
    print(f"CWD={Path.cwd().resolve()} DEVICE={jax.devices()}", flush=True)
    run(args.kind, args.cells, args.cache, args.bias_row, args.newton_row, args.out)


if __name__ == "__main__":
    main()
