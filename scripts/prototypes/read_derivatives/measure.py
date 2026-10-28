"""Compare cheaper second-derivative paths for the topology read.

The topology read needs a nodal flux jet: value, gradient and symmetric Hessian
at every hex-mesh node. The reference oracle obtains the Hessian by
differentiating the analytic booked-moment kernel a second time, which is what
makes the compiled read program large. Two alternatives are measured against it:

``fd``
    central finite differences of the analytic gradient, differentiating the
    kernel once rather than twice;
``interp``
    a local quadratic least-squares fit of the nodal flux over each hex cell's
    seven-node stencil. The flux is one contraction of the cached coupling
    operand with the booked moments, so no kernel autodiff runs in the read.

Each variant is the full read program: it builds the nodal jet the read
consumes and returns the topology state. Its cold compile, instruction count
and warm per-call wall are recorded, and its axis, admitted X-point and
boundary flux are compared with the oracle's at the limited and diverted
certificate states.

Run on a GPU compute node; the process prints one JSON row per program.
"""

# Precision precedes imports with array-valued defaults.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.polygonanalytic import packed_analytic_moments
from nova.equilibrium import current, topology
from nova.equilibrium.clip_quadrature import ClippedCurrentMoments
from nova.equilibrium.solve_request import TopologyPolicy

assert jax.config.jax_enable_x64 is True
ROOT = Path(__file__).resolve().parents[3]
JET = 6


def value_gradient(point, coupling):
    """Analytic booked-moment value and gradient at one point."""
    rows = packed_analytic_moments(
        jnp,
        jnp.broadcast_to(point[0], coupling.norm.shape),
        jnp.broadcast_to(point[1], coupling.norm.shape),
        coupling.edge,
        coupling.weight,
        coupling.norm,
        coupling.centre.T,
        coupling.centre.T,
        coupling.reflection_axis,
        coupling.reflection_partner,
    )
    value = jnp.stack(rows[:3], axis=-1)
    gradient = (
        2
        * jnp.pi
        * point[0]
        * jnp.stack(
            (jnp.stack(rows[6:9], axis=-1), -jnp.stack(rows[3:6], axis=-1)),
            axis=-1,
        )
    )
    return value, gradient


def jets_oracle(point, coupling):
    """Value, gradient and Hessian from a second kernel differentiation."""
    value, gradient = value_gradient(point, coupling)
    hessian = jax.jacfwd(lambda target: value_gradient(target, coupling)[1])(point)
    return jnp.concatenate(
        (
            value[..., None],
            gradient,
            hessian[..., 0, 0, None],
            hessian[..., 0, 1, None],
            hessian[..., 1, 1, None],
        ),
        axis=-1,
    )


def jets_fd(point, coupling, step):
    """Value, analytic gradient and a differenced Hessian."""
    value, gradient = value_gradient(point, coupling)
    offset = jnp.asarray((step, 0.0))
    _, ahead = value_gradient(point + offset, coupling)
    _, behind = value_gradient(point - offset, coupling)
    _, up = value_gradient(point + jnp.asarray((0.0, step)), coupling)
    _, down = value_gradient(point - jnp.asarray((0.0, step)), coupling)
    dx = (ahead - behind) / (2 * step)
    dy = (up - down) / (2 * step)
    return jnp.concatenate(
        (value[..., None], gradient, dx[..., :1], dy[..., :1], dy[..., 1:2]),
        axis=-1,
    )


def jets_interp(flux, inverse, index, pitch):
    """Quadratic stencil fit of the nodal flux, differentiated analytically."""
    pressed = flux[index]
    coefficient = jnp.einsum("ckj,cj->ck", inverse, pressed)
    scale = pitch[:, None]
    gradient = jnp.stack((coefficient[:, 1], coefficient[:, 2]), -1) / scale
    second = coefficient[:, 3:6]
    hessian = jnp.stack(
        (
            jnp.stack((second[:, 0], second[:, 2]), -1),
            jnp.stack((second[:, 2], second[:, 1]), -1),
        ),
        -2,
    ) / (scale[:, :, None] * scale[:, None, :])
    jets = jnp.concatenate(
        (
            coefficient[:, :1],
            gradient,
            hessian[:, 0, 0, None],
            hessian[:, 0, 1, None],
            hessian[:, 1, 1, None],
        ),
        -1,
    )
    flat = index.reshape(-1)
    count = jnp.zeros(flux.shape[0]).at[flat].add(1.0)
    total = (
        jnp.zeros((flux.shape[0], JET))
        .at[flat]
        .add(jnp.repeat(jets, index.shape[1], axis=0))
    )
    return total / jnp.maximum(count, 1.0)[:, None]


def design(delta, pitch):
    """Seven-node scaled quadratic design matrix for one cell stencil."""
    x = delta.T[0] / pitch
    y = delta.T[1] / pitch
    return np.stack((np.ones_like(x), x, y, 0.5 * x * x, x * y, 0.5 * y * y), -1)


def timed(function, operands, label):
    """Cold compile, instruction count and warm per-call wall of one program."""
    started = perf_counter()
    lowered = jax.jit(function).lower(*operands)
    executable = lowered.compile()
    cold = perf_counter() - started
    text = lowered.as_text(dialect="hlo")
    instructions = sum(1 for line in text.splitlines() if " = " in line)
    walls = []
    for _ in range(3):
        started = perf_counter()
        jax.block_until_ready(executable(*operands))
        walls.append(perf_counter() - started)
    row = dict(
        program=label,
        cold_compile_seconds=cold,
        warm_seconds=min(walls),
        hlo_instructions=instructions,
    )
    print("DERIVATIVE_ROW " + json.dumps(row), flush=True)
    return row, executable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, default=550)
    parser.add_argument("--step", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    print(f"REVISION={revision} TREE={ROOT} KIND={args.kind}", flush=True)
    print(f"MODULE={topology.__file__} CWD={Path.cwd().resolve()}", flush=True)

    import tests.equilibrium.test_topology_read as fixture
    from benchmarks import solovev_certificate as certificate
    from scripts.analytic_oracle_fixtures.measure import analytic_profile
    from scripts.prototypes.read_route.jets import build_jet_matrix, read_field
    from scripts.prototypes.read_route.measure import sample_field

    oracle, analytic, wall, axis, saddle = fixture._analytic_inputs(args.kind)
    geometry = fixture._realised_hex_geometry(wall, args.cells)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy(
        normal_form_radius=0.054806712567833996 if saddle is not None else 0.0
    )
    analytic_read = jax.jit(topology.read)(analytic, geometry, convention, policy)
    jax.block_until_ready(analytic_read)
    sampled = sample_field(analytic, geometry)
    sampled_read = jax.jit(topology.read)(sampled, geometry, convention, policy)
    jax.block_until_ready(sampled_read)

    _, source, _ = certificate._case(
        "weak-rotation-reactor-static"
        if args.kind == "limited"
        else "diverted-single-null",
        clip_mode="exact",
    )
    profile = analytic_profile(source)
    psi = jax.vmap(jax.vmap(analytic.value))(geometry.sample_points)
    reference = fixture._kernel_backed_field(args.kind, oracle, analytic, geometry)
    coupling = reference.coupling
    reference_moments = reference.moments

    def book(read):
        return ClippedCurrentMoments(
            *current.book(
                psi,
                read,
                geometry,
                profile,
                current.Quadrature(sigma=convention.sigma, segments=64),
            )[:3]
        )

    converged_moments = book(analytic_read)
    trial_moments = book(sampled_read)

    matrix, exterior, nodes, sample_index, _build = build_jet_matrix(
        coupling, geometry, analytic, reference_moments, persist=False
    )
    jax.block_until_ready(matrix)
    pitch = geometry.pitch
    delta = jnp.asarray(geometry.sample_points) - jnp.asarray(geometry.centre)[:, None]
    inverse = jnp.asarray(
        np.linalg.pinv(
            np.stack(
                [
                    design(np.asarray(delta[c]), float(pitch[c]))
                    for c in range(len(geometry.centre))
                ]
            )
        )
    )
    value_operand = matrix[..., 0]
    index = jnp.asarray(sample_index)

    from scripts.prototypes.read_route.jets import HermiteField

    def run_prepared(jets):
        return read_field(
            HermiteField(jets, nodes, sample_index, geometry.centre, geometry.pitch),
            geometry,
            convention,
            policy,
            finite_curvature=True,
        )

    def oracle_batch(points, coupling):
        return jax.vmap(jets_oracle, in_axes=(0, None))(points, coupling)

    def fd_batch(points, coupling, step):
        return jax.vmap(jets_fd, in_axes=(0, None, None))(points, coupling, step)

    def interp_program(value, moments, inverse, index, pitch):
        return jets_interp(
            jnp.einsum("ncm,cm->n", value, moments), inverse, index, pitch
        )

    # The kernel derivative is evaluated over a node batch, as production builds
    # its operand; one call over the whole mesh materialises the point jet for
    # every node at once and exhausts device memory.
    batch = min(args.batch, len(nodes))
    point_batch = jnp.asarray(np.asarray(nodes)[:batch])
    derivative_rows = []
    programs = (
        ("oracle", oracle_batch, (point_batch, coupling)),
        ("fd", fd_batch, (point_batch, coupling, jnp.asarray(args.step))),
        (
            "interp",
            interp_program,
            (
                value_operand,
                jnp.stack(tuple(converged_moments), -1),
                inverse,
                index,
                pitch,
            ),
        ),
    )
    executables = {}
    for name, function, operands in programs:
        row, executable = timed(function, operands, name)
        row.update(batch=len(nodes) if name == "interp" else batch)
        derivative_rows.append(row)
        executables[name] = executable
    read_row, read_executable = timed(run_prepared, (exterior,), "read")

    def assemble(name, moments):
        if name == "interp":
            return exterior + executables[name](
                value_operand, moments, inverse, index, pitch
            )
        matrix_rows = []
        for first in range(0, len(nodes), batch):
            points = jnp.asarray(np.asarray(nodes)[first : first + batch])
            matrix_rows.append(executables[name](points, coupling))
        local = jnp.concatenate(matrix_rows, axis=0)
        return exterior + jnp.einsum("ncmq,cm->nq", local, moments)

    rows = []
    for name in ("oracle", "fd", "interp"):
        derivative = next(row for row in derivative_rows if row["program"] == name)
        for state_name, moments in (
            ("converged", converged_moments),
            ("trial", trial_moments),
        ):
            stacked = jnp.stack(tuple(moments), -1)
            jets = assemble(name, stacked)
            result = jax.block_until_ready(read_executable(jets))
            x_points = np.asarray(result.x_points)[np.asarray(result.x_point_valid)]
            row = dict(
                derivative,
                program=name,
                state=state_name,
                read_cold_compile_seconds=read_row["cold_compile_seconds"],
                read_warm_seconds=read_row["warm_seconds"],
                read_hlo_instructions=read_row["hlo_instructions"],
                valid=bool(result.valid),
                qualified=bool(result.qualified),
                read_derivative_warm_seconds=(
                    read_row["warm_seconds"] + derivative["warm_seconds"]
                ),
                axis=np.asarray(result.axis).tolist(),
                x_points=x_points.tolist(),
                boundary_flux=float(result.boundary_flux),
                axis_flux=float(result.axis_flux),
            )
            rows.append(row)

    best = {(row["program"], row["state"]): row for row in rows}
    origin = best[("oracle", "converged")]
    reference_flux = origin["boundary_flux"]
    span = abs(reference_flux - origin["axis_flux"])
    for row in rows:
        oracle_row = best[("oracle", row["state"])]
        probe = np.asarray(row["x_points"])
        reference_probe = np.asarray(oracle_row["x_points"])
        row.update(
            axis_error_m=float(
                np.linalg.norm(np.asarray(row["axis"]) - np.asarray(oracle_row["axis"]))
            ),
            boundary_flux_error_relative=(
                None
                if span == 0
                else float(
                    abs(row["boundary_flux"] - oracle_row["boundary_flux"]) / span
                )
            ),
            x_point_error_m=(
                None
                if len(probe) != 1 or len(reference_probe) != 1
                else float(np.linalg.norm(probe[0] - reference_probe[0]))
            ),
        )
        print("COMPARISON_ROW " + json.dumps(row), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dict(kind=args.kind, cells=args.cells, rows=rows), indent=2) + "\n"
    )
    print("RUN_COMPLETE " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
