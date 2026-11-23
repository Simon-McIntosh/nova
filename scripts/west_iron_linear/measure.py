"""Measure and replay geometry-bound affine WEST iron interactions."""

from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
import gc
import hashlib
import json
from pathlib import Path
import pickle
import resource
import subprocess
import time
from unittest.mock import patch

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.scipy.linalg as jla  # noqa: E402
import numpy as np  # noqa: E402
from nova.biot import magnetisation as iron  # noqa: E402
from nova.biot import tiledassembly as tiled  # noqa: E402
from scripts.iron_graded_refinement.measure import west_inputs, provenance  # noqa: E402
from scripts.iron_linear_magnetisation.assembly_cost import Evaluators  # noqa: E402
from scripts.west_iron_galerkin_pair.measure import (  # noqa: E402
    SOURCE,
    source_field,
)

EVIDENCE = Path("/work/projects/imas_gpu/nova/evidence/iron-core-magnetisation")


def interaction_agreement(reference, candidate):
    """Compare every entry in bounded row chunks, including transferred operands."""
    if reference.shape != candidate.shape:
        raise AssertionError("interaction shapes differ")
    squared_error = squared_reference = maximum_error = maximum_reference = 0.0
    entries = 0
    for first in range(0, len(reference), 256):
        expected = np.asarray(reference[first : first + 256])
        actual = np.asarray(candidate[first : first + 256])
        assert np.isfinite(expected).all() and np.isfinite(actual).all()
        difference = actual - expected
        squared_error += float(np.vdot(difference, difference))
        squared_reference += float(np.vdot(expected, expected))
        maximum_error = max(maximum_error, float(np.abs(difference).max()))
        maximum_reference = max(maximum_reference, float(np.abs(expected).max()))
        entries += expected.size
    assert maximum_reference > 0, "interaction comparison has no nonzero control"
    row = dict(
        entries=entries,
        frobenius_relative=(squared_error / squared_reference) ** 0.5,
        maximum_entry_relative=maximum_error / maximum_reference,
    )
    assert max(row["frobenius_relative"], row["maximum_entry_relative"]) <= 1e-12, row
    return row


def probe_agreement(reference, candidate):
    """Require each probe vector to agree relative to its reference magnitude."""
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    assert reference.shape == candidate.shape == (3, 2)
    scale = np.linalg.norm(reference, axis=1)
    assert np.isfinite(candidate).all() and np.all(scale > 0)
    relative = np.linalg.norm(candidate - reference, axis=1) / scale
    assert np.all(relative <= 1e-12), relative
    return relative.tolist()


def coefficient_agreement(reference, candidate):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    assert reference.shape == candidate.shape and np.isfinite(candidate).all()
    assert np.linalg.norm(reference) > 0
    row = dict(
        frobenius_relative=float(
            np.linalg.norm(candidate - reference) / np.linalg.norm(reference)
        ),
        maximum_entry_relative=float(
            np.abs(candidate - reference).max() / np.abs(reference).max()
        ),
    )
    assert max(row.values()) <= 1e-12, row
    return row


def relative_distance(reference, candidate):
    """Frobenius and maximum-entry distance with no agreement floor.

    The route comparison asks how far each route sits from a refined reference,
    so this reports the same two measures as :func:`coefficient_agreement`
    without asserting a bound the measurement exists to test.
    """
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise AssertionError("distance operands have different shapes")
    assert np.isfinite(reference).all() and np.isfinite(candidate).all()
    norm = float(np.linalg.norm(reference))
    assert norm > 0, "distance has no nonzero control"
    return dict(
        frobenius_relative=float(np.linalg.norm(candidate - reference) / norm),
        maximum_entry_relative=float(
            np.abs(candidate - reference).max() / np.abs(reference).max()
        ),
    )


def _linear_problem(inputs, size, interaction):
    """Rebuild the affine Newton operands the production solve builds, once.

    The persisted interaction and the external field are deterministic, so these
    operands match the ones ``solve_magnetisation`` constructs internally. They
    are built here because the refined reference needs the Jacobian, residual
    and right-hand side as explicit arrays.
    """
    cells = iron._mesh(inputs["region"], size)
    law = iron._fit_law(inputs["table_b"], inputs["table_mu"])
    geometry = iron.linear_magnetisation_geometry(cells.polygons)
    values = np.asarray(inputs["external"](geometry["points"]), dtype=np.float64)
    owners = geometry["owners"]
    project = np.einsum(
        "qij,qj,q->qi",
        np.linalg.inv(geometry["gram"])[owners],
        geometry["basis"],
        geometry["weights"],
    )
    projected = np.zeros((2, len(cells.polygons), 3))
    for component in range(2):
        np.add.at(projected[component], owners, project * values[:, component, None])
    operands = (
        iron._stage_linear_interaction(interaction.matrix),
        jnp.asarray(projected.reshape(-1)),
        *(jnp.asarray(value, dtype=jnp.float64) for value in law),
        jnp.asarray(geometry["basis"]),
        jnp.asarray(project),
        jnp.asarray(owners),
    )
    metric = np.zeros((len(cells.polygons), 6, 6))
    metric[:, :3, :3] = geometry["gram"]
    metric[:, 3:, 3:] = geometry["gram"]
    scale = max(float(np.linalg.norm(projected.reshape(-1))) / iron._MU0, 1e-30)
    return operands, jnp.asarray(metric), scale, geometry


def _extended_direction(factor, jacobian_host, rhs, passes=3, floor=1e-17):
    """Solve J d = rhs, refining with residuals formed in extended precision.

    The Jacobian is float64; the defect ``rhs - J d`` is accumulated in
    ``longdouble`` so the refinement drives the forward error down to the
    working precision independently of the condition number, which is what makes
    the reference accurate enough to measure a 1e-12 route gap.
    """
    extended = jacobian_host.astype(np.longdouble)
    rhs_extended = np.asarray(rhs).astype(np.longdouble)
    direction = jla.lu_solve(factor, rhs)
    history, corrections = [], []
    for _ in range(passes):
        defect = rhs_extended - extended @ np.asarray(direction).astype(np.longdouble)
        size = float(np.sqrt(np.vdot(defect, defect)))
        history.append(
            size / max(float(np.sqrt(np.vdot(rhs_extended, rhs_extended))), 1e-300)
        )
        correction = jla.lu_solve(factor, jnp.asarray(defect.astype(np.float64)))
        corrections.append(
            float(np.linalg.norm(np.asarray(correction)))
            / max(float(np.linalg.norm(np.asarray(direction))), 1e-300)
        )
        direction = direction + correction
        if history[-1] <= floor:
            break
    return direction, history, corrections


def _one_norm(matrix):
    """Maximum absolute column sum, accumulated over column blocks."""
    best = 0.0
    for start in range(0, matrix.shape[1], 2048):
        block = np.abs(matrix[:, start : start + 2048]).sum(axis=0)
        best = max(best, float(block.max()))
    return best


def _inverse_one_norm(factor, dimension):
    """Hager one-norm estimate of J^{-1} using an existing LU factor."""
    estimate = 0.0
    x = jnp.full((dimension,), 1.0 / dimension)
    for _ in range(6):
        y = jla.lu_solve(factor, x, trans=1)
        z = jla.lu_solve(factor, jnp.sign(y))
        magnitude = jnp.abs(z)
        total = float(magnitude.sum())
        if total <= estimate:
            break
        estimate = total
        x = jnp.zeros((dimension,)).at[int(jnp.argmax(magnitude))].set(1.0)
    return estimate


def _linear_step_routes(operands, metric):
    """Solve one Newton linear system by each route and by a refined reference.

    The system is the first Newton step, ``J(0) d = -F(0)``, which the persisted
    operator and the external field fix without a nonlinear iteration. The
    reference refines its solve with float64 residuals accumulated in extended
    precision, so its own forward error is about ``kappa * eps_extended`` and sits
    decades below the ``kappa * eps_float64`` both routes can reach.
    """
    state = jnp.zeros(int(operands[1].size), dtype=jnp.float64)
    dimension = int(state.size)
    jacobian = iron._linear_jacobian(state, operands, jnp.asarray(0.0))
    jacobian_host = np.asarray(jacobian)
    right = -iron._linear_residual(state, operands)
    factor = jla.lu_factor(jacobian)
    lu_direction = jla.lu_solve(factor, right)
    reference, refinement, corrections = _extended_direction(
        factor, jacobian_host, right
    )
    cholesky_factor = iron._linear_cholesky_factor(
        state, operands, metric, jnp.asarray(0.0)
    )
    cholesky_update = iron._linear_cholesky_programs(state.device)[3]
    candidate, _ = cholesky_update(state, operands, cholesky_factor, jnp.asarray(1.0))
    cholesky_direction = candidate - state
    extended = jacobian_host.astype(np.longdouble)
    right_extended = np.asarray(right).astype(np.longdouble)
    residual = right_extended - extended @ np.asarray(reference).astype(np.longdouble)
    residual_relative = float(
        float(np.sqrt(np.vdot(residual, residual)))
        / max(float(np.sqrt(np.vdot(right_extended, right_extended))), 1e-300)
    )
    matrix_norm = _one_norm(jacobian_host)
    inverse_norm = _inverse_one_norm(factor, dimension)
    condition = dict(
        method="Hager one-norm estimate at the first Newton step",
        matrix_one_norm=matrix_norm,
        inverse_one_norm=inverse_norm,
        condition_estimate=matrix_norm * inverse_norm,
    )
    precision_floor = max(
        float(np.finfo(np.float64).eps), corrections[-1] if corrections else 0.0
    )
    reference_error = dict(
        residual_relative=residual_relative,
        forward_error_estimate=precision_floor,
        refinement_history=refinement,
        correction_history=corrections,
    )
    del jacobian_host
    return dict(
        reference=np.asarray(reference),
        lu=np.asarray(lu_direction),
        cholesky=np.asarray(cholesky_direction),
        condition=condition,
        reference_error=reference_error,
    )


def route_refinement(inputs, size, interaction, *, tolerance, max_iterations):
    """Measure both linear routes against an iteratively refined reference.

    Runs the LU route and the Cholesky route over the nonlinear problem at one
    tolerance to reproduce their coefficient difference, then solves the first
    Newton linear system by each route and by an extended-precision-refined
    reference and reports each route's Frobenius and maximum-entry distance from
    it. The first Newton step carries the state to leading order, so its
    direction distance predicts the converged coefficient difference.
    """
    lu = iron.solve_magnetisation(
        inputs["region"],
        inputs["external"],
        inputs["table_b"],
        inputs["table_mu"],
        cell_size=size,
        basis="linear",
        interaction=interaction,
        tolerance=tolerance,
        max_iterations=max_iterations,
        linear_factorisation="combined",
    )
    cholesky = iron.solve_magnetisation(
        inputs["region"],
        inputs["external"],
        inputs["table_b"],
        inputs["table_mu"],
        cell_size=size,
        basis="linear",
        interaction=interaction,
        tolerance=tolerance,
        max_iterations=max_iterations,
        linear_factorisation="cholesky",
    )
    if not (lu.converged and cholesky.converged):
        raise AssertionError("a route did not converge at the measurement tolerance")
    operands, metric, _scale, _geometry = _linear_problem(inputs, size, interaction)
    step = _linear_step_routes(operands, metric)
    del metric
    reference_coefficients = step["reference"].reshape(2, -1, 3)
    distances = {
        "lu": relative_distance(reference_coefficients, step["lu"].reshape(2, -1, 3)),
        "cholesky": relative_distance(
            reference_coefficients, step["cholesky"].reshape(2, -1, 3)
        ),
    }
    routes = dict(
        coefficient=relative_distance(
            np.asarray(lu.linear_coefficients),
            np.asarray(cholesky.linear_coefficients),
        ),
        direction=relative_distance(
            step["lu"].reshape(2, -1, 3), step["cholesky"].reshape(2, -1, 3)
        ),
        lu_residual_history=list(lu.residual_history),
        cholesky_residual_history=list(cholesky.residual_history),
        lu_iterations=lu.iterations,
        cholesky_iterations=cholesky.iterations,
    )
    sizes = {name: max(value.values()) for name, value in distances.items()}
    condition = step["condition"]
    reference_error = step["reference_error"]
    floor = condition["condition_estimate"] * np.finfo(np.float64).eps
    nearest = min(sizes.values())
    resolvable = bool(nearest > 10 * reference_error["forward_error_estimate"])
    ratio = max(sizes.values()) / max(min(sizes.values()), 1e-300)
    rounding = (
        resolvable
        and all(value <= 10 * floor for value in sizes.values())
        and ratio <= 100
    )
    farther = max(sizes, key=sizes.get)
    if not resolvable:
        outcome = "indeterminate"
    elif rounding:
        outcome = "rounding"
    else:
        outcome = "defect"
    verdict = dict(
        machine_epsilon=float(np.finfo(np.float64).eps),
        extended_epsilon=float(np.finfo(np.longdouble).eps),
        rounding_scale=floor,
        distance_ratio=ratio,
        farther_route=farther,
        outcome=outcome,
    )
    if outcome == "defect":
        deviation = np.abs(step["lu"] - step["reference"])
        if farther == "cholesky":
            deviation = np.abs(step["cholesky"] - step["reference"])
        order = np.argsort(deviation)[-5:][::-1]
        verdict["largest_entries"] = [
            dict(
                index=int(i),
                distance=float(deviation[i]),
                relative=float(deviation[i] / abs(step["reference"][i]))
                if step["reference"][i] != 0
                else None,
            )
            for i in order
        ]
    planted = step["reference"] * (1 + 1e-10)
    planted_distance = relative_distance(
        reference_coefficients, planted.reshape(2, -1, 3)
    )
    negative_control = dict(
        planted_relative=1e-10,
        planted_distance=planted_distance,
        detected=bool(max(planted_distance.values()) > 1e-11),
    )
    return dict(
        routes=routes,
        reference=dict(
            converged=True,
            error=reference_error,
            resolvable=resolvable,
            correction_relative=reference_error["residual_relative"],
        ),
        condition=condition,
        distances=distances,
        negative_control=negative_control,
        verdict=verdict,
    )


def refinement_verdict(reference, candidate):
    """A one-percent change at any probe leaves the refinement blocked."""
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    assert reference.shape == candidate.shape == (3,)
    assert np.isfinite(candidate).all() and np.all(reference > 0)
    change = (candidate - reference) / reference
    return dict(
        status="complete" if np.all(np.abs(change) < 0.01) else "blocked",
        relative_change=change.tolist(),
        maximum_change=float(np.abs(change).max()),
    )


def save(path, row):
    path.write_text(json.dumps(row, indent=2) + "\n")


def mesh_size(label, inputs):
    if label.startswith("uniform-"):
        return float(label.removeprefix("uniform-"))
    fine, coarse = {"graded-16-80": (0.016, 0.08), "graded-8-40": (0.008, 0.04)}[label]
    return iron.graded_cell_size(
        inputs["surfaces"]["radial"],
        inputs["surfaces"]["vertical"],
        fine=fine,
        coarse=coarse,
        cells_across=20,
    )


def constant_reference(label):
    if label.startswith("uniform-"):
        path = EVIDENCE / "west-62751/mesh-pair/measurement.json"
        key = label.removeprefix("uniform-")
    else:
        path = EVIDENCE / "graded-refinement/west-cpu/measurement.json"
        key = {"graded-16-80": "graded-fine", "graded-8-40": "graded-half"}[label]
    if not path.is_file():
        return dict(
            receipt=str(path),
            missing=True,
            key=key,
            reason=(
                "the constant-basis receipt is absent from the evidence root: its "
                "former path lay under the retired shared root and it was not "
                "rebuilt there after the storage deletion of 2026-11-19"
            ),
        )
    document = json.loads(path.read_text())
    table = document["meshes"] if label.startswith("uniform-") else document["west"]
    row = table[key]
    return dict(
        receipt=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        key=key,
        cells=row["cells"],
        magnitude_T=row["magnitude_T"],
    )


def factor_capacity(cells, geometry, inputs, *, factorisation="combined"):
    """Compile the actual quadrature shape and measure live factor storage."""
    n = 6 * len(cells.polygons)

    def shape(dims):
        return jax.ShapeDtypeStruct(dims, np.float64)

    law = tuple(
        shape(v.shape) for v in iron._fit_law(inputs["table_b"], inputs["table_mu"])
    )
    operands = (
        shape((n, n)),
        shape((n,)),
        *law,
        shape(geometry["basis"].shape),
        shape(geometry["basis"].shape),
        jax.ShapeDtypeStruct(geometry["owners"].shape, np.int64),
    )
    started = time.monotonic()
    programs = {}
    functions = (
        [("factor", iron._linear_factor, (shape((n,)), operands))]
        if factorisation == "combined"
        else [
            ("jacobian", iron._linear_jacobian, (shape((n,)), operands)),
            ("lu", iron._donated_lu, (shape((n, n)),)),
        ]
    )
    for name, function, arguments in functions:
        executable = function.lower(*arguments).compile()
        stats = executable.memory_analysis()
        live = (
            stats.argument_size_in_bytes
            + stats.output_size_in_bytes
            + stats.temp_size_in_bytes
            - stats.alias_size_in_bytes
        )
        # LU receives only the Jacobian; the interaction remains resident.
        retained = 8 * n**2 if name == "lu" else 0
        programs[name] = dict(
            argument_bytes=stats.argument_size_in_bytes,
            output_bytes=stats.output_size_in_bytes,
            temporary_bytes=stats.temp_size_in_bytes,
            alias_bytes=stats.alias_size_in_bytes,
            retained_interaction_bytes=retained,
            live_bytes=live + retained,
        )
        del executable
    live = max(program["live_bytes"] for program in programs.values())
    physical = (
        int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        * 1024**2
    )
    row = dict(
        compile_seconds=time.monotonic() - started,
        factorisation=factorisation,
        programs=programs,
        compiled_live_bytes=live,
        physical_bytes=physical,
        allocator_budget_bytes=jax.devices()[0].memory_stats()["bytes_limit"],
        reserve_bytes=int(0.1 * physical),
        fits=bool(
            live < min(0.9 * physical, jax.devices()[0].memory_stats()["bytes_limit"])
        ),
    )
    gc.collect()
    return row


def cholesky_capacity(cells):
    """Measure donated column-major row construction and potrf at the full shape."""
    count = len(cells.polygons)
    n = 6 * count

    def shape(dimensions):
        return jax.ShapeDtypeStruct(dimensions, np.float64)

    _, write, factor, _ = iron._linear_cholesky_programs(jax.devices()[0])
    started = time.monotonic()
    programs = {}
    for name, function, arguments in (
        (
            "row_construction",
            write,
            (
                shape((n, n)),
                shape((n, n)),
                shape((count, 6, 6)),
                shape((count, 6, 6)),
                shape((2, count)),
                shape(()),
                jax.ShapeDtypeStruct((), np.int64),
                min(count, 64),
            ),
        ),
        ("cholesky", factor, (shape((n, n)),)),
    ):
        executable = function.lower(*arguments).compile()
        stats = executable.memory_analysis()
        assert stats.alias_size_in_bytes == 8 * n * n, "factor buffer donation was lost"
        retained = 8 * n * n if name == "cholesky" else 0
        programs[name] = dict(
            argument_bytes=stats.argument_size_in_bytes,
            output_bytes=stats.output_size_in_bytes,
            temporary_bytes=stats.temp_size_in_bytes,
            alias_bytes=stats.alias_size_in_bytes,
            retained_interaction_bytes=retained,
            live_bytes=stats.argument_size_in_bytes
            + stats.output_size_in_bytes
            + stats.temp_size_in_bytes
            - stats.alias_size_in_bytes
            + retained,
            input_formats=str(executable.input_formats),
            output_formats=str(executable.output_formats),
        )
        del executable
    limit = jax.devices()[0].memory_stats()["bytes_limit"]
    live = max(program["live_bytes"] for program in programs.values())
    return dict(
        programs=programs,
        compile_seconds=time.monotonic() - started,
        compiled_live_bytes=live,
        allocator_budget_bytes=limit,
        fits=live < limit,
        factor_dtype="float64",
    )


def solve(
    inputs,
    size,
    interaction,
    external=None,
    *,
    factorisation="combined",
    observe_interaction=None,
    observe_solution=None,
    progress=None,
    tolerance=1e-10,
):
    """Measure projection and Newton separately; forbid hidden assembly."""

    def refuse(*args, **kwargs):
        raise AssertionError("persisted solve attempted interaction assembly")

    residual = iron._linear_residual
    clock = {}
    preparation = {}
    memory = []

    def measured_phase(name, function):
        def call(*args, **kwargs):
            event = dict(phase=name, before=jax.devices()[0].memory_stats())
            memory.append(event)
            began = time.monotonic()
            if progress:
                progress(memory)
            try:
                value = function(*args, **kwargs)
                jax.block_until_ready(value)
                if name == "interaction_staging":
                    event.update(
                        dtype=str(value.dtype),
                        matrix_bytes=value.size * value.dtype.itemsize,
                        buffer_pointer=value.unsafe_buffer_pointer(),
                    )
                return value
            except Exception as error:
                event["error"] = repr(error)
                raise
            finally:
                event.update(
                    after=jax.devices()[0].memory_stats(),
                    seconds=time.monotonic() - began,
                )
                if progress:
                    progress(memory)

        return call

    def timed(name, function):
        def call(*args, **kwargs):
            began = time.monotonic()
            try:
                return function(*args, **kwargs)
            finally:
                preparation[name] = preparation.get(name, 0.0) + (
                    time.monotonic() - began
                )

        return call

    def timed_residual(state, operands):
        if isinstance(state, jax.core.Tracer):
            return residual(state, operands)
        if "start" not in clock:
            # Charge pending operand transfers to preparation, before Newton.
            jax.block_until_ready((state, operands))
            if observe_interaction is not None:
                observe_interaction(operands[0])
        clock.setdefault("start", time.monotonic())
        value = residual(state, operands)
        jax.block_until_ready(value)
        clock["end"] = time.monotonic()
        return value

    with ExitStack() as stack:
        for attribute, phase in (
            ("_stage_linear_interaction", "interaction_staging"),
            ("_linear_cholesky_factor", "cholesky_factor"),
        ):
            stack.enter_context(
                patch.object(
                    iron, attribute, measured_phase(phase, getattr(iron, attribute))
                )
            )
        for name, attribute in (
            ("law_fit_seconds", "_fit_law"),
            ("mesh_seconds", "_mesh"),
            ("linear_geometry_seconds", "linear_magnetisation_geometry"),
        ):
            stack.enter_context(
                patch.object(iron, attribute, timed(name, getattr(iron, attribute)))
            )
        stack.enter_context(
            patch.object(
                iron.MagnetisationInteraction,
                "validate",
                timed("interaction_validation_seconds", interaction.validate.__func__),
            )
        )
        stack.enter_context(patch.object(iron, "_galerkin_interaction", refuse))
        stack.enter_context(patch.object(iron, "_linear_residual", timed_residual))
        started = time.monotonic()
        result = iron.solve_magnetisation(
            inputs["region"],
            timed("external_field_seconds", external or inputs["external"]),
            inputs["table_b"],
            inputs["table_mu"],
            cell_size=size,
            basis="linear",
            interaction=interaction,
            tolerance=tolerance,
            max_iterations=100,
            linear_factorisation=factorisation,
        )
        wall = time.monotonic() - started
    if observe_solution:
        observe_solution(result)
    preparation["projection_checks_and_device_setup_seconds"] = (
        clock["start"] - started - sum(preparation.values())
    )
    air = (external or inputs["external"])(inputs["probes"])
    total = air + result.field(inputs["probes"])
    return dict(
        solve_seconds=wall,
        newton_seconds=clock["end"] - clock["start"],
        preparation_seconds=clock["start"] - started,
        preparation_breakdown=preparation,
        preparation_breakdown_note=(
            "Disjoint host wall intervals inside solve_magnetisation. Linear geometry "
            "includes quadrature, basis, Gram blocks and bound-current incidence. "
            "The remainder includes external-field projection, matrix finiteness "
            "scan, operand construction and synchronized host-to-device transfers. "
            "Probe evaluation and persisted-operator loading are outside this split."
        ),
        converged=bool(result.converged),
        iterations=result.iterations,
        termination=result.termination,
        residual_history=list(result.residual_history),
        regularisations=list(result.regularisation_history),
        mu_min=float(result.permeability_relative.min()),
        mu_max=float(result.permeability_relative.max()),
        field_T=total.tolist(),
        magnitude_T=np.linalg.norm(total, axis=1).tolist(),
        memory_phases=memory,
        factor_dtype="float64",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--blob", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--refinement", action="store_true")
    parser.add_argument("--resume-refinement", action="store_true")
    parser.add_argument("--compare-routes", action="store_true")
    parser.add_argument("--refine-routes", action="store_true")
    parser.add_argument("--operator", type=Path)
    parser.add_argument("--cholesky", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
    replay_mode = parser.add_mutually_exclusive_group()
    replay_mode.add_argument("--replay", action="store_true")
    replay_mode.add_argument("--half-drive-only", action="store_true")
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    jax.config.update("jax_enable_compilation_cache", False)
    stamp = provenance(Path(__file__).resolve(), args.revision, args.blob)
    assert jax.devices()[0].platform == "gpu"
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = west_inputs()
    size = mesh_size(args.mesh, inputs)
    cells = iron._mesh(inputs["region"], size)
    count = len(cells.polygons)
    path = args.output / f"{args.mesh}{'-cholesky' if args.cholesky else ''}.json"
    operator_path = args.output / f"{args.mesh}-operator.npy"
    row = dict(
        **stamp,
        mesh=args.mesh,
        cells=count,
        unknowns=6 * count,
        operator_bytes=8 * (6 * count) ** 2,
        completed=False,
        device=jax.devices()[0].device_kind,
        x64=True,
        probes_m=inputs["probes"].tolist(),
        air_drift=inputs["air_drift"],
        constant=constant_reference(args.mesh),
        module=str(Path(iron.__file__).resolve()),
        cwd=str(Path.cwd().resolve()),
        factor_dtype="float64",
        factorisation="cholesky" if args.cholesky else "split",
    )
    constant = row["constant"]
    if constant.get("missing"):
        assert row["cells"] == count
    else:
        assert constant["cells"] == count, "constant receipt names another mesh"
    if args.resume_refinement:
        assert args.refinement and args.mesh == "graded-8-40"
        original = json.loads((args.output / f"{args.mesh}.json").read_text())
        assert original["geometry_digest"] and original["operator_path"] == str(
            operator_path
        )
        archive = args.output / "graded-8-40-assembly.json"
        if not archive.exists():
            save(archive, original)
        assembly = json.loads(archive.read_text())
        row = original | row
        row["assembly_provenance"] = dict(
            revision=assembly["revision"],
            measure_py_blob=assembly["measure_py_blob"],
            receipt=str(archive),
            geometry_digest=assembly["geometry_digest"],
        )
        interaction = iron.load_magnetisation_interaction(
            operator_path, region=inputs["region"], cell_size=size, basis="linear"
        )
        assert interaction.geometry_digest == assembly["geometry_digest"]
        geometry = tiled.linear_magnetisation_geometry(cells.polygons)
        route = "cholesky" if args.cholesky else "split"
        row[route + "_capacity"] = (
            cholesky_capacity(cells)
            if args.cholesky
            else factor_capacity(cells, geometry, inputs, factorisation="split")
        )
        save(path, row)
        assert row[route + "_capacity"]["fits"], row[route + "_capacity"]

        def progress(events):
            row["memory_phases"] = events
            save(path, row)

        row["full_drive"] = solve(
            inputs,
            size,
            interaction,
            factorisation=route,
            progress=progress,
            tolerance=1e-12 if args.cholesky else 1e-10,
        )
        row["solve_device_memory_stats"] = jax.devices()[0].memory_stats()
        row["assembly_calls"] = 0
        reference_path = (
            args.output / "graded-16-80-cholesky.json"
            if args.cholesky
            else args.output.parent / "graded-16-80.json"
        )
        row["refinement_reference"] = str(reference_path)
        reference_row = json.loads(reference_path.read_text())
        reference = reference_row["combined" if args.cholesky else "full_drive"][
            "magnitude_T"
        ]
        row["refinement"] = refinement_verdict(
            reference, row["full_drive"]["magnitude_T"]
        )
        row["completed"] = bool(row["full_drive"]["converged"])
        save(path, row)
        assert row["completed"], row["full_drive"]["termination"]
        print(json.dumps(row), flush=True)
        return
    if args.refine_routes:
        assert args.mesh == "uniform-0.04"
        source = (
            args.operator
            if args.operator is not None
            else args.output.parent / f"{args.mesh}-operator.npy"
        )
        interaction = iron.load_magnetisation_interaction(
            source, region=inputs["region"], cell_size=size, basis="linear"
        )
        row["operator_path"] = str(source)
        row["geometry_digest"] = interaction.geometry_digest
        save(path, row)
        row["refined_reference"] = route_refinement(
            inputs,
            size,
            interaction,
            tolerance=1e-12,
            max_iterations=60,
        )
        row["completed"] = True
        save(path, row)
        print(json.dumps(row), flush=True)
        return
    if args.compare_routes:
        source = args.output.parent / f"{args.mesh}-operator.npy"
        interaction = iron.load_magnetisation_interaction(
            source, region=inputs["region"], cell_size=size, basis="linear"
        )
        row["operator_path"] = str(source)
        row["geometry_digest"] = interaction.geometry_digest
        save(path, row)
        coefficients = {}
        candidate_route = "cholesky" if args.cholesky else "split"
        if args.cholesky:
            row["cholesky_capacity"] = cholesky_capacity(cells)
            save(path, row)
            assert row["cholesky_capacity"]["fits"]
        for route in ("combined", candidate_route):

            def observe(matrix):
                row[f"{route}_interaction"] = interaction_agreement(
                    interaction.matrix, matrix
                )

            def solution(result):
                coefficients[route] = result.linear_coefficients.copy()

            def progress(events):
                row[route + "_memory_phases"] = events
                save(path, row)

            row[route] = solve(
                inputs,
                size,
                interaction,
                factorisation=route,
                observe_interaction=observe,
                observe_solution=solution,
                progress=progress,
                tolerance=1e-12 if args.cholesky else 1e-10,
            )
            save(path, row)
            assert row[route]["converged"], row[route]["termination"]
        row["probe_relative_difference"] = probe_agreement(
            row["combined"]["field_T"], row[candidate_route]["field_T"]
        )
        row["coefficient_relative_difference"] = coefficient_agreement(
            coefficients["combined"], coefficients[candidate_route]
        )
        row["completed"] = True
        save(path, row)
        print(json.dumps(row), flush=True)
        return
    if args.replay or args.half_drive_only:
        started = time.monotonic()
        interaction = iron.load_magnetisation_interaction(
            operator_path, region=inputs["region"], cell_size=size, basis="linear"
        )
        row["load_seconds"] = time.monotonic() - started
        row["geometry_digest"] = interaction.geometry_digest
        row["operator_path"] = str(operator_path)
        if args.replay:
            original = json.loads(path.read_text())
            row["full_drive"] = solve(inputs, size, interaction)
            row["bit_identical"] = (
                np.asarray(row["full_drive"]["field_T"], dtype=np.float64).tobytes()
                == np.asarray(
                    original["full_drive"]["field_T"], dtype=np.float64
                ).tobytes()
            )
            save(args.output / "replay.json", row)
            assert row["bit_identical"], "fresh-process probe fields changed"
        with SOURCE.open("rb") as stream:
            data = copy.deepcopy(pickle.load(stream))
        for key in data["currents"]:
            data["currents"][key] *= 0.5
        half_external = source_field(data)
        row["half_coil_drive"] = solve(inputs, size, interaction, half_external)
        row["plasma_unchanged"] = True
        # Load a real operator from the coarse mesh against this mesh.
        wrong = args.output / "uniform-0.08-operator.npy"
        try:
            iron.load_magnetisation_interaction(
                wrong, region=inputs["region"], cell_size=size, basis="linear"
            )
        except ValueError as error:
            assert "geometry digest" in str(error)
            row["negative_control"] = dict(operator=str(wrong), refusal=str(error))
        else:
            raise AssertionError("different-mesh interaction was accepted")
        row["assembly_calls"] = 0
        row["completed"] = True
        receipt = "half-drive.json" if args.half_drive_only else "replay.json"
        save(args.output / receipt, row)
        assert row["half_coil_drive"]["converged"]
        if args.replay:
            assert row["full_drive"]["converged"]
        print(json.dumps(row), flush=True)
        return
    save(path, row)
    geometry = tiled.linear_magnetisation_geometry(cells.polygons)
    row["capacity"] = factor_capacity(cells, geometry, inputs)
    if args.refinement:
        row["split_capacity"] = factor_capacity(
            cells, geometry, inputs, factorisation="split"
        )
    save(path, row)
    capacity = row["split_capacity"] if args.refinement else row["capacity"]
    if not capacity["fits"]:
        row["skipped"] = (
            "compiled factor and interaction exceed card capacity with 10% reserve"
        )
        save(path, row)
        return
    # Assembly compile is measured independently of the Newton-shape compilation.
    timing = Evaluators()
    for name in ("_mutual_edge_evaluator", "_polygon_current_evaluator"):
        setattr(tiled, name, timing.factory(name, getattr(tiled, name)))
    started = time.monotonic()
    matrix, _ = tiled.linear_galerkin_cell_interaction(cells.polygons, plan=iron._PLAN)
    row["assembly_seconds"] = time.monotonic() - started
    row["assembly_peak_host_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    row["assembly_device_memory_stats"] = jax.devices()[0].memory_stats()
    row["assembly_program_memory"] = {
        str(key): dict(
            argument_bytes=(
                stats := executable.memory_analysis()
            ).argument_size_in_bytes,
            output_bytes=stats.output_size_in_bytes,
            temporary_bytes=stats.temp_size_in_bytes,
            alias_bytes=stats.alias_size_in_bytes,
        )
        for key, executable in timing.executables.items()
    }
    row["programs"] = list(timing.programs.values())
    row["compile_seconds"] = sum(p["compile_seconds"] for p in row["programs"])
    row["execute_seconds"] = sum(p["execute_seconds"] for p in row["programs"])
    row["host_seconds"] = (
        row["assembly_seconds"] - row["compile_seconds"] - row["execute_seconds"]
    )
    row["seconds_per_cell_pair"] = row["assembly_seconds"] / count**2
    row["projected_11708_seconds"] = row["seconds_per_cell_pair"] * 11708**2
    save(path, row)
    interaction = iron.save_magnetisation_interaction(
        operator_path, matrix, region=inputs["region"], cell_size=size, basis="linear"
    )
    row["operator_path"] = str(operator_path)
    row["geometry_digest"] = interaction.geometry_digest
    if args.assemble_only:
        row["assembly_only"] = True
        save(path, row)
        print(json.dumps(row), flush=True)
        return
    save(path, row)
    row["full_drive"] = solve(
        inputs,
        size,
        interaction,
        factorisation="split" if args.refinement else "combined",
    )
    if args.refinement:
        reference_path = args.output.parent / "graded-16-80.json"
        row["refinement_reference"] = str(reference_path)
        reference = json.loads(reference_path.read_text())["full_drive"]["magnitude_T"]
        row["refinement"] = refinement_verdict(
            reference, row["full_drive"]["magnitude_T"]
        )
        row["solve_device_memory_stats"] = jax.devices()[0].memory_stats()
    row["completed"] = True
    save(path, row)
    print(json.dumps(row), flush=True)
    assert row["full_drive"]["converged"], row["full_drive"]["termination"]


if __name__ == "__main__":
    main()
