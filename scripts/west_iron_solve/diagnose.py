"""Diagnose a saved material Newton state without changing the production solve."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shlex
import subprocess
import sys

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np
import shapely

from nova.biot import magnetisation as material
from nova.imas.machine import StaticMachineDescription
from nova.jax.config import configure_dtypes
from scripts.west_iron_solve.measure import MACHINE, MU0, save, source_field


@jax.jit
def linearize(state, operands):
    return jax.jacfwd(material._residual)(state, operands)


@jax.jit
def condition_infinity(matrix, factor):
    inverse = jla.lu_solve(factor, jnp.eye(matrix.shape[0], dtype=matrix.dtype))
    return jnp.linalg.norm(matrix, ord=jnp.inf) * jnp.linalg.norm(inverse, ord=jnp.inf)


def correction(state, operands):
    residual = material._residual(state, operands)
    factor = material._factor(state, operands)
    direction = jla.lu_solve(factor, -residual)
    matrix = linearize(state, operands)
    linear_residual = float(
        jnp.linalg.norm(matrix @ direction + residual) / jnp.linalg.norm(residual)
    )
    norm = float(jnp.linalg.norm(residual))
    full_ratio = (
        float(jnp.linalg.norm(material._residual(state + direction, operands))) / norm
    )
    for grade in range(15):
        scale = 0.5**grade
        candidate, trial = material._update(state, operands, factor, jnp.asarray(scale))
        if float(jnp.linalg.norm(trial)) < norm:
            return candidate, direction, scale, linear_residual, full_ratio
    return state, direction, 0.0, linear_residual, full_ratio


def relative(state, operands, reference):
    return float(jnp.linalg.norm(material._residual(state, operands))) / max(
        float(jnp.linalg.norm(state)), reference
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    device = jax.devices()[0]
    assert device.platform == "gpu" and "P100" in device.device_kind, device
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"# revision: {revision} tree: {Path.cwd().resolve()} command: "
        + shlex.join([sys.executable, *sys.argv]),
        flush=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.saved / "solution.pkl").open("rb") as stream:
        solution = pickle.load(stream)
    receipt = json.loads((args.saved / "receipt.json").read_text())
    with Path(receipt["source"]).open("rb") as stream:
        source = pickle.load(stream)
    cells = solution._mesh
    gram = np.load(args.saved / "gram.npy")
    assert gram.shape == (2 * solution.cell_count,) * 2 and np.isfinite(gram).all()
    assert np.max(np.abs(gram)) > 0
    external = source_field(source)(cells.centroids)
    b, mu = np.asarray(receipt["table_b_T"]), np.asarray(receipt["table_mu_relative"])
    law = material._fit_law(b, mu)
    operands = tuple(
        jnp.asarray(x, dtype=jnp.float64) for x in (gram, external.T.reshape(-1), *law)
    )
    stall = jnp.asarray(solution.magnetisation.T.reshape(-1))
    reference = max(np.linalg.norm(external) / MU0, 1e-30)
    measured = relative(stall, operands, reference)
    np.testing.assert_allclose(measured, receipt["residual_history"][-1], rtol=1e-10)
    result = dict(
        revision=revision,
        module=material.__file__,
        cwd=str(Path.cwd().resolve()),
        module_sha256=hashlib.sha256(Path(material.__file__).read_bytes()).hexdigest(),
        device=device.device_kind,
        saved=str(args.saved),
        cell_size_m=receipt["cell_size_m"],
        cells=solution.cell_count,
        matrix_reused=True,
        matrix_builds=0,
        stall_residual=measured,
        source_sha256=receipt["source_sha256"],
        history=[],
        jacobian_checks=[],
        top_cells=[],
    )
    output = args.output / "diagnosis.json"
    save(output, result)
    state = jnp.zeros_like(stall)
    for iteration in range(10):
        state, last_direction, scale, solve_residual, _ = correction(state, operands)
        assert scale == receipt["step_scales"][iteration], (iteration, scale)
    replay_error = float(jnp.linalg.norm(state - stall) / jnp.linalg.norm(stall))
    assert replay_error < 1e-10, replay_error
    result["replay_relative_error"] = replay_error
    result["last_direction_norm_A_per_m"] = float(jnp.linalg.norm(last_direction))
    factor = material._factor(stall, operands)
    matrix = linearize(stall, operands)
    result["stall_condition_infinity"] = float(condition_infinity(matrix, factor))
    total = (operands[1] + operands[0] @ stall).reshape(2, -1)
    frozen_coefficient = jnp.tile(
        material._coefficient(jnp.linalg.norm(total, axis=0), *operands[2:]), 2
    )
    rng = np.random.default_rng(17419)
    directions = [rng.normal(size=stall.size) for _ in range(10)] + [
        np.asarray(last_direction)
    ]
    for index, vector in enumerate(directions):
        vector = jnp.asarray(vector / np.linalg.norm(vector))
        product = matrix @ vector
        frozen_product = vector - frozen_coefficient * (operands[0] @ vector)
        for step in (1.0, 0.1, 0.01):
            difference = (
                material._residual(stall + step * vector, operands)
                - material._residual(stall - step * vector, operands)
            ) / (2 * step)
            mismatch = float(
                jnp.linalg.norm(product - difference) / jnp.linalg.norm(product)
            )
            check = dict(
                direction="last-newton" if index == 10 else f"random-{index}",
                step_A_per_m=step,
                relative_mismatch=mismatch,
                jacobian_product_norm=float(jnp.linalg.norm(product)),
                frozen_mu_mismatch=float(
                    jnp.linalg.norm(frozen_product - difference)
                    / jnp.linalg.norm(product)
                ),
            )
            result["jacobian_checks"].append(check)
            print("JACOBIAN " + json.dumps(check), flush=True)
        save(output, result)
    field = np.asarray((operands[1] + operands[0] @ stall).reshape(2, -1).T)
    magnitude = np.linalg.norm(field, axis=1)
    coefficient = np.asarray(
        material._coefficient(jnp.asarray(magnitude), *operands[2:])
    )
    permeability = 1 / (1 - MU0 * coefficient)
    residual = np.asarray(material._residual(stall, operands)).reshape(2, -1).T
    contribution = np.sum(residual**2, axis=1)
    ranked = np.argsort(contribution)[::-1][:20]
    knots, coefficients, ends = law
    width = np.diff(knots)
    a, c, d, e = coefficients
    left_h = ((a * width + c) * width + d) * width + e
    left_slope = (3 * a * width + 2 * c) * width + d
    result["law_value_jump_max_A_per_m"] = float(
        np.max(np.abs(left_h[:-1] - coefficients[3, 1:]))
    )
    result["law_derivative_jump_max_relative"] = float(
        np.max(
            np.abs(left_slope[:-1] - coefficients[2, 1:])
            / np.maximum(np.abs(left_slope[:-1]), np.abs(coefficients[2, 1:]))
        )
    )
    result["stall_b_range_T"] = [float(magnitude.min()), float(magnitude.max())]
    indices = np.clip(
        np.searchsorted(knots, magnitude, side="right") - 1, 0, len(knots) - 2
    )
    offset = magnitude - knots[indices]
    derivative = (
        3 * coefficients[0, indices] * offset + 2 * coefficients[1, indices]
    ) * offset + coefficients[2, indices]
    derivative = np.where(magnitude >= knots[-1], ends[1], derivative)
    differential_mu = 1 / (MU0 * derivative)
    machine = StaticMachineDescription.read_iron_core(
        f"imas:hdf5?path={MACHINE}", dd_version="4.1.0"
    )
    regions = [
        shapely.Polygon(np.column_stack((s.outline.r, s.outline.z)))
        for s in machine.iron_segments
    ]
    for index in ranked:
        polygon = shapely.Polygon(cells.polygons[index])
        overlaps = [polygon.intersection(region).area for region in regions]
        segment_index = int(np.argmax(overlaps))
        segment = (
            machine.iron_segments[segment_index].name or f"segment[{segment_index}]"
        )
        row = dict(
            cell=int(index),
            centroid_m=cells.centroids[index].tolist(),
            residual_norm_A_per_m=float(np.sqrt(contribution[index])),
            residual_fraction=float(contribution[index] / contribution.sum()),
            b_T=float(magnitude[index]),
            mu_r=float(permeability[index]),
            differential_mu_r=float(differential_mu[index]),
            nearest_knot_distance_T=float(np.min(np.abs(knots - magnitude[index]))),
            area_fraction=float(cells.areas[index] / receipt["cell_size_m"] ** 2),
            clearance_m=polygon.centroid.distance(polygon.boundary),
            segment=segment,
            segment_index=segment_index,
        )
        result["top_cells"].append(row)
        print("CELL " + json.dumps(row), flush=True)
    result.update(
        top_twenty_residual_fraction=float(
            contribution[ranked].sum() / contribution.sum()
        ),
        differential_mu_range=[
            float(differential_mu.min()),
            float(differential_mu.max()),
        ],
        global_minimum_knot_distance_T=float(
            np.min(np.abs(magnitude[:, None] - knots[None, :]))
        ),
        law_knots_T=knots.tolist(),
        authored_knots_T=b.tolist(),
    )
    save(output, result)
    state = stall
    for iteration in range(10, 41):
        row = dict(
            corrections=iteration,
            relative_residual=relative(state, operands, reference),
        )
        result["history"].append(row)
        save(output, result)
        if iteration == 40:
            print("ITERATION " + json.dumps(row), flush=True)
            break
        state, direction, scale, solve_residual, full_ratio = correction(
            state, operands
        )
        row.update(
            step_scale=scale,
            linear_solve_relative_residual=solve_residual,
            direction_norm_A_per_m=float(jnp.linalg.norm(direction)),
            full_step_residual_ratio=full_ratio,
        )
        print("ITERATION " + json.dumps(row), flush=True)
        save(output, result)
        if scale == 0:
            result["termination"] = "backtracking-refused"
            break
    result["terminal_residual"] = relative(state, operands, reference)
    result["terminal_condition_infinity"] = float(
        condition_infinity(
            linearize(state, operands), material._factor(state, operands)
        )
    )
    np.save(args.output / "terminal.npy", np.asarray(state))
    result["phase"] = "complete"
    save(output, result)
    print(
        "SUMMARY "
        + json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in (
                    "history",
                    "jacobian_checks",
                    "top_cells",
                    "law_knots_T",
                    "authored_knots_T",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
