"""Run the frozen-permeability Picard control beside Newton on WEST 62751.

Frozen-permeability Picard evaluates the secant coefficient c(|B|) at the
current field, holds it fixed, and solves the linear system M = c B_ext + c G M
for the next iterate. Newton instead differentiates c, so it corrects the
permeability along with the magnetisation. The plan requires a record of the
Picard iteration failing to converge on WEST from M = 0 at full drive, with the
same continuation Newton used, as the negative control for choosing Newton.

Two positive controls run the same iteration where it should converge: the
analytic shell at mu_r = 10 with a constant law, which the frozen coefficient
solves exactly in one step, and the same shell with a weakly rising law, which
needs several steps and still converges.

    python scripts/west_picard_control/measure.py --output <run-dir>

The mesh, law and external field come from the recorded WEST source recipe in
scripts/west_iron_galerkin_pair/measure.py; nothing here rebuilds them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shlex
import subprocess
import sys
import time

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402
from nova.imas.machine import StaticMachineDescription  # noqa: E402
from scripts.west_iron_galerkin_pair.measure import (  # noqa: E402
    SOURCE,
    SOURCE_SHA256,
    machine_source,
    save,
    source_field,
)
from tests.test_magnetisation import loop, shell  # noqa: E402

PAIR_RECEIPT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261112T201826132383-icm-west-62751-galerkin-pair/"
    "measurement/measurement.json"
)
NEWTON_TOLERANCE = 1e-10
PICARD_ITERATIONS = 50
WEST_SIZE = 0.08
PROBES = np.array([[2.0, 0.0], [0.0, 2.0], [3.0, 1.0]])


def iterate(gram, external, law, tolerance, max_iterations, initial=None):
    """Run frozen-permeability Picard from ``initial`` (default: demagnetised).

    Each step holds c at the field of the incoming iterate and solves the
    linear system exactly, so the record is the method rather than a truncated
    inner solve. The residual is the same F(M) = M - c(|B|) B Newton uses, so
    the two histories are comparable; ``relative`` is the solver's own
    normalisation, ||F|| / max(||M||, ||B_ext||/mu0), and ``residual_over_m`` is
    ||F|| / ||M|| as the control records, undefined at the demagnetised seed.
    """
    knots, coefficients, ends = law
    prescribed = np.asarray(external).T.reshape(-1)
    gram_array = np.asarray(gram, dtype=np.float64)
    operands = tuple(
        jnp.asarray(value, dtype=jnp.float64)
        for value in (gram_array, prescribed, knots, coefficients, ends)
    )
    reference = max(np.linalg.norm(prescribed) / iron._MU0, 1e-30)
    residual = jax.jit(iron._residual)
    coefficient_of = jax.jit(iron._coefficient)
    identity = np.eye(gram_array.shape[0])
    state = np.zeros_like(prescribed)
    if initial is not None:
        state = np.asarray(initial, dtype=np.float64).T.reshape(-1)
    rows, termination = [], "iteration-budget"
    previous_state, previous_permeability = None, None
    for iteration in range(max_iterations + 1):
        defect = np.asarray(residual(jnp.asarray(state), operands))
        norm = float(np.linalg.norm(defect))
        magnetisation_norm = float(np.linalg.norm(state))
        relative = norm / max(magnetisation_norm, reference)
        field = jnp.asarray(prescribed) + jnp.asarray(gram_array) @ jnp.asarray(state)
        strength = jnp.linalg.norm(field.reshape(2, -1), axis=0)
        coefficient = np.asarray(coefficient_of(strength, knots, coefficients, ends))
        permeability = 1 / (1 - iron._MU0 * coefficient)
        rows.append(
            dict(
                iteration=iteration,
                residual_norm=norm,
                magnetisation_norm=magnetisation_norm,
                residual_over_m=(
                    norm / magnetisation_norm if magnetisation_norm else None
                ),
                relative=relative,
                maximum_magnetisation_A_per_m=float(
                    np.linalg.norm(state.reshape(2, -1), axis=0).max()
                ),
                mu_min=float(permeability.min()),
                mu_max=float(permeability.max()),
                maximum_delta_mu_r=(
                    None
                    if previous_permeability is None
                    else float(np.max(np.abs(permeability - previous_permeability)))
                ),
                maximum_delta_M_A_per_m=(
                    None
                    if previous_state is None
                    else float(
                        np.max(
                            np.abs(state.reshape(2, -1) - previous_state.reshape(2, -1))
                        )
                    )
                ),
            )
        )
        if relative <= tolerance:
            termination = "converged"
            break
        if iteration == max_iterations:
            break
        previous_state, previous_permeability = state, permeability
        # Frozen coefficient: one exact solve of (I - diag(c) G) M = c B_ext.
        lifted = np.tile(coefficient, 2)
        matrix = identity - lifted[:, None] * gram_array
        state = np.linalg.solve(matrix, lifted * prescribed)
    return rows, termination, state, permeability


def classify(rows, tolerance):
    """Label a Picard history converged, diverging, oscillating or stagnating.

    The label is decided from recorded quantities a reader can re-derive: the
    residual's own trend, the magnetisation's growth, and whether the residual
    keeps changing sign, which is an iterate swinging about a fixed point
    rather than approaching it.
    """
    relative = np.asarray([row["relative"] for row in rows])
    final = float(relative[-1])
    magnetisation = np.asarray([row["maximum_magnetisation_A_per_m"] for row in rows])
    window = min(20, len(relative) - 1)
    tail = relative[-max(window, 2) :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = tail[1:] / tail[:-1]
    finite = ratios[np.isfinite(ratios) & (ratios > 0)]
    growth = float(np.exp(np.mean(np.log(finite)))) if finite.size else float("nan")
    change = np.diff(relative)
    sign_changes = (
        int(np.sum(np.sign(change[1:]) != np.sign(change[:-1])))
        if len(change) > 1
        else 0
    )
    peak = float(relative[1:].max()) if len(relative) > 1 else final
    if final <= tolerance:
        mode = "converged"
    elif final >= peak and magnetisation[-1] > 1.5 * float(magnetisation[1]):
        mode = "diverging"
    elif sign_changes >= 0.3 * max(len(change) - 1, 1):
        mode = "oscillating"
    else:
        mode = "stagnating"
    extrapolated = None
    if np.isfinite(growth) and 0 < growth < 1 and final > 0:
        extrapolated = float(np.log(tolerance / final) / np.log(growth))
    return dict(
        mode=mode,
        final_relative=final,
        best_relative=float(relative[1:].min()) if len(relative) > 1 else None,
        residual_growth_per_iteration=(growth if np.isfinite(growth) else None),
        iterations_to_tolerance=extrapolated,
        sign_changes_in_residual=sign_changes,
        maximum_magnetisation_seed=float(magnetisation[0]),
        maximum_magnetisation_first_correction=float(magnetisation[1]),
        maximum_magnetisation_last=float(magnetisation[-1]),
        late_magnetisation_relative_span=float(
            (magnetisation[-10:].max() - magnetisation[-10:].min())
            / magnetisation[-10:].max()
        ),
    )


def analytic_shielding(mu, radius=0.9):
    """Exact shielding of a mu_r shell of radius 1 and inner radius ``radius``."""
    return ((2 * mu + 1) * (mu + 2) - 2 * (mu - 1) ** 2 * radius**3) / (9 * mu)


def control(region, field, table_b, table_mu, size, tolerance, label):
    """Run Picard and Newton on an analytic case and compare their roots."""
    cells = iron._mesh(region, size)
    gram = iron._galerkin_interaction(cells)
    external = iron._average_external(cells, field)
    law = iron._fit_law(table_b, table_mu)
    started = time.monotonic()
    rows, termination, state, _ = iterate(
        gram, external, law, tolerance, PICARD_ITERATIONS
    )
    picard_seconds = time.monotonic() - started
    started = time.monotonic()
    newton = iron.solve_magnetisation(
        region,
        field,
        table_b,
        table_mu,
        cell_size=size,
        tolerance=tolerance,
        max_iterations=100,
    )
    newton_seconds = time.monotonic() - started
    picard_magnetisation = state.reshape(2, -1).T
    difference = float(
        np.linalg.norm(picard_magnetisation - newton.magnetisation, axis=1).max()
    )
    air = field(PROBES)
    total = air + newton.field(PROBES)
    shielding = np.linalg.norm(air, axis=1) / np.linalg.norm(total, axis=1)
    entry = dict(
        label=label,
        cells=int(len(cells.centroids)),
        table_b_T=[float(value) for value in table_b],
        table_mu_relative=[float(value) for value in table_mu],
        termination=termination,
        iterations=len(rows) - 1,
        converged=termination == "converged",
        residual_history=[row["relative"] for row in rows],
        final_relative=rows[-1]["relative"],
        maximum_magnetisation_A_per_m=rows[-1]["maximum_magnetisation_A_per_m"],
        mu_min=rows[-1]["mu_min"],
        mu_max=rows[-1]["mu_max"],
        picard_seconds=picard_seconds,
        newton=dict(
            converged=newton.converged,
            iterations=newton.iterations,
            residual_history=list(newton.residual_history),
            maximum_magnetisation_A_per_m=float(
                np.linalg.norm(newton.magnetisation, axis=1).max()
            ),
            mu_min=float(newton.permeability_relative.min()),
            mu_max=float(newton.permeability_relative.max()),
            seconds=newton_seconds,
        ),
        root_difference_over_picard=float(
            difference / max(rows[-1]["maximum_magnetisation_A_per_m"], 1e-30)
        ),
        shielding=shielding.tolist(),
        shielding_exact_at_table_end=float(analytic_shielding(float(table_mu[-1]))),
        rows=rows,
    )
    entry["shielding_error_at_table_end"] = float(
        np.max(np.abs(shielding / entry["shielding_exact_at_table_end"] - 1))
    )
    print(
        "CONTROL "
        + json.dumps({key: value for key, value in entry.items() if key != "rows"}),
        flush=True,
    )
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-receipt", type=Path, default=PAIR_RECEIPT)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    root = script.parents[2]
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    )
    if dirty.strip():
        raise SystemExit("refusing to run on a dirty tree:\n" + dirty)
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    blob = subprocess.check_output(
        ["git", "-C", str(root), "hash-object", str(script)], text=True
    ).strip()
    source_bytes = Path(iron.__file__).read_bytes()
    print(
        f"# revision: {revision} tree: {Path.cwd().resolve()} command: "
        + shlex.join([sys.executable, *sys.argv]),
        flush=True,
    )
    print(
        f"MODULE={iron.__file__} CWD={Path.cwd().resolve()} DEVICE={jax.devices()}",
        flush=True,
    )
    assert jax.config.jax_enable_x64, "the material solve requires float64"
    assert Path(iron.__file__).resolve().is_relative_to(root), iron.__file__
    assert "P100" in jax.devices()[0].device_kind, jax.devices()[0].device_kind
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == SOURCE_SHA256, digest
    with SOURCE.open("rb") as stream:
        data = pickle.load(stream)
    uri, dd_version = machine_source()
    machine = StaticMachineDescription.read_iron_core(uri, dd_version=dd_version)
    table = machine.iron_segments[0].material
    table_b = np.asarray(table.b_field)
    table_mu = np.asarray(table.permeability_relative)
    for segment in machine.iron_segments:
        np.testing.assert_array_equal(segment.material.b_field, table_b)
        np.testing.assert_array_equal(segment.material.permeability_relative, table_mu)
    region = shapely.union_all(
        [
            shapely.Polygon(np.column_stack((s.outline.r, s.outline.z)))
            for s in machine.iron_segments
        ]
    )
    assert region.is_valid
    external = source_field(data)
    record = dict(
        revision=revision,
        script=str(script.relative_to(root)),
        script_blob=blob,
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        material_source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        material_module=iron.__file__,
        cwd=str(Path.cwd().resolve()),
        device=jax.devices()[0].device_kind,
        machine=uri.removeprefix("imas:hdf5?path="),
        dd_version=dd_version,
        cell_size_m=WEST_SIZE,
        newton_tolerance=NEWTON_TOLERANCE,
        picard_iterations=PICARD_ITERATIONS,
        picard_definition=(
            "c(|B|) is evaluated at the current iterate's field and held fixed; "
            "the next iterate solves M = c B_ext + c G M exactly. Same start "
            "(M = 0 at full drive), same mesh, same Galerkin operator and same "
            "tolerance as the recorded Newton run. No damping is added."
        ),
        positive_controls=[],
    )
    save(out / "measurement.json", record)

    started = time.monotonic()
    cells = iron._mesh(region, WEST_SIZE)
    gram = iron._galerkin_interaction(cells)
    build_seconds = time.monotonic() - started
    averaged = iron._average_external(cells, external)
    law = iron._fit_law(table_b, table_mu)
    print(
        "OPERATOR "
        + json.dumps(
            dict(cells=int(len(cells.centroids)), build_seconds=build_seconds)
        ),
        flush=True,
    )
    pair = json.loads(args.pair_receipt.read_text())
    recorded = pair["meshes"][f"{WEST_SIZE:.4g}"]
    assert recorded["cells"] == len(cells.centroids), (
        recorded["cells"],
        len(cells.centroids),
    )
    started = time.monotonic()
    newton = iron.solve_magnetisation(
        region,
        external,
        table_b,
        table_mu,
        cell_size=WEST_SIZE,
        tolerance=NEWTON_TOLERANCE,
        max_iterations=100,
    )
    newton_seconds = time.monotonic() - started
    history = np.asarray(newton.residual_history)
    prior = np.asarray(recorded["residual_history"])
    assert history.shape == prior.shape, (history.shape, prior.shape)
    reproduction = float(np.max(np.abs(history / prior - 1)))
    print(
        "NEWTON "
        + json.dumps(
            dict(
                converged=newton.converged,
                iterations=newton.iterations,
                residual_history=list(newton.residual_history),
                recorded_residual_history=list(recorded["residual_history"]),
                reproduction_max_relative=reproduction,
                seconds=newton_seconds,
            )
        ),
        flush=True,
    )
    assert newton.converged and newton.iterations == recorded["iterations"]
    assert reproduction < 1e-6, reproduction
    record["operator"] = dict(
        cells=int(len(cells.centroids)), build_seconds=build_seconds
    )
    record["newton"] = dict(
        converged=newton.converged,
        iterations=newton.iterations,
        residual_history=list(newton.residual_history),
        recorded_residual_history=list(recorded["residual_history"]),
        reproduction_max_relative=reproduction,
        termination=newton.termination,
        mu_min=float(newton.permeability_relative.min()),
        mu_max=float(newton.permeability_relative.max()),
        maximum_magnetisation_A_per_m=float(
            np.linalg.norm(newton.magnetisation, axis=1).max()
        ),
        seconds=newton_seconds,
        source=str(args.pair_receipt),
    )
    save(out / "measurement.json", record)

    started = time.monotonic()
    rows, termination, state, _ = iterate(
        gram, averaged, law, NEWTON_TOLERANCE, PICARD_ITERATIONS
    )
    picard_seconds = time.monotonic() - started
    verdict = classify(rows, NEWTON_TOLERANCE)
    record["picard"] = dict(
        termination=termination,
        iterations=len(rows) - 1,
        converged=termination == "converged",
        residual_history=[row["relative"] for row in rows],
        rows=rows,
        verdict=verdict,
        seconds=picard_seconds,
    )
    save(out / "measurement.json", record)
    print("PICARD " + json.dumps(verdict), flush=True)
    print(
        "PICARD_TAIL "
        + json.dumps(
            [
                {
                    key: row[key]
                    for key in (
                        "iteration",
                        "relative",
                        "maximum_delta_mu_r",
                        "maximum_magnetisation_A_per_m",
                    )
                }
                for row in rows[-5:]
            ]
        ),
        flush=True,
    )
    if record["picard"]["converged"]:
        raise SystemExit("the negative control converged")
    np.save(
        out / "picard-maximum-M.npy",
        np.asarray([row["maximum_magnetisation_A_per_m"] for row in rows]),
    )
    np.save(out / "picard-state.npy", state.reshape(2, -1).T)

    record["positive_controls"].append(
        control(
            shell(),
            loop,
            [0.0, 10.0],
            [10.0, 10.0],
            0.04,
            NEWTON_TOLERANCE,
            "shell-linear-mu10",
        )
    )
    save(out / "measurement.json", record)
    record["positive_controls"].append(
        control(
            shell(),
            loop,
            [0.0, 1.0],
            [10.0, 12.0],
            0.04,
            NEWTON_TOLERANCE,
            "shell-rising-mu10to12",
        )
    )
    save(out / "measurement.json", record)
    linear = record["positive_controls"][0]
    assert linear["converged"] and linear["iterations"] <= 1, linear["label"]
    for entry in record["positive_controls"]:
        assert entry["converged"], entry["label"]
        assert entry["newton"]["converged"], entry["label"]
        assert entry["root_difference_over_picard"] < 1e-8, entry["label"]
    assert linear["shielding_error_at_table_end"] < 5e-3, linear
    record["complete"] = True
    save(out / "measurement.json", record)
    print(
        "SUMMARY "
        + json.dumps(
            dict(
                complete=True,
                cell_size_m=WEST_SIZE,
                newton_iterations=record["newton"]["iterations"],
                picard_iterations=record["picard"]["iterations"],
                picard_mode=verdict["mode"],
                picard_final_relative=verdict["final_relative"],
                positive_controls=[
                    (entry["label"], entry["iterations"], entry["termination"])
                    for entry in record["positive_controls"]
                ],
            )
        ),
        flush=True,
    )
    assert Path(iron.__file__).read_bytes() == source_bytes


if __name__ == "__main__":
    main()
