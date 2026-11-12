"""Replay saved iron roots, continue the drive, and diagnose collocation stability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.scipy.linalg as jla  # noqa: E402
import numpy as np  # noqa: E402
from scipy.sparse.linalg import LinearOperator, eigsh  # noqa: E402
import shapely  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402
from nova.imas.machine import StaticMachineDescription  # noqa: E402

RUNS = Path("/home/ITER/mcintos/.config/reckon/crew/runs")
SAVED = RUNS / "r-20261111T130417404611-icm-west-62751-solve/measurement-pair/0.0566"
PRIOR = RUNS / "r-20261111T205006716359-icm-newton-trust-region"
SMOOTH = RUNS / "r-20261112T000127065672-icm-magnetisation-smoothness"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@jax.jit
def multiply(matrix, vector):
    return matrix @ vector


@jax.jit
def inverse_normal(factor, vector):
    return jla.lu_solve(factor, jla.lu_solve(factor, vector, trans=1))


def extremes(matrix):
    """Ritz extremes with explicit residuals; matrices stay on the device."""
    size = matrix.shape[0]

    def action(vector):
        return np.asarray(multiply(matrix, jnp.asarray(vector)))

    operator = LinearOperator((size, size), matvec=action, dtype=np.float64)
    values, vectors = eigsh(
        operator,
        k=2,
        which="BE",
        tol=1e-9,
        maxiter=10000,
        v0=np.random.default_rng(1701).normal(size=size),
        ncv=48,
    )
    defects = [
        float(np.linalg.norm(action(vectors[:, k]) - values[k] * vectors[:, k]))
        for k in range(2)
    ]
    assert max(defects) < 1e-6 * max(1.0, float(np.max(np.abs(values))))
    return values, vectors, defects


def singular_minimum(matrix):
    factor = jla.lu_factor(matrix)

    def action(vector):
        return np.asarray(inverse_normal(factor, jnp.asarray(vector)))

    size = matrix.shape[0]
    operator = LinearOperator((size, size), matvec=action, dtype=np.float64)
    values, vectors = eigsh(
        operator,
        k=1,
        which="LA",
        tol=1e-10,
        maxiter=10000,
        v0=np.random.default_rng(2718).normal(size=size),
        ncv=32,
    )
    value = float(1 / np.sqrt(values[0]))
    right = jnp.asarray(vectors[:, 0])
    product = matrix @ right
    left = product / jnp.linalg.norm(product)
    measured = float(jnp.linalg.norm(product))
    defect = float(jnp.linalg.norm(matrix.T @ left - measured * right))
    return dict(value=value, direct_norm=measured, transpose_defect=defect)


def locate(vector, mesh, segments, strip):
    energy = np.sum(vector.reshape(2, -1).T ** 2, axis=1)
    energy /= energy.sum()
    indices = np.argsort(energy)[-12:][::-1]
    return dict(
        segment_fraction={
            str(s): float(energy[segments == s].sum()) for s in np.unique(segments)
        },
        strip_fraction=float(energy[strip].sum()),
        cells=[
            dict(
                cell=int(i),
                segment=int(segments[i]),
                centroid_m=mesh.centroids[i].tolist(),
                area_m2=float(mesh.areas[i]),
                fraction=float(energy[i]),
            )
            for i in indices
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"# revision: {revision} tree: {Path.cwd().resolve()} "
        f"command: {sys.executable} {Path(__file__).resolve()} --output {out}",
        flush=True,
    )
    print(
        f"MODULE={iron.__file__} CWD={Path.cwd().resolve()} DEVICE={jax.devices()}",
        flush=True,
    )
    assert jax.config.jax_enable_x64
    assert jax.devices()[0].platform == "gpu" and "P100" in jax.devices()[0].device_kind
    blob = subprocess.check_output(
        ["git", "show", revision + ":nova/biot/magnetisation.py"]
    )
    assert Path(iron.__file__).read_bytes() == blob
    # Both saved roots must use exactly the same material equation as the ramp.
    prior_blob = subprocess.check_output(
        [
            "git",
            "show",
            "39e67e073374daae459db8f327284e0e4de53a4f:nova/biot/magnetisation.py",
        ]
    )
    assert prior_blob == blob
    receipt = json.loads((SAVED / "receipt.json").read_text())
    with (SAVED / "solution.pkl").open("rb") as stream:
        seed = pickle.load(stream)
    with (PRIOR / "solution.pkl").open("rb") as stream:
        baseline = pickle.load(stream)
    roots = dict(
        A=np.asarray(baseline.magnetisation), B=np.load(SMOOTH / "west-0.0001-M.npy")
    )
    np.testing.assert_array_equal(roots["A"], np.load(SMOOTH / "baseline-M.npy"))
    gram = np.load(SAVED / "gram.npy")
    external = np.load(PRIOR / "external.npy")
    probe = np.load(SMOOTH / "probe-response.npy")
    assert (
        gram.shape == (external.size, external.size)
        and external.shape == roots["A"].shape
    )
    assert probe.shape == (6, external.size)
    assert np.isfinite(gram).all() and np.max(np.abs(gram)) > 0
    assert np.isfinite(probe).all() and np.max(np.abs(probe)) > 0
    law = iron._fit_law(receipt["table_b_T"], receipt["table_mu_relative"])
    device_gram = jnp.asarray(gram)
    device_law = tuple(jnp.asarray(item) for item in law)

    def operands(drive):
        return (device_gram, jnp.asarray(drive * external.T.reshape(-1)), *device_law)

    def residual(m, drive):
        f = iron._residual(jnp.asarray(m.T.reshape(-1)), operands(drive))
        return float(jnp.linalg.norm(f)) / max(
            np.linalg.norm(m), drive * np.linalg.norm(external) / iron._MU0, 1e-30
        )

    record = dict(
        revision=revision,
        module=iron.__file__,
        cwd=str(Path.cwd().resolve()),
        source_sha256=hashlib.sha256(blob).hexdigest(),
        device=jax.devices()[0].device_kind,
        matrix_builds=0,
        saved=str(SAVED),
        law_identical_to_saved_newton=True,
        cells=len(seed._mesh.centroids),
        steps=[],
        attempts=[],
        spectra={},
    )
    record["replay"] = {name: residual(m, 1.0) for name, m in roots.items()}
    print("REPRODUCTION " + json.dumps(record["replay"]), flush=True)
    assert max(record["replay"].values()) < 1e-8
    delta = np.linalg.norm(roots["A"] - roots["B"], axis=1)
    record["root_pair_max_difference_A_per_m"] = float(delta.max())
    assert delta.max() > 1e6
    known = jnp.diag(jnp.asarray([-2.0, 1.0, 3.0, 4.0]))
    ev, _, errors = extremes(known)
    np.testing.assert_allclose(ev, [-2.0, 4.0], atol=1e-10)
    control = singular_minimum(
        jnp.asarray(
            [
                [2.0, 1.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 5.0],
            ]
        )
    )
    np.testing.assert_allclose(
        control["value"],
        np.linalg.svd(np.asarray([[2.0, 1.0], [0.0, 3.0]]), compute_uv=False)[-1],
        rtol=1e-10,
    )
    assert ev[0] < 0
    print(
        "INSTRUMENT_CONTROL known negative direction detected; "
        "known nonsymmetric singular value reproduced",
        flush=True,
    )
    record["instrument_control"] = dict(
        extremes=ev.tolist(), residuals=errors, singular=control
    )
    write_json(out / "measurement.json", record)
    machine = StaticMachineDescription.read_iron_core(
        "imas:hdf5?path=" + receipt["machine"], dd_version=receipt["dd_version"]
    )
    bodies = [
        shapely.Polygon(np.column_stack((s.outline.r, s.outline.z)))
        for s in machine.iron_segments
    ]
    segments = np.asarray(
        [
            next(s for s, body in enumerate(bodies) if body.covers(shapely.Point(p)))
            for p in seed._mesh.centroids
        ]
    )
    strip_cells = np.load(SMOOTH / "strip-column.npz")["cells"]
    strip = np.zeros(len(segments), dtype=bool)
    strip[strip_cells] = True
    areas = seed._mesh.areas
    for label, weights in [
        ("area", areas),
        ("axisymmetric_volume", areas * seed._mesh.centroids[:, 0]),
    ]:
        scale = jnp.asarray(np.tile(np.sqrt(weights), 2))
        weighted = device_gram * scale[:, None] / scale[None, :]
        record["G_" + label] = dict(
            asymmetry=float(
                jnp.linalg.norm(weighted - weighted.T) / jnp.linalg.norm(weighted)
            ),
            skew_fraction=float(
                jnp.linalg.norm((weighted - weighted.T) / 2) / jnp.linalg.norm(weighted)
            ),
            symmetric_fraction=float(
                jnp.linalg.norm((weighted + weighted.T) / 2) / jnp.linalg.norm(weighted)
            ),
        )
        values, _, defects = extremes((weighted + weighted.T) / (2 * iron._MU0))
        record["G_" + label].update(
            symmetric_extremes_over_mu0=values.tolist(), eigen_residuals=defects
        )
        print("G_METRIC " + label + " " + json.dumps(record["G_" + label]), flush=True)
    write_json(out / "measurement.json", record)
    # Reuse saved cells and collocation operands through the production Newton.
    iron._mesh = lambda region, size: seed._mesh
    calls = []

    def interaction(cells, points, plan):
        assert cells is seed._mesh
        np.testing.assert_array_equal(points, seed._mesh.centroids)
        calls.append("saved")
        return gram

    iron._interaction = interaction
    current = np.zeros_like(external)
    record["steps"].append(
        dict(
            drive=0.0, iterations=0, residual=residual(current, 0.0), max_M_A_per_m=0.0
        )
    )
    drive, increment = 0.0, 0.05
    while drive < 1 - 1e-12:
        target = min(1.0, drive + increment)
        start = time.monotonic()
        result = iron.solve_magnetisation(
            None,
            lambda points: target * external,
            receipt["table_b_T"],
            receipt["table_mu_relative"],
            cell_size=receipt["cell_size_m"],
            tolerance=1e-10,
            max_iterations=400,
            initial_magnetisation=current,
        )
        entry = dict(
            drive=target,
            from_drive=drive,
            increment=target - drive,
            iterations=result.iterations,
            residual=result.residual_history[-1],
            max_M_A_per_m=float(np.linalg.norm(result.magnetisation, axis=1).max()),
            converged=result.converged,
            wall_seconds=time.monotonic() - start,
            max_change_A_per_m=float(
                np.linalg.norm(result.magnetisation - current, axis=1).max()
            ),
            regularised_steps=int(np.count_nonzero(result.regularisation_history)),
        )
        record["attempts"].append(entry)
        print("RAMP_ATTEMPT " + json.dumps(entry), flush=True)
        if not result.converged or (result.iterations > 40 and increment > 0.0002):
            increment /= 2
            write_json(out / "measurement.json", record)
            assert increment >= 0.00001, (
                "Ramp cannot continue within the declared drive refinement"
            )
            continue
        current, drive = result.magnetisation, target
        record["steps"].append(entry)
        np.save(out / "ramp-latest.npy", current)
        write_json(out / "measurement.json", record)
        if result.iterations < 10:
            increment = min(0.05, increment * 1.5)
    roots["C"] = current
    np.save(out / "root-C.npy", current)
    air = np.asarray(receipt["air_field_T"])
    record["probes_m"] = receipt["probes_m"]
    record["roots"] = {}
    fields = {}
    for name, m in roots.items():
        fields[name] = air + (probe @ m.T.reshape(-1)).reshape(2, -1).T
        record["roots"][name] = dict(
            residual=residual(m, 1.0),
            max_M_A_per_m=float(np.linalg.norm(m, axis=1).max()),
            field_T=fields[name].tolist(),
            field_magnitude_T=np.linalg.norm(fields[name], axis=1).tolist(),
        )
    record["comparisons"] = {}
    for name in ("A", "B"):
        record["comparisons"]["C_vs_" + name] = dict(
            max_difference_A_per_m=float(
                np.linalg.norm(current - roots[name], axis=1).max()
            ),
            relative_field_change=(
                np.linalg.norm(fields["C"] - fields[name], axis=1)
                / np.linalg.norm(fields[name], axis=1)
            ).tolist(),
        )
    np.savez(
        out / "strip-roots.npz",
        z=seed._mesh.centroids[strip_cells, 1],
        cells=strip_cells,
        **{name: m[strip_cells] for name, m in roots.items()},
    )
    write_json(out / "measurement.json", record)
    jacobian = jax.jit(jax.jacfwd(iron._residual))
    for name, m, strength in [
        ("zero", np.zeros_like(current), 0.0),
        *[(name, m, 1.0) for name, m in roots.items()],
    ]:
        start = time.monotonic()
        matrix = jacobian(jnp.asarray(m.T.reshape(-1)), operands(strength))
        symmetric = (matrix + matrix.T) / 2
        values, vectors, defects = extremes(symmetric)
        spectrum = dict(
            symmetric_extremes=values.tolist(),
            eigen_residuals=defects,
            asymmetry=float(
                jnp.linalg.norm(matrix - matrix.T) / jnp.linalg.norm(matrix)
            ),
            smallest_singular=singular_minimum(matrix),
            negative_direction=locate(vectors[:, 0], seed._mesh, segments, strip),
        )
        # The residual is preconditioned by a local constitutive derivative;
        # report the area inner product separately rather than equating raw J
        # with a magnetic energy Hessian.
        scale = jnp.asarray(np.tile(np.sqrt(areas), 2))
        weighted = matrix * scale[:, None] / scale[None, :]
        weighted_values, _, weighted_defects = extremes((weighted + weighted.T) / 2)
        spectrum.update(
            area_weighted_extremes=weighted_values.tolist(),
            area_weighted_eigen_residuals=weighted_defects,
            wall_seconds=time.monotonic() - start,
        )
        record["spectra"][name] = spectrum
        np.save(out / ("negative-direction-" + name + ".npy"), vectors[:, 0])
        write_json(out / "measurement.json", record)
        print("SPECTRUM " + name + " " + json.dumps(spectrum), flush=True)
    record["complete"] = True
    record["saved_operator_uses"] = len(calls)
    assert len(record["spectra"]) == 4 and record["steps"][-1]["drive"] == 1.0
    assert Path(iron.__file__).read_bytes() == blob
    write_json(out / "measurement.json", record)
    print(
        "SUMMARY "
        + json.dumps(
            dict(
                complete=True,
                accepted_steps=len(record["steps"]) - 1,
                attempts=len(record["attempts"]),
                roots=record["roots"],
                comparisons=record["comparisons"],
                matrix_builds=0,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
