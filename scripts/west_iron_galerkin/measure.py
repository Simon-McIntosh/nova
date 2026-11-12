"""Measure reciprocal iron operators, analytic controls and saved WEST roots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402
from nova.biot.tiledassembly import galerkin_cell_interaction  # noqa: E402
from nova.imas.machine import StaticMachineDescription  # noqa: E402
from tests.test_magnetisation import sphere, shell, uniform, loop  # noqa: E402

RUNS = Path("/home/ITER/mcintos/.config/reckon/crew/runs")
SAVED = RUNS / "r-20261111T130417404611-icm-west-62751-solve/measurement-pair/0.0566"
PRIOR = RUNS / "r-20261111T205006716359-icm-newton-trust-region"
SMOOTH = RUNS / "r-20261112T000127065672-icm-magnetisation-smoothness"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def spectrum(gram, cells, output, name):
    volume = np.tile(2 * np.pi * cells.areas * cells.centroids[:, 0], 2)
    weighted_energy = volume[:, None] * gram
    symmetry = np.linalg.norm(weighted_energy - weighted_energy.T) / np.linalg.norm(
        weighted_energy
    )
    scale = jnp.asarray(np.sqrt(volume))
    weighted = jnp.asarray(gram) / iron._MU0 * scale[:, None] / scale[None, :]
    eigenvalues = np.asarray(jnp.linalg.eigvalsh((weighted + weighted.T) / 2))
    np.save(output / (name + "-eigenvalues.npy"), eigenvalues)
    result = dict(
        cells=len(cells.centroids),
        weighted_symmetry=float(symmetry),
        minimum=float(eigenvalues[0]),
        maximum=float(eigenvalues[-1]),
        tolerance=2e-7,
    )
    print("SPECTRUM " + name + " " + json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source = Path(iron.__file__).read_bytes()
    print(
        f"# revision: {revision} tree: {Path.cwd().resolve()} "
        f"command: measure.py --output {out}",
        flush=True,
    )
    print(
        f"MODULE={iron.__file__} CWD={Path.cwd().resolve()} DEVICE={jax.devices()}",
        flush=True,
    )
    assert jax.config.jax_enable_x64 and "P100" in jax.devices()[0].device_kind
    record = dict(
        revision=revision,
        source_sha256=hashlib.sha256(source).hexdigest(),
        module=iron.__file__,
        cwd=str(Path.cwd().resolve()),
        device=jax.devices()[0].device_kind,
        spectra={},
        ramp=[],
        starts={},
        analytic=[],
    )

    def persist():
        save(out / "measurement.json", record)

    receipt = json.loads((SAVED / "receipt.json").read_text())
    with (SAVED / "solution.pkl").open("rb") as stream:
        seed = pickle.load(stream)
    with (PRIOR / "solution.pkl").open("rb") as stream:
        old_a = pickle.load(stream).magnetisation
    roots = dict(A=old_a, B=np.load(SMOOTH / "west-0.0001-M.npy"))
    np.testing.assert_array_equal(old_a, np.load(SMOOTH / "baseline-M.npy"))
    external = np.load(PRIOR / "external.npy")
    old = np.load(SAVED / "gram.npy")
    assert old.shape == (external.size, external.size)
    assert np.isfinite(old).all() and np.max(np.abs(old)) > 0
    record["spectra"]["collocation_saved"] = spectrum(
        old, seed._mesh, out, "collocation-saved"
    )
    assert record["spectra"]["collocation_saved"]["minimum"] < -0.01
    assert record["spectra"]["collocation_saved"]["maximum"] > 1.01
    persist()
    print("BUILD saved mesh " + str(len(seed._mesh.centroids)), flush=True)
    started = time.monotonic()
    gram = iron._interaction(seed._mesh, None, iron._PLAN)
    record["build_seconds"] = time.monotonic() - started
    np.save(out / "galerkin-gram.npy", gram)
    record["spectra"]["galerkin_saved"] = spectrum(
        gram, seed._mesh, out, "galerkin-saved"
    )
    record["external_projection"] = (
        "Saved external operands define a piecewise constant prescribed field "
        "on the saved cells; their volume averages equal the saved vectors. "
        "No interpolation or new source build is introduced."
    )
    persist()
    del old
    original_mesh, original_interaction, original_external = (
        iron._mesh,
        iron._interaction,
        iron._average_external,
    )
    iron._mesh = lambda region, size: seed._mesh
    iron._interaction = lambda cells, points, plan: gram
    iron._average_external = lambda cells, callback: callback(cells.centroids)
    current = np.zeros_like(external)
    for drive in np.linspace(0, 1, 21):
        result = iron.solve_magnetisation(
            None,
            lambda points: drive * external,
            receipt["table_b_T"],
            receipt["table_mu_relative"],
            cell_size=receipt["cell_size_m"],
            max_iterations=100,
            tolerance=1e-10,
            initial_magnetisation=current,
        )
        entry = dict(
            drive=float(drive),
            iterations=result.iterations,
            residual=result.residual_history[-1],
            maximum_M_A_per_m=float(np.linalg.norm(result.magnetisation, axis=1).max()),
            converged=result.converged,
        )
        record["ramp"].append(entry)
        print("RAMP " + json.dumps(entry), flush=True)
        persist()
        assert result.converged, entry
        current = result.magnetisation
        np.save(out / "ramp-M.npy", current)
    for name, initial in dict(zero=np.zeros_like(external), **roots).items():
        result = iron.solve_magnetisation(
            None,
            lambda points: external,
            receipt["table_b_T"],
            receipt["table_mu_relative"],
            cell_size=receipt["cell_size_m"],
            max_iterations=100,
            tolerance=1e-10,
            initial_magnetisation=initial,
        )
        relative = (
            np.linalg.norm(result.magnetisation - current, axis=1).max()
            / np.linalg.norm(current, axis=1).max()
        )
        record["starts"][name] = dict(
            iterations=result.iterations,
            residual=result.residual_history[-1],
            converged=result.converged,
            relative_root_difference=float(relative),
        )
        print("START " + name + " " + json.dumps(record["starts"][name]), flush=True)
        persist()
        assert result.converged and relative <= 1e-6
    probe = np.load(SMOOTH / "probe-response.npy")
    assert probe.shape == (6, external.size) and np.max(np.abs(probe)) > 0
    air = np.asarray(receipt["air_field_T"])
    record["far_field"] = {
        name: dict(
            field_T=(air + (probe @ m.T.reshape(-1)).reshape(2, -1).T).tolist(),
            magnitude_T=np.linalg.norm(
                air + (probe @ m.T.reshape(-1)).reshape(2, -1).T, axis=1
            ).tolist(),
        )
        for name, m in dict(A=roots["A"], B=roots["B"], Galerkin=current).items()
    }
    record["probes_m"] = receipt["probes_m"]
    operands = tuple(
        jnp.asarray(x)
        for x in (
            gram,
            external.T.reshape(-1),
            *iron._fit_law(receipt["table_b_T"], receipt["table_mu_relative"]),
        )
    )
    jacobian = jax.jit(jax.jacfwd(iron._residual))
    record["jacobian"] = {}
    for name, m in dict(zero=np.zeros_like(current), root=current).items():
        matrix = jacobian(jnp.asarray(m.T.reshape(-1)), operands)
        eigenvalues = np.asarray(jnp.linalg.eigvalsh((matrix + matrix.T) / 2))
        record["jacobian"][name] = dict(
            symmetric_minimum=float(eigenvalues[0]),
            symmetric_maximum=float(eigenvalues[-1]),
        )
        volume_scale = jnp.asarray(
            np.tile(np.sqrt(seed._mesh.areas * seed._mesh.centroids[:, 0]), 2)
        )
        weighted = matrix * volume_scale[:, None] / volume_scale[None, :]
        weighted_values = np.asarray(jnp.linalg.eigvalsh((weighted + weighted.T) / 2))
        record["jacobian"][name]["volume_weighted_symmetric_minimum"] = float(
            weighted_values[0]
        )
        print(
            "JACOBIAN " + name + " " + json.dumps(record["jacobian"][name]), flush=True
        )
        persist()
    iron._mesh, iron._interaction, iron._average_external = (
        original_mesh,
        original_interaction,
        original_external,
    )
    machine = StaticMachineDescription.read_iron_core(
        "imas:hdf5?path=" + receipt["machine"], dd_version=receipt["dd_version"]
    )
    region = shapely.union_all(
        [
            shapely.Polygon(np.column_stack((s.outline.r, s.outline.z)))
            for s in machine.iron_segments
        ]
    )
    cells = iron._mesh(region, 0.08)
    response = iron._interaction(cells, None, iron._PLAN)
    record["spectra"]["west_008"] = spectrum(response, cells, out, "west-008")
    persist()
    del response, operands, matrix
    # Each geometry is assembled once while the linear permeability varies.
    cached, production = {}, iron.galerkin_cell_interaction

    def interaction(polygons, *, plan):
        key = tuple(p.tobytes() for p in polygons)
        if key not in cached:
            cached[key] = production(polygons, plan=plan)
        return cached[key]

    iron.galerkin_cell_interaction = interaction
    for label, region, size, field in [
        ("sphere", sphere(), 0.1, uniform),
        ("shell", shell(), 0.04, loop),
        ("shell", shell(), 0.02, loop),
    ]:
        for mu in (100, 1000):
            result = iron.solve_magnetisation(
                region, field, [0, 10], [mu, mu], cell_size=size
            )
            assert result.converged
            points = (
                np.array([[1.0, 0.0], [0.0, 1.0]])
                if label == "sphere"
                else np.array([[2.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
            )
            total = field(points) + result.field(points)
            if label == "sphere":
                fraction = (mu - 1) / (mu + 2)
                expected = 0.01 * np.array([1 - fraction / 8, 1 + fraction / 4])
                error = np.abs(total[:, 1] / expected - 1)
                inside = np.linalg.norm(result._mesh.centroids, axis=1) < 0.3
                interior = field(result._mesh.centroids) + result.field(
                    result._mesh.centroids
                )
                extra = dict(
                    interior_error=float(
                        np.max(
                            np.abs(interior[inside, 1] / (3 * mu / (mu + 2) * 0.01) - 1)
                        )
                    )
                )
            else:
                expected = ((2 * mu + 1) * (mu + 2) - 2 * (mu - 1) ** 2 * 0.9**3) / (
                    9 * mu
                )
                shielding = np.linalg.norm(field(points), axis=1) / np.linalg.norm(
                    total, axis=1
                )
                error = np.abs(shielding / expected - 1)
                extra = dict(shielding=shielding.tolist())
            entry = dict(
                shape=label,
                size=size,
                mu=mu,
                errors=error.tolist(),
                iterations=result.iterations,
                **extra,
            )
            record["analytic"].append(entry)
            print("ANALYTIC " + json.dumps(entry), flush=True)
            if mu == 100:
                scaled = tuple(
                    p * iron._SHEET_LENGTH_SCALE for p in result._mesh.polygons
                )
                g = interaction(scaled, plan=iron._PLAN)
                record["spectra"][label + str(size)] = spectrum(
                    g, result._mesh, out, label + str(size)
                )
            persist()
    coarse = iron._mesh(shell(), 0.04)
    scaled = tuple(p * iron._SHEET_LENGTH_SCALE for p in coarse.polygons)
    response = galerkin_cell_interaction(scaled, plan=iron._PLAN, order=32)
    iron.galerkin_cell_interaction = lambda polygons, plan: response
    check = iron.solve_magnetisation(
        shell(), loop, [0, 10], [1000, 1000], cell_size=0.04
    )
    probes = np.array([[2.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
    exact = ((2001 * 1002) - 2 * 999**2 * 0.9**3) / 9000
    shielding = np.linalg.norm(loop(probes), axis=1) / np.linalg.norm(
        loop(probes) + check.field(probes), axis=1
    )
    errors = np.abs(shielding / exact - 1)
    regular = next(
        row
        for row in record["analytic"]
        if row["shape"] == "shell" and row["size"] == 0.04 and row["mu"] == 1000
    )
    change = errors - np.asarray(regular["errors"])
    record["shell_quadrature_check"] = dict(
        order=32, errors=errors.tolist(), change_from_order24=change.tolist()
    )
    print(
        "QUADRATURE_CHECK " + json.dumps(record["shell_quadrature_check"]), flush=True
    )
    assert np.max(np.abs(change)) < 2e-4
    iron.galerkin_cell_interaction = production
    record["analytic_no_worse"] = False
    record["acceptance_limitation"] = (
        "The coarse high-permeability shell error is larger than collocation; "
        "the sphere centroid-uniformity test also changes. "
        "No existing analytic tolerance is relaxed."
    )
    persist()
    for name, value in record["spectra"].items():
        if name.startswith("collocation"):
            continue
        assert value["weighted_symmetry"] < 5e-14, (name, value)
        assert value["minimum"] >= -2e-7 and value["maximum"] <= 1 + 2e-7, (name, value)
    assert Path(iron.__file__).read_bytes() == source
    record["complete"] = True
    persist()
    print("SUMMARY " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
