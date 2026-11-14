"""Assemble gradient comparisons and physical trace residuals at retained nodes."""

# ruff: noqa: E402
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.greens import MU0
from scripts.prototypes.bank_hessian_diagnosis.measure import ROOT, STATES, reference


def relative(delta, scale):
    return np.sqrt(np.sum(delta**2, axis=0) / np.sum(scale**2, axis=0))


def bank_input():
    prior = json.loads(
        (Path.home() / ".cache/nova-hessian-reference/inputs.json").read_text()
    )
    item = next(row for row in prior if row["path"].endswith("/bank.npz"))
    target = ROOT / "bank.npz"
    if not target.exists():
        shutil.copy2(item["path"], target)
    digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
    assert digest == item["sha256"]
    records = json.loads((ROOT / "inputs.json").read_text())
    record = dict(source=item["path"], copy=str(target), sha256=digest)
    if record not in records:
        records.append(record)
    (ROOT / "inputs.json").write_text(json.dumps(records, indent=2))
    return dict(np.load(target))


def density(points, polygons, packed, moments):
    """Evaluate the authored piecewise-linear current density, including vacuum."""
    from matplotlib.path import Path as PolygonPath

    masks = np.stack([PolygonPath(p).contains_points(points) for p in polygons], axis=1)
    assert np.all(masks.sum(axis=1) <= 1)
    area = np.array(
        [
            abs(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)))
            / 2
            for p in polygons
        ]
    )
    coefficient = np.concatenate(
        (
            moments[:, :, :1],
            np.einsum(
                "cij,scj->sci", packed["inverse_second_moment"], moments[:, :, 1:]
            ),
        ),
        axis=-1,
    )
    basis = np.concatenate(
        (np.ones((*masks.shape, 1)), points[:, None] - packed["centre"][None]), axis=-1
    )
    values = np.einsum("pcm,scm->spc", basis, coefficient) / area
    return np.sum(values * masks[None], axis=-1), masks.sum(axis=1)


def coil_reference(points, coils, moments):
    """Measure the same stencil on a field with no plasma or material at targets."""
    target = ROOT / "coil-stencils.npz"
    if target.exists():
        return dict(np.load(target))
    ref = reference()
    gradient, hessian, bound, direct_gradient, direct_hessian = [], [], [], [], []
    from nova.biot.polygonanalytic import _packed_analytic_moments_primal

    device = {k: jnp.asarray(v) for k, v in coils.items()}
    nearest_height = np.min(
        abs(coils["edge"][:, (1, 3), None, :] - points[None, None, :, 1, None])
    )
    assert nearest_height > 1e-7, "coil sample requires the height tangent branch"

    def uniform_gradient(point):
        rows = _packed_analytic_moments_primal(
            jnp,
            jnp.broadcast_to(point[0], device["norm"].shape),
            jnp.broadcast_to(point[1], device["norm"].shape),
            device["edge"],
            device["weight"],
            device["norm"],
            device["centre"].T,
            device["centre"].T,
            device["reflection_axis"],
            device["reflection_partner"],
        )
        return 2 * jnp.pi * point[0] * jnp.stack((rows[6], -rows[3]), axis=-1)

    direct_value = jax.jit(uniform_gradient)
    direct_derivative = jax.jit(jax.jacfwd(uniform_gradient))
    for i, point in enumerate(points):
        shifts = [np.array(point, dtype=np.longdouble)]
        steps = np.array([0.001, 0.0005, 0.00025], dtype=np.longdouble)
        for axis in range(2):
            for step in steps:
                for sign in (1, -1):
                    p = np.array(point, dtype=np.longdouble)
                    p[axis] += sign * step
                    shifts.append(p)
        values = np.concatenate(
            [
                ref.gradient(np.array(shifts[first : first + 2]), coils, moments)
                for first in range(0, len(shifts), 2)
            ],
            axis=1,
        )
        raw = (values[:, 1::2] - values[:, 2::2]).reshape(3, 2, 3, 2)
        raw /= 2 * steps[None, None, :, None]
        fourth = (4 * raw[:, :, 1:] - raw[:, :, :-1]) / 3
        sixth = (16 * fourth[:, :, 1] - fourth[:, :, 0]) / 15
        gradient.append(values[:, 0])
        hessian.append(sixth.transpose(0, 2, 1))
        bound.append((4 * abs(sixth - fourth[:, :, 1])).transpose(0, 2, 1))
        direct_gradient.append(
            np.einsum(
                "cd,sc->sd",
                np.asarray(direct_value(jnp.asarray(point))),
                moments[:, :, 0],
            )
        )
        direct_hessian.append(
            np.einsum(
                "cij,sc->sij",
                np.asarray(direct_derivative(jnp.asarray(point))),
                moments[:, :, 0],
            )
        )
        print("COIL_NODE", i, flush=True)
    result = dict(
        gradient=np.array(gradient).transpose(1, 0, 2),
        hessian=np.array(hessian).transpose(1, 0, 2, 3),
        bound=np.array(bound).transpose(1, 0, 2, 3),
        direct_gradient=np.array(direct_gradient).transpose(1, 0, 2),
        direct_hessian=np.array(direct_hessian).transpose(1, 0, 2, 3),
    )
    np.savez(target, **result)
    return result


def jet(gradient, hessian):
    return np.concatenate((gradient, hessian[:, 0], hessian[:, 1, 1, None]), axis=-1)


def trace(values, points, forcing):
    terms = np.stack(
        (values[:, 2], values[:, 4], -values[:, 0] / points[:, 0]), axis=-1
    )
    residual = terms.sum(axis=-1) - forcing
    scale = max(float(np.linalg.norm(forcing)), float(np.linalg.norm(terms)))
    return dict(
        residual=residual.tolist(),
        relative=float(np.linalg.norm(residual) / scale),
        maximum_scaled=float(np.max(np.abs(residual)) / scale),
        scale=scale,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutate", action="store_true")
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    ref = reference()
    geometry = dict(np.load(ROOT / "h200/geometry.npz"))
    polygons = tuple(
        p[:n]
        for p, n in zip(geometry["vertices"], geometry["vertex_count"], strict=True)
    )
    packed = ref.coupling(polygons)
    split = json.loads((ROOT / "split.json").read_text())
    indices = np.array([row["node"] for row in split["nodes"] if row["interior"]])
    old = dict(np.load(ROOT / "measurement/qualified-reference.npz"))
    old_index = np.array([np.flatnonzero(old["indices"] == i)[0] for i in indices])
    nodes = [dict(np.load(ROOT / f"node-{i}.npz")) for i in indices]
    plasma_grad = np.stack([a["gradient"] for a in nodes], axis=1)
    plasma_hess = np.stack([a["hessian"] for a in nodes], axis=1)
    plasma_bound = np.stack([a["bound"] for a in nodes], axis=1)
    archives = [dict(np.load(ROOT / f"diagnosis/accuracy-{s}.npz")) for s in STATES]
    points = archives[0]["points"][indices]
    bank = bank_input()
    np.testing.assert_array_equal(bank["nodes"][indices], points)
    currents = np.stack([a["coil_currents"] for a in archives])
    coil_bank = np.einsum("nkq,sk->snq", bank["coil_exterior"][indices], currents)
    coil_moments = np.zeros((3, len(geometry["turns"]), 3))
    coil_moments[:, :, 0] = currents[:, geometry["owners"]] * geometry["turns"]
    coils = ref.coupling(
        tuple(geometry[f"coil_{i}"] for i in range(len(geometry["turns"])))
    )
    coil_ref = coil_reference(points, coils, coil_moments)
    moments = np.stack([a["moments"] for a in archives])
    densities, owner_count = density(points, polygons, packed, moments)
    forcing = -2 * np.pi * MU0 * points[None, :, 0] * densities
    rows, arrays = [], dict(indices=indices, points=points, density=densities)
    for s, state in enumerate(STATES):
        np.testing.assert_array_equal(archives[s]["points"][indices], points)
        plasma = jet(plasma_grad[s], plasma_hess[s])
        coil = jet(coil_ref["gradient"][s], coil_ref["hessian"][s])
        total = plasma + coil
        methods = dict(
            bank=archives[s]["actual"][indices, 1:],
            direct=archives[s]["expected"][indices, 1:],
            reference=total,
        )
        row = dict(
            state=state,
            count=len(indices),
            gradient_relative={},
            trace={},
            coil_trace={},
            plasma_trace={},
            hessian_relative={},
            plasma_gradient_relative={},
        )
        row["reproduced_prior_total_hessian"] = relative(
            methods["bank"][:, 3:5]
            - old["plasma"][s, old_index]
            - old["coils"][s, old_index],
            old["plasma"][s, old_index] + old["coils"][s, old_index],
        ).tolist()
        np.testing.assert_allclose(
            row["reproduced_prior_total_hessian"],
            split["states"][s]["interior"]["bank"],
            rtol=1e-12,
            atol=1e-14,
        )
        row["fresh_vs_prior_plasma_hessian"] = relative(
            plasma[:, 3:5] - old["plasma"][s, old_index], old["plasma"][s, old_index]
        ).tolist()
        row["reference_cross_derivative_asymmetry"] = float(
            relative(
                plasma_hess[s, :, 0, 1] - plasma_hess[s, :, 1, 0],
                plasma_hess[s, :, 0, 1],
            )
        )
        row["reference_bound_relative"] = relative(
            jet(np.zeros_like(plasma_grad[s]), plasma_bound[s])[:, 2:], plasma[:, 2:]
        ).tolist()
        for name, values in methods.items():
            row["gradient_relative"][name] = relative(
                values[:, :2] - total[:, :2], total[:, :2]
            ).tolist()
            row["hessian_relative"][name] = relative(
                values[:, 2:] - total[:, 2:], total[:, 2:]
            ).tolist()
            row["trace"][name] = trace(values, points, forcing[s])
            direct_coil = jet(
                coil_ref["direct_gradient"][s], coil_ref["direct_hessian"][s]
            )
            coil_method = {
                "bank": coil_bank[s, :, 1:],
                "direct": direct_coil,
                "reference": coil,
            }[name]
            source = values - coil_method if name != "reference" else plasma
            row["plasma_gradient_relative"][name] = relative(
                source[:, :2] - plasma[:, :2], plasma[:, :2]
            ).tolist()
            row["plasma_trace"][name] = trace(source, points, forcing[s])
            row["coil_trace"][name] = trace(
                coil_method,
                points,
                np.zeros(len(points)),
            )
            arrays[f"{state}_{name}"] = values
        row["coil_gradient_relative"] = relative(
            coil_bank[s, :, 1:3] - coil[:, :2], coil[:, :2]
        ).tolist()
        row["coil_hessian_relative"] = relative(
            coil_bank[s, :, 3:] - coil[:, 2:], coil[:, 2:]
        ).tolist()
        row["direct_coil_gradient_relative"] = relative(
            direct_coil[:, :2] - coil[:, :2], coil[:, :2]
        ).tolist()
        row["direct_coil_hessian_relative"] = relative(
            direct_coil[:, 2:] - coil[:, 2:], coil[:, 2:]
        ).tolist()
        positive = trace(total, points, forcing[s])["relative"]
        mutated = total.copy()
        mutated[:, 4] *= np.longdouble("1.000001")
        negative = trace(mutated, points, forcing[s])["relative"]
        row["control"] = dict(
            threshold=1e-8,
            positive=positive,
            negative=negative,
            positive_pass=positive < 1e-8,
            negative_refused=negative > 1e-8,
        )
        tested = mutated if args.mutate else total
        row["guard_accepted"] = trace(tested, points, forcing[s])["relative"] < 1e-8
        rows.append(row)
        print("STATE", json.dumps(row, default=float), flush=True)
    output = dict(
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        states=rows,
        node_owner_count=owner_count.tolist(),
        input_manifest=str(ROOT / "inputs.json"),
        node_receipts=[str(ROOT / f"node-{i}.npz") for i in indices],
        cells=json.loads((ROOT / "cell.json").read_text()),
    )
    from nova.biot import polygonanalytic
    import inspect

    output["localization"] = dict(
        function="_packed_moments_height_tangent",
        file="nova/biot/polygonanalytic.py",
        line=inspect.getsourcelines(polygonanalytic._packed_moments_height_tangent)[1],
        source_sha256=hashlib.file_digest(
            Path(polygonanalytic.__file__).open("rb"), "sha256"
        ).hexdigest(),
        mechanism=(
            "ordinary autodiff of contour pole reductions near a source edge's "
            "extended line; the height-only tangent guard misses this geometry"
        ),
        terms=json.loads((ROOT / "terms.json").read_text()),
        projected=json.loads((ROOT / "projected.json").read_text()),
        quadrature_map_experiment=json.loads((ROOT / "localize.json").read_text()),
    )
    for cell in output["cells"]:
        truth = np.array(cell["independent"])
        cell["relative"] = {
            name: (abs(np.array(cell[name]) - truth) / abs(truth)).tolist()
            for name in ("bank", "direct", "reference")
        }
        assert max(cell["relative"]["reference"]) < 1e-8
    projected = output["localization"]["projected"]
    for cell in projected:
        truth = np.array(cell["independent"])
        cell["bank_relative"] = (
            abs(np.array(cell["bank"])[4:6] - truth) / abs(truth)
        ).tolist()
        cell["reference_relative"] = (
            abs(np.array(cell["reference"]) - truth) / abs(truth)
        ).tolist()
        assert max(cell["reference_relative"]) < 1e-8
    assert projected[0]["bank_relative"][0] > 1e-4
    assert max(projected[1]["bank_relative"]) < 1e-6
    output["verdict"] = (
        "The bank and direct-kernel plasma Hessians carry the error; "
        "the source-gradient Richardson reference is corroborated."
    )
    path = ROOT / ("mutation.json" if args.mutate else "receipt.json")
    path.write_text(json.dumps(output, indent=2, default=float))
    np.savez(ROOT / "comparison.npz", **arrays)
    assert all(row["control"]["positive_pass"] for row in rows), "source trace fails"
    assert all(
        max(row["gradient_relative"][name]) < 1e-10
        for row in rows
        for name in ("bank", "direct")
    ), "source gradients differ"
    assert all(
        row["trace"][name]["relative"] > 1e-3
        for row in rows
        for name in ("bank", "direct")
    ), "route trace defect absent"
    assert all(row["control"]["negative_refused"] for row in rows), (
        "trace mutation escaped"
    )
    assert all(max(row["coil_hessian_relative"]) < 1e-8 for row in rows), (
        "coil positive control fails"
    )
    assert all(max(row["direct_coil_hessian_relative"]) < 1e-8 for row in rows), (
        "direct coil positive control fails"
    )
    assert all(
        row["coil_trace"][name]["relative"] < 1e-8
        for row in rows
        for name in ("bank", "direct", "reference")
    ), "coil vacuum trace fails"
    assert all(row["guard_accepted"] for row in rows), (
        "psi_ZZ relative perturbation refused by trace identity"
    )
    summary = [
        dict(
            state=row["state"],
            gradient=row["gradient_relative"],
            plasma_gradient=row["plasma_gradient_relative"],
            trace={name: value["relative"] for name, value in row["trace"].items()},
            hessian=row["hessian_relative"],
            perturbed_trace=row["control"]["negative"],
        )
        for row in rows
    ]
    print("DIAGNOSIS_SUMMARY", json.dumps(summary), flush=True)
    print(
        "DIAGNOSIS_COMPLETE", len(indices), "interior nodes; three states", flush=True
    )


if __name__ == "__main__":
    main()
