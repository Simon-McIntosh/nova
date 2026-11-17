"""Retain the independent rectangle and production-coil value controls."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import json
from pathlib import Path
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot import polygonanalytic
from nova.equilibrium import topology
from scripts.prototypes.bank_hessian_diagnosis.measure import ROOT, STATES
from scripts.prototypes.bank_hessian_diagnosis.report import jet
from scripts.prototypes.polygon_trace_miss.measure import evaluate
from tests.biot.test_kernel_compile_cost import _baseline_kernel

assert jax.config.jax_enable_x64 is True
jax.config.update("jax_enable_compilation_cache", False)


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    receipt = {"completed": False}
    saved = json.loads((ROOT / "cell.json").read_text())[:2]
    points = np.array([row["point"] for row in saved])
    truth = np.array([row["independent"] for row in saved], dtype=np.longdouble)
    polygon = np.array([[5.5, -0.6], [6.5, -0.6], [6.5, 0.6], [5.5, 0.6]])
    gradient, hessian = evaluate(
        topology.BiotMomentCoupling.from_polygons((polygon,)),
        points,
        np.array([[[240000.0, 0.0, 0.0]]]),
        output,
        "rectangle",
    )
    measured = jet(gradient[0], hessian[0])
    assert np.isfinite(measured).all() and np.all(truth != 0)
    error = np.max(abs((measured - truth) / truth), axis=0)
    control = np.max(abs((1.000001 * measured - truth) / truth), axis=0)
    receipt["rectangle"] = dict(
        points=points.tolist(),
        component_relative=error.tolist(),
        control_relative=control.tolist(),
        reference=str(ROOT / "cell.json"),
    )
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, default=float))
    print(
        "RECTANGLE_CONTROL", json.dumps(receipt["rectangle"], default=float), flush=True
    )
    assert np.max(error) <= 1e-12
    assert np.min(control) > 1e-12

    geometry = dict(np.load(ROOT / "h200/geometry.npz"))
    archives = [dict(np.load(ROOT / f"diagnosis/accuracy-{s}.npz")) for s in STATES]
    split = json.loads((ROOT / "split.json").read_text())
    selected = [row["node"] for row in split["nodes"] if row["interior"]]
    points = archives[0]["points"][selected]
    coils = tuple(geometry[f"coil_{i}"] for i in range(len(geometry["turns"])))
    coupling = topology.BiotMomentCoupling.from_polygons(coils)
    moments = np.zeros((3, len(coils), 3))
    currents = np.stack([row["coil_currents"] for row in archives])
    moments[:, :, 0] = currents[:, geometry["owners"]] * geometry["turns"]
    results = []
    with _baseline_kernel() as baseline:
        for kernel in (baseline, polygonanalytic):
            with patch.object(
                polygonanalytic,
                "packed_analytic_moments",
                kernel.packed_analytic_moments,
            ):

                def sample(point, coefficients, source):
                    return topology.BiotMomentCoupling.value_gradient.__wrapped__(
                        source, point, tuple(coefficients.T)
                    )

                program = jax.jit(
                    jax.vmap(
                        jax.vmap(sample, in_axes=(0, None, None)),
                        in_axes=(None, 0, None),
                    )
                )
                result = program(jnp.asarray(points), jnp.asarray(moments), coupling)
                results.append(jax.tree.map(np.asarray, result))
    expected, actual = results
    scale = np.linalg.norm(expected[1].reshape(3, -1), axis=1)
    assert np.all(scale > 0) and all(np.isfinite(row).all() for row in actual)
    relative = np.linalg.norm((actual[1] - expected[1]).reshape(3, -1), axis=1) / scale
    control = (
        np.linalg.norm((1.000001 * actual[1] - expected[1]).reshape(3, -1), axis=1)
        / scale
    )
    receipt["coil"] = dict(
        states=list(STATES),
        count=len(points),
        sources=len(coils),
        gradient_relative=relative.tolist(),
        control_relative=control.tolist(),
        different_values=int(
            np.count_nonzero(actual[0].view(np.uint64) != expected[0].view(np.uint64))
        ),
        different_gradients=int(
            np.count_nonzero(actual[1].view(np.uint64) != expected[1].view(np.uint64))
        ),
    )
    receipt["completed"] = True
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, default=float))
    print("COIL_CONTROL", json.dumps(receipt["coil"]), flush=True)
    assert max(relative) <= 1e-12
    assert min(control) > 1e-12


if __name__ == "__main__":
    main()
