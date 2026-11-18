"""Re-certify the ITER bank's plasma row against the retained stencil reference.

The reference differenced the *primal* analytic gradient at three steps and
Richardson-extrapolated it, so it never reads the polygon tangent under test.
The bank's plasma row is the row builder
``nova.equilibrium.fixed_points._source_jets`` produces, which is the row the
published bank stores.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import support_read, topology
from nova.equilibrium.fixed_points import _source_jets

assert jax.config.jax_enable_x64 is True

STATES = (220, 333, 336)
ROOT = Path.home() / ".cache/nova-hessian-diagnosis"
TOLERANCE = 1e-8


def relative(delta, scale):
    return np.sqrt(np.sum(delta**2, axis=0) / np.sum(scale**2, axis=0))


def retained(root):
    """Load the retained interior nodes, sampled moments and reference jets."""
    split = json.loads((root / "split.json").read_text())
    indices = np.array([row["node"] for row in split["nodes"] if row["interior"]])
    archives = [
        dict(np.load(root / f"diagnosis/accuracy-{state}.npz")) for state in STATES
    ]
    points = archives[0]["points"][indices]
    for archive, state in zip(archives, STATES, strict=True):
        np.testing.assert_array_equal(archive["points"][indices], points)
        assert archive["points"].shape[1] == 2
    moments = np.stack([archive["moments"] for archive in archives])
    nodes = [dict(np.load(root / f"node-{node}.npz")) for node in indices]
    gradient = np.stack([node["gradient"] for node in nodes], axis=1)
    hessian = np.stack([node["hessian"] for node in nodes], axis=1)
    return dict(
        indices=indices,
        points=points,
        moments=moments,
        gradient=gradient.astype(np.longdouble),
        hessian=hessian.astype(np.longdouble),
    )


def polygons_of(root):
    geometry = dict(np.load(root / "h200/geometry.npz"))
    polygons = tuple(
        p[:n]
        for p, n in zip(geometry["vertices"], geometry["vertex_count"], strict=True)
    )
    return geometry, polygons


def plasma_rows(points, polygons, batch):
    """Evaluate the bank's plasma row at every target with the row builder."""
    coupling = topology.BiotMomentCoupling.from_polygons(polygons)
    program = jax.jit(jax.vmap(_source_jets, in_axes=(0, None)))
    output = []
    for first in range(0, len(points), batch):
        last = min(first + batch, len(points))
        size = last - first
        batch_points = points[first:last]
        if size < batch:
            batch_points = np.concatenate(
                (batch_points, np.repeat(batch_points[-1:], batch - size, axis=0))
            )
        output.append(np.asarray(program(jnp.asarray(batch_points), coupling))[:size])
        print("ROW_BATCH", first, flush=True)
    return np.concatenate(output), coupling


def contraction(rows, moments):
    """Contract booked moments into every node's value and five derivatives."""
    return np.einsum("scm,ncmq->snq", moments, rows)


def components(values, points, gradient, hessian):
    """Compare the four jet entries this node reports against the reference."""
    result = {}
    for name, actual, expected in (
        ("psi_R", values[..., 1], gradient[..., 0]),
        ("psi_Z", values[..., 2], gradient[..., 1]),
        ("psi_RZ", values[..., 4], hessian[..., 0, 1]),
        ("psi_ZZ", values[..., 5], hessian[..., 1, 1]),
    ):
        result[name] = relative(
            np.asarray(actual, dtype=np.longdouble) - expected, expected
        ).tolist()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--store", type=Path, help="Published bank content store")
    parser.add_argument("--digest", help="Digest of the published bank")
    parser.add_argument(
        "--current-bank", type=Path, help="Previously published bank npz"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-rows",
        action="store_true",
        help="read only the published bank, never evaluate the row builder",
    )
    args = parser.parse_args()
    data = retained(args.root)
    if args.limit:
        for key in ("indices", "points", "gradient", "hessian", "moments"):
            axis = 1 if key in ("gradient", "hessian", "moments") else 0
            data[key] = np.take(data[key], np.arange(args.limit), axis=axis)
    tolerance = dict(psi_R=1e-10, psi_Z=1e-10, psi_RZ=TOLERANCE, psi_ZZ=TOLERANCE)
    result = dict(
        count=int(len(data["indices"])),
        states=list(STATES),
        points=data["points"].tolist(),
        indices=data["indices"].tolist(),
    )
    rows = None
    if not args.skip_rows:
        _, polygons = polygons_of(args.root)
        rows, _ = plasma_rows(data["points"], polygons, args.batch)
        values = contraction(rows, data["moments"])
        result["cells"] = len(polygons)
        result["rows_mode"] = components(
            values, data["points"], data["gradient"], data["hessian"]
        )
        result["rows_pass"] = passes(result["rows_mode"], tolerance)
        np.savez(
            args.output.with_suffix(".rows.npz"),
            values=values,
            rows=rows,
            indices=data["indices"],
            points=data["points"],
        )
        print(
            "ROWS_MODE " + json.dumps(result["rows_mode"], sort_keys=True),
            flush=True,
        )
    if args.store is not None:
        bank = support_read.read_bank(args.store, args.digest, None, None)
        stored = np.asarray(bank.rows)
        bank_values = contraction(stored[data["indices"]], data["moments"])
        result["bank_mode"] = components(
            bank_values, data["points"], data["gradient"], data["hessian"]
        )
        result["bank_pass"] = passes(result["bank_mode"], tolerance)
        print(
            "BANK_MODE " + json.dumps(result["bank_mode"], sort_keys=True), flush=True
        )
        if rows is not None:
            result["rows_max_abs_delta"] = float(
                np.max(np.abs(stored[data["indices"]] - rows))
            )
        if args.current_bank is not None:
            with np.load(args.current_bank, allow_pickle=False) as archive:
                current = np.asarray(archive["rows"])
                current_coil = np.asarray(archive["coil_exterior"])
                active = int(archive["active_nodes"])
            current_values = contraction(current[data["indices"]], data["moments"])
            result["current_bank"] = components(
                current_values, data["points"], data["gradient"], data["hessian"]
            )
            result["unchanged"] = dict(
                first_derivative_rows=relative(
                    stored[data["indices"], :, :, 1:3]
                    - current[data["indices"], :, :, 1:3],
                    current[data["indices"], :, :, 1:3],
                ).tolist(),
                value_rows=relative(
                    stored[data["indices"], :, :, :1]
                    - current[data["indices"], :, :, :1],
                    current[data["indices"], :, :, :1],
                ).tolist(),
                first_derivative_rows_active=relative(
                    stored[:active, :, :, 1:3] - current[:active, :, :, 1:3],
                    current[:active, :, :, 1:3],
                ).tolist(),
                coil_rows=relative(
                    np.asarray(bank.coil_exterior)[:active] - current_coil[:active],
                    current_coil[:active],
                ).tolist(),
            )
            result["unchanged_pass"] = all(
                np.all(np.array(value) <= 1e-10)
                for value in result["unchanged"].values()
            )
            print(
                "UNCHANGED " + json.dumps(result["unchanged"], sort_keys=True),
                flush=True,
            )
    result["completed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print("RECERTIFY_COMPLETE " + json.dumps(digest_of(result)), flush=True)


def passes(errors, tolerance):
    """Hold every reported component to its stated relative budget."""
    return all(
        all(np.array(error) <= tolerance[name] for error in errors[name])
        for name in tolerance
    )


def digest_of(result):
    return {
        key: result[key]
        for key in ("count", "rows_pass", "bank_pass", "unchanged_pass")
        if key in result
    }


if __name__ == "__main__":
    main()
