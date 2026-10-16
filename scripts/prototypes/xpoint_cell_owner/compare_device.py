"""Compare the traced conic owner with independently integrated cell rows."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import nova
from scripts.prototypes.xpoint_cell_owner.device_owner import integrate_cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    source = json.loads(args.input.read_text())
    reference = json.loads(args.reference.read_text())
    by_cell = {row["cell"]: row for row in reference["rows"]}
    evaluate = jax.jit(integrate_cell)
    capacity = max(len(row["vertices"]) for row in source["rows"])
    result = {"cells": source["cells"], "device": jax.default_backend(), "rows": []}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in source["rows"]:
        vertices = np.asarray(row["vertices"])
        padded = np.pad(vertices, ((0, capacity - len(vertices)), (0, 0)))
        output, valid = evaluate(
            jnp.asarray(padded),
            jnp.asarray(len(vertices), dtype=jnp.int32),
            jnp.asarray(row["centre"]),
            jnp.asarray(row["pitch"]),
            jnp.asarray(row["field_coefficients"]),
            jnp.asarray(source["boundary_level"]),
            jnp.asarray(source["sigma"]),
            jnp.asarray(source["saddle"]),
            jnp.asarray(source["axis"]),
            jnp.asarray(row["density_coefficients"]),
        )
        actual = np.asarray(output)
        expected = np.asarray(by_cell[row["cell"]]["dense"])
        scale = np.maximum.reduce(
            (
                np.abs(expected),
                np.array(
                    (
                        row["full_area"],
                        abs(expected[1]),
                        abs(expected[1]) * row["pitch"],
                        abs(expected[1]) * row["pitch"],
                    )
                ),
                np.full(4, 1e-14),
            )
        )
        error = np.abs(actual - expected) / scale
        item = {
            "cell": row["cell"],
            "valid": bool(valid),
            "moments": actual.tolist(),
            "scaled_relative_error": error.tolist(),
        }
        result["rows"].append(item)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"CELL {json.dumps(item)}", flush=True)
    result["max_relative_error"] = max(
        max(row["scaled_relative_error"]) for row in result["rows"]
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"SUMMARY max_relative={result['max_relative_error']:.12g}", flush=True)
    if result["max_relative_error"] > 1e-12 or not all(
        row["valid"] for row in result["rows"]
    ):
        raise ValueError("traced conic owner differs from dense quadrature")


if __name__ == "__main__":
    main()
