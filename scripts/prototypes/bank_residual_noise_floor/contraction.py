"""Compare recorded nodal contractions with 80-digit sums of the same operands."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from decimal import Decimal, localcontext
import json
from pathlib import Path

import jax
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_booking_determinism.measure import emit
from scripts.prototypes.bank_residual_noise_floor.measure import nudged
from scripts.prototypes.bank_solve_profile import fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    emit(
        "HEADER",
        revision=args.revision,
        nova=nova.__file__,
        module=support_read.__file__,
        cwd=str(Path.cwd().resolve()),
        device=jax.devices()[0].device_kind,
        x64=jax.config.jax_enable_x64,
    )
    assert jax.config.jax_enable_x64 and jax.default_backend() == "cpu"
    batch = fixture.build(16)
    states = np.load(args.run / "gpu-states.npz")
    capture = np.load(args.run / "propagation.npz")
    worst = json.loads((args.run / "propagation.json").read_text())["worst"]
    lane, sample = worst["lane"], worst["sample"]
    moments = tuple(states[f"moment_{i}"][lane] for i in range(3))
    moved = tuple(np.asarray(nudged(x, sample + 1, 1 + sample % 4)) for x in moments)
    cell = int(capture["field_0_cell"])
    triangle = int(capture["field_0_triangle"])
    slots = np.asarray(batch.operands.mesh.triangles[cell, triangle])
    rows = np.asarray(batch.operands.rows[slots])
    exterior = np.asarray(batch.operands.exterior[slots])
    results = []
    precise = []
    with localcontext() as context:
        context.prec = 80
        for arm, state in enumerate((moments, moved)):
            flat = np.stack(state, axis=-1).reshape(-1)
            state_decimal = [Decimal.from_float(float(v)) for v in flat]
            ideal = []
            for node in range(len(slots)):
                node_values = []
                for channel in range(6):
                    terms = [
                        Decimal.from_float(float(a)) * b
                        for a, b in zip(
                            rows[node, :, :, channel].reshape(-1), state_decimal
                        )
                    ]
                    terms.append(Decimal.from_float(float(exterior[node, channel])))
                    exact = sum(terms)
                    actual = float(capture[f"field_{arm}_jets"][node, channel])
                    error = Decimal.from_float(actual) - exact
                    node_values.append(exact)
                    results.append(
                        dict(
                            arm=arm,
                            node=int(slots[node]),
                            channel=channel,
                            actual=actual,
                            exact=str(exact),
                            error=float(error),
                            error_ulps=float(
                                abs(error)
                                / Decimal.from_float(abs(float(np.spacing(actual))))
                            ),
                            sum_absolute=float(sum(abs(v) for v in terms)),
                            cancellation=float(sum(abs(v) for v in terms) / abs(exact)),
                        )
                    )
                ideal.append(node_values)
            precise.append(ideal)
        exact_change = np.asarray(
            [
                [float(b - a) for a, b in zip(left, right)]
                for left, right in zip(precise[0], precise[1])
            ]
        )
    observed = capture["field_1_jets"] - capture["field_0_jets"]
    record = dict(
        worst=worst,
        terms_per_jet=len(flat) + 1,
        rows=results,
        exact_change=exact_change.tolist(),
        observed_change=observed.tolist(),
        rounding_change=(observed - exact_change).tolist(),
    )
    (args.run / "contraction.json").write_text(json.dumps(record, indent=2))
    emit(
        "CONTRACTION",
        maximum_error=max(abs(r["error"]) for r in results),
        maximum_ulps=max(r["error_ulps"] for r in results),
        maximum_cancellation=max(r["cancellation"] for r in results),
        exact_change_max=float(np.abs(exact_change).max()),
        observed_change_max=float(np.abs(observed).max()),
        rounding_change_max=float(np.abs(observed - exact_change).max()),
    )
    emit("COMPLETE", output=str(args.run / "contraction.json"))


if __name__ == "__main__":
    main()
