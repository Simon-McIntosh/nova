"""Compare fixed cut-bucket capacities on the same lane states."""

from dataclasses import replace
import json
from unittest.mock import patch

import jax


def run(run_dir, machine, inputs, jacobians, timer, emit):
    bank, _, _, _, _, profile, quadrature, *_ = machine
    rows = {}
    for lanes in (1, 16):
        states, support, booked = inputs(lanes, machine)
        for width in (8, 16, 32):
            rule = replace(quadrature, cut_batch_width=width)

            def record(kind, **data):
                emit(kind, lanes=lanes, width=width, **data)
                if kind == "STAGE":
                    rows[f"{lanes}:{width}"] = data["median_ms"]

            with patch.object(timer, "emit", record):
                timer.measure(
                    "assembly",
                    lambda b, m, s, p: jacobians(
                        b, m, s, p, profile, rule, lambda f, x: jax.vmap(f)(x)
                    ),
                    bank,
                    states,
                    support,
                    booked,
                )
            (run_dir / "bucket-widths.json").write_text(json.dumps(rows, indent=2))
