"""Measure the topology read with saddle normal-form support removed."""

from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path

from nova.equilibrium import topology
from scripts.prototypes.ladder import measure


SUPPORT_SELECTION = """    represented = saddle_live & (
        owner
        | (
            jnp.linalg.norm(geometry.centre - form.position, axis=1)
            < policy.normal_form_radius
        )
    )"""


def _without_normal_form() -> None:
    source = inspect.getsource(topology._support_at_level)
    if source.count(SUPPORT_SELECTION) != 1:
        raise AssertionError("the saddle support selection changed")
    source = source.replace(
        SUPPORT_SELECTION, "    represented = jnp.zeros_like(owner)"
    )
    namespace: dict = {}
    exec(
        compile(source, str(Path(__file__).resolve()), "exec"),
        topology.__dict__,
        namespace,
    )
    topology._support_at_level = namespace["_support_at_level"]


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _without_normal_form()
    row = measure._measure(args.kind, args.cells, "B")
    row["normal_form_support_disabled"] = True
    row["normal_form_owner_cells"] = 0
    row = _json_safe(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        "QUADRATIC_ROW " + json.dumps(row, sort_keys=True, allow_nan=False), flush=True
    )


if __name__ == "__main__":
    main()
