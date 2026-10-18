"""Compare two stationary proposal implementations on one moving carrier."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
import subprocess
from pathlib import Path

import jax
import numpy as np

from scripts.prototypes.fast_map_production import tie_probe

MAP_FIELDS = (
    "boundary_class",
    "wall_candidate",
    "saddle_candidate",
    "boundary_flux",
    "raw_current",
    "valid",
    "qualified",
    "boundary_position",
    "saddle_form_position",
)


def load_census(revision, tree):
    """Compile the census source from a recorded revision without a checkout."""
    source = subprocess.check_output(
        [
            "git",
            "-C",
            str(tree),
            "show",
            f"{revision}:scripts/prototypes/read_robustness/census.py",
        ],
        text=True,
    )
    namespace = {"__name__": "recorded_census", "__file__": f"{revision}:census.py"}
    exec(compile(source, namespace["__file__"], "exec"), namespace)
    return namespace["census"]


def measure(state, function, offsets, warm_stop):
    tie_probe.census = function
    sampled = tie_probe.make_sampler(state)
    prior, warm_count = tie_probe.warm(state, sampled, -0.2, warm_stop, 0.01)
    return [
        tie_probe.row(offset, state, sampled, prior) for offset in offsets
    ], warm_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--scan")
    parser.add_argument("--offsets")
    parser.add_argument("--warm-stop", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (args.scan is None) == (args.offsets is None):
        parser.error("provide exactly one of --scan and --offsets")
    assert jax.config.jax_enable_x64 is True
    import nova

    tree = Path.cwd().resolve()
    print(
        f"nova={nova.__file__} module={__file__} cwd={tree} "
        f"backend={jax.default_backend()}",
        flush=True,
    )
    if args.scan:
        low, high, count = args.scan.split(",")
        offsets = np.linspace(float(low), float(high), int(count)).tolist()
    else:
        offsets = [float(value) for value in args.offsets.split(",")]
    base = load_census(args.base_revision, tree)
    from scripts.prototypes.read_robustness.census import census as head

    state = tie_probe.build("diverted", args.cells, args.cache)
    arms = {}
    for name, function in (("base", base), ("head", head)):
        rows, warm_count = measure(state, function, offsets, args.warm_stop)
        arms[name] = rows
        print(
            f"{name} warm={warm_count} counts="
            + " ".join(f"{row['offset']:.8g}:{row['root_count']}" for row in rows),
            flush=True,
        )
    agreement = [
        all(before[key] == after[key] for key in MAP_FIELDS)
        for before, after in zip(arms["base"], arms["head"], strict=True)
    ]
    count_agreement = [
        before["root_count"] == after["root_count"]
        for before, after in zip(arms["base"], arms["head"], strict=True)
    ]
    result = {
        "base_revision": args.base_revision,
        "cells": len(state["geometry"].centre),
        "backend": jax.default_backend(),
        "map_equal": agreement,
        "counts_equal": count_agreement,
        "arms": arms,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"MAP_EQUAL={sum(agreement)}/{len(agreement)} "
        f"COUNT_EQUAL={sum(count_agreement)}/{len(count_agreement)}",
        flush=True,
    )
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
