"""Measure batched warm corrections from saved matched Bell-rule terminals."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from pathlib import Path

import jax
import nova

from scripts.prototypes.lean_map import forward as lean_forward
from scripts.prototypes.lean_map import probe_forward
from scripts.prototypes.sub_millisecond.measure_slice import arc_cut_moments


def run(kind: str, cache: Path, output: Path):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    lean_forward.cut_moments = arc_cut_moments
    probe_forward.apply = lean_forward.apply
    probe_forward.run(kind, 250, cache, output, batch=True, terminal_only=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.kind, args.cache, args.output)


if __name__ == "__main__":
    main()
