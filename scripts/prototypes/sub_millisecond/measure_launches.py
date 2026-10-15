"""Trace map and warm-correction launches in the complete prototype route."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from pathlib import Path

import jax
import nova

from scripts.prototypes.sub_millisecond import measure_slice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    original_jit = jax.jit
    profile_dir = args.profile_dir

    class Program:
        def __init__(self, program, name):
            self.program = program
            self.name = name
            self.calls = 0

        def __call__(self, *operands):
            self.calls += 1
            target = 2 if self.name == "mapped" else 1
            if self.calls == target:
                directory = profile_dir / self.name
                directory.mkdir(parents=True, exist_ok=True)
                with jax.profiler.trace(str(directory)):
                    result = self.program(*operands)
                    jax.block_until_ready(result)
                print(f"TRACE name={self.name} path={directory}", flush=True)
                return result
            return self.program(*operands)

    class Lowered:
        def __init__(self, lowered, name):
            self.lowered = lowered
            self.name = name

        def compile(self, *arguments, **options):
            return Program(self.lowered.compile(*arguments, **options), self.name)

    class Jitted:
        def __init__(self, jitted, name):
            self.jitted = jitted
            self.name = name

        def lower(self, *arguments, **options):
            return Lowered(self.jitted.lower(*arguments, **options), self.name)

    def traced_jit(function, *arguments, **options):
        jitted = original_jit(function, *arguments, **options)
        name = getattr(function, "__name__", "")
        if name in ("mapped", "warm_correct"):
            return Jitted(jitted, name)
        return jitted

    jax.jit = traced_jit
    try:
        measure_slice.run(args.kind, args.cache, args.output, terminal_only=True)
    finally:
        jax.jit = original_jit


if __name__ == "__main__":
    main()
