"""Retain the compiled small-fixture moments before considering byte pins."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import hashlib
import json
from pathlib import Path
import sys

import jax
import numpy as np

import nova
from nova.equilibrium import current
from tests.test_support_read import _case


def main():
    output = Path(sys.argv[1])
    print("HEADER", nova.__file__, current.__file__, Path.cwd().resolve(), flush=True)
    assert jax.config.jax_enable_x64
    operands, cache, roots, pitch, step, _, _ = _case("diverted")
    _, booked = step(operands, cache, roots + 0.03 * pitch, roots)
    arrays = {
        name: np.asarray(value)
        for name, value in zip(booked._fields, booked, strict=True)
    }
    np.savez(output.with_suffix(".npz"), **arrays)
    pins = {
        name: hashlib.sha256(value.tobytes()).hexdigest()
        for name, value in arrays.items()
    }
    output.with_suffix(".json").write_text(json.dumps(pins, indent=2))
    print("PINS", json.dumps(pins), flush=True)


if __name__ == "__main__":
    main()
