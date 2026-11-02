"""Run an archived bank solve with the archive's source on the import path."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import os
from pathlib import Path

import jax

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_step_booking.measure import emit, run


def main():
    root = Path(__file__).resolve().parents[3]
    emit(
        "HEADER",
        revision=os.environ["BASE_REVISION"],
        tree=str(root),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
    )
    assert jax.config.jax_enable_x64
    run("fresh", None, 30)


if __name__ == "__main__":
    main()
