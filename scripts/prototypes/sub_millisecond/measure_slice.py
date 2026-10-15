"""Run a complete warm moment correction with analytic Bell arc cuts."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import nova

from scripts.prototypes.lean_map import forward as lean_forward
from scripts.prototypes.lean_map import probe_forward
from scripts.prototypes.lean_map.quadrature import BookingStatus, LeanBooking
from scripts.prototypes.sub_millisecond.measure_bell_cut import bell_moments


exact_cut_moments = lean_forward.cut_moments


def arc_cut_moments(psi, reading, geometry, profile, sigma, _capacity):
    full = (reading.membership >= 1 - 1e-14) & (geometry.vertex_count == 6)
    cut = (reading.membership > 0) & ~full
    ordinary = cut & ~reading.normal_form_cells
    analytic, valid = bell_moments(geometry, reading, profile, sigma, 4)
    saddle_read = reading._replace(
        membership=jnp.where(ordinary, 0.0, reading.membership)
    )
    saddle = exact_cut_moments(psi, saddle_read, geometry, profile, sigma, 8)
    moments = saddle.moments + jnp.where(ordinary[:, None], analytic, 0.0)
    invalid = jnp.any(ordinary & ~valid)
    status = jnp.where(invalid, int(BookingStatus.INVALID), saddle.status)
    return LeanBooking(moments, jnp.sum(cut, dtype=jnp.int32), status)


def run(kind, cache, output, terminal_only):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}",
        flush=True,
    )
    lean_forward.cut_moments = arc_cut_moments
    probe_forward.apply = lean_forward.apply
    probe_forward.run(kind, 250, cache, output, terminal_only=terminal_only)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminal-only", action="store_true")
    args = parser.parse_args()
    run(args.kind, args.cache, args.output, args.terminal_only)


if __name__ == "__main__":
    main()
