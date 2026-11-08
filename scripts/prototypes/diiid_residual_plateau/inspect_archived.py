"""Revalue an archived physical state with the current production maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from measure import host_read, measure, write_json
from nova.equilibrium import fixed_point
from nova.jax.config import configure_dtypes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_dtypes()
    assert jax.config.jax_enable_x64
    prepared = measure._prepare_frame(
        args.frame, float(measure.bench.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION)
    )
    profile = prepared["profile"]
    with np.load(args.archive) as archive:
        state = jnp.asarray(archive["state"])
    operator = profile.operator
    external = operator.external(jnp.asarray(prepared["current"]))
    target = prepared["target_current_a"]
    shadow = operator.residual_shadow_mask(state)
    promoted = operator.residual_shadow_mask(state, previous_shadow=shadow)
    live = jax.jit(operator.traced_flux_map(target_current=target))
    frozen = jax.jit(operator.traced_flux_map_with_shadow(target_current=target))
    mapped = live(state, external, operator, target)
    shadowed = frozen(state, shadow, external, operator, target)
    reconciled = frozen(state, promoted, external, operator, target)
    local, _result = host_read(operator, state)
    print("ARCHIVED REVALUATION", args.frame, flush=True)
    record = dict(
        frame=args.frame,
        support=int(np.asarray(local["connected"]).sum()),
        shadow=int(np.asarray(shadow).sum()),
        changed_shadow=int(np.sum(np.asarray(shadow) != np.asarray(promoted))),
        live_residual=float(fixed_point._relative_residual(mapped, state)),
        frozen_residual=float(fixed_point._relative_residual(shadowed, state)),
        reconciled_residual=float(fixed_point._relative_residual(reconciled, state)),
        live_merit=float(fixed_point._smooth_relative_sup_merit(mapped, state)),
        frozen_merit=float(fixed_point._smooth_relative_sup_merit(shadowed, state)),
        map_difference=float(jnp.max(jnp.abs(mapped - shadowed))),
        module=fixed_point.__file__,
        archive=str(args.archive),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "revaluation.json", record)
    np.savez_compressed(
        args.output / "revaluation.npz",
        state=state,
        live=mapped,
        frozen=shadowed,
        reconciled=reconciled,
        shadow=shadow,
        promoted=promoted,
        connected=local["connected"],
        radius=profile.lattice.radius,
        height=profile.lattice.height,
    )
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
