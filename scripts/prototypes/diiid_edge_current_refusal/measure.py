"""Retain spatial Newton operands beside the profile-extent cohort receipt.

The existing operand-capture seams observe executed inner solves and scored
line-search proposals. This observer adds their array-valued diagnostics;
every solver function returns its original result without modification.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "extent_measure", ROOT / "scripts/prototypes/diiid_xpoint_profile_extent/measure.py"
)
extent = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extent
SPEC.loader.exec_module(extent)
branch = extent.held_measure.core.branch_measure
load = branch.original_load
captures = []


def observed_load(name, path):
    """Extend the cohort's existing capture without changing its solve route."""
    module = load(name, path)
    if name != "operand_capture":
        return module
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nova.equilibrium import fixed_point

    class SpatialCapture(module.Capture):
        def __init__(self, output):
            super().__init__(output)
            self.trace_count = 0
            self.proposals = []
            captures.append(self)

        def install(self):
            super().install()
            inner = fixed_point._newton_krylov_inner
            ladder = fixed_point._backtracking_scores

            def observed_inner(*args, **kwargs):
                result = inner(*args, **kwargs)
                terminal = result if hasattr(result, "state") else result[0]
                names = tuple(
                    name
                    for name in terminal._fields
                    if name.startswith(("inner_iteration_", "promotion_"))
                )
                jax.debug.callback(
                    lambda *values: self.save_diagnostics(names, values),
                    *(getattr(terminal, name) for name in names),
                    kwargs["shadow_mask_fn"](args[1])
                    if kwargs.get("shadow_mask_fn") is not None
                    else jnp.zeros_like(args[1], dtype=bool),
                    terminal.trajectory_state,
                    ordered=True,
                )
                return result

            def observed_ladder(*args, **kwargs):
                result = ladder(*args, **kwargs)
                jax.debug.callback(
                    self.save_proposal,
                    args[2],
                    args[3],
                    result,
                    ordered=True,
                )
                return result

            fixed_point._newton_krylov_inner = observed_inner
            fixed_point._backtracking_scores = observed_ladder

        def save_diagnostics(self, names, values):
            np.savez_compressed(
                self.output / f"diagnostics-{self.trace_count}.npz",
                **dict(zip(names, values[:-2], strict=True)),
                frozen_shadow=values[-2],
                trajectory_state=values[-1],
            )
            self.trace_count += 1

        def save_proposal(self, state, step, scores):
            index = len(self.proposals)
            np.savez_compressed(
                self.output / f"proposal-{index}.npz",
                state=state,
                step=step,
                **scores._asdict(),
            )
            self.proposals.append(
                dict(
                    index=index,
                    inner=self.trace_count,
                    accepted=bool(scores.ladder_accepted),
                    distrusted=bool(scores.ladder_distrusted),
                )
            )
            (self.output / "proposals.json").write_text(
                json.dumps(self.proposals, indent=2) + "\n"
            )

    module.Capture = SpatialCapture
    return module


if __name__ == "__main__":
    branch.original_load = observed_load
    extent.main()
    assert len(captures) == 1
    capture = captures[0]
    assert capture.trace_count == capture.counts["inner"] > 0
    sys.path.insert(0, str(HERE))
    from analyze import analyze

    analyze(capture)
