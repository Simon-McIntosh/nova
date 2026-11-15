"""Repeat the limited edge row with the inactive field contribution suppressed.

This counterfactual retains the static zero-unknown row and the original solve
policy. It removes only that inactive row's flux contribution at runtime; no
product source is edited and no position target is introduced.
"""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import jax.numpy as jnp

from nova.equilibrium.constraint import UniformRadialFieldUnknown

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "edge_current_observer", HERE / "measure.py"
)
observer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = observer
SPEC.loader.exec_module(observer)


if __name__ == "__main__":
    original = UniformRadialFieldUnknown.flux_delta

    def inactive_field_suppressed(self, profile, context, functional, payload, unknown):
        assert payload is not None, "control requires an explicit active flag"
        value = original(self, profile, context, functional, payload, unknown)
        return jnp.where(jnp.asarray(payload, dtype=bool), value, jnp.zeros_like(value))

    UniformRadialFieldUnknown.flux_delta = inactive_field_suppressed
    folder = Path(sys.argv[sys.argv.index("--output") + 1])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "counterfactual.json").write_text(
        json.dumps(
            dict(
                changed="inactive UniformRadialFieldUnknown.flux_delta only",
                constraint="inactive scalar row retained; no position target",
                purpose="test whether the inactive radial field drives plasma loss",
            ),
            indent=2,
        )
        + "\n"
    )
    observer.branch.original_load = observer.observed_load
    observer.extent.main()
    assert len(observer.captures) == 1
    sys.path.insert(0, str(HERE))
    from analyze import analyze

    analyze(observer.captures[0])
