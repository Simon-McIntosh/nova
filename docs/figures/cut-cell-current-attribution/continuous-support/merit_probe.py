"""Read the production merit and trust ladder on stated trial directions."""

import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.argv = [str(ROOT / "probe.py"), str(ROOT / "merit-tangent.json")]
context = runpy.run_path(str(ROOT / "probe.py"))
jax, jnp = context["jax"], context["jnp"]
mapped, state = context["mapped"], context["state"]
from nova.equilibrium.fixed_point import (  # noqa: E402
    _backtracking_scores,
    _smooth_relative_sup_merit,
)

image, tangent = jax.linearize(mapped, state)
reference = _smooth_relative_sup_merit(image, state)


def evaluate(step):
    return _backtracking_scores(
        mapped,
        lambda candidate: image + tangent(candidate - state),
        state,
        step,
        reference,
        True,
        own_mask_acceptance=True,
    )


evaluate = jax.jit(evaluate)
data = {}
for name, vector in (
    ("map_defect", image - state),
    ("toward_analytic", jnp.asarray(context["analytic"]) - state),
):
    scores = evaluate(vector)
    jax.block_until_ready(scores)
    data[name] = {
        key: context["np"].asarray(getattr(scores, key)).tolist()
        for key in scores._fields
        if key != "candidates"
    }
    (ROOT / "merit-scores.json").write_text(json.dumps(data, indent=2))
    print(name, json.dumps(data[name]), flush=True)
