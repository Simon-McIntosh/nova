"""Measure continuation storage before lowering the shared evaluator."""

import json
import jax
import jax.numpy as jnp
from machine import Constant, Machine, Register, operator_evaluation
from nova.equilibrium.fixed_point import newton_krylov
from nova.jax.config import configure_dtypes

configure_dtypes()
op = operator_evaluation(lambda x: 0.2 * x**2 + jnp.ones_like(x))


def solve(x):
    return newton_krylov(op, x, newton_steps=3, gmres_iterations=2, warmup=2)


x = jnp.zeros(2, dtype=jnp.float64)
closed = jax.make_jaxpr(solve)(x)
machine = Machine()
outputs = [Register(v.aval) for v in closed.jaxpr.outvars]
machine.used.update((r, None) for r in outputs)
entry = machine.lower(closed, [Constant(x)], outputs, -1)
print(
    json.dumps(
        dict(
            blocks=len(machine.blocks),
            registers=len(machine.used),
            operators=len(machine.operators),
            entry=entry,
            branch_register_product=len(machine.blocks) * len(machine.used),
        )
    ),
    flush=True,
)

for index, closed in enumerate(machine.operators):
    print(
        "operator",
        index,
        "inputs",
        [str(v.aval) for v in closed.jaxpr.invars],
        "outputs",
        [str(v.aval) for v in closed.jaxpr.outvars],
        flush=True,
    )
