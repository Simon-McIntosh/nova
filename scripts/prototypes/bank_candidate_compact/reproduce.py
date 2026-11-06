"""Expose the truncation diagnostic and measure candidate support programs."""

import inspect
from unittest.mock import patch

import jax
from nova.equilibrium import topology
from tests.test_candidate_inspection_batched import (
    candidate_program,
    small_case,
    stationary_adapter,
    test_candidate_slots_share_one_program,
)

source = inspect.getsource(topology._read_at_nulls)
with patch.object(
    inspect, "getsource", return_value=source.replace("saddle_index", "selected_index")
):
    try:
        candidate_program()
    except Exception as error:
        print("MISSING_ASSIGNMENT", type(error).__name__, repr(str(error)), flush=True)

unrolled = source.replace(
    "accepted, forms = jax.vmap(inspect)(indices)",
    "accepted, forms = jax.tree.map(lambda *leaves: jax.numpy.stack(leaves), "
    "*[inspect(i) for i in range(policy.null_capacity)])",
)
assert unrolled != source
with patch.object(inspect, "getsource", return_value=unrolled):
    test_candidate_slots_share_one_program()
    print("UNROLLED_GUARD_PASSED", flush=True)

with patch.object(
    topology,
    "_support_at_level",
    jax.named_call(topology._support_at_level, name="candidate_support"),
):
    program = jax.make_jaxpr(stationary_adapter(candidate_program()))(
        *small_case()
    ).jaxpr


def walk(program):
    for eqn in program.eqns:
        if eqn.primitive.name in ("while", "scan"):
            print(
                "LOOP", eqn.primitive.name, str(eqn.source_info.name_stack), flush=True
            )
        for value in eqn.params.values():
            for child in value if isinstance(value, (tuple, list)) else (value,):
                child = getattr(child, "jaxpr", child)
                if hasattr(child, "eqns"):
                    walk(child)


walk(program)
