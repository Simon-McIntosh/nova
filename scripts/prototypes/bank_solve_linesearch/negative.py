"""Restore pending-lane read accounting in a disposable source tree."""

from pathlib import Path
import sys

path = Path(sys.argv[1]) / "nova/equilibrium/solve.py"
source = path.read_text()
replacements = {
    "calls, pending, tried = carry": "calls, pending, tried, lane_calls = carry",
    "tried | (pending & valid),": (
        "tried | (pending & valid),\n"
        "            lane_calls + pending.astype(jnp.int32),"
    ),
    "jnp.zeros_like(active),": (
        "jnp.zeros_like(active),\n        jnp.zeros_like(active, dtype=jnp.int32),"
    ),
    "scale, _, calls, _, _ = jax.lax.while_loop(": (
        "scale, _, calls, _, _, lane_calls = jax.lax.while_loop("
    ),
    "_DampedResult(best, best_aux, best_norm, scale, calls)": (
        "_DampedResult(best, best_aux, best_norm, scale, lane_calls)"
    ),
}
for before, after in replacements.items():
    assert source.count(before) == 1, before
    source = source.replace(before, after)
path.write_text(source)
print("MUTATION count reads only for pending lanes", path)
test = path.parents[2] / "tests/test_bank_solve_driver.py"
source = test.read_text()
anchor = (
    '    """Every trial reads even the lanes accepted earlier or already at a root."""'
)
assert source.count(anchor) == 1
source = source.replace(
    anchor,
    anchor + "\n    import nova\n    from pathlib import Path\n"
    '    print("PROVENANCE", nova.__file__, support_read.__file__, '
    'Path.cwd().resolve())',
)
test.write_text(source)
