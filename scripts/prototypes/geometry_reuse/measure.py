"""Adapt the committed slice-profile instrument for paired geometry carry."""

import hashlib
from pathlib import Path

from scripts.prototypes.bank_h200_slice_profile import measure


def program():
    path = Path(measure.__file__)
    source = path.read_text()

    def substitute(before, after, count=1):
        nonlocal source
        assert source.count(before) == count, (before, source.count(before), count)
        source = source.replace(before, after)

    substitute(
        "    args = parser.parse_args()",
        '    parser.add_argument("--arm", choices=("off", "on"), required=True)\n'
        "    args = parser.parse_args()",
    )
    substitute(
        "    assert jax.config.jax_enable_x64 is True",
        "    from nova.equilibrium import topology\n"
        "    if args.arm == 'off':\n"
        "        topology._fragment_coefficients = lambda _v, values: values\n"
        "        original_fragments = topology._fragments_at_level\n"
        "        def uncached(geometry, coefficient, sigma, level, form, "
        "represented, policy, cached=None, *, return_cache=False):\n"
        "            return original_fragments(geometry, coefficient, sigma, "
        "level, form, represented, policy, None, return_cache=return_cache)\n"
        "        topology._fragments_at_level = uncached\n"
        "    assert jax.config.jax_enable_x64 is True",
    )
    substitute(
        "        revision=args.revision,",
        "        arm=args.arm,\n        revision=args.revision,",
    )
    substitute(
        "            compile_seconds=perf_counter() - started,",
        "            lowered_sha256=hashlib.sha256("
        "lowered.as_text(debug_info=False).encode()).hexdigest(),\n"
        "            compile_seconds=perf_counter() - started,",
    )
    # This measurement requests native walls, not a second device attribution.
    substitute(
        "    def capture(label, program, inputs):\n        if args.smoke:",
        "    def capture(label, program, inputs):\n        if True:",
    )
    substitute(
        "        def solve(b, c, m, s, amplitude):",
        "        def solve(b, c, m, s, amplitude, carried):",
    )
    substitute(
        "                reuse_factor=False,",
        "                reuse_factor=False,\n"
        "                initial_support=carried if args.arm == 'on' else None,",
    )
    substitute(
        '        program = compile_program(\n            "solve", solve, '
        "(bank, cache, state, seeds, jnp.asarray(1.0))\n        )",
        "        support = jax.block_until_ready("
        "jax.jit(read)(bank, cache, state, seeds, None))\n"
        '        program = compile_program(\n            "solve", solve, '
        "(bank, cache, state, seeds, jnp.asarray(1.0), support)\n        )",
    )
    substitute(
        "program(bank, cache, predicted, seeds, jnp.asarray(target))",
        "program(bank, cache, predicted, seeds, jnp.asarray(target), support)",
        2,
    )
    substitute(
        "solve_inputs = (bank, cache, predicted, seeds, jnp.asarray(target))",
        "solve_inputs = (bank, cache, predicted, seeds, jnp.asarray(target), support)",
    )
    substitute(
        "            previous_state, state, amplitude, seeds = (",
        "            support = result.support\n"
        "            save()\n"
        "            previous_state, state, amplitude, seeds = (",
    )
    adapter_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    instrument_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    substitute(
        "        instrument_revision=args.instrument,",
        "        instrument_revision=args.instrument,\n"
        f"        adapter_sha256={adapter_digest!r},\n"
        f"        base_instrument_sha256={instrument_digest!r},",
    )
    namespace = dict(measure.__dict__, __name__="geometry_measure")
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["main"]


if __name__ == "__main__":
    program()()
