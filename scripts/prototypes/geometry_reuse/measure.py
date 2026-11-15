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

    substitute("import json\n", "import json\nimport os\n")
    substitute(
        "    args = parser.parse_args()",
        '    parser.add_argument("--arm", choices=("off", "on"), required=True)\n'
        '    parser.add_argument("--lower-only", action="store_true")\n'
        "    args = parser.parse_args()",
    )
    substitute(
        "        program = lowered.compile()",
        "        if args.lower_only and label == 'solve':\n"
        "            record.update(complete=True, lowering_only=True, "
        "lowered_solve_sha256=hashlib.sha256("
        "lowered.as_text(debug_info=False).encode()).hexdigest())\n"
        "            save()\n"
        "            print('LOWERED_ONLY', record['lowered_solve_sha256'], "
        "flush=True)\n"
        "            raise SystemExit(0)\n"
        "        program = lowered.compile()",
    )
    substitute(
        "sr.read_batch(b, c, m, s, s, cv, policy, tracking=tracking)",
        "sr.read_batch(b, c, m, s, s, cv, policy, tracking=tracking, "
        "reuse_geometry=args.arm == 'on')",
    )
    substitute(
        "        revision=args.revision,",
        "        arm=args.arm,\n        revision=args.revision,",
    )
    substitute(
        "        x64=True,",
        "        x64=True,\n"
        "        allocator=dict(preallocate=os.environ.get("
        "'XLA_PYTHON_CLIENT_PREALLOCATE'), memory_fraction=os.environ.get("
        "'XLA_CLIENT_MEM_FRACTION')),",
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
        "                reuse_geometry=args.arm == 'on',\n"
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
