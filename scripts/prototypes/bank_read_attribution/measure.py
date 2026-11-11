"""Attribute native bank reads with function-owned, runtime-operand partitions."""

import argparse
from contextlib import ExitStack
import inspect
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from unittest.mock import patch


def coverage(stage_ms, whole_ms):
    gap = sum(stage_ms.values()) / whole_ms - 1
    assert abs(gap) <= 0.10, f"read coverage gap {gap:.6%} exceeds 10%"
    return gap


def specialize(sr, topology):
    """Expose the known steady branch; native parity and rebuild flags guard it."""
    source = inspect.getsource(topology._read_at_nulls)
    anchor = """    accepted, forms = jax.lax.cond(
        any_lane_overflow(jnp.sum(retained) > compact_capacity),
        full_candidates,
        compact_candidates,
    )"""
    assert source.count(anchor) == 1
    source = source.replace(
        anchor,
        "    accepted, forms = compact_candidates()\n"
        "    accepted = accepted & (jnp.sum(retained) <= compact_capacity)",
    )
    namespace = dict(vars(topology))
    exec(compile(source, "<read-candidates>", "exec"), namespace)
    full = namespace["_read_at_nulls"]
    source = inspect.getsource(sr.read_batch)
    anchor = """        return jax.lax.cond(
            jnp.any(changed), lambda _: full_verdict(), lambda _: carried, None
        )"""
    assert source.count(anchor) == 1
    source = source.replace(
        anchor,
        "        return carried._replace(valid=carried.valid & ~jnp.any(changed))",
    )
    anchor = "    return jax.lax.cond(needed, rebuild, carry, None)"
    assert source.count(anchor) == 1
    source = source.replace(anchor, "    return carry(None)")
    namespace = dict(vars(sr))
    exec(compile(source, "<read-steady-branch>", "exec"), namespace)
    return full, namespace["read_batch"]


def owner(eqn, *, full=False):
    frames = eqn.source_info.traceback.frames
    names = {frame.function_name.split(".")[-1] for frame in frames}
    if eqn.primitive.name == "jit" and eqn.params.get("name") == "solve":
        return "fixed_points.solve"
    # Outer booking ownership includes its clipped quadrature and ring loops.
    if any(
        frame.file_name.endswith("/current.py") and frame.function_name == "book"
        for frame in frames
    ):
        return "current.book"
    if "solve" in names and any(
        frame.file_name.endswith("/fixed_points.py") and frame.function_name == "solve"
        for frame in frames
    ):
        return "fixed_points.solve"
    full_geometry = full and "_rebuilt_reading" in names
    if full_geometry and not names & {
        "_wall_events",
        "_curvature_derivatives",
        "saddle_normal_form",
    }:
        # Initial and terminal support share geometry inside this owner.
        return "support_read._rebuilt_reading"
    if "_wall_events" in names:
        return "topology._wall_events"
    if "_fragments_at_level" in names:
        return "topology._fragments_at_level"
    if names & {
        "_connected_fragment_labels",
        "_fragment_connections",
        "_support_at_level",
        "carried_decision_changed",
    }:
        return "topology._support_at_level/connectivity"
    if names & {"nodal_jets", "field_from_jets", "sampled_coefficient"}:
        return "support_read.nodal_jets/field_from_jets"
    if names & {"_curvature_derivatives", "saddle_normal_form"}:
        return "topology._curvature_derivatives/saddle_normal_form"
    if names & {
        "stationary_stencil",
        "_read_at_nulls",
        "read_at_stationary_points",
        "refresh_at_stationary_points",
    }:
        return "topology._read_at_nulls/qualification"
    return "support_read.read_batch/verdict"


def partition(closed, *, full=False):
    """Cut a traced straight-line graph, retaining every crossing variable."""
    from jax.extend import core

    graph = closed.jaxpr
    live = {v for v in graph.outvars if isinstance(v, core.Var)}
    kept = []
    for eqn in reversed(graph.eqns):
        assert not eqn.effects, eqn
        if any(v in live for v in eqn.outvars):
            kept.append(eqn)
            live.difference_update(eqn.outvars)
            live.update(v for v in eqn.invars if isinstance(v, core.Var))
    graph = core.Jaxpr(
        graph.constvars, graph.invars, graph.outvars, list(reversed(kept))
    )
    groups = []
    for eqn in graph.eqns:
        name = owner(eqn, full=full)
        if not groups or groups[-1][0] != name:
            groups.append((name, []))
        groups[-1][1].append(eqn)
    live = {v for v in graph.outvars if isinstance(v, core.Var)}
    regions = []
    for name, equations in reversed(groups):
        defined = {
            v for e in equations for v in e.outvars if not isinstance(v, core.DropVar)
        }
        outputs = [
            v
            for e in equations
            for v in e.outvars
            if v in live and not isinstance(v, core.DropVar)
        ]
        seen = set()
        inputs = []
        for equation in equations:
            assert not equation.effects, equation
            for var in equation.invars:
                if isinstance(var, core.Var) and var not in defined and var not in seen:
                    inputs.append(var)
                    seen.add(var)
        live = (live - defined) | set(inputs)
        region = core.Jaxpr([], inputs, outputs, equations)
        core.check_jaxpr(region)
        regions.append(
            (name, inputs, outputs, core.jaxpr_as_fun(core.ClosedJaxpr(region, [])))
        )
    return list(reversed(regions))


def common_subexpressions(closed):
    """Share pure expressions before compilation boundaries can duplicate them."""
    import numpy as np
    from jax.extend import core

    fingerprints = {}

    def freeze(value):
        if isinstance(value, core.ClosedJaxpr):
            return ("closed", graph_key(value.jaxpr), freeze(value.consts))
        if isinstance(value, core.Jaxpr):
            return ("graph", graph_key(value))
        if isinstance(value, dict):
            return tuple((k, freeze(v)) for k, v in sorted(value.items()))
        if isinstance(value, tuple | list):
            return (type(value).__name__, tuple(map(freeze, value)))
        if isinstance(value, set | frozenset):
            return ("set", frozenset(map(freeze, value)))
        if isinstance(value, np.ndarray | np.generic | float | complex):
            array = np.asarray(value)
            return (array.dtype.str, array.shape, array.tobytes())
        try:
            hash(value)
        except TypeError:
            return (type(value), id(value))
        return (type(value), value)

    def graph_key(graph):
        if id(graph) in fingerprints:
            return fingerprints[id(graph)]
        variables = {}

        def atom(var):
            if isinstance(var, core.Literal):
                return ("literal", freeze(var.val))
            if var not in variables:
                variables[var] = len(variables)
            return ("var", variables[var], str(var.aval))

        signature = tuple(map(atom, (*graph.constvars, *graph.invars)))
        instructions = tuple(
            (
                eqn.primitive,
                tuple(map(atom, eqn.invars)),
                tuple(map(atom, eqn.outvars)),
                freeze(eqn.params),
                freeze(eqn.effects),
            )
            for eqn in graph.eqns
        )
        result = (signature, instructions, tuple(map(atom, graph.outvars)))
        fingerprints[id(graph)] = result
        return result

    aliases, seen, equations = {}, {}, []

    def representative(var):
        return var if isinstance(var, core.Literal) else aliases.get(var, var)

    for eqn in closed.jaxpr.eqns:
        inputs = list(map(representative, eqn.invars))
        key = (
            eqn.primitive,
            tuple(
                ("literal", freeze(v.val)) if isinstance(v, core.Literal) else v
                for v in inputs
            ),
            freeze(eqn.params),
            tuple(str(v.aval) for v in eqn.outvars),
        )
        shareable = not eqn.effects and all(
            not isinstance(v, core.DropVar) for v in eqn.outvars
        )
        if shareable and key in seen:
            for var, prior in zip(eqn.outvars, seen[key], strict=True):
                aliases[var] = prior
        else:
            equations.append(eqn.replace(invars=inputs))
            if shareable:
                seen[key] = eqn.outvars
    graph = closed.jaxpr
    shared = core.Jaxpr(
        graph.constvars,
        graph.invars,
        list(map(representative, graph.outvars)),
        equations,
    )
    core.check_jaxpr(shared)
    return core.ClosedJaxpr(shared, closed.consts), len(graph.eqns) - len(equations)


def counts(lowered, compiled):
    text = compiled.as_text()
    return dict(
        optimized_hlo_ops=len(re.findall(r"^\s*%[^\n]* = ", text, re.M)),
        stablehlo_ops=str(lowered.compiler_ir()).count("stablehlo."),
        fusion_sites=len(re.findall(r"\bfusion\(", text)),
        while_sites=len(re.findall(r"\bwhile\(", text)),
        custom_call_sites=len(re.findall(r"\bcustom-call\(", text)),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=["full", "carried"], required=True)
    parser.add_argument("--widths", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.extend import core
    from nova.equilibrium import support_read as sr, topology
    from scripts.prototypes.bank_newton_step.measure import machine

    assert jax.config.jax_enable_x64 and args.repeats >= 12
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    payload = dict(
        revision=revision,
        module=sr.__file__,
        cwd=str(Path.cwd().resolve()),
        backend=jax.default_backend(),
        device=str(jax.devices()[0]),
        kind=args.kind,
        job=os.environ.get("SLURM_JOB_ID"),
        instrument_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=[],
    )
    print("HEADER", json.dumps(payload), flush=True)
    print("LOAD_BANK", flush=True)
    bank, cache, roots, convention, policy, profile, quadrature, moments, _ = machine(
        "diverted"
    )
    print("BANK_READY", bank.rows.shape, flush=True)
    full_owner, steady_read = specialize(sr, topology)

    def whole(
        bank,
        cache,
        state,
        seeds,
        prior,
        tracking,
        convention,
        policy,
        quadrature,
    ):
        support = sr.read_batch(
            bank, cache, state, seeds, prior, convention, policy, tracking=tracking
        )

        def booking(values, reading):
            jets = sr.nodal_jets(bank, values)
            psi = jets[bank.sample_index, 0]
            return sr.book(bank, reading, psi, profile, quadrature)

        booked = jax.vmap(booking)(state, support)
        return support, booked

    def wall(program, inputs, repeats):
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = jax.block_until_ready(program(*inputs))
            samples.append(1000 * (time.perf_counter() - start))
        return result, samples

    for width in args.widths:
        scales = jnp.linspace(2e-5, 6e-5, 16)[:width]
        state = tuple(value[None] * (1 + scales[:, None]) for value in moments)
        seeds = jnp.broadcast_to(
            roots + 0.03 * jnp.median(bank.geometry.pitch), (width, *roots.shape)
        )
        prior = jnp.broadcast_to(roots, seeds.shape)
        native = jax.jit(whole)
        initial = jax.block_until_ready(
            native(
                bank,
                cache,
                state,
                seeds,
                prior,
                None,
                convention,
                policy,
                quadrature,
            )
        )[0]
        assert bool(jnp.all(initial.valid))
        assert bool(jnp.all(initial.reading.x_point_valid.sum(axis=-1) == 1))
        tracking = sr.tracking_of(initial) if args.kind == "carried" else None
        inputs = (
            bank,
            cache,
            state,
            seeds,
            prior,
            tracking,
            convention,
            policy,
            quadrature,
        )
        print("NATIVE_COMPILE", width, args.kind, flush=True)
        lowered = native.lower(*inputs)
        program = lowered.compile()
        expected = jax.block_until_ready(program(*inputs))
        assert bool(jnp.all(expected[0].valid)) and bool(jnp.all(expected[1].valid))
        assert bool(jnp.all(expected[0].rebuilt == (args.kind == "full")))
        _, native_before = wall(program, inputs, args.repeats // 2)
        with ExitStack() as stack:
            stack.enter_context(patch.object(topology, "_read_at_nulls", full_owner))
            stack.enter_context(patch.object(sr, "read_batch", steady_read))
            closed = jax.make_jaxpr(lambda *values: whole(*values))(*inputs)
        closed, removed = common_subexpressions(closed)
        print("SHARED_EXPRESSIONS", width, removed, flush=True)
        flat_inputs, _ = jax.tree.flatten(inputs)
        env = dict(zip(closed.jaxpr.invars, flat_inputs, strict=True))
        env.update(zip(closed.jaxpr.constvars, closed.consts, strict=True))
        regions = partition(closed, full=args.kind == "full")
        row = dict(
            width=width,
            kind=args.kind,
            regions=[],
            native_counts=counts(lowered, program),
            native_before_ms=native_before,
        )
        stage_samples = {}
        compiled_regions = []
        for index, (name, invars, outvars, fn) in enumerate(regions):
            values = [env[v] for v in invars]
            print(
                "REGION_COMPILE",
                width,
                index,
                name,
                len(invars),
                len(outvars),
                flush=True,
            )
            lower = jax.jit(fn).lower(*values)
            compiled = lower.compile()
            result = jax.block_until_ready(compiled(*values))
            env.update(zip(outvars, result, strict=True))
            compiled_regions.append((name, invars, outvars, compiled))
            entry = dict(index=index, owner=name, **counts(lower, compiled))
            row["regions"].append(entry)
        actual = [
            var.val if isinstance(var, core.Literal) else env[var]
            for var in closed.jaxpr.outvars
        ]
        expected_leaves = jax.tree.leaves(expected)
        assert len(actual) == len(expected_leaves)
        maximum = 0.0
        for value, reference in zip(actual, expected_leaves, strict=True):
            a, b = np.asarray(value), np.asarray(reference)
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True)
            if np.issubdtype(a.dtype, np.floating) and np.any(np.isfinite(a)):
                maximum = max(maximum, float(np.nanmax(np.abs(a - b))))
        # Repeat the whole dependency chain; each exclusive region is synchronized.
        chain_ms = []
        for sample in range(args.repeats):
            total_start = time.perf_counter()
            per_owner = {}
            for name, invars, outvars, compiled in compiled_regions:
                values = [env[v] for v in invars]
                start = time.perf_counter()
                result = jax.block_until_ready(compiled(*values))
                elapsed = 1000 * (time.perf_counter() - start)
                env.update(zip(outvars, result, strict=True))
                per_owner[name] = per_owner.get(name, 0.0) + elapsed
            chain_ms.append(1000 * (time.perf_counter() - total_start))
            for name, value in per_owner.items():
                stage_samples.setdefault(name, []).append(value)
        _, native_after = wall(program, inputs, args.repeats - args.repeats // 2)
        row.update(
            stages={
                name: dict(median_ms=statistics.median(samples), samples_ms=samples)
                for name, samples in stage_samples.items()
            },
            native_after_ms=native_after,
            native_ms=statistics.median(native_before + native_after),
            chain_ms=chain_ms,
            maximum_absolute_difference=maximum,
            valid=True,
            rebuilt=np.asarray(expected[0].rebuilt).tolist(),
            cut_cells=np.sum(
                (np.asarray(expected[0].reading.membership) > 0)
                & (np.asarray(expected[0].reading.membership) < 1 - 1e-14),
                axis=-1,
            ).tolist(),
        )
        times = {name: entry["median_ms"] for name, entry in row["stages"].items()}
        row["coverage_gap"] = sum(times.values()) / row["native_ms"] - 1
        payload["rows"].append(row)
        args.output.write_text(json.dumps(payload, indent=2))
        print(
            "SUMMARY",
            json.dumps(
                {
                    key: value
                    for key, value in row.items()
                    if key
                    not in (
                        "regions",
                        "native_before_ms",
                        "native_after_ms",
                        "chain_ms",
                    )
                }
            ),
            flush=True,
        )
        coverage(times, row["native_ms"])
        omitted = max(times, key=times.get)
        try:
            coverage(
                {name: value for name, value in times.items() if name != omitted},
                row["native_ms"],
            )
        except AssertionError as error:
            row["negative_control"] = dict(
                omitted=omitted, refusal=str(error), passed=True
            )
            print("NEGATIVE_CONTROL", omitted, str(error), flush=True)
        else:
            raise AssertionError(
                "omitting the dominant stage escaped the coverage guard"
            )
        args.output.write_text(json.dumps(payload, indent=2))
    payload["complete"] = True
    args.output.write_text(json.dumps(payload, indent=2))
    print("ALL_PASS", args.kind, args.widths, flush=True)


if __name__ == "__main__":
    main()
