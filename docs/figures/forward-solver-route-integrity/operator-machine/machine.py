"""Prototype control-flow outlining for shared live operator evaluations."""

from dataclasses import dataclass

import jax
from jax.extend import core
import jax.numpy as jnp


@dataclass(eq=False)
class Register:
    aval: object


@dataclass(eq=False)
class Constant:
    value: object


def operator_evaluation(function):
    def shared_operator_evaluation(*args):
        return function(*args)

    return jax.jit(shared_operator_evaluation)


class Machine:
    def __init__(self):
        self.blocks = []
        self.used = {}
        self.contains_cache = {}
        self.operators = {}

    def emit(self, reads, builder):
        self.used.update((r, None) for r in reads if isinstance(r, Register))
        index = len(self.blocks)
        self.blocks.append(builder)
        return index

    def copy(self, sources, targets, next_pc):
        def build(read, write):
            def block(state):
                values = [read(state, r) for r in sources]
                return write(state, dict(zip(targets, values)), next_pc)

            return block

        return self.emit(sources, build)

    def contains(self, closed):
        jaxpr = closed.jaxpr if isinstance(closed, core.ClosedJaxpr) else closed
        if jaxpr not in self.contains_cache:
            self.contains_cache[jaxpr] = any(
                self.request(e)
                or any(self.contains(j) for j in core.jaxprs_in_params(e.params))
                for e in jaxpr.eqns
            )
        return self.contains_cache[jaxpr]

    @staticmethod
    def request(eqn):
        return (
            eqn.primitive.name == "jit"
            and eqn.params.get("name") == "shared_operator_evaluation"
        )

    def lower(self, closed, inputs, outputs, next_pc):
        if isinstance(closed, core.ClosedJaxpr):
            jaxpr, consts = closed.jaxpr, closed.consts
        else:
            jaxpr, consts = closed, ()
        env = dict(zip(jaxpr.constvars, map(Constant, consts)))
        env.update(zip(jaxpr.invars, inputs))
        for eqn in jaxpr.eqns:
            for var in eqn.outvars:
                env[var] = Register(var.aval)

        def resolve(var):
            return Constant(var.val) if isinstance(var, core.Literal) else env[var]

        pc = self.copy([resolve(v) for v in jaxpr.outvars], outputs, next_pc)
        groups = []
        straight = []
        for eqn in jaxpr.eqns:
            special = self.request(eqn) or any(
                self.contains(j) for j in core.jaxprs_in_params(eqn.params)
            )
            if special:
                if straight:
                    groups.append(straight)
                    straight = []
                groups.append(eqn)
            else:
                straight.append(eqn)
        if straight:
            groups.append(straight)
        for group in reversed(groups):
            if isinstance(group, list):
                pc = self.straight(group, resolve, pc)
            else:
                pc = self.control(
                    group,
                    [resolve(v) for v in group.invars],
                    [env[v] for v in group.outvars],
                    pc,
                )
        return pc

    def straight(self, equations, resolve, next_pc):
        produced = set()
        reads = []
        for eqn in equations:
            reads.extend(
                resolve(v)
                for v in eqn.invars
                if not isinstance(v, core.Literal) and v not in produced
            )
            produced.update(eqn.outvars)

        def build(read, write):
            def block(state):
                env = {}
                for eqn in equations:
                    args = [
                        v.val
                        if isinstance(v, core.Literal)
                        else env[v]
                        if v in env
                        else read(state, resolve(v))
                        for v in eqn.invars
                    ]
                    params = eqn.params
                    bind_params = eqn.primitive.get_bind_params(params)
                    result = eqn.primitive.bind(*args, **bind_params)
                    values = result if eqn.primitive.multiple_results else [result]
                    env.update(zip(eqn.outvars, values))
                return write(
                    state, {resolve(v): value for v, value in env.items()}, next_pc
                )

            return block

        return self.emit(reads, build)

    def control(self, eqn, inputs, outputs, next_pc):
        params = eqn.params
        name = eqn.primitive.name
        if self.request(eqn):
            closed = params["jaxpr"]
            key = closed
            if key not in self.operators:
                args = [Register(v.aval) for v in closed.jaxpr.invars]
                results = [Register(v.aval) for v in closed.jaxpr.outvars]
                resume = Register(jax.ShapeDtypeStruct((), jnp.int32))

                def build(read, write):
                    def block(state):
                        values = core.jaxpr_as_fun(closed)(
                            *[read(state, r) for r in args]
                        )
                        return write(
                            state, dict(zip(results, values)), read(state, resume)
                        )

                    return block

                entry = self.emit([*args, resume], build)
                self.operators[key] = args, results, resume, entry
            args, results, resume, entry = self.operators[key]
            returned = self.copy(results, outputs, next_pc)
            return self.copy(
                [*inputs, Constant(jnp.int32(returned))], [*args, resume], entry
            )
        if name == "custom_linear_solve":
            lengths = params["const_lengths"]
            offset = lengths.matvec + lengths.vecmat
            solve_inputs = (
                inputs[offset : offset + lengths.solve] + inputs[sum(lengths) :]
            )
            return self.lower(params["jaxprs"].solve, solve_inputs, outputs, next_pc)
        if name == "jit":
            return self.lower(params["jaxpr"], inputs, outputs, next_pc)
        if name == "cond":
            entries = [
                self.lower(branch, inputs[1:], outputs, next_pc)
                for branch in params["branches"]
            ]

            def build(read, write):
                def block(state):
                    pc = jnp.asarray(entries, dtype=jnp.int32)[
                        jnp.clip(read(state, inputs[0]), 0, len(entries) - 1)
                    ]
                    return write(state, {}, pc)

                return block

            return self.emit(inputs[:1], build)
        if name == "while":
            nc = params["cond_nconsts"]
            nb = params["body_nconsts"]
            carry = [Register(r.aval) for r in outputs]
            predicate = Register(jax.ShapeDtypeStruct((), jnp.bool_))
            choose = self.emit([predicate], None)
            test = self.lower(
                params["cond_jaxpr"], [*inputs[:nc], *carry], [predicate], choose
            )
            body = self.lower(
                params["body_jaxpr"], [*inputs[nc : nc + nb], *carry], carry, test
            )
            done = self.copy(carry, outputs, next_pc)

            def build(read, write):
                return lambda state: write(
                    state, {}, jnp.where(read(state, predicate), body, done)
                )

            self.blocks[choose] = build
            return self.copy(inputs[nc + nb :], carry, test)
        if name == "scan":
            const_tree, carry_tree, _ = params["ft_in"].unpack()
            nc, nk, length = len(const_tree), len(carry_tree), params["length"]
            closed = params["jaxpr"]
            carry = [Register(r.aval) for r in outputs[:nk]]
            slices = [Register(v.aval) for v in closed.jaxpr.invars[nc + nk :]]
            ys = [Register(r.aval) for r in outputs[nk:]]
            values = [Register(v.aval) for v in closed.jaxpr.outvars[nk:]]
            index = Register(jax.ShapeDtypeStruct((), jnp.int32))
            choose = self.emit([index], None)

            def advance_build(read, write):
                def block(state):
                    i = read(state, index)
                    position = length - 1 - i if params["reverse"] else i
                    updates = {
                        y: jax.lax.dynamic_update_index_in_dim(
                            read(state, y), read(state, v), position, 0
                        )
                        for y, v in zip(ys, values)
                    }
                    updates[index] = i + 1
                    return write(state, updates, choose)

                return block

            advance = self.emit([index, *ys, *values], advance_build)
            body = self.lower(
                closed, [*inputs[:nc], *carry, *slices], [*carry, *values], advance
            )

            def slice_build(read, write):
                def block(state):
                    i = read(state, index)
                    position = length - 1 - i if params["reverse"] else i
                    updates = {
                        target: jax.lax.dynamic_index_in_dim(
                            read(state, source), position, 0, False
                        )
                        for target, source in zip(slices, inputs[nc + nk :])
                    }
                    return write(state, updates, body)

                return block

            prepare = self.emit([index, *inputs[nc + nk :]], slice_build)
            done = self.copy([*carry, *ys], outputs, next_pc)

            def choose_build(read, write):
                return lambda state: write(
                    state, {}, jnp.where(read(state, index) < length, prepare, done)
                )

            self.blocks[choose] = choose_build
            return self.copy(
                [
                    *inputs[nc : nc + nk],
                    Constant(jnp.int32(0)),
                    *[Constant(jnp.zeros(r.aval.shape, r.aval.dtype)) for r in ys],
                ],
                [*carry, index, *ys],
                choose,
            )
        raise ValueError(f"shared operator under unsupported control primitive {name}")

    def execute(self, closed, arguments):
        inputs = [Constant(v) for v in arguments]
        outputs = [Register(v.aval) for v in closed.jaxpr.outvars]
        self.used.update((r, None) for r in outputs)
        entry = self.lower(closed, inputs, outputs, -1)
        registers = tuple(self.used)
        slots = {r: i + 1 for i, r in enumerate(registers)}

        def read(state, reg):
            return reg.value if isinstance(reg, Constant) else state[slots[reg]]

        def write(state, updates, pc):
            values = list(state)
            values[0] = jnp.asarray(pc, dtype=jnp.int32)
            for reg, value in updates.items():
                if reg in slots:
                    values[slots[reg]] = value
            return tuple(values)

        branches = [builder(read, write) for builder in self.blocks]
        initial = (
            jnp.int32(entry),
            *[jnp.zeros(r.aval.shape, r.aval.dtype) for r in registers],
        )
        terminal = jax.lax.while_loop(
            lambda s: s[0] >= 0, lambda s: jax.lax.switch(s[0], branches, s), initial
        )
        return [read(terminal, r) for r in outputs]


def shared_operator_call(function, *args):
    flat, tree = jax.tree.flatten(args)
    result_tree = None

    def flattened(*leaves):
        nonlocal result_tree
        result = function(*jax.tree.unflatten(tree, leaves))
        values, result_tree = jax.tree.flatten(result)
        return values

    closed = jax.make_jaxpr(flattened)(*flat)
    result = Machine().execute(closed, flat)
    return jax.tree.unflatten(result_tree, result)
