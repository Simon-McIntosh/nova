"""Traversal helpers over a lowered jaxpr, shared by the scan and loop guards.

A lowered program nests sub-jaxprs inside equation parameters, in a different
position for each primitive - a ``scan`` carries its body in ``jaxpr``, a
``while`` in ``body_jaxpr``/``cond_jaxpr``, a ``jit`` in ``jaxpr`` - so every
guard that reads a program's structure ends up writing the same descent. This
module owns that descent once. :func:`subjaxprs` yields the immediate nested
programs of one equation; :func:`scan_sites` walks a whole program and yields
the mapped length and unroll factor of every scan in it; :func:`scan_axes`
projects :func:`scan_sites` onto the lengths, which is what a guard keyed on
the scan axis needs.
"""

from __future__ import annotations


def subjaxprs(equation):
    """Yield the immediate nested jaxprs carried in one equation's parameters."""
    for parameter in equation.params.values():
        for value in parameter if isinstance(parameter, tuple | list) else (parameter,):
            inner = getattr(value, "jaxpr", value)
            if hasattr(inner, "eqns"):
                yield inner


def scan_sites(traced):
    """Yield ``(mapped length, unroll)`` for every scan in a lowered program.

    The walk descends through equation parameters and nested jaxprs and guards
    against revisiting a shared sub-jaxpr by identity, so a scan reachable by
    two paths is reported once.
    """
    visited = set()

    def walk(value):
        if id(value) in visited:
            return
        visited.add(id(value))
        if hasattr(value, "eqns"):
            for equation in value.eqns:
                if equation.primitive.name == "scan":
                    yield equation.params["length"], equation.params["unroll"]
                yield from walk(equation.params)
        elif hasattr(value, "jaxpr"):
            yield from walk(value.jaxpr)
        elif isinstance(value, dict):
            for item in value.values():
                yield from walk(item)
        elif isinstance(value, list | tuple):
            for item in value:
                yield from walk(item)

    return tuple(walk(traced))


def scan_axes(traced):
    """Yield the leading axis length of every scan in a lowered program."""
    return tuple(length for length, _unroll in scan_sites(traced))
