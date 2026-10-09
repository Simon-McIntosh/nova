"""Taylor-mode differentiation of the Biot point jet against nested jacfwd.

The topology read evaluates the analytic polygon kernel at a moving target
point. Its present construction differentiates the kernel point jet by nesting
``jax.jacfwd``, and ``value_gradient``'s autodiff accounts for essentially every
expanded equation the read traces. This prototype measures whether one Taylor
pass (``jax.experimental.jet``) to a given order costs less to compile than the
nested forward differentiation, and whether the two agree.

Definitions used throughout, stated so the numbers are unambiguous:

* the base point jet is ``g(point) = [value(point), gradient(point)]`` of the
  analytic polygon coupling (``BiotMomentCoupling.value_gradient``), with the
  target point the only varying input;
* order ``N`` means ``N`` nested ``jax.jacfwd`` applications to ``g`` for the
  nested arm, and a single ``jet`` call asked for ``N`` Taylor coefficients for
  the Taylor arm. Order 1 is therefore the point jet's first derivative (the
  Hessian of the kernel value), which is what the read's ``evaluate`` takes;
* an equation counts as *unique* when its jaxpr node has not been visited, and
  *expanded* at every nested call site, matching the counters the topology-read
  suite already uses.

Run as separate fresh processes so every compile is cold. The persistent
compilation cache is disabled in this module's import prelude, before ``jax`` is
imported, and every row records what it read for the cache configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "0"
for _name in (
    "JAX_COMPILATION_CACHE_DIR",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
    "JAX_PERSISTENT_CACHE_MAX_SIZE_BYTES",
):
    os.environ.pop(_name, None)

import numpy as np  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.experimental.jet as taylor  # noqa: E402

from nova.jax.config import configure_dtypes  # noqa: E402

configure_dtypes()
assert jax.config.jax_enable_x64, "extended precision must be on before any array"

from nova.equilibrium.topology import BiotMomentCoupling  # noqa: E402
from nova.biot.polygonanalytic import packed_analytic_moments  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jet_broadcast_rules import patch_broadcast_minmax  # noqa: E402

from jax import lax  # noqa: E402

STOCK_REFUSALS = {}

# Stock `jax.experimental.jet` has no rule for `lax.atan_p` and its min/max rules
# assume equal shapes, which a broadcast scalar against an array violates. The
# patches below are the workarounds; each is applied only when requested so the
# refusal can be measured with the library unmodified.
JET_PATCHES = ("minmax-broadcast", "atan-derivative", "asin-acos-derivative")


APPLIED_PATCHES: list[str] = []


def apply_jet_patches() -> None:
    patch_broadcast_minmax()
    APPLIED_PATCHES.extend(JET_PATCHES)
    taylor.def_deriv(lax.atan_p, lambda x: 1 / (1 + lax.square(x)))
    taylor.def_deriv(lax.asin_p, lambda x: 1 / lax.sqrt(1 - lax.square(x)))
    taylor.def_deriv(lax.acos_p, lambda x: -1 / lax.sqrt(1 - lax.square(x)))

# A single convex-ish polygon spanning the D-shaped plasma outboard midplane, the
# same section shape the kernel-compile rows use.
POLYGON = np.asarray(
    [[4.91, -0.10], [5.08, -0.14], [5.17, 0.02], [5.04, 0.15], [4.88, 0.08]]
)

_COUPLING = BiotMomentCoupling.from_polygons([POLYGON])
_MOMENTS = (
    jnp.asarray([1.0]),
    jnp.asarray([0.003]),
    jnp.asarray([-0.002]),
)


def value_gradient(point):
    """The read's point-jet body, with target point the only varying input.

    Mirrors ``BiotMomentCoupling.value_gradient`` exactly but without the
    ``jax.jit`` wrapper, so the jaxpr under test is the differentiated kernel
    expression and not a call boundary around it. Both arms share this base, so
    the comparison between them is unaffected.
    """
    coupling = _COUPLING
    rows = packed_analytic_moments(
        jnp,
        jnp.broadcast_to(point[0], coupling.norm.shape),
        jnp.broadcast_to(point[1], coupling.norm.shape),
        coupling.edge,
        coupling.weight,
        coupling.norm,
        coupling.centre.T,
        coupling.centre.T,
        coupling.reflection_axis,
        coupling.reflection_partner,
    )
    coefficient = coupling.coefficients(_MOMENTS).T
    value = jnp.sum(jnp.stack(rows[:3]) * coefficient)
    radial_field = jnp.sum(jnp.stack(rows[3:6]) * coefficient)
    vertical_field = jnp.sum(jnp.stack(rows[6:]) * coefficient)
    gradient = (
        2 * jnp.pi * point[0] * jnp.stack((vertical_field, -radial_field))
    )
    return jnp.concatenate(
        (jnp.reshape(value, (1,)), jnp.reshape(gradient, (2,)))
    )


def nested_function(order: int):
    def f(point):
        return value_gradient(point)

    for _ in range(order):
        f = jax.jacfwd(f)
    return f


def taylor_function(order: int):
    def f(point):
        _, series = taylor.jet(
            value_gradient,
            (point,),
            tuple([jnp.ones_like(point)]
                  + [jnp.zeros_like(point)] * (order - 1)),
        )
        return series[order - 1]

    return f


def build(mode: str, order: int):
    if mode == "nested":
        return nested_function(order)
    if mode == "taylor":
        return taylor_function(order)
    raise ValueError(mode)


def _equation_counts(closed):
    seen = set()
    unique = 0

    def walk(value):
        nonlocal unique
        if hasattr(value, "eqns"):
            fresh = id(value) not in seen
            seen.add(id(value))
            if fresh:
                unique += len(value.eqns)
            return len(value.eqns) + sum(
                walk(parameter)
                for equation in value.eqns
                for parameter in equation.params.values()
            )
        if hasattr(value, "jaxpr"):
            return walk(value.jaxpr)
        if isinstance(value, tuple | list):
            return sum(walk(item) for item in value)
        if isinstance(value, dict):
            return sum(walk(item) for item in value.values())
        return 0

    expanded = walk(closed)
    return {"unique_equations": unique, "expanded_equations": expanded}


def _primitive_census(closed):
    from collections import Counter

    primitives = Counter()

    def walk(value):
        if hasattr(value, "eqns"):
            for equation in value.eqns:
                primitives[equation.primitive.name] += 1
                for parameter in equation.params.values():
                    walk(parameter)
        elif hasattr(value, "jaxpr"):
            walk(value.jaxpr)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, tuple | list):
            for item in value:
                walk(item)

    walk(closed)
    return dict(primitives)


def _cache_facts():
    facts = {}
    for name in (
        "jax_enable_compilation_cache",
        "jax_persistent_cache_min_compile_time_secs",
    ):
        try:
            facts[name] = str(jax.config.__getattr__(name))
        except Exception as error:  # pragma: no cover - config naming varies
            facts[name] = f"unreadable: {type(error).__name__}"
    facts["env_JAX_ENABLE_COMPILATION_CACHE"] = os.environ.get(
        "JAX_ENABLE_COMPILATION_CACHE", ""
    )
    facts["jax_compilation_cache_dir"] = str(
        getattr(jax.config, "jax_compilation_cache_dir", "")
    )
    return facts


def _emit(out: str, row: dict) -> None:
    line = json.dumps(row, sort_keys=True)
    print("ROW " + line, flush=True)
    if out:
        with open(out, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())


POINT = jnp.asarray([5.05, 0.02])


def run_row(mode: str, order: int, out: str, budget: float) -> dict:
    row = {
        "kind": "row",
        "mode": mode,
        "order": order,
        "target_point": [5.05, 0.02],
        "cache": _cache_facts(),
        "jet_patches": list(APPLIED_PATCHES),
    }
    try:
        fn = build(mode, order)
        t0 = time.perf_counter()
        closed = jax.make_jaxpr(fn)(POINT)
        row["trace_seconds"] = time.perf_counter() - t0
        row.update(_equation_counts(closed))
        row["primitives"] = _primitive_census(closed)
        t1 = time.perf_counter()
        lowered = jax.jit(fn).lower(POINT)
        row["lower_seconds"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        compiled = lowered.compile()
        row["compile_seconds"] = time.perf_counter() - t2
        row["cold_compile_seconds"] = time.perf_counter() - t1
        result = compiled(POINT)
        jax.block_until_ready(result)
        row["total_seconds"] = time.perf_counter() - t0
        row["output_shape"] = list(np.shape(np.asarray(result)))
        row["status"] = "ok"
    except Exception as error:  # noqa: BLE001 - the refusal is the measurement
        row["status"] = "refused"
        row["error_type"] = type(error).__name__
        row["error"] = str(error)[:2000]
    _emit(out, row)
    return row


def _directions(key, count: int, far: bool):
    """Target points and unit directions spanning near and far field."""
    radius = jax.random.uniform(key, (count,), minval=4.7, maxval=5.35)
    if far:
        radius = jax.random.uniform(key, (count,), minval=6.0, maxval=60.0)
    height = jax.random.uniform(
        jax.random.fold_in(key, 1), (count,), minval=-3.0, maxval=3.0
    )
    points = jnp.stack((radius, height), axis=1)
    angle = jax.random.uniform(
        jax.random.fold_in(key, 2), (count,), minval=0.0, maxval=2 * np.pi
    )
    directions = jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=1)
    return points, directions


def _nested_directional(order: int):
    """Point function: N-fold jacfwd contracted ``order`` times with a direction."""

    def f(point, direction):
        fn = value_gradient
        for _ in range(order):
            fn = jax.jacfwd(fn)
        value = fn(point)
        for _ in range(order):
            value = jnp.tensordot(value, direction, axes=([value.ndim - 1], [0]))
        return value

    return f


def _taylor_directional(order: int):
    def f(point, direction):
        _, series = taylor.jet(
            value_gradient,
            (point,),
            tuple([direction] + [jnp.zeros_like(direction)] * (order - 1)),
        )
        return series[order - 1]

    return f


def run_identity(out: str, points_per_band: int) -> dict:
    """Jet coefficients against nested jacfwd, point by point.

    Compiled once per order and executed over the point set in a Python loop,
    never vmapped: a vmap would duplicate the already-large derivative graph
    once per sample and the compile would dominate the measurement.
    """
    row = {
        "kind": "identity",
        "points_per_band": points_per_band,
        "jet_patches": list(APPLIED_PATCHES),
    }
    key = jax.random.PRNGKey(20261009)
    maxima = {}
    for order in (1, 2, 3, 4):
        nested = jax.jit(_nested_directional(order))
        taylored = jax.jit(_taylor_directional(order))
        for band, far in (("near", False), ("far", True)):
            points, directions = _directions(key, points_per_band, far)
            nested_rows = []
            taylor_rows = []
            for index in range(points_per_band):
                point = points[index]
                direction = directions[index]
                nested_rows.append(
                    np.asarray(nested(point, direction), dtype=np.float64)
                )
                taylor_rows.append(
                    np.asarray(taylored(point, direction), dtype=np.float64)
                )
            nested_value = np.stack(nested_rows)
            taylor_value = np.stack(taylor_rows)
            absolute = np.abs(nested_value - taylor_value)
            scale = np.maximum(np.abs(nested_value), np.abs(taylor_value))
            relative = np.where(
                scale > 0, absolute / np.maximum(scale, 1e-300), 0.0
            )
            maxima[f"order{order}_{band}"] = {
                "max_abs": float(absolute.max()),
                "max_rel": float(relative.max()),
                "max_nested": float(np.abs(nested_value).max()),
                "samples": int(points_per_band),
            }
            print(
                f"IDENTITY order={order} band={band} "
                f"max_abs={absolute.max():.6g} max_rel={relative.max():.6g}",
                flush=True,
            )
    row["maxima"] = maxima
    row["status"] = "ok"
    _emit(out, row)
    return row


def run_primitives(out: str) -> dict:
    """Report which primitives jet refuses, one module import at a time."""
    row = {"kind": "primitives"}
    failures = []
    for order in (1, 2, 3, 4):
        for name, fn in (
            ("value_gradient", lambda point: value_gradient(point)),
            (
                "taylor",
                lambda point, order=order: taylor.jet(
                    value_gradient,
                    (point,),
                    tuple([jnp.ones_like(point)]
                          + [jnp.zeros_like(point)] * (order - 1)),
                ),
            ),
        ):
            try:
                closed = jax.make_jaxpr(fn)(POINT)
                failures.append(
                    {
                        "order": order,
                        "name": name,
                        "status": "traced",
                        "primitives": _primitive_census(closed),
                    }
                )
            except Exception as error:  # noqa: BLE001
                failures.append(
                    {
                        "order": order,
                        "name": name,
                        "status": "refused",
                        "error_type": type(error).__name__,
                        "error": str(error)[:2000],
                    }
                )
    row["rows"] = failures
    row["status"] = "ok"
    _emit(out, row)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.environ.get("TAYLOR_JSONL", ""))
    parser.add_argument(
        "--stock-jet",
        action="store_true",
        help="measure the library jet rules unmodified (refusal rows)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    row = sub.add_parser("row")
    row.add_argument("--mode", choices=("nested", "taylor"), required=True)
    row.add_argument("--order", type=int, required=True)
    row.add_argument("--budget", type=float, default=900.0)

    identity = sub.add_parser("identity")
    identity.add_argument("--points-per-band", type=int, default=500)

    sub.add_parser("primitives")

    args = parser.parse_args(argv)
    print(
        f"HOST jax={jax.__version__} devices={jax.devices()} "
        f"module={__file__} cwd={os.getcwd()} interp={sys.executable}",
        flush=True,
    )
    if not args.stock_jet:
        apply_jet_patches()
    if args.command == "row":
        run_row(args.mode, args.order, args.out, args.budget)
    elif args.command == "identity":
        run_identity(args.out, args.points_per_band)
    elif args.command == "primitives":
        run_primitives(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())