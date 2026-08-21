"""Adjudicate one packed contour with arbitrary-precision closed forms."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import mpmath as mp
import numpy as np

import nova.biot.completeelliptic as completeelliptic
import nova.biot.gradedresidual as gradedresidual
import nova.biot.polygonanalytic as polygonanalytic
from nova.biot.polygon import MU0
from nova.biot.polygonanalytic import _Edge, _Vertex


HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
DEFAULT_RUNGS = (64, 128, 256, 512, 1024)


class ArbitraryPrecisionNamespace:
    """Small array namespace backed by object arrays of ``mpmath.mpf`` values."""

    @staticmethod
    def _scalar(value):
        if isinstance(value, bool | np.bool_):
            return bool(value)
        if isinstance(value, mp.mpf):
            return value
        if isinstance(value, np.generic):
            value = value.item()
        return mp.mpf(value)

    @classmethod
    def asarray(cls, value):
        array = np.asarray(value)
        if array.dtype == np.bool_:
            return array
        convert = np.frompyfunc(cls._scalar, 1, 1)
        return np.asarray(convert(array), dtype=object)

    @staticmethod
    def _map(function, value):
        return np.frompyfunc(function, 1, 1)(value)

    def abs(self, value):
        return self._map(mp.fabs, value)

    def arcsinh(self, value):
        return self._map(mp.asinh, value)

    def arctan(self, value):
        return self._map(mp.atan, value)

    def clip(self, value, lower, upper):
        return np.minimum(np.maximum(value, lower), upper)

    def log(self, value):
        return self._map(mp.log, value)

    def log1p(self, value):
        return self._map(mp.log1p, value)

    def minimum(self, left, right):
        return np.minimum(left, right)

    def ones_like(self, value):
        return np.full(np.shape(value), mp.mpf(1), dtype=object)

    def sign(self, value):
        return self._map(mp.sign, value)

    def sin(self, value):
        return self._map(mp.sin, value)

    def sinh(self, value):
        return self._map(mp.sinh, value)

    def sqrt(self, value):
        return self._map(mp.sqrt, value)

    def where(self, condition, left, right):
        return np.where(condition, left, right)

    def zeros_like(self, value):
        return np.full(np.shape(value), mp.mpf(0), dtype=object)


@lru_cache(maxsize=None)
def _high_precision_rule(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the production panel's Gauss rule at the active decimal precision."""
    points, weights = mp.gauss_quadrature(nodes // 2, "legendre")
    return np.asarray(list(points), dtype=object), np.asarray(
        list(weights), dtype=object
    )


def _install_precision_context() -> ArbitraryPrecisionNamespace:
    """Route every closed-form constant and residual node through ``mpmath``."""
    namespace = ArbitraryPrecisionNamespace()
    completeelliptic._HALF_PI = mp.pi / 2
    gradedresidual._rule = _high_precision_rule
    polygonanalytic._PANEL = (mp.mpf(0), mp.pi / 4)
    polygonanalytic.np = SimpleNamespace(pi=mp.pi)
    return namespace


def _arbitrary_flux(
    target_r: np.ndarray, target_z: np.ndarray, vertices: np.ndarray, *, nodes: int
) -> list[mp.mpf]:
    """Evaluate the scalar host contour order with arbitrary-precision terms."""
    xp = _install_precision_context()
    section = xp.asarray(vertices)
    local = section - section[0]
    following = np.roll(local, -1, axis=0)
    signed_twice_area = np.sum(
        local[:, 0] * following[:, 1] - following[:, 0] * local[:, 1]
    )
    sign = -mp.sign(signed_twice_area)
    area = mp.mpf("0.5") * abs(signed_twice_area)
    norm = sign * mp.mpf(str(MU0)) / area
    rolled = np.roll(section, -1, axis=0)
    edges = np.column_stack([section[:, 0], section[:, 1], rolled[:, 0], rolled[:, 1]])

    radius = xp.abs(xp.asarray(target_r)).ravel()
    height = xp.asarray(target_z).ravel()
    flux = xp.zeros_like(radius)
    corners: dict[int, _Vertex] = {}

    def corner(index: int) -> _Vertex:
        if index not in corners:
            corners[index] = _Vertex(
                radius,
                height,
                section[index, 0],
                section[index, 1],
                nodes,
                residual=False,
                xp=xp,
            )
        return corners[index]

    for index, edge in enumerate(edges):
        lower = corner(index)
        upper = corner((index + 1) % len(edges))
        part = _Edge(radius, height, edge, nodes, xp=xp)
        flux -= part.terms(upper)[0] - part.terms(lower)[0]
    return list(mp.mpf("0.5") * norm * radius * flux)


def _evaluate_rung(payload: tuple[int, int, np.ndarray, np.ndarray, np.ndarray]):
    """Evaluate one independent residual rung in a worker process."""
    nodes, digits, target_r, target_z, vertices = payload
    mp.mp.dps = digits
    values = _arbitrary_flux(target_r, target_z, vertices, nodes=nodes)
    return nodes, [mp.nstr(value, digits) for value in values]


def _deviation(values: list[mp.mpf], oracle: list[mp.mpf]) -> dict[str, str]:
    difference = [
        abs(value - expected) for value, expected in zip(values, oracle, strict=True)
    ]
    relative = [
        delta / max(abs(expected), mp.mpf("1e-999"))
        for delta, expected in zip(difference, oracle, strict=True)
    ]
    return {
        "max_absolute": mp.nstr(max(difference), 40),
        "max_relative": mp.nstr(max(relative), 40),
        "median_absolute": mp.nstr(median(difference), 40),
    }


def _rung_change(lower: list[mp.mpf], upper: list[mp.mpf]) -> dict[str, str | int]:
    difference = [abs(left - right) for left, right in zip(lower, upper, strict=True)]
    index = max(range(len(difference)), key=difference.__getitem__)
    relative = difference[index] / max(abs(upper[index]), mp.mpf("1e-999"))
    return {
        "max_absolute_change": mp.nstr(difference[index], 40),
        "max_relative_change": mp.nstr(relative, 40),
        "target_index": index,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, default=100)
    parser.add_argument("--source", type=int, default=184)
    parser.add_argument("--rungs", type=int, nargs="+", default=DEFAULT_RUNGS)
    parser.add_argument("--workers", type=int, default=len(DEFAULT_RUNGS))
    args = parser.parse_args()
    if args.digits < 100:
        raise ValueError("digits must be at least 100")
    if sorted(set(args.rungs)) != args.rungs:
        raise ValueError("rungs must be unique and increasing")
    if len(args.rungs) < 3:
        raise ValueError("at least three rungs are required to bound the tail")

    mp.mp.dps = args.digits
    data = np.load(WORK / "coarse-input.npz")
    present = ~np.signbit(data["weight"][:, args.source])
    vertices = data["edge"][present, :2, args.source]
    payloads = [
        (
            nodes,
            args.digits,
            data["target_r"],
            data["target_z"],
            vertices,
        )
        for nodes in args.rungs
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(payloads))) as pool:
        evaluated = dict(pool.map(_evaluate_rung, payloads))
    values = {
        nodes: [mp.mpf(value) for value in evaluated[nodes]] for nodes in args.rungs
    }
    terminal = values[args.rungs[-1]]

    production = [
        mp.mpf(value)
        for value in np.load(WORK / "coarse-numpy-reference.npy")[:, args.source]
    ]
    extended = {
        nodes: [
            mp.mpf(value)
            for value in json.loads(
                (
                    HERE
                    / (
                        "extended-precision-oracle.json"
                        if nodes == 128
                        else f"extended-precision-oracle-{nodes}.json"
                    )
                ).read_text()
            )["oracle_values"]
        ]
        for nodes in args.rungs
    }
    ladder = []
    for index in range(len(args.rungs) - 1):
        lower = args.rungs[index]
        upper = args.rungs[index + 1]
        ladder.append(
            {
                "from_nodes": lower,
                "to_nodes": upper,
                **_rung_change(values[lower], values[upper]),
            }
        )

    last_change = mp.mpf(ladder[-1]["max_absolute_change"])
    previous_change = mp.mpf(ladder[-2]["max_absolute_change"])
    ratio = last_change / previous_change
    tail_bound = mp.inf if ratio >= 1 else last_change * ratio / (1 - ratio)
    fp64_floor = max(abs(value) for value in terminal) * mp.mpf(
        str(np.finfo(np.float64).eps)
    )
    production_deviation = _deviation(production, terminal)
    production_lower_bound = max(
        mp.mpf(production_deviation["max_absolute"]) - tail_bound, mp.mpf(0)
    )
    dominated = production_lower_bound > fp64_floor

    receipt = {
        "arithmetic": {
            "decimal_digits": args.digits,
            "library": f"mpmath {mp.__version__}",
            "residual_rule": "arbitrary-precision Gauss-Legendre",
        },
        "deviation_from_terminal_oracle": {
            "banked_63_bit_ladder": {
                str(nodes): _deviation(extended[nodes], terminal)
                for nodes in args.rungs
            },
            "production_numpy": production_deviation,
        },
        "oracle_values": {
            str(nodes): [mp.nstr(value, args.digits) for value in values[nodes]]
            for nodes in args.rungs
        },
        "residual_error": {
            "fp64_absolute_floor": mp.nstr(fp64_floor, 40),
            "geometric_tail_bound": mp.nstr(tail_bound, 40),
            "last_change_ratio": mp.nstr(ratio, 40),
        },
        "richardson_ladder": ladder,
        "source_column": args.source,
        "verdict": {
            "production_value_dominated_by_amplified_cancellation": dominated,
            "reason": (
                "production deviation remains above the bounded residual tail "
                "and fp64 floor"
                if dominated
                else "the residual tail does not separate the production "
                "deviation from fp64"
            ),
            "shared_conditioned_rule_authorized": dominated,
        },
        "vertex_count": int(len(vertices)),
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
