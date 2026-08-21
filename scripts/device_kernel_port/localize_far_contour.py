"""Localize finite-section contour cancellation against a high-precision path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np

from nova.biot.pairedfloat import add, scale, subtract
from nova.biot.polygon import MU0, pack_section
from nova.biot.polygonanalytic import _Edge, _Vertex
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.device_kernel_port.extended_precision_oracle import (
    _install_precision_context,
)


HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
ORACLE = HERE / "arbitrary-precision-oracle.json"
OUTPUT = HERE / "far-contour-localization.json"
PAIR_GATE = mp.mpf("8.04e-5")

_PLAIN_LABELS = (
    "flux_second_arsinh_plain",
    "flux_arctangent_plain",
    "radial_second_arsinh_plain",
    "vertical_second_arsinh_plain",
    "vertical_arctangent_plain",
)
_ACROSS_LABELS = (
    "flux_second_arsinh_plane",
    "flux_arctangent_ring",
    "flux_arctangent_plane",
    "radial_second_arsinh_plane",
    "vertical_second_arsinh_plane",
    "vertical_arctangent_ring",
    "vertical_arctangent_plane",
)
_ROOT_LABELS = ("flux_root", "radial_root")


def _scalar(item):
    """Return one scalar from a one-target array or scalar value."""
    array = np.asarray(item, dtype=object)
    return array.reshape(-1)[0] if array.ndim else array.item()


def _mp_scalar(item) -> mp.mpf:
    """Convert a NumPy or object scalar without losing its binary value."""
    item = _scalar(item)
    return mp.mpf(item.item() if isinstance(item, np.generic) else item)


def _pair_strings(pair, digits: int) -> dict[str, str]:
    high = _mp_scalar(pair[0])
    low = _mp_scalar(pair[1])
    return {
        "high": mp.nstr(high, digits),
        "low": mp.nstr(low, digits),
        "value": mp.nstr(high + low, digits),
    }


def _range_components(term) -> list[tuple[str, tuple]]:
    bulk, near, far = term
    return [
        *[(f"bulk_{index}", coefficient) for index, coefficient in enumerate(bulk)],
        ("near", near),
        ("far", far),
    ]


class _PrimitiveTrace:
    """Capture the paired contractions invoked by one edge endpoint."""

    def __init__(self, edge: _Edge, vertex: _Vertex):
        self.records: list[dict[str, Any]] = []
        self._plain = 0
        self._across = 0
        self._root = 0

        plain = vertex.plain_paired
        across = vertex.across_paired
        root = vertex.against_root_paired
        residual = edge._second_residual

        def capture_plain(term):
            label = _PLAIN_LABELS[self._plain]
            self._plain += 1
            result = plain(term)
            self.records.append(self._record(label, term, result))
            return result

        def capture_across(term, split):
            label = _ACROSS_LABELS[self._across]
            self._across += 1
            result = across(term, split)
            self.records.append(self._record(label, term, result))
            return result

        def capture_root(term):
            label = _ROOT_LABELS[self._root]
            self._root += 1
            result = root(term)
            self.records.append(self._record(label, term, result))
            return result

        def capture_residual(active_vertex, *, paired=False):
            result = residual(active_vertex, paired=paired)
            self.records.append({"label": "second_arsinh_residual", "output": result})
            return result

        vertex.plain_paired = capture_plain
        vertex.across_paired = capture_across
        vertex.against_root_paired = capture_root
        edge._second_residual = capture_residual

    @staticmethod
    def _record(label: str, term, output) -> dict[str, Any]:
        return {
            "input_range": dict(_range_components(term)),
            "label": label,
            "output": output,
        }


def _exact_norm(vertices: np.ndarray) -> mp.mpf:
    section = np.asarray(
        [[_mp_scalar(value) for value in row] for row in vertices], dtype=object
    )
    local = section - section[0]
    following = np.roll(local, -1, axis=0)
    signed_twice_area = sum(
        local[index, 0] * following[index, 1] - following[index, 0] * local[index, 1]
        for index in range(len(local))
    )
    return (
        -mp.sign(signed_twice_area)
        * mp.mpf(str(MU0))
        / (mp.mpf("0.5") * abs(signed_twice_area))
    )


def _evaluate(
    target_r,
    target_z,
    vertices: np.ndarray,
    norm,
    *,
    nodes: int,
    paired: bool,
    xp,
) -> dict[str, Any]:
    section = xp.asarray(vertices)
    rolled = np.roll(section, -1, axis=0)
    edges = np.column_stack([section[:, 0], section[:, 1], rolled[:, 0], rolled[:, 1]])
    radius = xp.asarray([target_r])
    height = xp.asarray([target_z])
    contour = (xp.zeros_like(radius), xp.zeros_like(radius))
    evaluated_edges = []

    for index, coordinates in enumerate(edges):
        endpoint_terms = {}
        endpoint_traces = {}
        for endpoint, corner in (
            ("lower", coordinates[:2]),
            ("upper", coordinates[2:]),
        ):
            part = _Edge(radius, height, coordinates, nodes, xp=xp)
            vertex = _Vertex(
                radius,
                height,
                corner[0],
                corner[1],
                nodes,
                residual=False,
                paired=paired,
                xp=xp,
            )
            trace = _PrimitiveTrace(part, vertex)
            endpoint_terms[endpoint] = part.terms(vertex, paired=True)[0]
            endpoint_traces[endpoint] = trace.records
        difference = subtract(endpoint_terms["lower"], endpoint_terms["upper"])
        contour = add(contour, difference)
        evaluated_edges.append(
            {
                "contour_partial": contour,
                "difference": difference,
                "endpoints": endpoint_terms,
                "index": index,
                "primitives": endpoint_traces,
            }
        )

    normalized = scale(contour, 0.5 * norm * target_r)
    return {"edges": evaluated_edges, "normalized": normalized, "contour": contour}


def _deviation(pair, oracle_pair, digits: int) -> dict[str, Any]:
    actual = _mp_scalar(pair[0]) + _mp_scalar(pair[1])
    expected = _mp_scalar(oracle_pair[0]) + _mp_scalar(oracle_pair[1])
    absolute = abs(actual - expected)
    relative = absolute / max(abs(expected), mp.mpf("1e-999"))
    return {
        "absolute_deviation": mp.nstr(absolute, digits),
        "exceeds_relative_gate": bool(relative > PAIR_GATE),
        "oracle_value": mp.nstr(expected, digits),
        "paired_fp64": _pair_strings(pair, digits),
        "relative_deviation": mp.nstr(relative, digits),
    }


def _compare_range(actual: dict[str, tuple], expected: dict[str, tuple], digits: int):
    rows = []
    for name in expected:
        row = {"coefficient": name, **_deviation(actual[name], expected[name], digits)}
        rows.append(row)
    return rows


def _first_breach(receipt: dict[str, Any]) -> dict[str, Any] | None:
    for edge in receipt["edges"]:
        for endpoint in ("lower", "upper"):
            for primitive in edge["primitives"][endpoint]:
                for coefficient in primitive.get("input_range", []):
                    if coefficient["exceeds_relative_gate"]:
                        return {
                            "assembly_stage": "range_function_terms",
                            "coefficient": coefficient["coefficient"],
                            "edge": edge["index"],
                            "endpoint": endpoint,
                            "primitive": primitive["label"],
                            "relative_deviation": coefficient["relative_deviation"],
                        }
                if primitive["output"]["exceeds_relative_gate"]:
                    return {
                        "assembly_stage": "primitive_contraction",
                        "edge": edge["index"],
                        "endpoint": endpoint,
                        "primitive": primitive["label"],
                        "relative_deviation": primitive["output"]["relative_deviation"],
                    }
        for stage in ("endpoint_difference", "contour_partial_sum"):
            if edge[stage]["exceeds_relative_gate"]:
                return {
                    "assembly_stage": stage,
                    "edge": edge["index"],
                    "relative_deviation": edge[stage]["relative_deviation"],
                }
    if receipt["inverse_area_normalization"]["exceeds_relative_gate"]:
        return {
            "assembly_stage": "inverse_area_normalization",
            "relative_deviation": receipt["inverse_area_normalization"][
                "relative_deviation"
            ],
        }
    return None


def _assemble_receipt(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    digits: int,
    source: int,
    target: int,
    norm: float,
    banked_oracle: str,
) -> dict[str, Any]:
    edges = []
    for actual_edge, expected_edge in zip(
        actual["edges"], expected["edges"], strict=True
    ):
        primitives = {}
        endpoints = {}
        for endpoint in ("lower", "upper"):
            compared = []
            for actual_primitive, expected_primitive in zip(
                actual_edge["primitives"][endpoint],
                expected_edge["primitives"][endpoint],
                strict=True,
            ):
                if actual_primitive["label"] != expected_primitive["label"]:
                    raise RuntimeError(
                        "primitive trace order changed between namespaces"
                    )
                row = {
                    "label": actual_primitive["label"],
                    "output": _deviation(
                        actual_primitive["output"],
                        expected_primitive["output"],
                        digits,
                    ),
                }
                if "input_range" in actual_primitive:
                    row["input_range"] = _compare_range(
                        actual_primitive["input_range"],
                        expected_primitive["input_range"],
                        digits,
                    )
                compared.append(row)
            primitives[endpoint] = compared
            endpoints[endpoint] = _deviation(
                actual_edge["endpoints"][endpoint],
                expected_edge["endpoints"][endpoint],
                digits,
            )
        edges.append(
            {
                "contour_partial_sum": _deviation(
                    actual_edge["contour_partial"],
                    expected_edge["contour_partial"],
                    digits,
                ),
                "endpoint_difference": _deviation(
                    actual_edge["difference"], expected_edge["difference"], digits
                ),
                "endpoints": endpoints,
                "index": actual_edge["index"],
                "primitives": primitives,
            }
        )

    gross = sum(
        abs(_mp_scalar(edge["difference"][0]) + _mp_scalar(edge["difference"][1]))
        for edge in expected["edges"]
    )
    contour = abs(
        _mp_scalar(expected["contour"][0]) + _mp_scalar(expected["contour"][1])
    )
    normalized = _deviation(actual["normalized"], expected["normalized"], digits)
    receipt = {
        "arithmetic": {
            "decimal_digits": digits,
            "oracle_path": (
                "the identical paired finite-section graph over mpmath objects"
            ),
            "relative_gate": mp.nstr(PAIR_GATE, 8),
        },
        "edges": edges,
        "far_target_scale": {
            "banked_column_max_absolute_deviation": "1.4944969930161633e-9",
            "banked_column_max_relative_deviation": "0.0019760683526076146",
            "exact_contour_condition_number": mp.nstr(gross / contour, digits),
            "gross_edge_difference_scale": mp.nstr(gross, digits),
        },
        "inverse_area_normalization": normalized,
        "oracle_cross_check": {
            "banked_1024_rung_value": banked_oracle,
            "localized_graph_value": normalized["oracle_value"],
            "relative_deviation": mp.nstr(
                abs(mp.mpf(normalized["oracle_value"]) - mp.mpf(banked_oracle))
                / abs(mp.mpf(banked_oracle)),
                digits,
            ),
        },
        "source_column": source,
        "target_index": target,
        "inverse_area_norm_fp64": repr(norm),
    }
    receipt["first_gate_breach"] = _first_breach(receipt)
    expected_normalized = _mp_scalar(expected["normalized"][0]) + _mp_scalar(
        expected["normalized"][1]
    )
    normalization = abs(expected_normalized / contour)
    receipt["route_estimates"] = {
        "analytic_closed_loop_roundoff_floor": mp.nstr(
            normalization * gross * mp.mpf(str(np.finfo(np.float64).eps)), digits
        ),
        "pairing_extension_recoverable_scale": normalized["absolute_deviation"],
        "verdict": (
            "recoverable by carrying paired values into the named first-breach stage"
            if receipt["first_gate_breach"]
            and receipt["first_gate_breach"]["assembly_stage"]
            in {"range_function_terms", "primitive_contraction"}
            else (
                "the paired graph remains accurate until closed-loop assembly; "
                "analytic reformulation is required"
            )
        ),
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, default=100)
    parser.add_argument("--nodes", type=int, default=1024)
    parser.add_argument("--source", type=int, default=184)
    parser.add_argument("--target", type=int, default=525)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.digits < 100:
        raise ValueError("digits must be at least 100")

    input_path = WORK / "coarse-input.npz"
    if input_path.exists():
        data = np.load(input_path)
        present = ~np.signbit(data["weight"][:, args.source])
        vertices = data["edge"][present, :2, args.source]
        target_r = data["target_r"][args.target]
        target_z = data["target_z"][args.target]
        norm = float(data["norm"][args.source])
    else:
        case = fixture.analytic_case()
        coarse = fixture.cached_machine(
            case,
            fixture.FIXTURE_REQUESTS["coarse"],
            wall_nodes=fixture.WALL_POINT_COUNT,
        )
        vertices = np.asarray(coarse.cell_polygons[args.source], dtype=np.float64)
        target_r = np.asarray(coarse.node)[args.target, 0]
        target_z = np.asarray(coarse.node)[args.target, 1]
        norm = float(pack_section(vertices)[2])
    actual = _evaluate(
        target_r,
        target_z,
        vertices,
        norm,
        nodes=128,
        paired=True,
        xp=np,
    )

    mp.mp.dps = args.digits
    xp = _install_precision_context()
    expected = _evaluate(
        _mp_scalar(target_r),
        _mp_scalar(target_z),
        vertices,
        _exact_norm(vertices),
        nodes=args.nodes,
        paired=True,
        xp=xp,
    )
    banked = json.loads(ORACLE.read_text(encoding="utf-8"))["oracle_values"][
        str(args.nodes)
    ][args.target]
    receipt = _assemble_receipt(
        actual,
        expected,
        digits=args.digits,
        source=args.source,
        target=args.target,
        norm=norm,
        banked_oracle=banked,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"LOCALIZED first={receipt['first_gate_breach']} "
        f"final_relative={receipt['inverse_area_normalization']['relative_deviation']} "
        f"oracle_cross_check={receipt['oracle_cross_check']['relative_deviation']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
