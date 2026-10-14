"""Evaluate one radial carrier entry with high precision quadrature."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import xarray as xr

from nova.biot.greens import section_centroid
from nova.biot.polygonanalytic import (
    _flux_area_rule,
    _section_centroid,
    pack_section,
    polygon_analytic_flux_moments,
)


def _number(value) -> mp.mpf:
    return mp.mpf(float(value))


def reference(target: np.ndarray, section: np.ndarray) -> tuple[float, bool]:
    mp.mp.dps = 50
    radius, height = map(_number, target)
    centre = np.asarray(_section_centroid(section))
    expansion = tuple(map(_number, section_centroid(section)))
    edge, weight, norm = pack_section(section)
    present = ~np.signbit(weight)
    extent = max(
        (float(point[0] - centre[0]) ** 2 + float(point[1] - centre[1]) ** 2)
        for point in section
    )
    near = float(np.sum((target - centre) ** 2)) < 16.0 * extent
    origin = (radius, height) if near else tuple(map(_number, centre))
    u, wu, v, wv = _flux_area_rule(near)
    result = mp.mpf("0")
    for index in range(len(edge)):
        if not present[index]:
            continue
        ar = _number(edge[index, 0]) - origin[0]
        az = _number(edge[index, 1]) - origin[1]
        br = _number(edge[index, 2]) - origin[0]
        bz = _number(edge[index, 3]) - origin[1]
        determinant = ar * bz - az * br
        if determinant == 0:
            continue
        delta_r, delta_z = br - ar, bz - az
        length_squared = delta_r**2 + delta_z**2
        peak = -(ar * delta_r + az * delta_z) / length_squared
        width = max(abs(determinant) / length_squared, mp.mpf("1e-15"))
        lower = mp.asinh(-peak / width)
        span_angle = mp.asinh((1 - peak) / width) - lower
        for radial_node, radial_weight in zip(u, wu, strict=True):
            radial_node, radial_weight = _number(radial_node), _number(radial_weight)
            for angular_node, angular_weight in zip(v, wv, strict=True):
                angular_node, angular_weight = (
                    _number(angular_node),
                    _number(angular_weight),
                )
                if near:
                    angle = lower + span_angle * angular_node
                    edge_coordinate = peak + width * mp.sinh(angle)
                    angular_weight *= span_angle * width * mp.cosh(angle)
                else:
                    edge_coordinate = angular_node
                dr = radial_node * ((1 - edge_coordinate) * ar + edge_coordinate * br)
                dz = radial_node * ((1 - edge_coordinate) * az + edge_coordinate * bz)
                source_r = origin[0] + dr
                source_z = origin[1] + dz
                gap = (radius - source_r) ** 2 + (height - source_z) ** 2
                span = (radius + source_r) ** 2 + (height - source_z) ** 2
                complement = gap / span
                elliptic_parameter = 1 - complement
                flux = (1 + complement) * mp.ellipk(elliptic_parameter) / 2 - mp.ellipe(
                    elliptic_parameter
                )
                result += (
                    -_number(norm)
                    * determinant
                    * radial_weight
                    * angular_weight
                    * mp.sqrt(span)
                    * flux
                    * (source_r - expansion[0])
                )
        print(f"REFERENCE_EDGE={index + 1}/{len(edge)}", flush=True)
    return float(result), near


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("group")
    parser.add_argument("inspection", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--top-index", type=int, default=0)
    args = parser.parse_args()
    inspection = json.loads(args.inspection.read_text())
    pair = inspection["top"][args.top_index]
    with xr.open_zarr(args.cache, group=args.group, consolidated=False) as data:
        all_targets = np.asarray(data["sample_coordinates"].values)
        target = all_targets[pair["target"]]
        offsets = np.asarray(data["cell_polygon_offsets"].values)
        section = np.asarray(
            data["cell_polygon_vertices"][
                offsets[pair["cell"]] : offsets[pair["cell"] + 1]
            ].values
        )
    high_precision, near = reference(target, section)
    host_recomputed = polygon_analytic_flux_moments(
        np.asarray([target[0]]),
        np.asarray([target[1]]),
        section,
        expansion_point=section_centroid(section),
    )[1][0]
    host_batch = polygon_analytic_flux_moments(
        all_targets[:, 0],
        all_targets[:, 1],
        section,
        expansion_point=section_centroid(section),
    )[1][pair["target"]]
    print(
        f"HOST_REPLAY stored={pair['host']:.17e} full={host_batch:.17e} "
        f"scalar={host_recomputed:.17e} reference={high_precision:.17e} "
        f"centre={section_centroid(section)}",
        flush=True,
    )
    replay_difference = abs(host_batch - pair["host"])
    replay_limit = 1e-13 * inspection["global_scale"]
    if replay_difference > replay_limit:
        raise ValueError("stored host entry does not match its full batch")
    report = {
        **pair,
        "top_index": args.top_index,
        "high_precision": high_precision,
        "near": near,
        "host_recomputed": float(host_recomputed),
        "host_recomputed_full_batch": float(host_batch),
        "host_replay_absolute": float(replay_difference),
        "host_replay_limit": float(replay_limit),
        "host_reference_absolute": abs(pair["host"] - high_precision),
        "device_reference_absolute": abs(pair["device"] - high_precision),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
