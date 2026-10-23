"""Compare a wall-adjacent contour carrier pair with its analytic flux field."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.collections import LineCollection
from scipy.optimize import root
from scipy.spatial import Delaunay, cKDTree

from nova.jax.config import configure_dtypes

configure_dtypes()
matplotlib.rcParams["svg.hashsalt"] = "sol-wall-pair"

from benchmarks.contour_tree_brute_force import (  # noqa: E402
    PART_ROOT,
    certificate_rung_fixtures,
)
from benchmarks import solovev_certificate as certificate  # noqa: E402
from nova.media.poloidal import draw_wall  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FIGURE = (
    ROOT
    / "docs/figures/contour-tree-topology-authority/sol-wall-pair/sol-wall-pair.svg"
)
FOCUS = "diverted-single-null-340-cells"


def fixture_payload(name: str) -> dict:
    case, count_text, _ = name.rsplit("-", 2)
    count = int(count_text)
    rung = {340: 300, 550: 500, 1074: 1000}[count]
    path = PART_ROOT / f"{case}-production-route-cells-{rung}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def exact_case(case: str):
    return certificate._case(case)[2]


def smooth_field(case: str, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    exact = exact_case(case)
    values = certificate._exact_state(case, exact, points)
    gradient, _ = certificate._exact_derivatives(case, exact, points)
    return np.asarray(values), np.asarray(gradient)


def components(mesh):
    coords = np.asarray(mesh.vertex_rz)
    values = np.asarray(mesh.vertex_psi)
    valid = np.asarray(mesh.vertex_valid)
    walls = np.asarray(mesh.vertex_is_wall)
    edges = np.asarray(mesh.edges)[np.asarray(mesh.edge_valid)]
    triangles = np.asarray(mesh.triangles)[np.asarray(mesh.triangle_valid)]
    links = [[] for _ in range(len(coords))]
    for a, b in edges:
        if valid[a] and valid[b]:
            links[a].append(int(b))
            links[b].append(int(a))
    return coords, values, valid, walls, edges, triangles, links


def stationary_near(case: str, point: np.ndarray, pitch: float, polygon) -> bool:
    def gradient(position):
        return smooth_field(case, np.asarray(position)[None, :])[1][0]

    starts = [
        point,
        *(point + pitch * offset for offset in np.eye(2)),
        *(point - pitch * offset for offset in np.eye(2)),
    ]
    for start in starts:
        result = root(gradient, start, method="hybr", options={"xtol": 1e-10})
        if (
            np.linalg.norm(gradient(result.x)) < 1e-8
            and np.linalg.norm(result.x - point) <= 2 * pitch
            and polygon.covers(shapely.Point(result.x))
        ):
            return True
    return False


def source_triangle(
    point: np.ndarray, centres: np.ndarray
) -> tuple[list[int], np.ndarray]:
    triangulation = Delaunay(centres)
    simplex = int(triangulation.find_simplex(point))
    if simplex < 0:
        simplex = int(
            cKDTree(centres[triangulation.simplices].mean(axis=1)).query(point)[1]
        )
    cells = triangulation.simplices[simplex]
    corners = centres[cells]
    bary = np.linalg.solve(
        np.column_stack((corners[1] - corners[0], corners[2] - corners[0])),
        point - corners[0],
    )
    return cells.tolist(), np.array((1 - bary.sum(), bary[0], bary[1]))


def audit_fixture(fixture) -> dict:
    payload = fixture_payload(fixture.name)
    coords, values, valid, walls, _, _, links = components(fixture.mesh)
    wall = np.asarray(payload["render_data"]["wall_units_rz_m"][0])
    polygon = shapely.Polygon(wall)
    pitch = float(payload["characteristic_pitch_m"])
    candidates = []
    all_extrema = 0
    smooth_all = np.full(len(coords), np.nan)
    smooth_all[valid] = smooth_field(payload["case"], coords[valid])[0]
    for index in np.flatnonzero(valid & ~walls):
        neighbours = links[index]
        if not neighbours:
            continue
        low = all(values[index] < values[neighbour] for neighbour in neighbours)
        high = all(values[index] > values[neighbour] for neighbour in neighbours)
        if not (low or high):
            continue
        all_extrema += 1
        if not stationary_near(payload["case"], coords[index], pitch, polygon):
            wall_neighbours = [j for j in neighbours if walls[j]]
            wall_resolves = any(
                smooth_all[j] < values[index] if low else smooth_all[j] > values[index]
                for j in wall_neighbours
            )
            candidates.append(
                {
                    "vertex": int(index),
                    "kind": "minimum" if low else "maximum",
                    "wall_neighbours": wall_neighbours,
                    "wall_interpolation_creates_extremum": bool(wall_resolves),
                }
            )
    return {
        "fixture": fixture.name,
        "pitch_m": pitch,
        "interior_carrier_extrema": all_extrema,
        "nonstationary_extrema": candidates,
    }


def focus_measure(fixture) -> tuple[dict, dict]:
    payload = fixture_payload(fixture.name)
    coords, values, valid, walls, edges, triangles, links = components(fixture.mesh)
    case = payload["case"]
    wall = np.asarray(payload["render_data"]["wall_units_rz_m"][0])
    polygon = shapely.Polygon(wall)
    pitch = float(payload["characteristic_pitch_m"])
    centre = coords[10]
    neighbour_ids = sorted(links[10])
    if 447 not in neighbour_ids:
        raise AssertionError(
            "recorded wall vertex 447 is no longer linked to vertex 10"
        )
    if not (valid[10] and not walls[10] and walls[447]):
        raise AssertionError("recorded interior/wall identity has changed")
    if not all(values[10] < values[j] for j in neighbour_ids):
        raise AssertionError("recorded carrier minimum no longer reproduces")
    # An independent receipt value establishes that this is the intended carrier.
    if abs(values[10] - (-0.0040097101310506145)) > 1e-10:
        raise AssertionError("carrier flux does not match the published receipt")
    ring = [10, *neighbour_ids]
    smooth, gradient = smooth_field(case, coords[ring])
    centre_coordinates = np.asarray(
        payload["render_data"]["coordinates_rz_m"][: payload["realised_cells"]]
    )
    rows = []
    for index, exact, derivative in zip(ring, smooth, gradient):
        touching = [n for n, triangle in enumerate(triangles) if index in triangle]
        if walls[index]:
            cells, weights = source_triangle(coords[index], centre_coordinates)
            origin = {
                "kind": "wall interpolation",
                "source_cells": cells,
                "source_cell_valid": valid[cells].tolist(),
                "source_cell_flux_wb": values[cells].tolist(),
                "barycentric_weights": weights.tolist(),
                "reconstructed_wb": float(np.dot(weights, values[cells])),
                "carrier_triangles": touching,
            }
        else:
            origin = {
                "kind": "cell sample",
                "cell": index,
                "carrier_triangles": touching,
            }
        rows.append(
            {
                "vertex": index,
                "rz_m": coords[index].tolist(),
                "carrier_wb": float(values[index]),
                "smooth_wb": float(exact),
                "carrier_minus_smooth_wb": float(values[index] - exact),
                "gradient_wb_per_m": float(np.linalg.norm(derivative)),
                "origin": origin,
            }
        )
    axes = (
        np.linspace(centre[0] - 2 * pitch, centre[0] + 2 * pitch, 121),
        np.linspace(centre[1] - 2 * pitch, centre[1] + 2 * pitch, 121),
    )
    rr, zz = np.meshgrid(*axes)
    samples = np.column_stack((rr.ravel(), zz.ravel()))
    inside = np.asarray(
        shapely.covers(polygon, shapely.points(samples[:, 0], samples[:, 1]))
    )
    inside_vessel = inside.copy()
    inside &= np.linalg.norm(samples - centre, axis=1) <= 2 * pitch
    sampled_flux, sampled_gradient = smooth_field(case, samples)
    strength = np.linalg.norm(sampled_gradient, axis=1)
    if not inside.any() or not np.isfinite(strength[inside]).all():
        raise AssertionError("smooth-gradient window is missing or nonfinite")
    minimum_index = np.flatnonzero(inside)[np.argmin(strength[inside])]
    typical_points = coords[valid & ~walls]
    typical_flux, typical_gradient = smooth_field(case, typical_points)
    # The SOL lies below the X-point flux for this sigma=+1 certificate.
    exact = exact_case(case)
    x_flux = float(exact.flux(exact.x_point[None, :])[0])
    axis_gradient = float(
        np.linalg.norm(exact.gradient(exact.magnetic_axis[None, :])[0])
    )
    if axis_gradient > 1e-8 or not stationary_near(
        case, exact.magnetic_axis, pitch, polygon
    ):
        raise AssertionError(
            "analytic axis positive control did not detect a stationary point"
        )
    sol = typical_flux < x_flux
    if sol.sum() < 10:
        raise AssertionError("SOL reference population unexpectedly empty")
    typical = float(np.median(np.linalg.norm(typical_gradient[sol], axis=1)))
    direction = (coords[447] - centre) / np.linalg.norm(coords[447] - centre)
    line_points = centre[None, :] + np.linspace(0, 1, 201)[:, None] * (
        coords[447] - centre
    )
    line_flux, line_gradient = smooth_field(case, line_points)
    directional_derivative = line_gradient @ direction
    measure = {
        "fixture": fixture.name,
        "pitch_m": pitch,
        "reproduction": "vertex 10 is below all links, including wall vertex 447",
        "neighbours": rows,
        "smooth_window_min_gradient_wb_per_m": float(strength[minimum_index]),
        "smooth_window_min_gradient_rz_m": samples[minimum_index].tolist(),
        "typical_sol_gradient_wb_per_m": typical,
        "smooth_window_min_gradient_over_typical": float(
            strength[minimum_index] / typical
        ),
        "smooth_line_flux_change_wb": float(line_flux[-1] - line_flux[0]),
        "carrier_line_flux_change_wb": float(values[447] - values[10]),
        "smooth_line_directional_derivative_range_wb_per_m": [
            float(np.min(directional_derivative)),
            float(np.max(directional_derivative)),
        ],
        "smooth_stationary_within_two_pitches": stationary_near(
            case, centre, pitch, polygon
        ),
        "smooth_window_flux_range_wb": [
            float(np.min(sampled_flux[inside])),
            float(np.max(sampled_flux[inside])),
        ],
        "smooth_value_at_vertex_10_wb": float(smooth[0]),
        "x_point_flux_wb": x_flux,
        "analytic_axis_gradient_wb_per_m": axis_gradient,
    }
    state = {
        "coords": coords,
        "values": values,
        "valid": valid,
        "walls": walls,
        "edges": edges,
        "triangles": triangles,
        "wall": wall,
        "rows": rows,
        "rr": rr,
        "zz": zz,
        "sampled_flux": sampled_flux.reshape(rr.shape),
        "inside": inside_vessel.reshape(rr.shape),
        "case": case,
        "pitch": pitch,
    }
    return measure, state


def draw_figure(state: dict) -> None:
    coords = state["coords"]
    rows = state["rows"]
    pitch = state["pitch"]
    centre = coords[10]
    wall_point = coords[447]
    fig, (spatial, profile) = plt.subplots(
        1, 2, figsize=(14, 7.2), gridspec_kw={"width_ratios": [1.55, 1]}
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.94, bottom=0.14, wspace=0.14)
    local = np.all(np.abs(coords - centre) < 2.6 * pitch, axis=1)
    segments = state["coords"][state["edges"]]
    segments = segments[np.all(local[state["edges"]], axis=1)]
    spatial.add_collection(
        LineCollection(segments, colors="#929292", linewidths=0.65, zorder=1)
    )
    field = np.ma.masked_where(~state["inside"], state["sampled_flux"])
    lo, hi = float(field.min()), float(field.max())
    levels = np.linspace(lo, hi, 9)[1:-1]
    spatial.contour(
        state["rr"],
        state["zz"],
        field,
        levels=levels,
        colors="#547b80",
        linewidths=1.2,
        zorder=2,
    )
    draw_wall(
        spatial,
        radius=state["wall"][:, 0],
        height=state["wall"][:, 1],
        color="#292929",
        linewidth=2.0,
        zorder=3,
    )
    offsets = {10: (6, 11), 31: (-15, 6), 32: (5, 5), 447: (6, 2), 448: (-31, -15)}
    for row in rows:
        index = row["vertex"]
        color = "#b6543d" if index == 10 else ("#3c6381" if index == 447 else "#616161")
        spatial.scatter(
            *coords[index], s=65 if index in (10, 447) else 28, color=color, zorder=5
        )
        spatial.annotate(
            str(index),
            coords[index],
            xytext=offsets[index],
            textcoords="offset points",
            fontsize=11,
            color=color,
            zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.75},
        )
    spatial.set_xlim(centre[0] - 2.2 * pitch, centre[0] + 2.2 * pitch)
    spatial.set_ylim(centre[1] - 2.2 * pitch, centre[1] + 2.2 * pitch)
    spatial.set_aspect("equal")
    spatial.set_axis_off()
    # The numbered labels form a separate key so crowded mesh vertices remain legible.
    listed = sorted(
        rows, key=lambda row: (row["vertex"] not in (10, 447), row["vertex"])
    )
    for rank, row in enumerate(listed):
        difference = row["carrier_minus_smooth_wb"] * 1e6
        if abs(difference) < 0.05:
            difference = 0.0
        spatial.text(
            0.01,
            0.98 - rank * 0.047,
            f"{row['vertex']}: {difference:+.1f} µWb",
            transform=spatial.transAxes,
            va="top",
            fontsize=10,
            color="#b6543d"
            if row["vertex"] == 10
            else ("#3c6381" if row["vertex"] == 447 else "#454545"),
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.88},
        )
    positions = centre[None, :] + np.linspace(0, 1, 200)[:, None] * (
        wall_point - centre
    )
    _, gradient = smooth_field(state["case"], positions)
    distance = np.linalg.norm(positions - centre, axis=1) * 100
    profile.plot(
        distance,
        np.linalg.norm(gradient, axis=1) * 1e3,
        color="#3c6381",
        linewidth=3,
        zorder=2,
    )
    profile.scatter(
        [0, distance[-1]],
        np.linalg.norm(gradient[[0, -1]], axis=1) * 1e3,
        color=["#b6543d", "#3c6381"],
        s=64,
        zorder=3,
    )
    profile.text(0, np.linalg.norm(gradient[0]) * 1e3, "  10", va="bottom", fontsize=12)
    profile.text(
        distance[-1],
        np.linalg.norm(gradient[-1]) * 1e3,
        "447  ",
        ha="right",
        va="bottom",
        fontsize=12,
    )
    profile.set_xlabel("distance from vertex 10 [cm]", fontsize=16)
    profile.set_ylabel("|∇ψ| [mWb/m]", fontsize=16)
    profile.tick_params(labelsize=14, width=1.2)
    profile.spines[["top", "right"]].set_visible(False)
    profile.spines[["bottom", "left"]].set_linewidth(1.2)
    profile.grid(False)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, format="svg", metadata={"Date": None})
    plt.close(fig)
    print(f"figure={FIGURE}")
    print(f"contour_levels_wb={levels.tolist()}")


def main() -> None:
    fixtures = certificate_rung_fixtures()
    selected = [fixture for fixture in fixtures if fixture.name == FOCUS]
    if len(selected) != 1:
        raise AssertionError(f"expected one focus fixture, found {len(selected)}")
    measure, state = focus_measure(selected[0])
    print("focus=" + json.dumps(measure, sort_keys=True))
    draw_figure(state)
    for fixture in fixtures:
        print("census=" + json.dumps(audit_fixture(fixture), sort_keys=True))


if __name__ == "__main__":
    main()
