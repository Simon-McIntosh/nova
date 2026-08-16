"""Measure the field step caused by centroid-labelled source participation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nova.biot.polygonanalytic import (
    polygon_analytic_field_moments,
    polygon_analytic_flux_moments,
)
from nova.equilibrium.domain import PlasmaDomain, classify_domains
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.jax.config import configure_dtypes

ARTIFACT_DIRECTORY = Path("docs/figures/plasma-edge-current-representation")
FINDINGS_PATH = ARTIFACT_DIRECTORY / "domain_participation_jump_findings.json"
FIGURE_PATH = ARTIFACT_DIRECTORY / "domain_participation_jump.svg"

PITCH = 0.18
CENTRE = np.asarray([1.5, 0.0])
CELL_AREA = np.sqrt(3.0) / 2.0 * PITCH**2
SECOND_MEAN = 5.0 / 72.0 * PITCH**2
TARGET_CURRENT = -50.19
STALL_FRACTION = 0.517
PROFILE_SLOPE = 0.2
PROFILE_CURVATURE = 0.15
FLUX_RADIAL_SLOPE = 0.25 / PITCH
FLUX_VERTICAL_SLOPE = -0.17 / PITCH
EPSILON = 1.0e-4


def _hexagon(centre: np.ndarray) -> np.ndarray:
    angle = np.arange(6) * np.pi / 3.0 + np.pi / 6.0
    return centre + PITCH / np.sqrt(3.0) * np.c_[np.cos(angle), np.sin(angle)]


def _patch() -> tuple[np.ndarray, list[np.ndarray]]:
    neighbour_angle = np.arange(6) * np.pi / 3.0
    centres = np.vstack(
        [
            CENTRE,
            CENTRE + PITCH * np.c_[np.cos(neighbour_angle), np.sin(neighbour_angle)],
        ]
    )
    return centres, [_hexagon(centre) for centre in centres]


def _source() -> tuple[ForwardSource, float]:
    pressure_gradient = -TARGET_CURRENT / (2.0 * np.pi * CENTRE[0] * CELL_AREA)
    source = ForwardSource(
        core=DomainProfile(
            p_prime=lambda psi: (
                pressure_gradient
                * (
                    1.0
                    + PROFILE_SLOPE * (psi - 1.0)
                    + PROFILE_CURVATURE * (psi - 1.0) ** 2
                )
            ),
            ff_prime=lambda psi: np.zeros_like(psi),
        )
    )
    return source, pressure_gradient


def _fixed_blocks(cells: list[np.ndarray]) -> np.ndarray:
    target_r = np.asarray([1.22, 1.79, 2.55, 3.8])
    target_z = np.asarray([0.19, -0.27, 0.41, -0.63])
    blocks = []
    for cell, centre in zip(cells, _patch()[0], strict=True):
        flux = polygon_analytic_flux_moments(
            target_r, target_z, cell, expansion_point=centre
        )
        radial, vertical = polygon_analytic_field_moments(
            target_r, target_z, cell, expansion_point=centre
        )
        blocks.append([np.stack(flux), np.stack(radial), np.stack(vertical)])
    return np.transpose(np.asarray(blocks), (1, 0, 2, 3))


def _state(
    displacement: float,
    centres: np.ndarray,
    source: ForwardSource,
    pressure_gradient: float,
    blocks: np.ndarray,
) -> dict[str, np.ndarray]:
    psi_norm = np.full(len(centres), 0.82) - displacement
    psi_norm[0] = 1.0 - displacement
    masks = classify_domains(
        psi_norm,
        closed=psi_norm <= 1.0,
        connected=np.ones(len(centres), dtype=bool),
        inside_material=np.ones(len(centres), dtype=bool),
    )
    radius = centres[:, 0]
    area = np.full(len(centres), CELL_AREA)
    current = np.asarray(source.cell_current(radius, area, masks), dtype=float)
    support = np.asarray(source.declared_support(masks), dtype=float)

    profile_offset = psi_norm - 1.0
    profile_factor = (
        1.0 + PROFILE_SLOPE * profile_offset + PROFILE_CURVATURE * profile_offset**2
    )
    profile_derivative = PROFILE_SLOPE + 2.0 * PROFILE_CURVATURE * profile_offset
    radial_gradient = (
        -2.0
        * np.pi
        * pressure_gradient
        * (profile_factor + radius * profile_derivative * FLUX_RADIAL_SLOPE)
    )
    vertical_gradient = (
        -2.0
        * np.pi
        * radius
        * pressure_gradient
        * profile_derivative
        * FLUX_VERTICAL_SLOPE
    )
    moments = np.stack(
        [
            current,
            support * area * SECOND_MEAN * radial_gradient,
            support * area * SECOND_MEAN * vertical_gradient,
        ],
        axis=1,
    )
    fields = np.einsum("cm,qcmt->qt", moments, blocks)
    return {
        "psi_norm": psi_norm,
        "label": np.asarray(masks.label),
        "current": current,
        "moments": moments,
        "fields": fields,
    }


def _ratio(fine_minus: dict, fine_plus: dict, coarse_minus: dict, coarse_plus: dict):
    ratios = {}
    for name in ("current", "moments", "fields"):
        fine = np.abs(fine_plus[name] - fine_minus[name])
        coarse = np.abs(coarse_plus[name] - coarse_minus[name])
        ratios[name] = coarse / fine
    return ratios


def _write_figure(result: dict) -> None:
    positions = result["sweep"]["displacement"]
    current = result["sweep"]["target_cell_current_A"]
    x = np.asarray(positions)
    y = np.asarray(current)
    x_screen = 420.0 + 240.0 * x / max(abs(x))
    y_screen = 205.0 - 95.0 * (y - min(y)) / (max(y) - min(y))
    points = " ".join(
        f"{a:.1f},{b:.1f}" for a, b in zip(x_screen, y_screen, strict=True)
    )
    cell = _hexagon(CENTRE)
    cell_screen = " ".join(
        f"{55 + 420 * (r - 1.35):.1f},{150 - 420 * z:.1f}" for r, z in cell
    )
    boundary_x = float(result["target_cell"]["polyline_radius"])
    boundary_screen = 55 + 420 * (boundary_x - 1.35)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 260" '
        'role="img" aria-label="Non-crossing cell and current step">\n'
        "  <style>text{font:12px sans-serif;fill:#222} "
        ".axis{stroke:#777;stroke-width:1} "
        ".data{fill:none;stroke:#b42318;stroke-width:2}</style>\n"
        f'  <polygon points="{cell_screen}" fill="#dbeafe" stroke="#2563eb" '
        'stroke-width="2"/>\n'
        '  <circle cx="118" cy="150" r="3" fill="#111"/>\n'
        f'  <line x1="{boundary_screen:.1f}" y1="72" '
        f'x2="{boundary_screen:.1f}" y2="228" stroke="#0f766e" '
        'stroke-width="2"/>\n'
        '  <text x="32" y="28">target label flips; polyline remains outside</text>\n'
        '  <text x="360" y="28">whole-cell current selected by label</text>\n'
        '  <line class="axis" x1="360" y1="205" x2="680" y2="205"/>\n'
        '  <line class="axis" x1="420" y1="70" x2="420" y2="220"/>\n'
        f'  <polyline class="data" points="{points}"/>\n'
        '  <text x="490" y="246">boundary displacement</text>\n'
        '  <text x="432" y="88">0 A</text>\n'
        f'  <text x="432" y="201">{min(y):.2f} A</text>\n'
        "</svg>"
    )
    FIGURE_PATH.write_text(svg)


def measure_participation_jump(*, write_artifacts: bool = True) -> dict:
    """Return the discontinuity receipt for a non-crossing labelled cell."""
    configure_dtypes()
    centres, cells = _patch()
    source, pressure_gradient = _source()
    blocks = _fixed_blocks(cells)
    positions = np.asarray([-4, -2, -1, 1, 2, 4], dtype=float) * EPSILON
    states = [
        _state(value, centres, source, pressure_gradient, blocks) for value in positions
    ]
    ratios = _ratio(states[2], states[3], states[1], states[4])
    all_ratios = np.concatenate([ratios[name].ravel() for name in ratios])
    labels = [PlasmaDomain(int(state["label"][0])).name for state in states]
    polyline_radius = CENTRE[0] + 0.7 * PITCH
    polygon_gap = polyline_radius - max(cells[0][:, 0])
    jump = abs(states[3]["current"][0] - states[2]["current"][0])
    result = {
        "configuration": {
            "pitch_m": PITCH,
            "epsilon": EPSILON,
            "cell_count": len(centres),
            "field_targets": 4,
        },
        "target_cell": {
            "index": 0,
            "polyline_intersects": False,
            "polyline_radius": polyline_radius,
            "minimum_polyline_gap_m": polygon_gap,
            "labels_across_fine_pair": labels[2:4],
        },
        "sweep": {
            "displacement": positions.tolist(),
            "centroid_psi_norm": [float(state["psi_norm"][0]) for state in states],
            "target_cell_label": labels,
            "target_cell_current_A": [float(state["current"][0]) for state in states],
            "target_moment_vector": [state["moments"][0].tolist() for state in states],
            "composed_fields": [state["fields"].tolist() for state in states],
        },
        "epsilon_doubling_ratios": {
            "cell_current": ratios["current"].tolist(),
            "moment_vector": ratios["moments"].tolist(),
            "composed_fields": ratios["fields"].tolist(),
            "all_entries": all_ratios.tolist(),
            "minimum": float(all_ratios.min()),
            "maximum": float(all_ratios.max()),
        },
        "measured_jump": {
            "target_cell_current_A": float(jump),
            "whole_cell_reference_A": abs(TARGET_CURRENT),
            "stall_amplitude_fraction": STALL_FRACTION,
            "implied_reference_stall_amplitude_A": float(jump / STALL_FRACTION),
            "field_by_quantity": np.abs(
                states[3]["fields"] - states[2]["fields"]
            ).tolist(),
        },
        "verdict": "centroid-labelled participation is discontinuous",
    }
    if write_artifacts:
        ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        FINDINGS_PATH.write_text(json.dumps(result, indent=2) + "\n")
        _write_figure(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FINDINGS_PATH)
    args = parser.parse_args()
    result = measure_participation_jump(write_artifacts=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.output == FINDINGS_PATH:
        _write_figure(result)
    print(json.dumps(result["measured_jump"], indent=2))


if __name__ == "__main__":
    main()
