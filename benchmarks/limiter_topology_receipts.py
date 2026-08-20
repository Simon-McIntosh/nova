"""Bank limited-boundary and topology-transition receipts.

The synthetic round trip uses an analytic single-axis equilibrium whose
limiting contact is prescribed before the topology read.  The MAST agreement
bank uses the in-house machine-description limiter and analytic limited flux
maps; it deliberately consumes no EFIT geometry or topology labels.  Those
frames are a classifier fixture, not evidence of experimental reconstruction
fidelity.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nova.biot.greens import hybrid_greens
from nova.biot.null import Null1D, Null2D
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.topology import (
    BoundaryMode,
    Topology,
    TopologyState,
    topology_solve_receipt,
)
from nova.equilibrium.wall_mask import inside_polygon
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes

WALL_CONTACT_TOLERANCE_GRID_CELLS = 1.0
MAST_FRAME_PARAMETERS = (
    (0.72, -0.18, 0.42),
    (0.78, 0.00, 0.40),
    (0.86, 0.20, 0.44),
    (1.02, -0.16, 0.46),
    (1.16, 0.12, 0.48),
    (1.28, 0.00, 0.50),
)


@dataclass(frozen=True)
class _TopologyGrid:
    """Topology locator plus the host geometry used to construct it."""

    topology: Topology
    coordinate: np.ndarray
    inside: np.ndarray
    wall: np.ndarray
    radial_spacing_m: float
    vertical_spacing_m: float


def _topology_grid(
    wall: np.ndarray,
    *,
    radial_points: int,
    vertical_points: int,
    margin_m: float,
) -> _TopologyGrid:
    """Build one structured topology grid around a closed wall polygon."""

    wall = np.asarray(wall, dtype=np.float64)
    radial = np.linspace(
        float(wall[:, 0].min() - margin_m),
        float(wall[:, 0].max() + margin_m),
        radial_points,
    )
    vertical = np.linspace(
        float(wall[:, 1].min() - margin_m),
        float(wall[:, 1].max() + margin_m),
        vertical_points,
    )
    radial_mesh, vertical_mesh = np.meshgrid(radial, vertical, indexing="ij")
    coordinate = np.column_stack([radial_mesh.ravel(), vertical_mesh.ravel()])
    inside = np.asarray(
        inside_polygon(coordinate[:, 0], coordinate[:, 1], wall[:, 0], wall[:, 1]),
        dtype=bool,
    )
    topology = Topology(
        Null2D.from_coordinates(
            coordinate, hex_stencil((radial_points, vertical_points)), maxsize=5
        ),
        Null1D(jnp.asarray(wall)),
    )
    return _TopologyGrid(
        topology=topology,
        coordinate=coordinate,
        inside=inside,
        wall=wall,
        radial_spacing_m=float(radial[1] - radial[0]),
        vertical_spacing_m=float(vertical[1] - vertical[0]),
    )


def _gaussian_flux(coordinate: np.ndarray, axis: tuple[float, float], width: float):
    """Return one positive single-axis flux map on arbitrary coordinates."""

    radius = coordinate[:, 0] - axis[0]
    height = coordinate[:, 1] - axis[1]
    return np.exp(-(radius**2 + height**2) / width**2)


def _read_limited_state(
    grid: _TopologyGrid, axis: tuple[float, float], width: float
) -> TopologyState:
    """Read one analytic wall-limited equilibrium through the JAX locator."""

    grid_flux = _gaussian_flux(grid.coordinate, axis, width)
    wall_flux = _gaussian_flux(grid.wall, axis, width)
    flux = jnp.asarray(np.concatenate([grid_flux, wall_flux]))
    return grid.topology.read(flux, 1.0, jnp.asarray(grid.inside))[1]


def _synthetic_limited_round_trip() -> tuple[TopologyState, dict[str, Any], np.ndarray]:
    """Read a prescribed circular-wall contact and score it in grid cells."""

    angle = np.linspace(0.0, 2.0 * np.pi, 144, endpoint=False)
    wall = np.column_stack([1.0 + 0.35 * np.cos(angle), 0.35 * np.sin(angle)])
    grid = _topology_grid(wall, radial_points=49, vertical_points=49, margin_m=0.04)
    prescribed_contact = np.array([1.35, 0.0])
    state = _read_limited_state(grid, (1.06, 0.0), 0.32)
    receipt = topology_solve_receipt((state,), solver_succeeded=True)
    observed_contact = np.asarray(receipt.wall_contact_point_m, dtype=float)
    distance_m = float(np.linalg.norm(observed_contact - prescribed_contact))
    cell_size_m = max(grid.radial_spacing_m, grid.vertical_spacing_m)
    distance_cells = distance_m / cell_size_m
    score = {
        "prescribed_wall_contact_point_m": prescribed_contact.tolist(),
        "recovered_wall_contact_point_m": observed_contact.tolist(),
        "contact_distance_m": distance_m,
        "grid_cell_size_m": cell_size_m,
        "contact_distance_grid_cells": distance_cells,
        "tolerance_grid_cells": WALL_CONTACT_TOLERANCE_GRID_CELLS,
        "passes": bool(distance_cells <= WALL_CONTACT_TOLERANCE_GRID_CELLS),
    }
    return state, score, wall


def _diverted_state() -> TopologyState:
    """Return the read of one analytic double-null flux map."""

    lattice = FluxLattice(np.linspace(0.55, 1.45, 45), np.linspace(-0.75, 0.75, 71))
    coordinate = np.asarray(lattice.coordinate)
    angle = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    wall = np.column_stack([1.0 + 0.42 * np.cos(angle), 0.62 * np.sin(angle)])
    grid = Topology(
        Null2D.from_coordinates(coordinate, hex_stencil(lattice.shape), maxsize=5),
        Null1D(jnp.asarray(wall)),
    )
    ring = np.array([[1.0, 0.0], [1.0, -0.62], [1.0, 0.62]])
    current = np.array([1.0e6, 5.0e5, 4.0e5])

    def ring_flux(target: np.ndarray) -> np.ndarray:
        columns = np.stack(
            [
                hybrid_greens(target[:, 0], target[:, 1], radius, height, 0.06, 0.06)[0]
                for radius, height in ring
            ],
            axis=1,
        )
        return columns @ current

    inside = ((coordinate[:, 0] - 1.0) / 0.42) ** 2 + (
        coordinate[:, 1] / 0.62
    ) ** 2 <= 1.0
    flux = jnp.asarray(np.concatenate([ring_flux(coordinate), ring_flux(wall)]))
    return grid.read(flux, 1.0, jnp.asarray(inside))[1]


def _mast_limited_frames() -> tuple[list[TopologyState], np.ndarray]:
    """Read the analytic limited-frame bank on the in-house MAST wall."""

    catalog_path = Path(__file__).parents[1] / "nova" / "catalog" / "mast_geometry.json"
    catalog = json.loads(catalog_path.read_text())
    configuration = next(iter(catalog["configurations"].values()))
    wall = np.asarray(configuration["geometry"]["limiter"], dtype=np.float64)
    grid = _topology_grid(wall, radial_points=53, vertical_points=69, margin_m=0.03)
    states = [
        _read_limited_state(grid, (radius, height), width)
        for radius, height, width in MAST_FRAME_PARAMETERS
    ]
    return states, wall


def build_receipt() -> dict[str, Any]:
    """Build the strict-JSON limited-mode evidence receipt."""

    configure_dtypes()
    limited_state, round_trip, synthetic_wall = _synthetic_limited_round_trip()
    diverted_state = _diverted_state()
    transition = topology_solve_receipt(
        (limited_state, diverted_state, limited_state, diverted_state),
        solver_succeeded=True,
    )
    mast_states, mast_wall = _mast_limited_frames()
    mast_receipts = [
        topology_solve_receipt((state,), solver_succeeded=True) for state in mast_states
    ]
    agreement = [
        receipt.topology_class is BoundaryMode.LIMITED for receipt in mast_receipts
    ]
    solves = [
        {
            "solve_id": "synthetic-limited-round-trip",
            **topology_solve_receipt((limited_state,), solver_succeeded=True).as_dict(),
        },
        {
            "solve_id": "synthetic-transition-traversal",
            **transition.as_dict(),
        },
        *[
            {
                "solve_id": f"mast-limited-frame-{index}",
                "prescribed_topology_class": "limited",
                **receipt.as_dict(),
            }
            for index, receipt in enumerate(mast_receipts)
        ],
    ]
    return {
        "schema": "nova-limiter-topology-receipt",
        "tolerances_fixed_before_scoring": {
            "wall_contact_distance_grid_cells": WALL_CONTACT_TOLERANCE_GRID_CELLS
        },
        "synthetic_limited_round_trip": round_trip,
        "solves": solves,
        "summary": {
            "solve_count": len(solves),
            "topology_class_by_solve": {
                row["solve_id"]: row["topology_class"] for row in solves
            },
            "mid_solve_transition_count": transition.transition_count,
            "mid_solve_transitions_traversed_without_solver_failure": (
                transition.transitions_without_solver_failure
            ),
            "mast_limited_frame_count": len(mast_receipts),
            "mast_limited_classifier_agreement_count": int(sum(agreement)),
            "mast_limited_classifier_agreement_fraction": float(np.mean(agreement)),
        },
        "mast_frame_provenance": {
            "machine_geometry": "nova/catalog/mast_geometry.json",
            "flux_source": "analytic in-house limited-equilibrium fixture bank",
            "reference_topology": "prescribed limited construction",
            "efit_inputs_used": False,
            "qualification": (
                "classifier fixture only; not experimental reconstruction fidelity"
            ),
        },
        "_figure_geometry": {
            "synthetic_wall": synthetic_wall.tolist(),
            "mast_wall": mast_wall.tolist(),
        },
    }


def render_figure(receipt: dict[str, Any], path: Path) -> None:
    """Render the prescribed/recovered contact and the MAST contact bank."""

    geometry = receipt.pop("_figure_geometry")
    synthetic_wall = np.asarray(geometry["synthetic_wall"])
    mast_wall = np.asarray(geometry["mast_wall"])
    round_trip = receipt["synthetic_limited_round_trip"]
    mast_contacts = np.asarray(
        [
            row["wall_contact_point_m"]
            for row in receipt["solves"]
            if row["solve_id"].startswith("mast-limited-frame-")
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 4.2), constrained_layout=True)
    axes[0].plot(synthetic_wall[:, 0], synthetic_wall[:, 1], color="#777777")
    axes[0].plot(
        *round_trip["prescribed_wall_contact_point_m"],
        marker="x",
        color="#cc0000",
        label="prescribed",
    )
    axes[0].plot(
        *round_trip["recovered_wall_contact_point_m"],
        marker="o",
        fillstyle="none",
        color="#111111",
        label="recovered",
    )
    axes[0].set_title("Synthetic wall contact")
    axes[0].legend(frameon=False)
    axes[1].plot(mast_wall[:, 0], mast_wall[:, 1], color="#777777")
    axes[1].scatter(
        mast_contacts[:, 0], mast_contacts[:, 1], marker="o", color="#111111"
    )
    axes[1].set_title("MAST limited-frame contacts")
    for axis in axes:
        axis.set_xlabel("R [m]")
        axis.set_ylabel("Z [m]")
        axis.set_aspect("equal")
        axis.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    """Emit the JSON receipt and optionally bank it with a companion figure."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figure", type=Path)
    arguments = parser.parse_args(argv)
    receipt = build_receipt()
    if arguments.figure is not None:
        render_figure(receipt, arguments.figure)
    else:
        receipt.pop("_figure_geometry")
    encoded = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded)
    sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
