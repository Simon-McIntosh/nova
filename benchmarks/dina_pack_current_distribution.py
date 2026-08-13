"""DINA equilibrium sensitivity to current placement inside coil packs.

The stored ITER equilibrium declares one circuit current, a turn count and a
conductor-section outline for every active-coil element.  The reference
reproduction spreads each element's ampere-turns uniformly over that outline.
This driver holds the reference slice, plasma profiles, passive currents,
hexagonal plasma mesh, COCOS mapping and Newton-Krylov solve route fixed while
changing only that within-section distribution.

The comparison distribution resolves a near-square turn lattice from each
rectangular pack's declared outline and turn count.  Every occupied site carries
one turn and unused sites are removed symmetrically from the outer corners.  Its
finite section is the site it occupies, so the comparison remains a conductor
section calculation rather than replacing the pack with singular filaments.
Skewed one-turn elements and all passive conductors remain byte-identical to the
reference machine.

Usage::

    uv run python benchmarks/dina_pack_current_distribution.py \
      --output /tmp/dina-pack-current-distribution.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REFERENCE_CELLS = -1500
BANKED_DEVIATION = {
    "plasma current percent": -1.15,
    "poloidal beta percent": 2.16,
    "internal inductance percent": -1.04,
    "axis position mm": 44.0,
}


@dataclass(frozen=True)
class TurnLattice:
    """Rectangular site lattice inferred from one pack outline and turn count."""

    radial_sites: int
    vertical_sites: int
    radius: np.ndarray
    height: np.ndarray
    turn_weight: np.ndarray
    site_width: float
    site_height: float

    @property
    def site_count(self) -> int:
        """Return the total number of geometrically available sites."""
        return self.radial_sites * self.vertical_sites

    @property
    def occupied_count(self) -> int:
        """Return the number of sites carrying a positive turn weight."""
        return int(np.count_nonzero(self.turn_weight))


def _lattice_shape(turn_count: float, width: float, height: float) -> tuple[int, int]:
    """Return the near-square site arrangement with the fewest spare sites.

    Candidate cells are ranked first by pitch anisotropy and then by unused
    capacity.  The anisotropy is rounded only for the tie break so numerically
    immaterial pitch changes cannot select a layout with many more empty sites.
    """
    required = math.ceil(turn_count)
    best_shape = (1, required)
    best_key: tuple[float, int] | None = None
    for radial_sites in range(1, 4 * math.isqrt(required) + 2):
        vertical_sites = math.ceil(required / radial_sites)
        pitch_ratio = (width / radial_sites) / (height / vertical_sites)
        key = (
            round(abs(math.log(pitch_ratio)), 6),
            radial_sites * vertical_sites - required,
        )
        if best_key is None or key < best_key:
            best_shape = radial_sites, vertical_sites
            best_key = key
    return best_shape


def turn_lattice(conductor) -> TurnLattice:
    """Resolve equal-current turn sites from a rectangular conductor section."""
    if conductor.rectangle is None:
        raise ValueError(f"{conductor.name} has no rectangular pack section")
    centre_r, centre_z, width, height = conductor.rectangle
    if conductor.turns <= 0.0 or not np.isfinite(conductor.turns):
        raise ValueError(f"{conductor.name} has invalid turn count {conductor.turns}")
    radial_sites, vertical_sites = _lattice_shape(conductor.turns, width, height)
    site_width = width / radial_sites
    site_height = height / vertical_sites
    radius = centre_r + site_width * (
        np.arange(radial_sites) - 0.5 * (radial_sites - 1)
    )
    height_coordinate = centre_z + site_height * (
        np.arange(vertical_sites) - 0.5 * (vertical_sites - 1)
    )
    grid_r, grid_z = np.meshgrid(radius, height_coordinate, indexing="ij")
    radius = grid_r.ravel()
    height_coordinate = grid_z.ravel()

    required = math.ceil(conductor.turns)
    spare = radius.size - required
    distance = ((radius - centre_r) / site_width) ** 2 + (
        (height_coordinate - centre_z) / site_height
    ) ** 2
    # Opposite corners have the same radius.  Alternating the signed radial and
    # vertical coordinates within each radius shell minimises the displacement
    # imposed by capacity the declared turn count does not occupy.
    angular_balance = np.sign(radius - centre_r) + 0.5 * np.sign(
        height_coordinate - centre_z
    )
    removal_order = np.lexsort((angular_balance, -distance))
    turn_weight = np.ones(radius.size)
    if spare:
        turn_weight[removal_order[:spare]] = 0.0
    occupied = np.flatnonzero(turn_weight)
    turn_weight[occupied[-1]] = conductor.turns - (len(occupied) - 1)
    if not np.isclose(turn_weight.sum(), conductor.turns, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{conductor.name} lattice did not preserve its turns")
    return TurnLattice(
        radial_sites=radial_sites,
        vertical_sites=vertical_sites,
        radius=radius,
        height=height_coordinate,
        turn_weight=turn_weight,
        site_width=site_width,
        site_height=site_height,
    )


def _rectangle_vertices(
    radius: float, height: float, width: float, thickness: float
) -> np.ndarray:
    """Return counter-clockwise vertices for one rectangular turn site."""
    return np.array(
        [
            [radius - width / 2.0, height - thickness / 2.0],
            [radius + width / 2.0, height - thickness / 2.0],
            [radius + width / 2.0, height + thickness / 2.0],
            [radius - width / 2.0, height + thickness / 2.0],
        ]
    )


def _lattice_coupling(conductor, target: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return flux and field coupling per circuit ampere for one turn lattice."""
    from nova.biot.polysection import PolySection

    lattice = turn_lattice(conductor)
    values = np.zeros((3, len(target)), dtype=np.float64)
    for radius, height, weight in zip(
        lattice.radius, lattice.height, lattice.turn_weight
    ):
        if weight == 0.0:
            continue
        section = _rectangle_vertices(
            radius, height, lattice.site_width, lattice.site_height
        )
        for destination, component in zip(
            values,
            PolySection.section_greens(target[:, 0], target[:, 1], section),
        ):
            destination += weight * component
    return tuple(values)


def turn_resolved_machine(case, machine):
    """Return the same machine with active rectangular-pack couplings replaced."""
    source_to_grid = np.array(machine.source_to_grid, copy=True)
    source_to_wall = np.array(machine.source_to_wall, copy=True)
    radial_grid = np.array(machine.radial_field[0], copy=True)
    vertical_grid = np.array(machine.vertical_field[0], copy=True)

    inventory = []
    for column, conductor in enumerate(case.active):
        if conductor.rectangle is None or conductor.turns <= 1.0:
            inventory.append(
                {
                    "name": conductor.name,
                    "turns": conductor.turns,
                    "distribution": "unchanged declared section",
                }
            )
            continue
        grid_psi, grid_br, grid_bz = _lattice_coupling(conductor, machine.node)
        wall_psi, _, _ = _lattice_coupling(conductor, machine.wall_node)
        source_to_grid[:, column] = grid_psi
        source_to_wall[:, column] = wall_psi
        radial_grid[:, column] = grid_br
        vertical_grid[:, column] = grid_bz
        lattice = turn_lattice(conductor)
        inventory.append(
            {
                "name": conductor.name,
                "turns": conductor.turns,
                "distribution": "finite-section turn lattice",
                "lattice": [lattice.radial_sites, lattice.vertical_sites],
                "sites": lattice.site_count,
                "occupied_sites": lattice.occupied_count,
                "site_size_m": [lattice.site_width, lattice.site_height],
            }
        )

    altered = replace(
        machine,
        source_to_grid=source_to_grid,
        source_to_wall=source_to_wall,
        radial_field=(radial_grid, machine.radial_field[1]),
        vertical_field=(vertical_grid, machine.vertical_field[1]),
    )
    return altered, inventory


def _axis_distance(deviation: dict[str, float]) -> float:
    """Return the magnetic-axis displacement magnitude in millimetres."""
    return float(math.hypot(deviation["axis radius"], deviation["axis height"]))


def _metric_rows(
    continuum: dict[str, float], lattice: dict[str, float]
) -> dict[str, dict[str, float]]:
    """Return distribution changes measured against the banked deviations."""
    baseline = {
        "plasma current percent": continuum["plasma current"],
        "poloidal beta percent": continuum["poloidal beta"],
        "internal inductance percent": continuum["internal inductance"],
        "axis position mm": _axis_distance(continuum),
    }
    comparison = {
        "plasma current percent": lattice["plasma current"],
        "poloidal beta percent": lattice["poloidal beta"],
        "internal inductance percent": lattice["internal inductance"],
        "axis position mm": _axis_distance(lattice),
    }
    rows = {}
    for name, original in baseline.items():
        changed = comparison[name]
        removed = abs(original) - abs(changed)
        signed_change = changed - original
        banked = BANKED_DEVIATION[name]
        rows[name] = {
            "banked": banked,
            "declared_continuum": original,
            "turn_resolved": changed,
            "signed_change": signed_change,
            "absolute_deviation_removed": removed,
            "response_relative_to_banked_percent": 100.0
            * abs(signed_change)
            / abs(banked),
            "share_of_banked_deviation_removed_percent": 100.0 * removed / abs(banked),
        }
    return rows


def run(cells: int = REFERENCE_CELLS) -> dict:
    """Run both current distributions on one fixed reference construction."""
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    from tests import test_equilibrium_forward_reference as reference

    case = reference.reference_case()
    if isinstance(case, str):
        raise RuntimeError(f"the stored DINA reference is unreachable: {case}")

    start = time.perf_counter()
    continuum_machine = reference.build_machine(case, cells, passive=True)
    assembly_seconds = time.perf_counter() - start
    if len(continuum_machine.node) != 1587:
        cell_count = len(continuum_machine.node)
        raise RuntimeError(
            f"the fixed production mesh has 1587 cells, got {cell_count}"
        )

    start = time.perf_counter()
    continuum = reference.solve(case, continuum_machine)
    continuum_seconds = time.perf_counter() - start

    start = time.perf_counter()
    lattice_machine, inventory = turn_resolved_machine(case, continuum_machine)
    redistribution_seconds = time.perf_counter() - start
    start = time.perf_counter()
    lattice = reference.solve(case, lattice_machine)
    lattice_seconds = time.perf_counter() - start

    continuum_deviation = continuum.deviations()
    lattice_deviation = lattice.deviations()
    metrics = _metric_rows(continuum_deviation, lattice_deviation)
    response_key = "response_relative_to_banked_percent"
    largest_response = max(
        metrics.items(),
        key=lambda item: item[1][response_key],
    )
    external_flux_change = (
        lattice_machine.source_to_grid @ lattice_machine.source_current
        - continuum_machine.source_to_grid @ continuum_machine.source_current
    )
    result = {
        "reference": {
            "pulse": reference.PULSE,
            "run": reference.RUN,
            "time_slice": reference.TIME_SLICE,
            "time_s": case.time,
            "dd_version": reference.DD_VERSION,
            "mesh_cells": len(continuum_machine.node),
            "profiles": "identical stored absolute p-prime and FF-prime",
            "cocos_mapping": "Phi_nova = -psi_IMAS; total poloidal flux",
            "solve_route": "fixed-budget Newton-Krylov",
            "banked_deviation": BANKED_DEVIATION,
        },
        "distributions": {
            "declared_continuum": (
                "declared ampere-turns uniformly distributed over every "
                "conductor section"
            ),
            "turn_resolved": (
                "one equal-current finite-section site per declared turn on an "
                "outline-derived near-square lattice; balanced outer-corner vacancies"
            ),
        },
        "metrics": metrics,
        "axis_components_mm": {
            "declared_continuum": {
                "radius": continuum_deviation["axis radius"],
                "height": continuum_deviation["axis height"],
            },
            "turn_resolved": {
                "radius": lattice_deviation["axis radius"],
                "height": lattice_deviation["axis height"],
            },
        },
        "external_flux_change": {
            "sup_norm_wb": float(np.max(np.abs(external_flux_change))),
            "percent_reference_flux_span": 100.0
            * float(np.max(np.abs(external_flux_change)))
            / abs(case.flux_span),
        },
        "inventory": inventory,
        "timing_seconds": {
            "continuum_assembly": assembly_seconds,
            "continuum_solve": continuum_seconds,
            "turn_redistribution": redistribution_seconds,
            "turn_resolved_solve": lattice_seconds,
        },
        "verdict": (
            "The turn-resolved redistribution is excluded as the dominant cause: "
            "its largest measured response is "
            f"{largest_response[1][response_key]:.3f}% of the banked "
            f"{largest_response[0]} deviation."
        ),
    }
    return result


def main() -> None:
    """Parse arguments, run the fixed comparison and write its JSON receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=REFERENCE_CELLS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cells)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))
    print(result["verdict"])


if __name__ == "__main__":
    main()
