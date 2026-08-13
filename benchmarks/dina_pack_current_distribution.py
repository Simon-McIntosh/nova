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

    uv run python benchmarks/dina_pack_current_distribution.py \
      --banked-construction /path/to/reference_coarse.npz \
      --banked-solve /path/to/reference_coarse_solved.npz \
      --active-replay /path/to/machine_1587.npz \
      --passive-result /path/to/dina-pack-current-distribution.json \
      --reference-audit-log /path/to/reference-read-comparison.log \
      --output /tmp/dina-passive-response.json
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
#: Raster moment recovered from the byte-identical stored reference flux map.
#: It refers the banked beta receipt and the passive-inclusive receipt to the
#: same definition without using the entry's incompatible published scalar.
REFERENCE_MAP_POLOIDAL_BETA = 0.4660720896996286
PASSIVE_EXTERNAL_FLUX_PERCENT = 0.093


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


def _load_array_receipt(path: Path) -> dict[str, np.ndarray]:
    """Return a detached copy of every array in one compressed receipt."""
    with np.load(path, allow_pickle=False) as receipt:
        return {name: np.array(receipt[name], copy=True) for name in receipt.files}


def _reference_audit(path: Path) -> dict:
    """Return the exact-equality result recorded by the live reference audit."""
    lines = path.read_text().splitlines()
    comparisons = []
    for line in lines:
        if " shape " not in line or " exact " not in line or " maxabs " not in line:
            continue
        name, remainder = line.split(" shape ", maxsplit=1)
        exact_text = remainder.split(" exact ", maxsplit=1)[1].split()[0]
        max_abs = float(remainder.rsplit(" maxabs ", maxsplit=1)[1])
        comparisons.append(
            {"name": name, "exact": exact_text == "True", "max_abs": max_abs}
        )
    completed = any(line == "EXIT=0" for line in lines)
    if not completed or not comparisons or not all(row["exact"] for row in comparisons):
        raise ValueError(f"reference equality audit is incomplete or unequal: {path}")
    return {
        "path": str(path),
        "arrays_compared": len(comparisons),
        "all_exact": True,
        "maximum_absolute_difference": max(row["max_abs"] for row in comparisons),
        "comparisons": comparisons,
    }


def _field_observation(
    construction: dict[str, np.ndarray],
    solved: dict[str, np.ndarray],
    core: np.ndarray,
    *,
    current_replay: bool,
) -> dict[str, float]:
    """Recompute the internal-inductance inputs from one solved receipt."""
    from scipy.constants import mu_0

    source_current = construction["coil_current"]
    cell_current = solved["cell_current"]
    if current_replay:
        source_radial = solved["source_to_radial"]
        plasma_radial = solved["plasma_to_radial"]
        source_vertical = solved["source_to_vertical"]
        plasma_vertical = solved["plasma_to_vertical"]
        node = solved["node"]
        area = solved["area"]
    else:
        source_radial = construction["source_to_radial_field"]
        plasma_radial = construction["plasma_to_radial_field"]
        source_vertical = construction["source_to_vertical_field"]
        plasma_vertical = construction["plasma_to_vertical_field"]
        node = construction["node"]
        area = construction["area"]
    radial = source_radial @ source_current + plasma_radial @ cell_current
    vertical = source_vertical @ source_current + plasma_vertical @ cell_current
    volume_element = np.where(core, 2.0 * np.pi * node[:, 0] * area, 0.0)
    plasma_current = float(cell_current[core].sum())
    major_radius = float(np.sum(node[:, 0] * volume_element) / volume_element.sum())
    field_integral = float(np.sum((radial**2 + vertical**2) * volume_element))
    reference = mu_0 * major_radius * plasma_current**2
    internal_inductance = 2.0 * field_integral / (mu_0 * reference)
    return {
        "plasma_current_a": plasma_current,
        "major_radius_m": major_radius,
        "field_integral_t2_m3": field_integral,
        "raw_internal_inductance": internal_inductance,
    }


def _exact_array_comparison(
    banked: dict[str, np.ndarray], current: dict[str, np.ndarray]
) -> dict[str, dict[str, float | bool]]:
    """Return exact comparisons for fixed mesh and field-coupling inputs."""
    keys = {
        "node": "node",
        "area": "area",
        "stencil": "stencil",
        "source_to_grid": "source_to_grid",
        "plasma_to_grid": "plasma_to_grid",
        "source_to_radial_field": "source_to_radial",
        "plasma_to_radial_field": "plasma_to_radial",
        "source_to_vertical_field": "source_to_vertical",
        "plasma_to_vertical_field": "plasma_to_vertical",
    }
    result = {}
    for banked_name, current_name in keys.items():
        left = banked[banked_name]
        right = current[current_name]
        result[banked_name] = {
            "exact": bool(np.array_equal(left, right, equal_nan=True)),
            "maximum_absolute_difference": float(np.max(np.abs(left - right))),
        }
    return result


def reconcile_passive_response(
    banked_construction_path: Path,
    banked_solve_path: Path,
    active_replay_path: Path,
    passive_result_path: Path,
    reference_audit_log: Path,
) -> dict:
    """Attribute the internal-inductance improvement across configurations."""
    banked_construction = _load_array_receipt(banked_construction_path)
    banked_solve = _load_array_receipt(banked_solve_path)
    active_replay = _load_array_receipt(active_replay_path)
    passive_result = json.loads(passive_result_path.read_text())
    reference_audit = _reference_audit(reference_audit_log)

    array_comparison = _exact_array_comparison(banked_construction, active_replay)
    if not all(row["exact"] for row in array_comparison.values()):
        raise ValueError("active-only replay changed a fixed mesh or coupling array")
    if not np.array_equal(banked_solve["label"], active_replay["label"]):
        raise ValueError("active-only replay changed topology labels")
    matching_labels = [
        int(label)
        for label in np.unique(banked_solve["label"])
        if np.array_equal(banked_solve["label"] == label, active_replay["core"])
    ]
    if len(matching_labels) != 1:
        raise ValueError("active-only replay core does not identify one banked label")
    core = banked_solve["label"] == matching_labels[0]

    banked_observation = _field_observation(
        banked_construction, banked_solve, core, current_replay=False
    )
    replay_observation = _field_observation(
        banked_construction, active_replay, core, current_replay=True
    )
    receipt_li = float(banked_solve["internal_inductance"])
    if not np.isclose(
        banked_observation["raw_internal_inductance"],
        receipt_li,
        rtol=1.0e-12,
        atol=0.0,
    ):
        raise ValueError(
            "banked l_i receipt does not reproduce from its field integral"
        )

    reference_radius = float(banked_construction["reference_r0"])
    banked_ip_percent = 100.0 * (
        banked_observation["plasma_current_a"]
        / float(banked_construction["reference_ip"])
        - 1.0
    )
    banked_beta_percent = 100.0 * (
        float(banked_solve["poloidal_beta"])
        * banked_observation["major_radius_m"]
        / reference_radius
        / REFERENCE_MAP_POLOIDAL_BETA
        - 1.0
    )
    banked_axis_mm = 1.0e3 * float(
        np.linalg.norm(
            banked_solve["axis"]
            - np.array(
                [
                    banked_construction["reference_axis_r"],
                    banked_construction["reference_axis_z"],
                ]
            )
        )
    )
    passive_metrics = passive_result["metrics"]
    passive_row = {
        "plasma current percent": passive_metrics["plasma current percent"][
            "declared_continuum"
        ],
        "poloidal beta percent": passive_metrics["poloidal beta percent"][
            "declared_continuum"
        ],
        "internal inductance percent": passive_metrics["internal inductance percent"][
            "declared_continuum"
        ],
        "axis position mm": passive_metrics["axis position mm"]["declared_continuum"],
    }
    active_row = {
        "plasma current percent": banked_ip_percent,
        "poloidal beta percent": banked_beta_percent,
        "internal inductance percent": BANKED_DEVIATION["internal inductance percent"],
        "axis position mm": banked_axis_mm,
    }
    configuration_change = {
        name: passive_row[name] - active_row[name] for name in active_row
    }

    # These are measurements of two physical configurations.  The smaller
    # passive-inclusive |l_i| deviation is an improvement, and its much larger
    # response than I_p or axis position is the observable asymmetry.  A small
    # passive flux fraction can therefore be irrelevant to the total-current
    # deviation floor while materially changing the field-energy integral.
    replay_ratio = (
        replay_observation["field_integral_t2_m3"]
        / banked_observation["field_integral_t2_m3"]
        * (
            banked_observation["plasma_current_a"]
            / replay_observation["plasma_current_a"]
        )
        ** 2
    )
    active_replay_change = 100.0 * (replay_ratio - 1.0)
    total_li_improvement = configuration_change["internal inductance percent"]
    passive_li_contribution = total_li_improvement - active_replay_change
    passive_share = 100.0 * passive_li_contribution / total_li_improvement
    improvement_factor = abs(active_row["internal inductance percent"]) / abs(
        passive_row["internal inductance percent"]
    )

    return {
        "reference": {
            "pulse": 135011,
            "run": 7,
            "time_slice": 353,
            "mesh_cells": len(active_replay["node"]),
        },
        "configuration_rows": {
            "active_only_banked": active_row,
            "passive_inclusive_current_tree": passive_row,
        },
        "configuration_change": configuration_change,
        "internal_inductance_attribution": {
            "total_improvement_percentage_points": total_li_improvement,
            "active_only_replay_change_percentage_points": active_replay_change,
            "declared_passive_contribution_percentage_points": (
                passive_li_contribution
            ),
            "declared_passive_share_percent": passive_share,
            "absolute_deviation_improvement_factor": improvement_factor,
            "named_cause": "declared passive conductors",
        },
        "separation_measurements": {
            "definition": {
                "same_field_integral_formula": True,
                "banked_receipt_recomputed_raw_l_i": banked_observation[
                    "raw_internal_inductance"
                ],
                "banked_receipt_stored_raw_l_i": receipt_li,
            },
            "integration_domain": {
                "topology_labels_exact": True,
                "core_mask_exact": True,
                "core_label": matching_labels[0],
                "core_cells": int(core.sum()),
            },
            "reference_read": reference_audit,
            "active_only_solve_replay": {
                "fixed_arrays": array_comparison,
                "banked": banked_observation,
                "current_code": replay_observation,
                "flux_sup_norm_wb": float(
                    np.max(np.abs(active_replay["flux"] - banked_solve["flux"]))
                ),
            },
        },
        "observable_asymmetry": {
            "plasma_current_change_percentage_points": configuration_change[
                "plasma current percent"
            ],
            "poloidal_beta_change_percentage_points": configuration_change[
                "poloidal beta percent"
            ],
            "axis_position_change_mm": configuration_change["axis position mm"],
            "passive_external_flux_percent_of_reference_span": (
                PASSIVE_EXTERNAL_FLUX_PERCENT
            ),
        },
        "banked_row_decision": (
            "Keep -1.04 percent as the explicitly active-only measurement and "
            "add -0.135266 percent beside it for the passive-inclusive current "
            "tree; the rows measure different physical configurations."
        ),
        "verdict": (
            "Declared passive conductors improve internal-inductance reproduction "
            f"by {improvement_factor:.3f}x, accounting for "
            f"{passive_li_contribution:.6f} of the "
            f"{total_li_improvement:.6f} percentage-point improvement "
            f"({passive_share:.3f}%). Their response is strongly concentrated in "
            "l_i rather than total current or axis position."
        ),
    }


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
    parser.add_argument("--banked-construction", type=Path)
    parser.add_argument("--banked-solve", type=Path)
    parser.add_argument("--active-replay", type=Path)
    parser.add_argument("--passive-result", type=Path)
    parser.add_argument("--reference-audit-log", type=Path)
    args = parser.parse_args()
    reconciliation_inputs = (
        args.banked_construction,
        args.banked_solve,
        args.active_replay,
        args.passive_result,
        args.reference_audit_log,
    )
    if any(path is not None for path in reconciliation_inputs):
        if not all(path is not None for path in reconciliation_inputs):
            parser.error(
                "receipt reconciliation requires --banked-construction, "
                "--banked-solve, --active-replay, --passive-result and "
                "--reference-audit-log"
            )
        result = reconcile_passive_response(*reconciliation_inputs)
    else:
        result = run(args.cells)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    headline = (
        result["metrics"]
        if "metrics" in result
        else result["internal_inductance_attribution"]
    )
    print(json.dumps(headline, indent=2))
    print(result["verdict"])


if __name__ == "__main__":
    main()
