"""Recover DIII-D PF-coil wiring evidence from current-trace identities.

The IMAS circuit and supply structures are checked first and reported empty.
Only ``pf_active`` is loaded: no magnetics signal or equilibrium reconstruction
is read.  Trace relationships recover shared drive, series opposition and fixed
gain only where the preregistered numeric tolerances support those statements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imas
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from nova.imas.diiid_description import POLOIDAL_CONDUCTORS

DEFAULT_ENTRY = Path("/home/ITER/tribolp/Public/imasdb/DIII-D/200000.nc")
DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/wiring")
DEFAULT_PREREGISTRATION = DEFAULT_OUTPUT / "wiring_preregistration.json"
DD_VERSION = "3.41.0"

CLASS_CODES = {
    "identical": 0,
    "exactly_negated": 1,
    "proportional": 2,
    "independent": 3,
}


def load_preregistration(path: Path) -> dict[str, Any]:
    declared = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "current_absolute_tolerance_A",
        "current_relative_tolerance",
        "proportional_residual_relative_tolerance",
        "proportional_correlation_floor",
        "time_base_absolute_tolerance_s",
        "classification_order",
    }
    missing = sorted(required - set(declared))
    if missing or not declared.get("declared_before_pairwise_current_comparison"):
        raise ValueError(f"incomplete wiring preregistration: missing {missing}")
    if declared["classification_order"] != list(CLASS_CODES):
        raise ValueError("classification order does not match the scorer")
    return declared


def scale_free_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = float(
        np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    )
    if denominator == 0.0:
        return None
    return float(np.dot(first_centered, second_centered) / denominator)


def compare_pair(
    first: np.ndarray,
    second: np.ndarray,
    tolerances: dict[str, Any],
) -> dict[str, Any]:
    """Classify the second trace relative to the first trace."""
    scale = float(max(np.max(np.abs(first)), np.max(np.abs(second))))
    equality_limit = (
        float(tolerances["current_absolute_tolerance_A"])
        + float(tolerances["current_relative_tolerance"]) * scale
    )
    maximum_difference = float(np.max(np.abs(first - second)))
    maximum_sum = float(np.max(np.abs(first + second)))
    correlation = scale_free_correlation(first, second)

    denominator = float(np.dot(first, first))
    ratio = float(np.dot(first, second) / denominator) if denominator else None
    proportional_residual = (
        float(np.max(np.abs(second - ratio * first))) if ratio is not None else None
    )
    proportional_limit = float(tolerances["current_absolute_tolerance_A"]) + float(
        tolerances["proportional_residual_relative_tolerance"]
    ) * float(np.max(np.abs(second)))

    if maximum_difference <= equality_limit:
        classification = "identical"
    elif maximum_sum <= equality_limit:
        classification = "exactly_negated"
    elif (
        correlation is not None
        and abs(correlation) >= float(tolerances["proportional_correlation_floor"])
        and proportional_residual is not None
        and proportional_residual <= proportional_limit
    ):
        classification = "proportional"
    else:
        classification = "independent"

    return {
        "classification": classification,
        "maximum_absolute_difference_A": maximum_difference,
        "maximum_absolute_sum_A": maximum_sum,
        "scale_free_correlation": correlation,
        "ratio_second_to_first": ratio,
        "maximum_proportional_residual_A": proportional_residual,
        "equality_tolerance_A": equality_limit,
        "proportional_residual_tolerance_A": proportional_limit,
    }


def _function_identifiers(coil: Any) -> list[int]:
    values = []
    for index in range(len(coil.function)):
        function_index = coil.function[index].index
        if function_index.has_value:
            values.append(int(function_index))
    return values


def _coil_record(coil: Any) -> dict[str, Any]:
    elements = []
    for index in range(len(coil.element)):
        element = coil.element[index]
        elements.append(
            {
                "name": str(element.name),
                "identifier": str(element.identifier),
                "turns_with_sign": float(element.turns_with_sign),
            }
        )
    return {
        "name": str(coil.name),
        "identifier": str(coil.identifier),
        "function_identifiers_verbatim": _function_identifiers(coil),
        "element_count": len(elements),
        "elements": elements,
        "current_time_base_length": int(np.asarray(coil.current.time).size),
    }


def _empty_structure_receipt(pf_active: Any, filled_paths: set[str]) -> dict[str, Any]:
    circuit_count = len(pf_active.circuit)
    supply_count = len(pf_active.supply)
    connection_count = 0
    for index in range(circuit_count):
        connection_count += len(pf_active.circuit[index].connections)
    searched = (
        ("pf_active/circuit", "circuit", circuit_count),
        ("pf_active/circuit/connections", "circuit/connections", connection_count),
        ("pf_active/supply", "supply", supply_count),
    )
    return {
        "statement": (
            "Measured negative result: pf_active has no circuit structure and no "
            "supply structure. The declared circuit matrix and power supplies are "
            "absent, so wiring cannot be read from a connection matrix."
        ),
        "method": (
            "imas-python structure lengths and netCDF-backend filled paths; no file "
            "header or raw netCDF inspection"
        ),
        "paths_searched": [
            {
                "path": exact,
                "access_layer_relative_path": relative,
                "filled": relative in filled_paths
                or any(path.startswith(relative + "/") for path in filled_paths),
                "structure_or_connection_count": count,
            }
            for exact, relative, count in searched
        ],
        "circuit_count": circuit_count,
        "connection_count": connection_count,
        "supply_count": supply_count,
    }


def build_receipt(
    entry_path: Path, preregistration_path: Path
) -> tuple[dict[str, Any], np.ndarray]:
    tolerances = load_preregistration(preregistration_path)
    with imas.DBEntry(entry_path, "r", dd_version=DD_VERSION) as entry:
        filled_paths = set(entry.list_filled_paths("pf_active", autoconvert=False))
        pf_active = entry.get("pf_active", autoconvert=False)
        written_dd = str(pf_active.ids_properties.version_put.data_dictionary)
        if written_dd != DD_VERSION:
            raise RuntimeError(f"expected DD {DD_VERSION}, read {written_dd}")

        negative_result = _empty_structure_receipt(pf_active, filled_paths)
        if any(
            path["filled"] or path["structure_or_connection_count"]
            for path in negative_result["paths_searched"]
        ):
            raise RuntimeError("pf_active unexpectedly contains circuit or supply data")

        coils = [
            _coil_record(pf_active.coil[index]) for index in range(len(pf_active.coil))
        ]
        names = [coil["name"] for coil in coils]
        times = [
            np.asarray(pf_active.coil[index].current.time, dtype=float)
            for index in range(len(pf_active.coil))
        ]
        currents = np.stack(
            [
                np.asarray(pf_active.coil[index].current.data, dtype=float)
                for index in range(len(pf_active.coil))
            ]
        )
        if not np.all(np.isfinite(currents)) or any(
            not np.all(np.isfinite(time)) for time in times
        ):
            raise RuntimeError("PF current comparison requires finite traces and times")

        reference_time = times[0]
        time_differences = [
            float(np.max(np.abs(time - reference_time))) for time in times
        ]
        allowed_time_difference = float(tolerances["time_base_absolute_tolerance_s"])
        if (
            any(time.shape != reference_time.shape for time in times)
            or max(time_differences) > allowed_time_difference
        ):
            raise RuntimeError(
                "PF coils do not share the preregistered common time base"
            )

    pairwise = []
    matrix = np.full((len(names), len(names)), CLASS_CODES["independent"], dtype=int)
    np.fill_diagonal(matrix, CLASS_CODES["identical"])
    for first_index, first_name in enumerate(names):
        for second_index in range(first_index + 1, len(names)):
            second_name = names[second_index]
            comparison = compare_pair(
                currents[first_index], currents[second_index], tolerances
            )
            comparison.update({"first": first_name, "second": second_name})
            pairwise.append(comparison)
            code = CLASS_CODES[comparison["classification"]]
            matrix[first_index, second_index] = code
            matrix[second_index, first_index] = code

    class_counts = {
        name: sum(pair["classification"] == name for pair in pairwise)
        for name in CLASS_CODES
    }
    element_counts = {coil["name"]: coil["element_count"] for coil in coils}
    expected_element_counts = {
        "ECOILA": 48,
        "ECOILB": 48,
        "E567UP": 6,
        "E567DN": 6,
        "E89UP": 7,
        "E89DN": 7,
        **{name: 1 for name in names if name.startswith("F")},
    }
    if element_counts != expected_element_counts or sum(element_counts.values()) != 140:
        raise RuntimeError(f"unexpected PF element grouping: {element_counts}")

    competition = set(POLOIDAL_CONDUCTORS)
    absent_from_competition = [name for name in names if name not in competition]
    receipt = {
        "source": str(entry_path),
        "backend": "imas-python netCDF",
        "dd_version": DD_VERSION,
        "ids_read": ["pf_active"],
        "ids_explicitly_not_read": ["magnetics", "equilibrium"],
        "negative_result_first": negative_result,
        "preregistered_tolerances": tolerances,
        "classification_meanings": {
            "identical": (
                "same current trace within the declared tolerance; common drive"
            ),
            "exactly_negated": (
                "opposite current trace within the declared tolerance; series "
                "opposition"
            ),
            "proportional": (
                "fixed through-origin current ratio within the declared residual and "
                "correlation tolerances"
            ),
            "independent": (
                "none of the three exact wiring relationships above; this does not by "
                "itself prove a physically separate power supply"
            ),
        },
        "shared_current_time_base": {
            "coil_count": len(coils),
            "sample_count": int(reference_time.size),
            "time_span_s": [float(reference_time[0]), float(reference_time[-1])],
            "maximum_pairwise_time_difference_s": max(time_differences),
        },
        "intra_coil_series_structure": {
            "coil_count": len(coils),
            "element_count": sum(element_counts.values()),
            "element_counts": element_counts,
            "coils": coils,
        },
        "competition_19_conductor_table": {
            "shipped": list(POLOIDAL_CONDUCTORS),
            "netcdf_coils_absent_from_competition_table": absent_from_competition,
            "statement": (
                "The competition table omits ECOILB and the four multi-element E-coil "
                "groups. ECOILB is the structurally incomplete ohmic-set candidate "
                "named by the vacuum-gate diagnosis and is now confirmed to have a "
                "real 48-element geometry and measured current trace in this entry."
            ),
        },
        "pairwise_current_relationships": pairwise,
        "pair_count": len(pairwise),
        "classification_counts": class_counts,
        "fitting_performed": False,
    }
    return receipt, matrix


def plot_matrix(names: list[str], matrix: np.ndarray, output_path: Path) -> None:
    colors = ["#2f855a", "#805ad5", "#3182ce", "#d6dde5"]
    labels = ["common drive", "series opposition", "fixed ratio", "independent"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N)
    figure, axis = plt.subplots(figsize=(9.4, 8.2), constrained_layout=True)
    axis.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest")
    axis.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    axis.set_yticks(range(len(names)), names, fontsize=7)
    axis.set_xlabel("PF coil")
    axis.set_ylabel("PF coil")
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=color, label=label)
        for color, label in zip(colors, labels, strict=True)
    ]
    axis.legend(
        handles=handles,
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=8,
    )
    for spine in axis.spines.values():
        spine.set_visible(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_outputs(
    receipt: dict[str, Any], matrix: np.ndarray, output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "diiid_coil_wiring.json"
    figure_path = output_dir / "pairwise_wiring_matrix.png"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    names = [coil["name"] for coil in receipt["intra_coil_series_structure"]["coils"]]
    plot_matrix(names, matrix, figure_path)
    return receipt_path, figure_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", type=Path, default=DEFAULT_ENTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    args = parser.parse_args()
    receipt, matrix = build_receipt(args.entry, args.preregistration)
    receipt_path, figure_path = write_outputs(receipt, matrix, args.output_dir)
    print(f"wrote {receipt_path} and {figure_path}")
    print(
        f"{receipt['pair_count']} pairs; classes "
        f"{receipt['classification_counts']}; missing from competition "
        f"{receipt['competition_19_conductor_table']['netcdf_coils_absent_from_competition_table']}"
    )


if __name__ == "__main__":
    main()
