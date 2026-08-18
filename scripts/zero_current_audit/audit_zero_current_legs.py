"""Adjudicate lower-leg current errors on the banked coarse fixture."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nova.equilibrium.domain import DomainMasks, PlasmaDomain
from nova.equilibrium.stencil_mesh import CellCurrentMoments


SOURCE_AXIS_FLUX_WB = -86.01817570002173
SOURCE_BOUNDARY_FLUX_WB = -4.7117712394715845
CURRENT_RESOLUTION_A = 1.0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=Path,
        default=(
            root.parent / "ring_quadrature/inputs/coarse-fixture-reference-inputs.npz"
        ),
    )
    parser.add_argument(
        "--localization",
        type=Path,
        default=root.parent / "ring_quadrature/inputs/source-shift-localization.npz",
    )
    parser.add_argument(
        "--ring-fields",
        type=Path,
        default=root.parent / "ring_quadrature/results/ring-quadrature-fields.npz",
    )
    parser.add_argument("--output", type=Path, default=root / "results")
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path(
            "docs/figures/boundary-ring-source-completion/"
            "ring-m0-current-weighted-error.png"
        ),
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as stored:
        return {name: stored[name] for name in stored.files}


def load_reference():
    path = Path("tests/test_equilibrium_forward_reference.py")
    spec = importlib.util.spec_from_file_location("zero_current_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.configure_dtypes()
    return module, module.require_reference()


def source_partition_probe(
    source,
    centroid_psi_norm: float,
    support_node_psi_norm: float,
    domain: PlasmaDomain,
) -> float:
    """Return core node density sent to quadrature for one labelled cell."""
    captured: list[np.ndarray] = []

    def support_moments(centroid_density, shared_density, support):
        del centroid_density, support
        captured.append(np.asarray(shared_density))
        zero = jnp.zeros_like(shared_density)
        return CellCurrentMoments(shared_density, zero, zero)

    def interior_moments(centroid_density, shared_density):
        del shared_density
        zero = jnp.zeros_like(centroid_density)
        return CellCurrentMoments(zero, zero, zero)

    label = jnp.asarray([int(domain)])
    masks = DomainMasks(label=label, psi_norm=jnp.asarray([centroid_psi_norm]))
    shared_masks = DomainMasks(
        label=label, psi_norm=jnp.asarray([support_node_psi_norm])
    )
    source.current_moments(
        radius=jnp.asarray([5.0]),
        masks=masks,
        shared_radius=jnp.asarray([5.0]),
        shared_masks=shared_masks,
        interior_moments=interior_moments,
        support_moments=support_moments,
        core_support=object(),
        common_support=object(),
    )
    if len(captured) != 1:
        raise AssertionError("core support quadrature was not called exactly once")
    return float(captured[0][0])


def current_weighted_error(
    attributed: np.ndarray, oracle: np.ndarray, selection: np.ndarray
) -> dict[str, float | int]:
    denominator = float(np.sum(np.abs(oracle[selection])))
    error = attributed[selection] - oracle[selection]
    if denominator <= 0.0:
        raise AssertionError("current-weighted error has a zero denominator")
    return {
        "cells": int(np.count_nonzero(selection)),
        "current_bearing_cells": int(
            np.count_nonzero(selection & (np.abs(oracle) > CURRENT_RESOLUTION_A))
        ),
        "oracle_absolute_current_a": denominator,
        "absolute_error_current_a": float(np.sum(np.abs(error))),
        "signed_error_current_a": float(np.sum(error)),
        "l1_relative_error": float(np.sum(np.abs(error)) / denominator),
        "maximum_cell_error_share": float(np.max(np.abs(error)) / denominator),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_current_weighted_errors(
    path: Path,
    centres: np.ndarray,
    nonempty: np.ndarray,
    ring: np.ndarray,
    leg: np.ndarray,
    oracle: np.ndarray,
    candidates: list[tuple[str, np.ndarray]],
) -> None:
    denominator = float(np.sum(np.abs(oracle[nonempty])))
    shares = [100.0 * (values - oracle) / denominator for _, values in candidates]
    finite = np.concatenate([np.abs(values[nonempty]) for values in shares])
    colour_limit = max(float(np.quantile(finite, 0.995)), np.finfo(float).eps)

    figure, axes = plt.subplots(
        1, len(candidates), figsize=(12.8, 4.7), sharex=True, sharey=True
    )
    for axes_one, (name, _), share in zip(axes, candidates, shares, strict=True):
        axes_one.scatter(
            centres[nonempty, 0],
            centres[nonempty, 1],
            c=share[nonempty],
            cmap="RdBu_r",
            vmin=-colour_limit,
            vmax=colour_limit,
            s=15,
            linewidths=0,
        )
        axes_one.scatter(
            centres[ring, 0],
            centres[ring, 1],
            facecolors="none",
            edgecolors="black",
            s=34,
            linewidths=0.45,
        )
        axes_one.scatter(
            centres[leg, 0],
            centres[leg, 1],
            marker="x",
            c="gold",
            edgecolors="black",
            s=38,
            linewidths=1.2,
        )
        axes_one.set_title(name)
        axes_one.set_xlabel("R [m]")
        axes_one.set_aspect("equal")
        axes_one.grid(alpha=0.16)
    axes[0].set_ylabel("Z [m]")
    colour = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=-colour_limit, vmax=colour_limit), cmap="RdBu_r"
    )
    colour.set_array([])
    colour_axes = figure.add_axes([0.89, 0.16, 0.018, 0.68])
    bar = figure.colorbar(colour, cax=colour_axes)
    bar.set_label("signed cell error / total |topology oracle current| [%]")
    figure.suptitle(
        "Current-weighted cell error; gold crosses are topology-zero lower-leg cells"
    )
    figure.subplots_adjust(left=0.06, right=0.86, bottom=0.12, top=0.84, wspace=0.10)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = load_npz(args.fixture)
    localization = load_npz(args.localization)
    fields = load_npz(args.ring_fields)
    reference_module, case = load_reference()
    source = reference_module.forward_source(case)

    centres = fixture["consistent_centres"]
    available = fixture["consistent_available"]
    nonempty = fixture["support_vertex_count"] >= 3
    ring = nonempty & ~available
    baseline = fields["baseline_m0"]
    own_geometry = fields["own_geometry_m0"]
    one_sided = fields["one_sided_m0"]
    bank_oracle = localization["analytic_m0"]

    lower_xpoint = case.x_point[np.argmin(case.x_point[:, 1])]
    flux = np.asarray(case.flux(centres[:, 0], centres[:, 1]))
    psi_norm = (flux - SOURCE_AXIS_FLUX_WB) / (
        SOURCE_BOUNDARY_FLUX_WB - SOURCE_AXIS_FLUX_WB
    )
    lower_leg = nonempty & (centres[:, 1] < lower_xpoint[1])
    private_flux = lower_leg & (psi_norm <= 1.0)
    open_leg = lower_leg & (psi_norm > 1.0)
    topology_oracle = bank_oracle.copy()
    topology_oracle[lower_leg] = 0.0

    if source.common_sol is not None or source.private_flux is not None:
        raise AssertionError(
            "reference source unexpectedly declares open-domain current"
        )
    if np.any(available[lower_leg]):
        raise AssertionError("lower-leg population unexpectedly includes stencil cells")
    if np.any(baseline[lower_leg] != 0.0):
        raise AssertionError(
            "banked baseline is nonzero in a topology-zero lower-leg cell"
        )
    if np.any(own_geometry[lower_leg] == 0.0) or np.any(one_sided[lower_leg] == 0.0):
        raise AssertionError(
            "a repair candidate did not expose the lower-leg attribution"
        )

    private_probe = source_partition_probe(source, 0.99, 0.9, PlasmaDomain.PRIVATE_FLUX)
    open_probe = source_partition_probe(source, 1.01, 0.9, PlasmaDomain.COMMON_SOL)
    if private_probe == 0.0 or open_probe == 0.0:
        raise AssertionError(
            "support partition probe did not expose core-density attribution"
        )

    rows: list[dict[str, object]] = []
    for cell in np.flatnonzero(lower_leg):
        bank_signal = (
            "metric_artifact"
            if abs(bank_oracle[cell]) > CURRENT_RESOLUTION_A
            else "no_relative_signal"
        )
        rows.append(
            {
                "cell": int(cell),
                "radius_m": float(centres[cell, 0]),
                "height_m": float(centres[cell, 1]),
                "psi_norm": float(psi_norm[cell]),
                "domain": "open_divertor_leg" if open_leg[cell] else "private_flux",
                "stencil_available": bool(available[cell]),
                "topology_oracle_current_a": 0.0,
                "bank_support_oracle_current_a": float(bank_oracle[cell]),
                "baseline_attributed_current_a": float(baseline[cell]),
                "own_geometry_attributed_current_a": float(own_geometry[cell]),
                "one_sided_attributed_current_a": float(one_sided[cell]),
                "bank_figure_classification": bank_signal,
                "own_geometry_classification": "genuine_misattributed_current",
                "one_sided_classification": "genuine_misattributed_current",
                "causal_function": (
                    "ForwardFluxOperator.cell_current_moments -> "
                    "ForwardSource.current_moments.partitioned_moments"
                ),
            }
        )
    write_rows(args.output / "leg-cell-classification.csv", rows)

    ranking = sorted(
        rows,
        key=lambda row: max(
            abs(float(row["own_geometry_attributed_current_a"])),
            abs(float(row["one_sided_attributed_current_a"])),
        ),
        reverse=True,
    )[:10]
    write_rows(args.output / "worst-leg-cells.csv", ranking)

    candidates = [
        ("Banked baseline", baseline),
        ("Own-geometry repair", own_geometry),
        ("One-sided repair", one_sided),
    ]
    plot_current_weighted_errors(
        args.figure,
        centres,
        nonempty,
        ring,
        lower_leg,
        topology_oracle,
        candidates,
    )

    interior = nonempty & available
    results = {
        "inputs": {
            "fixture": str(args.fixture.resolve()),
            "localization": str(args.localization.resolve()),
            "ring_fields": str(args.ring_fields.resolve()),
        },
        "population": {
            "nonempty_cells": int(np.count_nonzero(nonempty)),
            "stencil_available_cells": int(np.count_nonzero(interior)),
            "ring_cells": int(np.count_nonzero(ring)),
            "topology_zero_lower_leg_cells": int(np.count_nonzero(lower_leg)),
            "private_flux_cells": int(np.count_nonzero(private_flux)),
            "open_divertor_leg_cells": int(np.count_nonzero(open_leg)),
            "metric_artifact_cells": int(
                np.count_nonzero(
                    lower_leg & (np.abs(bank_oracle) > CURRENT_RESOLUTION_A)
                )
            ),
            "no_relative_signal_cells": int(
                np.count_nonzero(
                    lower_leg & (np.abs(bank_oracle) <= CURRENT_RESOLUTION_A)
                )
            ),
            "corrected_ring_current_bearing_cells": int(
                np.count_nonzero(
                    ring & (np.abs(topology_oracle) > CURRENT_RESOLUTION_A)
                )
            ),
        },
        "probe": {
            "source_declares_common_sol": source.common_sol is not None,
            "source_declares_private_flux": source.private_flux is not None,
            "support_node_psi_norm": 0.9,
            "private_flux_core_node_density_sent_to_support_a_m2": private_probe,
            "open_leg_core_node_density_sent_to_support_a_m2": open_probe,
            "causal_function": (
                "ForwardFluxOperator.cell_current_moments supplies a "
                "topology-unqualified "
                "core_support to ForwardSource.current_moments; its nested "
                "partitioned_moments evaluates the clipped core profile without a "
                "domain mask"
            ),
        },
        "current_weighted_error": {
            "definition": (
                "sum(abs(attributed - topology_oracle)) / sum(abs(topology_oracle))"
            ),
            "interior_baseline": current_weighted_error(
                baseline, topology_oracle, interior
            ),
            "ring_baseline": current_weighted_error(baseline, topology_oracle, ring),
            "ring_own_geometry": current_weighted_error(
                own_geometry, topology_oracle, ring
            ),
            "ring_one_sided": current_weighted_error(one_sided, topology_oracle, ring),
        },
        "lower_leg_attribution": {
            "topology_oracle_total_current_a": 0.0,
            "bank_support_oracle_total_current_a": float(
                np.sum(bank_oracle[lower_leg])
            ),
            "baseline_total_current_a": float(np.sum(baseline[lower_leg])),
            "own_geometry_total_current_a": float(np.sum(own_geometry[lower_leg])),
            "one_sided_total_current_a": float(np.sum(one_sided[lower_leg])),
            "own_geometry_max_absolute_cell_current_a": float(
                np.max(np.abs(own_geometry[lower_leg]))
            ),
            "one_sided_max_absolute_cell_current_a": float(
                np.max(np.abs(one_sided[lower_leg]))
            ),
        },
        "artifacts": {
            "classification": str(
                (args.output / "leg-cell-classification.csv").resolve()
            ),
            "worst_cells": str((args.output / "worst-leg-cells.csv").resolve()),
            "figure": str(args.figure.resolve()),
        },
    }
    results_path = args.output / "zero-current-results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
