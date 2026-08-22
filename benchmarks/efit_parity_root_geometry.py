"""Attribute the LCFS metric miss of one converged constrained MAST root.

The registered LCFS score compares the longest unit-normalised boundary
polyline with the stored polygon through an unordered symmetric nearest-
neighbour mean.  This diagnostic exposes both directed distance distributions
and repeats the comparison with the closed boundary branch that encloses the
magnetic axis.  The single recovered terminal state is serialized so later
diagnostics can observe it without another nonlinear solve.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from contourpy import contour_generator
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from benchmarks.efit_forward_parity_slice import (  # noqa: E402
    DECOMPOSITION_BANK,
    FIXED_POINT_CRITERION,
    NEWTON_STEPS,
    _contour,
    _mast_case_from_selection,
    _passive_inclusive_case,
    _passive_inclusive_solve,
    _symmetric_mean_distance,
    select_slices_by_shot,
)
from benchmarks.efit_topology_boundary_score import _stored_lcfs  # noqa: E402
from nova.imas.mast_solve_inputs import SHOT_STORE  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402

OUTPUT_DIRECTORY = Path("docs/figures/efit-forward-parity")
RECEIPT_NAME = "converged-root-geometry-attribution.json"
FIGURE_NAME = "converged-root-geometry-attribution.png"
TARGET_SHOT = 22086
EXPECTED_BANKED_FILE_COUNT = 23
ORDER_OF_MAGNITUDE = 10.0

EXPECTED_REFERENCE = {
    "shot": 22086,
    "slice_index": 43,
    "time_s": 0.22999998927116394,
    "span_wb": 1.5839675751331153,
}
EXPECTED_ROOT = {
    "residual": 2.911868346631881e-16,
    "plasma_current_a": 933034.875,
    "amplitude": 1.008098186771406,
}


def _sha256_array(values: np.ndarray) -> str:
    """Return a stable digest of one float64 state vector."""
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _banked_digests(directory: Path) -> dict[str, str]:
    """Digest the pre-existing evidence while excluding this node's outputs."""
    owned = {RECEIPT_NAME, FIGURE_NAME}
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name not in owned
    }


def _contour_geometry(points: np.ndarray) -> dict[str, Any]:
    """Return sampling and geometric extent for one polyline."""
    if len(points) < 2:
        raise ValueError("a contour needs at least two points")
    return {
        "point_count": int(len(points)),
        "total_arclength_m": float(
            np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        ),
        "endpoint_closure_gap_m": float(np.linalg.norm(points[0] - points[-1])),
        "bounding_box_m": {
            "r_min": float(np.min(points[:, 0])),
            "r_max": float(np.max(points[:, 0])),
            "z_min": float(np.min(points[:, 1])),
            "z_max": float(np.max(points[:, 1])),
        },
    }


def _nearest_neighbour_distribution(
    source: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    """Return the directed nearest-neighbour distance distribution."""
    if len(source) < 2 or len(target) < 2:
        raise ValueError("nearest-neighbour distributions need two point sets")
    distance = np.asarray(cKDTree(target).query(source)[0], dtype=np.float64)
    return {
        "min_m": float(np.min(distance)),
        "median_m": float(np.median(distance)),
        "p90_m": float(np.percentile(distance, 90.0)),
        "max_m": float(np.max(distance)),
        "mean_m": float(np.mean(distance)),
    }


def _unit_boundary_branches(
    radius: np.ndarray,
    height: np.ndarray,
    flux: np.ndarray,
    axis_flux: float,
    boundary_flux: float,
) -> list[np.ndarray]:
    """Return every finite unit-normalised boundary polyline."""
    normalised = (flux - axis_flux) / (boundary_flux - axis_flux)
    lines = contour_generator(x=radius, y=height, z=normalised.T).lines(1.0)
    branches = [line[np.all(np.isfinite(line), axis=1)] for line in lines]
    return [line for line in branches if len(line) >= 4]


def _closed_axis_branch(
    branches: list[np.ndarray], axis_point: np.ndarray
) -> np.ndarray:
    """Select the longest explicitly closed branch enclosing an axis point."""
    candidates: list[np.ndarray] = []
    for branch in branches:
        scale = max(float(np.ptp(branch[:, 0])), float(np.ptp(branch[:, 1])), 1.0)
        closure_gap = float(np.linalg.norm(branch[0] - branch[-1]))
        if closure_gap > 1.0e-9 * scale:
            continue
        if MplPath(branch, closed=True).contains_point(axis_point):
            candidates.append(branch)
    if not candidates:
        raise RuntimeError("no closed unit-boundary branch encloses the solved axis")
    return max(
        candidates,
        key=lambda line: float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()),
    )


def _distance_pair(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Expose the aggregate and both directed components of the LCFS metric."""
    left_to_right = _nearest_neighbour_distribution(left, right)
    right_to_left = _nearest_neighbour_distribution(right, left)
    symmetric = _symmetric_mean_distance(left, right)
    rebuilt = 0.5 * (left_to_right["mean_m"] + right_to_left["mean_m"])
    if not np.isclose(symmetric, rebuilt, rtol=0.0, atol=1.0e-15):
        raise RuntimeError("the exposed directed means do not reproduce the metric")
    return {
        "symmetric_mean_distance_m": symmetric,
        "left_to_right": left_to_right,
        "right_to_left": right_to_left,
        "correspondence_constraint": "none; unordered cKDTree nearest neighbours",
    }


def _lcfs_attribution(profile, equilibrium, group, row: int) -> tuple[dict, dict]:
    """Compare the registered longest branch with the axis-enclosing branch."""
    solved_flux = np.asarray(
        equilibrium.flux[: profile.lattice.node_count], dtype=np.float64
    ).reshape(profile.lattice.shape)
    topology = equilibrium.topology
    axis_point = np.asarray(topology.axis, dtype=np.float64)
    stored_axis = np.asarray(
        [group["magnetic_axis_r"][row], group["magnetic_axis_z"][row]],
        dtype=np.float64,
    )
    branches = _unit_boundary_branches(
        profile.lattice.radius,
        profile.lattice.height,
        solved_flux,
        float(topology.axis_flux),
        float(topology.boundary_flux),
    )
    longest = _contour(
        profile.lattice.radius,
        profile.lattice.height,
        solved_flux,
        float(topology.axis_flux),
        float(topology.boundary_flux),
    )
    closed = _closed_axis_branch(branches, axis_point)
    stored = _stored_lcfs(group, row)

    aggregate = _distance_pair(longest, stored)
    restricted = _distance_pair(closed, stored)
    ratio = aggregate["symmetric_mean_distance_m"] / max(
        restricted["symmetric_mean_distance_m"], np.finfo(np.float64).tiny
    )
    classification = (
        "METRIC_SELECTION_ARTIFACT"
        if ratio >= ORDER_OF_MAGNITUDE
        else "PHYSICAL_BOUNDARY_DISPLACEMENT"
    )
    record = {
        "registered_metric_mechanism": {
            "solved_selection": "longest unit-normalised psi_N=1 polyline",
            "stored_selection": "declared finite stored LCFS polygon",
            "match": "symmetric mean of directed unordered nearest-neighbour distances",
        },
        "magnetic_axes_m": {
            "solved": axis_point.tolist(),
            "stored": stored_axis.tolist(),
            "distance_m": float(np.linalg.norm(axis_point - stored_axis)),
        },
        "unit_boundary_branch_count": len(branches),
        "aggregate_longest_branch": {
            "solved_contour": _contour_geometry(longest),
            "stored_contour": _contour_geometry(stored),
            "distance": aggregate,
        },
        "closed_branch_enclosing_solved_axis": {
            "solved_contour": _contour_geometry(closed),
            "stored_polygon_encloses_stored_axis": bool(
                MplPath(stored, closed=True).contains_point(stored_axis)
            ),
            "distance": restricted,
        },
        "aggregate_over_closed_branch_distance_ratio": ratio,
        "classification_threshold_ratio": ORDER_OF_MAGNITUDE,
        "classification": classification,
        "supported_reading": (
            "The longest-polyline selector chose a different boundary component; "
            "the axis-enclosing separatrix is at least ten times closer."
            if classification == "METRIC_SELECTION_ARTIFACT"
            else "The axis-enclosing separatrix remains displaced; branch selection "
            "does not explain the registered LCFS miss."
        ),
        "registered_tolerance_adjusted": False,
    }
    return record, {
        "longest": longest,
        "closed": closed,
        "stored": stored,
        "solved_axis": axis_point,
        "stored_axis": stored_axis,
    }


def _plot_contours(fields: dict[str, np.ndarray], record: dict, path: Path) -> None:
    """Plot only the spatial relation needed to interpret branch selection."""
    figure, axis = plt.subplots(figsize=(5.4, 5.7), constrained_layout=True)
    axis.plot(
        fields["longest"][:, 0],
        fields["longest"][:, 1],
        color="#d95f02",
        lw=1.2,
        label="solved longest branch",
    )
    axis.plot(
        fields["closed"][:, 0],
        fields["closed"][:, 1],
        color="#1b9e77",
        lw=1.5,
        label="solved closed axis branch",
    )
    axis.plot(
        fields["stored"][:, 0],
        fields["stored"][:, 1],
        color="#222222",
        lw=1.1,
        ls="--",
        label="stored LCFS",
    )
    axis.scatter(*fields["solved_axis"], marker="+", s=50, color="#1b9e77")
    axis.scatter(*fields["stored_axis"], marker="x", s=35, color="#222222")
    aggregate = record["aggregate_longest_branch"]["distance"][
        "symmetric_mean_distance_m"
    ]
    restricted = record["closed_branch_enclosing_solved_axis"]["distance"][
        "symmetric_mean_distance_m"
    ]
    axis.set_title(
        f"MAST 22086/43 LCFS: longest {aggregate:.4f} m; axis branch {restricted:.4f} m"
    )
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.set_aspect("equal")
    axis.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _assert_reference(reference: dict[str, Any]) -> None:
    """Fail closed if frozen-six selection or units drift."""
    for key in ("shot", "slice_index"):
        if int(reference[key]) != int(EXPECTED_REFERENCE[key]):
            raise RuntimeError(f"target reference {key} drifted")
    for key in ("time_s", "span_wb"):
        if not np.isclose(
            float(reference[key]), float(EXPECTED_REFERENCE[key]), rtol=0.0, atol=1e-15
        ):
            raise RuntimeError(f"target reference {key} drifted")


def run_lcfs_attribution(
    store: Path = SHOT_STORE,
    bank: Path = DECOMPOSITION_BANK,
    output: Path = OUTPUT_DIRECTORY,
) -> dict[str, Any]:
    """Re-solve only the selected row and bank its LCFS mechanism receipt."""
    configure_dtypes()
    baseline = _banked_digests(output)
    if len(baseline) != EXPECTED_BANKED_FILE_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_BANKED_FILE_COUNT} banked files, found {len(baseline)}"
        )

    selected = next(
        item
        for item in select_slices_by_shot(bank)
        if int(item[0]["shot"]) == TARGET_SHOT
    )
    mast_case, context = _mast_case_from_selection(store, *selected)
    _assert_reference(mast_case["reference"])
    passive_case, profile, policy = _passive_inclusive_case(mast_case, context, None)
    target_current = abs(float(mast_case["reference"]["plasma_current_a"]))
    solve, _trace, branch = _passive_inclusive_solve(
        passive_case,
        context,
        profile,
        newton_budget=NEWTON_STEPS,
        target_current=target_current,
    )
    root = solve["forward_branch_receipt"]
    terminal = solve["terminal_state"]
    if not np.isclose(root["residual"], EXPECTED_ROOT["residual"], rtol=1e-12):
        raise RuntimeError("the constrained root residual did not reproduce")
    if float(terminal["plasma_current_a"]) != EXPECTED_ROOT["plasma_current_a"]:
        raise RuntimeError("the constrained root current did not reproduce exactly")
    if not np.isclose(
        terminal["normalisation_amplitude"],
        EXPECTED_ROOT["amplitude"],
        rtol=0.0,
        atol=1e-15,
    ):
        raise RuntimeError("the constrained root amplitude did not reproduce")

    attribution, fields = _lcfs_attribution(
        profile, branch.equilibrium, context["group"], context["row"]
    )
    state = np.asarray(branch.equilibrium.flux, dtype=np.float64)
    receipt = {
        "receipt": "converged constrained-root geometry attribution",
        "backend": "JAX_PLATFORMS=cpu",
        "reference": mast_case["reference"],
        "constrained_root": {
            "terminal_residual": float(root["residual"]),
            "registered_fixed_point_criterion": FIXED_POINT_CRITERION,
            "terminal_plasma_current_a": float(terminal["plasma_current_a"]),
            "target_current_a": target_current,
            "recovered_amplitude": float(terminal["normalisation_amplitude"]),
            "terminal_state": {
                "dtype": "float64",
                "value_count": int(state.size),
                "sha256": _sha256_array(state),
                "values": state.tolist(),
            },
        },
        "lcfs_shape_match_attribution": attribution,
        "reuse": {
            "map": "scripts/constrained_parity_reuse/reuse-report.md",
            "symbols": [
                "select_slices_by_shot",
                "_mast_case_from_selection",
                "_passive_inclusive_case",
                "_passive_inclusive_solve",
                "_contour",
                "_symmetric_mean_distance",
                "_stored_lcfs",
            ],
        },
        "banked_artifact_integrity": {
            "directory": str(output),
            "banked_file_count": len(baseline),
            "verified_digest_count": len(baseline),
            "before_equals_after": True,
            "sha256": baseline,
        },
        "passive_inclusive_policy": {
            key: policy[key]
            for key in (
                "stored_circuit_count",
                "active_circuit_count",
                "passive_or_vessel_circuit_count",
                "response_shape",
                "stored_field_closure_sup_wb",
            )
        },
    }
    _plot_contours(fields, attribution, output / FIGURE_NAME)
    if _banked_digests(output) != baseline:
        raise RuntimeError("a banked parity artifact changed during measurement")
    (output / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2) + "\n")
    if _banked_digests(output) != baseline:
        raise RuntimeError("a banked parity artifact changed while writing the receipt")
    return receipt


def main() -> None:
    """Run the LCFS attribution and print its quantitative verdict."""
    receipt = run_lcfs_attribution()
    attribution = receipt["lcfs_shape_match_attribution"]
    aggregate = attribution["aggregate_longest_branch"]["distance"][
        "symmetric_mean_distance_m"
    ]
    restricted = attribution["closed_branch_enclosing_solved_axis"]["distance"][
        "symmetric_mean_distance_m"
    ]
    print(
        "LCFS_ATTRIBUTION "
        f"classification={attribution['classification']} "
        f"aggregate_m={aggregate:.12g} restricted_m={restricted:.12g} "
        f"ratio={attribution['aggregate_over_closed_branch_distance_ratio']:.6g}"
    )


if __name__ == "__main__":
    main()
