"""Illustrate converged fixed-boundary DIII-D label re-solves.

The released coil channels do not define a nearby free-boundary root: the
root-existence receipt attributes 84.99 percent of the residual magnitude to
the boundary field.  This benchmark therefore shows the route that is
well-posed with the released data, the fixed-boundary relaxed Picard solve,
and labels the result as a demonstration rather than a production-gate pass.

Each selected map crosses the corpus flux convention once, supplies its own
extracted p-prime and FF-prime, and retains its labelled Dirichlet border.
Nothing is fitted and no current is adjusted.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from benchmarks.diiid_forward_gs_match import _LABEL_COLUMNS, _read
from benchmarks.diiid_label_resolve_gate import (
    FULL_CONVERGENCE_INITIAL_RELAXATION,
    FULL_CONVERGENCE_MAX_ITERATIONS,
    FULL_CONVERGENCE_MINIMUM_RELAXATION,
    PICARD_RELATIVE_TOLERANCE,
    RELAXATION_REDUCTION_INTERVAL,
    _frame_fields,
    _normalise_flux,
    _operator,
)
from nova.equilibrium.convention import grad_shafranov_source
from nova.equilibrium.map_extraction import extract_flux_functions


DEFAULT_DATA = Path("/work/projects/imas_gpu/sophelio/raw/data/diii_d_train")
DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/forward-gs")
ROOT_EXISTENCE_RECEIPT = Path(
    "docs/figures/diiid-forward-onboarding/root-existence/root_existence_receipt.json"
)
RECEIPT_NAME = "forward_gs_illustration_receipt.json"
FIGURE_NAME = "forward_gs_fixed_boundary_illustration.png"

LABEL_REPRESENTABILITY_CEILING = 0.0429
FREE_BOUNDARY_BOUNDARY_FIELD_SHARE = 0.8498610048171014
EXTRACTION_SURFACE_COUNT = 19
ROUTE = "fixed_boundary_relaxed_picard"
CAPTION = (
    "Converged fixed-boundary Nova GS re-solves from label-extracted p-prime "
    "and FF-prime. This is a demonstration, not a passed production gate: "
    "the released coil channels leave 0.8499 of the free-boundary residual "
    "attributable to the boundary field."
)
FIGURE_CAPTION = (
    "Converged fixed-boundary Nova GS re-solves from label-extracted p-prime "
    "and FF-prime.\n"
    "This is a demonstration, not a passed production gate: released coil "
    "channels leave 0.8499 of the free-boundary residual attributable to the "
    "boundary field."
)
NAMED_FRAMES = (
    ("d3d_shot_00000a10ac.parquet", 29),
    ("d3d_shot_00000c4a7b.parquet", 179),
    ("d3d_shot_0003ff34e7.parquet", 44),
)


@dataclass(frozen=True)
class FrameMetrics:
    """Convergence and label-match metrics for one illustrated frame."""

    shot: str
    frame: int
    time_ms: float
    route: str
    converged: bool
    convergence_criterion: str
    iterations: int
    final_relative_update: float
    final_fixed_point_relative_update: float
    final_relaxation: float
    reliable_extraction_surfaces: int
    interior_fractional_rms: float
    representability_ceiling: float
    fractional_rms_to_ceiling_ratio: float
    interior_r_squared: float


@dataclass(frozen=True)
class FrameMaps:
    """Plot-ready fields and receipt metrics for one frame."""

    metrics: FrameMetrics
    radius: np.ndarray
    height: np.ndarray
    predicted: np.ndarray
    labelled: np.ndarray
    labelled_lcfs: np.ndarray


def _r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    if denominator <= np.finfo(float).tiny:
        return float("nan")
    return float(1.0 - np.sum((actual - predicted) ** 2) / denominator)


def field_metrics(labelled: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Return the label-range fractional RMS and interior R-squared."""

    labelled = np.asarray(labelled, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    difference = predicted[1:-1, 1:-1] - labelled[1:-1, 1:-1]
    span = float(np.ptp(labelled))
    if span <= np.finfo(float).tiny:
        raise ValueError("labelled flux range is zero")
    fractional_rms = float(np.sqrt(np.mean(difference**2)) / span)
    return fractional_rms, _r_squared(labelled[1:-1, 1:-1], predicted[1:-1, 1:-1])


def boundary_field_share(path: Path = ROOT_EXISTENCE_RECEIPT) -> float:
    """Read and verify the pooled free-boundary attribution."""

    record = json.loads(path.read_text())
    share = float(record["result"]["pooled"]["attribution"]["vacuum_difference_share"])
    if not np.isclose(
        share, FREE_BOUNDARY_BOUNDARY_FIELD_SHARE, rtol=0.0, atol=5.0e-13
    ):
        raise RuntimeError("root-existence boundary-field share has changed")
    return share


def solve_frame(path: Path, frame: int) -> FrameMaps:
    """Converge one fixed-boundary profile solve without adjusting its sources."""

    row = _read(path, _LABEL_COLUMNS)
    operator = _operator(row["efit_grid_R"], row["efit_grid_Z"])
    labelled, contour, plasma_mask, label_span = _frame_fields(row, frame, operator)
    axis_point = np.array([[row["efit_r_axis"][frame], row["efit_z_axis"][frame]]])
    axis_flux = float(
        RegularGridInterpolator((operator.radius, operator.height), labelled)(
            axis_point
        )[0]
    )
    labelled_normalised = (labelled - axis_flux) / label_span
    surfaces = np.linspace(0.05, 0.95, EXTRACTION_SURFACE_COUNT)
    extraction = extract_flux_functions(
        operator.radius,
        operator.height,
        labelled,
        labelled_normalised,
        surfaces=surfaces,
        plasma_mask=plasma_mask,
        min_samples=6,
    )
    reliable = (
        extraction.reliable
        & np.isfinite(extraction.p_prime)
        & np.isfinite(extraction.ff_prime)
    )
    reliable_count = int(np.count_nonzero(reliable))
    if reliable_count < 2:
        raise ValueError(f"only {reliable_count} reliable extracted flux surfaces")

    surface = extraction.psi_norm[reliable]
    p_prime = extraction.p_prime[reliable]
    ff_prime = extraction.ff_prime[reliable]
    radius_map = np.broadcast_to(operator.radius[:, None], labelled.shape)
    predicted = np.array(labelled, copy=True)
    relaxation = FULL_CONVERGENCE_INITIAL_RELAXATION
    relative_update = float("inf")
    fixed_update = float("inf")
    for iteration in range(1, FULL_CONVERGENCE_MAX_ITERATIONS + 1):
        if iteration > 1 and (iteration - 1) % RELAXATION_REDUCTION_INTERVAL == 0:
            relaxation = max(FULL_CONVERGENCE_MINIMUM_RELAXATION, relaxation / 2.0)
        normalised, own_span = _normalise_flux(
            predicted,
            operator.radius,
            operator.height,
            contour,
            plasma_mask,
            np.sign(label_span),
        )
        active = plasma_mask & (normalised >= 0.0) & (normalised <= 1.0)
        evaluated_p = np.interp(normalised, surface, p_prime)
        evaluated_ff = np.interp(normalised, surface, ff_prime)
        source = np.zeros_like(labelled)
        source[active] = grad_shafranov_source(
            radius_map[active], evaluated_p[active], evaluated_ff[active]
        )
        solved = operator.solve(source, labelled)
        fixed_update = float(
            np.sqrt(np.mean((solved[1:-1, 1:-1] - predicted[1:-1, 1:-1]) ** 2))
            / abs(own_span)
        )
        predicted = relaxation * solved + (1.0 - relaxation) * predicted
        relative_update = relaxation * fixed_update
        if relative_update <= PICARD_RELATIVE_TOLERANCE:
            break

    converged = relative_update <= PICARD_RELATIVE_TOLERANCE
    fractional_rms, r_squared = field_metrics(labelled, predicted)
    metrics = FrameMetrics(
        shot=path.name,
        frame=frame,
        time_ms=float(row["efit_times"][frame]),
        route=ROUTE,
        converged=converged,
        convergence_criterion=(
            f"relaxed fractional update <= {PICARD_RELATIVE_TOLERANCE:.1e} "
            f"within {FULL_CONVERGENCE_MAX_ITERATIONS} iterations"
        ),
        iterations=iteration,
        final_relative_update=relative_update,
        final_fixed_point_relative_update=fixed_update,
        final_relaxation=relaxation,
        reliable_extraction_surfaces=reliable_count,
        interior_fractional_rms=fractional_rms,
        representability_ceiling=LABEL_REPRESENTABILITY_CEILING,
        fractional_rms_to_ceiling_ratio=(
            fractional_rms / LABEL_REPRESENTABILITY_CEILING
        ),
        interior_r_squared=r_squared,
    )
    return FrameMaps(
        metrics=metrics,
        radius=operator.radius,
        height=operator.height,
        predicted=predicted,
        labelled=labelled,
        labelled_lcfs=contour,
    )


def render(frames: list[FrameMaps], output: Path, share: float) -> Path:
    """Render predicted, labelled, and difference maps for each named frame."""

    if len(frames) < 3:
        raise ValueError("the illustration requires at least three named frames")
    figure, axes = plt.subplots(
        len(frames), 3, figsize=(11.8, 10.2), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)
    for row_index, frame in enumerate(frames):
        metric = frame.metrics
        flux_min = float(min(np.min(frame.predicted), np.min(frame.labelled)))
        flux_max = float(max(np.max(frame.predicted), np.max(frame.labelled)))
        flux_levels = np.linspace(flux_min, flux_max, 25)
        difference = frame.predicted - frame.labelled
        difference_limit = max(float(np.max(np.abs(difference))), 1.0e-12)
        difference_levels = np.linspace(-difference_limit, difference_limit, 25)
        panels = (
            (frame.predicted, flux_levels, "viridis", "Predicted"),
            (frame.labelled, flux_levels, "viridis", "Labelled"),
            (difference, difference_levels, "RdBu_r", "Predicted - labelled"),
        )
        metric_line = (
            f"frac RMS {metric.interior_fractional_rms:.4f} / "
            f"{LABEL_REPRESENTABILITY_CEILING:.4f} ceiling "
            f"({metric.fractional_rms_to_ceiling_ratio:.2f}x)"
        )
        for column_index, (field, levels, colour_map, title) in enumerate(panels):
            axis = axes[row_index, column_index]
            axis.contourf(
                frame.radius,
                frame.height,
                field.T,
                levels=levels,
                cmap=colour_map,
                extend="both",
            )
            axis.plot(
                frame.labelled_lcfs[:, 0],
                frame.labelled_lcfs[:, 1],
                color="#cc0000",
                linewidth=0.8,
            )
            axis.set_title(f"{title}\n{metric_line}", fontsize=8.5)
            axis.set_aspect("equal", adjustable="box")
            axis.tick_params(labelsize=8)
            if column_index == 0:
                shot = Path(metric.shot).stem
                axis.set_ylabel(
                    f"{shot}\nframe {metric.frame}, {metric.time_ms:.0f} ms\nZ [m]",
                    fontsize=8.5,
                )
            if row_index == len(frames) - 1:
                axis.set_xlabel("R [m]", fontsize=9)

    figure.suptitle(
        "DIII-D labelled maps re-solved by Nova on the convergent fixed boundary",
        fontsize=13,
        y=0.985,
    )
    figure.text(
        0.995,
        0.52,
        "Free-boundary context\n"
        f"boundary-field share = {share:.4f}\n"
        "(pooled residual attribution)\n\n"
        "Demonstration only -\nnot a passed production gate",
        ha="right",
        va="center",
        fontsize=9,
        color="#8b0000",
    )
    figure.text(0.5, 0.015, FIGURE_CAPTION, ha="center", va="bottom", fontsize=8.5)
    figure.subplots_adjust(left=0.12, right=0.82, top=0.93, bottom=0.1, wspace=0.18)
    output.mkdir(parents=True, exist_ok=True)
    path = output / FIGURE_NAME
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def receipt(frames: list[FrameMaps], share: float) -> dict[str, Any]:
    """Build the qualified, machine-readable illustration receipt."""

    metrics = [asdict(frame.metrics) for frame in frames]
    return {
        "measurement": "DIII-D converged fixed-boundary forward GS illustration",
        "route": {
            "name": ROUTE,
            "boundary_condition": "labelled Dirichlet border",
            "all_frames_converged": all(item["converged"] for item in metrics),
            "convergence_criterion": (
                f"relaxed fractional update <= {PICARD_RELATIVE_TOLERANCE:.1e}"
            ),
            "maximum_iterations": FULL_CONVERGENCE_MAX_ITERATIONS,
        },
        "comparison": {
            "label_representability_fractional_rms_ceiling": (
                LABEL_REPRESENTABILITY_CEILING
            ),
            "metric": "interior RMS divided by the full labelled flux range",
            "coefficients_fitted": 0,
        },
        "free_boundary_context": {
            "boundary_field_share": share,
            "displayed_share": round(share, 4),
            "source": str(ROOT_EXISTENCE_RECEIPT),
            "production_gate_passed": False,
            "reason": (
                "released conductor channels do not supply a nearby labelled "
                "free-boundary root"
            ),
        },
        "caption": CAPTION,
        "figure": FIGURE_NAME,
        "named_shot_frame_pairs": len(metrics),
        "frames": metrics,
    }


def run(data: Path, output: Path) -> dict[str, Any]:
    """Solve the named cohort and write its receipt and figure."""

    share = boundary_field_share()
    frames = [solve_frame(data / shot, frame) for shot, frame in NAMED_FRAMES]
    if not all(frame.metrics.converged for frame in frames):
        failed = [
            f"{frame.metrics.shot}:{frame.metrics.frame}"
            for frame in frames
            if not frame.metrics.converged
        ]
        raise RuntimeError(f"fixed-boundary illustration did not converge: {failed}")
    figure_path = render(frames, output, share)
    result = receipt(frames, share)
    result["figure"] = str(figure_path)
    output.mkdir(parents=True, exist_ok=True)
    (output / RECEIPT_NAME).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
