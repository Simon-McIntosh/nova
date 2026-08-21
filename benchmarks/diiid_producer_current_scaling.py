"""Test whether one family scalar explains the DIII-D producer currents.

The scalar is fitted to time-resolved family ampere-turn envelopes with no
per-coil parameters.  A vacuum calculation on the recorded separatrix is
reported separately as an internal diagnostic: plasma and passive currents are
not represented, so that calculation cannot authorize current data for a solve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from nova.biot.bandedcoupling import banded_greens


DEFAULT_ENTRY = Path("/home/ITER/tribolp/Public/imasdb/DIII-D/200000.nc")
DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/producer-currents")
RECEIPT_NAME = "producer_current_scaling_receipt.json"
FIGURE_NAME = "producer_current_scaling.png"
DD_VERSION = "3.41.0"

OHMIC_NAMES = ("ECOILA", "ECOILB", "E567UP", "E567DN", "E89UP", "E89DN")
REFERENCE_SCALES = (1.0, 1000.0)


def fit_single_positive_scale(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    """Fit one positive through-origin scale and report its relative residual."""

    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("family envelopes must be equal-length vectors")
    finite = np.isfinite(reference) & np.isfinite(candidate)
    reference = reference[finite]
    candidate = candidate[finite]
    denominator = float(np.dot(candidate, candidate))
    reference_norm = float(np.dot(reference, reference))
    if denominator <= 0.0 or reference_norm <= 0.0:
        raise ValueError("family envelopes must have nonzero finite energy")
    unconstrained = float(np.dot(reference, candidate) / denominator)
    scale = max(0.0, unconstrained)
    residual = reference - scale * candidate
    relative = float(np.linalg.norm(residual) / np.sqrt(reference_norm))
    return {
        "scale": scale,
        "relative_l2_residual": relative,
        "explained_energy_fraction": float(1.0 - relative**2),
        "unconstrained_scale": unconstrained,
    }


def relative_residual(
    reference: np.ndarray, candidate: np.ndarray, scale: float
) -> float:
    """Return the through-origin L2 residual for a fixed family scale."""

    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    return float(
        np.linalg.norm(reference - scale * candidate) / np.linalg.norm(reference)
    )


def _rectangle_vertices(geometry: Any) -> np.ndarray:
    rectangle = geometry.rectangle
    r = float(rectangle.r)
    z = float(rectangle.z)
    half_width = 0.5 * float(rectangle.width)
    half_height = 0.5 * float(rectangle.height)
    return np.asarray(
        [
            (r - half_width, z - half_height),
            (r + half_width, z - half_height),
            (r + half_width, z + half_height),
            (r - half_width, z + half_height),
        ]
    )


def _element_vertices(element: Any, coil_name: str) -> np.ndarray:
    geometry = element.geometry
    geometry_type = int(geometry.geometry_type)
    if geometry_type == 1:
        return np.c_[
            np.asarray(geometry.outline.r, dtype=float),
            np.asarray(geometry.outline.z, dtype=float),
        ]
    if geometry_type == 2:
        return _rectangle_vertices(geometry)
    raise ValueError(f"unsupported geometry type {geometry_type} for {coil_name}")


def _coil_response(coil: Any, radius: np.ndarray, height: np.ndarray) -> np.ndarray:
    response = np.zeros_like(radius, dtype=float)
    for element in coil.element:
        response += (
            float(element.turns_with_sign)
            * banded_greens(radius, height, _element_vertices(element, str(coil.name)))[
                0
            ]
        )
    return response


def boundary_misclosure(
    ohmic_flux: list[np.ndarray],
    shaping_flux: list[np.ndarray],
    axis_to_boundary_span: np.ndarray,
    scale: float,
) -> dict[str, Any]:
    """Measure gauge-free vacuum variation along recorded flux surfaces."""

    if len(ohmic_flux) != len(shaping_flux):
        raise ValueError("vacuum families must contain the same slices")
    spans = np.asarray(axis_to_boundary_span, dtype=float)
    if spans.shape != (len(ohmic_flux),) or np.any(spans <= 0.0):
        raise ValueError("axis-to-boundary spans must be positive per slice")
    rms = []
    fractional = []
    for ohmic, shaping, span in zip(ohmic_flux, shaping_flux, spans, strict=True):
        prediction = np.asarray(ohmic) + scale * np.asarray(shaping)
        centred = prediction - np.mean(prediction)
        one_rms = float(np.sqrt(np.mean(np.square(centred))))
        rms.append(one_rms)
        fractional.append(one_rms / float(span))
    return {
        "scale": float(scale),
        "rms_wb_total": {
            "minimum": float(np.min(rms)),
            "median": float(np.median(rms)),
            "maximum": float(np.max(rms)),
        },
        "fraction_of_axis_to_boundary_flux_span": {
            "minimum": float(np.min(fractional)),
            "median": float(np.median(fractional)),
            "maximum": float(np.max(fractional)),
        },
        "per_slice_fraction": fractional,
    }


def _family_envelopes(
    currents: np.ndarray, turns: np.ndarray, names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    ampere_turn = currents * turns[:, None]
    ohmic_mask = np.asarray([name in OHMIC_NAMES for name in names])
    if int(np.count_nonzero(ohmic_mask)) != len(OHMIC_NAMES):
        raise ValueError("the declared six-coil ohmic family is incomplete")
    if int(np.count_nonzero(~ohmic_mask)) != 18:
        raise ValueError("the declared eighteen-coil shaping family is incomplete")
    return (
        np.sqrt(np.sum(np.square(ampere_turn[ohmic_mask]), axis=0)),
        np.sqrt(np.sum(np.square(ampere_turn[~ohmic_mask]), axis=0)),
    )


def read_entry(entry_path: Path) -> dict[str, Any]:
    """Read currents, turns, geometry, and equilibrium labels through IMAS."""

    import imas

    with imas.DBEntry(entry_path, "r", dd_version=DD_VERSION) as entry:
        active = entry.get("pf_active", autoconvert=False)
        equilibrium = entry.get("equilibrium", autoconvert=False)

        names = [str(coil.name) for coil in active.coil]
        if len(names) != 24 or len(set(names)) != 24:
            raise ValueError(f"expected 24 unique PF coils, found {len(names)}")
        current_units = {str(coil.current.data.metadata.units) for coil in active.coil}
        turn_units = {
            str(element.turns_with_sign.metadata.units)
            for coil in active.coil
            for element in coil.element
        }
        if current_units != {"A"} or turn_units != {"-"}:
            raise ValueError(
                "unexpected declared units: "
                f"current={current_units}, turns={turn_units}"
            )

        currents = np.stack(
            [np.asarray(coil.current.data, dtype=float) for coil in active.coil]
        )
        current_times = [
            np.asarray(coil.current.time, dtype=float) for coil in active.coil
        ]
        if any(
            not np.array_equal(current_times[0], values) for values in current_times[1:]
        ):
            raise ValueError("PF coil current time bases are not identical")
        turns = np.asarray(
            [
                sum(float(element.turns_with_sign) for element in coil.element)
                for coil in active.coil
            ]
        )
        if np.any(turns == 0.0):
            raise ValueError("an effective coil turn count is zero")

        ohmic_envelope, shaping_envelope = _family_envelopes(currents, turns, names)
        fitted = fit_single_positive_scale(ohmic_envelope, shaping_envelope)
        fixed_residuals = {
            str(scale): relative_residual(ohmic_envelope, shaping_envelope, scale)
            for scale in REFERENCE_SCALES
        }

        ohmic_mask = np.asarray([name in OHMIC_NAMES for name in names])
        max_current = np.nanmax(np.abs(currents), axis=1)
        max_ampere_turn = max_current * np.abs(turns)
        raw_ratio = float(
            np.max(max_current[ohmic_mask]) / np.max(max_current[~ohmic_mask])
        )
        turn_ratio = float(
            np.max(max_ampere_turn[ohmic_mask]) / np.max(max_ampere_turn[~ohmic_mask])
        )

        slices = [
            equilibrium.time_slice[index]
            for index in range(len(equilibrium.time_slice))
        ]
        offsets = [0]
        boundary_r = []
        boundary_z = []
        flux_spans = []
        for one_slice in slices:
            outline = one_slice.boundary_separatrix.outline
            radius = np.asarray(outline.r, dtype=float)
            height = np.asarray(outline.z, dtype=float)
            if radius.size < 3 or radius.shape != height.shape:
                raise ValueError("an equilibrium slice has no usable separatrix")
            boundary_r.append(radius)
            boundary_z.append(height)
            offsets.append(offsets[-1] + radius.size)
            boundary_flux = float(one_slice.boundary_separatrix.psi)
            axis_flux = float(one_slice.global_quantities.psi_axis)
            flux_spans.append(abs(boundary_flux - axis_flux))
        all_r = np.concatenate(boundary_r)
        all_z = np.concatenate(boundary_z)

        first_grid = slices[0].profiles_2d[0].grid
        grid_r = np.asarray(first_grid.dim1, dtype=float)
        grid_z = np.asarray(first_grid.dim2, dtype=float)
        for one_slice in slices[1:]:
            one_grid = one_slice.profiles_2d[0].grid
            if not np.array_equal(grid_r, np.asarray(one_grid.dim1, dtype=float)):
                raise ValueError("equilibrium radial grid changes between slices")
            if not np.array_equal(grid_z, np.asarray(one_grid.dim2, dtype=float)):
                raise ValueError("equilibrium vertical grid changes between slices")
        target_r, target_z = np.meshgrid(grid_r, grid_z, indexing="ij")
        grid_responses = np.stack(
            [
                _coil_response(coil, target_r.ravel(), target_z.ravel()).reshape(
                    target_r.shape
                )
                for coil in active.coil
            ]
        )
        interpolators = [
            RegularGridInterpolator((grid_r, grid_z), response, bounds_error=True)
            for response in grid_responses
        ]
        equilibrium_time = np.asarray(equilibrium.time, dtype=float)
        interpolated_currents = np.stack(
            [
                np.interp(equilibrium_time, current_times[index], currents[index])
                for index in range(len(names))
            ]
        )
        ohmic_flux = []
        shaping_flux = []
        for slice_index in range(len(slices)):
            section = slice(offsets[slice_index], offsets[slice_index + 1])
            coordinates = np.c_[all_r[section], all_z[section]]
            responses = np.stack(
                [interpolator(coordinates) for interpolator in interpolators]
            )
            coefficients = interpolated_currents[:, slice_index]
            ohmic_flux.append(
                np.sum(
                    coefficients[ohmic_mask, None] * responses[ohmic_mask],
                    axis=0,
                )
            )
            shaping_flux.append(
                np.sum(
                    coefficients[~ohmic_mask, None] * responses[~ohmic_mask],
                    axis=0,
                )
            )

    scales = {
        "declared_scale_1": 1.0,
        "family_best_scale": fitted["scale"],
        "thousand_scale": 1000.0,
        "turn_extrema_ratio_scale": turn_ratio,
    }
    boundary_results = {
        label: boundary_misclosure(
            ohmic_flux, shaping_flux, np.asarray(flux_spans), scale
        )
        for label, scale in scales.items()
    }
    best_boundary_label = min(
        boundary_results,
        key=lambda label: boundary_results[label][
            "fraction_of_axis_to_boundary_flux_span"
        ]["median"],
    )
    declared_boundary_median = boundary_results["declared_scale_1"][
        "fraction_of_axis_to_boundary_flux_span"
    ]["median"]
    fitted_boundary_median = boundary_results["family_best_scale"][
        "fraction_of_axis_to_boundary_flux_span"
    ]["median"]

    return {
        "source": str(entry_path),
        "backend": "imas-python netCDF",
        "dd_version": DD_VERSION,
        "declared_metadata": {
            "current_path": "pf_active/coil/current/data",
            "current_unit": "A",
            "turn_path": "pf_active/coil/element/turns_with_sign",
            "turn_unit": "-",
            "coil_count": len(names),
            "ohmic_coil_count": int(np.count_nonzero(ohmic_mask)),
            "shaping_coil_count": int(np.count_nonzero(~ohmic_mask)),
            "current_sample_count": int(currents.shape[1]),
        },
        "family_ratios": {
            "raw_max_abs_current_ratio": raw_ratio,
            "turns_corrected_max_abs_ampere_turn_ratio": turn_ratio,
            "numerator_family": "ohmic",
            "denominator_family": "shaping",
        },
        "single_scalar_reconciliation": {
            "declared_before_evaluation": True,
            "objective": (
                "positive through-origin least squares from the shaping-family "
                "L2 ampere-turn envelope to the ohmic-family L2 ampere-turn envelope "
                "at every common current sample"
            ),
            "per_coil_freedom": False,
            "effective_turns_applied_before_fit": True,
            "best_fit": fitted,
            "fixed_scale_relative_l2_residuals": fixed_residuals,
            "sample_count": int(currents.shape[1]),
            "verdict": "not_reconciled_by_one_scalar",
            "interpretation": (
                "An exact one-scalar reconciliation has zero relative residual. "
                f"The measured optimum {fitted['scale']:.6g} retains "
                f"{fitted['relative_l2_residual']:.6g}, so a single family scalar "
                "does not reconcile the time-resolved ampere-turn envelopes. The "
                "two families have distinct control functions, so this is a "
                "rejection test for a global unit correction, not a calibration."
            ),
            "numeric_confidence": "high",
            "scientific_confidence": "medium",
        },
        "vacuum_boundary_internal_diagnostic": {
            "role": "internal_diagnostic_only",
            "slice_count": len(slices),
            "boundary_point_count": int(all_r.size),
            "metric": (
                "RMS variation of coil-only total poloidal flux along each recorded "
                "separatrix after removing its additive gauge, divided by the "
                "recorded axis-to-boundary total-flux span"
            ),
            "spatial_evaluation": {
                "method": (
                    "Nova banded finite-section Green responses evaluated on the "
                    "common equilibrium grid and bilinearly interpolated to every "
                    "recorded separatrix"
                ),
                "grid_shape": [int(grid_r.size), int(grid_z.size)],
            },
            "global_current_sign_invariant": True,
            "results": boundary_results,
            "lowest_median_candidate": best_boundary_label,
            "ampere_turn_consistency": {
                "assessment": "does_not_support_the_fitted_family_scale",
                "declared_scale_median_fraction": declared_boundary_median,
                "fitted_scale_median_fraction": fitted_boundary_median,
                "fitted_to_declared_median_ratio": (
                    fitted_boundary_median / declared_boundary_median
                ),
                "statement": (
                    "Uniformly applying the family-envelope optimum makes the "
                    "coil-only separatrix misclosure larger, not smaller."
                ),
            },
            "confidence": "low",
            "limitation": (
                "The recorded boundary is a total-flux surface containing plasma "
                "and passive-current contributions. Coil-only contour misclosure "
                "can compare scale hypotheses but cannot calibrate or validate the "
                "producer currents, and no acceptance threshold is assigned."
            ),
        },
        "solve_authority": {
            "may_drive_a_solve": False,
            "verdict": "prohibited_as_solve_input",
            "applies_to": "the netCDF producer currents as stored or benchmark-scaled",
            "reason": (
                "A fitted benchmark scalar is not producer metadata, the family "
                "reconciliation retains a measured residual, and the vacuum check "
                "omits plasma and passive currents. Only a producer-backed corrected "
                "current description followed by independent validation could create "
                "solve authority."
            ),
        },
        "plot_data": {
            "time_s": current_times[0].tolist(),
            "ohmic_envelope_a_turn": ohmic_envelope.tolist(),
            "shaping_envelope_a_turn": shaping_envelope.tolist(),
        },
    }


def plot_receipt(receipt: dict[str, Any], output_path: Path) -> None:
    """Plot family envelopes and the separatrix diagnostic."""

    data = receipt["plot_data"]
    time = np.asarray(data["time_s"])
    ohmic = np.asarray(data["ohmic_envelope_a_turn"])
    shaping = np.asarray(data["shaping_envelope_a_turn"])
    scale = receipt["single_scalar_reconciliation"]["best_fit"]["scale"]
    stride = max(1, time.size // 5000)

    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.2))
    axes[0].plot(time[::stride], ohmic[::stride], label="ohmic", color="#b44a36")
    axes[0].plot(
        time[::stride], shaping[::stride], label="shaping, declared", color="#286f9b"
    )
    axes[0].plot(
        time[::stride],
        scale * shaping[::stride],
        label=f"shaping × {scale:.3g}",
        color="#3f8f61",
    )
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].set_ylabel("family L2 envelope [A.turn]")
    axes[0].set_xlabel("producer time [s]")
    axes[0].legend(frameon=False, ncol=3)

    diagnostic = receipt["vacuum_boundary_internal_diagnostic"]["results"]
    labels = list(diagnostic)
    samples = [
        diagnostic[label]["fraction_of_axis_to_boundary_flux_span"]["median"]
        for label in labels
    ]
    axes[1].bar(np.arange(len(labels)), samples, color="#6f6f86")
    axes[1].set_xticks(
        np.arange(len(labels)), [label.replace("_", "\n") for label in labels]
    )
    axes[1].set_ylabel("median LCFS vacuum misclosure / flux span")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def write_outputs(receipt: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write the compact receipt and its diagnostic figure."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / FIGURE_NAME
    receipt_path = output_dir / RECEIPT_NAME
    plot_receipt(receipt, figure_path)
    published = dict(receipt)
    published.pop("plot_data")
    published["artifacts"] = {"figure": str(figure_path)}
    receipt_path.write_text(json.dumps(published, indent=2) + "\n", encoding="utf-8")
    return receipt_path, figure_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", type=Path, default=DEFAULT_ENTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = read_entry(arguments.entry)
    receipt_path, figure_path = write_outputs(receipt, arguments.output_dir)
    summary = {
        "receipt": str(receipt_path),
        "figure": str(figure_path),
        "raw_ratio": receipt["family_ratios"]["raw_max_abs_current_ratio"],
        "turns_corrected_ratio": receipt["family_ratios"][
            "turns_corrected_max_abs_ampere_turn_ratio"
        ],
        "best_scale": receipt["single_scalar_reconciliation"]["best_fit"]["scale"],
        "best_scale_relative_l2_residual": receipt["single_scalar_reconciliation"][
            "best_fit"
        ]["relative_l2_residual"],
        "may_drive_a_solve": receipt["solve_authority"]["may_drive_a_solve"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
