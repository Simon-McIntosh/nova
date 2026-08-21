"""Retest whether omitted DIII-D ohmic currents follow ECOILA deterministically.

The measurement uses every common sample in the located IMAS netCDF pulse.
Each target receives one through-origin gain and, in the final comparison, one
global polarity choice.  No offset, time-local coefficient, or smoothing is
allowed.  The physical verdict is based on maximum trace error as a percentage
of the measured ohmic-family peak, not on near-bitwise proportionality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ENTRY = Path("/home/ITER/tribolp/Public/imasdb/DIII-D/200000.nc")
DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/ohmic-circuit")
DEFAULT_PRODUCER_RECEIPT = Path(
    "docs/figures/diiid-forward-onboarding/producer-currents/"
    "producer_current_scaling_receipt.json"
)
DEFAULT_POLARITY_RECEIPT = Path(
    "docs/figures/diiid-forward-onboarding/current-polarity/"
    "current_polarity_audit_receipt.json"
)
RECEIPT_NAME = "ohmic_circuit_retest_receipt.json"
FIGURE_NAME = "ohmic_circuit_retest.png"
DD_VERSION = "3.41.0"

REFERENCE_COIL = "ECOILA"
TARGET_COILS = ("ECOILB", "E567UP", "E567DN", "E89UP", "E89DN")
OHMIC_COILS = (REFERENCE_COIL, *TARGET_COILS)
EXPECTED_SAMPLE_COUNT = 480_256
OHMIC_PEAK_REFERENCE_A = 55_483.703125
DETERMINISTIC_MAXIMUM_RESIDUAL_PERCENT = 1.0


def _correlation(reference: np.ndarray, target: np.ndarray) -> float:
    reference_centered = reference - np.mean(reference)
    target_centered = target - np.mean(target)
    denominator = float(
        np.linalg.norm(reference_centered) * np.linalg.norm(target_centered)
    )
    if denominator == 0.0:
        raise ValueError("correlation requires non-constant traces")
    return float(np.dot(reference_centered, target_centered) / denominator)


def relationship_metrics(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    scale: float | None = None,
    positive_scale_only: bool = False,
) -> dict[str, float]:
    """Return one through-origin relationship and its full-trace residuals."""

    reference = np.asarray(reference, dtype=float)
    target = np.asarray(target, dtype=float)
    if reference.shape != target.shape or reference.ndim != 1:
        raise ValueError("relationship traces must be equal-length vectors")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(target)):
        raise ValueError("relationship traces must be finite")
    target_energy = float(np.dot(target, target))
    reference_energy = float(np.dot(reference, reference))
    if target_energy <= 0.0 or reference_energy <= 0.0:
        raise ValueError("relationship traces must have nonzero energy")
    if scale is None:
        scale = float(np.dot(reference, target) / reference_energy)
        if positive_scale_only:
            scale = max(0.0, scale)
    residual = target - scale * reference
    return {
        "scale_target_from_ecoila": float(scale),
        "relative_l2_residual": float(
            np.linalg.norm(residual) / np.sqrt(target_energy)
        ),
        "maximum_absolute_residual": float(np.max(np.abs(residual))),
        "rms_residual": float(np.sqrt(np.mean(np.square(residual)))),
        "scale_free_correlation": _correlation(reference, target),
    }


def _basis_receipt(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    target_to_current: float,
) -> dict[str, Any]:
    before = relationship_metrics(reference, target, scale=1.0)
    gain_only = relationship_metrics(reference, target, positive_scale_only=True)
    gain_and_polarity = relationship_metrics(reference, target)
    for result in (before, gain_only, gain_and_polarity):
        equivalent_current = result["maximum_absolute_residual"] * target_to_current
        result["maximum_equivalent_current_residual_A"] = equivalent_current
        result["maximum_residual_percent_of_ohmic_peak"] = (
            100.0 * equivalent_current / OHMIC_PEAK_REFERENCE_A
        )
    return {
        "before_fault_accounting": before,
        "family_gain_accounted_positive_scale": gain_only,
        "family_gain_and_polarity_accounted_signed_scale": gain_and_polarity,
    }


def evaluate_coil(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    reference_turns: float,
    target_turns: float,
    common_family_scale: float = 1.0,
) -> dict[str, Any]:
    """Evaluate one omitted coil on current, turn-normalized, and ampere-turn bases."""

    current_basis = _basis_receipt(reference, target, target_to_current=1.0)
    per_turn_basis = _basis_receipt(
        reference / abs(reference_turns),
        target / abs(target_turns),
        target_to_current=abs(target_turns),
    )
    ampere_turn_basis = _basis_receipt(
        reference * reference_turns,
        target * target_turns,
        target_to_current=1.0 / abs(target_turns),
    )
    final = current_basis["family_gain_and_polarity_accounted_signed_scale"]
    common_scaled = relationship_metrics(
        common_family_scale * reference, common_family_scale * target
    )
    deterministic = (
        final["maximum_residual_percent_of_ohmic_peak"]
        <= DETERMINISTIC_MAXIMUM_RESIDUAL_PERCENT
    )
    return {
        "declared_turns": {
            "ecoila": reference_turns,
            "target": target_turns,
        },
        "current_A_basis": current_basis,
        "current_per_declared_turn_basis": per_turn_basis,
        "ampere_turn_basis": ampere_turn_basis,
        "fault_accounting": {
            "gain_magnitude": abs(final["scale_target_from_ecoila"]),
            "polarity": 1 if final["scale_target_from_ecoila"] >= 0.0 else -1,
            "time_local_freedom": False,
            "offset_fitted": False,
            "common_family_multiplier_check": {
                "multiplier_applied_to_both_ohmic_traces": common_family_scale,
                "scale_target_from_ecoila": common_scaled["scale_target_from_ecoila"],
                "relative_l2_residual": common_scaled["relative_l2_residual"],
                "relative_l2_difference_from_unscaled": (
                    common_scaled["relative_l2_residual"]
                    - final["relative_l2_residual"]
                ),
                "maximum_equivalent_current_residual_A": (
                    common_scaled["maximum_absolute_residual"]
                    / abs(common_family_scale)
                ),
            },
        },
        "verdict": {
            "deterministic_function_of_ecoila": deterministic,
            "criterion": (
                "maximum absolute residual after one signed through-origin scale "
                f"is at most {DETERMINISTIC_MAXIMUM_RESIDUAL_PERCENT:.1f}% of "
                f"the {OHMIC_PEAK_REFERENCE_A:.6f} A measured ohmic peak"
            ),
            "residual_bound_A": (
                DETERMINISTIC_MAXIMUM_RESIDUAL_PERCENT * OHMIC_PEAK_REFERENCE_A / 100.0
            ),
            "observed_maximum_residual_A": final[
                "maximum_equivalent_current_residual_A"
            ],
            "observed_percent_of_ohmic_peak": final[
                "maximum_residual_percent_of_ohmic_peak"
            ],
        },
    }


def _effective_turns(coil: Any) -> float:
    turns = [
        float(coil.element[index].turns_with_sign) for index in range(len(coil.element))
    ]
    if not turns:
        raise ValueError(f"{coil.name}: no declared conductor elements")
    return float(np.sum(turns))


def _fault_context(
    producer_receipt_path: Path, polarity_receipt_path: Path
) -> dict[str, Any]:
    producer = json.loads(producer_receipt_path.read_text(encoding="utf-8"))
    polarity = json.loads(polarity_receipt_path.read_text(encoding="utf-8"))
    producer_scale = float(
        producer["single_scalar_reconciliation"]["best_fit"]["scale"]
    )
    census = polarity["full_corpus_census"]
    return {
        "producer_family_scale_receipt": str(producer_receipt_path),
        "landed_shaping_to_ohmic_candidate_scale": producer_scale,
        "family_scale_treatment": (
            "All six traces tested here belong to the ohmic family. A common "
            "family multiplier therefore cancels exactly from fitted scale and "
            "relative residual; per-target gain is still measured explicitly."
        ),
        "corpus_polarity_receipt": str(polarity_receipt_path),
        "corpus_affected_shot_count": int(census["affected_shot_count"]),
        "corpus_total_shot_count": int(census["shot_count"]),
        "polarity_treatment": (
            "The polarity census describes the competition corpus, not this netCDF "
            "pulse. Each target is nevertheless allowed one signed scale so an "
            "opposed recording cannot be misclassified as non-deterministic."
        ),
    }


def build_receipt(
    entry_path: Path,
    producer_receipt_path: Path,
    polarity_receipt_path: Path,
) -> dict[str, Any]:
    """Read the IMAS pulse and evaluate all five omitted ohmic coils."""

    import imas

    with imas.DBEntry(entry_path, "r", dd_version=DD_VERSION) as entry:
        active = entry.get("pf_active", autoconvert=False)
        written_dd = str(active.ids_properties.version_put.data_dictionary)
        if written_dd != DD_VERSION:
            raise RuntimeError(f"expected DD {DD_VERSION}, read {written_dd}")
        by_name = {str(coil.name): coil for coil in active.coil}
        if any(name not in by_name for name in OHMIC_COILS):
            raise RuntimeError("the six-coil ohmic family is incomplete")

        currents = {
            name: np.asarray(by_name[name].current.data, dtype=float)
            for name in OHMIC_COILS
        }
        times = {
            name: np.asarray(by_name[name].current.time, dtype=float)
            for name in OHMIC_COILS
        }
        turns = {name: _effective_turns(by_name[name]) for name in OHMIC_COILS}
        units = {str(by_name[name].current.data.metadata.units) for name in OHMIC_COILS}
        if units != {"A"}:
            raise RuntimeError(f"unexpected current units {sorted(units)}")
        reference_time = times[REFERENCE_COIL]
        if reference_time.size != EXPECTED_SAMPLE_COUNT:
            raise RuntimeError(
                f"expected {EXPECTED_SAMPLE_COUNT} samples, found {reference_time.size}"
            )
        if any(
            time.shape != reference_time.shape
            or not np.array_equal(time, reference_time)
            for time in times.values()
        ):
            raise RuntimeError("ohmic coils do not share an identical time base")
        if any(not np.all(np.isfinite(values)) for values in currents.values()):
            raise RuntimeError("ohmic current traces contain non-finite samples")

    measured_peak = float(max(np.max(np.abs(values)) for values in currents.values()))
    if not np.isclose(measured_peak, OHMIC_PEAK_REFERENCE_A, rtol=0.0, atol=1.0e-9):
        raise RuntimeError(
            f"ohmic peak changed from {OHMIC_PEAK_REFERENCE_A} to {measured_peak} A"
        )
    fault_context = _fault_context(producer_receipt_path, polarity_receipt_path)
    common_family_scale = fault_context["landed_shaping_to_ohmic_candidate_scale"]
    results = {
        name: evaluate_coil(
            currents[REFERENCE_COIL],
            currents[name],
            reference_turns=turns[REFERENCE_COIL],
            target_turns=turns[name],
            common_family_scale=common_family_scale,
        )
        for name in TARGET_COILS
    }
    recovered = [
        name
        for name, result in results.items()
        if result["verdict"]["deterministic_function_of_ecoila"]
    ]
    return {
        "measurement": "DIII-D omitted-ohmic current determinism against ECOILA",
        "source": str(entry_path),
        "backend": "imas-python netCDF",
        "dd_version": DD_VERSION,
        "ids_read": ["pf_active"],
        "ids_not_read": ["equilibrium", "magnetics"],
        "declaration_before_comparison": {
            "reference_coil": REFERENCE_COIL,
            "target_coils": list(TARGET_COILS),
            "sample_count_required": EXPECTED_SAMPLE_COUNT,
            "fit": (
                "one through-origin least-squares gain per target over the full "
                "common time base; the final fit permits one global polarity sign"
            ),
            "per_time_freedom": False,
            "offset_fitted": False,
            "deterministic_maximum_residual_percent_of_ohmic_peak": (
                DETERMINISTIC_MAXIMUM_RESIDUAL_PERCENT
            ),
            "ohmic_peak_reference_A": OHMIC_PEAK_REFERENCE_A,
            "maximum_residual_bound_A": (
                DETERMINISTIC_MAXIMUM_RESIDUAL_PERCENT * OHMIC_PEAK_REFERENCE_A / 100.0
            ),
        },
        "common_time_base": {
            "sample_count": int(reference_time.size),
            "start_s": float(reference_time[0]),
            "end_s": float(reference_time[-1]),
            "maximum_elementwise_difference_s": 0.0,
        },
        "basis_definitions": {
            "current_A_basis": "stored current in declared amperes",
            "current_per_declared_turn_basis": (
                "stored current divided by the absolute sum of declared "
                "turns_with_sign for that coil"
            ),
            "ampere_turn_basis": (
                "stored current multiplied by the signed sum of declared "
                "turns_with_sign for that coil"
            ),
            "verdict_conversion": (
                "turn-normalized and ampere-turn residuals are converted back to "
                "equivalent target current before comparison with the physical bound"
            ),
        },
        "fault_context": fault_context,
        "per_coil": results,
        "summary": {
            "deterministic_coils": recovered,
            "non_deterministic_coils": [
                name for name in TARGET_COILS if name not in recovered
            ],
            "deterministic_count": len(recovered),
            "target_count": len(TARGET_COILS),
            "all_five_recoverable_from_ecoila": len(recovered) == len(TARGET_COILS),
            "verdict": (
                "all_five_recoverable"
                if len(recovered) == len(TARGET_COILS)
                else "only_a_subset_is_recoverable"
            ),
        },
    }


def plot_receipt(receipt: dict[str, Any], path: Path) -> None:
    """Plot the residual decision and the fitted current/turn scales."""

    names = list(receipt["per_coil"])
    before = [
        receipt["per_coil"][name]["current_A_basis"]["before_fault_accounting"][
            "maximum_residual_percent_of_ohmic_peak"
        ]
        for name in names
    ]
    after = [
        receipt["per_coil"][name]["current_A_basis"][
            "family_gain_and_polarity_accounted_signed_scale"
        ]["maximum_residual_percent_of_ohmic_peak"]
        for name in names
    ]
    current_scale = [
        receipt["per_coil"][name]["current_A_basis"][
            "family_gain_and_polarity_accounted_signed_scale"
        ]["scale_target_from_ecoila"]
        for name in names
    ]
    per_turn_scale = [
        receipt["per_coil"][name]["current_per_declared_turn_basis"][
            "family_gain_and_polarity_accounted_signed_scale"
        ]["scale_target_from_ecoila"]
        for name in names
    ]

    position = np.arange(len(names))
    width = 0.36
    figure, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True)
    axes[0].bar(position - width / 2, before, width, label="fixed scale +1")
    axes[0].bar(position + width / 2, after, width, label="gain + polarity fitted")
    axes[0].axhline(
        DETERMINISTIC_MAXIMUM_RESIDUAL_PERCENT,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="deterministic bound",
    )
    axes[0].set_ylabel("maximum residual [% of ohmic peak]")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].scatter(position, current_scale, label="current basis", marker="o")
    axes[1].scatter(position, per_turn_scale, label="current / turns", marker="s")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("signed target / ECOILA scale")
    axes[1].set_xticks(position, names)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def write_outputs(receipt: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / RECEIPT_NAME
    figure_path = output_dir / FIGURE_NAME
    plot_receipt(receipt, figure_path)
    receipt["artifacts"] = {"figure": str(figure_path)}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt_path, figure_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", type=Path, default=DEFAULT_ENTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--producer-receipt", type=Path, default=DEFAULT_PRODUCER_RECEIPT
    )
    parser.add_argument(
        "--polarity-receipt", type=Path, default=DEFAULT_POLARITY_RECEIPT
    )
    arguments = parser.parse_args()
    receipt = build_receipt(
        arguments.entry, arguments.producer_receipt, arguments.polarity_receipt
    )
    receipt_path, figure_path = write_outputs(receipt, arguments.output_dir)
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "figure": str(figure_path),
                **receipt["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
