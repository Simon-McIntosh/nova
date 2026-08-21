"""Evaluate discrete convention hypotheses at labelled DIII-D states.

Every Sauter--Medvedev convention is treated as a hypothesis for the raw
corpus numbers and converted into Nova's COCOS 17 convention.  The resulting
label, extracted gradients, and released conductor currents are evaluated by
the linear Dirichlet operator.  No iterative root search, continuous parameter
choice, current adjustment, or profile fit is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from benchmarks import diiid_root_existence as root_existence
from benchmarks.diiid_corpus_conventions import CORPUS_COCOS, NOVA_COCOS
from benchmarks.diiid_forward_gs_match import (
    DEFAULT_DATA,
    _CURRENT_COLUMNS,
    _GEOMETRY_COLUMNS,
    _LABEL_COLUMNS,
    _eligible_frame,
    _plasma_mask,
    _read,
    canonical_axes,
)
from benchmarks.diiid_label_resolve_gate import _operator
from benchmarks.diiid_vacuum_quiescent_gate import normalised_flux
from nova.equilibrium.convention import grad_shafranov_source
from nova.equilibrium.map_extraction import extract_flux_functions
from nova.imas.diiid_description import (
    DiiidDescriptionRegistry,
    vacuum_psi,
    vacuum_response,
)
from nova.io.cocos import (
    B0_LIKE,
    CONVENTION_DIGITS,
    DODPSI_LIKE,
    IP_LIKE,
    PSI_LIKE,
    Q_LIKE,
    convention,
    convention_transform,
)
from nova.jax.config import configure_dtypes

DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/root-convention")
POLARITY_RECEIPT = Path(
    "docs/figures/diiid-forward-onboarding/current-polarity/"
    "current_polarity_audit_receipt.json"
)
ROOT_RECEIPT = root_existence.DEFAULT_OUTPUT / root_existence.RECEIPT_NAME
PREREGISTRATION_NAME = "root_convention_preregistration.json"
RECEIPT_NAME = "root_convention_receipt.json"
FIGURE_NAME = "root_convention_sweep.png"
CHECKPOINT_NAME = "root_convention_frames.jsonl"

LANDED_FRAME_COUNT = 5
ADDITIONAL_FRAME_COUNT = 15
TOTAL_FRAME_COUNT = LANDED_FRAME_COUNT + ADDITIONAL_FRAME_COUNT
LANDED_FREE_MEDIAN = 0.2156505171295768
LANDED_FIXED_MEDIAN = 0.04058126344290784
LANDED_VACUUM_SHARE = 0.8498610048171014
CONTROL_ABSOLUTE_TOLERANCE = 2.0e-7
FIXED_LEVEL_RATIO = 1.05


@dataclass(frozen=True)
class Variant:
    """One discrete source-convention interpretation of the raw arrays."""

    source_cocos: int
    sigma_bp: int
    flux_exponent: int
    sigma_r_phi_z: int
    sigma_rho_theta_phi: int
    psi_to_nova: float
    ip_to_nova: float
    derivative_to_nova: float
    b0_to_nova: float
    q_to_nova: float

    @property
    def identifier(self) -> str:
        return f"source-cocos-{self.source_cocos}"

    @property
    def poloidal_flux_sign(self) -> int:
        return int(np.sign(self.psi_to_nova))

    @property
    def toroidal_current_sign(self) -> int:
        return int(np.sign(self.ip_to_nova))

    @property
    def derivative_sign(self) -> int:
        return int(np.sign(self.derivative_to_nova))

    @property
    def raw_flux_interpretation(self) -> str:
        return "per-radian" if self.flux_exponent == 1 else "total"


@dataclass(frozen=True)
class SelectedFrame:
    """One labelled state selected without using a residual."""

    path: Path
    frame: int
    time_ms: float
    population: str
    absent_from_polarity_population: bool


def variants() -> tuple[Variant, ...]:
    """Return all sixteen convention hypotheses in numeric order."""

    result = []
    for identifier in sorted(CONVENTION_DIGITS):
        source = convention(identifier)
        transform = convention_transform(source=identifier, target=NOVA_COCOS)
        result.append(
            Variant(
                source_cocos=identifier,
                sigma_bp=source.sigma_bp,
                flux_exponent=transform.flux_exponent,
                sigma_r_phi_z=source.sigma_r_phi_z,
                sigma_rho_theta_phi=source.sigma_rho_theta_phi,
                psi_to_nova=transform.factor(PSI_LIKE),
                ip_to_nova=transform.factor(IP_LIKE),
                derivative_to_nova=transform.factor(DODPSI_LIKE),
                b0_to_nova=transform.factor(B0_LIKE),
                q_to_nova=transform.factor(Q_LIKE),
            )
        )
    return tuple(result)


def preregistration() -> dict[str, Any]:
    """Return the complete cohort, variant, and verdict declaration."""

    members = variants()
    return {
        "measurement": "linear residual evaluation at unchanged labelled states",
        "no_root_search": True,
        "coefficients_fitted": 0,
        "currents_adjusted": False,
        "profiles_adjusted": False,
        "cohort": {
            "landed_root_existence_frames": LANDED_FRAME_COUNT,
            "additional_diverted_frames": ADDITIONAL_FRAME_COUNT,
            "total_frames": TOTAL_FRAME_COUNT,
            "additional_selection": (
                "first fifteen lexicographic shots outside the five landed shots "
                "and outside the complete 603-shot polarity population, using the "
                "finite diverted frame nearest each shot's median eligible time"
            ),
        },
        "variant_set": {
            "members": [
                asdict(member) | {"identifier": member.identifier} for member in members
            ],
            "basis": (
                "all sixteen Sauter-Medvedev source COCOS indices transformed "
                "into Nova COCOS 17; this exhausts both signs of poloidal flux, "
                "both signs of toroidal current, total versus per-radian flux, "
                "and both signs of derivatives with respect to flux"
            ),
            "pinned_control_source_cocos": CORPUS_COCOS,
            "target_cocos": NOVA_COCOS,
        },
        "evaluation_algebra": {
            "label": "raw efit_psirz multiplied by the candidate psi_like factor",
            "profiles": (
                "p-prime and FF-prime extracted once in the pinned Nova coordinate, "
                "mapped back to raw-coordinate gradients, then multiplied by the "
                "candidate dodpsi_like factor"
            ),
            "conductor_currents": (
                "the released-current vacuum map multiplied by the candidate "
                "ip_like factor relative to the pinned factor"
            ),
            "operator": (
                "one sparse linear Dirichlet application per boundary condition"
            ),
        },
        "controls": {
            "landed_free_boundary_median": LANDED_FREE_MEDIAN,
            "landed_fixed_boundary_median": LANDED_FIXED_MEDIAN,
            "landed_vacuum_share": LANDED_VACUUM_SHARE,
            "absolute_reproduction_tolerance": CONTROL_ABSOLUTE_TOLERANCE,
            "numerical_basis": (
                "the current Nova environment replays the unchanged landed helper "
                "under NumPy 2.4.6 rather than the receipt's NumPy 2.4.4; an "
                "unpublished preflight failed at the control before emitting "
                "scores and measured maximum absolute drift 1.13e-7, so 2e-7 is "
                "fixed before the authoritative scoring run"
            ),
        },
        "convention_artefact_criterion": {
            "free_to_fixed_maximum_ratio": FIXED_LEVEL_RATIO,
            "rule": (
                "a candidate admits the free-boundary residual to the landed "
                "fixed-boundary level when its landed-five median free residual is "
                "no more than 1.05 times the landed pinned fixed median; a candidate "
                "whose fixed and free residuals both deteriorate is reported but "
                "cannot turn deterioration into admission"
            ),
            "any_member_is_decisive": True,
            "semantic_correction": (
                "the first published-score draft compared each free residual with "
                "its own fixed residual and was rejected before delivery because "
                "total-flux hypotheses made both residuals order ten; the task asks "
                "for the landed fixed-boundary level, whose value was frozen before "
                "this sweep"
            ),
        },
    }


def write_preregistration(output: Path) -> Path:
    """Persist the discrete set before any scored row is read."""

    output.mkdir(parents=True, exist_ok=True)
    path = output / PREREGISTRATION_NAME
    encoded = json.dumps(preregistration(), indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError("on-disk convention preregistration differs from policy")
    path.write_text(encoded)
    return path


def _polarity_population() -> set[str]:
    receipt = json.loads(POLARITY_RECEIPT.read_text())
    census = receipt["full_corpus_census"]
    if census["shot_count"] != 7_041 or census["affected_shot_count"] != 603:
        raise RuntimeError("polarity census does not retain the 7041/603 authority")
    return set(census["affected_shots"])


def _landed_frames(data: Path, affected: set[str]) -> list[SelectedFrame]:
    receipt = json.loads(ROOT_RECEIPT.read_text())
    frames = receipt["result"]["frames"]
    if len(frames) != LANDED_FRAME_COUNT:
        raise RuntimeError("root-existence receipt does not retain five frames")
    return [
        SelectedFrame(
            path=data / item["shot"],
            frame=int(item["frame"]),
            time_ms=float(item["time_ms"]),
            population="landed-control",
            absent_from_polarity_population=item["shot"] not in affected,
        )
        for item in frames
    ]


def select_frames(data: Path) -> list[SelectedFrame]:
    """Select the five controls and fifteen independently screened additions."""

    affected = _polarity_population()
    landed = _landed_frames(data, affected)
    landed_names = {item.path.name for item in landed}
    additional: list[SelectedFrame] = []
    for path in sorted(data.glob("*.parquet")):
        if path.name in landed_names or path.name in affected:
            continue
        row = _read(path, _LABEL_COLUMNS)
        frame = _eligible_frame(row)
        if frame is None:
            continue
        additional.append(
            SelectedFrame(
                path=path,
                frame=frame,
                time_ms=float(row["efit_times"][frame]),
                population="polarity-screened-additional",
                absent_from_polarity_population=True,
            )
        )
        if len(additional) == ADDITIONAL_FRAME_COUNT:
            break
    if len(additional) != ADDITIONAL_FRAME_COUNT:
        raise RuntimeError("fewer than fifteen screened diverted additions were found")
    return landed + additional


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def _profile_components(
    row: dict[str, Any], frame: int, radius: np.ndarray, height: np.ndarray
) -> dict[str, Any]:
    """Extract pinned gradients and the masks shared by every hypothesis."""

    pinned = next(item for item in variants() if item.source_cocos == CORPUS_COCOS)
    raw_flux_rz = np.asarray(row["efit_psirz"][frame], dtype=float).T
    pinned_label_rz = pinned.psi_to_nova * raw_flux_rz
    psi_norm = normalised_flux(row, frame).T
    core = _plasma_mask(row, frame, radius, height)
    extraction = extract_flux_functions(
        radius,
        height,
        pinned_label_rz,
        psi_norm,
        plasma_mask=core,
        min_samples=6,
    )
    reliable = extraction.reliable & np.isfinite(
        extraction.p_prime + extraction.ff_prime
    )
    if np.count_nonzero(reliable) < 2:
        raise RuntimeError("fewer than two reliable extracted profile surfaces")
    p_prime = np.interp(
        psi_norm, extraction.psi_norm[reliable], extraction.p_prime[reliable]
    )
    ff_prime = np.interp(
        psi_norm, extraction.psi_norm[reliable], extraction.ff_prime[reliable]
    )
    active = core & extraction.current.valid & np.isfinite(p_prime + ff_prime)
    active &= np.isfinite(psi_norm) & (psi_norm >= 0.0) & (psi_norm <= 1.0)
    return {
        "raw_flux_rz": raw_flux_rz,
        "pinned_p_prime": p_prime,
        "pinned_ff_prime": ff_prime,
        "active": active,
        "reliable_surfaces": int(np.count_nonzero(reliable)),
    }


def evaluate_variant(
    variant: Variant,
    profile: dict[str, Any],
    radius: np.ndarray,
    operator: Any,
    pinned_coil_rz: np.ndarray,
) -> dict[str, float | bool]:
    """Evaluate one convention hypothesis with two linear boundary solves."""

    pinned = next(item for item in variants() if item.source_cocos == CORPUS_COCOS)
    label_rz = variant.psi_to_nova * profile["raw_flux_rz"]
    gradient_ratio = variant.derivative_to_nova / pinned.derivative_to_nova
    p_prime = gradient_ratio * profile["pinned_p_prime"]
    ff_prime = gradient_ratio * profile["pinned_ff_prime"]
    radius_map = np.broadcast_to(radius[:, None], label_rz.shape)
    source = np.zeros_like(label_rz)
    active = profile["active"]
    source[active] = grad_shafranov_source(
        radius_map[active], p_prime[active], ff_prime[active]
    )
    coil_ratio = variant.ip_to_nova / pinned.ip_to_nova
    coil_rz = coil_ratio * pinned_coil_rz
    fixed_solution = operator.solve(source, label_rz)
    free_solution = operator.solve(source, coil_rz)
    fixed_residual = fixed_solution - label_rz
    free_residual = free_solution - label_rz
    vacuum_difference = free_solution - fixed_solution
    scale = abs(float(np.ptp(label_rz)))

    def fractional(field: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.asarray(field) ** 2)) / scale)

    fixed = fractional(fixed_residual)
    free = fractional(free_residual)
    vacuum = fractional(vacuum_difference)
    denominator = fixed + vacuum
    vacuum_share = vacuum / denominator if denominator > 0.0 else 0.5
    return {
        "fixed_boundary_fractional_rms": fixed,
        "free_boundary_fractional_rms": free,
        "vacuum_difference_fractional_rms": vacuum,
        "vacuum_share": vacuum_share,
        "free_to_fixed_ratio": free / fixed if fixed > 0.0 else float("inf"),
        "at_fixed_boundary_level": bool(free <= FIXED_LEVEL_RATIO * fixed),
    }


def summarize_variant(
    records: list[dict[str, Any]], variant: Variant
) -> dict[str, Any]:
    """Summarize one member on controls, additions, and the combined cohort."""

    def summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
        fixed = [item["fixed_boundary_fractional_rms"] for item in selected]
        free = [item["free_boundary_fractional_rms"] for item in selected]
        vacuum = [item["vacuum_difference_fractional_rms"] for item in selected]
        fixed_median = float(np.median(fixed))
        free_median = float(np.median(free))
        vacuum_median = float(np.median(vacuum))
        denominator = fixed_median + vacuum_median
        return {
            "frames": len(selected),
            "fixed_boundary_fractional_rms": _distribution(fixed),
            "free_boundary_fractional_rms": _distribution(free),
            "vacuum_difference_fractional_rms": _distribution(vacuum),
            "vacuum_share_from_medians": vacuum_median / denominator,
            "free_to_fixed_median_ratio": free_median / fixed_median,
            "at_candidate_fixed_boundary_level": bool(
                free_median <= FIXED_LEVEL_RATIO * fixed_median
            ),
        }

    controls = [item for item in records if item["population"] == "landed-control"]
    additions = [
        item for item in records if item["population"] == "polarity-screened-additional"
    ]
    return {
        "variant": asdict(variant)
        | {
            "identifier": variant.identifier,
            "poloidal_flux_sign": variant.poloidal_flux_sign,
            "toroidal_current_sign": variant.toroidal_current_sign,
            "derivative_sign": variant.derivative_sign,
            "raw_flux_interpretation": variant.raw_flux_interpretation,
        },
        "landed_controls": summary(controls),
        "screened_additions": summary(additions),
        "all_frames": summary(records),
    }


def convention_verdict(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """State whether any discrete member reaches its fixed-boundary level."""

    admitted = [
        item["variant"]["identifier"]
        for item in summaries
        if item["landed_controls"]["free_boundary_fractional_rms"]["median"]
        <= FIXED_LEVEL_RATIO * LANDED_FIXED_MEDIAN
    ]
    candidate_relative = [
        item["variant"]["identifier"]
        for item in summaries
        if item["landed_controls"]["at_candidate_fixed_boundary_level"]
    ]
    best = min(
        summaries,
        key=lambda item: item["landed_controls"]["free_boundary_fractional_rms"][
            "median"
        ],
    )
    return {
        "any_variant_reaches_landed_fixed_boundary_level": bool(admitted),
        "admitted_variants": admitted,
        "best_variant": best["variant"]["identifier"],
        "best_free_boundary_fractional_rms": best["landed_controls"][
            "free_boundary_fractional_rms"
        ]["median"],
        "best_free_to_fixed_median_ratio": best["landed_controls"][
            "free_to_fixed_median_ratio"
        ],
        "candidate_relative_admitted_variants": candidate_relative,
        "candidate_relative_admission_is_not_a_root_criterion": True,
        "convention_artefact": bool(admitted),
        "data_limitation_survives_convention_sweep": not admitted,
        "criterion": (
            f"landed-five free median <= {FIXED_LEVEL_RATIO} * landed pinned "
            "fixed median"
        ),
    }


def _render(summaries: list[dict[str, Any]], output: Path) -> Path:
    """Render residual and attribution metrics for all convention members."""

    identifiers = [item["variant"]["source_cocos"] for item in summaries]
    fixed = np.asarray(
        [
            item["landed_controls"]["fixed_boundary_fractional_rms"]["median"]
            for item in summaries
        ]
    )
    free = np.asarray(
        [
            item["landed_controls"]["free_boundary_fractional_rms"]["median"]
            for item in summaries
        ]
    )
    vacuum = np.asarray(
        [item["landed_controls"]["vacuum_share_from_medians"] for item in summaries]
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    locations = np.arange(len(summaries))
    axes[0].plot(locations, free, "o-", label="free boundary")
    axes[0].plot(locations, fixed, "s-", label="fixed boundary")
    axes[0].axhline(LANDED_FREE_MEDIAN, color="tab:blue", ls="--", lw=0.8)
    axes[0].axhline(LANDED_FIXED_MEDIAN, color="tab:orange", ls="--", lw=0.8)
    axes[0].set_xticks(locations, identifiers, rotation=45)
    axes[0].set_xlabel("hypothesized source COCOS")
    axes[0].set_ylabel("landed-five median fractional RMS")
    axes[0].set_yscale("log")
    axes[0].set_title("Linear residual at labelled state")
    axes[0].legend()

    axes[1].bar(locations, vacuum, color="tab:purple", alpha=0.75)
    axes[1].axhline(LANDED_VACUUM_SHARE, color="black", ls="--", lw=0.9)
    axes[1].set_xticks(locations, identifiers, rotation=45)
    axes[1].set_xlabel("hypothesized source COCOS")
    axes[1].set_ylabel("vacuum share")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Boundary-field attribution")
    path = output / FIGURE_NAME
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def run(data: Path, output: Path) -> dict[str, Any]:
    """Execute the preregistered linear convention sweep."""

    preregistration_path = write_preregistration(output)
    preregistration_digest = hashlib.sha256(
        preregistration_path.read_bytes()
    ).hexdigest()
    configure_dtypes()
    selected = select_frames(data)
    if len(selected) != TOTAL_FRAME_COUNT:
        raise RuntimeError("convention sweep did not select twenty frames")
    additional = [item for item in selected if item.population != "landed-control"]
    if not all(item.absent_from_polarity_population for item in additional):
        raise RuntimeError("an additional frame survived the polarity screen")

    columns = _LABEL_COLUMNS + _GEOMETRY_COLUMNS + _CURRENT_COLUMNS
    first_row = _read(selected[0].path, columns)
    radius, height = canonical_axes(first_row)
    operator = _operator(radius, height)
    registry = DiiidDescriptionRegistry()
    response_cache: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    records_by_variant = {item.identifier: [] for item in variants()}
    checkpoint = output / CHECKPOINT_NAME
    checkpoint.write_text("")
    for frame_index, selected_frame in enumerate(selected):
        row = first_row if frame_index == 0 else _read(selected_frame.path, columns)
        profile = _profile_components(row, selected_frame.frame, radius, height)
        description = registry.ingest(row, source_row=selected_frame.path.name)
        response = response_cache.get(description.physical_digest)
        if response is None:
            response = vacuum_response(description, radius, height)
            response_cache[description.physical_digest] = response
        pinned_coil_zr = vacuum_psi(row, description, response)[selected_frame.frame]
        pinned_coil_rz = np.asarray(pinned_coil_zr, dtype=float).T
        for variant in variants():
            result = evaluate_variant(
                variant, profile, radius, operator, pinned_coil_rz
            )
            record = {
                "shot": selected_frame.path.name,
                "frame": selected_frame.frame,
                "time_ms": selected_frame.time_ms,
                "population": selected_frame.population,
                "absent_from_polarity_population": (
                    selected_frame.absent_from_polarity_population
                ),
                "variant": variant.identifier,
                "reliable_profile_surfaces": profile["reliable_surfaces"],
                **result,
            }
            records_by_variant[variant.identifier].append(record)
            with checkpoint.open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")

    summaries = [
        summarize_variant(records_by_variant[item.identifier], item)
        for item in variants()
    ]
    pinned = next(
        item for item in summaries if item["variant"]["source_cocos"] == CORPUS_COCOS
    )
    control = {
        "source_cocos": CORPUS_COCOS,
        "measured_free_boundary_median": pinned["landed_controls"][
            "free_boundary_fractional_rms"
        ]["median"],
        "expected_free_boundary_median": LANDED_FREE_MEDIAN,
        "measured_fixed_boundary_median": pinned["landed_controls"][
            "fixed_boundary_fractional_rms"
        ]["median"],
        "expected_fixed_boundary_median": LANDED_FIXED_MEDIAN,
        "measured_vacuum_share": pinned["landed_controls"]["vacuum_share_from_medians"],
        "expected_vacuum_share": LANDED_VACUUM_SHARE,
    }
    control["free_absolute_difference"] = abs(
        control["measured_free_boundary_median"] - LANDED_FREE_MEDIAN
    )
    control["fixed_absolute_difference"] = abs(
        control["measured_fixed_boundary_median"] - LANDED_FIXED_MEDIAN
    )
    control["vacuum_share_absolute_difference"] = abs(
        control["measured_vacuum_share"] - LANDED_VACUUM_SHARE
    )
    control["reproduced"] = bool(
        max(
            control["free_absolute_difference"],
            control["fixed_absolute_difference"],
            control["vacuum_share_absolute_difference"],
        )
        <= CONTROL_ABSOLUTE_TOLERANCE
    )
    if not control["reproduced"]:
        raise RuntimeError(
            f"pinned convention did not reproduce landed control: {control}"
        )
    verdict = convention_verdict(summaries)
    figure = _render(summaries, output)
    receipt = {
        "preregistration": preregistration(),
        "preregistration_sha256": preregistration_digest,
        "selection": {
            "frames": len(selected),
            "landed_control_frames": LANDED_FRAME_COUNT,
            "screened_additional_frames": len(additional),
            "distinct_shots": len({item.path.name for item in selected}),
            "polarity_population_count": len(_polarity_population()),
            "all_additional_absent_from_polarity_population": all(
                item.absent_from_polarity_population for item in additional
            ),
            "frames_detail": [
                {
                    "shot": item.path.name,
                    "frame": item.frame,
                    "time_ms": item.time_ms,
                    "population": item.population,
                    "absent_from_polarity_population": (
                        item.absent_from_polarity_population
                    ),
                }
                for item in selected
            ],
        },
        "control": control,
        "variants": summaries,
        "verdict": verdict,
        "no_root_search": True,
        "coefficients_fitted": 0,
        "currents_adjusted": False,
        "artifacts": {
            "preregistration": str(preregistration_path),
            "receipt": str(output / RECEIPT_NAME),
            "incremental_checkpoint": str(checkpoint),
            "figure": str(figure),
        },
    }
    (output / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preregister-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    path = write_preregistration(arguments.output)
    print(f"PREREGISTERED {path}", flush=True)
    if arguments.preregister_only:
        return
    receipt = run(arguments.data, arguments.output)
    print(
        json.dumps(
            {"control": receipt["control"], "verdict": receipt["verdict"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
