"""Classify registered acceptance quantities by their continuum reference.

This is a receipt builder, not an acceptance repair.  It reads the committed
registration and measurements, records the semantic class of every registered
quantity, and leaves every criterion unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parents[1]
CRITERION_SOURCE = (
    HERE / "docs/figures/forward-operator-refinement/criterion-family.json"
)
ACCEPTANCE_SOURCE = (
    HERE / "docs/figures/derived-observable-parity/integrated-acceptance.json"
)
PROCESS_SOURCE = (
    HERE / "docs/figures/same-device-label-determinism/executable-boundary-arms.json"
)
DEFAULT_OUTPUT = (
    HERE / "docs/figures/roundoff-scale-acceptance-bounds/bound-classification.json"
)

ZERO_IDENTITIES = {
    "conservation.divergence_b": {
        "reason": (
            "The axisymmetric magnetic field is constructed from one flux "
            "function, so div(B) is identically zero in the continuum for any "
            "flux map, not merely for a converged equilibrium."
        ),
        "discretisation_reason": (
            "The scored production ring-mesh least-squares derivative operators "
            "do not commute, so the mixed derivatives do not cancel exactly; "
            "the observable has a second-order truncation floor."
        ),
    },
    "conservation.divergence_j": {
        "reason": (
            "The poloidal current is constructed from the single-valued field "
            "function F(psi), so div(J) is identically zero in the continuum "
            "for any differentiable flux-function closure."
        ),
        "discretisation_reason": (
            "The scored production ring-mesh least-squares derivative operators "
            "do not commute, so the composed field-function gradients retain a "
            "second-order truncation floor."
        ),
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registration(criterion: dict[str, Any]) -> list[dict[str, Any]]:
    rows = criterion["criterion_family"]["terminal_compiled_parity"][
        "terminal_observable_registration"
    ]["bounds"]
    if len(rows) != 69 or len({row["observable"] for row in rows}) != 69:
        raise RuntimeError("the registered acceptance bank is not 69 unique bounds")
    return rows


def _nonzero_reason(name: str) -> str:
    if name == "cell_current":
        return (
            "The scored cell-current image carries the nonzero plasma-current "
            "distribution."
        )
    if name == "conservation.checked_cells":
        return (
            "The scored value is the positive count of cells carrying a complete "
            "conservation stencil."
        )
    if name.startswith("conservation."):
        quantity = name.removeprefix("conservation.").replace("_", " ")
        return (
            f"The scored {quantity} is a finite computed residual or its physical "
            "normalising scale; unlike the two divergence identities, its value "
            "for a stopped discrete solve is not identically zero for every flux map."
        )
    if name.startswith("continuation."):
        _, domain, field = name.split(".")
        return (
            f"The scored {domain.replace('_', ' ')} continuation "
            f"{field.replace('_', ' ')} "
            "is declared policy or closure data, not a continuum identity residual; "
            "zero is one legitimate policy value rather than its defining value."
        )
    if name.startswith("domains."):
        field = name.removeprefix("domains.").replace("_", " ")
        return (
            f"The scored domain {field} is a topology label or normalised-flux field "
            "with case-dependent nonzero values, not an identically zero residual."
        )
    if name.startswith("finite."):
        field = name.removeprefix("finite.").replace("_", " ")
        return (
            f"The scored {field} finiteness predicate has the exact nonzero value "
            "true for a valid result."
        )
    if name.startswith("fixed_point."):
        field = name.removeprefix("fixed_point.").replace("_", " ")
        return (
            f"The scored fixed-point {field} records the stopped numerical solve; "
            "its reference is the nonzero achieved state, trace, residual, or "
            "qualification, "
            "not an identity that vanishes for every continuum field."
        )
    if name == "flux":
        return "The scored total-flux map is a case-dependent nonzero physical field."
    if name.startswith("ledger."):
        field = name.removeprefix("ledger.").replace("_", " ")
        return (
            f"The scored {field} current ledger entry is an integral over a labelled "
            "domain; it may be zero for an empty domain but is not defined to "
            "vanish in the continuum."
        )
    if name.startswith("moments."):
        field = name.removeprefix("moments.").replace("_", " ")
        return (
            f"The scored {field} is a case-dependent nonzero equilibrium integral "
            "or geometric moment."
        )
    if name.startswith("normalisation."):
        field = name.removeprefix("normalisation.").replace("_", " ")
        return (
            f"The scored normalisation {field} is policy or a scale factor; its exact "
            "reference is case-dependent and is not a vanishing continuum identity."
        )
    if name.startswith("rotation."):
        field = name.removeprefix("rotation.").replace("_", " ")
        return (
            f"The scored rotation {field} is closure policy or a physical parameter; "
            "zero can describe a static case but is not an identity imposed on "
            "all cases."
        )
    if name.startswith("topology."):
        field = name.removeprefix("topology.").replace("_", " ")
        return (
            f"The scored topology {field} is a location, flux, or branch predicate "
            "with a case-dependent exact reference, not an identically zero residual."
        )
    raise RuntimeError(f"unclassified registered observable: {name}")


def _classifications(registration: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for bound in registration:
        name = str(bound["observable"])
        identity = ZERO_IDENTITIES.get(name)
        result.append(
            {
                "observable": name,
                "criterion_kind": bound["criterion_kind"],
                "registered_absolute_bound": bound.get("absolute_bound"),
                "registered_relative_bound": bound.get("relative_bound"),
                "has_nonzero_continuum_value": identity is None,
                "continuum_class": (
                    "nonzero_reference" if identity is None else "zero_identity"
                ),
                "reason": _nonzero_reason(name)
                if identity is None
                else identity["reason"],
                "identity_preserved_by_scored_discretisation": (
                    None if identity is None else False
                ),
                "discretisation_reason": (
                    None if identity is None else identity["discretisation_reason"]
                ),
                "floor_kind_if_needed": None if identity is None else "truncation",
            }
        )
    return result


def _failing_cases(acceptance: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for batch in acceptance["batch_results"]:
        for observable in batch["per_observable"]:
            for case in observable["cases"]:
                failed = [member for member in case["members"] if not member["passes"]]
                if not failed:
                    continue
                absolute_bound = float(observable["absolute_bound"])
                relative_bound = float(observable["relative_bound"])
                failures.append(
                    {
                        "batch_size": int(batch["batch_size"]),
                        "observable": observable["observable"],
                        "case_id": case["case_id"],
                        "failed_member_count": len(failed),
                        "maximum_absolute_difference_by_failed_member": [
                            member["maximum_absolute_difference"] for member in failed
                        ],
                        "registered_absolute_bound": absolute_bound,
                        "absolute_bound_satisfied_by_every_failed_member": all(
                            member["maximum_absolute_difference"] <= absolute_bound
                            for member in failed
                        ),
                        "maximum_relative_difference_by_failed_member": [
                            member["maximum_relative_difference"] for member in failed
                        ],
                        "registered_relative_bound": relative_bound,
                        "relative_bound_failed_by_every_failed_member": all(
                            member["maximum_relative_difference"] > relative_bound
                            for member in failed
                        ),
                    }
                )
    all_absolute = all(
        row["absolute_bound_satisfied_by_every_failed_member"] for row in failures
    )
    all_relative = all(
        row["relative_bound_failed_by_every_failed_member"] for row in failures
    )
    return {
        "tested_first": True,
        "hypothesis": (
            "Every currently failing member satisfies its registered absolute "
            "bound and fails only because the relative bound is also applied."
        ),
        "outcome": "REJECTED" if not (all_absolute and all_relative) else "SUPPORTED",
        "every_failure_satisfies_absolute_bound": all_absolute,
        "every_failure_violates_relative_bound": all_relative,
        "failure_group_count": len(failures),
        "failing_member_evaluation_count": sum(
            row["failed_member_count"] for row in failures
        ),
        "reason": (
            "The relative criterion fails for every current failure, but width 4 "
            "case 22086/43 on conservation.divergence_j also exceeds its registered "
            "absolute bound, so not every failure is relative-only."
            if not all_absolute
            else "Every current failure is relative-only."
        ),
        "per_failing_case": failures,
    }


def _cache_off_arm(process: dict[str, Any]) -> dict[str, Any]:
    return next(
        arm for arm in process["arms"].values() if arm["name"] == "cache_off"
    )


def _required_exemplar(process: dict[str, Any]) -> dict[str, Any]:
    row = next(
        row
        for row in _cache_off_arm(process)["pass_status_changes"]
        if row["observable"] == "conservation.divergence_b"
        and row["case_id"] == "21978/35"
        and row["batch_size"] == 1
    )
    return {
        "source_arm": "cache_off",
        "case_id": row["case_id"],
        "batch_size": row["batch_size"],
        "observable": row["observable"],
        "absolute_value_by_invocation": row["absolute_value_by_invocation"],
        "registered_absolute_bound": row["bound"]["absolute_bound"],
        "every_invocation_below_absolute_bound": all(
            value <= row["bound"]["absolute_bound"]
            for value in row["absolute_value_by_invocation"]
        ),
        "registered_relative_bound": row["bound"]["relative_bound"],
        "passes_by_invocation": row["passes_by_invocation"],
    }


def _scale_separation(
    process: dict[str, Any], acceptance: dict[str, Any]
) -> list[dict[str, Any]]:
    changes = _cache_off_arm(process)["pass_status_changes"]
    roundoff_scale = max(
        occurrence["maximum_absolute_difference"]
        for row in acceptance["remaining_failures"]
        if row["observable"] == "conservation.divergence_b"
        for occurrence in row["occurrences"]
    )
    result = []
    for name in sorted(ZERO_IDENTITIES):
        rows = [row for row in changes if row["observable"] == name]
        magnitude = max(
            value for row in rows for value in row["absolute_value_by_invocation"]
        )
        result.append(
            {
                "observable": name,
                "maximum_observable_magnitude": magnitude,
                "banked_process_roundoff_scale": roundoff_scale,
                "orders_of_magnitude_separation": (
                    None
                    if roundoff_scale == 0.0
                    else __import__("math").log10(magnitude / roundoff_scale)
                ),
                "interpretation": (
                    "The observable magnitude is truncation-dominated on the "
                    "non-commuting ring mesh; the quoted process scale is the "
                    "last-bit variation banked alongside the acceptance evidence."
                ),
            }
        )
    return result


def build_receipt() -> dict[str, Any]:
    criterion = _read_json(CRITERION_SOURCE)
    acceptance = _read_json(ACCEPTANCE_SOURCE)
    process = _read_json(PROCESS_SOURCE)
    registration = _registration(criterion)
    classifications = _classifications(registration)
    return {
        "artifact": "bound_reference_classification",
        "status": "complete",
        "scope": "classification_only_no_bound_change_no_repair",
        "registered_bound_count": len(classifications),
        "nonzero_continuum_value_count": sum(
            row["has_nonzero_continuum_value"] for row in classifications
        ),
        "zero_in_continuum_count": sum(
            not row["has_nonzero_continuum_value"] for row in classifications
        ),
        "absolute_bound_already_satisfied_hypothesis": _failing_cases(acceptance),
        "required_banked_exemplar": _required_exemplar(process),
        "zero_identity_scale_separation": _scale_separation(process, acceptance),
        "classifications": classifications,
        "bound_changes_authored": 0,
        "repairs_authored": 0,
        "evidence_sources": [
            {
                "path": str(CRITERION_SOURCE.relative_to(HERE)),
                "sha256": _sha256(CRITERION_SOURCE),
            },
            {
                "path": str(ACCEPTANCE_SOURCE.relative_to(HERE)),
                "sha256": _sha256(ACCEPTANCE_SOURCE),
            },
            {
                "path": str(PROCESS_SOURCE.relative_to(HERE)),
                "sha256": _sha256(PROCESS_SOURCE),
            },
            {
                "path": "nova/equilibrium/conservation.py",
                "sha256": _sha256(HERE / "nova/equilibrium/conservation.py"),
            },
            {
                "path": "nova/equilibrium/stencil_mesh.py",
                "sha256": _sha256(HERE / "nova/equilibrium/stencil_mesh.py"),
            },
        ],
    }


def _validate(receipt: dict[str, Any]) -> None:
    rows = receipt["classifications"]
    if len(rows) != 69 or len({row["observable"] for row in rows}) != 69:
        raise RuntimeError("receipt does not classify 69 unique bounds")
    if any(not row["reason"] for row in rows):
        raise RuntimeError("every classification requires a per-bound reason")
    zero = [row for row in rows if not row["has_nonzero_continuum_value"]]
    if {row["observable"] for row in zero} != set(ZERO_IDENTITIES):
        raise RuntimeError("zero-identity classification drifted")
    if any(
        row["identity_preserved_by_scored_discretisation"] is not False for row in zero
    ):
        raise RuntimeError("each zero identity must record stencil preservation")
    hypothesis = receipt["absolute_bound_already_satisfied_hypothesis"]
    if (
        hypothesis["outcome"] != "REJECTED"
        or hypothesis["failing_member_evaluation_count"] != 18
    ):
        raise RuntimeError("cheap-hypothesis evidence changed")
    exemplar = receipt["required_banked_exemplar"]
    if exemplar["passes_by_invocation"] != [True, False, False]:
        raise RuntimeError("required banked exemplar changed")
    if not exemplar["every_invocation_below_absolute_bound"]:
        raise RuntimeError(
            "required banked exemplar no longer satisfies the absolute bound"
        )
    if receipt["bound_changes_authored"] or receipt["repairs_authored"]:
        raise RuntimeError("classification receipt must not author a repair")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    receipt = build_receipt()
    _validate(receipt)
    rendered = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.check:
        if (
            not args.output.exists()
            or args.output.read_text(encoding="utf-8") != rendered
        ):
            raise RuntimeError(f"classification receipt is stale: {args.output}")
        print(
            "bound classification check passed: 69 bounds, 2 zero identities, "
            "18 current failing members"
        )
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
