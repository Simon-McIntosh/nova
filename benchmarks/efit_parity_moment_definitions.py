"""Rescore published equilibrium integrals under matched moment definitions."""

from __future__ import annotations

import argparse
import hashlib
import json
from html import escape
from pathlib import Path
from typing import Any

from scipy.constants import mu_0

INPUT_RECEIPT = Path(
    "docs/figures/efit-forward-parity/converged-root-geometry-attribution.json"
)
OUTPUT_RECEIPT = Path("docs/figures/efit-forward-parity/moment-definition-rescore.json")
EXPECTED_REFERENCE_BETA = 0.3351304829120636
EXPECTED_REFERENCE_LI = 0.7693799734115601
REPRODUCTION_RELATIVE_TOLERANCE = 1.0e-12


def _relative_error(observed: float, expected: float) -> float:
    return observed / expected - 1.0


def _metric_value(
    value: float,
    current_support: str,
    current_a: float | None,
    denominator: float,
) -> dict[str, Any]:
    return {
        "value": value,
        "current_support": current_support,
        "current_a": current_a,
        "denominator_t2_m3": denominator,
    }


def _definition_values(
    pressure_integral: float,
    field_integral: float,
    beta_denominators: dict[str, float],
    li_denominators: dict[str, float] | float,
    currents: dict[str, float],
) -> dict[str, Any]:
    beta_numerator = 2.0 * mu_0 * pressure_integral
    beta = {
        support: _metric_value(
            beta_numerator / denominator,
            support,
            currents[support],
            denominator,
        )
        for support, denominator in beta_denominators.items()
    }
    if isinstance(li_denominators, dict):
        internal_inductance = {
            support: _metric_value(
                field_integral / denominator,
                support,
                currents[support],
                denominator,
            )
            for support, denominator in li_denominators.items()
        }
    else:
        internal_inductance = {
            "current_independent": _metric_value(
                field_integral / li_denominators,
                "not_used_mean_squared_boundary_field",
                None,
                li_denominators,
            )
        }
    return {
        "poloidal_beta": beta,
        "internal_inductance": internal_inductance,
    }


def _format_metric(values: dict[str, dict[str, Any]]) -> str:
    items = []
    for support, record in values.items():
        current = record["current_a"]
        support_text = escape(support.replace("_", " "))
        if current is None:
            label = support_text
        else:
            label = f"{support_text}, Ip={current:.9f} A"
        items.append(f"<li>{label}: {record['value']:.15g}</li>")
    return "<ul>" + "".join(items) + "</ul>"


def _html_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{escape(row['definition_label'])}</td>"
            f"<td>{escape(row['side'])}</td>"
            f"<td>{_format_metric(row['poloidal_beta'])}</td>"
            f"<td>{_format_metric(row['internal_inductance'])}</td>"
            "</tr>"
        )
    return (
        '<div style="overflow-x:auto"><table>'
        "<thead><tr><th>Definition</th><th>Integral side</th>"
        "<th>beta_p</th><th>l_i</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _verify_protected_artifacts(
    input_receipt: Path, banked_integrity: dict[str, Any]
) -> dict[str, Any]:
    expected = banked_integrity["sha256"]
    directory = input_receipt.parent
    observed = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in expected
    }
    mismatches = {
        name: {"expected": expected[name], "observed": observed[name]}
        for name in expected
        if observed[name] != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"protected banked artifacts changed: {mismatches}")
    if len(expected) != banked_integrity["verified_digest_count"]:
        raise RuntimeError("protected artifact count disagrees with its receipt")
    return {
        "declared_count": banked_integrity["banked_file_count"],
        "verified_digest_count": len(observed),
        "all_digests_match": True,
        "source_receipt_and_this_receipt_are_outside_protected_set": True,
    }


def rescore(input_receipt: Path = INPUT_RECEIPT) -> dict[str, Any]:
    """Return a zero-solve rescore built only from published integrals."""
    source = json.loads(input_receipt.read_text())
    attribution = source["moment_normalisation_attribution"]
    solved = attribution["solved_live_equilibrium"]
    reference = attribution["reference"]

    currents = {
        "confined_core": solved["moment_confined_core_current_a"],
        "all_domain_constrained": solved["constrained_terminal_cell_current_sum_a"],
    }
    nova_denominators = {
        support: 0.5 * mu_0**2 * solved["volume_weighted_major_radius_m"] * current**2
        for support, current in currents.items()
    }
    reference_beta_denominators = {
        support: (mu_0 * current / reference["stored_lcfs_perimeter_m"]) ** 2
        * reference["plasma_volume_m3"]
        for support, current in currents.items()
    }
    reference_li_denominator = (
        reference["lcfs_surface_mean_poloidal_field_squared_t2"]
        * reference["plasma_volume_m3"]
    )

    sides = {
        "solved": {
            "pressure": solved["pressure_volume_integral_pa_m3"],
            "field": solved["poloidal_field_squared_volume_integral_t2_m3"],
        },
        "reference": {
            "pressure": reference["pressure_volume_integral_implied_by_betap_pa_m3"],
            "field": reference["poloidal_field_squared_volume_integral_t2_m3"],
        },
    }
    definitions = {
        "nova_shared_volume_current": {
            side: _definition_values(
                values["pressure"],
                values["field"],
                nova_denominators,
                nova_denominators,
                currents,
            )
            for side, values in sides.items()
        },
        "reference_boundary_field": {
            side: _definition_values(
                values["pressure"],
                values["field"],
                reference_beta_denominators,
                reference_li_denominator,
                currents,
            )
            for side, values in sides.items()
        },
    }

    reproduced_beta = definitions["reference_boundary_field"]["reference"][
        "poloidal_beta"
    ]["all_domain_constrained"]["value"]
    reproduced_li = definitions["reference_boundary_field"]["reference"][
        "internal_inductance"
    ]["current_independent"]["value"]
    beta_error = _relative_error(reproduced_beta, EXPECTED_REFERENCE_BETA)
    li_error = _relative_error(reproduced_li, EXPECTED_REFERENCE_LI)
    reproduction = {
        "poloidal_beta": {
            "reproduced": reproduced_beta,
            "published": EXPECTED_REFERENCE_BETA,
            "signed_relative_error": beta_error,
            "absolute_relative_error": abs(beta_error),
            "passes": abs(beta_error) <= REPRODUCTION_RELATIVE_TOLERANCE,
        },
        "internal_inductance": {
            "reproduced": reproduced_li,
            "published": EXPECTED_REFERENCE_LI,
            "signed_relative_error": li_error,
            "absolute_relative_error": abs(li_error),
            "passes": abs(li_error) <= REPRODUCTION_RELATIVE_TOLERANCE,
        },
        "relative_tolerance": REPRODUCTION_RELATIVE_TOLERANCE,
    }
    if not all(
        reproduction[name]["passes"]
        for name in ("poloidal_beta", "internal_inductance")
    ):
        raise RuntimeError(
            "the boundary-field definition does not reproduce the reference"
        )

    beta_deviation = sides["solved"]["pressure"] / sides["reference"]["pressure"] - 1.0
    li_deviation = sides["solved"]["field"] / sides["reference"]["field"] - 1.0
    matched = {
        "poloidal_beta": {
            "signed_relative_deviation": beta_deviation,
            "absolute_relative_deviation": abs(beta_deviation),
            "verdict": "FAIL_REFERENCE_REPRODUCTION",
            "reading": (
                "The solved pressure integral is smaller under either common "
                "scoring scale."
            ),
        },
        "internal_inductance": {
            "signed_relative_deviation": li_deviation,
            "absolute_relative_deviation": abs(li_deviation),
            "verdict": "FAIL_REFERENCE_REPRODUCTION",
            "reading": (
                "The solved Bp-squared volume integral is smaller under either "
                "common scoring scale."
            ),
        },
        "invariant_across_definitions_and_current_supports": True,
    }

    rows = []
    labels = {
        "nova_shared_volume_current": "Nova shared volume-current denominator",
        "reference_boundary_field": "Reference boundary-field denominators",
    }
    for definition, by_side in definitions.items():
        for side, values in by_side.items():
            rows.append(
                {
                    "definition": definition,
                    "definition_label": labels[definition],
                    "side": side,
                    **values,
                }
            )

    protected = _verify_protected_artifacts(
        input_receipt, source["banked_artifact_integrity"]
    )
    return {
        "receipt": {
            "kind": "moment_definition_rescore",
            "shot": 22086,
            "slice_index": 43,
            "input_receipt": str(input_receipt),
            "input_receipt_sha256": hashlib.sha256(
                input_receipt.read_bytes()
            ).hexdigest(),
        },
        "execution_contract": {
            "nonlinear_solve_calls": 0,
            "new_equilibria": 0,
            "input_mode": "banked published integrals only",
        },
        "definitions": attribution["definitions"],
        "current_supports": {
            **currents,
            "confined_core_over_all_domain": currents["confined_core"]
            / currents["all_domain_constrained"],
            "outside_confined_core_a": currents["all_domain_constrained"]
            - currents["confined_core"],
        },
        "four_by_two_rescore": {
            "rows": rows,
            "html_table": _html_table(rows),
        },
        "reference_definition_reproduction": reproduction,
        "matched_definition_deviations": matched,
        "parity_verdict_scoreability": {
            "scoreable": True,
            "verdict": "SCOREABLE_MATCHED_DEFINITIONS_FAIL_REPRODUCTION",
            "reason": (
                "Both published numerator integrals can be scored on either common "
                "denominator family; the matched solved/reference deviations are "
                "therefore definition- and support-invariant."
            ),
            "additional_quantity_required": None,
        },
        "protected_banked_artifacts": protected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_RECEIPT)
    parser.add_argument("--output", type=Path, default=OUTPUT_RECEIPT)
    arguments = parser.parse_args()
    receipt = rescore(arguments.input)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
