"""Adjudicate a COCOS current-sign dispute from the defining sign algebra."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import tau
from pathlib import Path
from typing import Any, Mapping


SOURCE_COCOS = 11
TARGET_COCOS = 17
DEFAULT_OUTPUT = Path(
    "docs/figures/diiid-forward-onboarding/cocos-ip-sign/receipt.json"
)


@dataclass(frozen=True)
class CocosDefinition:
    """The four signs and flux exponent that define one COCOS index."""

    identifier: int
    sigma_bp: int
    e_bp: int
    sigma_r_phi_z: int
    sigma_rho_theta_phi: int


@dataclass(frozen=True)
class QuantityFactors:
    """Factors for the five quantity classes needed by the adjudication."""

    psi_like: float
    ip_like: float
    b0_like: float
    q_like: float
    dodpsi_like: float

    def as_dict(self) -> dict[str, float]:
        """Return factors in stable receipt order."""

        return {
            "psi_like": self.psi_like,
            "ip_like": self.ip_like,
            "b0_like": self.b0_like,
            "q_like": self.q_like,
            "dodpsi_like": self.dodpsi_like,
        }


SOURCE_DEFINITION = CocosDefinition(
    identifier=SOURCE_COCOS,
    sigma_bp=+1,
    e_bp=1,
    sigma_r_phi_z=+1,
    sigma_rho_theta_phi=+1,
)
TARGET_DEFINITION = CocosDefinition(
    identifier=TARGET_COCOS,
    sigma_bp=-1,
    e_bp=1,
    sigma_r_phi_z=+1,
    sigma_rho_theta_phi=+1,
)

MIGRATION_GUIDE_IP_PATHS = (
    "pf_active/coil/current",
    "pf_active/circuit/current",
    "pf_active/supply/current",
)
MIGRATION_GUIDE_IP_FACTOR = -1.0


def derive_factors(
    source: CocosDefinition,
    target: CocosDefinition,
) -> QuantityFactors:
    """Derive quantity factors using only the published COCOS definitions.

    Sauter and Medvedev define the effective signs between two conventions as
    products of their defining signs. Toroidal current and field follow the
    relative cylindrical handedness. Poloidal flux additionally follows the
    poloidal-field sign and flux exponent, while a flux derivative takes its
    reciprocal. Safety factor follows the relative flux-surface and cylindrical
    handednesses.
    """

    sigma_bp_eff = source.sigma_bp * target.sigma_bp
    sigma_r_phi_z_eff = source.sigma_r_phi_z * target.sigma_r_phi_z
    sigma_rho_theta_phi_eff = source.sigma_rho_theta_phi * target.sigma_rho_theta_phi
    flux_exponent_delta = target.e_bp - source.e_bp
    psi_factor = float(sigma_bp_eff * sigma_r_phi_z_eff) * (tau**flux_exponent_delta)
    return QuantityFactors(
        psi_like=psi_factor,
        ip_like=float(sigma_r_phi_z_eff),
        b0_like=float(sigma_r_phi_z_eff),
        q_like=float(sigma_rho_theta_phi_eff * sigma_r_phi_z_eff),
        dodpsi_like=1.0 / psi_factor,
    )


def effective_signs(
    source: CocosDefinition,
    target: CocosDefinition,
) -> dict[str, int]:
    """Expose the intermediate products used by the factor derivation."""

    return {
        "sigma_bp_eff": source.sigma_bp * target.sigma_bp,
        "sigma_r_phi_z_eff": source.sigma_r_phi_z * target.sigma_r_phi_z,
        "sigma_rho_theta_phi_eff": (
            source.sigma_rho_theta_phi * target.sigma_rho_theta_phi
        ),
        "e_bp_delta": target.e_bp - source.e_bp,
    }


def current_round_trip(current: float) -> dict[str, float | bool]:
    """Carry a current through both convention directions and require identity."""

    forward_factor = derive_factors(SOURCE_DEFINITION, TARGET_DEFINITION).ip_like
    reverse_factor = derive_factors(TARGET_DEFINITION, SOURCE_DEFINITION).ip_like
    converted = current * forward_factor
    restored = converted * reverse_factor
    return {
        "source_value_a": current,
        "forward_factor": forward_factor,
        "target_value_a": converted,
        "reverse_factor": reverse_factor,
        "restored_value_a": restored,
        "exact_equal": restored == current,
    }


def nova_engine_claim() -> dict[str, float]:
    """Read Nova's claim only after the independent derivation is complete."""

    from nova.io.cocos import (  # noqa: PLC0415
        B0_LIKE,
        DODPSI_LIKE,
        IP_LIKE,
        PSI_LIKE,
        Q_LIKE,
        convention_transform,
    )

    transform = convention_transform(source=SOURCE_COCOS, target=TARGET_COCOS)
    return {
        "psi_like": transform.factor(PSI_LIKE),
        "ip_like": transform.factor(IP_LIKE),
        "b0_like": transform.factor(B0_LIKE),
        "q_like": transform.factor(Q_LIKE),
        "dodpsi_like": transform.factor(DODPSI_LIKE),
    }


def build_receipt() -> dict[str, Any]:
    """Build the definition-first derivation and authority verdict."""

    factors = derive_factors(SOURCE_DEFINITION, TARGET_DEFINITION)
    derived = factors.as_dict()
    engine = nova_engine_claim()
    round_trip = current_round_trip(-123456.75)
    if engine != derived:
        raise ValueError(
            f"Nova's engine disagrees with the definition-derived factors: {engine}"
        )
    if MIGRATION_GUIDE_IP_FACTOR == factors.ip_like:
        raise ValueError("the recorded authority dispute is no longer present")
    if not round_trip["exact_equal"]:
        raise ValueError(f"current round trip changed the value: {round_trip}")

    return {
        "measurement": "COCOS current-sign adjudication from defining signs",
        "conventions": {
            "source": asdict(SOURCE_DEFINITION),
            "target": asdict(TARGET_DEFINITION),
        },
        "effective_signs": effective_signs(SOURCE_DEFINITION, TARGET_DEFINITION),
        "derivation": {
            "psi_like": "sigma_bp_eff * sigma_r_phi_z_eff * (2*pi)^e_bp_delta",
            "ip_like": "sigma_r_phi_z_eff",
            "b0_like": "sigma_r_phi_z_eff",
            "q_like": "sigma_rho_theta_phi_eff * sigma_r_phi_z_eff",
            "dodpsi_like": "1 / psi_like",
            "factors": derived,
        },
        "authority_claims": {
            "nova_convention_engine": {
                "factors": engine,
                "matches_definition_derivation": True,
            },
            "imas_data_dictionary_migration_guide": {
                "quantity_class": "ip_like",
                "displayed_factor": MIGRATION_GUIDE_IP_FACTOR,
                "paths": list(MIGRATION_GUIDE_IP_PATHS),
                "matches_definition_derivation": False,
            },
        },
        "current_round_trip": round_trip,
        "adjudication": {
            "correct_ip_like_factor": factors.ip_like,
            "wrong_authority": (
                "IMAS Data Dictionary migration guide display for the "
                "COCOS-11 to COCOS-17 ip_like factor"
            ),
            "authority_matching_definitions": "Nova convention engine",
            "reason": (
                "COCOS 11 and 17 both have sigma_RphiZ=+1, so Sauter and "
                "Medvedev Eq. 39 gives sigma_Ip_eff=(+1)*(+1)=+1. Only "
                "sigma_Bp changes, from +1 to -1; that flips psi_like and "
                "dodpsi_like, not ip_like or b0_like. The guide's displayed "
                "-1 ip_like factor is therefore inconsistent with both the "
                "COCOS definition and the DD's ip_like expression "
                "sigma_ip_eff."
            ),
        },
        "sources": {
            "definition": {
                "title": "Tokamak coordinate conventions: COCOS",
                "authors": "O. Sauter and S. Yu. Medvedev",
                "doi": "10.1016/j.cpc.2012.09.010",
                "used_parts": "Table I and Eqs. 39-42",
                "url": (
                    "https://www.epfl.ch/research/domains/swiss-plasma-center/"
                    "wp-content/uploads/2018/10/"
                    "Sauter_COCOS_Tokamak_Coordinate_Conventions.pdf"
                ),
            },
            "dd_metadata": {
                "statement": (
                    "The DD developer guide labels pf_active coil current "
                    "ip_like and specifies the expression sigma_ip_eff."
                ),
                "url": (
                    "https://imas-data-dictionary.readthedocs.io/en/4.1.1/"
                    "dd_developer_guide.html"
                ),
            },
        },
    }


def write_receipt(receipt: Mapping[str, Any], output: Path | str) -> Path:
    """Write a stable JSON receipt."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = build_receipt()
    path = write_receipt(receipt, args.output)
    print(json.dumps({"output": str(path), **receipt["adjudication"]}, indent=2))


if __name__ == "__main__":
    main()
