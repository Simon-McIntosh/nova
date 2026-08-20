"""Definition-first tests for the COCOS current-sign adjudication."""

from benchmarks.cocos_ip_like_adjudication import (
    MIGRATION_GUIDE_IP_FACTOR,
    SOURCE_DEFINITION,
    TARGET_DEFINITION,
    build_receipt,
    current_round_trip,
    derive_factors,
    effective_signs,
)


def test_quantity_factors_follow_only_the_defining_signs():
    signs = effective_signs(SOURCE_DEFINITION, TARGET_DEFINITION)
    factors = derive_factors(SOURCE_DEFINITION, TARGET_DEFINITION)

    assert signs == {
        "sigma_bp_eff": -1,
        "sigma_r_phi_z_eff": +1,
        "sigma_rho_theta_phi_eff": +1,
        "e_bp_delta": 0,
    }
    assert factors.as_dict() == {
        "psi_like": -1.0,
        "ip_like": +1.0,
        "b0_like": +1.0,
        "q_like": +1.0,
        "dodpsi_like": -1.0,
    }


def test_current_bearing_quantity_round_trips_at_exact_equality():
    receipt = current_round_trip(-123456.75)

    assert receipt["forward_factor"] == +1.0
    assert receipt["reverse_factor"] == +1.0
    assert receipt["restored_value_a"] == receipt["source_value_a"]
    assert receipt["exact_equal"] is True


def test_receipt_adjudicates_the_disputed_authorities():
    receipt = build_receipt()

    assert receipt["derivation"]["factors"]["ip_like"] == +1.0
    assert MIGRATION_GUIDE_IP_FACTOR == -1.0
    assert (
        receipt["authority_claims"]["nova_convention_engine"][
            "matches_definition_derivation"
        ]
        is True
    )
    guide = receipt["authority_claims"]["imas_data_dictionary_migration_guide"]
    assert guide["displayed_factor"] == -1.0
    assert guide["matches_definition_derivation"] is False
    assert len(guide["paths"]) == 3
    assert receipt["adjudication"]["correct_ip_like_factor"] == +1.0
    assert receipt["adjudication"]["wrong_authority"].startswith(
        "IMAS Data Dictionary migration guide"
    )
    assert "sigma_Ip_eff=(+1)*(+1)=+1" in receipt["adjudication"]["reason"]
