from math import tau

import numpy as np

from benchmarks import diiid_corpus_conventions as cocos


def _frame() -> cocos.DiscriminatorFrame:
    return cocos.DiscriminatorFrame(
        shot="shot.parquet",
        frame=3,
        time_ms=1000.0,
        plasma_current_ka=1000.0,
        bcoil=-100.0,
        q95=4.0,
        psi_axis_wb_per_rad=-0.3,
        psi_boundary_wb_per_rad=-0.05,
        current_from_per_radian_a=1.05e6,
        current_from_total_flux_a=1.05e6 / tau,
    )


def test_pinned_transform_maps_cocos_five_to_seventeen():
    assert cocos.CORPUS_COCOS == 5
    assert cocos.NOVA_COCOS == 17
    assert cocos.FIXED_FLUX_SIGN == -1.0
    assert cocos.TOTAL_FLUX_FACTOR == tau
    assert cocos.PSI_TO_NOVA == -tau
    assert cocos.IP_TO_NOVA == 1.0
    assert cocos.F_TO_NOVA == 1.0
    assert cocos.Q_TO_NOVA == -1.0
    assert cocos.D_PSI_TO_NOVA == -1.0 / tau


def test_flux_transform_round_trips_exactly_once():
    source = np.array([-0.4, -0.1, 0.2])
    nova = cocos.corpus_flux_to_nova_total(source)
    np.testing.assert_allclose(nova, -tau * source)
    np.testing.assert_allclose(cocos.nova_total_flux_to_corpus(nova), source)


def test_numeric_discriminators_identify_cocos_five():
    receipt = cocos.summarize([_frame() for _ in range(20)])
    assert receipt["corpus_cocos"] == 5
    assert receipt["measured_digits"] == {
        "sigma_bp": 1,
        "e_bp": 0,
        "sigma_r_phi_z": 1,
        "sigma_rho_theta_phi": -1,
    }
    discriminators = receipt["discriminators"]
    assert discriminators["psi_relative_to_ip"]["sigma_bp_positive"] == 20
    assert discriminators["q95_sign"]["sigma_rho_theta_phi_negative"] == 20
    assert discriminators["axis_to_boundary_psi_wb_per_rad"]["increasing"] == 20
    assert (
        discriminators["current_integral_ratio_to_recorded_ip"]["corpus_as_per_radian"][
            "median"
        ]
        == 1.05
    )
