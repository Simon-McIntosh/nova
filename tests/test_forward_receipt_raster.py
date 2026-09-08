"""Receipt availability on tensor and unstructured forward carriers."""

from __future__ import annotations

import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import ForwardProfile
from nova.equilibrium.forward import RasterFluxReceiptStatus
from nova.equilibrium.stencil_mesh import StencilMesh
from scripts.analytic_oracle_fixtures import measure as oracle_fixture


def test_reduced_hex_request_receipt_records_absent_raster_flux() -> None:
    """The public request seam retains a receipt on the reduced hex oracle."""

    certificate.configure_dtypes()
    case_name = "strong-rotation-compact-static"
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = oracle_fixture.cached_machine(
        carrier_case,
        -110,
        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = certificate._exact_state(case_name, exact, coordinates)
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical = oracle_fixture.exact_current_moments(
        source_case,
        empty_operator,
        oracle_state,
    )
    exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
    exact_internal = oracle_fixture._internal_flux_image(
        empty_operator,
        exact_coefficients,
    )
    operator = oracle_fixture.forward_operator(
        source_case,
        machine,
        oracle_state - exact_internal,
    )
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=certificate.recovery.NEWTON_STEPS,
    )
    target_current, centroid, current_receipt = certificate._closed_form_current_target(
        case_name,
        source_case,
        operator,
        exact_physical,
    )
    seed, _requested_class, _seed_receipt = certificate._production_seed(
        profile,
        case_name,
        target_current,
        centroid,
        current_receipt,
    )

    result = profile.solve(
        certificate._certificate_solve_request(
            profile,
            seed,
            target_current,
            carrier_identity="reduced-hex-raster-receipt",
        )
    )

    assert result.equilibrium.raster_flux is None
    assert int(result.equilibrium.raster_flux_status) == int(
        RasterFluxReceiptStatus.UNAVAILABLE_NON_TENSOR_PRODUCT_GRID
    )
