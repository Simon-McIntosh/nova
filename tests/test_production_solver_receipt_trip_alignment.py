"""The certificate preserves trip identity across route-specific histories."""

from types import SimpleNamespace

import json
from pathlib import Path
import re

import numpy as np
import pytest

from benchmarks import carrier_step_attribution as carrier_step
from benchmarks import exact_placement_discriminator as placement_discriminator
from benchmarks import fieldnull_production_route as fieldnull_route
from benchmarks import plasma_cell_production_ladder as production_ladder
from benchmarks import solovev_certificate as certificate
from nova.equilibrium.fixed_point import FixedPointResult

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_RECEIPT = (
    ROOT / "docs/figures/gs-absolute-accuracy/production-solver-receipt/receipt.json"
)
SOURCE_REVISION = re.compile(r"[0-9a-f]{40}")


def _equilibrium(**fields):
    history = FixedPointResult(
        state=np.zeros(1),
        residual=np.asarray(0.125),
        trace=np.asarray([0.5, 0.25, 0.125]),
        active_set_iterations=3,
        active_set_residuals=np.asarray([0.5, 0.25, 0.125]),
        active_set_mask_differences=np.asarray([7, 3, 0, -1]),
    )._replace(**fields)
    continuation = SimpleNamespace(
        active=False,
        domain_name="none",
        form_name="none",
        continuity_name="none",
        support=0.0,
        decay_width=0.0,
        truncated_fraction=0.0,
    )
    return SimpleNamespace(
        fixed_point=history,
        continuation=SimpleNamespace(
            common_sol=continuation, private_flux=continuation
        ),
    )


def test_reduced_newton_preserves_trip_index_and_unavailable_damping():
    receipt = certificate._production_solver_receipt(_equilibrium())
    assert receipt["trip_count"] == 3
    assert receipt["per_trip_residual_history"] == [
        {
            "trip": 1,
            "live_relative_residual": 0.5,
            "mask_difference_cells": 7,
            "cycle_damping_activated": None,
        },
        {
            "trip": 2,
            "live_relative_residual": 0.25,
            "mask_difference_cells": 3,
            "cycle_damping_activated": None,
        },
        {
            "trip": 3,
            "live_relative_residual": 0.125,
            "mask_difference_cells": 0,
            "cycle_damping_activated": None,
        },
    ]
    assert receipt["globalisation_decisions"] == []
    assert receipt["promotion_globalisation"] == []


def test_padded_arrays_share_executed_trip_indices():
    receipt = certificate._production_solver_receipt(
        _equilibrium(
            active_set_residuals=np.asarray([0.5, 0.25, 0.125, np.nan, np.nan]),
            active_set_cycle_damping_activations=np.asarray([0, 1, -1, -1]),
        )
    )
    trips = receipt["per_trip_residual_history"]
    assert [trip["live_relative_residual"] for trip in trips] == [0.5, 0.25, 0.125]
    assert [trip["mask_difference_cells"] for trip in trips] == [7, 3, 0]
    assert [trip["cycle_damping_activated"] for trip in trips] == [False, True, None]


@pytest.mark.parametrize(
    "field,value",
    [
        ("active_set_residuals", np.asarray([0.5, 0.25])),
        ("active_set_mask_differences", np.asarray([7, 3])),
        ("active_set_cycle_damping_activations", np.asarray([0, 1])),
        ("active_set_cycle_damping_activations", np.asarray(0)),
        ("active_set_mask_differences", np.asarray([[7, 3, 0]])),
    ],
)
def test_incomplete_trip_telemetry_refuses_by_name(field, value):
    with pytest.raises(
        ValueError, match=f"production_solver_receipt_trip_alignment.*{field}"
    ):
        certificate._production_solver_receipt(_equilibrium(**{field: value}))


def test_empty_history_has_no_invented_trips():
    receipt = certificate._production_solver_receipt(
        _equilibrium(
            active_set_iterations=0,
            active_set_residuals=np.asarray(np.nan),
            active_set_mask_differences=-1,
        )
    )
    assert receipt["trip_count"] == 0
    assert receipt["per_trip_residual_history"] == []


def test_negative_trip_count_refuses_by_name():
    with pytest.raises(
        ValueError,
        match="production_solver_receipt_trip_alignment.*active_set_iterations",
    ):
        certificate._production_solver_receipt(_equilibrium(active_set_iterations=-1))


def _assert_source_revision(payload):
    """Assert the receipt names the 40-hex revision that produced it."""
    revision = payload.get("source_revision")
    assert isinstance(revision, str), (
        "a production-route receipt must name the source revision it was produced at"
    )
    assert SOURCE_REVISION.fullmatch(revision), (
        f"source_revision is not a 40-hex sha: {revision!r}"
    )


def test_committed_production_receipt_names_its_source_revision():
    """The committed receipt carries the revision it was measured at."""
    payload = json.loads(PRODUCTION_RECEIPT.read_text(encoding="utf-8"))
    _assert_source_revision(payload)


def test_production_solver_receipt_builder_stamps_source_revision():
    """The production-solver payload builder stamps the revision on its payload."""
    _assert_source_revision(certificate._production_solver_receipt(_equilibrium()))


def _synthetic_fieldnull_capture():
    """Return a minimal capture pair the fieldnull assembler accepts."""
    return {
        "revision": "a" * 40,
        "source_hashes": {"fixture": "b" * 64},
        "legacy": {
            "coordinates": [[0.0, 0.0], [1.0, 1.0]],
            "candidate_count": 49,
            "fixed_overflow": 0,
            "source_cells": 10,
            "matched_reference_count": 49,
            "extra_reference_count": 0,
        },
        "scientific": {
            "noise_only_resolved_false_positive_fields": 0,
            "generation_recall_from_cell_snr_0_4_minimum_family": 0.4,
            "resolved_recall_from_cell_snr_1_6_minimum_family": 1.0,
            "resolved_false_candidates": 0,
        },
        "timing": {
            "resident_fields_per_second": 1.0,
            "transfer_dispatch_fraction": 0.1,
            "double_buffered_hbm_fraction": 0.5,
        },
    }


def test_fieldnull_receipt_builder_stamps_source_revision(tmp_path):
    """fieldnull's own assemble() stamps the payload it returns."""
    capture = json.dumps(_synthetic_fieldnull_capture())
    cpu = tmp_path / "cpu.json"
    gpu = tmp_path / "gpu.json"
    cpu.write_text(capture, encoding="utf-8")
    gpu.write_text(capture, encoding="utf-8")
    _assert_source_revision(fieldnull_route.assemble(cpu, gpu))


def test_plasma_ladder_receipt_builder_stamps_source_revision(tmp_path):
    """The production-ladder assemble() stamps the payload it returns."""
    _assert_source_revision(production_ladder.assemble(tmp_path))


def test_carrier_receipt_builder_stamps_source_revision():
    """The carrier step-attribution receipt payload builder stamps the revision."""
    _assert_source_revision(carrier_step.receipt_payload({}))


def test_discriminator_receipt_builder_stamps_source_revision():
    """The exact-placement discriminator receipt builder stamps the revision."""
    _assert_source_revision(placement_discriminator.receipt_payload({}))
