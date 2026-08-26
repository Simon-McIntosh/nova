"""Focused contract tests for the MAST EFIT corroboration adapter."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.boundary_comparison import BoundaryMode


SCRIPT = (
    Path(__file__).parents[1]
    / "docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py"
)


def _adapter():
    spec = spec_from_file_location("efit_topology_corroboration_adapter", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ring() -> np.ndarray:
    return np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)))


def _line(start, end):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    return np.stack((start, (2 * start + end) / 3, (start + 2 * end) / 3, end))


def _operand(**overrides):
    row = {
        "identity": "22086/43",
        "shot": 22086,
        "slice_index": 43,
        "time_s": 0.25,
        "arm": "pure",
        "converged": True,
        "terminal_residual": 1.0e-9,
        "tolerance": 1.0e-8,
        "termination_reason": "converged",
        "efit_label": "diverted",
        "nova_achieved_class": "diverted",
        "radius": np.linspace(0.0, 1.0, 3),
        "height": np.linspace(0.0, 1.0, 3),
        "flux": np.zeros((3, 3)),
        "axis": np.asarray((0.5, 0.5)),
        "wall": _ring(),
        "binding_flux": 0.0,
        "selected_saddle": np.asarray((1.0, 0.0)),
        "limiter_coordinate": np.asarray((0.0, 1.0)),
        "class_margin": 0.25,
        "efit_lcfs": _ring(),
        "efit_x_points": np.asarray(((1.0, 0.0),)),
    }
    row.update(overrides)
    return row


def test_branch_adapter_uses_only_valid_rows_and_keeps_open_legs(monkeypatch):
    adapter = _adapter()
    closed = np.stack(
        (
            _line((0.0, 0.0), (1.0, 0.0)),
            _line((1.0, 0.0), (1.0, 1.0)),
            _line((1.0, 1.0), (0.0, 1.0)),
            _line((0.0, 1.0), (0.0, 0.0)),
            np.full((4, 2), 999.0),
        )
    )
    open_controls = np.zeros((2, 2, 4, 2))
    open_controls[0, 0] = _line((0.0, 0.0), (0.0, -4.0))
    open_controls[0, 1] = 777.0
    open_controls[1] = 888.0
    assembled = {
        "well_formed": True,
        "closed_controls_rz": closed,
        "closed_valid": np.asarray((True, True, True, True, False)),
        "open_controls_rz": open_controls,
        "open_valid": np.asarray(((True, False), (False, False))),
        "open_branch_valid": np.asarray((True, False)),
    }
    monkeypatch.setattr(
        adapter, "assemble_separatrix_branches", lambda *args: assembled
    )

    boundary, legs = adapter._assembled_branch_polylines(
        {
            "radius": np.linspace(0.0, 1.0, 3),
            "height": np.linspace(0.0, 1.0, 3),
            "flux": np.zeros((3, 3)),
            "axis": np.asarray((0.5, 0.5)),
            "boundary_flux": 0.0,
        }
    )

    assert boundary is not None
    assert np.max(np.abs(boundary)) <= np.nextafter(1.0, 2.0)
    assert len(legs) == 1
    assert legs[0][-1].tolist() == [0.0, -4.0]
    assert np.max(np.abs(legs[0])) < 777.0


def test_open_leg_extent_is_serialized_but_cannot_influence_shared_metrics(
    monkeypatch,
):
    adapter = _adapter()
    ring = _ring()
    long_leg = np.asarray(((0.0, 0.0), (0.0, -1000.0)))
    monkeypatch.setattr(
        adapter, "_assembled_branch_polylines", lambda geometry: (ring, [long_leg])
    )

    row = adapter._score_operand(_operand())

    assert row["binding_to_efit_lcfs_sup_m"] == 0.0
    assert row["binding_to_efit_lcfs_rms_m"] == 0.0
    assert row["selected_saddle_to_efit_x_point_m"] == 0.0
    assert row["nova_open_legs_m"] == [long_leg.tolist()]
    assert row["comparison_failures"] == []


def test_margin_classification_ignores_cached_achieved_label(monkeypatch):
    adapter = _adapter()
    monkeypatch.setattr(
        adapter, "_assembled_branch_polylines", lambda geometry: (_ring(), [])
    )

    row = adapter._score_operand(
        _operand(
            class_margin=-0.5,
            efit_label="limited",
            nova_achieved_class="diverted",
        )
    )

    assert row["nova_achieved_class"] == "limited"
    assert row["label_agreement"] is True


def test_all_missing_comparator_inputs_remain_one_strict_json_bank_row(monkeypatch):
    adapter = _adapter()
    monkeypatch.setattr(
        adapter, "_assembled_branch_polylines", lambda geometry: (None, [])
    )

    row = adapter._score_operand(
        _operand(
            efit_label=None,
            class_margin=None,
            selected_saddle=None,
            efit_lcfs=None,
            efit_x_points=None,
        )
    )

    assert row["identity"] == "22086/43"
    assert row["comparison_failures"] == [
        "missing_predicted_closed_boundary",
        "missing_reference_closed_boundary",
        "missing_achieved_topology_class",
        "missing_reference_topology_class",
        "missing_predicted_saddle",
        "missing_reference_x_points",
    ]
    assert row["nova_closed_boundary_m"] is None
    assert row["binding_to_efit_lcfs_sup_m"] is None
    assert row["label_agreement"] is None
    json.dumps(row, allow_nan=False)


def test_convergence_receipts_qualify_rms_without_shrinking_cohort(
    monkeypatch, tmp_path
):
    adapter = _adapter()
    monkeypatch.setattr(
        adapter, "_assembled_branch_polylines", lambda geometry: (_ring(), [])
    )
    monkeypatch.setattr(adapter, "CACHE_PATH", tmp_path / "operands.npz")

    operand_rows = [_operand()]
    operand_rows.append(
        _operand(
            identity="22086/43",
            arm="mixed",
            converged=False,
            terminal_residual=2.5e-3,
            tolerance=1.0e-8,
            termination_reason="iteration_budget_exhausted",
        )
    )
    operand_rows.extend(
        _operand(identity=f"synthetic/{index}", arm="pure") for index in range(10)
    )
    carrier = {"carrier": {"semantic_response_identity": "synthetic-carrier"}}
    adapter._write_operand_cache(operand_rows, carrier)
    restored = adapter._read_operand_cache(carrier)
    assert all(
        {"converged", "terminal_residual", "tolerance", "termination_reason"}
        <= row.keys()
        for row in restored
    )
    converged, nonconverged = [adapter._score_operand(row) for row in restored[:2]]
    eligibility = adapter._rms_threshold_eligibility([converged, nonconverged])

    assert converged["converged"] is True
    assert converged["terminal_residual"] == 1.0e-9
    assert converged["tolerance"] == 1.0e-8
    assert converged["termination_reason"] == "converged"
    assert converged["qualified_terminal"] is True
    assert converged["rms_threshold_eligible"] is True
    assert nonconverged["converged"] is False
    assert nonconverged["terminal_residual"] == 2.5e-3
    assert nonconverged["tolerance"] == 1.0e-8
    assert nonconverged["termination_reason"] == "iteration_budget_exhausted"
    assert nonconverged["qualified_terminal"] is False
    assert nonconverged["rms_threshold_eligible"] is False
    assert eligibility == {
        "eligible_count": 1,
        "declared_arm_denominator": 2,
        "eligible_arms": ["22086/43 pure"],
        "excluded_nonconverged_arms": ["22086/43 mixed"],
    }
    json.dumps([converged, nonconverged], allow_nan=False)


def test_post_cutover_geometry_routes_class_through_public_classifier(monkeypatch):
    adapter = _adapter()
    observed = []
    observed_material = []
    monkeypatch.setattr(
        adapter,
        "classify_boundary_mode",
        lambda margin: observed.append(margin) or BoundaryMode.LIMITED,
    )

    def diagnostics(*args):
        observed_material.append(args[3])
        return {
            "selected_typed_candidate": jnp.asarray((0.8, -0.6, 3.0)),
            "class_margin": jnp.asarray(0.75),
            "limiter_coordinate": jnp.asarray((0.4, 1.1)),
            "limiter_flux": jnp.asarray(2.0),
        }

    monkeypatch.setattr(
        adapter,
        "traced_margin_candidate_diagnostics",
        diagnostics,
    )
    coordinate = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    connectivity_material = jnp.asarray((True, False, False, True))
    operator = SimpleNamespace(
        physical_node_number=4,
        grid=SimpleNamespace(coordinate=coordinate),
        topology=SimpleNamespace(
            split_flux_map=lambda physical: (physical, jnp.asarray((0.0, 0.0)))
        ),
        _fixed_design_topology=SimpleNamespace(
            grid=lambda flux: (None, jnp.zeros((1, 4)))
        ),
        connectivity_axis_seed=lambda axis: (axis, connectivity_material),
        inside_material=jnp.zeros(4, dtype=bool),
        wall=SimpleNamespace(coordinate=np.asarray(((0.0, 0.0), (1.0, 0.0)))),
    )
    profile = SimpleNamespace(operator=operator)
    topology = SimpleNamespace(
        wall_point=jnp.asarray((0.4, 1.1)),
        wall_point_flux=jnp.asarray(2.0),
        axis=jnp.asarray((0.5, 0.5)),
    )

    result = adapter._post_cutover_geometry(profile, jnp.arange(4.0), topology)

    assert observed == [0.75]
    np.testing.assert_array_equal(
        observed_material, connectivity_material.reshape((2, 2)).T[None]
    )
    assert result["achieved_class"] == "limited"
    assert result["binding_flux"] == 2.0


@pytest.mark.parametrize(
    "exception_type",
    ("NoQualifiedAxisError", "ConstraintViolationError"),
)
def test_named_arm_failure_retains_twelve_row_census(
    monkeypatch, tmp_path, exception_type
):
    adapter = _adapter()
    exception_class = getattr(adapter, exception_type)
    ring = _ring()

    def grid_geometry(_profile, state):
        if state == 5 and exception_type == "NoQualifiedAxisError":
            raise exception_class("synthetic axis disqualification")
        return {
            "radius": np.linspace(0.0, 1.0, 3),
            "height": np.linspace(0.0, 1.0, 3),
            "flux": np.zeros((3, 3)),
            "axis": np.asarray((0.5, 0.5)),
            "wall": ring,
        }

    def operator_read(state):
        if state == 5 and exception_type == "ConstraintViolationError":
            raise exception_class("synthetic constraint violation")
        return None, SimpleNamespace()

    reachability = SimpleNamespace(_grid_geometry=grid_geometry)
    profile = SimpleNamespace(operator=SimpleNamespace(read=operator_read))
    monkeypatch.setattr(
        adapter,
        "_post_cutover_geometry",
        lambda *_args: {
            "achieved_class": "diverted",
            "binding_flux": 0.0,
            "selected_saddle": np.asarray((1.0, 0.0)),
            "limiter_coordinate": np.asarray((0.0, 1.0)),
            "class_margin": 0.25,
        },
    )
    monkeypatch.setattr(
        adapter, "_assembled_branch_polylines", lambda _geometry: (ring, [])
    )
    monkeypatch.setattr(adapter, "CACHE_PATH", tmp_path / "operands.npz")

    operands = [
        adapter._build_arm_operand(
            reachability,
            profile,
            SimpleNamespace(
                state=index,
                converged=True,
                terminal_residual=1.0e-9,
                tolerance=1.0e-8,
                termination_reason="converged",
            ),
            identity=f"synthetic/{index // 2}",
            shot=20_000 + index // 2,
            slice_index=index // 2,
            time_s=0.1 * index,
            arm="pure" if index % 2 == 0 else "mixed",
            efit_label="diverted",
            efit_lcfs=ring,
            efit_x_points=np.asarray(((1.0, 0.0),)),
        )
        for index in range(12)
    ]
    carrier = {"carrier": {"semantic_response_identity": "synthetic-carrier"}}
    adapter._write_operand_cache(operands, carrier)
    rows = [
        adapter._score_operand(operand)
        for operand in adapter._read_operand_cache(carrier)
    ]
    eligibility = adapter._rms_threshold_eligibility(rows)
    failed = rows[5]

    assert len(rows) == 12
    assert [(row["identity"], row["arm"]) for row in rows] == [
        (f"synthetic/{index // 2}", "pure" if index % 2 == 0 else "mixed")
        for index in range(12)
    ]
    assert failed["converged"] is False
    assert failed["termination_reason"] == exception_type
    assert failed["failure_exception_class"] == exception_type
    assert failed["terminal_residual"] is None
    assert failed["nova_achieved_class"] is None
    assert failed["nova_closed_boundary_m"] is None
    assert failed["binding_to_efit_lcfs_rms_m"] is None
    assert failed["label_agreement"] is None
    assert eligibility["declared_arm_denominator"] == 12
    assert eligibility["eligible_count"] == 11
    assert eligibility["excluded_nonconverged_arms"] == ["synthetic/2 mixed"]
    json.dumps(rows, allow_nan=False)


def test_unrelated_arm_geometry_exception_propagates(monkeypatch):
    adapter = _adapter()
    reachability = SimpleNamespace(
        _grid_geometry=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("unrelated failure")
        )
    )
    arm_result = SimpleNamespace(
        state=0,
        converged=True,
        terminal_residual=1.0e-9,
        tolerance=1.0e-8,
        termination_reason="converged",
    )

    with pytest.raises(RuntimeError, match="unrelated failure"):
        adapter._build_arm_operand(
            reachability,
            SimpleNamespace(),
            arm_result,
            identity="synthetic/0",
            shot=20_000,
            slice_index=0,
            time_s=0.0,
            arm="pure",
            efit_label="diverted",
            efit_lcfs=_ring(),
            efit_x_points=np.asarray(((1.0, 0.0),)),
        )
