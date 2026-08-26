import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "diiid_forward_gs_match.py"
SPEC = importlib.util.spec_from_file_location(
    "diiid_forward_gs_match_material", MODULE_PATH
)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def _diagnostic(class_margin):
    return {
        "class_margin": jnp.asarray(class_margin),
        "axis_flux": jnp.asarray(0.0),
        "outward_flux_span": jnp.asarray(1.0),
        "typed_candidates": jnp.zeros((1, 4)),
        "typed_candidate_present": jnp.zeros(1, dtype=bool),
        "selected_typed_candidate_index": jnp.asarray(0),
        "connectivity_candidates": jnp.zeros((1, 4)),
        "connectivity_candidate_present": jnp.zeros(1, dtype=bool),
        "connectivity_candidate_admitted": jnp.zeros(1, dtype=bool),
        "connectivity_candidate_resolved": jnp.zeros(1, dtype=bool),
        "connectivity_candidate_state": jnp.zeros(1, dtype=int),
        "connectivity_candidate_confidence": jnp.zeros(1),
        "connectivity_candidate_class_margin": jnp.zeros(1),
        "connectivity_candidate_boundary_snr": jnp.zeros(1),
        "connectivity_candidate_root_support_cell": jnp.zeros(1),
        "selected_typed_candidate": jnp.zeros(4),
        "selected_typed_candidate_present": jnp.asarray(False),
        "selected_x_normalized_flux_operand": jnp.asarray(jnp.nan),
        "wall_candidate": jnp.zeros(3),
        "wall_candidate_present": jnp.asarray(False),
        "wall_normalized_flux_operand_before_shadow": jnp.asarray(jnp.nan),
        "wall_normalized_flux_operand": jnp.asarray(jnp.nan),
        "wall_shadowed": jnp.asarray(False),
        "typed_candidate_count": jnp.asarray(0),
        "connectivity_admitted_slot_count": jnp.asarray(0),
        "connectivity_candidate_count_before_capacity": jnp.asarray(0),
        "connectivity_candidate_overflow": jnp.asarray(False),
        "connectivity_discarded_score_upper_bound": jnp.asarray(jnp.nan),
    }


def _inputs():
    coordinate = np.array([[1.0, -1.0], [1.0, 1.0], [2.0, -1.0], [2.0, 1.0]])
    repaired_material = jnp.array([True, False, True, True])
    axis = jnp.array([1.5, 0.0])
    calls = []

    def connectivity_axis_seed(received_axis):
        calls.append(received_axis)
        return jnp.asarray(2), repaired_material

    operator = SimpleNamespace(
        physical_node_number=6,
        grid=SimpleNamespace(coordinate=coordinate),
        wall=SimpleNamespace(coordinate=jnp.array([[0.8, -1.0], [0.8, 1.0]])),
        topology=SimpleNamespace(
            split_flux_map=lambda physical: (physical[:4], physical[4:])
        ),
        _fixed_design_topology=SimpleNamespace(
            grid=lambda grid_flux: (jnp.empty((0, 4)), jnp.empty((0, 4)))
        ),
        inside_material=jnp.array([False, False, False, False]),
        connectivity_axis_seed=connectivity_axis_seed,
    )
    topology = SimpleNamespace(
        axis=axis,
        wall_point=jnp.array([0.8, 0.0]),
        wall_point_flux=jnp.asarray(0.5),
        class_margin=0.25,
    )
    profile = SimpleNamespace(operator=operator)
    state = jnp.arange(6.0)
    return profile, state, topology, repaired_material, calls


def test_terminal_diagnostic_uses_axis_seeded_connectivity_material(monkeypatch):
    profile, state, topology, repaired_material, calls = _inputs()
    captured = {}

    def diagnostics(*args):
        captured["material"] = args[3]
        return _diagnostic(topology.class_margin)

    monkeypatch.setattr(gate, "traced_margin_candidate_diagnostics", diagnostics)

    serialized = gate._terminal_xpoint_diagnostics(profile, state, topology)

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], topology.axis)
    expected_material = repaired_material.reshape(2, 2).T
    np.testing.assert_array_equal(captured["material"], expected_material)
    assert serialized["class_margin_from_operands"] == topology.class_margin


def test_terminal_diagnostic_refuses_any_class_margin_mismatch(monkeypatch):
    profile, state, topology, _repaired_material, _calls = _inputs()
    changed_margin = np.nextafter(topology.class_margin, np.inf)
    monkeypatch.setattr(
        gate,
        "traced_margin_candidate_diagnostics",
        lambda *args: _diagnostic(changed_margin),
    )

    with pytest.raises(RuntimeError, match="changed the exact class-margin operand"):
        gate._terminal_xpoint_diagnostics(profile, state, topology)
