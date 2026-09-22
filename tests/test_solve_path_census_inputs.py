"""Solve receipts preserve the direct samples required by the null census."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.jax.config import configure_dtypes


_CENSUS_READS = {
    "read_qualification",
    "_fixed_design_read",
    "read_census",
    "candidate_table_status",
    "candidate_census",
    "grid",
    "locator",
}
_CONSTRAINT_PREFIX_ALLOWLIST = {
    (
        "nova/equilibrium/reduced_newton.py",
        "_constraint_qualification",
        2329,
        "_fixed_design_read",
    ),
}


def _prefix_sources(expression, aliases):
    sources = set()
    for node in ast.walk(expression):
        if isinstance(node, ast.Name):
            sources.update(aliases.get(node.id, ()))
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Slice)
            and node.slice.lower is None
            and isinstance(node.slice.upper, ast.Attribute)
            and node.slice.upper.attr == "physical_node_number"
        ):
            sources.add(node.lineno)
    return sources


def _prefix_census_calls(source, path):
    """Trace function-local prefix slices through aliases to census inputs."""
    tree = ast.parse(source)
    findings = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        aliases = {}
        events = sorted(
            (
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Assign | ast.AnnAssign | ast.Call)
            ),
            key=lambda node: (node.lineno, isinstance(node, ast.Call)),
        )
        for node in events:
            if isinstance(node, ast.Assign | ast.AnnAssign):
                value = node.value
                sources = (
                    _prefix_sources(value, aliases) if value is not None else set()
                )
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            aliases[name.id] = sources.copy()
                continue
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else (node.func.id if isinstance(node.func, ast.Name) else "")
            )
            if name not in _CENSUS_READS:
                continue
            operands = [*node.args, *(keyword.value for keyword in node.keywords)]
            sources = set().union(*(_prefix_sources(arg, aliases) for arg in operands))
            for line in sources:
                findings.append(
                    {
                        "path": path,
                        "function": function.name,
                        "slice_line": line,
                        "call_line": node.lineno,
                        "target": name,
                    }
                )
    return findings


def test_prefix_census_scanner_sees_the_known_positive_controls():
    for target in sorted(_CENSUS_READS):
        source = (
            "def read(operator, state):\n"
            "    physical = state[:operator.physical_node_number]\n"
            "    alias = physical\n"
            f"    return operator.{target}(alias)\n"
        )
        findings = _prefix_census_calls(source, "control.py")
        assert len(findings) == 1
        assert findings[0]["slice_line"] == 2
        assert findings[0]["target"] == target
    safe = (
        "def read(operator, state):\n"
        "    physical = state[:operator.physical_node_number]\n"
        "    grid, wall = operator.topology.split_flux_map(physical)\n"
        "    return operator.read_qualification(state)\n"
    )
    assert _prefix_census_calls(safe, "control.py") == []


def test_solve_path_prefix_slices_do_not_feed_the_null_census():
    root = Path(__file__).resolve().parents[1]
    paths = [
        "nova/equilibrium/forward.py",
        "nova/equilibrium/forward_operator.py",
        "nova/equilibrium/topology.py",
        "nova/equilibrium/reduced_newton.py",
    ]
    findings = [
        item
        for path in paths
        for item in _prefix_census_calls((root / path).read_text(), path)
    ]
    print("PREFIX_CENSUS_INPUTS " + json.dumps(findings, sort_keys=True), flush=True)
    assert not [item for item in findings if item["path"] == paths[0]]
    observed = {
        (item["path"], item["function"], item["slice_line"], item["target"])
        for item in findings
    }
    assert observed == _CONSTRAINT_PREFIX_ALLOWLIST


@pytest.fixture(
    scope="module",
    params=["diverted-single-null", "weak-rotation-reactor-static"],
    ids=["single-null", "discriminator-row"],
)
def analytic_carrier(request):
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    from benchmarks import solovev_certificate as certificate
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.forward_operator import (
        set_support_clip_mode,
        support_clip_mode,
    )
    from nova.equilibrium.stencil_mesh import StencilMesh
    from scripts.analytic_oracle_fixtures import measure as oracle_fixture
    from scripts.oracle_rebaseline import measure as recovery

    previous_mode = support_clip_mode()
    set_support_clip_mode("exact")
    try:
        case_name = request.param
        if case_name == "diverted-single-null":
            case_name = certificate.DIVERTED_CASE_NAME
        requested_cells = -110
        carrier_case, source_case, exact = certificate._case(case_name)
        machine = certificate._case_machine(
            case_name, carrier_case, exact, requested_cells
        )
        coordinates = np.vstack(
            (machine.node, machine.wall_node, machine.sample_coordinates)
        )
        analytic = certificate._exact_state(case_name, exact, coordinates)
        empty = oracle_fixture.forward_operator(source_case, machine)
        moments, exterior, _cache = oracle_fixture.cached_fixture_exterior(
            source_case, exact, machine, empty, analytic
        )
        operator = oracle_fixture.forward_operator(source_case, machine, exterior)
        profile = ForwardProfile(
            operator,
            StencilMesh(machine.node, machine.stencil, machine.area),
            newton_steps=recovery.NEWTON_STEPS,
        )
        target_current, _centroid, _current_receipt = (
            certificate._closed_form_current_target(
                case_name, source_case, operator, moments
            )
        )
        solve_request = certificate._certificate_solve_request(
            profile,
            jnp.asarray(analytic, dtype=jnp.float64),
            float(target_current),
            carrier_identity=f"analytic-hex:{case_name}:{requested_cells}",
        )
        solve_request = replace(
            solve_request, policy=replace(solve_request.policy, active_set_steps=1)
        )
        positive = operator._fixed_design_topology.grid.candidate_table_status(
            operator.null_flux_pool(jnp.asarray(analytic))
        )
        assert int(positive["retained_count"][0]) > 0
        print(
            f"SOLVE_INPUT_CONTROL case={case_name} realised={len(machine.node)} "
            f"state_values={len(analytic)} "
            f"physical_values={operator.physical_node_number} "
            f"retained={np.asarray(positive['retained_count']).tolist()}",
            flush=True,
        )
        yield SimpleNamespace(
            case_name=case_name,
            machine=machine,
            coordinates=coordinates,
            analytic=analytic,
            operator=operator,
            profile=profile,
            solve_request=solve_request,
            positive=positive,
        )
    finally:
        set_support_clip_mode(previous_mode)


def _finite_polish_count(polish):
    assert polish is not None
    position = np.asarray(polish["selected_position_rz"])
    value = np.asarray(polish["selected_value"])
    assert np.all(np.isfinite(position[0])) and np.isfinite(value[0])
    valid = np.isfinite(value)
    assert np.all(np.isfinite(position[valid]))
    return int(np.sum(valid))


def test_terminal_polish_receipt_keeps_direct_samples(analytic_carrier):
    case = analytic_carrier
    equilibrium = SimpleNamespace(
        flux=case.analytic,
        topology=SimpleNamespace(axis=case.positive["retained_candidate"][0, 0, :2]),
    )
    assert equilibrium.flux.size == case.operator.node_number
    assert equilibrium.flux.size > case.operator.physical_node_number
    polish = case.profile._terminal_polish_receipt(equilibrium)
    finite = _finite_polish_count(polish)
    print(
        f"TERMINAL_SEAM case={case.case_name} "
        f"state_values={equilibrium.flux.size} finite_landmarks={finite}",
        flush=True,
    )


def test_reduced_newton_acceptance_read_keeps_direct_samples(analytic_carrier):
    case = analytic_carrier
    _masks, topology, _connected, admitted = case.operator._fixed_design_read(
        jnp.asarray(case.analytic), None
    )
    assert bool(admitted)
    assert np.all(np.isfinite(np.asarray(topology.axis)))
    assert np.isfinite(float(topology.axis_flux))
    with pytest.raises(ValueError, match="requires the direct sampling flux values"):
        case.operator._fixed_design_read(
            jnp.asarray(case.analytic)[: case.operator.physical_node_number], None
        )
    print(
        f"ACCEPTANCE_SEAM case={case.case_name} axis_admitted={bool(admitted)} "
        f"state_values={len(case.analytic)} truncated_input_refused=True",
        flush=True,
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not any(device.platform == "gpu" for device in jax.devices()),
    reason="the full exact-mode solve requires a GPU backend",
)
def test_exact_mode_solve_retains_census_samples(analytic_carrier, monkeypatch):
    from nova.equilibrium.forward import ForwardProfile

    case = analytic_carrier
    operator = case.operator
    visits = []
    read_receipt = ForwardProfile._terminal_polish_receipt

    def observe_receipt(self, equilibrium):
        visits.append(equilibrium.flux.shape[0])
        print(f"TERMINAL_RECEIPT_INPUT values={visits[-1]}", flush=True)
        return read_receipt(self, equilibrium)

    monkeypatch.setattr(ForwardProfile, "_terminal_polish_receipt", observe_receipt)
    receipt = case.profile.solve(case.solve_request)
    assert visits == [operator.node_number]
    assert visits[0] > operator.physical_node_number
    finite = _finite_polish_count(receipt.polish_receipt)
    terminal = np.asarray(receipt.equilibrium.flux)
    assert np.all(np.isfinite(terminal))
    history = receipt.equilibrium.fixed_point
    print(
        f"SOLVE_POLISH_RECEIPT case={case.case_name} "
        f"realised={len(case.machine.node)} finite_landmarks={finite} "
        f"residual={float(history.residual):.12g} "
        f"converged={bool(history.converged)} backend={jax.default_backend()}",
        flush=True,
    )
    destination = os.environ.get("NOVA_SOLVE_PATH_EVIDENCE")
    if destination:
        output = Path(destination)
        output.mkdir(parents=True, exist_ok=True)
        _reference_masks, reference_topology = operator.read(case.analytic)
        _reference_o, reference_x = operator._fixed_design_topology.grid(
            operator.null_flux_pool(case.analytic)
        )
        _terminal_o, terminal_x = operator._fixed_design_topology.grid(
            operator.null_flux_pool(terminal)
        )
        np.savez(
            output / f"{case.case_name}.npz",
            coordinates=case.coordinates,
            grid_count=len(case.machine.node),
            analytic=case.analytic,
            terminal=terminal,
            wall=case.machine.wall_node,
            axis=np.asarray(receipt.equilibrium.topology.axis),
            x_point=np.asarray(receipt.equilibrium.topology.x_point),
            reference_axis=np.asarray(reference_topology.axis),
            reference_x_point=np.asarray(reference_topology.x_point),
            reference_candidates=np.asarray(reference_x[:, :2]),
            terminal_candidates=np.asarray(terminal_x[:, :2]),
            residual=float(history.residual),
            converged=bool(history.converged),
            finite_landmarks=finite,
        )
