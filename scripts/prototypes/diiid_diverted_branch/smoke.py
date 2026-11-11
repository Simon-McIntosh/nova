"""Exercise receipt readers and nested solve instrumentation without a solve."""

from __future__ import annotations

import argparse
import ast
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
READERS = (
    "scripts/prototypes/diiid_forward_convergence/measure_branches.py",
    "scripts/prototypes/diiid_material_mask/measure.py",
    "scripts/prototypes/diiid_residual_plateau/measure.py",
    "scripts/prototypes/diiid_stopping_rule/diagnose_step_gate.py",
    "scripts/prototypes/diiid_support_mask/trace_support.py",
)


def function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import jax
    import numpy as np

    from nova.equilibrium import solve_request
    from nova.equilibrium.solve_request import (
        ExplicitSolveSeed,
        ForwardSolveReceipt,
        ForwardSolveRequest,
    )
    from nova.equilibrium.topology import TopologyClass
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64
    assert jax.default_backend() == "cpu"
    # Both product revisions carry this same receipt API.
    for revision in (
        "04337b991e8aed9d0904494437e02a1b66fa24dc",
        "8fb2118e30f902fce846de297220c3703ac27417",
    ):
        source = subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "show",
                revision + ":nova/equilibrium/solve_request.py",
            ]
        )
        assert source == Path(solve_request.__file__).read_bytes()
    state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    fixed = SimpleNamespace(
        residual=0.125,
        converged=False,
        warmup_refused_proposals=np.array([2]),
        warmup_first_nonfinite_index=np.array([1]),
    )
    topology = SimpleNamespace(
        boundary_flux=1.0,
        axis_flux=2.0,
        axis=state[:2],
        x_point=state[:2],
        diverted=True,
    )
    equilibrium = SimpleNamespace(fixed_point=fixed, topology=topology, flux=state)
    values = {field.name: None for field in fields(ForwardSolveReceipt)}
    values.update(
        terminal_state=equilibrium,
        qualified=False,
        termination_reason=12,
        residual_history=np.array([1.0, 0.125]),
        changed_mask_entries=np.array([4, 0]),
        support_cell_count=np.array([4, 4]),
        globalisation_decisions=(np.array([1]), np.array([0.5])),
        amplitude_history=np.array([0.5]),
        attempted_newton_promotions=4,
        accepted_newton_promotions=2,
    )
    receipt = ForwardSolveReceipt(**values)
    assert not hasattr(receipt, "mask_history")
    helper = SimpleNamespace(_int_list=lambda values: [int(value) for value in values])
    reader_checks = {}
    for relative in READERS:
        path = ROOT / relative
        tree = ast.parse(path.read_text())
        assert not any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "receipt"
            and node.attr == "mask_history"
            for node in ast.walk(tree)
        )
        call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and any(
                isinstance(arg, ast.Attribute)
                and isinstance(arg.value, ast.Name)
                and arg.value.id == "receipt"
                and arg.attr == "changed_mask_entries"
                for arg in node.args
            )
        )
        result = eval(
            compile(ast.Expression(call), str(path), "eval"),
            dict(
                receipt=receipt,
                np=np,
                probe=helper,
                measure=helper,
                measure_branches=helper,
                _int_list=helper._int_list,
            ),
        )
        assert np.asarray(result).tolist() == [4, 0]
        reader_checks[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    namespace = dict(
        Any=Any,
        np=np,
        TopologyClass=TopologyClass,
        _finite_list=lambda values: np.asarray(values).tolist(),
        _int_list=helper._int_list,
        _inner_step_counts=lambda *_: {},
        _core_mask=lambda *_: np.ones(4, bool),
        _presence=lambda *_: {"fixture": True},
        classify_terminal=lambda *_: "fixture",
        inside_wall_units=lambda *_: np.ones(1, bool),
        _wall_distance=lambda *_: (0.0, 0.0),
        bench=SimpleNamespace(
            _assembled_boundary_geometry=lambda *_: (state.reshape(2, 2),)
        ),
    )
    record = function(ROOT / READERS[0], "_record_receipt", namespace)
    prepared = dict(
        profile=SimpleNamespace(
            lattice=SimpleNamespace(
                node_count=4, shape=(2, 2), radius=state[:2], height=state[:2]
            )
        ),
        wall=state.reshape(2, 2),
        target_current_a=1.0,
    )
    row = record(receipt, 0.0, prepared, TopologyClass.DIVERTED)
    assert row["mask_history"] == [4, 0] and row["status"] == "solved"
    assert row["terminal_residual"] == 0.125
    (args.output / "single-row-receipt.json").write_text(
        json.dumps(row, default=lambda value: np.asarray(value).tolist()) + "\n"
    )

    request = object.__new__(ForwardSolveRequest)
    object.__setattr__(request, "seed_policy", ExplicitSolveSeed(state))
    calls = []
    context = {}

    def backend(profile, supplied, *_a, **_kw):
        if isinstance(supplied, ForwardSolveRequest):
            calls.append("typed")
            assert (
                context["persisted"](profile, supplied.seed_policy.state) is equilibrium
            )
            return receipt
        calls.append("internal_array")
        assert np.array_equal(supplied, state)
        return equilibrium

    context.update(
        np=np,
        hashlib=hashlib,
        json=json,
        Path=Path,
        seeds=[],
        branches=[],
        ForwardSolveRequest=ForwardSolveRequest,
        solve=backend,
        args=SimpleNamespace(revision="receipt-smoke"),
        fixed_point=SimpleNamespace(__file__=solve_request.__file__),
        self=SimpleNamespace(output=args.output),
    )
    persisted = function(Path(__file__).with_name("measure.py"), "persisted", context)
    assert persisted(None, request) is receipt
    assert calls == ["typed", "internal_array"]
    saved = json.loads((args.output / "solve-receipt.json").read_text())
    assert saved["changed_mask_entries"] == [4, 0]
    summary = dict(
        passed=True,
        numerical_solves=0,
        single_row_receipt_read=True,
        typed_and_internal_calls=calls,
        reader_sha256=reader_checks,
        product_module=solve_request.__file__,
        product_api_identical_at_both_arms=True,
    )
    (args.output / "smoke.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RECEIPT_SMOKE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
