"""Build each corpus adapter call and compare its registered frame solve.

Callers run to the adapter boundary, where their actual profile and current
arguments are captured. Both adapter revisions complete that same input. Each
completed profile then goes through the benchmark's registered typed-request
solve, preserving its caller's wall and response geometry. Driver-specific
route repertoires and scoring are outside this current-adapter measurement.
The saved-state operator reader has no callable corpus preparation entry; its
build-through-adapter statements are executed from its parsed source instead.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
from contextlib import contextmanager
from pathlib import Path

import jax
import numpy as np

from nova.jax.config import configure_dtypes
from scripts.prototypes.diiid_adapter_current_refusal.refusal_arms import (
    FRAME,
    ROOT,
    _corpus_row,
    _digest,
    _module,
    _write,
)


class Captured(Exception):
    """Return a caller's arguments before it advances to its solve or scores."""

    def __init__(self, profile, options):
        self.profile = profile
        self.options = options
        super().__init__("captured the current-adapter boundary")


def _capture(profile, **options):
    raise Captured(profile, options)


@contextmanager
def _patched(module):
    original = module.complete_profile_current_adapter
    module.complete_profile_current_adapter = _capture
    try:
        yield
    finally:
        module.complete_profile_current_adapter = original


def _inventory():
    found = set()
    for folder in ("benchmarks", "scripts"):
        for path in (ROOT / folder).rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith("scripts/prototypes/diiid_adapter_current_refusal/"):
                continue
            tree = ast.parse(path.read_text(), filename=relative)
            for function in ast.walk(tree):
                if not isinstance(function, ast.FunctionDef):
                    continue
                for call in ast.walk(function):
                    if not isinstance(call, ast.Call):
                        continue
                    name = getattr(call.func, "id", getattr(call.func, "attr", ""))
                    if name == "complete_profile_current_adapter":
                        found.add((relative, function.name))
    # This caller constructs synthetic targets and has no corpus or solve.
    synthetic = ("scripts/prototypes/diiid_finite_edge_booking/seam_check.py", "main")
    if synthetic not in found:
        raise AssertionError("inventory missed the known-present synthetic caller")
    found.remove(synthetic)
    if ("benchmarks/diiid_forward_gs_match.py", "solve_frame") not in found:
        raise AssertionError("inventory missed the known-present benchmark caller")
    return found, synthetic


def _operator_reader(bench, row, path):
    """Execute the saved-state reader's actual build-through-adapter statements."""

    module = importlib.import_module(
        "scripts.prototypes.diiid_xpoint_attribution.operator_probe"
    )
    tree = ast.parse(Path(module.__file__).read_text())
    function = next(node for node in tree.body if getattr(node, "name", "") == "probe")
    loop = next(node for node in function.body if isinstance(node, ast.For))
    statements = []
    started = False
    for statement in loop.body:
        targets = getattr(statement, "targets", [])
        target = getattr(targets[0], "id", "") if targets else ""
        if target == "built":
            started = True
        if started:
            statements.append(statement)
        if target == "adapter":
            break
    if not statements or target != "adapter":
        raise AssertionError("saved-state reader's adapter prefix was not found")
    namespace = dict(vars(module), bench=bench, row=row, frame=FRAME)
    namespace["result"] = {
        "archive": str(path),
        "time_ms": float(row["efit_times"][FRAME]),
    }
    prefix = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))
    exec(compile(prefix, str(module.__file__), "exec"), namespace)


def _entries(bench, row, path, profile, time_ms):
    pinned = importlib.import_module("benchmarks.diiid_current_pinned_forward")
    diverted = importlib.import_module("benchmarks.diiid_diverted_root_full_currents")
    repaired = importlib.import_module("benchmarks.diiid_repaired_solve_remeasure")
    cold = importlib.import_module("benchmarks.diiid_constrained_cold_start")
    strict = importlib.import_module("benchmarks.strict_exit_incidence")
    circuit = importlib.import_module(
        "benchmarks.diiid_circuit_driven_forward_validation"
    )
    fidelity = importlib.import_module("scripts.diiid_map_fidelity.measure")
    script_directory = ROOT / "scripts/prototypes/diiid_forward_convergence"
    sys.path.insert(0, str(script_directory))
    try:
        convergence = importlib.import_module(
            "scripts.prototypes.diiid_forward_convergence.measure_branches"
        )
    finally:
        sys.path.pop(0)
    propagation = importlib.import_module(
        "scripts.prototypes.diiid_external_current_propagation.measure"
    )
    expansion = bench.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION
    selected = circuit.SelectedFrame(
        path, FRAME, time_ms, bench._target_current(row, time_ms)
    )
    cold._RESPONSE_CACHE.clear()
    return [
        (bench, "solve_frame", lambda: bench.solve_frame(row, FRAME, expansion)),
        (
            bench,
            "_solve_wall_topology_frame",
            lambda: bench._solve_wall_topology_frame(
                row,
                FRAME,
                None,
                bench.DEFAULT_MACHINE_ARTIFACT_CACHE,
                bench.DEFAULT_MACHINE_ARTIFACT_DIGEST,
            ),
        ),
        (
            pinned,
            "description_driven_currents",
            lambda: pinned.description_driven_currents(row, profile, time_ms),
        ),
        (
            diverted,
            "description_driven_currents",
            lambda: diverted.description_driven_currents(row, profile, time_ms),
        ),
        (repaired, "_prepare_frame", lambda: repaired._prepare_frame(row, FRAME)),
        (cold, "_fixed_wiring_adapter", lambda: cold.prepare_frame(path, FRAME)),
        (
            strict,
            "_build_diiid_member_row",
            lambda: strict._build_diiid_member_row(
                {"shot": path.name, "frame": FRAME},
                number=1,
                machine_cache=bench.DEFAULT_MACHINE_ARTIFACT_CACHE,
            ),
        ),
        (circuit, "solve_frame", lambda: circuit.solve_frame(selected)),
        (bench, "prepare_diiid", lambda: fidelity.prepare_diiid(FRAME)),
        (bench, "_prepare_frame", lambda: convergence._prepare_frame(FRAME)),
        (
            importlib.import_module("nova.imas.diiid_current"),
            "_adapted",
            lambda: propagation._adapted(bench, row, expansion),
        ),
        (bench, "probe", lambda: _operator_reader(bench, row, path)),
    ]


def _identity(module, function):
    # The function's owner can differ from the module holding its adapter alias.
    owners = {
        "prepare_diiid": "scripts/diiid_map_fidelity/measure.py",
        "_adapted": "scripts/prototypes/diiid_external_current_propagation/measure.py",
        "probe": "scripts/prototypes/diiid_xpoint_attribution/operator_probe.py",
    }
    if (
        module.__name__ == "benchmarks.diiid_forward_gs_match"
        and function == "_prepare_frame"
    ):
        return (
            "scripts/prototypes/diiid_forward_convergence/measure_branches.py",
            function,
        )
    return owners.get(
        function, Path(module.__file__).relative_to(ROOT).as_posix()
    ), function


def _solve(bench, adapter, row):
    current = np.asarray(adapter.resolution.current(()), dtype=float)
    if current.size != 24 or adapter.resolution.unknown_names:
        raise AssertionError("caller did not complete all 24 conductor currents")
    equilibrium, termination, selection, iterations, _seed, converged, defaults = (
        bench._solve_registered(
            adapter.profile,
            np.asarray([]),
            row,
            FRAME,
            current,
            bench._target_current(row, float(row["efit_times"][FRAME])),
        )
    )
    flux = np.asarray(equilibrium.flux, dtype=float)
    return flux, {
        "flux_digest": _digest(flux),
        "flux_shape": list(flux.shape),
        "finite": bool(np.all(np.isfinite(flux))),
        "converged": bool(converged),
        "residual": float(equilibrium.fixed_point.residual),
        "termination": termination,
        "iterations": iterations,
        "selection": selection,
        "resolved_defaults": defaults,
        "current_a": current.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-module", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    configure_dtypes()
    assert jax.config.jax_enable_x64
    from benchmarks import diiid_forward_gs_match as bench

    head = _module("tree")
    base = _module(options.base_module)
    inventory, synthetic = _inventory()
    row, path = _corpus_row(bench)
    profile, *_rest = bench.build_profile(
        row, FRAME, bench.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION
    )
    entries = _entries(bench, row, path, profile, float(row["efit_times"][FRAME]))
    covered = {_identity(module, function) for module, function, _ in entries}
    if covered != inventory:
        raise AssertionError(
            f"caller inventory differs: missing={inventory - covered}, "
            f"extra={covered - inventory}"
        )
    _write(
        options.output,
        "inventory",
        {
            "corpus_callers": sorted(inventory),
            "synthetic_caller": synthetic,
            "measurement": (
                "each caller's adapter input and registered typed-request solve"
            ),
            "head_module": str(head.__file__),
            "base_module": str(base.__file__),
        },
    )
    results = []
    for index, (module, function, invoke) in enumerate(entries):
        identity = _identity(module, function)
        name = f"caller-{index:02d}"
        record = {"caller": identity, "status": "started"}
        _write(options.output, name, record)
        try:
            with _patched(module):
                try:
                    invoke()
                except Captured as captured:
                    original_profile, kwargs = captured.profile, captured.options
                else:
                    raise AssertionError("caller did not reach the adapter")
            head_adapter = head.complete_profile_current_adapter(
                original_profile, **kwargs
            )
            base_adapter = base.complete_profile_current_adapter(
                original_profile, **kwargs
            )
            head_flux, record["head"] = _solve(bench, head_adapter, row)
            _write(options.output, name, record)
            base_flux, record["base"] = _solve(bench, base_adapter, row)
            identical = np.array_equal(head_flux, base_flux)
            record["terminal_flux_identical"] = bool(identical)
            record["max_abs_flux_difference_wb"] = float(
                np.max(np.abs(head_flux - base_flux))
            )
            record["status"] = (
                "passed" if identical and record["head"]["finite"] else "failed"
            )
            np.savez_compressed(
                options.output / (name + ".npz"), head=head_flux, base=base_flux
            )
        except Exception as error:
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
        results.append(record)
        _write(options.output, name, record)
        if record["status"] != "passed":
            break
    passed = sum(record["status"] == "passed" for record in results)
    _write(
        options.output,
        "summary",
        {
            "expected": len(entries),
            "passed": passed,
            "failed": [
                record["caller"] for record in results if record["status"] != "passed"
            ],
            "completed": len(results) == len(entries),
        },
    )
    return 0 if passed == len(entries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
