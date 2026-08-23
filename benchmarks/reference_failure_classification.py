"""Classify the reference-suite failures against their intended contracts.

The benchmark executes the four registered test functions unchanged.  Their
expensive immutable carrier is restored through the suite's native cache and
validated byte for byte before the fixtures consume it; every solve, trace and
assertion remains the test module's own implementation.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pytest

from nova.equilibrium import fixed_point
from nova.jax.config import configure_dtypes
from tests import test_equilibrium_forward_reference as reference


DEFAULT_RECEIPT = Path(
    "docs/figures/forward-operator-refinement/reference-failure-classification.json"
)
TEST_MODULE = "tests/test_equilibrium_forward_reference.py"
TEST_NAMES = (
    "test_the_passive_closure_moves_the_reproduction_by_a_tenth_of_a_percent",
    "test_the_relaxed_routes_leave_the_equilibrium_on_a_bounded_budget",
    "test_the_relaxed_route_walks_down_the_vertical_instability",
    "test_the_published_solve_runs_on_the_production_mesh",
)


class _ResultRecorder:
    """Keep the call result for each selected pytest item."""

    def __init__(self) -> None:
        self.results: dict[str, dict[str, object]] = {}

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Record only the assertion-bearing call phase."""
        if report.when != "call":
            return
        name = report.nodeid.rsplit("::", 1)[-1]
        if name not in TEST_NAMES:
            return
        self.results[name] = {
            "outcome": report.outcome,
            "duration_seconds": float(report.duration),
        }


def _git_output(*arguments: str) -> str:
    """Return one stable Git identity string."""
    return subprocess.check_output(
        ("git", *arguments), text=True, stderr=subprocess.DEVNULL
    ).strip()


def _install_cached_carrier():
    """Route the suite's immutable machine request through its validated cache."""

    @lru_cache(maxsize=4)
    def machine(cells: int, passive: bool):
        configure_dtypes()
        case = reference.require_reference()
        return case, reference.cached_machine(case, cells, passive=passive)

    reference._machine = machine
    reference._solved.cache_clear()
    reference._published.cache_clear()
    return machine


def _closure_measurement(solved) -> dict[str, object]:
    """Reproduce the passive-closure assertion operands."""
    without_current = solved.machine.source_current.copy()
    without_current[-solved.machine.passive_columns :] = 0.0
    without_machine = reference.replace(
        solved.machine, source_current=without_current, passive_columns=0
    )
    without = reference.solve(solved.case, without_machine).deviations()
    structure = solved.deviations()
    closure = without["flux sup-norm"] - structure["flux sup-norm"]
    return {
        "prior_observation_percentage_points": 0.6098,
        "current_closure_percentage_points": float(closure),
        "registered_ceiling_percentage_points": float(
            reference.PASSIVE_REPRODUCTION_MOVE_CEILING
        ),
        "without_passives_flux_deviation_percent": float(
            without["flux sup-norm"]
        ),
        "with_passives_flux_deviation_percent": float(
            structure["flux sup-norm"]
        ),
    }


def _relaxed_measurement(profile, equilibrium) -> dict[str, object]:
    """Reproduce both fixed-budget mixed-route traces."""
    traces = {}
    mapped = profile.flux_map()
    for scheme in (fixed_point.picard, fixed_point.anderson):
        history = scheme(
            mapped,
            equilibrium.flux,
            evaluations=reference.RELAXED_EVALUATIONS,
            relaxation=reference.RELAXATION,
        )
        trace = np.asarray(history.trace)
        traces[scheme.__name__] = {
            "samples": int(trace.size),
            "initial_relative_residual": float(trace[0]),
            "terminal_relative_residual": float(trace[-1]),
            "terminal_over_initial": float(trace[-1] / trace[0]),
            "finite": bool(np.all(np.isfinite(trace))),
        }
    return {
        "prior_anderson_terminal_relative_residual": 5.25e-14,
        "registered_terminal_growth_factor": float(reference.DRIFT_GROWTH),
        "schemes": traces,
    }


def _branch_measurement(solved) -> dict[str, object]:
    """Reproduce the fixed-budget Picard branch-selection trace."""
    operator = reference.forward_operator(solved.case, solved.machine)
    history = fixed_point.picard(
        operator.flux_map(),
        reference.seed_flux(solved.case, solved.machine),
        evaluations=reference.RELAXED_EVALUATIONS,
        relaxation=reference.RELAXATION,
    )
    trace = np.asarray(history.trace)
    width = reference.RELAXED_EVALUATIONS // 4
    early = trace[:width]
    tail = trace[-width:]
    masks, topology = operator.read(history.state)
    current = operator.source.cell_current(
        np.asarray(solved.machine.radius), operator.area, masks
    )
    return {
        "prior_early_relative_residual": {
            "initial": 0.01486,
            "terminal": 0.00845,
        },
        "current_early_relative_residual": {
            "initial": float(early[0]),
            "terminal": float(early[-1]),
        },
        "trace_maximum": float(np.max(trace)),
        "tail_relative_residual": {
            "initial": float(tail[0]),
            "terminal": float(tail[-1]),
        },
        "terminal_core_cells": int(np.asarray(masks.core).sum()),
        "terminal_diverted": bool(topology.diverted),
        "terminal_plasma_current_a": float(np.sum(np.asarray(current))),
    }


def _published_measurement(profile, equilibrium) -> dict[str, object]:
    """Reproduce the published solve's arithmetic-floor operands."""
    relative = float(equilibrium.fixed_point.residual)
    scale = float(np.max(np.abs(np.asarray(equilibrium.flux))))
    amplitude = relative * scale
    resolution = reference.flux_resolution(profile, equilibrium)
    return {
        "prior_residual_amplitude_wb": 8.16e-10,
        "current_relative_residual": relative,
        "registered_relative_residual_ceiling": float(
            reference.RESIDUAL_TOLERANCE
        ),
        "current_residual_amplitude_wb": amplitude,
        "arithmetic_resolution_wb": float(resolution),
        "residual_over_resolution": float(amplitude / resolution),
        "registered_resolution_margin": float(reference.RESOLUTION_MARGIN),
        "finite_receipt": bool(equilibrium.finite.passed),
        "diverted": bool(equilibrium.topology.diverted),
        "plasma_current_a": float(equilibrium.moments.plasma_current),
    }


def _classifications(measurements: dict[str, object]) -> list[dict[str, object]]:
    """State each disposition from the registered semantic contract."""
    return [
        {
            "test": TEST_NAMES[0],
            "disposition": "regression",
            "owner": "map-gain chain",
            "mechanism": "noncontractive_free_boundary_map_resolvent_gain",
            "justification": (
                "The passive external field is within its cross-source budget, "
                "but the coupled free-boundary feedback amplifies it. The direct "
                "input is not the closure defect, so the registered closure "
                "ceiling remains valid and its repair belongs to the map-gain chain."
            ),
            "measurement": measurements[TEST_NAMES[0]],
        },
        {
            "test": TEST_NAMES[1],
            "disposition": "stale assertion",
            "proposed_expectation": (
                "Require exactly 60 finite samples from each fixed-budget route; "
                "retain the Picard non-contraction observation, but do not require "
                "Anderson's mixed iterates to amplify a residual from a converged seed."
            ),
            "justification": (
                "An expanding eigen-direction of the raw fixed-point map constrains "
                "Picard. Anderson applies history-dependent mixing and is not "
                "semantically required to follow or amplify that eigen-direction."
            ),
            "measurement": measurements[TEST_NAMES[1]],
        },
        {
            "test": TEST_NAMES[2],
            "disposition": "stale assertion",
            "proposed_expectation": (
                "Require the finite fixed-budget trace to select the zero-current, "
                "non-diverted vacuum branch, exhibit a nontrivial transient above its "
                "contracting tail, and contract by at least tenfold over that tail; "
                "do not prescribe growth over the first fifteen samples."
            ),
            "justification": (
                "The identified coupled map is strongly non-normal and its measured "
                "transient contains contractions and bursts. A monotone early rise "
                "does not follow from non-contraction, while the terminal branch and "
                "tail behaviour are the intended branch-selection semantics."
            ),
            "measurement": measurements[TEST_NAMES[2]],
        },
        {
            "test": TEST_NAMES[3],
            "disposition": "stale assertion",
            "proposed_expectation": (
                "Require the registered normalized fixed-point residual below 1e-6, "
                "a finite receipt, diverted topology and nonzero plasma current; bank "
                "residual-over-arithmetic-resolution as a diagnostic rather than an "
                "additional acceptance bound."
            ),
            "justification": (
                "The production solve contract is the normalized fixed-point bound "
                "plus physical receipt qualification. Floating-point spacing is a "
                "lower resolvability floor, not a promise that a fixed-budget Krylov "
                "solve terminates within ten units of that floor."
            ),
            "measurement": measurements[TEST_NAMES[3]],
        },
    ]


def run(receipt_path: Path, pytest_log: Path) -> dict[str, object]:
    """Execute the exact tests and write their semantic classification."""
    started = perf_counter()
    machine = _install_cached_carrier()
    recorder = _ResultRecorder()
    node_ids = [f"{TEST_MODULE}::{name}" for name in TEST_NAMES]
    pytest_arguments = [
        "-q",
        "-p",
        "no:cacheprovider",
        *node_ids,
    ]
    executed_command = "pytest " + " ".join(pytest_arguments)
    with pytest_log.open("w", encoding="utf-8") as stream:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = stream
            sys.stderr = stream
            pytest_exit = int(pytest.main(pytest_arguments, plugins=[recorder]))
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    solved = reference._solved(reference.SUITE_CELLS, True)
    profile, equilibrium = reference._published(reference.SUITE_CELLS)
    _, carrier = machine(reference.SUITE_CELLS, True)
    cache = carrier.cache_receipt
    if cache is None:
        raise RuntimeError("the reference carrier has no cache receipt")

    measurements = {
        TEST_NAMES[0]: _closure_measurement(solved),
        TEST_NAMES[1]: _relaxed_measurement(profile, equilibrium),
        TEST_NAMES[2]: _branch_measurement(solved),
        TEST_NAMES[3]: _published_measurement(profile, equilibrium),
    }
    expected_failures = all(
        recorder.results.get(name, {}).get("outcome") == "failed"
        for name in TEST_NAMES
    )
    receipt = {
        "schema": "nova.reference_failure_classification",
        "schema_version": 1,
        "source": {
            "head_commit": _git_output("rev-parse", "HEAD"),
            "tree_sha": _git_output("rev-parse", "HEAD^{tree}"),
        },
        "execution": {
            "command": executed_command,
            "pytest_exit_code": pytest_exit,
            "selected_test_count": len(TEST_NAMES),
            "observed_failure_count": sum(
                result["outcome"] == "failed" for result in recorder.results.values()
            ),
            "all_expected_failures_reproduced": expected_failures,
            "elapsed_seconds": perf_counter() - started,
            "pytest_log": str(pytest_log),
            "carrier_substitution": (
                "Exact test functions and assertions; suite-native cached_machine "
                "replaces only direct reconstruction of the immutable carrier."
            ),
            "carrier": {
                "store": cache.store,
                "key": cache.key,
                "cache_hit": bool(cache.hit),
                "arrays_verified": int(cache.arrays_verified),
                "bytes_verified": int(cache.bytes_verified),
                "bitwise_stored_precision": bool(cache.bitwise_stored_precision),
            },
            "per_test": recorder.results,
        },
        "classifications": _classifications(measurements),
        "verdict": {
            "complete": expected_failures and len(recorder.results) == len(TEST_NAMES),
            "regressions": 1,
            "stale_assertions": 3,
            "assertions_edited": False,
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    """Run the measurement and return success only for a complete receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--pytest-log", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = run(arguments.receipt, arguments.pytest_log)
    print(json.dumps(receipt["verdict"], sort_keys=True))
    return 0 if receipt["verdict"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
