"""Attribute the stored-state refusals' drift to one differing harness input.

Each case runs two arms in one job at today's fixtures and the solver's default
policy, started from the stored imas-efit flux map:

    <case>-input-stored      the stored fitted currents this branch reads
    <case>-input-comparison  the same solve with the one differing input set to
                             what the earlier comparison run used: the currents
                             that run's fitted map carries for TCV, and the
                             measured conductor currents (no fitted binding)
                             for ITER, which is how that run built its request

Run it through ``submit.sh`` here with ``--runner attribute_input_drift.py``;
it needs a GPU lane.
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT = None
if "STORED_STATE_REFUSALS_OUTPUT" in os.environ:
    OUTPUT = Path(os.environ["STORED_STATE_REFUSALS_OUTPUT"]).resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ["NOVA_EFIT_COMPARISON_OUTPUT"] = str(OUTPUT)

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402
from tests.test_magnetics_fit_efit_comparison import (  # noqa: E402
    _attempt,
    _fitted_current_binding,
    _problem,
    _read_case,
    _residual_fields,
    _write,
)
from records import mesh_record, profile_coefficients  # noqa: E402

CASES = ("tcv", "iter")

COMPARISON_CURRENT_MAP = {
    "tcv": Path(
        "/home/ITER/mcintos/.cache/nova-inverse-equilibrium-baseline/"
        "efit-comparison-run-copy/fitted-physical-measurement-1318907/"
        "physical-tcv/tcv-fitted-current-map.json"
    ),
}

RECORDED = {
    "tcv": dict(
        residual=0.030364200526436996, termination_reason=12, axis_admitted=True
    ),
    "iter": dict(
        residual=None, residual_state="nan", termination_reason=3, axis_admitted=False
    ),
}

RECORD_SOURCE = (
    "docs/evidence/fragments/inverse-equilibrium-baseline/efit-comparison.html"
)


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _bind_columns(data, columns):
    element_current = data["projection"] @ columns
    case = data["case"]
    items = [
        replace(item, current=float(value))
        for item, value in zip(
            (*case.active, *case.passive), element_current, strict=True
        )
    ]
    count = len(case.active)
    data["case"] = replace(
        case,
        active=tuple(items[:count]),
        passive=tuple(items[count:]),
    )
    data["fitted_conductor_current"] = columns
    return data


def _comparison_columns(data, case):
    path = COMPARISON_CURRENT_MAP.get(case)
    if path is None:
        return None, "no fitted-current binding in the comparison arm"
    effective = json.loads(path.read_text())["effective_current_a"]
    names = [str(name) for name in data["conductor_names"]]
    missing = [name for name in names if name not in effective]
    if missing:
        raise ValueError(f"{case}: comparison map is missing {len(missing)} conductors")
    return np.array([float(effective[name]) for name in names]), str(path)


def _current_drift(columns, baseline):
    difference = np.abs(np.asarray(columns) - np.asarray(baseline))
    relative = difference / np.maximum(np.abs(np.asarray(baseline)), 1.0)
    return dict(
        conductors=int(np.size(columns)),
        identical_within_1e_6_a=int(np.count_nonzero(difference < 1e-6)),
        max_absolute_difference_a=float(np.max(difference)),
        max_relative_difference=float(np.max(relative)),
    )


def _axis_admission(data, flux):
    if flux is None:
        return dict(admitted=None, note="no terminal flux was returned")
    try:
        _, topology = data["profile"].operator.read(jax.numpy.asarray(flux))
    except (ValueError, RuntimeError) as error:
        return dict(admitted=False, error=type(error).__name__, message=str(error))
    return dict(
        admitted=True,
        magnetic_axis=[float(x) for x in np.asarray(topology.axis)],
        diverted=bool(topology.diverted),
    )


def _history(receipt, error=None):
    if receipt is not None:
        history = np.asarray(receipt.residual_history).ravel()
    elif error is not None:
        found = [
            frame.f_locals["history"]
            for frame, _ in traceback.walk_tb(error.__traceback__)
            if "history" in frame.f_locals
            and hasattr(frame.f_locals["history"], "residual")
        ]
        if not found:
            return None
        history = np.asarray(found[-1].residual).ravel()
    else:
        return None
    finite = history[np.isfinite(history)]
    return dict(
        entries=int(history.size),
        first=float(finite[0]) if finite.size else None,
        last=float(finite[-1]) if finite.size else None,
        lowest=float(np.min(finite)) if finite.size else None,
        values=[float(x) if np.isfinite(x) else None for x in history],
    )


def _failure_trajectory(label):
    """Read the trajectory an arm's generating failure stored beside its receipt.

    A refused arm never returns a receipt, so its residual trace survives only in
    the trajectory the solver dumped when it raised; that trace is what says
    whether the arm started where the record's arm started.
    """
    path = OUTPUT / f"{label}-generating-failure.json"
    if not path.is_file():
        return None
    failure = json.loads(path.read_text())
    history_path = failure.get("history_path")
    if not history_path:
        return None
    import pickle

    with open(history_path, "rb") as handle:
        result = pickle.load(handle)

    def array(name):
        value = getattr(result, name, None)
        return None if value is None else np.asarray(value).ravel().tolist()

    return dict(
        exception=failure.get("exception"),
        trips=failure.get("trips"),
        termination_reason=failure.get("termination_reason"),
        active_set_residuals=array("active_set_residuals"),
        inner_iteration_residuals_before=array("inner_iteration_residuals_before"),
        inner_iteration_residuals_after=array("inner_iteration_residuals_after"),
    )


def _arm(data, request, label, case, applied):
    started = perf_counter()
    record = dict(
        label=label, case=case, recorded=RECORDED[case], applied_current=applied
    )
    error = None
    receipt = None
    try:
        receipt = _attempt(data, request, label)
    except Exception as raised:
        error = raised
        print("ARM_EXC", label, type(raised).__name__, flush=True)
    flux = data.get("terminal_flux")
    history = _history(receipt, error)
    trajectory = None if history is not None else _failure_trajectory(label)
    record.update(
        policy=request.policy.to_dict(),
        route=request.policy.route,
        coefficients=profile_coefficients(request.source_profile).tolist(),
        seed_policy=type(request.seed_policy).__name__,
        mesh=mesh_record(data),
        device=str(jax.devices()[0]),
        qualified=bool(receipt is not None),
        converged=bool(data.get("terminal_converged")),
        termination_reason=data.get("terminal_termination_reason"),
        wall_seconds=perf_counter() - started,
        **_residual_fields(_finite(data.get("terminal_residual"))),
        axis=_axis_admission(data, flux),
        residual_history=history,
        failure_trajectory=trajectory,
    )
    recorded = RECORDED[case]
    if recorded.get("residual") and record.get("residual") is not None:
        factor = abs(record["residual"] / recorded["residual"])
        record["factor_of_recorded_residual"] = factor
        record["within_factor_three"] = 1.0 / 3.0 <= factor <= 3.0
    record["same_reason_code"] = (
        record.get("termination_reason") == recorded["termination_reason"]
    )
    record["same_axis_admission"] = (
        record["axis"].get("admitted") == recorded["axis_admitted"]
    )
    if error is not None:
        record["raised"] = f"{type(error).__name__}: {error}"
    _write(OUTPUT / f"{label}.json", record)
    return record


def _prepare(case, monkeypatch, fitted):
    data = _problem(_read_case(case), monkeypatch)
    if fitted:
        data = _fitted_current_binding(data)
    return data


def _solve(data, columns, label, case, applied):
    """Solve one arm with the given conductor currents applied to operator and request.

    ``columns`` of ``None`` means "the currents this branch already bound", which
    is the stored arm; an arm that applies no fitted currents at all reaches here
    with no binding in ``data`` and leaves the request as the case reader built
    it. Both the operator's external current and the request's current are set,
    because the operator carries the field and the request carries the receipt.
    """
    request, profile = data["request"], data["profile"]
    if columns is None:
        columns = data.get("fitted_conductor_current")
    if columns is not None:
        columns = np.asarray(columns, dtype=float)
        _bind_columns(data, columns)
        current = jax.numpy.asarray(columns)
        request = replace(request, current=current)
        profile = replace(
            profile, operator=replace(profile.operator, external_current=current)
        )
        data["profile"] = profile
    return _arm(data, request, label, case, applied)


def main():
    if OUTPUT is None:
        raise SystemExit(
            "STORED_STATE_REFUSALS_OUTPUT must name this job's output directory"
        )
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu", "a GPU lane is required"
    monkeypatch = pytest.MonkeyPatch()
    summary = dict(
        qualification_tolerance=1.0e-10,
        device=str(jax.devices()[0]),
        source_revision=os.environ.get("SOURCE_REVISION"),
        record_source=RECORD_SOURCE,
        measurement_module=str(ROOT / "tests/test_magnetics_fit_efit_comparison.py"),
        working_directory=str(Path.cwd().resolve()),
        cases={},
    )
    _write(OUTPUT / "measurement.json", summary)
    for case in CASES:
        entry = dict()
        data = _prepare(case, monkeypatch, fitted=True)
        stored = np.asarray(data["fitted_conductor_current"], dtype=float)
        comparison, provenance = _comparison_columns(data, case)
        entry["comparison_input"] = provenance
        if comparison is not None:
            entry["stored_vs_comparison"] = _current_drift(stored, comparison)
        entry["stored"] = _solve(
            data, None, f"{case}-input-stored", case, "stored fitted currents"
        )
        _write(OUTPUT / "measurement.json", summary)
        del data

        data = _prepare(case, monkeypatch, fitted=comparison is not None)
        entry["comparison"] = _solve(
            data,
            comparison,
            f"{case}-input-comparison",
            case,
            provenance,
        )
        del data
        summary["cases"][case] = entry
        _write(OUTPUT / "measurement.json", summary)
        print("CASE_DONE " + case, flush=True)
    summary["finished"] = True
    _write(OUTPUT / "measurement.json", summary)
    print("INPUT_ATTRIBUTION_FINISHED", flush=True)


if __name__ == "__main__":
    main()
