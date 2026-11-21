"""Reproduce the TCV and ITER stored-state refusals from the recorded fixtures.

The earlier comparison met these refusals inside its own drive: the case
reader, the case's fixture machine, the receipt mesh it is differentiated on,
the stored flux map as the seed, the stored fitted conductor currents and the
tightening policy are all that record's, and this runner calls that record's
own functions rather than restating any of them.

Two arms per case, both started from the stored imas-efit flux map:

    <case>-stored-default    the record's default policy, the arm the record
                             shows refusing at the stored state
    <case>-stored-tightened  the comparison's continuation policy (internal
                             stopping tolerance 1e-12, 40 Newton steps, 64
                             active-set steps) seeded from the first arm's
                             terminal state, which is the construction the
                             record used for the arm it shows qualifying on
                             MAST-U

Each arm is written as its own receipt as it lands, so an expiry loses at most
the running arm.  Each arm's terminal state is drawn beside the stored
imas-efit state on one shared level array, with its null set and the wall.
Run it through ``scripts/prototypes/stored_state_refusals/submit.sh`` with
``--runner run_recorded_cases.py``; it needs a GPU lane.  ``--render <receipts>
<figures>`` redraws the panels from the terminal fluxes a run already stored,
so a caption or null-set change costs no solve.
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

OUTPUT = (
    Path(os.environ["STORED_STATE_REFUSALS_OUTPUT"]).resolve()
    if "STORED_STATE_REFUSALS_OUTPUT" in os.environ
    else None
)
FIGURES = (
    Path(os.environ["STORED_STATE_REFUSALS_FIGURES"]).resolve()
    if "STORED_STATE_REFUSALS_FIGURES" in os.environ
    else None
)
if OUTPUT is not None and FIGURES is not None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    os.environ["NOVA_EFIT_COMPARISON_OUTPUT"] = str(OUTPUT / "comparison")
    os.environ["NOVA_EFIT_COMPARISON_FIGURES"] = str(OUTPUT / "comparison")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nova.equilibrium.solve_request import ExplicitSolveSeed  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402
from tests.test_magnetics_fit_efit_comparison import (  # noqa: E402
    _attempt,
    _fitted_current_binding,
    _problem,
    _read_case,
    _residual_fields,
    _write,
)

import panels  # noqa: E402
from records import mesh_record, profile_coefficients  # noqa: E402

CASES = ("tcv", "iter")
RECORDED = {
    "tcv": dict(
        arm="stored-field seed, default policy",
        qualified=False,
        residual=0.030364200526436996,
        termination_reason=12,
        termination_name="ACTIVE_SET_STEP_REFUSED",
        axis_admitted=True,
        wall_seconds=1611.8093862347305,
    ),
    "iter": dict(
        arm="stored-field seed, default policy",
        qualified=False,
        residual=None,
        residual_state="nan",
        termination_reason=3,
        termination_name="NONFINITE_RESIDUAL",
        axis_admitted=False,
        failure="NoQualifiedAxisError",
        wall_seconds=None,
    ),
}
RECORD_SOURCE = (
    "docs/evidence/fragments/inverse-equilibrium-baseline/efit-comparison.html"
)
RECORD_REVISION = "9047a3fe50708d6dd20ebe672d0dc22f40b0ae50"


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _solved_nulls_attribute(data, record):
    """Attach the arm's recorded refusal, or clear it when an axis was admitted."""
    axis = record.get("axis") or {}
    if axis.get("admitted"):
        data.pop("solved_nulls", None)
        return
    # A refused solved-null read is passed to the panel so it says which error
    # refused instead of reading a topology the solve never reached.
    data["solved_nulls"] = (
        f"refused: {axis.get('error')}: {axis.get('message')}"
        if axis.get("error")
        else "not read"
    )


def _axis_admission(data, flux):
    """Ask the topology reader whether a terminal state admits an axis."""
    if flux is None:
        return dict(admitted=None, error="no terminal flux was returned")
    try:
        _, topology = data["profile"].operator.read(jax.numpy.asarray(flux))
    except (ValueError, RuntimeError) as error:
        return dict(admitted=False, error=type(error).__name__, message=str(error))
    return dict(
        admitted=True,
        magnetic_axis=[float(x) for x in np.asarray(topology.axis)],
        diverted=bool(topology.diverted),
    )


def _device_memory(device):
    """Return the device's total memory, where the backend reports one."""
    stats = getattr(device, "memory_stats", None)
    if stats is None:
        return None
    return int(stats().get("bytes_limit", 0)) or None


def _band(record):
    """State the measured value beside the recorded one and their factor."""
    recorded, band = record["recorded"], {}
    residual, target = record.get("residual"), recorded.get("residual")
    band.update(
        recorded_residual=target,
        measured_residual=residual,
        recorded_termination_reason=recorded["termination_reason"],
        measured_termination_reason=record.get("termination_reason"),
    )
    if residual is not None and target:
        band["factor"] = abs(residual / target)
        band["within_factor_three"] = bool(1.0 / 3.0 <= band["factor"] <= 3.0)
    if recorded.get("residual_state") == "nan":
        band["recorded_residual_state"] = "nan"
        band["measured_residual_state"] = record.get("residual_state")
    if "failure" in recorded:
        axis = record.get("axis") or {}
        band["recorded_axis_admitted"] = recorded["axis_admitted"]
        band["measured_axis_admitted"] = axis.get("admitted")
        band["measured_axis_error"] = axis.get("error")
    return band


def _draw(data, case, label, record, figures=None):
    """Draw the arm's terminal state and record what the file actually carries."""
    if data.get("terminal_flux") is None:
        record["panel_refused"] = "the arm returned no terminal flux to contour"
        return
    destination = (figures or FIGURES) / f"{case}-{label[len(case) + 1 :]}.svg"
    try:
        drawn = panels.draw_panel(data, data["terminal_flux"], destination, "")
        drawn.update(
            label=label,
            residual=record.get("residual"),
            reason=record.get("termination_reason"),
        )
        panels.write_description(destination, panels.caption(drawn))
    except Exception as error:
        record["panel_refused"] = f"{error.__class__.__name__}: {error}"
        return
    record["panel"] = str(destination)
    record["panel_record"] = drawn


def _arm(data, request, label, case):
    started = perf_counter()
    record = dict(label=label, case=case, recorded=RECORDED[case])
    try:
        receipt = _attempt(data, request, label)
    except Exception as error:  # scaffolding failure, not a solve outcome
        record.update(
            failed_before_record=True,
            error=f"{error.__class__.__name__}: {error}",
            traceback=traceback.format_exc(),
            wall_seconds=perf_counter() - started,
        )
        _write(OUTPUT / f"{record['label']}.json", record)
        return record
    flux = data.get("terminal_flux")
    record.update(
        policy=request.policy.to_dict(),
        resolved_policy=(
            receipt.resolved_defaults.policy.to_dict() if receipt is not None else None
        ),
        route=request.policy.route,
        coefficients=profile_coefficients(request.source_profile).tolist(),
        seed_policy=type(request.seed_policy).__name__,
        current_present=bool(np.size(np.asarray(request.current))),
        mesh=mesh_record(data),
        device=str(jax.devices()[0]),
        qualified=bool(receipt is not None),
        converged=bool(data.get("terminal_converged")),
        device_memory_bytes=_device_memory(jax.devices()[0]),
        termination_reason=data.get("terminal_termination_reason"),
        wall_seconds=perf_counter() - started,
        **_residual_fields(_finite(data.get("terminal_residual"))),
        axis=_axis_admission(data, flux),
    )
    _solved_nulls_attribute(data, record)
    if flux is not None:
        path = OUTPUT / f"{label}-terminal-flux.npz"
        np.savez(path, flux=np.asarray(flux))
        record["terminal_flux"] = str(path)
    _draw(data, case, label, record)
    record["recorded_band"] = _band(record)
    _write(OUTPUT / f"{record['label']}.json", record)
    return record


def render(receipts: Path, figures: Path) -> None:
    """Redraw the arms' panels from the terminal fluxes their receipts name.

    The receipts keep each arm's measured outcome, so a caption or a null-set
    change is repaired by redrawing from the stored flux rather than by solving
    the case again.  Each redrawn panel's record is written back into the arm's
    receipt, so the captions and the fragment read what the files carry.
    """
    receipts, figures = receipts.resolve(), figures.resolve()
    figures.mkdir(parents=True, exist_ok=True)
    scratch = receipts / "render-scratch"
    scratch.mkdir(exist_ok=True)
    os.environ.setdefault("NOVA_EFIT_COMPARISON_OUTPUT", str(scratch))
    os.environ.setdefault("NOVA_EFIT_COMPARISON_FIGURES", str(scratch))
    configure_dtypes()
    monkeypatch = pytest.MonkeyPatch()
    data_by_case: dict = {}
    for case in CASES:
        for arm in ("default", "tightened"):
            label = f"{case}-stored-{arm}"
            path = receipts / f"{label}.json"
            if not path.is_file():
                continue
            record = json.loads(path.read_text())
            flux_path = record.get("terminal_flux")
            if not flux_path:
                continue
            if case not in data_by_case:
                data_by_case[case] = _problem(_read_case(case), monkeypatch)
            data = data_by_case[case]
            _solved_nulls_attribute(data, record)
            data["terminal_flux"] = np.load(flux_path)["flux"]
            _draw(data, case, label, record, figures)
            _write(path, record)
            print(
                "PANEL " + label + " " + json.dumps(record.get("panel_record", {})),
                flush=True,
            )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--render"]:
        render(Path(argv[1]), Path(argv[2]))
        return
    if OUTPUT is None or FIGURES is None:
        raise SystemExit(
            "STORED_STATE_REFUSALS_OUTPUT and STORED_STATE_REFUSALS_FIGURES must "
            "name this job's output directories"
        )
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu", "the recorded refusals need a GPU lane"
    monkeypatch = pytest.MonkeyPatch()
    summary = dict(
        qualification_tolerance=1.0e-10,
        device=str(jax.devices()[0]),
        source_revision=os.environ.get("SOURCE_REVISION"),
        record_source=RECORD_SOURCE,
        record_revision=RECORD_REVISION,
        measurement_module=str(ROOT / "tests/test_magnetics_fit_efit_comparison.py"),
        working_directory=str(Path.cwd().resolve()),
        arms={},
    )
    _write(OUTPUT / "measurement.json", summary)
    for case in CASES:
        data = _problem(_read_case(case), monkeypatch)
        data = _fitted_current_binding(data)
        fitted = np.asarray(data["fitted_conductor_current"])
        data["request"] = replace(data["request"], current=fitted)
        data["profile"] = replace(
            data["profile"],
            operator=replace(data["profile"].operator, external_current=fitted),
        )
        summary["arms"].setdefault(case, {})
        summary["arms"][case]["fitted_conductor_current"] = fitted.tolist()
        summary["arms"][case]["default"] = _arm(
            data, data["request"], f"{case}-stored-default", case
        )
        _write(OUTPUT / "measurement.json", summary)
        terminal = data.get("terminal_flux")
        if terminal is None:
            summary["arms"][case]["tightened"] = dict(
                label=f"{case}-stored-tightened",
                skipped="the default arm returned no terminal flux to continue from",
            )
            _write(OUTPUT / "measurement.json", summary)
            continue
        policy = replace(
            data["request"].policy,
            newton_steps=40,
            active_set_steps=64,
            kernel_tolerance=min(data["request"].policy.kernel_tolerance, 1.0e-12),
        )
        summary["arms"][case]["tightened"] = _arm(
            data,
            replace(
                data["request"],
                policy=policy,
                seed_policy=ExplicitSolveSeed(np.asarray(terminal)),
            ),
            f"{case}-stored-tightened",
            case,
        )
        _write(OUTPUT / "measurement.json", summary)
        del data
    summary["finished"] = True
    _write(OUTPUT / "measurement.json", summary)
    print(
        "RECORDED_CASES_FINISHED " + json.dumps(summary["arms"], default=str)[:6000],
        flush=True,
    )


if __name__ == "__main__":
    main()
