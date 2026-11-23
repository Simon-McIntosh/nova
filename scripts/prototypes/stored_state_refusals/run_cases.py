"""Reproduce the forward-solve refusals beside stored imas-efit states.

The three machines' stored reconstructions are read from the imas-efit test
data by the comparison reader, and the public forward solve is started from
each stored flux map at imas-efit's fitted profile coefficients.  Three solves
carry the refusals and two carry the positive controls:

    mastu-stored-default          MAST-U, stored coefficients, default policy
    mastu-stored-continuation     the positive control, from the terminal state
    mastu-displaced-default       the pressure coordinate moved by the fit step
    mastu-displaced-continuation  that displacement, from the qualified state
    tcv-stored-default            TCV, stored coefficients, default policy
    iter-stored-default           ITER, stored coefficients, default policy

Every arm is persisted as it lands, so an expiry loses at most the running
arm.  One line-contour panel per arm is drawn on the stored imas-efit state
with both null sets and the wall.  Run it through
``scripts/prototypes/stored_state_refusals/submit.sh``; it needs a GPU lane.
"""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The job's own output directory. The run directory that holds every job is its
# parent, and that is what a written sequence is counted against: a job that
# counted only itself would write sequence 1 on every submission.
OUTPUT = Path(os.environ["STORED_STATE_REFUSALS_OUTPUT"]).resolve()
FIGURES = Path(os.environ["STORED_STATE_REFUSALS_FIGURES"]).resolve()
(OUTPUT / "comparison").mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
os.environ["NOVA_EFIT_COMPARISON_OUTPUT"] = str(OUTPUT / "comparison")
os.environ["NOVA_EFIT_COMPARISON_FIGURES"] = str(FIGURES)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nova.equilibrium.source import (  # noqa: E402
    DomainProfile,
    PolynomialFluxFunction,
)
from nova.equilibrium.solve_request import ExplicitSolveSeed  # noqa: E402
from records import (  # noqa: E402
    mesh_record,
    next_sequence,
    profile_coefficients,
    residual_fields,
)
from tests.test_magnetics_fit_efit_comparison import (  # noqa: E402
    _attempt,
    _problem,
    _read_case,
    _residual_fields,
    _write,
)

import panels  # noqa: E402

DIFFERENCE_STEP = 1.0e-4


def _displaced_source(source, index, step):
    core = source.core
    pressure = np.asarray(core.p_prime.coefficients, dtype=float).copy()
    field = np.asarray(core.ff_prime.coefficients, dtype=float).copy()
    raw = np.r_[pressure, field]
    raw[index] -= step
    split = pressure.size
    return replace(
        source,
        core=DomainProfile(
            p_prime=PolynomialFluxFunction(jnp.asarray(raw[:split])),
            ff_prime=PolynomialFluxFunction(jnp.asarray(raw[split:])),
        ),
    )


def _record(data, label, request, receipt, record, started):
    flux = data.get("terminal_flux")
    residual = data.get("terminal_residual")
    record.update(
        policy=request.policy.to_dict(),
        coefficients=profile_coefficients(request.source_profile).tolist(),
        seed_policy=type(request.seed_policy).__name__,
        current_present=bool(np.size(np.asarray(request.current))),
        qualified=receipt is not None,
        converged=bool(data.get("terminal_converged")),
        termination_reason=data.get("terminal_termination_reason"),
        device=str(jax.devices()[0]),
        wall_seconds=perf_counter() - started,
        **_residual_fields(residual),
    )
    if receipt is None:
        record.update(
            residual_fields(OUTPUT / "comparison" / f"{label}-rejected-history.pkl")
        )
    if receipt is not None:
        record["resolved_policy"] = receipt.resolved_defaults.policy.to_dict()
    if flux is not None:
        path = OUTPUT / f"{label}-terminal-flux.npz"
        np.savez(path, flux=np.asarray(flux))
        record["terminal_flux"] = str(path)
    try:
        destination = FIGURES / f"{data['name']}-{label}.svg"
        draw = panels.draw_panel(data, flux, destination, "")
        draw.update(
            label=label,
            residual=data.get("terminal_residual"),
            reason=data.get("terminal_termination_reason"),
            qualified=receipt is not None,
        )
        panels.write_description(destination, panels.caption(draw))
        record["panel"] = str(destination)
        record["panel_record"] = draw
    except Exception as error:
        record["panel_refused"] = f"{type(error).__name__}: {error}"
    _write(OUTPUT / f"{label}.json", record)
    return record


def _arm(data, request, label):
    started = perf_counter()
    try:
        receipt = _attempt(data, request, label)
        record = dict(label=label, case=data["name"])
        return _record(data, label, request, receipt, record, started)
    except Exception as error:
        record = dict(
            label=label,
            case=data.get("name"),
            failed_before_record=True,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        return record


def _report(measurement):
    # Counted at the moment of the write, over the run directory that holds
    # every job, so this receipt sorts above all its predecessors and a
    # re-measured arm supersedes the run it replaces rather than tying it.
    measurement["sequence"] = next_sequence(OUTPUT.parent)
    _write(OUTPUT / "measurement.json", measurement)


def _mastu(monkeypatch, records):
    data = _problem(_read_case("mastu"), monkeypatch)
    stored = data["request"]
    records["mesh"] = mesh_record(data)
    receipt = getattr(data["machine"], "cache_receipt", None)
    if receipt is not None:
        records["machine_cache"] = dict(
            store=receipt.store,
            key=receipt.key,
            hit=receipt.hit,
            build_seconds=receipt.build_seconds,
            load_seconds=receipt.load_seconds,
            store_seconds=receipt.store_seconds,
        )
    records["default_policy"] = stored.policy.to_dict()
    records.setdefault("stored_coefficients", {})["mastu"] = profile_coefficients(
        stored.source_profile
    ).tolist()
    _report(records)
    records["mastu-stored-default"] = _arm(data, stored, "mastu-stored-default")
    _report(records)
    terminal = data.get("terminal_flux")
    if terminal is None:
        records["mastu-stored-continuation"] = dict(
            label="mastu-stored-continuation",
            skipped="the first solve returned no terminal flux",
        )
        _report(records)
        return data, stored, None
    policy = replace(
        stored.policy,
        newton_steps=40,
        active_set_steps=64,
        kernel_tolerance=min(stored.policy.kernel_tolerance, 1.0e-12),
    )
    records["mastu-stored-continuation"] = _arm(
        data,
        replace(
            stored,
            policy=policy,
            seed_policy=ExplicitSolveSeed(np.asarray(terminal)),
        ),
        "mastu-stored-continuation",
    )
    _report(records)
    return data, stored, data.get("terminal_flux")


def _displaced(monkeypatch, records, data, stored, qualified_terminal):
    raw = profile_coefficients(stored.source_profile)
    pressure_size = int(
        np.asarray(stored.source_profile.core.p_prime.coefficients).size
    )
    # The comparison defines no displacement of its own, so the coordinate is
    # the request's live coefficient of largest magnitude: a zero coefficient
    # moves no current net and would re-ask the stored solve.
    index = int(np.argmax(np.abs(raw)))
    step = DIFFERENCE_STEP * max(abs(float(raw[index])), 1.0)
    records["displacement"] = dict(
        coordinate=("p_prime" if index < pressure_size else "ff_prime")
        + f"_{index if index < pressure_size else index - pressure_size}",
        index=index,
        coefficient_frame="pressure then ff_prime",
        pressure_size=pressure_size,
        description=(
            "the request's live coefficient of largest magnitude in the "
            "polynomial flux frame, displaced by 1e-4 times max(|c|, 1); the "
            "comparison's own fit defines no displacement, so the coordinate "
            "is chosen from the coefficients the fit stored"
        ),
        size=float(step),
        relative_size=float(step / abs(float(raw[index]))),
        sign="negative",
        value_before=float(raw[index]),
        value_after=float(raw[index] - step),
    )
    _report(records)
    displaced = replace(
        stored, source_profile=_displaced_source(stored.source_profile, index, step)
    )
    records["mastu-displaced-default"] = _arm(
        data, displaced, "mastu-displaced-default"
    )
    _report(records)
    if qualified_terminal is None:
        return
    records["mastu-displaced-continuation"] = _arm(
        data,
        replace(
            displaced,
            seed_policy=ExplicitSolveSeed(np.asarray(qualified_terminal)),
        ),
        "mastu-displaced-continuation",
    )
    _report(records)


def _mastu_displaced_only(monkeypatch, records):
    """Re-run the displaced arms alone, seeded from the landed stored solve."""
    data = _problem(_read_case("mastu"), monkeypatch)
    stored = data["request"]
    records["mesh"] = mesh_record(data)
    records.setdefault("stored_coefficients", {})["mastu"] = profile_coefficients(
        stored.source_profile
    ).tolist()
    seed_path = os.environ.get("STORED_STATE_REFUSALS_SEED_FLUX")
    qualified_terminal = None
    if seed_path and Path(seed_path).is_file():
        qualified_terminal = np.load(seed_path)["flux"]
        records["seed_flux"] = str(Path(seed_path).resolve())
    _report(records)
    _displaced(monkeypatch, records, data, stored, qualified_terminal)
    del data


def _stored_case(name, monkeypatch, records):
    data = _problem(_read_case(name), monkeypatch)
    request = data["request"]
    records["meshes"] = records.get("meshes", {})
    records["meshes"][name] = mesh_record(data)
    records["stored_coefficients"] = records.get("stored_coefficients", {})
    records["stored_coefficients"][name] = profile_coefficients(
        request.source_profile
    ).tolist()
    _report(records)
    records[f"{name}-stored-default"] = _arm(data, request, f"{name}-stored-default")
    _report(records)
    return data, request


def main():
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu", "the standing cases need a GPU lane"
    monkeypatch = pytest.MonkeyPatch()
    records = dict(
        qualification_tolerance=1.0e-10,
        device=str(jax.devices()[0]),
        source_revision=os.environ.get("SOURCE_REVISION"),
        measurement_module="/".join(
            [str(ROOT / "tests/test_magnetics_fit_efit_comparison.py")]
        ),
        working_directory=str(Path.cwd().resolve()),
    )
    _report(records)
    selected = (
        {name for name in sys.argv[1].split(",") if name}
        if len(sys.argv) > 1
        else set()
    )
    if "mastu-displaced" in selected:
        _mastu_displaced_only(monkeypatch, records)
    if not selected or "mastu" in selected:
        data, stored, qualified_terminal = _mastu(monkeypatch, records)
        _displaced(monkeypatch, records, data, stored, qualified_terminal)
        del data
    for name in ("tcv", "iter"):
        if selected and name not in selected:
            continue
        case_data, _ = _stored_case(name, monkeypatch, records)
        del case_data
    records["finished"] = True
    _report(records)


if __name__ == "__main__":
    main()
