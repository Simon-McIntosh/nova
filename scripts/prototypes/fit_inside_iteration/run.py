"""Measure the least-squares fit inside the equilibrium iteration.

Runs the known states of the known-answer recovery record and the stored cases
of the imas-efit comparison record through
:class:`nova.equilibrium.flux_iteration.FluxIterationFit`, in sequence inside
one allocation, and writes a per-case table plus one overlaid line-contour
panel per case. Each case carries its negative control: the row the fit reads
(``measured - vacuum``), the measured plasma current and the measured conductor
currents all scaled by 1.10 must leave the 1.5 % plasma-current band when the
case converged. Both terms of the row are scaled, because scaling the
measurement alone leaves the row short of the factor wherever the vacuum
prediction dominates, and the control then tests something else.

The known states reuse the row construction, generating receipts and starting
fields of the known-answer recovery study; the stored cases reuse the reader,
rows, profile basis and fitted-conductor projection of the imas-efit comparison
study. No second construction of either is introduced here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

KNOWN = ("solovev", "mast", "iter")
STORED = {"mastu": "mastu", "tcv": "tcv", "iter-stored": "iter"}
NESTED_FORWARD_CALLS = {"solovev": 20, "mast": 15, "iter": 15}
STEP_RULE = "coupled"
STEP_BUDGET = 96


class _ControlNotJudged(Exception):
    """The fit did not converge, so the control's band is not applied."""


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


class _Factory:
    """Stand in for pytest's ``tmp_path_factory`` outside a session."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def mktemp(self, name: str) -> Path:
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return path


class _CellRows:
    """Read a per-cell plasma current vector into the measured row order."""

    def __init__(self, biot, sensors, reader) -> None:
        self.biot = biot
        self.sensors = sensors
        self.reader = reader

    def __call__(self, currents):
        self.biot.sloc["Ic"] = np.asarray(currents, dtype=np.float64)
        return np.asarray(self.reader(self.biot, self.sensors), dtype=np.float64)


def _known_case(name, factory):
    from tests.magnetics_fit_harness import (
        _physical_sensor_rows,
        physical_case_table,
    )

    case = physical_case_table(factory)(name)
    initial = case["initial"]
    seed_policy = initial.seed_policy
    if not hasattr(seed_policy, "state"):
        raise TypeError(
            "the known state's starting field must be an explicit solve seed"
        )
    flux = np.asarray(seed_policy.state, dtype=np.float64)
    profile = case["profile"]._with_source(initial.source_profile)
    rows = _CellRows(case["source"].biot, case["sensors"], _physical_sensor_rows)
    return dict(
        name=name,
        profile=profile,
        magnetics=case["sensors"],
        rows=rows,
        basis=case["basis"],
        measurement=case["measurement"],
        initial_flux=flux,
        generating=case["generated"],
        generated_flux=np.asarray(case["generated"].terminal_state.flux),
        generated_topology=case["generated"].terminal_state.topology,
        cells=len(case["coordinate"]),
        core=np.asarray(case["generated"].terminal_state.domains.core),
        plasma_current=abs(float(case["measurement"].plasma_current)),
        prescribed_current=getattr(case["request"], "prescribed_current", None),
        request=case["request"],
        figure=case,
    )


def _stored_case(name):
    import pytest
    from tests import test_magnetics_fit_efit_comparison as study
    from tests.magnetics_fit_harness import (
        _PlasmaDiagnosticSource,
        _physical_sensor_rows,
    )
    from nova.frame.coilset import CoilSet

    data = study._read_case(name)
    data = study._problem(data, pytest.MonkeyPatch())
    source = _PlasmaDiagnosticSource(data["machine"].cell_polygons)
    conductor = CoilSet(dcoil=-1, field_attrs=["Psi", "Br", "Bz"])
    for item in (*data["case"].active, *data["case"].passive):
        conductor.coil.insert(*item.placement, nturn=item.turns, Ic=0.0, name=item.name)
    for factory in ("point", "probe", "poloidal_flux_loop"):
        _PlasmaDiagnosticSource._cache_factory(getattr(conductor, factory))
    columns = []
    for column in data["projection"].T:
        conductor.sloc["Ic"] = column
        columns.append(_physical_sensor_rows(conductor, data["geometry"]))
    response = np.column_stack(columns)
    rows = _CellRows(source.biot, data["geometry"], _physical_sensor_rows)
    return dict(
        name=name,
        profile=data["profile"],
        magnetics=data["geometry"],
        rows=rows,
        basis=data["basis"],
        measurement=data["measurement"],
        initial_flux=np.asarray(data["seed"], dtype=np.float64),
        conductor_response=response,
        conductor_names=data["conductor_names"],
        prescribed_current=getattr(data.get("request"), "prescribed_current", None),
        plasma_current=abs(float(data["case"].plasma_current)),
        reference_plasma_current=float(data["case"].plasma_current),
        data=data,
    )


def _model_rows(case, fitted):
    """Return the model rows at the terminal state, or why they cannot be read."""
    try:
        return np.asarray(
            case["rows"](fitted.terminal_state.cell_current), dtype=np.float64
        ).tolist()
    except ValueError as refusal:  # a terminal state without cells still has a row
        return f"{type(refusal).__name__}: {refusal}"


GAIN_ERROR = 1.10


def _gain_error(measurement, factor=GAIN_ERROR):
    """Return the measurement with every physical input scaled by ``factor``.

    The row the fit reads is ``measured - vacuum``, so a gain error on the rows
    scales both terms. Scaling the measurement alone leaves the signature short
    of the factor wherever the vacuum prediction is the larger term, which is a
    different test from the one the control claims to be.
    """

    def scaled(value):
        return None if value is None else np.asarray(value, dtype=np.float64) * factor

    return replace(
        measurement,
        measured=scaled(measurement.measured),
        vacuum=scaled(measurement.vacuum),
        scale=scaled(measurement.scale),
        plasma_current=float(measurement.plasma_current) * factor,
        vacuum_flux=scaled(measurement.vacuum_flux),
        conductor_current=scaled(measurement.conductor_current),
        conductor_scale=scaled(measurement.conductor_scale),
    )


def _fit_summary(fitted):
    """Summarise a fit so a control run is read by what it reached, not inferred."""
    return dict(
        converged=bool(fitted.converged),
        passes=len(fitted.passes),
        last_flux_update=fitted.last_flux_update,
        coefficients=fitted.coefficients.tolist(),
        stopped_improving_pass=fitted.stopped_improving_pass,
        stalled_pass=fitted.stalled_pass,
        rejected_pass=fitted.rejected_pass,
        wall_seconds=fitted.wall_seconds,
        plasma_current_a=float(np.sum(fitted.terminal_state.cell_current)),
    )


def _progress(name):
    """Print each pass as it lands, since a solve-heavy pass can run for minutes."""

    started = perf_counter()

    def report(index, misfit, update):
        print(
            "PASS "
            + json.dumps(
                dict(
                    case=name,
                    index=index,
                    misfit=misfit,
                    update=update,
                    wall=perf_counter() - started,
                )
            ),
            flush=True,
        )

    return report


def _run_known(case, args):
    from nova.equilibrium.flux_iteration import FluxIterationFit
    from tests.magnetics_fit_harness import _draw_recovery_panels

    fitter = FluxIterationFit(
        case["profile"],
        case["magnetics"],
        case["rows"],
        coefficient_basis=case["basis"],
        prescribed_current=case["prescribed_current"],
    )
    fitted = fitter.fit(
        case["measurement"],
        case["initial_flux"],
        plasma_current_scale=case["plasma_current"],
        on_pass=_progress(case["name"]),
        max_passes=args.max_passes,
        flux_tolerance=args.flux_tolerance,
        relaxation=args.relaxation,
        step_rule=STEP_RULE,
    )
    flux = np.asarray(fitted.terminal_state.flux)
    span = abs(float(case["generated_topology"].flux_span))
    error = float(
        np.max(
            np.abs(flux[: case["cells"]] - case["generated_flux"][: case["cells"]])[
                case["core"]
            ]
        )
        / span
    )
    residual = float(np.max(np.abs(fitted.whitened_residuals)))
    scaled_measurement = _gain_error(case["measurement"])
    scaled = fitter.fit(
        scaled_measurement,
        case["initial_flux"],
        plasma_current_scale=case["plasma_current"] * GAIN_ERROR,
        max_passes=args.max_passes,
        flux_tolerance=args.flux_tolerance,
        relaxation=args.relaxation,
        step_rule=STEP_RULE,
    )
    scaled_current = abs(float(np.sum(scaled.terminal_state.cell_current)))
    control = abs(scaled_current - case["plasma_current"]) / case["plasma_current"]
    # Record the ratio the rows the fit reads actually moved by, so a control
    # that stays inside its band cannot be blamed on an unscaled input.
    plain_signature = case["measurement"].signature[case["measurement"].mask]
    scaled_signature = scaled_measurement.signature[case["measurement"].mask]
    resolved = np.abs(plain_signature) > 0.0
    ratio = np.abs(scaled_signature[resolved]) / np.abs(plain_signature[resolved])
    signature_ratio = float(np.median(ratio))
    shim = SimpleNamespace(
        forward=SimpleNamespace(terminal_state=fitted.terminal_state)
    )
    panel_error = None
    try:
        _draw_recovery_panels(case["name"], case["figure"], shim)
    except Exception as refusal:  # the row outlives a painter's refusal
        panel_error = f"{type(refusal).__name__}: {refusal}"
    row = dict(
        case=case["name"],
        construction="known state of the known-answer recovery record",
        converged=bool(fitted.converged),
        passes=len(fitted.passes),
        pass_flux_update=[item.flux_update for item in fitted.passes],
        pass_misfit=[item.misfit for item in fitted.passes],
        pass_fraction=[item.fraction for item in fitted.passes],
        pass_trials=[item.trials for item in fitted.passes],
        stopped_improving_pass=fitted.stopped_improving_pass,
        stalled_pass=fitted.stalled_pass,
        stalled_message=fitted.stalled_message,
        axis_lost_pass=fitted.axis_lost_pass,
        axis_loss_message=fitted.axis_loss_message,
        rejected_pass=fitted.rejected_pass,
        rejected_message=fitted.rejected_message,
        topology_error=fitted.topology_error,
        panel_error=panel_error,
        terminal_flux_finite=bool(np.all(np.isfinite(fitted.terminal_state.flux))),
        flux_map_error_over_span=error,
        flux_map_band=1.0e-6,
        whitened_residual_sup=residual,
        whitened_residual_band=1.0e-8,
        coefficients=fitted.coefficients.tolist(),
        coefficient_names=list(fitted.coefficient_names),
        singular_values=fitted.singular_values.tolist(),
        undetermined_coefficients=list(fitted.undetermined_coefficients),
        terminal_misfit=fitted.terminal_misfit,
        last_flux_update=fitted.last_flux_update,
        flux_evaluations=fitted.flux_evaluations,
        sensor_evaluations=fitted.sensor_evaluations,
        wall_seconds=fitted.wall_seconds,
        max_passes=args.max_passes,
        flux_tolerance=args.flux_tolerance,
        step_rule=STEP_RULE,
        nested_forward_calls=NESTED_FORWARD_CALLS[case["name"]],
        reconstructed_plasma_current_a=float(
            np.sum(fitted.terminal_state.cell_current)
        ),
        generating_plasma_current_a=float(case["measurement"].plasma_current),
        scaled_control_judged=bool(fitted.converged),
        scaled_control_relative_error=control,
        scaled_control_band=0.015,
        scaled_control_fired=bool(fitted.converged and control > 0.015),
        scaled_measurement_current_a=float(scaled_measurement.plasma_current),
        scaled_signature_ratio=signature_ratio,
        scaled_fit=_fit_summary(scaled),
        relaxed=fitted.relaxation,
        max_coefficient_step=fitted.max_coefficient_step,
        whitened_residuals=np.asarray(fitted.whitened_residuals).tolist(),
        measured=np.asarray(case["measurement"].measured).tolist(),
        vacuum=np.asarray(case["measurement"].vacuum).tolist(),
        scale=np.asarray(case["measurement"].scale).tolist(),
        mask=np.asarray(case["measurement"].mask).tolist(),
        model_rows=_model_rows(case, fitted),
        first_pass_values=np.asarray(fitted.first_pass_values).tolist(),
        first_pass_residual=np.asarray(fitted.first_pass_residual).tolist(),
    )
    if not fitted.converged:
        row["not_converged_reason"] = "flux update did not reach its tolerance"
    return row


def _run_stored(case, args):
    from nova.equilibrium.flux_iteration import FluxIterationFit
    from nova.imas.mast_efit_referee import score_with_efit_referee
    from tests import test_magnetics_fit_efit_comparison as study

    data = case["data"]
    fitter = FluxIterationFit(
        case["profile"],
        case["magnetics"],
        case["rows"],
        coefficient_basis=case["basis"],
        conductor_response=case["conductor_response"],
        conductor_names=case["conductor_names"],
        prescribed_current=case["prescribed_current"],
    )
    try:
        fitted = fitter.fit(
            case["measurement"],
            case["initial_flux"],
            plasma_current_scale=float(data["plasma_current_scale"]),
            on_pass=_progress(case["name"]),
            max_passes=args.max_passes,
            flux_tolerance=args.flux_tolerance,
            max_update=args.max_update,
            relaxation=args.relaxation,
            step_rule=STEP_RULE,
        )
    except Exception:
        # the starting map is still a state worth showing beside the reference
        data["terminal_residual"] = None
        study._panel(data, False, flux=case["initial_flux"], suffix="-refused")
        raise
    data["terminal_residual"] = fitted.terminal_misfit
    panel_error = None
    try:
        study._panel(data, bool(fitted.converged), flux=fitted.terminal_state.flux)
    except Exception as refusal:  # the row outlives a painter's refusal
        panel_error = f"{type(refusal).__name__}: {refusal}"
    scores, scores_error = None, None
    try:
        state, centroid = study._as_equilibrium(data, fitted, fitted.source_profile)
        scores = score_with_efit_referee(
            state,
            data["reference_eq"],
            current_centroid_z_m=centroid,
            reference_slice_index=data["reference_slice"],
            reference_current_centroid_z_m=study._reference_centroid(data),
        )
    except Exception as error:
        scores_error = f"{type(error).__name__}: {error}"
    control, control_error = None, None
    # the control is judged on the cases the fit converged, so a case it did
    # not converge does not spend a second full fit on a control no band reads
    try:
        if not fitted.converged:
            raise _ControlNotJudged
        scaled_measurement = _gain_error(case["measurement"])
        scaled = fitter.fit(
            scaled_measurement,
            case["initial_flux"],
            plasma_current_scale=float(data["plasma_current_scale"]) * GAIN_ERROR,
            on_pass=_progress(case["name"] + "-control"),
            max_passes=args.max_passes,
            flux_tolerance=args.flux_tolerance,
            max_update=args.max_update,
            relaxation=args.relaxation,
            step_rule=STEP_RULE,
        )
        scaled_current = abs(float(np.sum(scaled.terminal_state.cell_current)))
        control = abs(scaled_current - case["plasma_current"]) / case["plasma_current"]
    except _ControlNotJudged:
        control_error = "not judged: the fit did not converge, so no band reads it"
    except Exception as error:
        control_error = f"{type(error).__name__}: {error}"
    bands_applied = case["name"] != "iter"
    row = dict(
        case=case["name"],
        construction="stored case of the imas-efit comparison record",
        label=data["constraint_context"]["comparison_label"],
        converged=bool(fitted.converged),
        passes=len(fitted.passes),
        pass_flux_update=[item.flux_update for item in fitted.passes],
        pass_misfit=[item.misfit for item in fitted.passes],
        pass_fraction=[item.fraction for item in fitted.passes],
        pass_trials=[item.trials for item in fitted.passes],
        stopped_improving_pass=fitted.stopped_improving_pass,
        stalled_pass=fitted.stalled_pass,
        stalled_message=fitted.stalled_message,
        axis_lost_pass=fitted.axis_lost_pass,
        axis_loss_message=fitted.axis_loss_message,
        rejected_pass=fitted.rejected_pass,
        rejected_message=fitted.rejected_message,
        topology_error=fitted.topology_error,
        panel_error=panel_error,
        terminal_flux_finite=bool(np.all(np.isfinite(fitted.terminal_state.flux))),
        terminal_misfit=fitted.terminal_misfit,
        last_flux_update=fitted.last_flux_update,
        coefficients=fitted.coefficients.tolist(),
        coefficient_names=list(fitted.coefficient_names),
        singular_values=fitted.singular_values.tolist(),
        undetermined_coefficients=list(fitted.undetermined_coefficients),
        flux_evaluations=fitted.flux_evaluations,
        sensor_evaluations=fitted.sensor_evaluations,
        wall_seconds=fitted.wall_seconds,
        max_passes=args.max_passes,
        flux_tolerance=args.flux_tolerance,
        step_rule=STEP_RULE,
        bands_applied=bands_applied,
        scores=scores,
        scores_error=scores_error,
        reconstructed_plasma_current_a=float(
            np.sum(fitted.terminal_state.cell_current)
        ),
        stored_plasma_current_a=float(case["reference_plasma_current"]),
        scaled_control_judged=bool(fitted.converged),
        scaled_control_relative_error=control,
        scaled_control_band=0.015,
        scaled_control_fired=bool(
            fitted.converged and control is not None and control > 0.015
        ),
        scaled_control_error=control_error,
        terminal_axis=(
            np.asarray(fitted.terminal_state.topology.axis).tolist()
            if fitted.terminal_state.topology is not None
            else None
        ),
        terminal_flux_span=(
            float(fitted.terminal_state.topology.flux_span)
            if fitted.terminal_state.topology is not None
            else None
        ),
    )
    if not bands_applied:
        row["judged"] = "same magnetics, different constraints; gap reported"
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=",".join((*KNOWN, *STORED)))
    parser.add_argument("--output", required=True)
    parser.add_argument("--figures", required=True)
    parser.add_argument("--generating-receipts")
    parser.add_argument("--max-passes", type=int, default=STEP_BUDGET)
    parser.add_argument("--flux-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--max-update", type=float, default=1.0)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures = Path(args.figures).resolve()
    # the known-state and stored-case painters both name a case "iter"
    os.environ["NOVA_MAGNETICS_FIT_FIGURES"] = str(figures / "known-states")
    os.environ["NOVA_EFIT_COMPARISON_FIGURES"] = str(figures / "stored-cases")
    os.environ["NOVA_MAGNETICS_FIT_EVIDENCE"] = str(output / "known-state-evidence")
    os.environ["NOVA_EFIT_COMPARISON_OUTPUT"] = str(output / "stored-case-evidence")
    if args.generating_receipts:
        os.environ["NOVA_MAGNETICS_FIT_GENERATING_RECEIPTS"] = str(
            Path(args.generating_receipts).resolve()
        )
    import jax

    from nova.jax.config import configure_dtypes

    configure_dtypes()
    print(
        "RUN_HEADER "
        + json.dumps(
            dict(
                device=str(jax.devices()[0]),
                backend=str(jax.default_backend()),
                x64=bool(jax.config.jax_enable_x64),
                module=__file__,
                cwd=os.getcwd(),
                cases=args.cases,
                max_passes=args.max_passes,
                flux_tolerance=args.flux_tolerance,
                max_update=args.max_update,
                relaxation=args.relaxation,
                step_rule=STEP_RULE,
            )
        ),
        flush=True,
    )
    factory = _Factory(output / "known-state-evidence")
    rows = []
    status = 0
    for name in [item for item in args.cases.split(",") if item]:
        started = perf_counter()
        try:
            if name in KNOWN:
                row = _run_known(_known_case(name, factory), args)
            elif name in STORED:
                row = _run_stored(_stored_case(STORED[name]), args)
            else:
                raise ValueError(f"unknown case {name}")
        except Exception as error:  # persisted, then reported as failed evidence
            row = dict(
                case=name,
                construction="failed",
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
            status = 1
        row["case_wall_seconds"] = perf_counter() - started
        if name in KNOWN:
            row["bands_met"] = bool(
                row.get("flux_map_error_over_span", np.inf)
                <= row.get("flux_map_band", 0.0)
                and row.get("whitened_residual_sup", np.inf)
                <= row.get("whitened_residual_band", 0.0)
            )
            if not row["bands_met"]:
                status = 1
        rows.append(row)
        _write(output / f"{name}-result.json", row)
        print("CASE_RESULT " + json.dumps(row), flush=True)
    table = dict(
        device=str(jax.devices()[0]),
        cases=rows,
        converged=[row["case"] for row in rows if row.get("converged")],
        bands_met=[row["case"] for row in rows if row.get("bands_met")],
        failed=[row["case"] for row in rows if row.get("construction") == "failed"],
        control_fired=[row["case"] for row in rows if row.get("scaled_control_fired")],
        control_judged=[
            row["case"] for row in rows if row.get("scaled_control_judged")
        ],
        control_unfired=[
            row["case"]
            for row in rows
            if row.get("scaled_control_judged") and not row.get("scaled_control_fired")
        ],
    )
    _write(output / "result-table.json", table)
    print("RESULT_TABLE " + json.dumps(table), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
