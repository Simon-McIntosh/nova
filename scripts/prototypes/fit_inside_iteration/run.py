"""Measure the least-squares fit inside the equilibrium iteration.

Runs the known states of the §1 record and the stored imas-efit cases of the
§2 record through :class:`nova.equilibrium.flux_iteration.FluxIterationFit`, in
sequence inside one allocation, and writes a per-case table plus one overlaid
line-contour panel per case. Each case carries its negative control: every
magnetics row, the measured plasma current and the measured conductor currents
scaled by 1.10 must leave the 1.5 % plasma-current band when the case converged.

The known states reuse the row construction, generating receipts and starting
fields of the §1 study; the stored cases reuse the reader, rows, profile basis
and fitted-conductor projection of the §2 study. No second construction of
either is introduced here.

Usage, on a compute node with the shared environment's interpreter:

    python scripts/prototypes/fit_inside_iteration/run.py \
        --cases solovev,mast,iter,mastu,tcv,iter-stored \
        --output <run-dir>/evidence --figures <run-dir>/figures
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
    from tests import test_equilibrium_magnetics_fit as study

    case = study.physical_cases.__wrapped__(factory)(name)
    initial = case["initial"]
    flux = np.asarray(initial.seed_policy.flux, dtype=np.float64)
    profile = case["profile"]._with_source(initial.source_profile)
    rows = _CellRows(case["source"].biot, case["sensors"], study._physical_sensor_rows)
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
        figure=case,
    )


def _stored_case(name):
    import pytest
    from tests import test_magnetics_fit_efit_comparison as study
    from tests.test_equilibrium_magnetics_fit import (
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
        plasma_current=abs(float(data["case"].plasma_current)),
        data=data,
    )


def _run_known(case, args):
    from nova.equilibrium.flux_iteration import FluxIterationFit
    from tests import test_equilibrium_magnetics_fit as study

    fitter = FluxIterationFit(
        case["profile"],
        case["magnetics"],
        case["rows"],
        coefficient_basis=case["basis"],
    )
    fitted = fitter.fit(
        case["measurement"],
        case["initial_flux"],
        plasma_current_scale=case["plasma_current"],
        max_passes=args.max_passes,
        flux_tolerance=args.flux_tolerance,
        relaxation=args.relaxation,
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
    scaled_measurement = replace(
        case["measurement"],
        measured=case["measurement"].measured * 1.10,
        scale=case["measurement"].scale * 1.10,
        plasma_current=case["measurement"].plasma_current * 1.10,
    )
    scaled = fitter.fit(
        scaled_measurement,
        case["initial_flux"],
        plasma_current_scale=case["plasma_current"] * 1.10,
        max_passes=args.max_passes,
        flux_tolerance=args.flux_tolerance,
        relaxation=args.relaxation,
    )
    scaled_current = abs(float(np.sum(scaled.terminal_state.cell_current)))
    control = abs(scaled_current - case["plasma_current"]) / case["plasma_current"]
    shim = SimpleNamespace(
        forward=SimpleNamespace(terminal_state=fitted.terminal_state)
    )
    study._draw_recovery_panels(case["name"], case["figure"], shim)
    row = dict(
        case=case["name"],
        construction="known state of the §1 record",
        converged=bool(fitted.converged),
        passes=len(fitted.passes),
        pass_flux_update=[item.flux_update for item in fitted.passes],
        pass_misfit=[item.misfit for item in fitted.passes],
        stopped_improving_pass=fitted.stopped_improving_pass,
        axis_lost_pass=fitted.axis_lost_pass,
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
        nested_forward_calls=NESTED_FORWARD_CALLS[case["name"]],
        reconstructed_plasma_current_a=float(
            np.sum(fitted.terminal_state.cell_current)
        ),
        generating_plasma_current_a=float(case["measurement"].plasma_current),
        scaled_control_relative_error=control,
        scaled_control_band=0.015,
        scaled_control_fired=bool(control > 0.015),
        relaxed=fitted.relaxation,
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
    )
    fitted = fitter.fit(
        case["measurement"],
        case["initial_flux"],
        plasma_current_scale=float(data["plasma_current_scale"]),
        max_passes=args.max_passes,
        flux_tolerance=args.flux_tolerance,
        relaxation=args.relaxation,
    )
    state, centroid = study._as_equilibrium(data, fitted, fitted.source_profile)
    scores = score_with_efit_referee(
        state,
        data["reference_eq"],
        current_centroid_z_m=centroid,
        reference_slice_index=data["reference_slice"],
        reference_current_centroid_z_m=study._reference_centroid(data),
    )
    scaled_measurement = replace(
        case["measurement"],
        measured=case["measurement"].measured * 1.10,
        scale=case["measurement"].scale * 1.10,
        plasma_current=case["measurement"].plasma_current * 1.10,
        conductor_current=case["measurement"].conductor_current * 1.10,
        conductor_scale=case["measurement"].conductor_scale * 1.10,
    )
    scaled = fitter.fit(
        scaled_measurement,
        case["initial_flux"],
        plasma_current_scale=float(data["plasma_current_scale"]) * 1.10,
        max_passes=args.max_passes,
        flux_tolerance=args.flux_tolerance,
        relaxation=args.relaxation,
    )
    scaled_current = abs(float(np.sum(scaled.terminal_state.cell_current)))
    control = abs(scaled_current - case["plasma_current"]) / case["plasma_current"]
    data["terminal_residual"] = fitted.terminal_misfit
    study._panel(data, bool(fitted.converged), flux=fitted.terminal_state.flux)
    bands_applied = case["name"] != "iter"
    row = dict(
        case=case["name"],
        construction="stored imas-efit case of the §2 record",
        label=data["constraint_context"]["comparison_label"],
        converged=bool(fitted.converged),
        passes=len(fitted.passes),
        pass_flux_update=[item.flux_update for item in fitted.passes],
        pass_misfit=[item.misfit for item in fitted.passes],
        stopped_improving_pass=fitted.stopped_improving_pass,
        axis_lost_pass=fitted.axis_lost_pass,
        terminal_misfit=fitted.terminal_misfit,
        last_flux_update=fitted.last_flux_update,
        coefficients=fitted.coefficients.tolist(),
        coefficient_names=list(fitted.coefficient_names),
        singular_values=fitted.singular_values.tolist(),
        undetermined_coefficients=list(fitted.undetermined_coefficients),
        flux_evaluations=fitted.flux_evaluations,
        sensor_evaluations=fitted.sensor_evaluations,
        wall_seconds=fitted.wall_seconds,
        bands_applied=bands_applied,
        scores=scores,
        reconstructed_plasma_current_a=float(
            np.sum(fitted.terminal_state.cell_current)
        ),
        stored_plasma_current_a=float(case["plasma_current"]),
        scaled_control_relative_error=control,
        scaled_control_band=0.015,
        scaled_control_fired=bool(control > 0.015),
        relaxed=fitted.relaxation,
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
    parser.add_argument("--max-passes", type=int, default=64)
    parser.add_argument("--flux-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--relaxation", type=float, default=1.0)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["NOVA_MAGNETICS_FIT_FIGURES"] = str(Path(args.figures).resolve())
    os.environ["NOVA_EFIT_COMPARISON_FIGURES"] = str(Path(args.figures).resolve())
    os.environ["NOVA_MAGNETICS_FIT_EVIDENCE"] = str(output / "known-state-evidence")
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
                relaxation=args.relaxation,
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
        rows.append(row)
        _write(output / f"{name}-result.json", row)
        print("CASE_RESULT " + json.dumps(row), flush=True)
    table = dict(
        device=str(jax.devices()[0]),
        cases=rows,
        converged=[row["case"] for row in rows if row.get("converged")],
        failed=[row["case"] for row in rows if row.get("construction") == "failed"],
        control_fired=[row["case"] for row in rows if row.get("scaled_control_fired")],
        control_unfired=[
            row["case"]
            for row in rows
            if row.get("converged") and not row.get("scaled_control_fired")
        ],
    )
    _write(output / "result-table.json", table)
    print("RESULT_TABLE " + json.dumps(table), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
