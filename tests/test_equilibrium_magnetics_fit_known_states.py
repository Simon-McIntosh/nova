"""Known-state recovery cases for the magnetics fit."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from nova.equilibrium import MagneticsFit
from tests.magnetics_fit_harness import (
    _draw_recovery_panels,
    _persist_forward,
    _total_observations,
    physical_case_table,
)


def _recover_physical(case, displacement=0.0):
    import json

    sensors = case["sensors"]
    label = f"{case['name']}-" + ("displaced-fit" if displacement else "fit")
    initial_record = {}

    def magnetic_source(receipt):
        if not initial_record:
            _persist_forward(receipt, case["output"] / f"{label}-initial.pkl")
            expected = case["generated"].terminal_state
            count = len(case["coordinate"])
            core = np.asarray(expected.domains.core)
            error = float(
                np.max(
                    np.abs(
                        np.asarray(
                            receipt.terminal_state.flux[:count] - expected.flux[:count]
                        )
                    )[core]
                )
                / abs(float(expected.topology.flux_span))
            )
            initial_record["flux_sup_fraction"] = error
            (case["output"] / f"{label}-initial.json").write_text(
                json.dumps(initial_record, indent=2)
            )
            print("INITIAL_RECOVERY " + json.dumps(initial_record), flush=True)
            assert error > 1e-6, initial_record
        return case["source"](receipt)

    if displacement:
        radius = sensors.r.copy()
        radius[0] += displacement
        sensors = replace(sensors, r=radius)
    total, vacuum, plasma, scales = _total_observations(
        case["source"],
        case["generated"],
        sensors,
        case["known_field"],
        case["measurement"].scale,
    )
    measurement = replace(case["measurement"], vacuum=vacuum)
    observation = dict(
        displacement_m=displacement,
        conductor_flux_sup=float(np.max(np.abs(vacuum[sensors.flux_loop]))),
        conductor_probe_sup=float(np.max(np.abs(vacuum[~sensors.flux_loop]))),
        subtraction_error=float(np.max(np.abs((total - vacuum - plasma) / scales))),
        geometry=sensors.r.tolist(),
        total=total.tolist(),
        vacuum=vacuum.tolist(),
        plasma=plasma.tolist(),
    )
    if displacement:
        delta = float(np.max(np.abs((vacuum - case["measurement"].vacuum) / scales)))
        assert delta > 1e-12, delta
        observation["conductor_geometry_delta"] = delta
    (case["output"] / f"{label}-observations.json").write_text(
        json.dumps(observation, indent=2)
    )
    print(
        "TOTAL_OBSERVATIONS "
        + json.dumps(
            {
                k: v
                for k, v in observation.items()
                if k not in ("geometry", "total", "vacuum", "plasma")
            }
        ),
        flush=True,
    )
    fitted = MagneticsFit(
        case["profile"],
        case["initial"],
        sensors,
        magnetic_source,
        coefficient_basis=case["basis"],
    ).fit(
        measurement,
        plasma_current_scale=abs(case["measurement"].plasma_current),
    )
    forward_path = case["output"] / f"{label}-forward.pkl"
    _persist_forward(fitted.forward, forward_path)
    core = fitted.source_profile.core
    record = dict(
        coefficients=fitted.coefficients.tolist(),
        coefficient_names=list(fitted.coefficient_names),
        forward_receipt=str(forward_path),
        iterations=fitted.iterations,
        singular_values=fitted.singular_values.tolist(),
        undetermined_coefficients=list(fitted.undetermined_coefficients),
        null_directions=fitted.null_directions.tolist(),
        whitened_residuals=fitted.whitened_residuals.tolist(),
        covariance=fitted.covariance.tolist(),
        converged=fitted.converged,
        forward_evaluations=fitted.forward_evaluations,
        source_profile=dict(
            carrier=case["request"].carrier_identity,
            pressure_coefficients=np.asarray(core.p_prime.coefficients).tolist(),
            pressure_normalisation=float(core.p_prime.normalisation),
            diamagnetic_coefficients=np.asarray(core.ff_prime.coefficients).tolist(),
            diamagnetic_normalisation=float(core.ff_prime.normalisation),
            boundary_pressure=float(fitted.source_profile.boundary_pressure),
            boundary_field_function=float(
                fitted.source_profile.boundary_field_function
            ),
        ),
    )
    (case["output"] / f"{label}.json").write_text(json.dumps(record, indent=2))
    expected = case["generated"].terminal_state
    terminal = fitted.forward.terminal_state
    count = len(case["coordinate"])
    core = np.asarray(expected.domains.core)
    span = abs(float(expected.topology.flux_span))
    error = float(
        np.max(np.abs(np.asarray(terminal.flux[:count] - expected.flux[:count]))[core])
        / span
    )
    return fitted, error


@pytest.fixture(scope="module")
def physical_cases(tmp_path_factory):
    """Keep each carrier, compiled program and diagnostic table in one process."""
    return physical_case_table(tmp_path_factory)


@pytest.mark.slow
@pytest.mark.timeout(3500)
@pytest.mark.parametrize("name", ("solovev", "mast", "iter"))
def test_known_state_recovery(name, physical_cases):
    import json
    import os

    case = physical_cases(name)
    displacement = (
        float(os.environ.get("NOVA_MAGNETICS_FIT_DISPLACEMENT", "0"))
        if name == "solovev"
        else 0.0
    )
    fitted, error = _recover_physical(case, displacement)
    _persist_forward(fitted.forward, case["output"] / f"{name}-recovered.pkl")
    resolved = case["profile"].solve(
        replace(case["request"], source_profile=fitted.source_profile)
    )
    _persist_forward(resolved, case["output"] / f"{name}-resolved.pkl")
    agreement = float(
        np.max(
            np.abs(
                np.asarray(
                    resolved.terminal_state.flux - fitted.forward.terminal_state.flux
                )
            )
        )
        / np.max(np.abs(np.asarray(fitted.forward.terminal_state.flux)))
    )
    summary = dict(
        case=name,
        flux_sup_fraction=error,
        residual_sup=float(np.max(np.abs(fitted.whitened_residuals))),
        largest_residual_row=int(np.argmax(np.abs(fitted.whitened_residuals))),
        resolve_relative_sup=agreement,
        iterations=fitted.iterations,
        singular_values=fitted.singular_values.tolist(),
        undetermined_coefficients=list(fitted.undetermined_coefficients),
        forward_evaluations=fitted.forward_evaluations,
        coefficients=fitted.coefficients.tolist(),
        coefficient_names=list(fitted.coefficient_names),
        terminal_residual=float(fitted.forward.terminal_state.fixed_point.residual),
        terminal_converged=bool(fitted.forward.terminal_state.fixed_point.converged),
        forward_tolerance=case["request"].policy.kernel_tolerance,
    )
    (case["output"] / f"{name}-recovery.json").write_text(json.dumps(summary, indent=2))
    print("RECOVERY " + json.dumps(summary), flush=True)
    _draw_recovery_panels(name, case, fitted)
    assert error <= 1e-6, summary
    assert summary["residual_sup"] <= 1e-8, summary
    assert bool(resolved.qualified), summary
    assert agreement <= case["request"].policy.kernel_tolerance, summary
    assert fitted.singular_values.size == 2
    assert fitted.iterations > 0


@pytest.mark.slow
@pytest.mark.timeout(3500)
def test_displaced_flux_loop_is_identified(physical_cases):
    import json

    case = physical_cases("solovev")
    fitted, error = _recover_physical(case, 0.05)
    summary = dict(
        flux_sup_fraction=error,
        largest_residual_row=int(np.argmax(np.abs(fitted.whitened_residuals))),
        residual_sup=float(np.max(np.abs(fitted.whitened_residuals))),
    )
    (case["output"] / "displaced-loop.json").write_text(json.dumps(summary, indent=2))
    print("DISPLACED_LOOP " + json.dumps(summary), flush=True)
    assert error > 1e-6, summary
    assert summary["largest_residual_row"] == 0, summary
