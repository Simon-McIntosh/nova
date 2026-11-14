"""Profile-coefficient fitting and its public-forward authority boundary."""

from __future__ import annotations

from dataclasses import fields, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import MagneticsFit, MagneticsFitReceipt
from nova.equilibrium import magnetics_fit
from nova.equilibrium.measurement import Magnetics, SliceMeasurement, whitened_solve
from nova.equilibrium.solve_request import ForwardSolveReceipt, ForwardSolveRequest
from nova.equilibrium.source import DomainProfile, ForwardSource, PolynomialFluxFunction
from nova.jax.config import configure_dtypes


def test_rank_deficient_data_exposes_its_null_direction():
    response = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    target = np.array([2.0, 4.0, 6.0])
    coefficient, covariance, spectrum, null = whitened_solve(
        response, target, np.ones(3), 0.0, diagnostics=True
    )
    np.testing.assert_allclose(response @ coefficient, target, atol=2.0e-15)
    assert np.all(np.isfinite(covariance))
    assert spectrum[0] > 1.0
    assert spectrum[1] < 1.0e-14
    assert null.shape == (1, 2)
    np.testing.assert_allclose(response @ null.T, 0.0, atol=2.0e-15)


def test_regularisation_does_not_claim_data_identifiability():
    response = np.array([[1.0, 1.0]])
    _, _, spectrum, null = whitened_solve(
        response,
        np.array([1.0]),
        np.ones(1),
        1.0e-3,
        diagnostics=True,
        penalty=np.array([1.0, 4.0]),
    )
    assert spectrum.size == 1
    assert null.shape == (1, 2)
    np.testing.assert_allclose(response @ null.T, 0.0, atol=1.0e-15)


def test_current_anchor_participates_in_the_data_spectrum():
    response = np.array([[1.0, 1.0]])
    coefficient, _, spectrum, null = whitened_solve(
        response,
        np.array([3.0]),
        np.array([0.5]),
        0.0,
        anchor=(np.array([1.0, -1.0]), 1.0, 2.0),
        diagnostics=True,
    )
    np.testing.assert_allclose(coefficient, [2.0, 1.0], atol=1.0e-15)
    assert spectrum.shape == (2,)
    assert spectrum[-1] > 0.0
    assert null.shape == (0, 2)


def test_fit_receipt_and_forward_request_keep_the_authority_boundary():
    assert MagneticsFitReceipt.__module__ == "nova.equilibrium.magnetics_fit"
    assert MagneticsFit.__module__ == "nova.equilibrium.magnetics_fit"
    names = {item.name for item in fields(ForwardSolveRequest)}
    assert "measurement" not in names
    with pytest.raises(TypeError, match="measurement"):
        ForwardSolveRequest(measurement=np.zeros(4))


@pytest.fixture
def linear_fit(monkeypatch):
    """An algebraic instrument checks orchestration without a physics claim."""
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    source = ForwardSource(
        core=DomainProfile(
            p_prime=PolynomialFluxFunction(jnp.array([0.8, 0.1])),
            ff_prime=PolynomialFluxFunction(jnp.array([1.2])),
        )
    )
    request = ForwardSolveRequest.from_defaults(
        carrier_identity="linear-response",
        source_profile=source,
        seed_policy=magnetics_fit.ExplicitSolveSeed(jnp.zeros(3)),
    )
    response = np.array(
        [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [2.0, 1.0, 1.0], [1.0, -2.0, 1.0]]
    )
    calls = []

    def solve(active):
        assert isinstance(active, ForwardSolveRequest)
        calls.append(active)
        coefficients = np.r_[
            active.source_profile.core.p_prime.coefficients,
            active.source_profile.core.ff_prime.coefficients,
        ]
        return ForwardSolveReceipt(
            terminal_state=SimpleNamespace(
                flux=coefficients,
                cell_current=coefficients,
            ),
            qualified=True,
            termination_reason=0,
            residual_history=np.zeros(1),
            changed_mask_entries=np.zeros(1),
            support_cell_count=np.ones(1),
            globalisation_decisions=([], []),
            amplitude_history=np.ones(1),
            topology_read=None,
            polish_receipt=None,
            compilation_cache_hit=True,
            wall_seconds=0.0,
            resolved_defaults=None,
        )

    def signal(biot, coordinates, *args):
        indices = coordinates[:, 0].astype(int) - 1
        return SimpleNamespace(values=response[indices] @ biot)

    monkeypatch.setattr(magnetics_fit, "virtual_flux_loops", signal)
    monkeypatch.setattr(magnetics_fit, "virtual_poloidal_probes", signal)
    geometry = Magnetics(
        r=np.arange(1.0, 5.0),
        z=np.zeros(4),
        angle=np.zeros(4),
        flux_loop=np.array([True, False, True, False]),
    )
    fitter = MagneticsFit(
        SimpleNamespace(solve=solve),
        request,
        geometry,
        lambda receipt: receipt.terminal_state.flux,
    )
    truth = np.array([1.0, 0.3, 0.9])
    measurement = SliceMeasurement(
        measured=response @ truth,
        vacuum=np.zeros(4),
        mask=np.ones(4, dtype=bool),
        scale=np.array([0.5, 2.0, 1.0, 1.0]),
        plasma_current=truth.sum(),
    )
    return fitter, measurement, truth, calls


def test_coefficient_updates_issue_only_typed_source_requests(linear_fit):
    fitter, measurement, truth, calls = linear_fit
    receipt = fitter.fit(measurement, plasma_current_scale=1.0)
    assert receipt.converged
    assert receipt.iterations > 0
    assert receipt.forward_evaluations == len(calls)
    np.testing.assert_allclose(receipt.coefficients, truth, atol=1.0e-11)
    assert np.max(np.abs(receipt.whitened_residuals)) <= 1.0e-8
    assert receipt.undetermined_coefficients == ()
    assert receipt.singular_values.shape == (3,)
    assert all(active.target_current is None for active in calls)
    np.testing.assert_array_equal(
        fitter.request.source_profile.core.p_prime.coefficients, [0.8, 0.1]
    )


def test_masked_missing_rows_cannot_poison_the_fit(linear_fit):
    fitter, measurement, truth, _ = linear_fit
    missing = measurement.measured.copy()
    missing[1] = np.nan
    measurement = replace(
        measurement, measured=missing, mask=np.array([1, 0, 1, 1], bool)
    )
    receipt = fitter.fit(measurement, plasma_current_scale=1.0)
    assert receipt.converged
    assert receipt.whitened_residuals[1] == 0.0
    np.testing.assert_allclose(receipt.coefficients, truth, atol=1.0e-11)


def test_unqualified_forward_state_is_refused_before_sensor_read(linear_fit):
    fitter, measurement, _, _ = linear_fit
    solve = fitter.profile.solve
    fitter.profile.solve = lambda request: replace(solve(request), qualified=False)
    with pytest.raises(RuntimeError, match="did not qualify") as caught:
        fitter.fit(measurement, plasma_current_scale=1.0)
    assert isinstance(caught.value.receipt, ForwardSolveReceipt)
    assert not caught.value.receipt.qualified


def test_declared_basis_recovers_coefficients_and_a_resolvable_source(linear_fit):
    fitter, measurement, truth, _ = linear_fit
    basis = np.column_stack(([0.8, 0.1, 1.2], truth))
    fitted = MagneticsFit(
        fitter.profile,
        fitter.request,
        fitter.magnetics,
        fitter.magnetic_source,
        coefficient_basis=basis,
    ).fit(measurement, plasma_current_scale=1.0)
    assert fitted.converged
    np.testing.assert_allclose(fitted.coefficients, [0.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(
        np.r_[
            fitted.source_profile.core.p_prime.coefficients,
            fitted.source_profile.core.ff_prime.coefficients,
        ],
        truth,
        atol=1e-10,
    )
    assert fitted.singular_values.shape == (2,)


def test_dependent_basis_is_refused(linear_fit):
    fitter, _, _, _ = linear_fit
    with pytest.raises(ValueError, match="independent columns"):
        MagneticsFit(
            fitter.profile,
            fitter.request,
            fitter.magnetics,
            fitter.magnetic_source,
            coefficient_basis=np.ones((3, 2)),
        )


def _physical_problem(name):
    """Load fixture-owned machine operands and declare a polynomial source."""
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.forward_operator import ForwardFluxOperator
    from nova.equilibrium.solve_request import ColdSeedPortfolio

    if name == "solovev":
        from benchmarks import solovev_certificate as lane
        from benchmarks.profile_coefficients_recompile_audit import _scaled_source

        kind = "moderate-rotation-conventional-static"
        profile, _, request, _ = lane._certificate_compile_problem(kind, -300)
        carrier, _, exact = lane._case(kind)
        machine = lane._case_machine(kind, carrier, exact, -300)
        source = _scaled_source(profile.source, 1.0, 1.0)
        profile = ForwardProfile(
            replace(profile.operator, source=source), profile.lattice
        )
        request = replace(
            request, source_profile=source, target_current=None, vertical_pin="omit"
        )
        return profile, request, machine.node, machine.wall_node, machine.cell_polygons

    if name == "mast":
        from benchmarks import coil_edit_latency as cached
        from benchmarks import efit_forward_parity_slice as lane
        from benchmarks.mast_response_carrier_warm import DEFAULT_CARRIER
        from nova.imas.mast_solve_inputs import SHOT_STORE

        response, _ = cached._response_cache(DEFAULT_CARRIER)
        case, context = lane._mast_case_from_selection(
            SHOT_STORE,
            dict(shot=22086, slice_index=43),
            None,
        )
        case, declared, _ = lane._passive_inclusive_case(case, context, response)
        old = declared.operator
        operator = ForwardFluxOperator(
            grid=old.grid,
            wall=old.wall,
            source=old.source,
            external_current=old.external_current,
            area=old.area,
            polarity=old.polarity,
            inside_material=old.inside_material,
            use_linear_moments=False,
            prescribed_current_field=old.prescribed_current_field,
        )
        profile = ForwardProfile(operator, declared.lattice)
        seed = jnp.asarray(case["state"])
        coordinate = np.asarray(profile.lattice.coordinate)
        wall = np.asarray(profile.operator.wall.coordinate)
        dr = float(profile.lattice.radial_step)
        dz = float(profile.lattice.vertical_step)
        corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) * [
            dr,
            dz,
        ]
        sections = tuple(point + corners for point in coordinate)
        prescribed = old.prescribed_current_field.current
        plasma_current = float(case["reference"]["plasma_current_a"])
    else:
        from tests import test_equilibrium_forward_reference as lane

        case = lane.reference_case()
        if isinstance(case, str):
            raise RuntimeError(case)
        machine = lane.cached_machine(case, -500, passive=True)
        profile = lane.forward_profile(case, machine)
        seed = lane.seed_flux(case, machine)
        coordinate, wall, sections = (
            machine.node,
            machine.wall_node,
            machine.cell_polygons,
        )
        prescribed = None
        plasma_current = float(case.plasma_current)
    _, stored_topology = profile.operator.read(seed)
    centroid = tuple(np.asarray(stored_topology.axis, dtype=float))
    sign = -np.sign(plasma_current)
    source = ForwardSource(
        core=DomainProfile(
            p_prime=PolynomialFluxFunction(
                jnp.asarray([1.0, -1.0]), sign * 3.0e5 / centroid[0] ** 2
            ),
            ff_prime=PolynomialFluxFunction(jnp.asarray([1.0, -1.0]), sign * 0.25),
        ),
        boundary_pressure=profile.source.boundary_pressure,
        boundary_field_function=profile.source.boundary_field_function,
    )
    profile = ForwardProfile(replace(profile.operator, source=source), profile.lattice)
    estimated_current = float(np.sum(profile.operator.cell_current(seed)))
    amplitude = plasma_current / estimated_current
    assert np.isfinite(amplitude) and amplitude > 0.0
    source = replace(
        source,
        core=replace(
            source.core,
            p_prime=replace(
                source.core.p_prime,
                normalisation=source.core.p_prime.normalisation * amplitude,
            ),
            ff_prime=replace(
                source.core.ff_prime,
                normalisation=source.core.ff_prime.normalisation * amplitude,
            ),
        ),
    )
    profile = ForwardProfile(replace(profile.operator, source=source), profile.lattice)
    request = ForwardSolveRequest.from_defaults(
        carrier_identity=f"{name}-cached-polynomial-source",
        source_profile=source,
        seed_policy=ColdSeedPortfolio(
            plasma_current, centroid, requested_class="limited"
        ),
        prescribed_current=prescribed,
        vertical_pin="omit",
        policy_overrides=dict(kernel_tolerance=1e-12, qualification_tolerance=1e-12),
    )
    return profile, request, coordinate, wall, sections


def _persist_forward(receipt, path):
    """Persist the complete forward result before deriving any score."""
    import copyreg
    import pickle
    from types import MappingProxyType
    from nova.equilibrium.forward_operator import ForwardTopologyState

    copyreg.pickle(type(MappingProxyType({})), lambda value: (dict, (dict(value),)))
    copyreg.pickle(
        ForwardTopologyState,
        lambda value: (
            ForwardTopologyState.tree_unflatten,
            (None, value.tree_flatten()[0]),
        ),
    )
    pending = path.with_suffix(".pending")
    with pending.open("wb") as stream:
        pickle.dump(receipt, stream)
    pending.replace(path)


def _persist_forward_failure(error, output, name):
    """Retain the public call's terminal history when receipt qualification raises."""
    import json
    import pickle
    import traceback
    from nova.equilibrium.fixed_point import FixedPointResult

    frames = list(traceback.walk_tb(error.__traceback__))
    histories = [
        frame.f_locals["history"]
        for frame, _ in frames
        if isinstance(frame.f_locals.get("history"), FixedPointResult)
    ]
    summary = dict(case=name, exception=type(error).__name__, message=str(error))
    if histories:
        history = histories[-1]
        destination = output / f"{name}-rejected-history.pkl"
        with destination.open("wb") as stream:
            pickle.dump(history, stream)
        summary.update(
            residual=float(history.residual),
            converged=bool(history.converged),
            termination_reason=int(history.termination_reason),
            trips=int(
                np.count_nonzero(np.isfinite(np.asarray(history.active_set_residuals)))
            ),
            history_path=str(destination),
        )
    admitted = [
        frame.f_locals["admitted"]
        for frame, _ in frames
        if frame.f_code.co_name == "require_qualified_axis"
    ]
    if admitted:
        summary["axis_admitted"] = bool(admitted[-1])
    (output / f"{name}-generating-failure.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("GENERATING_FAILURE " + json.dumps(summary), flush=True)


class _PlasmaDiagnosticSource:
    """Bind terminal cell currents to fixed authored diagnostic sections."""

    def __init__(self, sections):
        from nova.frame.coilset import CoilSet

        self.biot = CoilSet(dcoil=-1, field_attrs=["Psi", "Br", "Bz"])
        for index, section in enumerate(sections):
            self.biot.coil.insert(
                np.asarray(section),
                nturn=1.0,
                Ic=0.0,
                name=f"cell_{index}",
            )
        for name in ("point", "probe", "poloidal_flux_loop"):
            factory = getattr(self.biot, name)
            self._cache_factory(factory)

    @staticmethod
    def _cache_factory(factory):
        """Retain factory-owned Green tables while source currents vary."""
        original = factory.solve
        tables = {}

        def solve(*args):
            if args:
                coordinates = np.asarray(args[0], dtype=np.float64)
            else:
                coordinates = np.column_stack(
                    (factory.target["x"], factory.target["z"])
                )
            key = coordinates.shape, coordinates.tobytes()
            if key not in tables:
                original(*args)
                tables[key] = factory.data.copy(deep=True)
            else:
                factory.data = tables[key]
                factory.load_operators()

        factory.solve = solve

    def __call__(self, receipt):
        self.biot.sloc["Ic"] = np.asarray(receipt.terminal_state.cell_current)
        return self.biot


@pytest.fixture(scope="module")
def physical_cases(tmp_path_factory):
    """Keep each carrier, compiled program and diagnostic table in one process."""
    import os
    from pathlib import Path

    output = Path(
        os.environ.get(
            "NOVA_MAGNETICS_FIT_EVIDENCE", tmp_path_factory.mktemp("magnetics")
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    cases = {}

    def get(name):
        if name in cases:
            return cases[name]
        profile, request, coordinate, wall, sections = _physical_problem(name)
        recorded = os.environ.get("NOVA_MAGNETICS_FIT_GENERATING_RECEIPTS")
        recorded_path = (
            Path(recorded) / f"{name}-generating-receipt.pkl" if recorded else None
        )
        if recorded and (Path(recorded) / f"{name}-unqualified.json").exists():
            raise RuntimeError(
                f"{name}: all three recorded prescribed profiles failed qualification"
            )
        if recorded_path is not None and recorded_path.exists():
            import pickle

            with recorded_path.open("rb") as stream:
                generated = pickle.load(stream)
            request_path = Path(recorded) / f"{name}-generating-request.pkl"
            if request_path.exists():
                with request_path.open("rb") as stream:
                    request = pickle.load(stream)
            print(f"GENERATING_RECEIPT {recorded_path}", flush=True)
        else:
            import json

            attempts = (1.0,) if name == "solovev" else (1.0, 0.7, 1.3)
            original = request
            qualified = False
            for attempt, factor in enumerate(attempts, 1):
                label = f"{name}-profile-{attempt}"
                source = original.source_profile
                source = replace(
                    source,
                    core=replace(
                        source.core,
                        p_prime=replace(
                            source.core.p_prime,
                            coefficients=source.core.p_prime.coefficients * factor,
                        ),
                        ff_prime=replace(
                            source.core.ff_prime,
                            coefficients=source.core.ff_prime.coefficients * factor,
                        ),
                    ),
                )
                request = replace(original, source_profile=source)
                inputs = dict(
                    pressure_coefficients=np.asarray(
                        source.core.p_prime.coefficients
                    ).tolist(),
                    pressure_normalisation=float(source.core.p_prime.normalisation),
                    diamagnetic_coefficients=np.asarray(
                        source.core.ff_prime.coefficients
                    ).tolist(),
                    diamagnetic_normalisation=float(source.core.ff_prime.normalisation),
                    seed=request.seed_policy.provenance().to_dict(),
                )
                (output / f"{label}-inputs.json").write_text(
                    json.dumps(inputs, indent=2)
                )
                try:
                    generated = profile.solve(request)
                except Exception as error:
                    _persist_forward_failure(error, output, label)
                    continue
                _persist_forward(generated, output / f"{label}-receipt.pkl")
                summary = dict(
                    case=name,
                    attempt=attempt,
                    qualified=bool(generated.qualified),
                    converged=bool(generated.terminal_state.fixed_point.converged),
                    residual=float(generated.terminal_state.fixed_point.residual),
                    termination_reason=int(generated.termination_reason),
                    trips=int(
                        np.count_nonzero(
                            np.isfinite(np.asarray(generated.residual_history))
                        )
                    ),
                )
                (output / f"{label}-receipt.json").write_text(
                    json.dumps(summary, indent=2)
                )
                print("PRESCRIBED_GENERATING " + json.dumps(summary), flush=True)
                if summary["qualified"] and summary["converged"]:
                    qualified = True
                    break
            if not qualified:
                (output / f"{name}-unqualified.json").write_text(
                    json.dumps(dict(case=name, attempts=len(attempts), qualified=False))
                )
                raise RuntimeError(
                    f"{name}: no qualified prescribed profile "
                    f"in {len(attempts)} attempts"
                )
        _persist_forward(generated, output / f"{name}-generating.pkl")
        _persist_forward(generated, output / f"{name}-generating-receipt.pkl")
        _persist_forward(request, output / f"{name}-generating-request.pkl")
        residual = float(generated.terminal_state.fixed_point.residual)
        print(
            f"GENERATING {name} qualified={bool(generated.qualified)} "
            f"residual={residual:.17g}",
            flush=True,
        )
        assert bool(generated.qualified), (
            name,
            residual,
            request.policy.qualification_tolerance,
        )
        assert bool(generated.terminal_state.fixed_point.converged), (name, residual)
        if os.environ.get("NOVA_MAGNETICS_FIT_GENERATE_ONLY") == "1":
            return dict(profile=profile, request=request, generated=generated)
        index = np.linspace(0, len(wall) - 1, 24, dtype=int)
        point = np.asarray(wall)[index]
        sensors = Magnetics(
            r=np.tile(point[:, 0], 2),
            z=np.tile(point[:, 1], 2),
            angle=np.r_[np.zeros(24), np.linspace(0.0, 360.0, 24, endpoint=False)],
            flux_loop=np.r_[np.ones(24, bool), np.zeros(24, bool)],
        )
        source = _PlasmaDiagnosticSource(sections)
        instrument = MagneticsFit(profile, request, sensors, source)
        raw = np.r_[
            request.source_profile.core.p_prime.coefficients,
            request.source_profile.core.ff_prime.coefficients,
        ]
        _, rows = instrument._evaluate(raw, generated.terminal_state.flux)
        scales = np.r_[
            np.full(24, max(float(np.max(np.abs(rows[:24]))), 1e-12)),
            np.full(24, max(float(np.max(np.abs(rows[24:48]))), 1e-12)),
        ]
        measurement = SliceMeasurement(
            measured=rows[:-1],
            vacuum=np.zeros(48),
            mask=np.ones(48, bool),
            scale=scales,
            plasma_current=rows[-1],
        )
        pressure = request.source_profile.core.p_prime.coefficients.size
        basis = np.zeros((raw.size, 2))
        basis[:pressure, 0] = raw[:pressure]
        basis[pressure:, 1] = raw[pressure:]
        initial_source = replace(
            request.source_profile,
            core=replace(
                request.source_profile.core,
                p_prime=replace(
                    request.source_profile.core.p_prime,
                    coefficients=jnp.asarray(raw[:pressure] * 0.98),
                ),
                ff_prime=replace(
                    request.source_profile.core.ff_prime,
                    coefficients=jnp.asarray(raw[pressure:] * 1.02),
                ),
            ),
        )
        initial = replace(request, source_profile=initial_source)
        cases[name] = dict(
            name=name,
            profile=profile,
            request=request,
            initial=initial,
            generated=generated,
            coordinate=np.asarray(coordinate),
            wall=np.asarray(wall),
            source=source,
            sensors=sensors,
            measurement=measurement,
            basis=basis,
            output=output,
        )
        return cases[name]

    return get


def _recover_physical(case, displacement=0.0):
    sensors = case["sensors"]
    if displacement:
        radius = sensors.r.copy()
        radius[0] += displacement
        sensors = replace(sensors, r=radius)
    fitted = MagneticsFit(
        case["profile"],
        case["initial"],
        sensors,
        case["source"],
        coefficient_basis=case["basis"],
    ).fit(
        case["measurement"],
        plasma_current_scale=abs(case["measurement"].plasma_current),
    )
    label = "displaced-fit.pkl" if displacement else "fit.pkl"
    _persist_forward(fitted, case["output"] / f"{case['name']}-{label}")
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


def _draw_recovery_panels(name, case, fitted):
    """Save shared-level contour panels through the existing poloidal painters."""
    import os
    from pathlib import Path
    import xml.etree.ElementTree as element_tree
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from nova.media import poloidal
    from nova.media.ink import DEFAULT_INK, poloidal_axes

    directory = os.environ.get("NOVA_MAGNETICS_FIT_FIGURES")
    if directory is None:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    generated = case["generated"].terminal_state
    recovered = fitted.forward.terminal_state
    levels = np.sort(
        np.linspace(
            float(generated.topology.axis_flux),
            float(generated.topology.boundary_flux),
            14,
        )
    )[1:-1]
    grey, blue = "#929292", DEFAULT_INK.flux_color
    original_style = DEFAULT_INK.variant(
        axis_color=grey, xpoint_color=grey, axis_markersize=9.0, xpoint_markersize=10.0
    )
    recovered_style = DEFAULT_INK.variant(axis_color=blue, xpoint_color=blue)
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 7.2), dpi=100)
    figure.subplots_adjust(left=0.03, right=0.97, bottom=0.06, top=0.84, wspace=0.12)
    coordinate = case["coordinate"]
    for axis, state, color, label in zip(
        axes,
        (generated, recovered),
        (grey, blue),
        ("generating", "recovered"),
        strict=True,
    ):
        poloidal_axes(axis)
        poloidal.draw_scattered_contours(
            axis,
            coordinate[:, 0],
            coordinate[:, 1],
            np.asarray(state.flux)[: len(coordinate)],
            levels,
            case["wall"],
            color=color,
            linewidth=1.2,
        )
        poloidal.draw_wall(axis, units=case["wall"])
        for landmarks, style in (
            (generated.topology, original_style),
            (recovered.topology, recovered_style),
        ):
            diverted = bool(landmarks.diverted)
            poloidal.draw_nulls(
                axis,
                magnetic_axis=np.asarray(landmarks.axis),
                x_points=np.asarray(landmarks.x_point) if diverted else None,
                other_x_points=np.asarray(landmarks.x_point) if not diverted else None,
                contain=case["wall"],
                style=style,
            )
        axis.text(0.5, 1.02, label, ha="center", transform=axis.transAxes, fontsize=16)
    legend = figure.legend(
        [Line2D([], [], color=grey, lw=2.6), Line2D([], [], color=blue, lw=3.0)],
        ["generating", "recovered"],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=14,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    texts = [
        *figure.texts,
        *legend.get_texts(),
        *(text for axis in axes for text in axis.texts),
    ]
    boxes = [(text.get_text(), text.get_window_extent(renderer)) for text in texts]
    for label, box in boxes:
        if not figure.bbox.contains(box.x0, box.y0) or not figure.bbox.contains(
            box.x1, box.y1
        ):
            raise ValueError(f"figure text outside canvas: {label}")
    for index, (label, box) in enumerate(boxes):
        for other, candidate in boxes[index + 1 :]:
            if box.overlaps(candidate):
                raise ValueError(f"figure text overlap: {label}, {other}")
    destination = directory / f"{name}.svg"
    figure.savefig(destination, format="svg")
    tree = element_tree.parse(destination)
    tree.getroot().set("width", "31.1cm")
    tree.getroot().set("height", "16cm")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    figure.savefig(case["output"] / f"{name}-preview.png", dpi=100)
    plt.close(figure)
    print(f"FIGURE {destination} text_bounds=passed text_overlap=passed", flush=True)


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
