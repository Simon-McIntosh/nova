"""Shared receipt, diagnostic-source and known-state helpers for the tests."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.measurement import Magnetics, SliceMeasurement
from nova.equilibrium.solve_request import ForwardSolveRequest
from nova.equilibrium.source import (
    DomainProfile,
    ForwardSource,
    PolynomialFluxFunction,
)
from nova.jax.config import configure_dtypes


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
            trips=int(history.active_set_iterations),
            finite_residual_entries=int(
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


def _physical_sensor_rows(biot, sensors):
    """Synthesize row-ordered observations through the diagnostic kernel owners."""
    from nova.equilibrium.observation_kernels import (
        virtual_flux_loops,
        virtual_poloidal_probes,
    )

    points = np.column_stack((sensors.r, sensors.z))
    loops = np.asarray(sensors.flux_loop, dtype=bool)
    rows = np.empty(sensors.number)
    loop_points, loop_rows, fractions = sensors.loop_quadrature
    rows[loops] = 0.0
    if len(loop_points):
        np.add.at(
            rows,
            loop_rows,
            virtual_flux_loops(biot, loop_points).values * fractions,
        )
    angle = np.deg2rad(sensors.angle[~loops])
    direction = np.column_stack((np.cos(angle), np.sin(angle)))
    rows[~loops] = virtual_poloidal_probes(biot, points[~loops], direction).values
    return rows


def _physical_problem(name):
    """Load fixture-owned machine operands and declare a polynomial source."""
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.forward_operator import ForwardFluxOperator
    from nova.equilibrium.solve_request import ColdSeedPortfolio, ExplicitSolveSeed

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
        coordinates = np.vstack(
            (machine.node, machine.wall_node, machine.sample_coordinates)
        )
        physical, _, _ = lane.oracle_fixture.cached_fixture_exterior(
            carrier,
            exact,
            machine,
            profile.operator,
            lane._exact_state(kind, exact, coordinates),
        )
        known = dict(
            analytic=exact, reference_current=np.asarray(physical.cell_current)
        )
        return (
            profile,
            request,
            machine.node,
            machine.wall_node,
            machine.cell_polygons,
            known,
        )

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
        group = context["group"]
        circuit = np.asarray(group["fcoil_circ"], dtype=int) - 1
        element = {
            key: np.asarray(group["fcoil_" + key])
            for key in ("r", "z", "width", "height", "ang1", "ang2", "turns", "xmult")
        }
        known = dict(
            conductors=tuple(
                (
                    (
                        lane.shaped_section_vertices(
                            *(
                                element[key][index]
                                for key in ("r", "z", "width", "height", "ang1", "ang2")
                            )
                        ),
                    ),
                    float(element["turns"][index] * element["xmult"][index]),
                    float(prescribed[circuit[index]]),
                )
                for index in range(len(circuit))
            )
        )
        plasma_current = float(case["reference"]["plasma_current_a"])
    else:
        from tests import test_equilibrium_forward_reference as lane

        case = lane.reference_case()
        if isinstance(case, str):
            raise RuntimeError(case)
        machine = lane.cached_machine(case, -500, passive=True)
        sample_nodes = machine.moment_geometry.sample_node_coordinates
        if len(machine.sample_coordinates) != len(sample_nodes):
            from scipy.spatial import cKDTree

            with pytest.raises(ValueError, match="direct sample target rows"):
                lane.forward_profile(case, machine)
            distance, rows = cKDTree(machine.sample_coordinates).query(sample_nodes)
            roundoff = (
                128.0
                * np.finfo(float).eps
                * max(float(np.max(np.abs(sample_nodes))), 1.0)
            )
            assert np.max(distance) <= roundoff
            assert len(np.unique(rows)) == len(sample_nodes)
            print(
                f"CACHED_SAMPLE_ALIGNMENT guard_refused=True "
                f"cached_rows={len(machine.sample_coordinates)} "
                f"required_rows={len(sample_nodes)} "
                f"max_coordinate_distance={np.max(distance):.17g}",
                flush=True,
            )
            machine = replace(
                machine,
                sample_coordinates=np.asarray(sample_nodes),
                **{
                    field: np.asarray(getattr(machine, field))[rows]
                    for field in (
                        "source_to_sample",
                        "plasma_to_sample",
                        "plasma_to_sample_r",
                        "plasma_to_sample_z",
                    )
                },
            )
        profile = lane.forward_profile(case, machine)
        seed = lane.seed_flux(case, machine)
        coordinate, wall, sections = (
            machine.node,
            machine.wall_node,
            machine.cell_polygons,
        )
        prescribed = None
        known = dict(
            conductors=tuple(
                (item.placement, item.turns, item.current)
                for item in case.drive(passive=True)
            )
        )
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
        seed_policy=(
            ColdSeedPortfolio(plasma_current, centroid, requested_class="limited")
            if name == "mast"
            else ExplicitSolveSeed(seed)
        ),
        prescribed_current=prescribed,
        vertical_pin="omit",
        policy_overrides=dict(kernel_tolerance=1e-12, qualification_tolerance=1e-12),
    )
    return profile, request, coordinate, wall, sections, known


def _known_field_reader(known, source):
    """Bind prescribed currents or the analytic fixture exterior independently."""
    if "conductors" in known:
        from nova.frame.coilset import CoilSet

        biot = CoilSet(dcoil=-1, field_attrs=["Psi", "Br", "Bz"])
        for index, (placement, turns, current) in enumerate(known["conductors"]):
            biot.coil.insert(*placement, nturn=turns, Ic=current, name=f"known_{index}")
        for name in ("point", "probe", "poloidal_flux_loop"):
            _PlasmaDiagnosticSource._cache_factory(getattr(biot, name))
        assert np.any(np.asarray(biot.sloc["Ic"]) != 0.0)

        def read(sensors):
            rows = _physical_sensor_rows(biot, sensors)
            current = np.asarray(biot.sloc["Ic"]).copy()
            biot.sloc["Ic"] = np.zeros_like(current)
            try:
                np.testing.assert_array_equal(_physical_sensor_rows(biot, sensors), 0.0)
            finally:
                biot.sloc["Ic"] = current
            return rows
    else:
        from nova.equilibrium.convention import TOTAL_FLUX_FACTOR

        def read(sensors):
            exact = known["analytic"]
            radial, vertical = exact.flux_gradient(sensors.r, sensors.z)
            angle = np.deg2rad(sensors.angle)
            analytic = np.where(
                sensors.flux_loop,
                TOTAL_FLUX_FACTOR * exact.flux(sensors.r, sensors.z),
                (-vertical * np.cos(angle) + radial * np.sin(angle)) / sensors.r,
            )
            source.biot.sloc["Ic"] = known["reference_current"]
            return analytic - _physical_sensor_rows(source.biot, sensors)

    return read


def _total_observations(source, generated, sensors, known_field, scales=None):
    """Measure total and known fields without calling the inverse evaluator."""
    plasma = _physical_sensor_rows(source(generated), sensors)
    vacuum = known_field(sensors)
    total = plasma + vacuum
    if scales is None:
        loops = sensors.flux_loop
        scales = np.where(
            loops,
            max(np.max(np.abs(plasma[loops])), 1e-12),
            max(np.max(np.abs(plasma[~loops])), 1e-12),
        )
    assert np.max(np.abs(vacuum[sensors.flux_loop] / scales[sensors.flux_loop])) > 1e-8
    assert (
        np.max(np.abs(vacuum[~sensors.flux_loop] / scales[~sensors.flux_loop])) > 1e-8
    )
    np.testing.assert_allclose(
        (total - vacuum) / scales, plasma / scales, rtol=0, atol=1e-12
    )
    return total, vacuum, plasma, scales


def physical_case_table(tmp_path_factory):
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
        import json

        if name in cases:
            return cases[name]
        profile, request, coordinate, wall, sections, known = _physical_problem(name)
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
        if name != "solovev":
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
        known_field = _known_field_reader(known, source)
        total, vacuum, plasma, scales = _total_observations(
            source, generated, sensors, known_field
        )
        shifted_radius = sensors.r.copy()
        shifted_radius[0] += 0.05
        shifted = replace(sensors, r=shifted_radius)
        shifted_total, shifted_vacuum, shifted_plasma, _ = _total_observations(
            source, generated, shifted, known_field, scales
        )
        geometry_delta = float(np.max(np.abs((shifted_vacuum - vacuum) / scales)))
        assert geometry_delta > 1e-12, geometry_delta
        subtraction = dict(
            case=name,
            conductor_count=len(known.get("conductors", ())),
            exterior_kind="stored_conductors"
            if "conductors" in known
            else "analytic_fixture",
            nominal_vacuum=vacuum.tolist(),
            displaced_vacuum=shifted_vacuum.tolist(),
            nominal_total=total.tolist(),
            displaced_total=shifted_total.tolist(),
            nominal_subtraction_error=float(
                np.max(np.abs((total - vacuum - plasma) / scales))
            ),
            displaced_subtraction_error=float(
                np.max(
                    np.abs((shifted_total - shifted_vacuum - shifted_plasma) / scales)
                )
            ),
            conductor_geometry_delta=geometry_delta,
        )
        (output / f"{name}-conductor-subtraction.json").write_text(
            json.dumps(subtraction, indent=2)
        )
        print(
            "CONDUCTOR_SUBTRACTION "
            + json.dumps(
                {
                    key: value
                    for key, value in subtraction.items()
                    if not key.endswith(("vacuum", "total"))
                }
            ),
            flush=True,
        )
        raw = np.r_[
            request.source_profile.core.p_prime.coefficients,
            request.source_profile.core.ff_prime.coefficients,
        ]
        measurement = SliceMeasurement(
            measured=total,
            vacuum=vacuum,
            mask=np.ones(48, bool),
            scale=scales,
            plasma_current=float(np.sum(generated.terminal_state.cell_current)),
        )
        pressure = request.source_profile.core.p_prime.coefficients.size
        basis = np.zeros((raw.size, 2))
        basis[:pressure, 0] = raw[:pressure]
        basis[pressure:, 1] = raw[pressure:]
        perturbation = 0.02 if name == "solovev" else 2e-4
        initial_source = replace(
            request.source_profile,
            core=replace(
                request.source_profile.core,
                p_prime=replace(
                    request.source_profile.core.p_prime,
                    coefficients=jnp.asarray(raw[:pressure] * (1.0 - perturbation)),
                ),
                ff_prime=replace(
                    request.source_profile.core.ff_prime,
                    coefficients=jnp.asarray(raw[pressure:] * (1.0 + perturbation)),
                ),
            ),
        )
        initial = replace(request, source_profile=initial_source)
        if name != "solovev":
            from nova.equilibrium.solve_request import ExplicitSolveSeed

            initial = replace(
                initial, seed_policy=ExplicitSolveSeed(generated.terminal_state.flux)
            )
        cases[name] = dict(
            name=name,
            profile=profile,
            request=request,
            initial=initial,
            generated=generated,
            coordinate=np.asarray(coordinate),
            wall=np.asarray(wall),
            source=source,
            known_field=known_field,
            sensors=sensors,
            measurement=measurement,
            basis=basis,
            output=output,
        )
        return cases[name]

    return get


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
    # Both maps share one panel and one level array: the generating map is the
    # wide line underneath and the recovered map the narrow line on top, so
    # agreement reads as a narrow line centred in a wide one and any
    # disagreement as the two lines parting.
    wide, narrow = 3.6, 1.1
    original_style = DEFAULT_INK.variant(
        axis_color=grey, xpoint_color=grey, axis_markersize=15.0, xpoint_markersize=16.0
    )
    recovered_style = DEFAULT_INK.variant(
        axis_color=blue, xpoint_color=blue, axis_markersize=7.0, xpoint_markersize=8.0
    )
    figure, axis = plt.subplots(1, 1, figsize=(14.0, 7.2), dpi=100)
    figure.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.90)
    coordinate = case["coordinate"]
    poloidal_axes(axis)
    for state, color, linewidth, landmark_style in (
        (generated, grey, wide, original_style),
        (recovered, blue, narrow, recovered_style),
    ):
        poloidal.draw_scattered_contours(
            axis,
            coordinate[:, 0],
            coordinate[:, 1],
            np.asarray(state.flux)[: len(coordinate)],
            levels,
            case["wall"],
            color=color,
            linewidth=linewidth,
        )
        landmarks = state.topology
        if landmarks is None:
            # a state that lost its axis has no nulls to draw; the contours
            # still show where it went, and the caption says the nulls are
            # absent rather than inheriting the other state's markers
            print(
                "PANEL_NULLS_UNAVAILABLE",
                name,
                color,
                flush=True,
            )
            continue
        diverted = bool(landmarks.diverted)
        poloidal.draw_nulls(
            axis,
            magnetic_axis=np.asarray(landmarks.axis),
            x_points=np.asarray(landmarks.x_point) if diverted else None,
            other_x_points=np.asarray(landmarks.x_point) if not diverted else None,
            contain=case["wall"],
            style=landmark_style,
        )
    poloidal.draw_wall(axis, units=case["wall"])
    legend = figure.legend(
        [Line2D([], [], color=grey, lw=wide), Line2D([], [], color=blue, lw=narrow)],
        ["generating", "recovered"],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=14,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    texts = [*figure.texts, *legend.get_texts(), *axis.texts]
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
