"""Stored magnetics-only reconstructions through the public forward seam."""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import traceback

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.spatial import cKDTree

from nova.equilibrium import MagneticsFit
from nova.equilibrium.measurement import Magnetics, SliceMeasurement
from nova.equilibrium.solve_request import (
    ColdSeedPortfolio,
    ExplicitSolveSeed,
    ForwardSolveRequest,
)
from nova.equilibrium.source import DomainProfile, ForwardSource, PolynomialFluxFunction
from nova.imas.dataset import open_db_entry
from nova.jax.config import configure_dtypes

INPUT_ROOT = Path("/home/ITER/mcintos/Code/imas-efit/tests/data/imas")
CASES = {
    "mastu": ("MASTU/45272", "magnetics", "4.1.0", 0, 0, 2),
    "tcv": ("TCV/workshop", "magnetics", "4.0.0", 2, 2, 2),
    "iter": ("ITER/135013", "standalone", "4.1.0", 0, 2, 3),
}


def _output():
    path = Path(os.environ["NOVA_EFIT_COMPARISON_OUTPUT"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def _sample(signal, ids, time, field="data"):
    values = np.asarray(getattr(signal, field), dtype=float)
    clock = np.asarray(
        ids.time if int(ids.ids_properties.homogeneous_time) == 1 else signal.time
    )
    if values.size == 0:
        return None
    if len(clock) != len(values):
        raise ValueError(f"time and data length differ for {signal.metadata.path}")
    return float(np.interp(time, clock, values))


def _probe_angle(clockwise_radians):
    """Convert the IMAS clockwise sensor normal to counterclockwise degrees."""
    return -np.rad2deg(clockwise_radians)


def _profile_basis(count, family):
    """Map the live Bernstein functions, with their endpoint dropped, to powers."""
    if count == 0:
        return np.zeros((1, 0))
    if family == "monomial":
        basis = np.vstack((np.eye(count), -np.ones((1, count))))
        return basis
    degree = count - 1
    basis = np.zeros((count, count - 1))
    for k in range(count - 1):
        for j in range(degree - k + 1):
            basis[k + j, k] = (
                math.comb(degree, k) * math.comb(degree - k, j) * (-1.0) ** j
            )
    return basis


def _read_case(name):
    from nova.equilibrium.wall_mask import wall_units_from_ids
    from tests import test_equilibrium_forward_reference as owner

    base, regression, dd, index, pressure_count, field_count = CASES[name]
    source = INPUT_ROOT / base
    with open_db_entry(source, dd_version=dd, backend="hdf5") as entry:
        occurrences = {
            key: [int(x) for x in entry.list_all_occurrences(key)]
            for key in (
                "magnetics",
                "pf_active",
                "pf_passive",
                "wall",
                "iron_core",
                "tf",
            )
        }
        if occurrences["iron_core"]:
            raise ValueError("the selected case carries an iron-core occurrence")
        ids = {
            key: entry.get(key, 0, lazy=False, autoconvert=False)
            for key in ("magnetics", "pf_active", "wall")
        }
    with open_db_entry(
        source / "regression" / regression, dd_version="4.1.0", backend="hdf5"
    ) as entry:
        eq = entry.get("equilibrium", 0, lazy=False, autoconvert=False)
    assert str(eq.ids_properties.version_put.data_dictionary) == "4.1.0"
    assert int(eq.ids_properties.homogeneous_time) == 1
    ts = eq.time_slice[index]
    time = float(eq.time[index])
    assert int(eq.code.output_flag[index]) == 0
    g, p = ts.global_quantities, ts.profiles_1d
    span = float(g.psi_boundary) - float(g.psi_axis)
    ip = float(g.ip)
    if not (np.isfinite(span) and span * ip < 0):
        raise ValueError(
            f"convention inconsistency: Ip={ip}, boundary_minus_axis={span}"
        )
    m = ids["magnetics"]
    sensors, values, scales, names, reference_prediction, weights = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for kind, channels, constraints, quantity in (
        ("loop", m.flux_loop, ts.constraints.flux_loop, "flux"),
        ("probe", m.b_field_pol_probe, ts.constraints.b_field_pol_probe, "field"),
    ):
        assert len(channels) == len(constraints)
        for channel_index, (channel, constraint) in enumerate(
            zip(channels, constraints, strict=True)
        ):
            signal = getattr(channel, quantity)
            measured = float(constraint.measured)
            sigma = _sample(signal, m, time, "data_error_upper")
            weight = float(constraint.weight)
            present = (
                sigma is not None
                and sigma > 0
                and np.isfinite(measured)
                and abs(measured) < 1e30
                and weight > 0
            )
            if kind == "loop":
                if len(channel.position) != 1:
                    raise ValueError(
                        "a comparison flux loop must have one poloidal position"
                    )
                position = channel.position[0]
                angle = 0.0
            else:
                position = channel.position
                angle = _probe_angle(float(channel.poloidal_angle))
            sensors.append(
                (float(position.r), float(position.z), angle, kind == "loop")
            )
            values.append(measured if present else 0.0)
            scales.append(sigma if present else 1.0)
            names.append(f"{kind}:{channel_index}:{str(channel.name)}")
            reference_prediction.append(
                float(constraint.reconstructed) if present else 0.0
            )
            weights.append(present)
    sensors = np.asarray(sensors)
    geometry = Magnetics(
        r=sensors[:, 0],
        z=sensors[:, 1],
        angle=sensors[:, 2],
        flux_loop=sensors[:, 3].astype(bool),
    )
    active = ids["pf_active"]
    conductors, element_coils, currents, uncertainty, coil_names = [], [], [], [], []
    for coil_index, coil in enumerate(active.coil):
        current = _sample(coil.current, active, time)
        sigma = _sample(coil.current, active, time, "data_error_upper")
        placed = []
        for element_index, element in enumerate(coil.element):
            turns = float(element.turns_with_sign)
            if abs(turns) > 1e20 or turns == 0:
                continue
            item = owner._conductor(
                f"coil_{coil_index}_{element_index}",
                element.geometry,
                (current or 0.0) * np.sign(turns),
                abs(turns),
            )
            if item is not None:
                placed.append((item, np.sign(turns)))
        if not placed:
            continue
        if current is None or sigma is None or sigma <= 0:
            raise ValueError(
                f"placed conductor lacks measured current/uncertainty: {str(coil.name)}"
            )
        column = len(currents)
        currents.append(current)
        uncertainty.append(sigma)
        coil_names.append(str(coil.name))
        for item, sign in placed:
            conductors.append(item)
            element_coils.append((column, sign))
    projection = np.zeros((len(conductors), len(currents)))
    for row, (column, sign) in enumerate(element_coils):
        projection[row, column] = sign
    wall_units = wall_units_from_ids(ids["wall"])
    vessels = [unit for unit in wall_units if unit.kind == "vessel"]
    if len(vessels) != 1:
        raise ValueError(f"expected one vessel unit, found {len(vessels)}")
    flux_1d = np.asarray(p.psi)
    x = (flux_1d - flux_1d[0]) / (flux_1d[-1] - flux_1d[0])
    surface = ts.profiles_2d[0]
    boundary = np.c_[ts.boundary.outline.r, ts.boundary.outline.z]
    nulls = np.array(
        [[float(point.r), float(point.z)] for point in ts.boundary_separatrix.x_point]
    ).reshape(-1, 2)
    case = owner.ReferenceCase(
        user=str(source),
        time=time,
        plasma_current=ip,
        poloidal_beta=float(g.beta_pol),
        internal_inductance=float(g.li_3),
        axis=np.array([float(g.magnetic_axis.r), float(g.magnetic_axis.z)]),
        flux_axis=float(g.psi_axis),
        flux_boundary=float(g.psi_boundary),
        reference_radius=float(eq.vacuum_toroidal_field.r0),
        psi_norm=x,
        p_prime=-np.asarray(p.dpressure_dpsi),
        ff_prime=-np.asarray(p.f_df_dpsi),
        pressure=np.asarray(p.pressure),
        field_function=np.asarray(p.f),
        safety_factor=np.asarray(p.q),
        boundary=boundary,
        separatrix=boundary,
        x_point=nulls,
        wall=vessels[0].vertices,
        active=tuple(conductors),
        passive=(),
        unplaced=(),
        grid_radius=np.asarray(surface.grid.dim1),
        grid_height=np.asarray(surface.grid.dim2),
        grid_flux=-np.asarray(surface.psi),
    )
    sigma_ip = _sample(m.ip[0], m, time, "data_error_upper")
    assert sigma_ip is not None and sigma_ip > 0
    measurement = SliceMeasurement(
        measured=np.array(values),
        vacuum=np.zeros(len(values)),
        mask=np.array(weights),
        scale=np.array(scales),
        plasma_current=float(ts.constraints.ip.measured),
        conductor_current=np.array(currents),
        conductor_scale=np.array(uncertainty),
    )
    family = "monomial" if name == "mastu" else "Bernstein"
    pressure_basis, field_basis = (
        _profile_basis(pressure_count, family),
        _profile_basis(field_count, family),
    )
    pressure_raw = np.linalg.lstsq(
        np.polynomial.polynomial.polyvander(x, pressure_basis.shape[0] - 1)
        @ pressure_basis,
        case.p_prime,
        rcond=None,
    )[0]
    field_raw = np.linalg.lstsq(
        np.polynomial.polynomial.polyvander(x, field_basis.shape[0] - 1) @ field_basis,
        case.ff_prime,
        rcond=None,
    )[0]
    basis = block_diag(pressure_basis, field_basis)
    raw = basis @ np.r_[pressure_raw, field_raw]
    split = pressure_basis.shape[0]
    fitted_source = ForwardSource(
        core=DomainProfile(
            p_prime=PolynomialFluxFunction(jnp.asarray(raw[:split])),
            ff_prime=PolynomialFluxFunction(jnp.asarray(raw[split:])),
        ),
        boundary_pressure=float(p.pressure[-1]),
        boundary_field_function=float(p.f[-1]),
    )
    summary = dict(
        case=name,
        input=str(source),
        input_dd=dd,
        reference=str(source / "regression" / regression),
        reference_dd="4.1.0",
        time_s=time,
        pressure_ndeg=pressure_count,
        field_ndeg=field_count,
        free_pressure=pressure_basis.shape[1],
        free_field=field_basis.shape[1],
        basis=family + ", edge constrained to zero",
        configuration_assumption=(
            "Committed configuration taken as generating configuration; "
            "stored output does not carry it."
        ),
        sensor_count=len(values),
        active_sensor_count=int(np.sum(weights)),
        conductor_count=len(currents),
        occurrences=occurrences,
        ip_a=ip,
        flux_span_wb=span,
    )
    _write(_output() / f"{name}-inputs.json", summary)
    print("CASE_INPUT " + json.dumps(summary), flush=True)
    return dict(
        name=name,
        case=case,
        geometry=geometry,
        measurement=measurement,
        plasma_current_scale=sigma_ip,
        source=fitted_source,
        basis=basis,
        projection=projection,
        conductor_names=tuple(coil_names),
        sensor_names=names,
        reference_predictions=np.array(reference_prediction),
        wall_units=wall_units,
        reference_eq=eq,
        reference_slice=index,
    )


def _problem(data, monkeypatch):
    from tests import test_equilibrium_forward_reference as owner
    from nova.equilibrium.forward import ForwardProfile

    case = data["case"]
    with monkeypatch.context() as context:
        context.setattr(owner, "PULSE", data["name"])
        context.setattr(owner, "RUN", "stored-magnetics")
        context.setattr(owner, "DD_VERSION", "4.1.0")
        context.setattr(owner, "TIME_SLICE", data["reference_slice"])
        machine = owner.cached_machine(case, -300, passive=False)
    sample_nodes = machine.moment_geometry.sample_node_coordinates
    distance, rows = cKDTree(machine.sample_coordinates).query(sample_nodes)
    assert np.max(distance) < 1e-12
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
    operator = owner.forward_operator(case, machine)
    projection = data["projection"]
    operator = replace(
        operator,
        source=data["source"],
        polarity=int(np.sign(case.plasma_current)),
        external_current=jnp.asarray(data["measurement"].conductor_current),
        grid=replace(
            operator.grid, source_target=operator.grid.source_target @ projection
        ),
        wall=replace(
            operator.wall, source_target=operator.wall.source_target @ projection
        ),
        sample=replace(
            operator.sample, source_target=operator.sample.source_target @ projection
        ),
    )
    profile = ForwardProfile(operator, owner.receipt_mesh(machine))
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    seed = jnp.asarray(case.flux(coordinates[:, 0], coordinates[:, 1]))
    request = ForwardSolveRequest.from_defaults(
        carrier_identity=f"{data['name']}-shared-magnetics",
        source_profile=data["source"],
        seed_policy=ExplicitSolveSeed(seed),
        current=jnp.asarray(data["measurement"].conductor_current),
        vertical_pin="omit",
    )
    data.update(machine=machine, profile=profile, request=request, seed=seed)
    return data


def _attempt(data, request, label):
    from tests.test_equilibrium_magnetics_fit import (
        _persist_forward,
        _persist_forward_failure,
    )

    started = perf_counter()
    try:
        receipt = data["profile"].solve(request)
    except (ValueError, RuntimeError) as error:
        _persist_forward_failure(error, _output(), label)
        histories = [
            frame.f_locals["history"]
            for frame, _ in traceback.walk_tb(error.__traceback__)
            if "history" in frame.f_locals
            and hasattr(frame.f_locals["history"], "state")
        ]
        if histories:
            data["terminal_flux"] = np.asarray(histories[-1].state)
            data["terminal_residual"] = float(histories[-1].residual)
        print(
            "PUBLIC_FORWARD_REFUSAL",
            label,
            type(error).__name__,
            str(error),
            flush=True,
        )
        return None
    _persist_forward(receipt, _output() / f"{label}-receipt.pkl")
    data["terminal_flux"] = np.asarray(receipt.terminal_state.flux)
    data["terminal_residual"] = float(receipt.terminal_state.fixed_point.residual)
    _write(
        _output() / f"{label}-receipt.json",
        dict(
            qualified=bool(receipt.qualified),
            converged=bool(receipt.terminal_state.fixed_point.converged),
            residual=data["terminal_residual"],
            termination_reason=int(receipt.termination_reason),
            wall_seconds=perf_counter() - started,
            device=str(jax.devices()[0]),
        ),
    )
    return receipt if bool(receipt.qualified) else None


def _generating(data):
    case, original = data["case"], data["request"]
    seeds = [
        original.seed_policy,
        ColdSeedPortfolio(case.plasma_current, tuple(case.axis), "limited"),
        ColdSeedPortfolio(case.plasma_current, tuple(case.axis), "diverted"),
    ]
    for index, seed in enumerate(seeds, 1):
        request = replace(original, seed_policy=seed)
        result = _attempt(data, request, f"{data['name']}-seed-{index}")
        if result is not None:
            return result
    return None


@pytest.mark.slow
@pytest.mark.timeout(7000)
@pytest.mark.parametrize("name", tuple(CASES))
def test_shared_magnetics_reconstruction(name, monkeypatch):
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu", "physical comparisons require the H200 lane"
    data = _problem(_read_case(name), monkeypatch)
    forward = _generating(data)
    if forward is None:
        _write(
            _output() / f"{name}-result.json",
            dict(
                case=name,
                status="forward-failed",
                attempts=3,
                residual=data.get("terminal_residual"),
                device=str(jax.devices()[0]),
            ),
        )
        _panel(data, qualified=False)
        pytest.fail(f"{name}: public forward solve did not qualify in three seeds")
    _compare_fit(data, forward)


def _panel(data, qualified, flux=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import xml.etree.ElementTree as element_tree
    from nova.media import poloidal
    from nova.media.ink import DEFAULT_INK, poloidal_axes

    flux = data.get("terminal_flux") if flux is None else flux
    if flux is None:
        print(
            "PANEL_UNAVAILABLE no terminal flux was returned", data["name"], flush=True
        )
        return
    directory = Path(os.environ["NOVA_EFIT_COMPARISON_FIGURES"])
    directory.mkdir(parents=True, exist_ok=True)
    case = data["case"]
    levels = np.sort(np.linspace(case.flux_axis, case.flux_boundary, 14))[1:-1]
    figure, ax = plt.subplots(figsize=(14.0, 7.2), dpi=100)
    figure.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.88)
    poloidal_axes(ax)
    grey, blue = "#929292", DEFAULT_INK.flux_color
    poloidal.draw_flux_contours(
        ax,
        case.grid_radius,
        case.grid_height,
        -case.grid_flux.T,
        levels,
        wall=data["wall_units"],
        color=grey,
        linewidth=2.6,
        zorder=1,
    )
    node = data["machine"].node
    poloidal.draw_scattered_contours(
        ax,
        node[:, 0],
        node[:, 1],
        np.asarray(flux)[: len(node)],
        levels,
        data["wall_units"],
        color=blue,
        linewidth=1.4,
        zorder=2,
    )
    poloidal.draw_wall(ax, units=data["wall_units"])
    reference_style = DEFAULT_INK.variant(
        axis_color=grey,
        xpoint_color=grey,
        axis_marker="^",
        axis_markersize=10.0,
        xpoint_markersize=11.0,
    )
    poloidal.draw_nulls(
        ax,
        magnetic_axis=case.axis,
        x_points=case.x_point,
        style=reference_style,
        contain=data["wall_units"],
    )
    try:
        _, topology = data["profile"].operator.read(jnp.asarray(flux))
        solved_style = DEFAULT_INK.variant(
            axis_color=blue,
            xpoint_color=blue,
            axis_marker="^",
            axis_markersize=6.0,
            xpoint_markersize=7.0,
        )
        poloidal.draw_nulls(
            ax,
            magnetic_axis=np.asarray(topology.axis),
            x_points=np.asarray(topology.x_point) if bool(topology.diverted) else None,
            other_x_points=np.asarray(topology.x_point)
            if not bool(topology.diverted)
            else None,
            style=solved_style,
            contain=data["wall_units"],
        )
    except ValueError as error:
        print(
            "TERMINAL_NULL_READ_REFUSED", type(error).__name__, str(error), flush=True
        )
    legend = figure.legend(
        [
            Line2D([], [], color=grey, lw=2.6, marker="^", markersize=9),
            Line2D([], [], color=blue, lw=1.4, marker="^", markersize=6),
        ],
        ["imas-efit", "nova"],
        loc="upper center",
        frameon=False,
        ncol=2,
        fontsize=14,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    texts = [*figure.texts, *legend.get_texts(), *ax.texts]
    boxes = [(text.get_text(), text.get_window_extent(renderer)) for text in texts]
    for label, box in boxes:
        if not figure.bbox.contains(box.x0, box.y0) or not figure.bbox.contains(
            box.x1, box.y1
        ):
            raise ValueError(f"figure text outside canvas: {label}")
    for i, (label, box) in enumerate(boxes):
        for other, other_box in boxes[i + 1 :]:
            if box.overlaps(other_box):
                raise ValueError(f"figure text overlap: {label}, {other}")
    destination = directory / f"{data['name']}.svg"
    figure.savefig(destination, format="svg")
    tree = element_tree.parse(destination)
    tree.getroot().set("width", "31.1cm")
    tree.getroot().set("height", "16cm")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    preview = _output() / f"{data['name']}-preview.png"
    figure.savefig(preview, dpi=100)
    plt.close(figure)
    print(
        "FIGURE",
        str(destination),
        "PNG_PREVIEW",
        str(preview),
        "text_bounds=passed text_overlap=passed",
        f"qualified={qualified}",
        f"residual={data.get('terminal_residual')}",
        flush=True,
    )


def _as_equilibrium(data, receipt, source):
    import imas
    import contourpy
    from matplotlib.path import Path as PolygonPath
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from nova.equilibrium.flux_surface_geometry import (
        FluxSurfaceGeometry,
        source_field_function,
    )

    state = receipt.terminal_state
    case, machine = data["case"], data["machine"]
    node = machine.node
    r = np.linspace(case.wall[:, 0].min(), case.wall[:, 0].max(), 129)
    z = np.linspace(case.wall[:, 1].min(), case.wall[:, 1].max(), 193)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    values = np.asarray(state.flux)[: len(node)]
    grid = LinearNDInterpolator(node, values)(rr, zz)
    missing = ~np.isfinite(grid)
    grid[missing] = NearestNDInterpolator(node, values)(rr[missing], zz[missing])
    top = state.topology
    field = source_field_function(source, float(top.flux_span))
    surface = FluxSurfaceGeometry.from_flux_map(
        SimpleNamespace(radius=r, height=z),
        grid.ravel(),
        field,
        axis=tuple(np.asarray(top.axis)),
        boundary_flux=float(top.boundary_flux),
        reference_radius=case.reference_radius,
        surfaces=33,
        angles=96,
        edge_psi_norm=0.97,
    )
    curves = contourpy.contour_generator(x=r, y=z, z=grid.T).lines(
        float(top.boundary_flux)
    )
    closed = [
        curve
        for curve in curves
        if len(curve) > 3
        and np.linalg.norm(curve[0] - curve[-1]) < 1e-8
        and PolygonPath(curve).contains_point(np.asarray(top.axis))
    ]
    if not closed:
        raise ValueError("no closed terminal boundary encloses the magnetic axis")
    boundary = max(closed, key=len)
    result = imas.IDSFactory("4.1.0").new("equilibrium")
    result.ids_properties.homogeneous_time = 1
    result.time = [case.time]
    result.time_slice.resize(1)
    ts = result.time_slice[0]
    g = ts.global_quantities
    g.ip = float(np.sum(state.cell_current))
    g.magnetic_axis.r, g.magnetic_axis.z = map(float, np.asarray(top.axis))
    g.psi_axis, g.psi_boundary = float(top.axis_flux), float(top.boundary_flux)
    ts.boundary.outline.r, ts.boundary.outline.z = boundary.T
    ts.profiles_1d.psi = float(top.axis_flux) + np.asarray(surface.psi_norm) * float(
        top.flux_span
    )
    ts.profiles_1d.q = np.asarray(surface.safety_factor)
    centroid = float(
        np.dot(np.asarray(state.cell_current), node[:, 1]) / np.sum(state.cell_current)
    )
    return result, centroid


def _reference_centroid(data):
    from matplotlib.path import Path as PolygonPath
    from nova.equilibrium.convention import toroidal_current_density

    case = data["case"]
    rr, zz = np.meshgrid(case.grid_radius, case.grid_height, indexing="ij")
    flux = -case.grid_flux
    s = np.clip((flux - case.flux_axis) / case.flux_span, 0.0, 1.0)
    current = np.asarray(
        toroidal_current_density(
            rr,
            np.interp(s, case.psi_norm, case.p_prime),
            np.interp(s, case.psi_norm, case.ff_prime),
        )
    )
    inside = (
        PolygonPath(case.boundary)
        .contains_points(np.c_[rr.ravel(), zz.ravel()])
        .reshape(rr.shape)
    )
    current = np.where(inside, current, 0.0)

    def integral(a):
        return float(
            np.trapezoid(np.trapezoid(a, case.grid_height, axis=1), case.grid_radius)
        )

    measured = integral(current)
    if measured * case.plasma_current <= 0:
        raise ValueError(
            f"stored-profile convention inconsistency: integrated Ip={measured}, "
            f"stored Ip={case.plasma_current}"
        )
    return integral(current * zz) / measured


def _compare_fit(data, forward):
    from tests.test_equilibrium_magnetics_fit import (
        _PlasmaDiagnosticSource,
        _physical_sensor_rows,
        _persist_forward,
    )
    from nova.imas.mast_efit_referee import score_with_efit_referee
    from nova.frame.coilset import CoilSet

    machine, case = data["machine"], data["case"]
    source = _PlasmaDiagnosticSource(machine.cell_polygons)
    conductor = CoilSet(dcoil=-1, field_attrs=["Psi", "Br", "Bz"])
    for item in case.active:
        conductor.coil.insert(*item.placement, nturn=item.turns, Ic=0.0, name=item.name)
    for name in ("point", "probe", "poloidal_flux_loop"):
        _PlasmaDiagnosticSource._cache_factory(getattr(conductor, name))
    columns = []
    for column in data["projection"].T:
        conductor.sloc["Ic"] = column
        columns.append(_physical_sensor_rows(conductor, data["geometry"]))
    response = np.column_stack(columns)
    np.testing.assert_array_equal(np.any(response != 0.0, axis=0), True)
    request = replace(
        data["request"], seed_policy=ExplicitSolveSeed(forward.terminal_state.flux)
    )
    fitter = MagneticsFit(
        data["profile"],
        request,
        data["geometry"],
        source,
        coefficient_basis=data["basis"],
        conductor_response=response,
        conductor_names=data["conductor_names"],
    )
    started = perf_counter()
    fitted = fitter.fit(
        data["measurement"],
        plasma_current_scale=data["plasma_current_scale"],
        tolerance=1e-5,
        max_iterations=16,
    )
    warm_wall = perf_counter() - started
    _persist_forward(fitted, _output() / f"{data['name']}-fit.pkl")
    state, centroid = _as_equilibrium(data, fitted.forward, fitted.source_profile)
    forward_state, forward_centroid = _as_equilibrium(data, forward, data["source"])
    reference_centroid = _reference_centroid(data)
    common = dict(
        reference_slice_index=data["reference_slice"],
        reference_current_centroid_z_m=reference_centroid,
    )
    scores = score_with_efit_referee(
        state, data["reference_eq"], current_centroid_z_m=centroid, **common
    )
    forward_part = score_with_efit_referee(
        forward_state,
        data["reference_eq"],
        current_centroid_z_m=forward_centroid,
        **common,
    )
    fit_part = score_with_efit_referee(
        state,
        forward_state,
        current_centroid_z_m=centroid,
        reference_current_centroid_z_m=forward_centroid,
    )
    m = data["measurement"]
    residuals = [
        dict(
            sensor=name,
            nova=float(value),
            efit=float((prediction - measured) / scale) if keep else 0.0,
            active=bool(keep),
        )
        for name, value, prediction, measured, scale, keep in zip(
            data["sensor_names"],
            fitted.whitened_residuals[: len(m.measured)],
            data["reference_predictions"],
            m.measured,
            m.scale,
            m.mask,
            strict=True,
        )
    ]
    scaled_measurement = replace(
        m,
        measured=m.measured * 1.10,
        scale=m.scale * 1.10,
        plasma_current=m.plasma_current * 1.10,
        conductor_current=m.conductor_current * 1.10,
        conductor_scale=m.conductor_scale * 1.10,
    )
    scaled = fitter.fit(
        scaled_measurement,
        plasma_current_scale=data["plasma_current_scale"] * 1.10,
        tolerance=1e-5,
        max_iterations=16,
    )
    scaled_error = abs(
        float(np.sum(scaled.forward.terminal_state.cell_current)) - case.plasma_current
    ) / abs(case.plasma_current)
    result = dict(
        case=data["name"],
        status="measured",
        fit_converged=bool(fitted.converged),
        scores=scores,
        forward_part=forward_part,
        fit_part=fit_part,
        sensor_residuals=residuals,
        warm_wall_seconds=warm_wall,
        forward_evaluations=fitted.forward_evaluations,
        device=fitted.device,
        scaled_current_relative_error=scaled_error,
        scaled_control_fired=scaled_error > 0.015,
        residual=float(fitted.forward.terminal_state.fixed_point.residual),
        terminal_converged=bool(fitted.forward.terminal_state.fixed_point.converged),
    )
    _write(_output() / f"{data['name']}-result.json", result)
    data["terminal_residual"] = result["residual"]
    _panel(
        data,
        qualified=bool(fitted.forward.qualified),
        flux=fitted.forward.terminal_state.flux,
    )
    assert fitted.converged, "inverse did not reach a stationary fit"
    assert all(scores["bands"].values()), scores
    assert scaled_error > 0.015, (
        "scaled-input control failed to leave plasma-current band"
    )


def test_probe_normal_uses_clockwise_angle_from_outward_radius():
    geometry = Magnetics(
        r=np.ones(3),
        z=np.zeros(3),
        angle=_probe_angle(np.array([0.0, np.pi / 2, np.pi])),
        flux_loop=np.zeros(3, dtype=bool),
    )
    np.testing.assert_allclose(
        geometry.project(np.zeros(3), np.full(3, 2.0), np.full(3, 3.0)),
        [2.0, -3.0, -2.0],
    )


def test_supplied_equilibrium_reports_vertical_offset_and_physical_bands():
    from nova.imas.mast_efit_referee import score_with_efit_referee

    def state(offset, ip):
        boundary = np.array(
            [[1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [1.0, 1.0], [1.0, -1.0]]
        )
        return SimpleNamespace(
            time_slice=[
                SimpleNamespace(
                    global_quantities=SimpleNamespace(
                        ip=ip,
                        psi_axis=2.0,
                        psi_boundary=1.0,
                        magnetic_axis=SimpleNamespace(r=1.5, z=offset),
                    ),
                    boundary=SimpleNamespace(
                        outline=SimpleNamespace(
                            r=boundary[:, 0], z=boundary[:, 1] + offset
                        )
                    ),
                    profiles_1d=SimpleNamespace(
                        psi=np.array([2.0, 1.0]), q=np.array([-1.0, -3.0])
                    ),
                )
            ]
        )

    score = score_with_efit_referee(
        state(0.3, 110.0),
        state(0.0, 100.0),
        current_centroid_z_m=0.3,
        reference_current_centroid_z_m=0.0,
    )
    assert score["magnetic_axis_distance_m"] == pytest.approx(0.3)
    assert score["aligned_magnetic_axis_distance_m"] == pytest.approx(0.0)
    assert score["aligned_boundary_distance_mm"] == pytest.approx(0.0)
    assert score["plasma_current_relative_error"] == pytest.approx(0.1)
    assert score["q95"] == pytest.approx(-2.9)
    assert score["bands"]["magnetic_axis_distance_m"] is False
    assert score["bands"]["plasma_current_relative_error"] is False
    assert score["bands"]["q95_magnitude_error"] is True
