"""Shared magnetic rows through the public forward seam, with prior context."""

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
from nova.equilibrium.convention import SIGMA_BP
from nova.equilibrium.measurement import Magnetics, SliceMeasurement
from nova.equilibrium.solve_request import (
    ColdSeedPortfolio,
    ExplicitSolveSeed,
    ForwardSolveReceipt,
    ForwardSolveRequest,
)
from nova.equilibrium.source import DomainProfile, ForwardSource, PolynomialFluxFunction
from nova.imas.dataset import open_db_entry
from nova.jax.config import configure_dtypes

INPUT_ROOT = Path("/home/ITER/mcintos/Code/imas-efit/tests/data/imas")
CASES = {
    "mastu": ("MASTU/45272", "magnetics", "4.1.0", 0, 2, 2),
    "tcv": ("TCV/workshop", "magnetics", "4.0.0", 2, 0, 2),
    "iter": ("ITER/135013", "standalone", "4.1.0", 0, 2, 3),
}


@pytest.fixture(scope="module", autouse=True)
def _record_comparison_source():
    """Bind a collected comparison module to its source bytes and revision."""
    import hashlib
    import subprocess

    path = Path(__file__).resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path.parents[1], text=True
    ).strip()
    print(
        "COMPARISON_SOURCE "
        + json.dumps(
            dict(
                revision_at_collection=revision,
                module=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        ),
        flush=True,
    )


def _constraint_context(name):
    different = name == "iter"
    label = (
        "same magnetics rows; different constraint sets"
        if different
        else "same magnetics rows; magnetics-only constraint set"
    )
    caveat = (
        "Live counts follow the current selector code. Stored baselines may "
        "predate ITER priors added after 2026-06-23; per-slice live counts "
        "await imas-efit confirmation."
    )
    reference_priors = ["synthesised pressure prior", "q0 prior"] if different else []
    pressure_ndeg, field_ndeg = CASES[name][-2:]
    caption = (
        f"{label}. Bernstein pp/FF-prime ndeg={pressure_ndeg}/{field_ndeg}, "
        f"live={max(pressure_ndeg - 1, 0)}/{max(field_ndeg - 1, 0)}"
    )
    if different:
        caption += (
            ". Under the generating-configuration assumption, imas-efit ITER "
            "carried a synthesised pressure prior (beta_pol=0.5) and a q0 prior; "
            "nova's magnetics fit carries neither. " + caveat
        )
    return dict(
        comparison_label=label,
        reference_extra_priors=reference_priors,
        nova_extra_priors=[],
        caption=caption,
        counts_caveat=caveat,
        pressure_ndeg=pressure_ndeg,
        field_ndeg=field_ndeg,
    )


def _output():
    path = Path(os.environ["NOVA_EFIT_COMPARISON_OUTPUT"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def _residual_fields(value):
    """Represent a refused numerical residual without emitting invalid JSON."""
    if value is None:
        return dict(residual=None, residual_state="unavailable")
    value = float(value)
    if math.isfinite(value):
        return dict(residual=value, residual_state="finite")
    state = (
        "nan"
        if math.isnan(value)
        else "positive_infinity"
        if value > 0
        else "negative_infinity"
    )
    return dict(residual=None, residual_state=state)


def _passive_signed_turns(value):
    """Preserve explicit zero turns as a zero response weight."""
    turns = float(value)
    if not np.isfinite(turns) or abs(turns) > 1e20:
        raise ValueError("invalid passive signed turns")
    return turns


def _circuit_groups(active, *, join_chains):
    """Read signed shared-node pairs and append coils without shared nodes."""
    supply_count, coil_count = len(active.supply), len(active.coil)
    count = supply_count + coil_count
    pairs = []
    for circuit in active.circuit:
        for raw in np.asarray(circuit.connections, dtype=int):
            row = np.where(np.abs(raw) > 1, 0, raw)
            if len(row) == 2 * count:
                row = row[::2] - row[1::2]
            indices = np.flatnonzero(row[:count])
            if len(indices) in (1, 2) and np.all(indices >= supply_count):
                pairs.append([int((i - supply_count + 1) * row[i]) for i in indices])
    groups, done = [], set()
    for index, pair in enumerate(pairs):
        if index in done:
            continue
        group = [-pair[0]]
        done.add(index)
        if len(pair) == 2:
            tail = pair[1]
            if join_chains:
                for other in range(index + 1, len(pairs)):
                    if other not in done and pairs[other][0] == -tail:
                        group.append(-pairs[other][0])
                        tail = pairs[other][1]
                        done.add(other)
            group.append(tail)
        groups.append(group)
    included = {abs(item) for pair in pairs for item in pair}
    groups.extend([i] for i in range(1, coil_count + 1) if i not in included)
    return groups


def _fitted_current_binding(data):
    """Bind the stored flattened circuit vector using its measured-row witness."""
    active, passive = data["input_ids"]["pf_active"], data["input_ids"]["pf_passive"]
    time = data["case"].time
    ts = data["reference_eq"].time_slice[data["reference_slice"]]
    rows = [
        (f"pf_current[{i}]", row) for i, row in enumerate(ts.constraints.pf_current)
    ] + [
        (f"pf_passive_current[{i}]", row)
        for i, row in enumerate(ts.constraints.pf_passive_current)
    ]
    passive_sources = [
        loop
        for loop in passive.loop
        if _sample_passive(loop, passive, time) is not None
        and any(int(e.geometry.geometry_type) in (2, 3, 5, 6) for e in loop.element)
    ]
    candidates = {}
    for join in (False, True):
        groups = [
            [
                i
                for i in group
                if _sample(active.coil[abs(i) - 1].current, active, time) is not None
            ]
            for group in _circuit_groups(active, join_chains=join)
        ]
        groups = [group for group in groups if group]
        measured = [
            _sample(active.coil[abs(group[0]) - 1].current, active, time)
            for group in groups
        ] + [_sample_passive(loop, passive, time) for loop in passive_sources]
        witness = np.array([float(row.measured) for _, row in rows])
        if len(measured) == len(rows) and np.allclose(
            measured, witness, rtol=1e-10, atol=1e-6
        ):
            candidates[tuple(tuple(g) for g in groups)] = (groups, measured)
    if len(candidates) != 1:
        raise ValueError(
            "stored circuit identities lack a unique measured-row witness: "
            f"{len(candidates)}"
        )
    groups, measured = next(iter(candidates.values()))
    effective, mapping = {}, []
    for index, ((path, row), expected) in enumerate(zip(rows, measured, strict=True)):
        fitted = float(row.reconstructed)
        if not np.isfinite(fitted) or abs(fitted) > 1e30:
            raise ValueError(f"missing fitted conductor current: {path}")
        if index < len(groups):
            group = groups[index]
            members = [
                (str(active.coil[abs(i) - 1].name), int(np.sign(i) * np.sign(group[0])))
                for i in group
            ]
        else:
            members = [("passive:" + str(passive_sources[index - len(groups)].name), 1)]
        for name, polarity in members:
            effective[name] = effective.get(name, 0.0) + polarity * fitted
        mapping.append(
            dict(
                output=path,
                source_index=index,
                source_kind="active" if index < len(groups) else "passive",
                members=[dict(name=n, polarity=p) for n, p in members],
                measured_a=float(row.measured),
                input_witness_a=float(expected),
                fitted_a=fitted,
                weight=float(row.weight),
            )
        )
    fitted_columns = np.array([effective[name] for name in data["conductor_names"]])
    element_current = data["projection"] @ fitted_columns
    case = data["case"]
    items = [
        replace(item, current=float(value))
        for item, value in zip(
            (*case.active, *case.passive), element_current, strict=True
        )
    ]
    data["case"] = replace(
        case,
        active=tuple(items[: len(case.active)]),
        passive=tuple(items[len(case.active) :]),
    )
    data["fitted_conductor_current"] = fitted_columns
    data["fitted_current_map"] = mapping
    _write(
        _output() / f"{data['name']}-fitted-current-map.json",
        dict(
            active_groups=len(groups),
            sampled_passive_sources=len(passive_sources),
            stored_active_entries=len(ts.constraints.pf_current),
            stored_passive_entries=len(ts.constraints.pf_passive_current),
            measured_witness_max_error_a=float(
                np.max(np.abs(np.array(measured) - witness))
            ),
            mapping=mapping,
            effective_current_a=effective,
            rule=(
                "Flatten both output constraint arrays before binding filtered "
                "source order; sum every overlapping circuit contribution "
                "at each physical coil."
            ),
        ),
    )
    return data


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


def _flux_loop_segments(channel):
    """Integrate declared toroidal legs of an axisymmetric diagnostic path."""
    if not len(channel.position):
        raise ValueError("a differential flux loop needs its declared references")
    points = np.array(
        [[float(p.r), float(p.z), float(p.phi)] for p in channel.position]
    )
    if not np.all(np.isfinite(points)):
        raise ValueError("flux-loop path coordinates must be finite")
    if len(points) == 1:
        return np.array([[*points[0, :2], float(SIGMA_BP)]])
    fraction = SIGMA_BP * np.diff(points[:, 2]) / (2.0 * np.pi)
    live = fraction != 0.0
    if np.any(np.abs(np.diff(points[:, :2], axis=0)[live]) > 1e-12):
        raise ValueError(
            "a slanted toroidal leg needs distributed segment observations"
        )
    return np.column_stack((points[1:, :2][live], fraction[live]))


def _sample_passive(loop, ids, time, field="current"):
    """Read passive-loop primitive currents on their declared time base."""
    values = np.asarray(getattr(loop, field), dtype=float)
    clock = np.asarray(
        ids.time if int(ids.ids_properties.homogeneous_time) == 1 else loop.time
    )
    if values.size == 0:
        return None
    if len(clock) != len(values):
        raise ValueError("passive current and IDS time lengths differ")
    return float(np.interp(time, clock, values))


def _omit_empty_unconnected_coil(active, index, deselect_when_no_data):
    """Mirror explicit circuit deselection without losing a wired source."""
    if not deselect_when_no_data or len(active.coil[index].current.data):
        return False
    column = len(active.supply) + index
    for circuit in active.circuit:
        matrix = np.asarray(circuit.connections)
        if not matrix.size:
            continue
        if matrix.ndim != 2 or matrix.shape[1] <= column:
            raise ValueError("cannot verify the missing coil's circuit connections")
        if np.any(matrix[:, column] != 0):
            return False
    return True


def _deselect_missing_current(path):
    """Read the circuit directive from an EFIT XML fragment collection."""
    import xml.etree.ElementTree as element_tree

    if not path.exists():
        return False
    content = path.read_text().lstrip()
    if content.startswith("<?xml"):
        content = content.partition("?>")[2]
    root = element_tree.fromstring("<configuration>" + content + "</configuration>")
    directive = next(root.iter("efitCircuits"), None)
    return directive is not None and directive.get("deselectWhenNoData") == "true"


def _profile_basis(count):
    """Map the live Bernstein functions, with their endpoint dropped, to powers."""
    if count == 0:
        return np.zeros((1, 0))
    degree = count - 1
    basis = np.zeros((count, count - 1))
    for k in range(count - 1):
        for j in range(degree - k + 1):
            basis[k + j, k] = (
                math.comb(degree, k) * math.comb(degree - k, j) * (-1.0) ** j
            )
    return basis


def _profile_audit(x, values, count):
    """Check sampled derivatives against the configured live function space."""
    values = np.asarray(values, dtype=float)
    scale = float(np.max(np.abs(values))) or 1.0
    basis = _profile_basis(count)
    design = np.polynomial.polynomial.polyvander(x, basis.shape[0] - 1) @ basis
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    error = float(np.max(np.abs(design @ coefficients - values)) / scale)
    errors = [
        float(
            np.max(
                np.abs(
                    np.polynomial.polynomial.polyval(
                        x, np.polynomial.polynomial.polyfit(x, values, degree)
                    )
                    - values
                )
            )
            / scale
        )
        for degree in range(5)
    ]
    return dict(
        ndeg=count,
        live_coefficients=basis.shape[1],
        identically_zero=bool(np.all(values == 0.0)),
        lowest_polynomial_degree=next(
            (degree for degree, residual in enumerate(errors) if residual < 1e-10),
            None,
        ),
        polynomial_relative_sup_errors=errors,
        configured_relative_sup_error=error,
        representable=error < 1e-10,
    )


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
            for key in ("magnetics", "pf_active", "pf_passive", "wall")
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
    flux_1d = np.asarray(p.psi)
    x = (flux_1d - flux_1d[0]) / (flux_1d[-1] - flux_1d[0])
    profile_audit = {
        "pressure": _profile_audit(x, p.dpressure_dpsi, pressure_count),
        "field": _profile_audit(x, p.f_df_dpsi, field_count),
    }
    _write(_output() / f"{name}-profile-audit.json", profile_audit)
    if not all(item["representable"] for item in profile_audit.values()):
        _write(
            _output() / f"{name}-result.json",
            dict(
                case=name,
                status="configuration-disagreement",
                profiles=profile_audit,
                constraint_context=_constraint_context(name),
            ),
        )
        pytest.fail(
            f"{name}: configured live basis disagrees with stored profiles: "
            f"{profile_audit}"
        )
    span = float(g.psi_boundary) - float(g.psi_axis)
    ip = float(g.ip)
    if not (np.isfinite(span) and span * ip < 0):
        raise ValueError(
            f"convention inconsistency: Ip={ip}, boundary_minus_axis={span}"
        )
    m = ids["magnetics"]
    sensors, values, scales, names, reference_prediction, weights, fractions = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    loop_segments = []
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
                position = channel.position[0]
                terms = _flux_loop_segments(channel)
                loop_segments.append(terms)
                fraction = SIGMA_BP if np.any(terms[:, 2] != 0) else 0.0
                angle = 0.0
            else:
                position = channel.position
                angle = _probe_angle(float(channel.poloidal_angle))
                fraction = 1.0
                loop_segments.append(np.empty((0, 3)))
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
            fractions.append(fraction)
    sensors = np.asarray(sensors)
    geometry = Magnetics(
        r=sensors[:, 0],
        z=sensors[:, 1],
        angle=sensors[:, 2],
        flux_loop=sensors[:, 3].astype(bool),
        flux_loop_fraction=np.asarray(fractions),
        flux_loop_segments=tuple(loop_segments),
    )
    active = ids["pf_active"]
    circuit_config = (
        INPUT_ROOT.parents[2]
        / "configuration/machines"
        / base.split("/")[0]
        / "constraints/magnetics.xml"
    )
    deselect_when_no_data = _deselect_missing_current(circuit_config)
    omitted_conductors = []
    conductors, element_coils, currents, uncertainty, coil_names = [], [], [], [], []
    for coil_index, coil in enumerate(active.coil):
        if _omit_empty_unconnected_coil(active, coil_index, deselect_when_no_data):
            omitted_conductors.append(
                dict(
                    name=str(coil.name),
                    reason="empty current and no circuit connection",
                    configuration=str(circuit_config),
                )
            )
            continue
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
    active_elements = len(conductors)
    passive_ids = ids["pf_passive"]
    for loop_index, loop in enumerate(passive_ids.loop):
        current = _sample_passive(loop, passive_ids, time)
        sigma = _sample_passive(loop, passive_ids, time, "current_error_upper")
        placed = []
        for element_index, element in enumerate(loop.element):
            turns = _passive_signed_turns(element.turns_with_sign)
            if turns == 0:
                continue
            item = owner._conductor(
                f"passive_{loop_index}_{element_index}",
                element.geometry,
                (current or 0.0) * np.sign(turns),
                abs(turns),
            )
            if item is not None:
                placed.append((item, np.sign(turns)))
        if not placed:
            omitted_conductors.append(
                dict(
                    name=str(loop.name), reason="no nonzero-turn placed passive element"
                )
            )
            continue
        if current is None and deselect_when_no_data:
            omitted_conductors.append(
                dict(
                    name=str(loop.name),
                    reason="explicit no-data passive deselection",
                    configuration=str(circuit_config),
                )
            )
            continue
        if (
            current is None
            or not np.isfinite(current)
            or sigma is None
            or not np.isfinite(sigma)
            or sigma <= 0
        ):
            raise ValueError(
                f"placed passive lacks measured current/uncertainty: {str(loop.name)}"
            )
        column = len(currents)
        currents.append(current)
        uncertainty.append(sigma)
        coil_names.append(f"passive:{str(loop.name)}")
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
    surface = ts.profiles_2d[0]
    boundary = np.c_[ts.boundary.outline.r, ts.boundary.outline.z]
    nulls = np.array(
        [
            [
                float(point.position_reconstructed.r),
                float(point.position_reconstructed.z),
            ]
            for point in ts.constraints.x_point
        ]
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
        active=tuple(conductors[:active_elements]),
        passive=tuple(conductors[active_elements:]),
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
    family = "Bernstein"
    pressure_basis, field_basis = (
        _profile_basis(pressure_count),
        _profile_basis(field_count),
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
        profile_audit=profile_audit,
        basis=family + ", edge constrained to zero",
        configuration_assumption=(
            "Committed configuration taken as generating configuration; "
            "stored output does not carry it."
        ),
        constraint_context=_constraint_context(name),
        sensor_count=len(values),
        active_sensor_count=int(np.sum(weights)),
        zero_extent_flux_rows=int(
            np.sum(geometry.flux_loop & (geometry.loop_fraction == 0))
        ),
        active_zero_extent_flux_rows=int(
            np.sum(
                geometry.flux_loop & (geometry.loop_fraction == 0) & measurement.mask
            )
        ),
        conductor_count=len(currents),
        active_element_count=active_elements,
        passive_element_count=len(conductors) - active_elements,
        flux_loop_polarity=SIGMA_BP,
        omitted_conductors=omitted_conductors,
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
        constraint_context=_constraint_context(name),
        input_ids=ids,
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
        machine = owner.cached_machine(case, -300, passive=bool(case.passive))
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
    data["terminal_converged"] = False
    data["terminal_termination_reason"] = None
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
            data["terminal_converged"] = bool(histories[-1].converged)
            data["terminal_termination_reason"] = int(histories[-1].termination_reason)
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
    data["terminal_converged"] = bool(receipt.terminal_state.fixed_point.converged)
    data["terminal_termination_reason"] = int(receipt.termination_reason)
    _write(
        _output() / f"{label}-receipt.json",
        dict(
            qualified=bool(receipt.qualified),
            converged=bool(receipt.terminal_state.fixed_point.converged),
            **_residual_fields(data["terminal_residual"]),
            termination_reason=int(receipt.termination_reason),
            wall_seconds=perf_counter() - started,
            device=str(jax.devices()[0]),
            constraint_context=data["constraint_context"],
        ),
    )
    if "continuation_provenance" in data:
        history = np.asarray(receipt.residual_history)
        finite = history[np.isfinite(history)]
        _write(
            _output() / f"{label}-history.json",
            dict(
                provenance=data["continuation_provenance"],
                resolved_defaults=receipt.resolved_defaults.to_dict(),
                qualified=bool(receipt.qualified),
                converged=data["terminal_converged"],
                **_residual_fields(data["terminal_residual"]),
                residual_history=[
                    float(x) if np.isfinite(x) else None for x in history.ravel()
                ],
                lowest_recorded_residual=float(np.min(finite)) if finite.size else None,
                last_finite_residual=float(finite[-1]) if finite.size else None,
                attempted_newton_promotions=int(receipt.attempted_newton_promotions),
                accepted_newton_promotions=int(receipt.accepted_newton_promotions),
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


def _record_forward_refusal(data):
    """Persist an exhausted public seed portfolio and its terminal panel."""
    result = dict(
        case=data["name"],
        status="forward-failed",
        attempts=data.get("attempts", 3),
        qualified=False,
        converged=bool(data.get("terminal_converged", False)),
        **_residual_fields(data.get("terminal_residual")),
        termination_reason=data.get("terminal_termination_reason"),
        device=str(jax.devices()[0]),
        constraint_context=data["constraint_context"],
    )
    if "recovery" in data:
        result["recovery"] = data["recovery"]
    if "continuation_provenance" in data:
        result["continuation_provenance"] = data["continuation_provenance"]
    _write(_output() / f"{data['name']}-result.json", result)
    _panel(data, qualified=False)
    print("FORWARD_REFUSAL_RECORDED " + json.dumps(result), flush=True)


def _require_replay_agreement(report):
    """Require completed fitted-source evidence before spending public seeds."""
    if (
        report.get("conductor_state") != "stored fitted"
        or report.get("equilibrium_solves") != 0
        or "wall_seconds" not in report
        or report.get("source_cells", 0) <= 0
        or report.get("current_relative_error", math.inf) > 1e-8
    ):
        raise ValueError("physical fit requires a completed fitted-source replay")
    nonzero = 0
    for name, family in report["families"].items():
        score = family["nova_vs_efit"]
        if score["reference_rms"] == 0.0:
            continue
        nonzero += 1
        value = score["relative_l2"]
        if value is None or not math.isfinite(value) or value > 0.05:
            raise ValueError(f"physical fit refused by replay family {name}: {value}")
    if nonzero == 0:
        raise ValueError("physical fit requires a nonzero reference family")


@pytest.mark.slow
@pytest.mark.timeout(27000)
@pytest.mark.parametrize("name", tuple(CASES))
def test_shared_magnetics_reconstruction(name, monkeypatch):
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu", "physical comparisons require a GPU lane"
    data = _read_case(name)
    replay_root = os.environ.get("NOVA_EFIT_FITTED_REPLAY_ROOT")
    if replay_root is not None:
        directory = Path(replay_root) / name
        report = json.loads((directory / f"{name}-row-replay.json").read_text())
        _require_replay_agreement(report)
        if not (directory / "gate.log").read_text().rstrip().endswith("EXIT=0"):
            raise ValueError("physical fit requires the native replay EXIT=0 receipt")
    data = _problem(data, monkeypatch)
    if replay_root is not None:
        data = _fitted_current_binding(data)
        initial = jnp.asarray(data["fitted_conductor_current"])
        data["request"] = replace(data["request"], current=initial)
        data["profile"] = replace(
            data["profile"],
            operator=replace(data["profile"].operator, external_current=initial),
        )
        print(
            "FIT_INITIAL_CURRENTS stored fitted guess; "
            "measured current priors unchanged",
            flush=True,
        )
    continuation_path = os.environ.get("NOVA_EFIT_CONTINUATION_RECEIPT")
    if continuation_path is None:
        forward = _generating(data)
    else:
        import hashlib
        import pickle

        if name != "mastu" or replay_root is None:
            raise ValueError("terminal continuation requires admitted MAST-U inputs")
        payload = Path(continuation_path).read_bytes()
        saved = pickle.loads(payload)
        if not isinstance(saved, ForwardSolveReceipt):
            raise ValueError("terminal continuation needs a public forward receipt")
        policy = saved.resolved_defaults.policy
        if (
            policy.qualification_tolerance
            != data["request"].policy.qualification_tolerance
        ):
            raise ValueError("terminal continuation cannot change qualification")
        policy = replace(
            policy,
            newton_steps=40,
            active_set_steps=64,
            kernel_tolerance=min(policy.kernel_tolerance, 1e-12),
        )
        data["request"] = replace(
            data["request"],
            seed_policy=ExplicitSolveSeed(saved.terminal_state.flux),
            policy=policy,
        )
        data["attempts"] = 1
        data["continuation_provenance"] = dict(
            receipt_sha256=hashlib.sha256(payload).hexdigest(),
            previous_residual=float(saved.terminal_state.fixed_point.residual),
            original_public_seeds=3,
            additional_continuations=1,
            policy=policy.to_dict(),
        )
        print(
            "TERMINAL_CONTINUATION " + json.dumps(data["continuation_provenance"]),
            flush=True,
        )
        forward = _attempt(data, data["request"], f"{name}-continuation")
    if forward is None:
        _record_forward_refusal(data)
        pytest.fail(f"{name}: admitted public forward attempt did not qualify")
    _compare_fit(data, forward)


def _density_quadrature(radius, height, density):
    """Carry the generator's uniform grid-node cell-area current rule."""
    radius, height, density = map(np.asarray, (radius, height, density))
    if density.shape != (len(radius), len(height)):
        raise ValueError("stored current density does not align with its grid")
    dr, dz = np.diff(radius), np.diff(height)
    if not (np.all(dr > 0) and np.all(dz > 0)):
        raise ValueError("stored current grid must be increasing")
    if not (np.allclose(dr, dr[0], rtol=1e-3) and np.allclose(dz, dz[0], rtol=1e-3)):
        raise ValueError("stored current quadrature requires its uniform grid")
    if not np.all(np.isfinite(density)) or not np.any(density != 0):
        raise ValueError("stored current density has no finite nonzero source")
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    keep = density != 0.0
    return rr[keep], zz[keep], density[keep] * dr[0] * dz[0], dr[0], dz[0]


def _row_agreement(prediction, reference, scale):
    """Report physical relative error beside the per-row uncertainty residual."""
    prediction, reference, scale = map(np.asarray, (prediction, reference, scale))
    error = prediction - reference
    norm = float(np.linalg.norm(reference))
    return dict(
        rows=len(reference),
        whitened_rms=float(np.sqrt(np.mean((error / scale) ** 2))),
        whitened_max=float(np.max(np.abs(error / scale))),
        physical_rms=float(np.sqrt(np.mean(error**2))),
        reference_rms=float(np.sqrt(np.mean(reference**2))),
        relative_l2=float(np.linalg.norm(error) / norm) if norm > 0 else None,
        signed_gain=float(prediction @ reference / (reference @ reference))
        if norm > 0
        else None,
        opposite_sign_rows=int(
            np.count_nonzero(
                (prediction * reference < 0)
                & (np.abs(reference) > 0.01 * np.max(np.abs(reference)))
            )
        ),
    )


def _replay_nulls(radius, height, values, case):
    """Read display-map stationary points using the shared spline polish."""
    from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
    from nova.linalg.tensor_spline import fit_tensor_spline
    from matplotlib.path import Path as PolygonPath

    rr, zz = np.meshgrid(radius[::4], height[::4], indexing="ij")
    seeds = np.vstack((case.axis[None], case.x_point, np.c_[rr.ravel(), zz.ravel()]))
    spline = fit_tensor_spline(
        jnp.asarray(radius), jnp.asarray(height), jnp.asarray(values)
    )
    polished = polish_stationary_points(
        spline,
        jnp.asarray(seeds),
        jnp.ones(len(seeds), dtype=bool),
        stationary_steps=24,
    )
    positions, types = (
        np.asarray(polished["position_rz"]),
        np.asarray(polished["hessian_type"]),
    )
    valid = np.asarray(polished["converged"]) & PolygonPath(case.wall).contains_points(
        positions
    )
    accepted, kinds = [], []
    for point, kind in zip(positions[valid], types[valid], strict=True):
        if not any(np.linalg.norm(point - old) < 1e-5 for old in accepted):
            accepted.append(point)
            kinds.append(kind)
    points = np.asarray(accepted).reshape(-1, 2)
    extrema, saddles = points[np.asarray(kinds) == 1], points[np.asarray(kinds) == -1]
    axis = (
        extrema[np.argmin(np.linalg.norm(extrema - case.axis, axis=1))]
        if len(extrema)
        else None
    )
    return dict(magnetic_axis=axis, other_x_points=saddles)


def _probe_response_audit(data, plasma, conductor, total):
    """Contrast probe projection and calibration hypotheses on one field."""
    from nova.equilibrium.observation_kernels import virtual_poloidal_probes

    sensors = data["geometry"]
    indices = np.flatnonzero(~sensors.flux_loop & data["measurement"].mask)
    points = np.c_[sensors.r[indices], sensors.z[indices]]
    fields = []
    for direction in ([1.0, 0.0], [0.0, 1.0]):
        normal = np.tile(direction, (len(indices), 1))
        fields.append(
            sum(
                np.asarray(virtual_poloidal_probes(biot, points, normal).values)
                for biot in (plasma, conductor)
            )
        )
    radial, vertical = fields
    angle = np.deg2rad(sensors.angle[indices])
    reference = data["reference_predictions"][indices]
    sigma = data["measurement"].scale[indices]
    probes = data["input_ids"]["magnetics"].b_field_pol_probe
    loop_count = len(data["input_ids"]["magnetics"].flux_loop)
    parameters = [
        dict(
            name=data["sensor_names"][i],
            angle_rad=float(probes[i - loop_count].poloidal_angle),
            area_m2=float(probes[i - loop_count].area),
            length_m=float(probes[i - loop_count].length),
            turns=int(probes[i - loop_count].turns),
            type_index=int(probes[i - loop_count].type.index),
            type_name=str(probes[i - loop_count].type.name),
        )
        for i in indices
    ]
    variants = dict(
        clockwise=radial * np.cos(angle) + vertical * np.sin(angle),
        counterclockwise=radial * np.cos(angle) - vertical * np.sin(angle),
        perpendicular=-radial * np.sin(angle) + vertical * np.cos(angle),
        area_multiplied=total[indices] * np.array([p["area_m2"] for p in parameters]),
    )
    ranking = np.argsort(-np.abs((total[indices] - reference) / sigma))
    return dict(
        variants={
            name: _row_agreement(value, reference, sigma)
            for name, value in variants.items()
        },
        ranked_rows=[
            dict(
                **parameters[j],
                measured_t=float(data["measurement"].measured[indices[j]]),
                efit_t=float(reference[j]),
                nova_t=float(total[indices[j]]),
                residual_t=float(total[indices[j]] - reference[j]),
                whitened_difference=float(
                    (total[indices[j]] - reference[j]) / sigma[j]
                ),
                variants_t={name: float(value[j]) for name, value in variants.items()},
            )
            for j in ranking
        ],
        response_source=(
            "EFIT responseFunctions.cpp passes R,Z,orientation,length,dLength; "
            "type affects selection/weights; area/turns do not multiply "
            "the Tesla response."
        ),
    )


@pytest.mark.slow
@pytest.mark.timeout(27000)
@pytest.mark.parametrize("name", tuple(CASES))
def test_shared_magnetics_reconstruction_observation_replay(name):
    """Image a stored density with the actual conductor inputs, without solving."""
    from nova.frame.coilset import CoilSet
    from tests.test_equilibrium_magnetics_fit import _physical_sensor_rows

    configure_dtypes()
    assert jax.config.jax_enable_x64
    started = perf_counter()
    data = _read_case(name)
    fitted_currents = os.environ.get("NOVA_EFIT_REPLAY_FITTED_CURRENTS") == "1"
    if fitted_currents:
        data = _fitted_current_binding(data)
    case, sensors, measurement = data["case"], data["geometry"], data["measurement"]
    ts = data["reference_eq"].time_slice[data["reference_slice"]]
    density = np.asarray(ts.profiles_2d[0].j_phi)
    r, z, current, dr, dz = _density_quadrature(
        case.grid_radius, case.grid_height, density
    )
    integrated = float(np.sum(current))
    if integrated * case.plasma_current <= 0:
        raise ValueError(
            f"stored density sign disagrees with Ip: grid={integrated}, "
            f"Ip={case.plasma_current}"
        )
    base, _, dd, *_ = CASES[name]
    with open_db_entry(INPUT_ROOT / base, dd_version=dd, backend="hdf5") as entry:
        passive_ids = entry.get("pf_passive", 0, lazy=False, autoconvert=False)
    passive = [
        dict(
            name=str(loop.name),
            samples=len(loop.current.data),
            current_a=_sample(loop.current, passive_ids, case.time),
        )
        for loop in passive_ids.loop
    ]
    inventory = dict(
        case=name,
        density_shape=list(density.shape),
        nonzero_density_nodes=len(current),
        integrated_current_a=integrated,
        stored_current_a=case.plasma_current,
        input_passive=passive,
        reference_passive=[
            dict(
                measured=float(row.measured),
                reconstructed=float(row.reconstructed),
                weight=float(row.weight),
            )
            for row in ts.constraints.pf_passive_current
        ],
        reference_active=[
            dict(
                measured=float(row.measured),
                reconstructed=float(row.reconstructed),
                weight=float(row.weight),
            )
            for row in ts.constraints.pf_current
        ],
    )
    _write(_output() / f"{name}-row-replay-inputs.json", inventory)
    print("ROW_REPLAY_INPUTS " + json.dumps(inventory), flush=True)
    plasma = CoilSet(dcoil=-1, field_attrs=["Psi", "Br", "Bz"])
    plasma.coil.insert(
        r,
        z,
        np.full(len(r), dr),
        np.full(len(r), dz),
        nturn=1.0,
        Ic=current,
        name=[f"density_{i}" for i in range(len(r))],
    )
    conductor = CoilSet(dcoil=-1, field_attrs=["Psi", "Br", "Bz"])
    for item in (*case.active, *case.passive):
        conductor.coil.insert(
            *item.placement, nturn=item.turns, Ic=item.current, name=item.name
        )
    plasma_rows = _physical_sensor_rows(plasma, sensors)
    vacuum = _physical_sensor_rows(conductor, sensors)
    total = plasma_rows + vacuum
    keep = measurement.mask
    families = {}
    for label, family in (
        ("whole flux loops", sensors.flux_loop & (sensors.loop_fraction != 0)),
        ("zero-extent flux paths", sensors.flux_loop & (sensors.loop_fraction == 0)),
        ("poloidal probes", ~sensors.flux_loop),
    ):
        selected = keep & family
        if not np.any(selected):
            continue

        def agreement(prediction, reference):
            return _row_agreement(
                prediction[selected], reference[selected], measurement.scale[selected]
            )

        families[label] = dict(
            nova_vs_measured=agreement(total, measurement.measured),
            efit_vs_measured=agreement(
                data["reference_predictions"], measurement.measured
            ),
            nova_vs_efit=agreement(total, data["reference_predictions"]),
            vacuum_vs_implied=agreement(
                vacuum, data["reference_predictions"] - plasma_rows
            ),
            vacuum_rms=float(np.sqrt(np.mean(vacuum[selected] ** 2))),
            plasma_rms=float(np.sqrt(np.mean(plasma_rows[selected] ** 2))),
        )
    rows = [
        dict(
            name=row,
            active=bool(keep[i]),
            sigma=float(measurement.scale[i]),
            measured=float(measurement.measured[i]),
            efit=float(data["reference_predictions"][i]),
            plasma=float(plasma_rows[i]),
            vacuum=float(vacuum[i]),
            nova=float(total[i]),
            nova_whitened=float(
                (total[i] - measurement.measured[i]) / measurement.scale[i]
            ),
            efit_whitened=float(
                (data["reference_predictions"][i] - measurement.measured[i])
                / measurement.scale[i]
            ),
        )
        for i, row in enumerate(data["sensor_names"])
    ]
    # Prove the current-bearing path before interpreting an empty family.
    assert np.any(np.abs(plasma_rows[keep & ~sensors.flux_loop]) > 0)
    assert np.any(np.abs(vacuum[keep & ~sensors.flux_loop]) > 0)
    report = dict(
        case=name,
        equilibrium_solves=0,
        integrated_current_a=integrated,
        stored_current_a=case.plasma_current,
        current_relative_error=abs(integrated / case.plasma_current - 1),
        source_cells=len(current),
        source_rule=(
            "stored j_phi at each grid node times uniform generator "
            "cell area; no current rescaling"
        ),
        conductors=list(data["conductor_names"]),
        conductor_current_a=measurement.conductor_current.tolist(),
        conductor_state="stored fitted" if fitted_currents else "measured inputs",
        fitted_conductor_current_a=data.get(
            "fitted_conductor_current", np.array([])
        ).tolist(),
        fitted_current_map=data.get("fitted_current_map", []),
        inventory=inventory,
        families=families,
        rows=rows,
        vacuum_reference=(
            "implied only: EFIT reconstructed total minus nova "
            "stored-density plasma; dependent on plasma quadrature, "
            "not an independent vacuum measurement"
        ),
        constraint_context=data["constraint_context"],
        device=str(jax.devices()[0]),
    )
    if name == "tcv":
        report["probe_audit"] = _probe_response_audit(data, plasma, conductor, total)
    _write(_output() / f"{name}-row-replay.json", report)
    print(
        "ROW_REPLAY " + json.dumps({k: v for k, v in report.items() if k != "rows"}),
        flush=True,
    )
    radius, height = case.grid_radius[::2], case.grid_height[::2]
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    points = np.c_[rr.ravel(), zz.ravel()]
    maps = []
    for biot in (plasma, conductor):
        pieces = []
        for start in range(0, len(points), 128):
            biot.point.solve(points[start : start + 128])
            pieces.append(np.asarray(biot.point.psi).copy())
        maps.append(np.concatenate(pieces).reshape(rr.shape))
    values = maps[0] + maps[1]
    stationary = _replay_nulls(radius, height, values, case)
    _panel(
        data,
        qualified=None,
        flux=values,
        sampled_grid=(radius, height),
        stationary=stationary,
        suffix="-row-replay",
    )
    report.update(
        wall_seconds=perf_counter() - started,
        map_relative_l2=float(
            np.linalg.norm(values - (-case.grid_flux)[::2, ::2])
            / np.linalg.norm(case.grid_flux[::2, ::2])
        ),
        stationary={
            key: np.asarray(value).tolist() if value is not None else None
            for key, value in stationary.items()
        },
    )
    _write(_output() / f"{name}-row-replay.json", report)
    print(
        "ROW_REPLAY_COMPLETE "
        + json.dumps(
            {k: v for k, v in report.items() if k not in ("rows", "families")}
        ),
        flush=True,
    )


def _panel(
    data, qualified, flux=None, *, sampled_grid=None, stationary=None, suffix=""
):
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
    if sampled_grid is None:
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
    else:
        radius, height = sampled_grid
        poloidal.draw_flux_contours(
            ax,
            radius,
            height,
            np.asarray(flux).T,
            levels,
            wall=data["wall_units"],
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
        if stationary is None:
            _, topology = data["profile"].operator.read(jnp.asarray(flux))
            stationary = dict(
                magnetic_axis=np.asarray(topology.axis),
                x_points=np.asarray(topology.x_point)
                if bool(topology.diverted)
                else None,
                other_x_points=np.asarray(topology.x_point)
                if not bool(topology.diverted)
                else None,
            )
        solved_style = DEFAULT_INK.variant(
            axis_color=blue,
            xpoint_color=blue,
            axis_marker="^",
            axis_markersize=6.0,
            xpoint_markersize=7.0,
        )
        poloidal.draw_nulls(
            ax,
            **stationary,
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
        ["imas-efit", "nova current replay" if sampled_grid is not None else "nova"],
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
    destination = directory / f"{data['name']}{suffix}.svg"
    figure.savefig(destination, format="svg")
    tree = element_tree.parse(destination)
    tree.getroot().set("width", "31.1cm")
    tree.getroot().set("height", "16cm")
    description = element_tree.SubElement(
        tree.getroot(), "{http://www.w3.org/2000/svg}desc"
    )
    description.text = data["constraint_context"]["caption"]
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    preview = _output() / f"{data['name']}{suffix}-preview.png"
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
        "constraint_context=" + json.dumps(data["constraint_context"]),
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
    for item in (*case.active, *case.passive):
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
    _persist_forward(scaled, _output() / f"{data['name']}-scaled-fit.pkl")
    scaled_error = abs(
        float(np.sum(scaled.forward.terminal_state.cell_current)) - case.plasma_current
    ) / abs(case.plasma_current)
    result = dict(
        case=data["name"],
        constraint_context=data["constraint_context"],
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
        scaled_fit_converged=bool(scaled.converged),
        scaled_forward_qualified=bool(scaled.forward.qualified),
        scaled_forward_evaluations=scaled.forward_evaluations,
        residual=float(fitted.forward.terminal_state.fixed_point.residual),
        terminal_converged=bool(fitted.forward.terminal_state.fixed_point.converged),
    )
    if "continuation_provenance" in data:
        result["continuation_provenance"] = data["continuation_provenance"]
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


def test_live_bernstein_space_admits_linear_and_zero_profiles_but_refuses_curvature():
    x = np.linspace(0.0, 1.0, 65)
    basis = _profile_basis(2)
    np.testing.assert_allclose(basis[:, 0], [1.0, -1.0])
    linear = _profile_audit(x, 7.0 * (1.0 - x), 2)
    zero = _profile_audit(x, np.zeros_like(x), 2)
    curved = _profile_audit(x, (1.0 - x) ** 2, 2)
    assert linear["representable"] and linear["live_coefficients"] == 1
    assert zero["representable"] and zero["identically_zero"]
    assert not curved["representable"]
    assert curved["lowest_polynomial_degree"] == 2
    assert curved["configured_relative_sup_error"] > 0.1


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
