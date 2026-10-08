"""Prescribed-source CORSICA equilibria on their own pinned ITER machine.

The bundle carries total flux in Nova's sense, with a negative-current axis at
its minimum. Nova's source derivatives are with respect to NEGATED total flux,
so both stored derivatives are negated, with no fitted amplitude or gauge shift.
Every solve starts from the vacuum field of the input conductor currents.
Reference flux, axis and boundary values are comparison data only; they never
enter the seed. A missing reference LCFS qualifies only boundary-based comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

# The batch process leaves GPU memory to the separate solver processes.
if __name__ == "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from nova.database.zarrstore import ZarrStore
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.solve_request import (
    ExplicitSolveSeed,
    ForwardSolveRequest,
    SampledFluxFunction,
)
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.imas import validator_case as vc
from nova.imas.machine import Annulus
from nova.jax.config import configure_dtypes
from tests import test_equilibrium_forward_reference as reference

pytestmark = pytest.mark.slow

# Absolute and scale-aware cross-code bands, fixed before scoring.
AXIS_BAND_M = 0.20
CURRENT_FLOOR_A = 200_000.0
CURRENT_FRACTION = 0.015
FLUX_FLOOR_WB = 5.0
FLUX_DEPTH_FRACTION = 0.05
NORMALISED_MAP_BAND = 0.10
LCFS_AREA_BAND_M2 = 2.0
MINOR_RADIUS_FLOOR_M = 0.5
GRAD_SHAFRANOV_BAND = 0.08
DIVERGENCE_MARGIN = 0.2
PLASMA_CELLS = -2100
SLICE_TIMES = (150.0, 200.0, 230.0, 231.0)


def _write_receipt(path, payload):
    temporary = path.with_suffix(path.suffix + ".partial")

    def finite_json(value):
        if isinstance(value, dict):
            return {key: finite_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [finite_json(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    temporary.write_text(json.dumps(finite_json(payload), indent=2, allow_nan=False))
    temporary.replace(path)


def _current_index(ids, time):
    assert int(ids.ids_properties.homogeneous_time) == 1
    matches = np.flatnonzero(np.asarray(ids.time) == time)
    assert len(matches) == 1, (time, np.asarray(ids.time))
    return int(matches[0])


@dataclass(frozen=True)
class MaterialConductor(reference.Conductor):
    """Carry an authored material polygon, including any annular hole."""

    material: object = None

    @property
    def placement(self):
        return (self.material,)

    @property
    def declared_area(self):
        return self.material.area


def _conductor(name, geometry, current, turns):
    section = int(geometry.geometry_type)
    if section in (2, 3):
        return reference._conductor(name, geometry, current, turns)
    if section == 5:
        material = Annulus(geometry).poly
    elif section == 6:
        line = geometry.thick_line
        start = np.array([float(line.first_point.r), float(line.first_point.z)])
        end = np.array([float(line.second_point.r), float(line.second_point.z)])
        delta = end - start
        normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
        offset = 0.5 * float(line.thickness) * normal
        material = Polygon([start + offset, end + offset, end - offset, start - offset])
    else:
        raise ValueError(f"unsupported conductor geometry {section}")
    assert material.is_valid and material.area > 0
    return MaterialConductor(
        name,
        current,
        turns,
        polygon=np.asarray(material.exterior.coords),
        material=material,
    )


def _conductors(ids, time, *, passive=False):
    index = _current_index(ids, time)
    conductors = []
    items = ids.loop if passive else ids.coil
    for item_index, item in enumerate(items):
        values = item.current if passive else item.current.data
        assert len(values) == len(ids.time)
        current = float(np.asarray(values)[index])
        for element_index, element in enumerate(item.element):
            turns = float(element.turns_with_sign)
            conductor = _conductor(
                f"{'passive' if passive else 'active'}_{item_index}_{element_index}",
                element.geometry,
                current * np.sign(turns),
                abs(turns),
            )
            assert conductor is not None, str(item.name)
            conductors.append(conductor)
    return tuple(conductors)


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    try:
        bundle = vc.resolve_validator_case(
            "iter_corsica_130506",
            store_root=(
                vc.default_store_root()
                if os.environ.get(vc.STORE_ROOT_ENVIRONMENT)
                else tmp_path_factory.mktemp("corsica-store")
            ),
        )
    except vc.LayerRegistryUnreachable as error:
        pytest.skip(f"validator registry unreachable: {error}")
    assert bundle.dd_version == "4.1.0"
    with bundle.entry("input") as entry:
        ids = {}
        for name in ("pf_active", "pf_passive", "wall"):
            assert 0 in entry.list_all_occurrences(name), name
            ids[name] = entry.get(name, 0, autoconvert=False)
    with bundle.entry("reference") as entry:
        equilibrium = entry.get("equilibrium", 0, autoconvert=False)
    np.testing.assert_array_equal(np.asarray(equilibrium.time), SLICE_TIMES)
    assert len(equilibrium.time_slice) == 4
    units = ids["wall"].description_2d[0].limiter.unit
    assert len(units) == 1, "machine mesher requires a single closed limiter"
    wall = np.column_stack((units[0].outline.r, units[0].outline.z))
    result = []
    for index, time in enumerate(SLICE_TIMES):
        slice_ = equilibrium.time_slice[index]
        global_ = slice_.global_quantities
        profile = slice_.profiles_1d
        surface = slice_.profiles_2d[0]
        psi = np.asarray(profile.psi)
        axis_flux = float(global_.psi_magnetic_axis)
        boundary_flux = float(global_.psi_boundary)
        assert np.isfinite(axis_flux) and abs(axis_flux) < 1e6
        assert axis_flux < boundary_flux
        result.append(
            reference.ReferenceCase(
                user=bundle.key,
                time=time,
                plasma_current=float(global_.ip),
                poloidal_beta=float(global_.beta_pol),
                internal_inductance=float(global_.li_3),
                axis=np.array(
                    [float(global_.magnetic_axis.r), float(global_.magnetic_axis.z)]
                ),
                flux_axis=axis_flux,
                flux_boundary=boundary_flux,
                reference_radius=float(equilibrium.vacuum_toroidal_field.r0),
                psi_norm=(psi - psi[0]) / (psi[-1] - psi[0]),
                p_prime=-np.asarray(profile.dpressure_dpsi),
                ff_prime=-np.asarray(profile.f_df_dpsi),
                pressure=np.asarray(profile.pressure),
                field_function=np.asarray(profile.f),
                safety_factor=np.asarray(profile.q),
                boundary=np.column_stack(
                    (slice_.boundary.outline.r, slice_.boundary.outline.z)
                ),
                separatrix=np.column_stack(
                    (
                        slice_.boundary.outline.r,
                        slice_.boundary.outline.z,
                    )
                ),
                x_point=np.array(
                    [
                        [float(p.r), float(p.z)]
                        for p in slice_.contour_tree.node
                        if int(p.critical_type) == 1
                    ]
                ),
                wall=wall,
                active=_conductors(ids["pf_active"], time),
                passive=_conductors(ids["pf_passive"], time, passive=True),
                unplaced=(),
                grid_radius=np.asarray(surface.grid.dim1),
                grid_height=np.asarray(surface.grid.dim2),
                # The shared carrier's interpolation accessor negates its stored grid.
                grid_flux=-np.asarray(surface.psi),
            )
        )
    print(
        f"INPUT dd={bundle.dd_version} times={SLICE_TIMES} "
        f"active_elements={len(result[0].active)} "
        f"passive_elements={len(result[0].passive)}",
        flush=True,
    )
    return result


@pytest.fixture(scope="module")
def machine(cases, tmp_path_factory):
    case = cases[0]
    root = Path(
        os.environ.get(
            "NOVA_CORSICA_ARTIFACTS", tmp_path_factory.mktemp("corsica-artifacts")
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    identity = reference.machine_cache_identity(case, PLASMA_CELLS)
    identity["reference_locator"] = vc.load_case_record("iter_corsica_130506")
    identity["material_polygons"] = [
        c.material.wkb_hex if isinstance(c, MaterialConductor) else None
        for c in case.drive()
    ]
    store = ZarrStore(filename="corsica-carrier", dirname=root)
    store.group = store.hash_attrs(identity)
    with reference._machine_cache_lock(store):
        if store.is_store():
            machine = reference._load_cached_machine(store, identity)
            print(f"CARRIER cache=hit key={store.group}", flush=True)
        else:
            print(f"CARRIER cache=miss key={store.group}", flush=True)
            machine = reference.build_machine(case, PLASMA_CELLS)
            store.data = reference._machine_dataset(machine, identity, store.group)
            store.store(mode=store.get_mode())
            restored = reference._load_cached_machine(store, identity)
            reference.assert_machine_arrays_bitwise_identical(machine, restored)
    receipt_root = Path(os.environ.get("NOVA_CORSICA_RECEIPTS", str(root)))
    receipt_root.mkdir(parents=True, exist_ok=True)
    return machine, receipt_root


def _lcfs(machine, flux, axis, boundary_flux):
    """Find the closed flux contour containing the solved magnetic axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    if not np.all(np.isfinite(flux)) or not np.all(np.isfinite(axis)):
        return Polygon()
    if not np.isfinite(boundary_flux) or np.ptp(flux) == 0:
        return Polygon()
    fig, axes = plt.subplots()
    contour = axes.tricontour(
        tri.Triangulation(machine.node[:, 0], machine.node[:, 1]),
        flux,
        levels=[boundary_flux],
    )
    rings = [
        Polygon(line)
        for line in contour.allsegs[0]
        if len(line) >= 4 and np.allclose(line[0], line[-1])
    ]
    plt.close(fig)
    enclosed = [ring for ring in rings if ring.is_valid and ring.contains(Point(axis))]
    return max(enclosed, key=lambda ring: ring.area) if enclosed else Polygon()


def _render(case, machine, equilibrium, metrics, path):
    import matplotlib.pyplot as plt
    from nova.media.ink import DEFAULT_INK, poloidal_axes
    from nova.media.poloidal import (
        draw_flux_contours,
        draw_nulls,
        draw_scattered_contours,
        draw_wall,
    )

    fig, axes = plt.subplots(figsize=(14, 9), dpi=100)
    poloidal_axes(axes)
    levels = np.linspace(case.flux_axis, case.flux_boundary, 10)[1:]
    stored = draw_flux_contours(
        axes,
        case.grid_radius,
        case.grid_height,
        -case.grid_flux.T,
        levels,
        color="#333333",
        linewidth=2.6,
        wall=case.wall,
    )
    stored.set_linestyle("dashed")
    terminal_flux = np.asarray(equilibrium.flux)[: len(machine.node)]
    if np.all(np.isfinite(terminal_flux)) and np.ptp(terminal_flux) > 0:
        draw_scattered_contours(
            axes,
            machine.node[:, 0],
            machine.node[:, 1],
            np.asarray(equilibrium.flux)[: len(machine.node)],
            levels,
            case.wall,
            color=DEFAULT_INK.flux_color,
            linewidth=3.0,
        )
    draw_wall(axes, units=(case.wall,))
    draw_nulls(
        axes,
        case.axis,
        case.x_point,
        style=DEFAULT_INK.variant(axis_color="#333333", xpoint_color="#333333"),
    )
    draw_nulls(
        axes,
        np.asarray(equilibrium.topology.axis),
        np.asarray(equilibrium.topology.x_point).reshape(-1, 2),
        style=DEFAULT_INK.variant(
            axis_color=DEFAULT_INK.flux_color, xpoint_color=DEFAULT_INK.flux_color
        ),
    )
    fig.text(
        0.5,
        0.04,
        f"t = {case.time:g} s; blue: solved, dashed black: CORSICA; "
        "both null sets marked\n"
        f"shared flux levels [Wb]; residual = {metrics['residual']:.3g}; "
        f"converged = {metrics['converged']}",
        ha="center",
        fontsize=20,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


@pytest.fixture(scope="module")
def solved(cases, machine):
    carrier, root = machine
    results = {}

    def solve(index):
        if index in results:
            if isinstance(results[index], BaseException):
                raise results[index]
            return results[index]
        case = cases[index]
        scale = (
            float(os.environ.get("NOVA_CORSICA_FF_SCALE", "1"))
            if case.time == float(os.environ.get("NOVA_CORSICA_CONTROL_TIME", "150"))
            else 1.0
        )
        current = np.array([c.current for c in case.drive()])
        active_carrier = replace(carrier, source_current=current)
        source = ForwardSource(
            core=DomainProfile(
                p_prime=SampledFluxFunction(case.psi_norm, case.p_prime),
                ff_prime=SampledFluxFunction(case.psi_norm, case.ff_prime * scale),
            ),
            boundary_pressure=float(case.pressure[-1]),
            boundary_field_function=float(case.field_function[-1]),
        )
        operator = replace(
            reference.forward_operator(case, active_carrier), source=source
        )
        profile = ForwardProfile(
            operator=operator, lattice=reference.receipt_mesh(active_carrier)
        )
        # Only conductor geometry and exact-time input currents enter this seed.
        seed = operator.external()
        request = ForwardSolveRequest.from_defaults(
            carrier_identity=f"corsica-{case.time:g}",
            source_profile=source,
            seed_policy=ExplicitSolveSeed(seed),
        )
        print(
            f"SOLVE time={case.time:g} FF_scale={scale} "
            f"defaults={request.policy.to_dict()}",
            flush=True,
        )
        receipt_path = root / f"slice-{case.time:g}-ff-{scale:g}.json"
        grid_r, grid_z = np.meshgrid(case.grid_radius, case.grid_height, indexing="ij")
        reference_lcfs = _lcfs(
            SimpleNamespace(node=np.column_stack((grid_r.ravel(), grid_z.ravel()))),
            -case.grid_flux.ravel(),
            case.axis,
            case.flux_boundary,
        )
        reference_closed = not reference_lcfs.is_empty
        metrics = {
            "time": case.time,
            "ff_scale": scale,
            "seed_kind": "input-conductor-vacuum",
            "phase": "solving",
            "completed": False,
            "iterations_taken": None,
            "accepted_steps": None,
            "residual": None,
            "reference_closed_lcfs": reference_closed,
            "reference_consistency_finding": (
                None
                if reference_closed
                else (
                    "no closed axis-containing contour "
                    "at declared reference boundary flux"
                )
            ),
            "seed_reference_difference_wb": float(
                np.max(
                    np.abs(
                        np.asarray(seed)[: len(carrier.node)]
                        - case.flux(carrier.node[:, 0], carrier.node[:, 1])
                    )
                )
            ),
        }
        _write_receipt(receipt_path, metrics)
        print(
            f"MEASUREMENT_MODULE={sys.modules[ForwardProfile.__module__].__file__}",
            flush=True,
        )
        print(f"MEASUREMENT_CWD={Path.cwd().resolve()}", flush=True)
        print(
            f"EXECUTION_BACKEND={jax.default_backend()} DEVICES={jax.devices()}",
            flush=True,
        )
        try:
            receipt = profile.solve(request)
        except BaseException as error:
            metrics.update(phase="solve-error", error=repr(error))
            _write_receipt(receipt_path, metrics)
            results[index] = error
            raise
        eq = receipt.terminal_state
        history = eq.fixed_point
        metrics.update(
            phase="terminal",
            completed=True,
            residual=float(history.residual),
            converged=bool(history.converged),
            iterations_taken=int(history.attempted_newton_promotions),
            iteration_count_definition="attempted Newton promotions",
            accepted_steps=int(history.accepted_newton_promotions),
            active_set_iterations=int(history.active_set_iterations),
            termination_reason=int(history.termination_reason),
            terminal_minus_seed_wb=float(
                np.max(np.abs(np.asarray(eq.flux) - np.asarray(seed)))
            ),
            residual_history=np.asarray(receipt.residual_history).tolist(),
            mask_history=np.asarray(receipt.mask_history).tolist(),
            defaults=receipt.resolved_defaults.to_dict(),
        )
        metrics["solver_finding"] = (
            "unchanged seed with nonzero residual"
            if metrics["terminal_minus_seed_wb"] == 0.0 and metrics["residual"] > 0.0
            else None
        )
        _write_receipt(receipt_path, metrics)
        np.savez_compressed(
            root / f"slice-{case.time:g}-ff-{scale:g}.npz",
            node=carrier.node,
            flux=np.asarray(eq.flux),
            seed=np.asarray(seed),
            current=np.asarray(eq.cell_current),
            axis=np.asarray(eq.topology.axis),
            x_point=np.asarray(eq.topology.x_point),
            axis_flux=np.asarray(eq.topology.axis_flux),
            boundary_flux=np.asarray(eq.topology.boundary_flux),
        )
        grid_flux = np.asarray(eq.flux)[: len(carrier.node)]
        lcfs = _lcfs(
            carrier,
            grid_flux,
            np.asarray(eq.topology.axis),
            float(eq.topology.boundary_flux),
        )
        ref_flux = case.flux(carrier.node[:, 0], carrier.node[:, 1])

        # Positive ranges absorb gauge and magnitude while retaining a sign reversal.
        def norm(values):
            return (values - values.min()) / np.ptp(values)

        map_error = float(np.max(np.abs(norm(grid_flux) - norm(ref_flux))))
        metrics.update(
            {
                "time": case.time,
                "ff_scale": scale,
                "axis_error_m": float(
                    np.linalg.norm(np.asarray(eq.topology.axis) - case.axis)
                ),
                "ip_a": float(eq.moments.plasma_current),
                "reference_ip_a": case.plasma_current,
                "ip_error_a": abs(
                    float(eq.moments.plasma_current) - case.plasma_current
                ),
                "axis_flux_error_wb": abs(
                    float(eq.topology.axis_flux) - case.flux_axis
                ),
                "boundary_flux_error_wb": abs(
                    float(eq.topology.boundary_flux) - case.flux_boundary
                ),
                "map_error": map_error,
                "lcfs_area_m2": lcfs.area,
                "reference_area_m2": Polygon(case.boundary).area,
                "residual": float(eq.fixed_point.residual),
                "converged": bool(eq.fixed_point.converged),
                "qualified": bool(receipt.qualified),
                "wall_seconds": receipt.wall_seconds,
                "q_axis_reference": float(case.safety_factor[0]),
                "beta_p": float(eq.moments.poloidal_beta),
                "beta_p_reference": case.poloidal_beta,
                "li": float(eq.moments.internal_inductance),
                "li_reference": case.internal_inductance,
                "grad_shafranov": float(eq.conservation.relative_grad_shafranov),
                "defaults": receipt.resolved_defaults.to_dict(),
            }
        )
        print("CORSICA " + json.dumps(metrics, sort_keys=True), flush=True)
        _write_receipt(receipt_path, metrics)
        results[index] = case, eq, receipt, lcfs, metrics
        figures = os.environ.get("NOVA_CORSICA_FIGURES")
        if figures and scale == 1.0:
            _render(
                case, carrier, eq, metrics, Path(figures) / f"corsica-{case.time:g}.svg"
            )
        results[index] = case, eq, receipt, lcfs, metrics
        return results[index]

    return solve


@pytest.mark.parametrize(
    "index", range(4), ids=[f"time-{time:g}" for time in SLICE_TIMES]
)
def test_prescribed_profiles_match_axis_and_current(solved, index):
    case, _, _, _, metrics = solved(index)
    assert metrics["axis_error_m"] <= AXIS_BAND_M, metrics
    assert metrics["ip_error_a"] <= max(
        CURRENT_FLOOR_A, CURRENT_FRACTION * abs(case.plasma_current)
    ), metrics


@pytest.mark.parametrize(
    "index", range(4), ids=[f"time-{time:g}" for time in SLICE_TIMES]
)
def test_flux_map_and_absolute_lcfs_area(solved, index):
    case, _, _, _, metrics = solved(index)
    band = max(FLUX_FLOOR_WB, FLUX_DEPTH_FRACTION * abs(case.flux_span))
    assert metrics["axis_flux_error_wb"] <= band, metrics
    assert metrics["map_error"] <= NORMALISED_MAP_BAND, metrics
    if metrics["reference_closed_lcfs"]:
        assert metrics["boundary_flux_error_wb"] <= band, metrics
        assert (
            abs(metrics["lcfs_area_m2"] - metrics["reference_area_m2"])
            <= LCFS_AREA_BAND_M2
        ), metrics


@pytest.mark.parametrize(
    "index", range(4), ids=[f"time-{time:g}" for time in SLICE_TIMES]
)
def test_containment_and_solve_receipts(solved, index):
    case, eq, receipt, lcfs, metrics = solved(index)
    axis = Point(np.asarray(eq.topology.axis))
    assert Polygon(case.wall).contains(axis)
    assert lcfs.contains(axis), metrics
    low_r, low_z, high_r, high_z = lcfs.bounds
    assert lcfs.area >= 0.5 * (high_r - low_r) * (high_z - low_z), metrics
    assert 0.5 * (high_r - low_r) >= MINOR_RADIUS_FLOOR_M
    assert metrics["converged"], metrics
    assert (
        metrics["residual"] <= receipt.resolved_defaults.policy.qualification_tolerance
    ), metrics
    assert bool(eq.finite.passed)
    assert metrics["grad_shafranov"] < GRAD_SHAFRANOV_BAND, metrics
    for name in ("relative_divergence_b", "relative_divergence_j"):
        assert (
            float(getattr(eq.conservation, name))
            < DIVERGENCE_MARGIN * metrics["grad_shafranov"]
        )
    for name in ("common_sol", "private_flux", "excluded_material"):
        assert float(getattr(eq.ledger, name)) == 0.0
    assert eq.normalisation.policy_name == "absolute"
    assert not bool(eq.normalisation.rescaled)
    assert float(eq.normalisation.amplitude) == 1.0


SLICE_CHECKS = (
    "test_prescribed_profiles_match_axis_and_current",
    "test_flux_map_and_absolute_lcfs_area",
    "test_containment_and_solve_receipts",
)


def _slice_process(root, time, scale, revision):
    """Run one bounded pytest process and persist its complete or partial verdict."""
    import shlex
    import tempfile
    import xml.etree.ElementTree as element_tree

    label = f"{'head' if scale == 1.0 else 'control'}-{time:g}"
    log_path = root / f"{label}.log"
    report_path = root / f"{label}.xml"
    source = Path(__file__).resolve()
    checks = SLICE_CHECKS if scale == 1.0 else SLICE_CHECKS[:1]
    with tempfile.TemporaryDirectory(prefix=f"corsica-{label}-") as temporary:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-vv",
            "-s",
            "--timeout=600",
            f"--basetemp={temporary}",
            f"--junitxml={report_path}",
            *(f"{source}::{name}[time-{time:g}]" for name in checks),
        ]
        environment = dict(os.environ)
        environment["JAX_PLATFORMS"] = "cuda,cpu"
        environment["NOVA_CORSICA_CONTROL_TIME"] = str(time)
        environment["NOVA_CORSICA_FF_SCALE"] = str(scale)
        timed_out = False
        with log_path.open("w") as log:
            log.write(
                f"revision={revision} tree={source.parents[1]} "
                f"command={shlex.join(command)}\n"
            )
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=source.parents[1],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                code = process.wait(timeout=660)
            except subprocess.TimeoutExpired:
                timed_out = True
                log.write(
                    f"Stopping owned pytest PID {process.pid}; "
                    "parent and other processes excluded.\n"
                )
                log.flush()
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    log.write(
                        f"Owned pytest PID {process.pid} still alive; "
                        "sending SIGKILL.\n"
                    )
                    log.flush()
                    process.kill()
                    process.wait()
                code = 124
            log.write(f"EXIT={code}\n")
    failures, passed, skipped = [], [], []
    completed = report_path.exists() and not timed_out
    if completed:
        try:
            report = element_tree.parse(report_path)
        except element_tree.ParseError:
            completed = False
        else:
            for test in report.findall(".//testcase"):
                identifier = (
                    f"tests/test_forward_profile_corsica.py::{test.attrib['name']}"
                )
                if test.find("failure") is not None or test.find("error") is not None:
                    failures.append(identifier)
                elif test.find("skipped") is not None:
                    skipped.append(identifier)
                else:
                    passed.append(identifier)
    suite = {
        "revision": revision,
        "command": shlex.join(command),
        "exit_status": code,
        "log_path": str(log_path),
        "completed": completed,
        "failure_count": len(failures),
        "failure_ids": failures,
        "passed_ids": passed,
        "skipped_ids": skipped,
        "time": time,
        "ff_scale": scale,
        "timed_out": timed_out,
    }
    _write_receipt(root / f"{label}-suite.json", suite)
    return suite


def _run_slice_processes():
    """Measure all reference-independent slices and the control in one allocation."""
    root = Path(os.environ["NOVA_CORSICA_RECEIPTS"])
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source.parents[1]), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    record = {
        "revision": revision,
        "seed_kind": "input-conductor-vacuum",
        "completed": False,
        "head_arms": [],
    }
    journal = root / "batch-receipt.json"
    _write_receipt(journal, record)
    for time in SLICE_TIMES:
        print(f"START_HEAD time={time:g}", flush=True)
        arm = _slice_process(root, time, 1.0, revision)
        record["head_arms"].append(arm)
        _write_receipt(journal, record)
        print(f"END_HEAD time={time:g} exit={arm['exit_status']}", flush=True)
    candidates = [
        arm
        for arm in record["head_arms"]
        if arm["completed"]
        and any(SLICE_CHECKS[0] in identifier for identifier in arm["passed_ids"])
    ]
    head = candidates[0] if candidates else record["head_arms"][0]
    control = _slice_process(root, head["time"], 1.3, revision)
    added = sorted(set(control["failure_ids"]) - set(head["failure_ids"]))
    record.update(
        control_arm=control,
        control_head_time=head["time"],
        control_head_passed_axis_current=bool(candidates),
        negative_control_added_failure_ids=added,
        negative_control_discriminating=bool(
            control["completed"]
            and head["completed"]
            and control["exit_status"] != 0
            and added
        ),
        completed=True,
    )
    _write_receipt(journal, record)
    return int(any(arm["exit_status"] for arm in [*record["head_arms"], control]))


if __name__ == "__main__":
    if sys.argv[1:] != ["--run-slices"]:
        raise SystemExit("usage: test_forward_profile_corsica.py --run-slices")
    raise SystemExit(_run_slice_processes())
