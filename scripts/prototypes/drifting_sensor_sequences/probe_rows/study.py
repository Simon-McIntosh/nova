"""Drifting sensor sequences over one held ITER state, nova fit beside imas-efit.

Probe rows are written through the row-exchange study's sensor construction:
the machine's own ``poloidal_angle`` is copied into the pulse unchanged, and
the row value is projected along ``Br cos(theta) - Bz sin(theta)``, which is
the IMAS clockwise poloidal-angle convention that imas-efit's reader negates
(``src/imas.cpp``: ``poloidalOrientation = -probe->poloidal_angle``).  The
prior head projected along ``+Bz sin(theta)``, so every probe row was the
reflection of what the reader infers and the reconstruction left the loops by
a metre.  ``--probe-rows prior`` writes the prior convention from the same
state and the same noise, which is the negative control.

The generating state and the sensor set are the section-1 reference machine
read through :mod:`tests.test_equilibrium_magnetics_fit`: a converged 500-cell
ITER forward solve, 24 flux loops and 48 poloidal probes at the machine's own
geometry.  A sequence holds that state fixed and adds, on every sensor and
every slice, noise from one of two models of the same marginal size:

``white``   independent Gaussian per sensor and slice;
``drift``   per-sensor Gaussian noise coloured with a spectral density flat
            below a corner frequency and falling as ``f**-index`` above it
            (index 1, i.e. 1/f, for the reference sequence).

Both codes reconstruct every slice from the same rows.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

STUDY = Path(os.environ.get("DRIFT_STUDY", str(Path(__file__).resolve().parent)))
REPO = Path(__file__).resolve().parents[4]
MACHINE_URI = "imas:hdf5?user=mcintos;pulse=135011;run=7;database=iter;version=3"
DD_VERSION = "3.39.0"
EFIT_BIN = Path("/home/ITER/mcintos/Code/imas-efit/build/sdcc/src/efit")
EFIT_ROOT = Path("/home/ITER/mcintos/Code/imas-efit")
REFERENCE_FACTOR = 0.01
FIT_BUDGET = 16
# The fixture slice's own time (pulse 135011 run 7, slice 353).
SLICE_TIME = 338.9275978120551
# The row-exchange node's pulse carries the fixture's prescribed coil
# currents at this slice, which are not the machine database's own
# pf_active currents; the conductor IDSs are taken from it.
REFERENCE_PULSE = (
    "/home/ITER/mcintos/.cache/nova-inverse-equilibrium-baseline"
    "/efit-on-nova-rows-case/pulse-10"
)
CORNER_CYCLES = 4.0
GRID_RADIUS = np.linspace(3.0, 9.0, 65)
GRID_HEIGHT = np.linspace(-6.0, 6.0, 129)


def serial_pools() -> None:
    """Run every Biot-Savart batch in this process; one shard is one process."""
    if os.environ.get("NOVA_FORCE_SERIAL_POOLS") != "1":
        return
    from nova.biot import polygonanalytic

    polygonanalytic._available_cpu_count = lambda: 1


EFIT_CONFIG = (
    """<Top>
  <include file="${EFIT_CONFIG_DIR}/machines/_common/debug.xml"/>
  <include file="${EFIT_CONFIG_DIR}/machines/_common/outputOptions.xml"/>
  <include file="${EFIT_CONFIG_DIR}/machines/_common/codeControls.xml"/>
  <include file="${EFIT_CONFIG_DIR}/machines/_common/constraints.xml"/>
  <include file="${EFIT_CONFIG_DIR}/machines/ITER/machine.xml"/>
  <include file="${EFIT_CONFIG_DIR}/machines/_common/solver/default.xml"/>
  <include file="${EFIT_CONFIG_DIR}/machines/ITER/solver/magnetics.xml"/>
  <include file="${EFIT_CONFIG_DIR}/machines/_common/regularisation/default.xml"/>
  <include file="${EFIT_CONFIG_DIR}/machines/ITER/constraints/magnetics.xml"/>
  <numericalControls pressurePriorBetaPol="0">
    <pp ndeg="2" edge="1" func="8" autoDegree="off"/>
    <ff ndeg="2" edge="1" func="8" autoDegree="off"/>
    <ww ndeg="0" edge="1" func="0"/>
  </numericalControls>
  <q0 weights="0" values="1.3" sigmas="0.3" interpolationMethod="1" """
    """timeWindow="0" timeMargin="1"/>
</Top>
"""
)


def probe_row_angle(dictionary_angle) -> np.ndarray:
    """Return the sensor angle that writes an IMAS probe row [deg].

    IMAS defines a probe's sensitive axis as ``cos(theta) Rhat - sin(theta)
    Zhat`` and imas-efit reads it by negating ``poloidal_angle``; nova's sensor
    projects ``Br cos(a) + Bz sin(a)``, so the row that reproduces the IMAS
    axis is written at ``-np.asarray(dictionary_angle)``.
    """
    return -np.asarray(dictionary_angle, dtype=float)


def sensor_geometry() -> dict:
    """Read the ITER pulse's own 24 loops and 48 probes, with orientations."""
    import imas

    entry = imas.DBEntry(MACHINE_URI, "r", dd_version=DD_VERSION)
    magnetics = entry.get("magnetics")
    equilibrium = entry.get("equilibrium")
    loop = np.array(
        [
            [float(item.position[0].r), float(item.position[0].z)]
            for item in magnetics.flux_loop
        ]
    )
    probe = np.array(
        [
            [float(item.position.r), float(item.position.z)]
            for item in magnetics.b_field_pol_probe
        ]
    )
    angle = np.degrees(
        [float(item.poloidal_angle) for item in magnetics.b_field_pol_probe]
    )
    b0 = abs(float(np.asarray(equilibrium.vacuum_toroidal_field.b0)[353]))
    r0 = float(equilibrium.vacuum_toroidal_field.r0)
    frame = equilibrium.time_slice[353]
    axis = np.array(
        [
            float(frame.global_quantities.magnetic_axis.r),
            float(frame.global_quantities.magnetic_axis.z),
        ]
    )
    entry.close()
    return dict(loop=loop, probe=probe, angle=angle, b0=b0, axis=axis, r0=r0)


def build_case(cells: int = 500, reuse_rows: bool = False) -> dict:
    """Solve the generating state and evaluate its noise-free rows.

    ``reuse_rows`` loads the rows a previous build of the same state and the
    same sensor set persisted, which skips the two Biot-Savart row evaluations
    (about half an hour at 500 cells on four cores) on a card that needs the
    fits, not the rows.  The state itself is still solved: the fit needs the
    generating flux as its seed.
    """
    import importlib

    from nova.equilibrium.measurement import Magnetics
    from nova.equilibrium.solve_request import ExplicitSolveSeed

    sys.path.insert(0, str(REPO))
    tm = importlib.import_module("tests.test_equilibrium_magnetics_fit")
    lane = importlib.import_module("tests.test_equilibrium_forward_reference")

    if cells != 500:
        original = lane.cached_machine
        lane.cached_machine = lambda case, cells_requested, **kwargs: original(
            case, -abs(cells), **kwargs
        )
        try:
            profile, request, coordinate, wall, sections, known = tm._physical_problem(
                "iter"
            )
        finally:
            lane.cached_machine = original
    else:
        profile, request, coordinate, wall, sections, known = tm._physical_problem(
            "iter"
        )
    generated = profile.solve(request)
    if not generated.qualified or not generated.terminal_state.fixed_point.converged:
        raise RuntimeError("generating state failed to qualify")
    print(
        "GENERATING residual=%.3e axis=%s"
        % (
            float(generated.terminal_state.fixed_point.residual),
            np.round(np.asarray(generated.terminal_state.topology.axis), 6).tolist(),
        ),
        flush=True,
    )

    geometry = sensor_geometry()
    # The pulse carries the machine's ``poloidal_angle`` unchanged; the row is
    # projected along the IMAS clockwise convention ``(cos, -sin)``, which is
    # ``angle = -theta`` in nova's own ``(Br cos + Bz sin)`` sensor projection.
    probe_angle = probe_row_angle(geometry["angle"])
    sensors = Magnetics(
        r=np.r_[geometry["loop"][:, 0], geometry["probe"][:, 0]],
        z=np.r_[geometry["loop"][:, 1], geometry["probe"][:, 1]],
        angle=np.r_[np.zeros(24), probe_angle],
        flux_loop=np.r_[np.ones(24, bool), np.zeros(48, bool)],
    )
    prior_sensors = Magnetics(
        r=np.r_[geometry["loop"][:, 0], geometry["probe"][:, 0]],
        z=np.r_[geometry["loop"][:, 1], geometry["probe"][:, 1]],
        angle=np.r_[np.zeros(24), -probe_angle],
        flux_loop=np.r_[np.ones(24, bool), np.zeros(48, bool)],
    )
    magnetic_source = tm._PlasmaDiagnosticSource(sections)
    known_field = tm._known_field_reader(known, magnetic_source)
    stored = STUDY / "rows.npz"
    if reuse_rows and stored.exists():
        cached = np.load(stored)
        total = np.asarray(cached["total"], dtype=float)
        total_prior = np.asarray(cached["total_prior"], dtype=float)
        vacuum = np.asarray(cached["vacuum"], dtype=float)
        plasma = np.asarray(cached["plasma"], dtype=float)
        print("ROWS-REUSED " + str(stored), flush=True)
    else:
        total, vacuum, plasma, scales = tm._total_observations(
            magnetic_source, generated, sensors, known_field
        )
        total_prior = tm._total_observations(
            magnetic_source, generated, prior_sensors, known_field
        )[0]
    loops = np.asarray(sensors.flux_loop, dtype=bool)
    sigma_loop = float(np.sqrt(np.mean(plasma[loops] ** 2)))
    sigma_probe = float(np.sqrt(np.mean(plasma[~loops] ** 2)))
    plasma_current = float(np.sum(generated.terminal_state.cell_current))
    del scales

    raw = np.r_[
        request.source_profile.core.p_prime.coefficients,
        request.source_profile.core.ff_prime.coefficients,
    ]
    pressure = request.source_profile.core.p_prime.coefficients.size
    basis = np.zeros((raw.size, 2))
    basis[:pressure, 0] = raw[:pressure]
    basis[pressure:, 1] = raw[pressure:]

    topology = generated.terminal_state.topology
    reference = dict(
        axis=np.asarray(topology.axis, dtype=float).tolist(),
        flux_axis=float(topology.axis_flux),
        flux_boundary=float(topology.boundary_flux),
        flux_span=float(topology.flux_span),
        plasma_current=plasma_current,
        sigma_loop=sigma_loop,
        sigma_probe=sigma_probe,
        b0=float(geometry["b0"]),
        r0=float(geometry["r0"]),
        generating_residual=float(generated.terminal_state.fixed_point.residual),
        coordinates=int(coordinate.shape[0]),
    )
    reference["cells"] = cells
    return dict(
        profile=profile,
        request=request,
        initial=replace(
            request, seed_policy=ExplicitSolveSeed(generated.terminal_state.flux)
        ),
        generated=generated,
        coordinate=np.asarray(coordinate),
        wall=np.asarray(wall),
        sections=sections,
        sensors=sensors,
        magnetic_source=magnetic_source,
        total=total,
        total_prior=total_prior,
        vacuum=vacuum,
        plasma=plasma,
        sigma_loop=sigma_loop,
        sigma_probe=sigma_probe,
        basis=basis,
        geometry=geometry,
        reference=reference,
        cells=cells,
        module=tm,
    )


def write_prepare(case: dict) -> None:
    """Persist the noise-free rows and the generating reference."""
    geometry = case["geometry"]
    for name in ("total", "vacuum", "plasma"):
        values = np.asarray(case[name], dtype=float)
        finite = np.isfinite(values)
        if values.shape != (case["sensors"].number,):
            raise RuntimeError(
                "rows have the wrong shape: %s %s" % (name, values.shape)
            )
        if not np.all(finite):
            print(
                "ROWS_NONFINITE %s count=%d rows=%s"
                % (name, int((~finite).sum()), np.flatnonzero(~finite).tolist()),
                flush=True,
            )
        if not np.any(finite):
            raise RuntimeError("every row is non-finite: %s" % name)
    temporary = STUDY / "rows.tmp.npz"
    np.savez(
        temporary,
        total=case["total"],
        total_prior=case["total_prior"],
        vacuum=case["vacuum"],
        plasma=case["plasma"],
        loop_r=geometry["loop"][:, 0],
        loop_z=geometry["loop"][:, 1],
        probe_r=geometry["probe"][:, 0],
        probe_z=geometry["probe"][:, 1],
        probe_angle=case["geometry"]["angle"],
    )
    temporary.replace(STUDY / "rows.npz")
    (STUDY / "generating.json").write_text(json.dumps(case["reference"], indent=2))


def noise_columns(
    sequence: str,
    slices: int,
    sigma: np.ndarray,
    seed: int,
    factor: float,
    corner_cycles: float = CORNER_CYCLES,
    index: float = 1.0,
) -> np.ndarray:
    """Return one noise matrix [slices, sensors] at the stated marginal size.

    Every sensor keeps the marginal standard deviation ``factor * sigma``; the
    coloured model only redistributes power in frequency, so the two sequences
    carry noise of the same size.  The coloured sequence shapes white noise in
    the Fourier domain, with a flat band below the corner and ``f**-index``
    above it, the DC bin included -- a constant sensor offset is exactly what
    a 1/f model is there, and it is what a 50-slice mean cannot average away.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((slices, sigma.size))
    if sequence == "white":
        return factor * sigma * white

    frequency = np.fft.rfftfreq(slices)
    corner = corner_cycles / slices
    gain = np.ones_like(frequency)
    above = frequency > corner
    gain[above] = (corner / frequency[above]) ** index
    shaped = np.fft.irfft(
        np.fft.rfft(white, axis=0) * np.sqrt(gain)[:, None], n=slices, axis=0
    )
    shaped = shaped / np.std(shaped, axis=0, ddof=0)
    return factor * sigma * shaped


def flux_map_of(radius, height, flux, gradient_psi, gradient, reference_radius):
    from nova.imas.mast_efit_referee import FluxMap

    return FluxMap(
        radius=np.asarray(radius, dtype=float),
        height=np.asarray(height, dtype=float),
        flux=np.asarray(flux, dtype=float),
        reference_radius=float(reference_radius),
        gradient_psi=np.asarray(gradient_psi, dtype=float),
        gradient=np.asarray(gradient, dtype=float),
    )


def q95_of(flux_map, axis, flux_axis, flux_boundary, b0) -> float:
    """Read q95 from one flux map with the shared calculator."""
    from types import SimpleNamespace

    from nova.imas.mast_efit_referee import q95_from_flux_map

    state = SimpleNamespace(
        magnetic_axis_m=np.asarray(axis, dtype=float),
        flux_axis=float(flux_axis),
        flux_boundary=float(flux_boundary),
        flux_map=flux_map,
    )
    return float(q95_from_flux_map(state, vacuum_b0=float(b0)))


def nova_flux_map(case: dict, state) -> object:
    """Resample a solved cell flux onto the uniform lattice the reader takes."""
    from scipy.interpolate import griddata
    from scipy.ndimage import uniform_filter

    coordinate = np.asarray(case["coordinate"], dtype=float)
    flux = np.asarray(state.flux, dtype=float)[: coordinate.shape[0]]
    rad, hei = np.meshgrid(GRID_RADIUS, GRID_HEIGHT, indexing="ij")
    grid = np.asarray(griddata(coordinate, flux, (rad, hei), method="linear"))
    holes = ~np.isfinite(grid)
    if np.any(holes):
        nearest = np.asarray(griddata(coordinate, flux, (rad, hei), method="nearest"))
        grid[holes] = nearest[holes]
    grid = uniform_filter(grid, size=(3, 3), mode="nearest")
    psi_norm = np.linspace(0.0, 1.0, GRID_RADIUS.size)
    gradient = np.asarray(case["request"].source_profile.core.ff_prime(psi_norm))
    span = float(state.topology.flux_span)
    gradient_psi = float(state.topology.axis_flux) + psi_norm * span
    return flux_map_of(
        GRID_RADIUS,
        GRID_HEIGHT,
        grid,
        gradient_psi,
        gradient,
        case["reference"]["r0"],
    )


def nova_metrics(case: dict, fitted) -> dict:
    topology = fitted.forward.terminal_state.topology
    flux_map = nova_flux_map(case, fitted.forward.terminal_state)
    axis = np.asarray(topology.axis, dtype=float)
    return dict(
        axis=axis.tolist(),
        flux_axis=float(topology.axis_flux),
        flux_boundary=float(topology.boundary_flux),
        flux_span=float(topology.flux_span),
        plasma_current=float(np.sum(fitted.forward.terminal_state.cell_current)),
        q95=q95_of(
            flux_map,
            axis,
            topology.axis_flux,
            topology.boundary_flux,
            case["reference"]["b0"],
        ),
        coefficients=np.asarray(fitted.coefficients, dtype=float).ravel().tolist(),
        covariance=np.asarray(fitted.covariance, dtype=float).tolist(),
        iterations=int(fitted.iterations),
        converged=bool(fitted.converged),
        wall_seconds=float(fitted.wall_seconds),
        undetermined=list(fitted.undetermined_coefficients),
        x_points=(
            np.asarray(topology.x_point, dtype=float).reshape(-1, 2).tolist()
            if getattr(topology, "x_point", None) is not None
            else []
        ),
    )


def fitted_measurement(case: dict, measured, sigma_row: np.ndarray, current: float):
    from nova.equilibrium.measurement import SliceMeasurement

    return SliceMeasurement(
        measured=measured,
        vacuum=case["vacuum"],
        mask=np.ones(case["sensors"].number, bool),
        scale=sigma_row,
        plasma_current=float(current),
    )


def coefficient_jacobian(case: dict) -> dict:
    """Derivative of axis and q95 with respect to the shared amplitude."""
    base = case["request"].source_profile.core
    step = 1.0e-3
    probes = []
    for sign in (1.0, -1.0):
        scale = 1.0 + sign * step
        core = replace(
            base,
            p_prime=replace(
                base.p_prime, normalisation=base.p_prime.normalisation * scale
            ),
            ff_prime=replace(
                base.ff_prime, normalisation=base.ff_prime.normalisation * scale
            ),
        )
        request = replace(
            case["request"],
            source_profile=replace(case["request"].source_profile, core=core),
        )
        state = case["profile"].solve(request)
        if not (state.qualified and state.terminal_state.fixed_point.converged):
            raise RuntimeError("jacobian probe state failed to qualify")
        topology = state.terminal_state.topology
        flux_map = nova_flux_map(case, state.terminal_state)
        probes.append(
            dict(
                axis=np.asarray(topology.axis, dtype=float),
                q95=q95_of(
                    flux_map,
                    topology.axis,
                    topology.axis_flux,
                    topology.boundary_flux,
                    case["reference"]["b0"],
                ),
            )
        )
    return dict(
        axis=((probes[0]["axis"] - probes[1]["axis"]) / (2.0 * step)).tolist(),
        q95=float((probes[0]["q95"] - probes[1]["q95"]) / (2.0 * step)),
        step=step,
    )


RECEIPTS = Path(os.environ.get("STUDY_RECEIPTS", STUDY / "receipts"))


from contextlib import contextmanager  # noqa: E402


@contextmanager
def guard(name: str, index: int):
    """Keep a slice failure on the record rather than ending the shard."""
    try:
        yield
    except Exception:
        RECEIPTS.mkdir(parents=True, exist_ok=True)
        failed = RECEIPTS / ("%s-%05d.failed" % (name, index))
        failed.write_text(traceback.format_exc())
        print("SLICE_FAILED " + name + " " + str(index), flush=True)
        traceback.print_exc()


def land(name: str, record: dict) -> None:
    """Write one row receipt beside the sequence, flushed as the row lands."""
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    path = RECEIPTS / ("%s-%05d.json" % (name, record["slice"]))
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record))
    temporary.replace(path)


def slice_range(args) -> range:
    """Slices one worker owns, interleaved so every worker sees every regime."""
    return (
        range(args.shard, args.slices, args.shards)
        if args.shards > 1
        else range(args.slices)
    )


def run_nova(args) -> None:
    from nova.equilibrium.magnetics_fit import MagneticsFit

    serial_pools()

    case = build_case(args.cells, reuse_rows=args.reuse_rows)
    if args.shard == 0 or args.shards == 1:
        write_prepare(case)
    reference = case["reference"]

    generating = case["generated"].terminal_state
    reference["q95"] = q95_of(
        nova_flux_map(case, generating),
        generating.topology.axis,
        generating.topology.axis_flux,
        generating.topology.boundary_flux,
        reference["b0"],
    )
    (STUDY / "generating.json").write_text(json.dumps(reference, indent=2))
    print(
        "NOVA_REFERENCE q95=%.6f axis=%s"
        % (reference["q95"], np.round(reference["axis"], 6).tolist()),
        flush=True,
    )

    # The coefficient jacobian serves the covariance ratio alone, which this
    # node's measure does not read; it costs two full solves on the card the
    # sequence needs, so it is computed only when asked for.
    stored = STUDY / "jacobian.json"
    if stored.exists() and args.cells == 500:
        jacobian = json.loads(stored.read_text())
        print("JACOBIAN-REUSED " + json.dumps(jacobian), flush=True)
    elif args.jacobian:
        jacobian = coefficient_jacobian(case)
        stored.write_text(json.dumps(jacobian, indent=2))
        print("JACOBIAN " + json.dumps(jacobian), flush=True)
    else:
        print("JACOBIAN-SKIPPED", flush=True)

    sigma = np.r_[np.full(24, case["sigma_loop"]), np.full(48, case["sigma_probe"])]
    sigma_ip = args.factor * abs(reference["plasma_current"])
    for sequence in filter(None, (args.sequences or "").split(",")):
        suffix = "" if args.shards == 1 else "-shard%d" % args.shard
        path = STUDY / (
            "nova-%s-%d-%.4f%s.jsonl" % (sequence, args.slices, args.factor, suffix)
        )
        noise = noise_columns(sequence, args.slices, sigma, args.seed, args.factor)
        rng = np.random.default_rng(args.seed + 991)
        current = reference["plasma_current"] + sigma_ip * rng.standard_normal(
            args.slices
        )
        with path.open("w") as stream:
            for index in slice_range(args):
                measured = case["total"] + noise[index]
                measurement = fitted_measurement(case, measured, sigma, current[index])
                function = guard("nova-%s-%.4f" % (sequence, args.factor), index)
                with function:
                    fitted = MagneticsFit(
                        case["profile"],
                        case["initial"],
                        case["sensors"],
                        case["magnetic_source"],
                        coefficient_basis=case["basis"],
                    ).fit(
                        measurement,
                        plasma_current_scale=sigma_ip,
                        max_iterations=FIT_BUDGET,
                    )
                    record = nova_metrics(case, fitted)
                    record["slice"] = index
                    maps = STUDY / "maps"
                    maps.mkdir(parents=True, exist_ok=True)
                    solved_map = nova_flux_map(case, fitted.forward.terminal_state)
                    np.savez(
                        maps
                        / ("nova-%s-%.4f-%05d.npz" % (sequence, args.factor, index)),
                        radius=GRID_RADIUS,
                        height=GRID_HEIGHT,
                        flux=np.asarray(solved_map.flux, dtype=float),
                    )
                    record["plasma_current_measured"] = float(current[index])
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    land("nova-%s-%.4f" % (sequence, args.factor), record)
        print("NOVA_SEQUENCE -> %s" % path, flush=True)

    factors = [
        float(text) for text in filter(None, (args.sweep_factors or "").split(","))
    ]
    factors = factors[args.shard :: args.shards]
    if args.sweep_part == "spectrum":
        factors = []
    for swept in [] if args.skip_sweep else factors:
        sweep_sigma_ip = swept * abs(reference["plasma_current"])
        path = STUDY / ("nova-sweep-%d-%.4f.jsonl" % (args.sweep_slices, swept))
        noise = noise_columns("white", args.sweep_slices, sigma, args.seed + 7, swept)
        rng = np.random.default_rng(args.seed + 1991)
        current = reference["plasma_current"] + sweep_sigma_ip * rng.standard_normal(
            args.sweep_slices
        )
        with path.open("w") as stream:
            for index in range(args.sweep_slices):
                measured = case["total"] + noise[index]
                measurement = fitted_measurement(case, measured, sigma, current[index])
                function = guard("nova-sweep-%.4f" % swept, index)
                with function:
                    fitted = MagneticsFit(
                        case["profile"],
                        case["initial"],
                        case["sensors"],
                        case["magnetic_source"],
                        coefficient_basis=case["basis"],
                    ).fit(
                        measurement,
                        plasma_current_scale=sweep_sigma_ip,
                        max_iterations=FIT_BUDGET,
                    )
                    record = nova_metrics(case, fitted)
                    record["slice"] = index
                    maps = STUDY / "maps"
                    maps.mkdir(parents=True, exist_ok=True)
                    solved_map = nova_flux_map(case, fitted.forward.terminal_state)
                    np.savez(
                        maps
                        / ("nova-%s-%.4f-%05d.npz" % (sequence, args.factor, index)),
                        radius=GRID_RADIUS,
                        height=GRID_HEIGHT,
                        flux=np.asarray(solved_map.flux, dtype=float),
                    )
                    record["factor"] = swept
                    record["plasma_current_measured"] = float(current[index])
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    land("nova-sweep-%.4f" % swept, record)
        print("NOVA_SWEEP %.4f -> %s" % (swept, path), flush=True)

    if args.sweep_part == "size":
        spectrum = []
    else:
        spectrum = [
            (float(index), float(corner))
            for index in filter(None, args.index_sweep.split(","))
            for corner in filter(None, args.corner_sweep.split(","))
        ][args.shard :: args.shards]
    for index, corner in spectrum:
        sigma_ip = args.factor * abs(reference["plasma_current"])
        path = STUDY / (
            "nova-spectrum-%d-index%.1f-corner%.4f.jsonl"
            % (args.sweep_slices, index, corner)
        )
        noise = noise_columns(
            "drift",
            args.sweep_slices,
            sigma,
            args.seed + 13,
            args.factor,
            corner_cycles=corner,
            index=index,
        )
        rng = np.random.default_rng(args.seed + 1992)
        current = reference["plasma_current"] + sigma_ip * rng.standard_normal(
            args.sweep_slices
        )
        with path.open("w") as stream:
            for position in range(args.sweep_slices):
                measurement = fitted_measurement(
                    case, case["total"] + noise[position], sigma, current[position]
                )
                name = "nova-spectrum-%.1f-%.4f" % (index, corner)
                function = guard(name, position)
                with function:
                    fitted = MagneticsFit(
                        case["profile"],
                        case["initial"],
                        case["sensors"],
                        case["magnetic_source"],
                        coefficient_basis=case["basis"],
                    ).fit(
                        measurement,
                        plasma_current_scale=sigma_ip,
                        max_iterations=FIT_BUDGET,
                    )
                    record = nova_metrics(case, fitted)
                    record["slice"] = position
                    record["index"] = index
                    record["corner"] = corner
                    record["plasma_current_measured"] = float(current[position])
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    land(name, record)
        print("NOVA_SPECTRUM %.1f %.4f -> %s" % (index, corner, path), flush=True)


GEOM: dict = {}


def run_overlay(args) -> None:
    from nova.imas.mast_efit_referee import read_equilibrium_state

    serial_pools()

    case = build_case(args.cells, reuse_rows=True)
    reference = case["reference"]
    sigma = np.r_[np.full(24, case["sigma_loop"]), np.full(48, case["sigma_probe"])]
    sigma_ip = args.factor * abs(reference["plasma_current"])
    noise = noise_columns(args.sequence, args.slices, sigma, args.seed, args.factor)
    rng = np.random.default_rng(args.seed + 991)
    current = reference["plasma_current"] + sigma_ip * rng.standard_normal(args.slices)
    index = args.slice
    measured = case["total"] + noise[index]
    measurement = fitted_measurement(case, measured, sigma, current[index])
    from nova.equilibrium.magnetics_fit import MagneticsFit

    fitted = MagneticsFit(
        case["profile"],
        case["initial"],
        case["sensors"],
        case["magnetic_source"],
        coefficient_basis=case["basis"],
    ).fit(
        measurement,
        plasma_current_scale=sigma_ip,
        max_iterations=FIT_BUDGET,
    )
    nova_map = nova_flux_map(case, fitted)
    output = STUDY / (
        "efit-%s-%d-%.4f" % (args.sequence, args.slices, args.factor)
        if args.index is None
        else "efit-%s-%d-%.4f-index%.1f-corner%.4f"
        % (args.sequence, args.slices, args.factor, args.index, args.corner)
    )
    state = read_equilibrium_state(output, time_index=index, wall_source=MACHINE_URI)
    stored = state.flux_map
    from scipy.interpolate import griddata

    rad, hei = np.meshgrid(GRID_RADIUS, GRID_HEIGHT, indexing="ij")
    points = np.column_stack(
        [
            np.repeat(np.asarray(stored.radius, float), stored.flux.shape[1]),
            np.tile(np.asarray(stored.height, float), stored.flux.shape[0]),
        ]
    )
    values = np.asarray(stored.flux, float).reshape(-1)
    efit_map = np.asarray(griddata(points, values, (rad, hei), method="linear"))
    holes = ~np.isfinite(efit_map)
    if np.any(holes):
        efit_map[holes] = np.asarray(
            griddata(points, values, (rad, hei), method="nearest")
        )[holes]
    wall_r, wall_z = wall_outline(case["wall"])
    path = STUDY / ("overlay-%s-%d.jsonl.npz" % (args.sequence, index))
    np.savez(
        path,
        radius=GRID_RADIUS,
        height=GRID_HEIGHT,
        psi_nova=np.asarray(nova_map.flux, float),
        psi_efit=efit_map,
        axis_nova=np.asarray(fitted.topology.axis, float),
        axis_efit=np.asarray(state.magnetic_axis_m, float),
        wall_r=wall_r,
        wall_z=wall_z,
        slice=np.array(index),
    )
    print("OVERLAY -> %s" % path, flush=True)


def wall_outline(wall) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the wall units into one polyline, NaN-separated."""
    radius: list[float] = []
    height: list[float] = []
    for unit in wall:
        radius.extend(np.asarray(unit.r, float))
        radius.append(np.nan)
        height.extend(np.asarray(unit.z, float))
        height.append(np.nan)
    return np.array(radius), np.array(height)


def write_pulse(
    path,
    times,
    flux,
    field,
    current,
    sigma_loop: float,
    sigma_probe: float,
    b0: float,
    r0: float,
    machine_uri: str = MACHINE_URI,
) -> None:
    """Write one IMAS pulse carrying the noisy magnetics rows and the machine."""
    import imas

    if path.exists():
        shutil.rmtree(path)
    entry = imas.DBEntry(f"imas:hdf5?path={path}", "w", dd_version=DD_VERSION)
    magnetics = entry.factory.magnetics()
    magnetics.ids_properties.homogeneous_time = 1
    magnetics.ids_properties.comment = "synthetic drifting-sensor rows"
    magnetics.time = times
    magnetics.flux_loop.resize(flux.shape[1])
    for index in range(flux.shape[1]):
        loop = magnetics.flux_loop[index]
        loop.position.resize(1)
        loop.position[0].r = float(GEOM["loop"][index, 0])
        loop.position[0].z = float(GEOM["loop"][index, 1])
        loop.flux.data = flux[:, index]
    magnetics.b_field_pol_probe.resize(field.shape[1])
    for index in range(field.shape[1]):
        probe = magnetics.b_field_pol_probe[index]
        probe.position.r = float(GEOM["probe"][index, 0])
        probe.position.z = float(GEOM["probe"][index, 1])
        probe.poloidal_angle = float(np.radians(GEOM["angle"][index]))
        probe.field.data = field[:, index]
        if os.environ.get("NOVA_PROBES_NULL"):
            probe.field.data = np.zeros_like(np.asarray(probe.field.data, dtype=float))
        if os.environ.get("NOVA_PROBES_NEGATE"):
            probe.field.data = -np.asarray(field[:, index], dtype=float)
    magnetics.ip.resize(1)
    magnetics.ip[0].data = current
    # The row-exchange node's pulses leave every data_error_upper unfilled, so
    # EFIT++ applies its own sigma from the machine configuration (auto_sigma);
    # a filled error of the fixture's own sigma reweights the fit and moves the
    # solution away from the state the rows came from.
    entry.put(magnetics)
    # The row-exchange node's pulses carry the machine's own pf_active,
    # pf_passive and wall plus a synthesised tf, so imas-efit runs in unified
    # mode with no --machine URI; the split-entry read of the machine's passive
    # circuits aborts on a data-error field DD 3.39 does not define for
    # pf_passive.loop.current.
    machine = imas.DBEntry(machine_uri, "r", dd_version=DD_VERSION)
    reference = imas.DBEntry(
        "imas:hdf5?path=" + REFERENCE_PULSE, "r", dd_version=DD_VERSION
    )
    # The machine database carries its own pf_active currents, which are not the
    # operating point the fixture's rows were evaluated at; the reference pulse
    # holds those prescribed currents at this slice.  The machine's time base is
    # kept (so the conductors interpolate at every slice time) and its current
    # values are replaced by the prescribed ones.
    for name in ("pf_active", "pf_passive", "tf", "wall"):
        prescribed = reference.get(name)
        identifier = reference.get(name) if name == "tf" else machine.get(name)
        if name == "pf_active":
            for coil, source in zip(identifier.coil, prescribed.coil):
                coil.current.data = np.full(
                    np.asarray(identifier.time).size,
                    float(np.asarray(source.current.data)[0]),
                )
        elif name == "pf_passive":
            for loop, source in zip(identifier.loop, prescribed.loop):
                loop.current = np.full(
                    np.asarray(identifier.time).size,
                    float(np.asarray(source.current)[0]),
                )
        elif name == "tf":
            value = float(np.asarray(prescribed.b_field_tor_vacuum_r.data)[0])
            identifier.b_field_tor_vacuum_r.data = np.full(times.size, value)
            identifier.time = np.asarray(times, dtype=float)
        entry.put(identifier)
    reference.close()
    machine.close()
    tf = entry.factory.tf()
    tf.ids_properties.homogeneous_time = 1
    tf.time = np.atleast_1d(times[:1])
    # The row-exchange node's synthesised tf carries the signed vacuum
    # toroidal flux function B0 * R0, which is what its pulse-10 holds.
    tf.b_field_tor_vacuum_r.data = np.atleast_1d(-float(b0) * float(r0))
    tf.b_field_tor_vacuum_r.time = np.atleast_1d(times[:1])
    entry.put(tf)
    entry.close()


def run_efit(args) -> None:
    global GEOM
    serial_pools()
    rows = np.load(STUDY / "rows.npz")
    reference = json.loads((STUDY / "generating.json").read_text())
    GEOM.clear()
    GEOM.update(
        loop=np.column_stack([rows["loop_r"], rows["loop_z"]]),
        probe=np.column_stack([rows["probe_r"], rows["probe_z"]]),
        angle=rows["probe_angle"],
    )
    sigma_loop = float(reference["sigma_loop"])
    sigma_probe = float(reference["sigma_probe"])
    sigma = np.r_[np.full(24, sigma_loop), np.full(48, sigma_probe)]

    label = "%s-%d-%.4f" % (args.sequence, args.slices, args.factor)
    probe_rows = getattr(args, "probe_rows", "corrected")
    if probe_rows == "prior":
        totals = np.asarray(rows["total_prior"], dtype=float)[None, :]
        label = "%s-prior" % label
    else:
        totals = np.asarray(rows["total"], dtype=float)[None, :]
    if getattr(args, "tag", None):
        label = "%s-%s" % (label, args.tag)
    if args.index is not None:
        label += "-index%.1f-corner%.4f" % (args.index, args.corner)
    pulse = STUDY / ("pulse-" + label)
    output = STUDY / ("efit-" + label)
    for target in (pulse, output):
        if target.exists():
            shutil.rmtree(target)
    noise_seed = args.seed if args.noise_seed is None else args.noise_seed
    current_seed = args.seed + 991 if args.current_seed is None else args.current_seed
    noise = noise_columns(
        args.sequence,
        args.slices,
        sigma,
        noise_seed,
        args.factor,
        corner_cycles=args.corner or CORNER_CYCLES,
        index=1.0 if args.index is None else args.index,
    )
    rng = np.random.default_rng(current_seed)
    sigma_ip = args.factor * abs(reference["plasma_current"])
    totals = totals + noise
    current = reference["plasma_current"] + sigma_ip * rng.standard_normal(args.slices)
    times = SLICE_TIME + np.arange(args.slices, dtype=float)
    write_pulse(
        pulse,
        times,
        totals[:, :24],
        totals[:, 24:],
        current,
        sigma_loop * args.factor,
        sigma_probe * args.factor,
        reference["b0"],
        reference["r0"],
    )

    config = STUDY / "conf" / "machines" / "ITER"
    config.mkdir(parents=True, exist_ok=True)
    (config / "efitOptions-nova2.xml").write_text(EFIT_CONFIG)
    log = STUDY / ("efit-%s.log" % label)
    argv = [
        str(EFIT_BIN),
        str(config / "efitOptions-nova2.xml"),
        "--pulse",
        str(pulse),
        "--output",
        str(output),
        "--tvals",
        ",".join("%g" % float(t) for t in times),
        "--dd-out",
        "4.1.0",
        "--options-dir",
        str(config),
        "--weight-dir",
        str(EFIT_ROOT / "configuration" / "machines" / "ITER" / "weights" / "standard"),
        "--timing",
    ]
    environment = dict(os.environ)
    environment["EFIT_CONFIG_DIR"] = str(EFIT_ROOT / "configuration")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [
            "/work/imas/opt/EasyBuild/software/GSL/2.7-GCC-13.2.0/lib",
            "/work/imas/opt/EasyBuild/software/GCCcore/13.2.0/lib64",
            "/work/imas/opt/EasyBuild/software/Boost/1.83.0-GCC-13.2.0/lib",
            "/work/imas/opt/EasyBuild/software/HDF5/1.14.3-gompi-2023b/lib",
            "/work/imas/opt/EasyBuild/software/netCDF/4.9.2-gompi-2023b/lib",
            "/work/imas/opt/EasyBuild/software/netCDF-Fortran/4.6.1-gompi-2023b/lib",
            "/work/imas/opt/EasyBuild/software/netCDF-C++4/4.3.1-gompi-2023b/lib",
            "/work/imas/opt/EasyBuild/software/PSPLINE/2.0.1-gompi-2023b/lib",
            "/work/imas/opt/EasyBuild/software/Mamba/23.11.0-0/lib",
            "/work/imas/opt/EasyBuild/software/Blitz++/1.0.2-GCCcore-13.2.0/lib64",
        ]
    )
    print("EFIT_ARGV " + json.dumps(argv), flush=True)
    with log.open("w") as stream:
        completed = subprocess.run(
            argv, stdout=stream, stderr=subprocess.STDOUT, env=environment
        )
    if completed.returncode:
        raise RuntimeError("imas-efit exited %d; see %s" % (completed.returncode, log))
    read_efit(output, reference, args, label)


def wall_npz(path: Path, machine_uri: str = MACHINE_URI) -> Path:
    """Write the machine's limiter ring as a two-column wall array.

    The limiter units are read by
    :func:`nova.equilibrium.wall_mask.wall_units_from_ids`, which owns the IDS
    walk, the unit kind and the closure test; this function only chooses the
    first unit that carries a contour and writes it beside the flux maps.
    """

    import imas

    from nova.equilibrium.wall_mask import wall_units_from_ids

    entry = imas.DBEntry(machine_uri, "r", dd_version=DD_VERSION)
    try:
        units = wall_units_from_ids(entry.get("wall"))
    finally:
        entry.close()
    rings = [unit.vertices for unit in units if unit.vertices.shape[0] >= 3]
    if not rings:
        raise RuntimeError("the machine carries no limiter ring")
    if path.exists():
        path.unlink()
    np.savez(path, wall=rings[0])
    return path


def read_efit(output, reference, args, label: str) -> None:
    from nova.imas.mast_efit_referee import q95_from_flux_map, read_equilibrium_state

    name = "efit-%s.jsonl" % label
    path = STUDY / name
    wall = str(wall_npz(STUDY / "conf" / "machine-wall.npz"))
    with path.open("w") as stream:
        for index in range(args.slices):
            state = read_equilibrium_state(output, time_index=index, wall_source=wall)
            q95 = float("nan")
            if state.flux_map is not None:
                q95 = float(q95_from_flux_map(state, vacuum_b0=reference["b0"]))
            record = dict(
                slice=index,
                axis=np.asarray(state.magnetic_axis_m, dtype=float).tolist(),
                flux_axis=float(state.flux_axis),
                flux_boundary=float(state.flux_boundary),
                flux_span=float(state.flux_boundary - state.flux_axis),
                plasma_current=float(state.plasma_current_a),
                q95=q95,
                q95_stored=float(state.q95),
            )
            stream.write(json.dumps(record) + "\n")
            stream.flush()
    print("EFIT_ROWS -> %s" % path, flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    nova = commands.add_parser("nova")
    nova.add_argument("--slices", type=int, default=200)
    nova.add_argument("--factor", type=float, default=REFERENCE_FACTOR)
    nova.add_argument("--sequences", default="white,drift")
    nova.add_argument("--seed", type=int, default=20261118)
    nova.add_argument("--sweep-factors", default="")
    nova.add_argument("--sweep-slices", type=int, default=50)
    nova.add_argument("--shard", type=int, default=0)
    nova.add_argument("--shards", type=int, default=1)
    nova.add_argument("--skip-sweep", action="store_true")
    nova.add_argument("--index-sweep", default="")
    nova.add_argument("--corner-sweep", default="")
    nova.add_argument("--cells", type=int, default=500)
    nova.add_argument("--sweep-part", default="both")
    nova.add_argument("--jacobian", action="store_true")
    nova.add_argument("--reuse-rows", action="store_true")
    nova.set_defaults(handler=run_nova)
    efit = commands.add_parser("efit")
    efit.add_argument("--slices", type=int, default=200)
    efit.add_argument("--factor", type=float, default=REFERENCE_FACTOR)
    efit.add_argument("--tag", default="")
    efit.add_argument(
        "--probe-rows", choices=("corrected", "prior"), default="corrected"
    )
    efit.add_argument("--sequence", default="white")
    efit.add_argument("--seed", type=int, default=20261118)
    efit.add_argument("--index", type=float, default=None)
    efit.add_argument("--corner", type=float, default=None)
    efit.add_argument("--noise-seed", type=int, default=None)
    efit.add_argument("--current-seed", type=int, default=None)
    efit.set_defaults(handler=run_efit)
    overlay = commands.add_parser("overlay")
    overlay.add_argument("--slices", type=int, default=200)
    overlay.add_argument("--factor", type=float, default=REFERENCE_FACTOR)
    overlay.add_argument("--sequence", default="drift")
    overlay.add_argument("--seed", type=int, default=20261118)
    overlay.add_argument("--slice", type=int, default=0)
    overlay.add_argument("--cells", type=int, default=500)
    overlay.set_defaults(handler=run_overlay)
    arguments = parser.parse_args(argv)
    arguments.handler(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
