"""Carry the solved WEST iron magnetisation through the CoilSet seam.

Two stages, both writing under the plan's GPFS evidence root:

``state``  solves the affine material problem on the persisted uniform-40 mm
           Galerkin operator and stores the affine state the frame carries.
``field``  rebuilds that state as bound-current sources, inserts it as one
           passive frame in the production WEST CoilSet, and measures from the
           same seam: the outboard stray field with and without the iron on the
           three sampling planes, the zero-magnetisation control, the iron
           contribution at every in-vessel poloidal probe and every flux loop,
           and the wall of each solve.

Usage: measure.py state | field
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()

import imas  # noqa: E402
import jax  # noqa: E402
import shapely  # noqa: E402
import pickle  # noqa: E402

from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402
from nova.biot.greens import greens_bz_br  # noqa: E402
from nova.frame.coilset import CoilSet  # noqa: E402
from nova.frame.iron import insert_iron, set_iron_currents  # noqa: E402
from nova.imas.machine import StaticMachineDescription  # noqa: E402
from nova.imas.plasma_source import PlasmaSource  # noqa: E402

logging.disable(logging.WARNING)
ROOT = "/home/ITER/mcintos/Code/imas-efit/tests/data/imas/WEST"
MACHINE = f"{ROOT}/machine/dd4-1-0"
PULSE = f"{ROOT}/62751"
EQUILIBRIUM = f"{ROOT}/62751/validation/nice"
INDEX = 14
CACHE = Path("/home/ITER/mcintos/.cache/nova-west-stray/hex/cache")
EVIDENCE = Path(
    "/work/projects/imas_gpu/sophelio/evidence/iron-core-magnetisation/field-source"
)
LINEAR = Path(
    "/work/projects/imas_gpu/sophelio/evidence/iron-core-magnetisation/"
    "linear-magnetisation/west-h200"
)
DEFAULT_MESH = "uniform-0.08"
MESH = DEFAULT_MESH
CELL_SIZE = 0.04
LINES = {"midplane": 0.0, "Fb plane": -0.637, "Fh plane": 0.637}

def state_path():
    return EVIDENCE / f"west-62751-{MESH}-state.npz"


def bound_path():
    return EVIDENCE / f"west-62751-{MESH}-bound-currents.pkl"


def field_path():
    return EVIDENCE / f"west-62751-{MESH}-field.npz"


def receipt_path():
    return EVIDENCE / f"west-62751-{MESH}-field-receipt.json"


COIL_RESPONSE = EVIDENCE / "coil-response"


def quadrature(data):
    """Split the prototype's external-term quadrature into coils and plasma."""
    radius, height, amps = [], [], []
    for coil in data["coils"]:
        offsets = (np.arange(6) + 0.5) / 6 - 0.5
        rr, zz = np.meshgrid(
            coil["r"] + offsets * coil["dr"], coil["z"] + offsets * coil["dz"]
        )
        radius.extend(rr.ravel())
        height.extend(zz.ravel())
        amps.extend(
            np.full(rr.size, data["currents"][coil["name"]] * coil["nturn"] / rr.size)
        )
    coils = (np.column_stack((radius, height)), np.asarray(amps))
    radius, height, amps = [], [], []
    for polygon, current in zip(
        data["plasma_cells"], data["plasma_weight"], strict=True
    ):
        centre = shapely.Polygon(polygon).centroid
        radius.append(centre.x)
        height.append(centre.y)
        amps.append(current)
    plasma = (np.array([radius, height]).T, np.asarray(amps))
    return coils, plasma


def field_at(points, rows):
    """Evaluate the (B_R, B_Z) of a quadrature set at ``points``."""
    points = np.asarray(points, dtype=np.float64)
    coords, amps = rows
    values = np.empty((len(points), 2))
    for start in range(0, len(points), 256):
        stop = start + 256
        bz, br = greens_bz_br(
            points[start:stop, 0, None],
            points[start:stop, 1, None],
            coords[None, :, 0],
            coords[None, :, 1],
        )
        values[start:stop] = np.column_stack((br @ amps, bz @ amps))
    return values


def coil_set_key(data):
    """Return the digest of the coil geometry and currents the response serves."""
    payload = [
        [
            coil["name"],
            coil["r"],
            coil["z"],
            coil["dr"],
            coil["dz"],
            coil["nturn"],
            data["currents"][coil["name"]],
        ]
        for coil in data["coils"]
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def persist_coil_field(points, rows, mesh_digest, coil_key):
    """Persist the coils' field through the package's response cache.

    The cache owner holds one container and one digest rule: the mesh identity
    names the ordered quadrature and the recorded coil key names the coil set.
    A request carrying either a different mesh or a different coil set raises
    the owner's refusal rather than serving another problem's field.
    """
    from nova.biot.magnetisation import (
        load_magnetisation_interaction,
        save_magnetisation_interaction,
    )

    region, _, _, _ = west_region()
    COIL_RESPONSE.mkdir(parents=True, exist_ok=True)
    path = COIL_RESPONSE / f"coil-{mesh_digest[:16]}-{coil_key[:16]}.npy"
    if path.exists():
        stored = load_magnetisation_interaction(
            path,
            region=region,
            cell_size=CELL_SIZE,
            basis="coil",
            coil_key=coil_key,
        )
        return np.asarray(stored.matrix), True
    values = field_at(points, rows)
    save_magnetisation_interaction(
        path,
        values,
        region=region,
        cell_size=CELL_SIZE,
        basis="coil",
        coil_key=coil_key,
    )
    return values, False


def entry(path, dd):
    return imas.DBEntry(f"imas:hdf5?path={path}", "r", dd_version=dd)


def west_region():
    """Return the authored WEST iron region, its material table and the machine."""
    machine = StaticMachineDescription.read_west_iron_outline(
        f"imas:hdf5?path={MACHINE}", dd_version="4.1.0"
    ).description
    table = machine.iron_segments[0].material
    region = shapely.union_all(
        [
            shapely.Polygon(np.column_stack((s.outline.r, s.outline.z)))
            for s in machine.iron_segments
        ]
    )
    return (
        region,
        np.asarray(table.b_field),
        np.asarray(table.permeability_relative),
        machine,
    )


def stage_state():
    """Solve the affine material problem and store the state the frame carries."""
    from scripts.iron_graded_refinement.measure import west_inputs

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    region, table_b, table_mu, _ = west_region()
    pinned = json.loads((LINEAR / f"{MESH}.json").read_text())
    assert pinned["completed"] and pinned["full_drive"]["converged"]
    assert float(pinned["mesh"].split("-")[1]) == CELL_SIZE
    started = perf_counter()
    interaction = iron.load_magnetisation_interaction(
        LINEAR / f"{MESH}-operator.npy",
        region=region,
        cell_size=CELL_SIZE,
        basis="linear",
    )
    print(f"operator loaded {perf_counter() - started:.1f} s", flush=True)
    inputs = west_inputs()
    from scripts.iron_graded_refinement.measure import SOURCE

    with SOURCE.open("rb") as stream:
        data = pickle.load(stream)
    mesh = iron._mesh(region, CELL_SIZE)
    geometry = iron.linear_magnetisation_geometry(mesh.polygons)
    points = np.asarray(geometry["points"], dtype=np.float64)
    mesh_digest = iron._geometry_digest(mesh, "linear")
    coil_rows, plasma_rows = quadrature(data)
    coil_started = perf_counter()
    coil_values, reused = persist_coil_field(
        points, coil_rows, mesh_digest, coil_set_key(data)
    )
    coil_seconds = perf_counter() - coil_started
    direct_started = perf_counter()
    direct_coil = field_at(points, coil_rows)
    direct_coil_seconds = perf_counter() - direct_started
    coil_relative = float(
        np.max(
            np.abs(coil_values - direct_coil) / np.maximum(np.abs(direct_coil), 1e-30)
        )
    )
    plasma_started = perf_counter()
    plasma_values = field_at(points, plasma_rows)
    plasma_seconds = perf_counter() - plasma_started
    external_values = coil_values + plasma_values
    direct_external = np.asarray(inputs["external"](points), dtype=np.float64)
    external_relative = float(
        np.max(
            np.abs(external_values - direct_external)
            / np.maximum(np.abs(direct_external), 1e-30)
        )
    )
    external_frobenius = float(
        np.linalg.norm(external_values - direct_external)
        / np.linalg.norm(direct_external)
    )

    def external(requests):
        requests = np.asarray(requests, dtype=np.float64)
        if requests.shape == points.shape and np.array_equal(requests, points):
            return external_values
        raise ValueError("the persisted coil response serves another mesh")

    refusals = {}
    try:
        external(points[: len(points) // 2])
    except ValueError as error:
        refusals["another mesh"] = str(error)
    try:
        persist_coil_field(points, coil_rows, mesh_digest, "0" * 64)
    except ValueError as error:
        refusals["another coil set"] = str(error)
    started = perf_counter()
    result = iron.solve_magnetisation(
        region,
        external,
        table_b,
        table_mu,
        cell_size=CELL_SIZE,
        basis="linear",
        interaction=interaction,
        tolerance=1e-8,
        max_iterations=60,
    )
    solve_seconds = perf_counter() - started

    # The slice wall with the persisted response against the same solve with the
    # coils evaluated on the slice: the persisted term must not change the state,
    # so this comparison is both the wall the plan asks for and its identity check.
    direct_started = perf_counter()
    direct_result = iron.solve_magnetisation(
        region,
        inputs["external"],
        table_b,
        table_mu,
        cell_size=CELL_SIZE,
        basis="linear",
        interaction=interaction,
        tolerance=1e-8,
        max_iterations=60,
    )
    direct_solve_seconds = perf_counter() - direct_started
    state_relative = float(
        np.max(
            np.abs(
                np.asarray(direct_result.linear_coefficients)
                - np.asarray(result.linear_coefficients)
            )
            / np.maximum(np.abs(np.asarray(direct_result.linear_coefficients)), 1e-30)
        )
    )
    bound = result.bound_currents()
    with bound_path().open("wb") as stream:
        pickle.dump(bound, stream)
    np.savez(
        state_path(),
        currents=bound.currents,
        coefficients=result.linear_coefficients,
        probes=inputs["probes"],
        air=inputs["air"],
        field=result.field(inputs["probes"]),
    )
    receipt = dict(
        stage="state",
        mesh=MESH,
        cells=result.cell_count,
        unknowns=bound.unknowns,
        converged=bool(result.converged),
        termination=result.termination,
        iterations=result.iterations,
        residual_history=list(result.residual_history),
        mu_min=float(result.permeability_relative.min()),
        mu_max=float(result.permeability_relative.max()),
        solve_seconds=solve_seconds,
        slice_seconds_with_response=solve_seconds,
        slice_seconds_without_response=direct_solve_seconds,
        coil_field_seconds_persisted=coil_seconds,
        coil_field_seconds_direct=direct_coil_seconds,
        plasma_field_seconds=plasma_seconds,
        state_relative_without_response=state_relative,
        air_drift=inputs["air_drift"],
        coil_response_reused=bool(reused),
        coil_response_points=int(len(points)),
        coil_response_relative=coil_relative,
        external_relative=external_relative,
        external_relative_frobenius=external_frobenius,
        coil_response_refusals=refusals,
        operator=str(LINEAR / f"{MESH}-operator.npy"),
        module=str(iron.__file__),
        cwd=str(Path.cwd()),
        x64=bool(jax.config.jax_enable_x64),
        backend=str(jax.default_backend()),
        completed=True,
    )
    (EVIDENCE / f"west-62751-{MESH}-state-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)


def bound_from_state():
    """Load the bound-current source set the frame carries."""
    with bound_path().open("rb") as stream:
        return pickle.load(stream)


def frame_coilset():
    """Load the production WEST CoilSet and apply the 62751 slice's currents."""
    with entry(MACHINE, "4.1.0") as md:
        pf_md = md.get("pf_active", autoconvert=False)
    coils = [
        dict(
            name=str(coil.name).strip(),
            r=float(element.geometry.rectangle.r),
            z=float(element.geometry.rectangle.z),
            dr=float(element.geometry.rectangle.width),
            dz=float(element.geometry.rectangle.height),
            nturn=float(element.turns_with_sign),
        )
        for coil in pf_md.coil
        for element in coil.element
    ]
    with entry(EQUILIBRIUM, "3.42.2") as eq_entry, entry(PULSE, "4.1.0") as pulse:
        equilibrium = eq_entry.get("equilibrium", autoconvert=False)
        active = pulse.get("pf_active", autoconvert=False)
        magnetics = pulse.get("magnetics", autoconvert=False)
        source = PlasmaSource.from_imas(
            equilibrium, active, provenance="WEST 62751 NICE", indices=[INDEX]
        )
        slice_ = source.slices[0]
        currents = dict(zip(source.coils.names, source.coils.at(slice_.time)))
        prescribed = source.prescribe(0)
        probes = np.column_stack(
            (
                [float(probe.position.r) for probe in magnetics.b_field_pol_probe],
                [float(probe.position.z) for probe in magnetics.b_field_pol_probe],
            )
        )
        loops = np.column_stack(
            (
                [float(loop.position[0].r) for loop in magnetics.flux_loop],
                [float(loop.position[0].z) for loop in magnetics.flux_loop],
            )
        )
    coilset = CoilSet(filename="west-62751-coilset", dirname=CACHE).load()
    for coil in coils:
        coilset.sloc[coil["name"], "Ic"] = currents[coil["name"]]
    plasma_rows = np.flatnonzero(np.asarray(coilset.subframe["part"]) == "plasma")
    polys = np.asarray(coilset.subframe["poly"], dtype=object)[plasma_rows]
    cells = [np.asarray(poly.boundary, float).reshape(-1, 2) for poly in polys]
    lcfs = shapely.Polygon(prescribed.boundary)
    density = RegularGridInterpolator(
        (prescribed.radius, prescribed.height),
        prescribed.current_density,
        bounds_error=False,
        fill_value=0.0,
    )
    weight = np.zeros(len(cells))
    for position, cell in enumerate(cells):
        part = shapely.Polygon(cell).intersection(lcfs)
        if part.area > 0:
            centroid = part.centroid
            weight[position] = density([[centroid.x, centroid.y]])[0] * part.area
    weight *= prescribed.plasma_current / weight.sum()
    ionize = weight != 0
    coilset.plasma.ionize = ionize
    coilset.plasma.nturn = weight[ionize] / weight[ionize].sum()
    plasma_name = [
        name
        for name, part in zip(coilset.frame.index, coilset.frame["part"])
        if part == "plasma"
    ][0]
    coilset.sloc[plasma_name, "Ic"] = prescribed.plasma_current
    return coilset, coils, probes, loops, float(slice_.time)


def stage_field():
    """Measure the seam on the production path and tabulate the iron's share."""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    bound = bound_from_state()
    coilset, coils, probes, loops, time = frame_coilset()
    edge = max(
        coil["r"] + coil["dr"] / 2 for coil in coils if coil["name"] in ("Fh", "Fb")
    )
    # the plasma closed-coupling assembly costs about 3.4 s per CPU target and
    # is linear in the target count, so the sampled set is sized to the turn
    radius = np.linspace(edge, 16.0, 10)
    line_targets = np.vstack(
        [np.column_stack((radius, np.full_like(radius, z))) for z in LINES.values()]
    )
    targets = np.vstack((line_targets, probes, loops))
    started = perf_counter()
    coilset.point.solve(targets)
    air = np.column_stack((np.asarray(coilset.point.br), np.asarray(coilset.point.bz)))
    air_psi = np.asarray(coilset.point.psi).copy()
    air_seconds = perf_counter() - started
    insert_iron(coilset, bound.zero())
    started = perf_counter()
    coilset.point.solve(targets)
    zero = np.column_stack((np.asarray(coilset.point.br), np.asarray(coilset.point.bz)))
    zero_psi = np.asarray(coilset.point.psi).copy()
    zero_seconds = perf_counter() - started
    control = float(np.max(np.abs(zero - air)))
    control_psi = float(np.max(np.abs(zero_psi - air_psi)))
    set_iron_currents(coilset, bound)
    started = perf_counter()
    coilset.point.solve(targets)
    with_iron = np.column_stack(
        (np.asarray(coilset.point.br), np.asarray(coilset.point.bz))
    )
    with_psi = np.asarray(coilset.point.psi).copy()
    with_seconds = perf_counter() - started
    contribution = with_iron - air
    nline = line_targets.shape[0]
    names = list(LINES)
    bmag_air = np.hypot(*air[:nline].T).reshape(len(names), -1)
    bmag_iron = np.hypot(*with_iron[:nline].T).reshape(len(names), -1)
    direct_probe = bound.field(probes)
    frame_probe = contribution[nline : nline + len(probes)]
    direct_loop = bound.psi(loops)
    frame_loop = with_psi[nline + len(probes) :] - air_psi[nline + len(probes) :]
    np.savez(
        field_path(),
        radius=radius,
        line_names=np.asarray(names),
        lines=np.asarray(list(LINES.values())),
        edge=edge,
        bmag_air=bmag_air,
        bmag_iron=bmag_iron,
        contribution=contribution,
        probes=probes,
        loops=loops,
        frame_probe=frame_probe,
        direct_probe=direct_probe,
        frame_loop=frame_loop,
        direct_loop=direct_loop,
        air=air,
        with_iron=with_iron,
    )
    receipt = dict(
        stage="field",
        cells=int(bound.volume.shape[1]),
        unknowns=int(bound.unknowns),
        atoms=int(len(bound.atoms)),
        targets=int(len(targets)),
        probe_count=int(len(probes)),
        loop_count=int(len(loops)),
        zero_control_field=control,
        zero_control_psi=control_psi,
        zero_control_seconds=zero_seconds,
        air_seconds=air_seconds,
        with_iron_seconds=with_seconds,
        frame_versus_direct_probe=float(np.max(np.abs(frame_probe - direct_probe))),
        frame_versus_direct_loop=float(np.max(np.abs(frame_loop - direct_loop))),
        probe_field_norm_max=float(np.max(np.hypot(*direct_probe.T))),
        line_planes=dict(
            zip(
                names,
                [
                    dict(
                        air_edge_mT=float(bmag_air[k][0] * 1e3),
                        iron_edge_mT=float(bmag_iron[k][0] * 1e3),
                        air_4m_mT=float(bmag_air[k][-1] * 1e3),
                        iron_4m_mT=float(bmag_iron[k][-1] * 1e3),
                    )
                    for k in range(len(names))
                ],
            )
        ),
        completed=True,
    )
    receipt_path().write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def main():
    global MESH, CELL_SIZE
    if len(sys.argv) > 2:
        MESH = sys.argv[2]
        CELL_SIZE = float(MESH.split("-")[1])
    match sys.argv[1:2]:
        case ["state"]:
            stage_state()
        case ["field"]:
            stage_field()
        case _:
            raise SystemExit("usage: measure.py state | field")


if __name__ == "__main__":
    main()
