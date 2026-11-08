"""Re-solve every EFIT slice of one MAST pulse through the forward-solve seam.

Each slice is a free-boundary forward solve: EFIT's own p' and FF' tables,
every one of its 101 fitted circuit currents (13 active and 88 passive or
vessel circuits) and its plasma current are prescribed, and Nova finds the
flux, the plasma region and the boundary itself. The EFIT flux map is only
the seed. The result is a stack of solved 2-D maps, which the Thomson
animation contours, instead of the rasterless labeller sessions it used to
draw.

Two choices set what the limited phase of the pulse looks like:

- **The wall is sampled densely.** EFIT's limiter is a 37-vertex polygon whose
  centre column is ONE segment from Z = -1.084 m to +1.084 m. A wall target
  placed only at the vertices has no sample between those ends. A plasma
  leaning on the column touches it near the midplane, so a vertex-only wall
  read cannot find the contact, and the limited boundary floats inside the
  vessel. Each segment is therefore resampled at a spacing well under the grid
  pitch.
- **The grid is uniform in metres and crops to the vessel.** The plasma cells
  are the grid nodes, so one choice sets both resolutions. The stored EFIT
  grid is 65 x 65 over 1.94 m x 4 m (30 mm x 62 mm cells). The default here is
  2 cm square cells over the limiter's bounding box.

Every row persists as it lands, so an interrupted run keeps the rows it
solved. ``--rows`` names a subset for a quick check.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

DEFAULT_PITCH = 0.02
WALL_SPACING_FRACTION = 0.25
NEWTON_STEPS = 12
GMRES_ITERATIONS = 12
FIXED_POINT_CRITERION = 1.0e-8
STORED_CIRCUIT_COUNT = 101


def densify_wall(limiter: np.ndarray, spacing: float) -> np.ndarray:
    """Return the closed limiter resampled at no more than ``spacing`` per step.

    Every original vertex is kept, so corners stay exact; only the straight
    runs between them gain samples.
    """
    loop = np.asarray(limiter, dtype=float)
    if not np.allclose(loop[0], loop[-1]):
        loop = np.vstack([loop, loop[:1]])
    points = []
    for start, end in zip(loop[:-1], loop[1:]):
        count = max(1, int(np.ceil(np.hypot(*(end - start)) / spacing)))
        fraction = np.arange(count)[:, None] / count
        points.append(start + fraction * (end - start))
    return np.vstack(points)


def pulse_rows(group) -> list[int]:
    """Return the EFIT rows carrying a finite map, boundary and profile pair."""
    flux = np.asarray(group["psirz"])
    count = np.asarray(group["lcfsn_c"])
    pprime = np.asarray(group["pprime"])
    ffprime = np.asarray(group["ffprime"])
    return [
        row
        for row in range(flux.shape[0])
        if np.isfinite(flux[row]).any()
        and np.isfinite(count[row])
        and np.all(np.isfinite(pprime[row]))
        and np.all(np.isfinite(ffprime[row]))
    ]


#: Sources whose change invalidates a stored operand set.
OPERAND_SOURCES = (
    "nova/biot/greens.py",
    "nova/biot/tiledassembly.py",
    "nova/imas/mast_vacuum_response.py",
    "benchmarks/efit_forward_parity_slice.py",
)


def operand_path(shot: int, coordinate: np.ndarray, wall: np.ndarray) -> Path:
    """Return the content-addressed store path of one pulse's Green operands.

    The key covers the shot (its stored circuit geometry), the grid and wall
    coordinates, and the source of every module that computes a coupling, so a
    changed input or kernel selects a new file and a stale one is never served.
    """
    import hashlib

    from nova.database.filepath import derived_cache_root

    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256(str(shot).encode())
    digest.update(np.ascontiguousarray(coordinate, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(wall, dtype=np.float64).tobytes())
    for name in OPERAND_SOURCES:
        digest.update((root / name).read_bytes())
    return (
        derived_cache_root()
        / "pulse-forward"
        / "operands"
        / f"{shot}-{digest.hexdigest()[:20]}.npz"
    )


class PulseMachine:
    """Shot-invariant operands: grid, dense wall and every Green block.

    These are built once per pulse; only the source tables, the circuit
    currents and the seed change between slices.
    """

    def __init__(self, group, shot: int, pitch: float):
        import jax.numpy as jnp

        from benchmarks.efit_forward_parity_slice import (
            _circuit_drives,
            _inside_limiter,
            _plasma_response,
            _source_response,
            _stored_circuit_fields,
        )
        from nova.biot.null import Null1D, Null2D
        from nova.biot.target import FluxTarget
        from nova.catalog.mast_geometry import MachineGeometryRegistry
        from nova.equilibrium.conservation import FluxLattice
        from nova.geometry.hexstencil import hex_stencil

        started = time.perf_counter()
        self.limiter = np.column_stack(
            [
                np.asarray(group["limiterr"], dtype=float),
                np.asarray(group["limiterz"], dtype=float),
            ]
        )
        margin = pitch
        lower = self.limiter.min(axis=0) - margin
        upper = self.limiter.max(axis=0) + margin
        stored_r = np.asarray(group["gridr"], dtype=float)
        lower[0] = max(lower[0], stored_r[0])
        nr = int(np.ceil((upper[0] - lower[0]) / pitch)) + 1
        nz = int(np.ceil((upper[1] - lower[1]) / pitch)) + 1
        self.radius = np.linspace(lower[0], upper[0], nr)
        self.height = np.linspace(lower[1], upper[1], nz)
        self.lattice = FluxLattice(self.radius, self.height)
        coordinate = np.asarray(self.lattice.coordinate)
        self.material = _inside_limiter(self.radius, self.height, self.limiter)
        dr = float(np.diff(self.radius).mean())
        dz = float(np.diff(self.height).mean())
        self.wall = densify_wall(self.limiter, WALL_SPACING_FRACTION * min(dr, dz))

        geometry = MachineGeometryRegistry.default().select(shot).configuration.geometry
        row = pulse_rows(group)[0]
        families, drive, active_mapping = _circuit_drives(
            group, row, geometry, "fcoil_c"
        )
        self.zero_drive = np.zeros_like(np.asarray(drive, dtype=float))
        cache = operand_path(shot, coordinate, self.wall)
        if cache.exists():
            with np.load(cache, allow_pickle=False) as stored:
                operands = {name: stored[name] for name in stored.files}
            self.circuit_audit = json.loads(str(operands.pop("audit")))
            self.operand_source = f"loaded {cache}"
        else:
            targets = np.vstack((coordinate, self.wall))
            circuits, audit = _stored_circuit_fields(
                group, row, targets, geometry, active_mapping
            )
            if len(circuits) != STORED_CIRCUIT_COUNT:
                raise RuntimeError(
                    f"{len(circuits)} stored circuits, expected {STORED_CIRCUIT_COUNT}"
                )
            self.circuit_audit = {
                key: value for key, value in audit.items() if np.isscalar(value)
            }
            operands = {
                "response": np.column_stack(
                    [item["response_wb_per_a"] for item in circuits]
                ),
                "source_to_grid": _source_response(geometry, coordinate, families),
                "source_to_wall": _source_response(geometry, self.wall, families),
                "plasma_to_grid": _plasma_response(coordinate, coordinate, dr, dz),
                "plasma_to_wall": _plasma_response(self.wall, coordinate, dr, dz),
            }
            cache.parent.mkdir(parents=True, exist_ok=True)
            pending = cache.with_suffix(".pending.npz")
            np.savez(pending, audit=json.dumps(self.circuit_audit), **operands)
            pending.replace(cache)
            self.operand_source = f"built and stored {cache}"
        self.response = operands["response"]
        source_to_grid = operands["source_to_grid"]
        source_to_wall = operands["source_to_wall"]
        plasma_to_grid = operands["plasma_to_grid"]
        plasma_to_wall = operands["plasma_to_wall"]
        self.grid_target = FluxTarget(
            source_target=jnp.asarray(source_to_grid),
            plasma_target=jnp.asarray(plasma_to_grid),
            null=Null2D.from_coordinates(
                coordinate, hex_stencil(self.lattice.shape), maxsize=5
            ),
        )
        self.wall_target = FluxTarget(
            source_target=jnp.asarray(source_to_wall),
            plasma_target=jnp.asarray(plasma_to_wall),
            null=Null1D(jnp.asarray(self.wall, dtype=jnp.float64)),
        )
        self.build_seconds = time.perf_counter() - started

    def provenance(self) -> dict[str, object]:
        """Return what was built, in units a reader can check."""
        return {
            "grid_shape": [self.radius.size, self.height.size],
            "grid_pitch_m": [
                float(np.diff(self.radius).mean()),
                float(np.diff(self.height).mean()),
            ],
            "grid_extent_m": [
                [float(self.radius[0]), float(self.radius[-1])],
                [float(self.height[0]), float(self.height[-1])],
            ],
            "plasma_cells_inside_limiter": int(np.count_nonzero(self.material)),
            "wall_samples": int(self.wall.shape[0]),
            "limiter_vertices": int(self.limiter.shape[0]),
            "circuits": {
                key: value
                for key, value in self.circuit_audit.items()
                if not isinstance(value, np.ndarray)
            },
            "build_seconds": self.build_seconds,
            "operands": self.operand_source,
        }


def row_source(group, row: int):
    """Return one slice's EFIT p' and FF' as array-owned flux functions.

    Array leaves rather than closures, so every slice enters the one compiled
    solve program as data instead of compiling its own.
    """
    from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
    from nova.equilibrium.solve_request import SampledFluxFunction
    from nova.equilibrium.source import DomainProfile, ForwardSource

    psi_norm = np.asarray(group["psi_norm"], dtype=float)
    p_prime = -np.asarray(group["pprime"][row], dtype=float) / TOTAL_FLUX_FACTOR
    ff_prime = -np.asarray(group["ffprime"][row], dtype=float) / TOTAL_FLUX_FACTOR
    return ForwardSource(
        core=DomainProfile(
            p_prime=SampledFluxFunction(psi_norm, p_prime),
            ff_prime=SampledFluxFunction(psi_norm, ff_prime),
        ),
        boundary_pressure=float(group["ppsi_c"][row, -1]),
        boundary_field_function=float(group["fpsi_c"][row, -1]),
    )


def pulse_profile(machine: PulseMachine, group, rows: list[int]):
    """Return the one forward profile every slice of the pulse is solved on.

    Geometry, Green blocks and the circuit response are fixed for the pulse;
    each request substitutes its own source tables and circuit currents. The
    plasma-current polarity is a static property of the program, so a pulse
    whose current changes sign across the requested rows is refused.
    """
    import jax.numpy as jnp

    from benchmarks.efit_forward_parity_slice import plasma_current_polarity
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.forward_operator import (
        ForwardFluxOperator,
        PrescribedCurrentField,
    )

    polarity = {plasma_current_polarity(group["plasma_current_c"][row]) for row in rows}
    if len(polarity) != 1:
        raise ValueError("the requested rows carry both plasma-current polarities")
    first = rows[0]
    operator = ForwardFluxOperator(
        grid=machine.grid_target,
        wall=machine.wall_target,
        source=row_source(group, first),
        external_current=jnp.asarray(machine.zero_drive),
        area=jnp.asarray(machine.lattice.cell_area),
        polarity=polarity.pop(),
        inside_material=jnp.asarray(machine.material),
        use_linear_moments=False,
        prescribed_current_field=PrescribedCurrentField(
            response=jnp.asarray(machine.response),
            current=jnp.asarray(np.asarray(group["fcoil_c"][first], dtype=float)),
        ),
    )
    return ForwardProfile(
        operator=operator, lattice=machine.lattice, newton_steps=NEWTON_STEPS
    )


#: Interval of the soft centroid pins, in metres.
CENTROID_PIN_INTERVAL = 0.01


def centroid_pins(group, row: int):
    """Return soft radial and vertical current-centroid pins for one slice.

    The targets are the current centroid stored in the shot's own EFIT record,
    used only as a position hint for a plausible state, each with a 1 cm
    deterministic interval. They hold the plasma where the shot put it when
    the unconstrained solve selects a saddle that does not bound the plasma.
    """
    from nova.equilibrium.observation import (
        ConstraintPinSet,
        MomentIntegralSupport,
        MomentPin,
        PinUncertainty,
    )

    interval = PinUncertainty(
        CENTROID_PIN_INTERVAL,
        "m",
        "EFIT current centroid of the slice, a soft position hint",
    )
    return ConstraintPinSet(
        moments=tuple(
            MomentPin(
                name,
                float(group[leaf][row]),
                interval,
                MomentIntegralSupport.CONFINED_CORE,
            )
            for name, leaf in (
                ("centroid_r", "current_centrd_r"),
                ("centroid_z", "current_centrd_z"),
            )
        )
    )


def solve_row(
    machine: PulseMachine,
    profile,
    group,
    shot: int,
    row: int,
    pin_centroid: bool = False,
) -> dict[str, object]:
    """Solve one EFIT slice on the pulse profile and return its terminal state."""
    from dataclasses import replace

    import jax.numpy as jnp
    from scipy.interpolate import RectBivariateSpline

    from benchmarks.efit_forward_parity_slice import _stored_map
    from nova.equilibrium.solve_request import ExplicitSolveSeed, ForwardSolveRequest

    source = row_source(group, row)
    stored_r, stored_z, stored_flux = _stored_map(group, row)
    spline = RectBivariateSpline(stored_r, stored_z, stored_flux, kx=3, ky=3, s=0.0)
    seed = np.r_[
        spline(machine.radius, machine.height).ravel(),
        spline.ev(machine.wall[:, 0], machine.wall[:, 1]),
    ]
    target_current = abs(float(group["plasma_current_c"][row]))
    request = replace(
        ForwardSolveRequest.from_defaults(
            carrier_identity=f"mast-pulse:{shot}",
            source_profile=source,
            seed_policy=ExplicitSolveSeed(jnp.asarray(seed)),
            policy_overrides={
                "newton_steps": NEWTON_STEPS,
                "gmres_iterations": GMRES_ITERATIONS,
                "kernel_tolerance": FIXED_POINT_CRITERION,
                "qualification_tolerance": FIXED_POINT_CRITERION,
            },
            target_current=target_current,
        ),
        prescribed_current=jnp.asarray(np.asarray(group["fcoil_c"][row], dtype=float)),
        # The default request appends an INACTIVE vertical pin. Any constraint
        # pair routes the solve through the augmented-constraint path, whose
        # outer scan runs eagerly with the operator closed in, so every slice
        # compiles its own program with the Green blocks as constants. With
        # the inactive pin omitted the same Newton-Krylov route runs as one
        # jitted program taking the operator as an argument, compiled once
        # for the pulse.
        vertical_pin="omit",
        constraint_pins=centroid_pins(group, row) if pin_centroid else None,
    )
    started = time.perf_counter()
    receipt = profile.solve(request)
    wall_seconds = time.perf_counter() - started
    equilibrium = receipt.equilibrium
    topology = equilibrium.topology
    nodes = machine.lattice.node_count
    state = np.asarray(equilibrium.flux, dtype=float)
    cell_current = np.asarray(equilibrium.cell_current, dtype=float)
    return {
        "row": row,
        "time_s": float(group["time"][row]),
        "flux": state[:nodes].reshape(machine.radius.size, machine.height.size),
        "wall_flux": state[nodes:],
        "cell_current": cell_current,
        "axis": np.asarray(topology.axis, dtype=float),
        "axis_flux": float(topology.axis_flux),
        "boundary_flux": float(topology.boundary_flux),
        "x_point": np.asarray(topology.x_point, dtype=float),
        "x_point_flux": float(topology.x_point_flux),
        "wall_point": np.asarray(topology.wall_point, dtype=float),
        "wall_point_flux": float(topology.wall_point_flux),
        "diverted": bool(topology.diverted),
        "residual": float(equilibrium.fixed_point.residual),
        "qualified": bool(receipt.qualified),
        "termination": int(np.asarray(receipt.termination_reason)),
        "plasma_current": float(cell_current.sum()),
        "target_current": target_current,
        "wall_seconds": wall_seconds,
    }


SCALARS = (
    "row",
    "time_s",
    "axis_flux",
    "boundary_flux",
    "x_point_flux",
    "wall_point_flux",
    "diverted",
    "residual",
    "qualified",
    "plasma_current",
    "target_current",
    "wall_seconds",
    "centroid_pinned",
)


def write_stack(path: Path, machine: PulseMachine, rows: list[dict]) -> None:
    """Persist the solved rows and the shared grid atomically."""
    pending = path.with_suffix(".pending.npz")
    arrays = {
        "radius": machine.radius,
        "height": machine.height,
        "material": machine.material,
        "limiter": machine.limiter,
        "wall": machine.wall,
        "flux": np.stack([item["flux"] for item in rows]),
        "wall_flux": np.stack([item["wall_flux"] for item in rows]),
        "cell_current": np.stack([item["cell_current"] for item in rows]),
        "axis": np.stack([item["axis"] for item in rows]),
        "x_point": np.stack([item["x_point"] for item in rows]),
        "wall_point": np.stack([item["wall_point"] for item in rows]),
        "termination": np.asarray([item["termination"] for item in rows]),
    }
    arrays |= {name: np.asarray([item[name] for item in rows]) for name in SCALARS}
    np.savez(pending, **arrays)
    pending.replace(path)


def main(argv: list[str] | None = None) -> int:
    """Solve the pulse and write the stack beside a JSON receipt."""
    import zarr

    from nova.database.filepath import derived_cache_root, mast_level1_shots
    from nova.jax.config import (
        configure_dtypes,
        configure_persistent_compilation_cache,
        default_persistent_compilation_cache_root,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, default=27079)
    parser.add_argument("--pitch", type=float, default=DEFAULT_PITCH)
    parser.add_argument("--rows", type=int, nargs="*", default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--pin-centroid",
        action="store_true",
        help="hold each slice with soft radial and vertical current-centroid pins",
    )
    arguments = parser.parse_args(argv)

    configure_dtypes()
    import jax

    assert jax.config.jax_enable_x64 is True
    configure_persistent_compilation_cache(default_persistent_compilation_cache_root())
    group = zarr.open_group(mast_level1_shots() / f"{arguments.shot}.zarr", mode="r")[
        "efm"
    ]
    rows = arguments.rows or pulse_rows(group)[:: arguments.stride]
    tag = f"{arguments.shot}-pitch{round(1e3 * arguments.pitch)}mm"
    output = arguments.output or derived_cache_root() / "pulse-forward" / f"{tag}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)

    machine = PulseMachine(group, arguments.shot, arguments.pitch)
    print(json.dumps(machine.provenance(), default=str), flush=True)
    profile = pulse_profile(machine, group, rows)
    solved = []
    for row in rows:
        try:
            result = solve_row(
                machine,
                profile,
                group,
                arguments.shot,
                row,
                pin_centroid=arguments.pin_centroid,
            )
            result["centroid_pinned"] = bool(arguments.pin_centroid)
        except Exception as error:  # a refused slice is a row of the receipt
            print(json.dumps({"row": row, "error": repr(error)}), flush=True)
            continue
        solved.append(result)
        print(
            json.dumps(
                {name: result[name] for name in SCALARS}
                | {
                    "termination": result["termination"],
                    "axis": result["axis"].tolist(),
                    "wall_point": result["wall_point"].tolist(),
                }
            ),
            flush=True,
        )
        write_stack(output, machine, solved)
    receipt = {
        "shot": arguments.shot,
        "stack": str(output),
        "machine": machine.provenance(),
        "rows_requested": len(rows),
        "rows_solved": len(solved),
        "rows_qualified": int(sum(item["qualified"] for item in solved)),
    }
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2, default=str))
    print(json.dumps(receipt, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
