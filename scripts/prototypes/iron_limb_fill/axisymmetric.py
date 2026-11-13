"""Compare two axisymmetric stand-ins for six discrete iron limbs.

``smear`` keeps full iron and thins the section by the toroidal fill fraction,
so arms taper as 1/R; this is how the WEST machine description stores the
circuit. ``fill`` keeps the true poloidal outline and gives each cell the fill
fraction f. The poloidal field is tangent to the limb side faces, so H is
shared by iron and the air between limbs and the toroidal mean obeys

    M_mean = f M_iron(B_iron),   B_iron = B_mean + mu0 (1/f - 1) M_mean.

Both models run through the production mesh, volume-tested interaction and
B-H fit of nova.biot.magnetisation; f = 1 recovers the production residual.

Usage: axisymmetric.py linear --case <case.json> --output <dir> --sizes 0.1 0.05
       axisymmetric.py west --source <hex.pkl> --output <dir> --size 0.08
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import time

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.scipy.linalg as jla  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402

from nova.biot import magnetisation as material  # noqa: E402
from nova.biot.tiledassembly import TilePlan  # noqa: E402

MU0 = 4e-7 * np.pi
PLAN = TilePlan(128, 128, 256, 16, 48)
COLUMN_RADIUS, ARM_HALF_WIDTH, ARM_Z, LIMB_R = 0.664, 0.4, (2.4, 3.6), (4.8, 6.0)
WEST_JOIN = 0.7639
# The interaction kernels are singular on the axis; the stored circuit has this bore.
BORE = 0.05
WEST_MACHINE = Path(
    "/home/ITER/mcintos/Code/imas-efit/tests/data/imas/WEST/machine/dd4-1-0"
)


def limb_fraction(radius):
    """Return the toroidal fraction of six 0.8 m wide limbs at a radius."""
    radius = np.asarray(radius, dtype=float)
    reach = np.maximum(radius, 2 * ARM_HALF_WIDTH)
    return np.where(
        radius <= 2 * ARM_HALF_WIDTH, 1.0, 6 / np.pi * np.arcsin(ARM_HALF_WIDTH / reach)
    )


def mirrored(polygon):
    return shapely.union_all(
        [polygon, shapely.affinity.scale(polygon, 1, -1, origin=(0, 0))]
    )


def linear_regions():
    """Return the fill and smear regions of the WEST-like control circuit."""
    column = shapely.box(BORE, -ARM_Z[0], COLUMN_RADIUS, ARM_Z[0])
    arm = mirrored(shapely.box(BORE, ARM_Z[0], LIMB_R[1], ARM_Z[1]))
    limb = shapely.box(LIMB_R[0], -ARM_Z[0], LIMB_R[1], ARM_Z[0])
    fill = shapely.union_all([column, arm, limb])
    radius = np.linspace(LIMB_R[0], LIMB_R[1], 2001)
    moment = np.trapezoid(limb_fraction(radius) * radius, radius)
    outer = np.sqrt(LIMB_R[0] ** 2 + 2 * moment)
    taper = np.r_[np.linspace(2 * ARM_HALF_WIDTH, outer, 200)]
    top = ARM_Z[0] + (ARM_Z[1] - ARM_Z[0]) * limb_fraction(taper)
    slab = shapely.Polygon(
        [
            (BORE, ARM_Z[0]),
            (outer, ARM_Z[0]),
            *zip(taper[::-1], top[::-1]),
            (BORE, ARM_Z[1]),
        ]
    )
    thin = shapely.box(LIMB_R[0], -ARM_Z[0], outer, ARM_Z[0])
    smear = shapely.union_all([column, mirrored(slab), thin])
    return fill, smear, outer


def linear_fill(centroids):
    outside = (np.abs(centroids[:, 1]) > ARM_Z[0]) | (centroids[:, 0] > LIMB_R[0])
    return np.where(outside, limb_fraction(centroids[:, 0]), 1.0)


def limb_demagnetising(centroids, arm_floor, limb_face, factor=0.4):
    """Return (N_R, N_Z) for field across a limb of 0.8 m by 1.2 m section.

    Along its own axis a limb shares H with the gaps beside it, so N = 0. Across
    it, in the poloidal plane, the bar's own poles oppose the field; the factor
    width / (width + depth) is the elliptic-section estimate.
    """
    arm = np.abs(centroids[:, 1]) > arm_floor
    limb = centroids[:, 0] > limb_face
    result = np.zeros((len(centroids), 2))
    result[arm & ~limb, 1] = factor
    result[limb & ~arm, 0] = factor
    return result


def residual(state, operands):
    """Return the toroidal-mean material residual for per-cell fill fraction."""
    gram, external, knots, coefficients, ends, fill, gain = operands
    total = (external + gram @ state).reshape(2, -1)
    iron = total + MU0 * gain * state.reshape(2, -1)
    squared = jnp.sum(iron**2, axis=0)
    magnitude = jnp.sqrt(jnp.where(squared > 0, squared, 1.0))
    magnitude = jnp.where(squared > 0, magnitude, 0.0)
    coefficient = material._coefficient(magnitude, knots, coefficients, ends)
    return state - (fill * coefficient * iron).reshape(-1)


@jax.jit
def _step(state, operands, damping):
    value = residual(state, operands)
    jacobian = jax.jacfwd(residual)(state, operands)
    jacobian += damping * jnp.eye(state.size, dtype=state.dtype)
    candidate = state + jla.solve(jacobian, -value)
    return candidate, residual(candidate, operands)


def newton(operands, reference, tolerance=1e-8, budget=40):
    """Solve by Newton with the production's adaptive diagonal regularisation."""
    state = jnp.zeros(operands[1].size, dtype=jnp.float64)
    history, damping = [], 1e-6
    for _ in range(budget + 1):
        value = residual(state, operands)
        norm = float(jnp.linalg.norm(value))
        history.append(norm / max(float(jnp.linalg.norm(state)), reference))
        print(f"NEWTON residual={history[-1]:.3e}", flush=True)
        if history[-1] <= tolerance:
            return state, history, True
        for attempt in range(35):
            shift = 0.0 if attempt == 0 else damping
            candidate, trial = _step(state, operands, jnp.asarray(shift))
            defect = trial + shift * (candidate - state)
            ratio = 1 - (float(jnp.linalg.norm(defect)) / norm) ** 2
            if ratio > 0.25:
                state = candidate
                if shift > 0:
                    damping *= 0.3 if ratio > 0.9 else 2.0 if ratio < 0.5 else 1.0
                break
            if attempt > 0:
                damping *= 10
        else:
            return state, history, False
    return state, history, False


class Model:
    """One meshed region with its interaction, ready for several laws."""

    def __init__(self, region, size, external, fill, demagnetising=None):
        started = time.perf_counter()
        self.cells = material._mesh(region, size)
        self.fill = np.asarray(fill(self.cells.centroids), dtype=float)
        factor = (
            np.zeros((len(self.fill), 2))
            if demagnetising is None
            else np.asarray(demagnetising(self.cells.centroids), dtype=float)
        )
        # B_iron = B_mean + mu0 gain M_mean, per (R, Z) component.
        self.gain = ((1 - self.fill) / self.fill)[None, :] * (1 - factor.T)
        self.external = material._average_external(self.cells, external)
        self.gram = np.asarray(material._galerkin_interaction(self.cells, PLAN))
        assert np.isfinite(self.gram).all()
        self.size = size
        print(
            f"MODEL cells={len(self.cells.polygons)} size={size} "
            f"seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )

    def linear(self, mu):
        """Solve constant iron permeability; the cell mean has 1 + f (mu - 1)."""
        kappa = (1 - 1 / mu) / MU0
        coefficient = (
            self.fill * kappa / (1 - self.fill * kappa * MU0 * self.gain)
        ).reshape(-1)
        drive = self.external.T.reshape(-1)
        # Host LAPACK: the dense factor of a fine mesh exceeds a 16 GB device.
        system = np.eye(drive.size) - coefficient[:, None] * self.gram
        return np.linalg.solve(system, coefficient * drive)

    def nonlinear(self, b, mu):
        operands = tuple(
            jnp.asarray(v, dtype=jnp.float64)
            for v in (
                self.gram,
                self.external.T.reshape(-1),
                *material._fit_law(b, mu),
                np.tile(self.fill, 2).reshape(2, -1),
                self.gain,
            )
        )
        reference = max(np.linalg.norm(self.external) / MU0, 1e-30)
        state, history, converged = newton(operands, reference)
        return np.asarray(state), history, converged

    def mean_field(self, state):
        """Return the cell-mean (B_R, B_Z) of the toroidal-mean field."""
        return (self.external.T.reshape(-1) + self.gram @ state).reshape(2, -1).T

    def iron_field(self, state):
        mean = self.mean_field(state)
        return mean + MU0 * self.gain.T * state.reshape(2, -1).T

    def probe(self, state, points):
        response = material._interaction(self.cells, points, PLAN)
        return np.asarray(response @ state).reshape(2, -1).T

    def column_flux(self, state, slab=0.2):
        """Return the flux through the central column, averaged over a slab."""
        c = self.cells.centroids
        core = (c[:, 0] < COLUMN_RADIUS) & (c[:, 1] > 0) & (c[:, 1] < slab)
        volume = 2 * np.pi * c[core, 0] * self.cells.areas[core]
        return float((self.mean_field(state)[core, 1] * volume).sum() / slab)


def axisymmetric_drive(sources):
    from nova.biot.greens import greens_bz_br

    radius, height, amps = (np.asarray(sources[k]) for k in ("r", "z", "amps"))

    def evaluate(points):
        result = np.empty_like(points)
        for start in range(0, len(points), 256):
            stop = start + 256
            bz, br = greens_bz_br(
                points[start:stop, 0, None], points[start:stop, 1, None], radius, height
            )
            result[start:stop] = np.column_stack((br @ amps, bz @ amps))
        return result

    return evaluate


def run_linear(args):
    case = json.loads(args.case.read_text())
    external = axisymmetric_drive(case["sources"])
    probes = np.asarray(case["probes"])
    fill_region, smear_region, outer = linear_regions()
    record = dict(probes=probes.tolist(), smear_outer_radius=outer, rows=[])
    for size in args.sizes:
        for name, region, fill, shape in (
            ("fill", fill_region, linear_fill, None),
            (
                "tensor",
                fill_region,
                linear_fill,
                lambda c: limb_demagnetising(c, ARM_Z[0], LIMB_R[0]),
            ),
            ("smear", smear_region, lambda c: np.ones(len(c)), None),
            ("solid", fill_region, lambda c: np.ones(len(c)), None),
        ):
            if name not in args.models:
                continue
            model = Model(region, size, external, fill, shape)
            for mu in args.mu:
                state = model.linear(mu)
                record["rows"].append(
                    dict(
                        model=name,
                        size=size,
                        mu=mu,
                        cells=len(model.cells.polygons),
                        iron_T=model.probe(state, probes).tolist(),
                        column_flux_Wb=model.column_flux(state),
                    )
                )
                print(
                    f"LINEAR {name} size={size} mu={mu:g} "
                    f"flux={record['rows'][-1]['column_flux_Wb']:.5f}",
                    flush=True,
                )
                (args.output / args.record).write_text(
                    json.dumps(record, indent=1) + "\n"
                )
            del model
            jax.clear_caches()


def west_regions():
    from nova.imas.machine import StaticMachineDescription

    machine = StaticMachineDescription.read_iron_core(
        f"imas:hdf5?path={WEST_MACHINE}", dd_version="4.1.0"
    )
    segments = [
        shapely.Polygon(np.column_stack((s.outline.r, s.outline.z)))
        for s in machine.iron_segments
    ]
    table = machine.iron_segments[0].material
    stored = shapely.union_all(segments)
    lower, upper = segments[2].bounds[1], segments[2].bounds[3]
    limb = segments[3].bounds
    arm = mirrored(shapely.box(0.05, lower, 6.0, upper))
    flat = shapely.union_all(
        [
            segments[0],
            segments[1],
            segments[5],
            arm,
            shapely.box(limb[0], limb[1], 6.0, limb[3]),
        ]
    )
    return (
        stored,
        flat,
        lower,
        limb[0],
        np.asarray(table.b_field),
        np.asarray(table.permeability_relative),
    )


def run_west(args):
    from scripts.west_iron_solve.measure import source_field

    with args.source.open("rb") as stream:
        data = pickle.load(stream)
    external = source_field(data)
    stored, flat, arm_floor, limb_face, b, mu = west_regions()

    def flat_fill(c):
        outside = (np.abs(c[:, 1]) > arm_floor) | (c[:, 0] > limb_face)
        return np.where(outside, np.minimum(1.0, WEST_JOIN / c[:, 0]), 1.0)

    angle = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    vessel = np.column_stack((2.45 + 0.75 * np.cos(angle), 0.75 * np.sin(angle)))
    stray = np.array(
        [
            [6.5, 0],
            [7.5, 0],
            [9.5, 0],
            [2.5, 4.0],
            [2.5, 4.5],
            [2.5, 5.5],
            [4.5, 4.0],
            [4.5, 5.0],
            [7.0, 3.0],
        ]
    )
    grid_r, grid_z = np.meshgrid(
        np.linspace(0.9, 10.0, 92), np.linspace(-6.0, 6.0, 121)
    )
    grid = np.column_stack((grid_r.ravel(), grid_z.ravel()))
    probes = np.vstack((vessel, stray))
    record = dict(
        time_s=data["time"],
        vessel=vessel.tolist(),
        stray=stray.tolist(),
        air_T=external(probes).tolist(),
        rows=[],
    )
    np.save(args.output / "grid.npy", grid)
    np.save(args.output / "grid-air.npy", external(grid))
    for name, region, fill, shape in (
        ("stored", stored, lambda c: np.ones(len(c)), None),
        ("fill", flat, flat_fill, None),
        (
            "tensor",
            flat,
            flat_fill,
            lambda c: limb_demagnetising(c, arm_floor, limb_face),
        ),
    ):
        if name not in args.models:
            continue
        model = Model(region, args.size, external, fill, shape)
        state, history, converged = model.nonlinear(b, mu)
        iron = np.linalg.norm(model.iron_field(state), axis=1)
        moment = np.linalg.norm(state.reshape(2, -1).T, axis=1) / model.fill
        relative = 1 / (1 - MU0 * moment / np.maximum(iron, 1e-300))
        record["rows"].append(
            dict(
                model=name,
                size=args.size,
                cells=len(model.cells.polygons),
                converged=converged,
                residual_history=history,
                iron_T=model.probe(state, probes).tolist(),
                column_flux_Wb=model.column_flux(state),
                mu_range=[float(relative.min()), float(relative.max())],
                iron_b_max_T=float(iron.max()),
            )
        )
        (args.output / args.record).write_text(json.dumps(record, indent=1) + "\n")
        np.savez(
            args.output / f"west-{name}.npz",
            centroids=model.cells.centroids,
            areas=model.cells.areas,
            fill=model.fill,
            state=state,
            iron_b=iron,
            mu=relative,
            grid_iron=model.probe(state, grid),
        )
        with (args.output / f"west-{name}-polygons.pkl").open("wb") as stream:
            pickle.dump(model.cells.polygons, stream)
        del model
        jax.clear_caches()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("linear", "west"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--sizes", type=float, nargs="+", default=[0.1, 0.05])
    parser.add_argument("--size", type=float, default=0.08)
    parser.add_argument("--record", default="record.json")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["fill", "tensor", "smear", "solid", "stored"],
    )
    parser.add_argument("--mu", type=float, nargs="+", default=[50.0, 200.0, 1000.0])
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    print(f"DEVICE={jax.devices()} MODULE={material.__file__}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (run_linear if args.mode == "linear" else run_west)(args)


if __name__ == "__main__":
    main()
