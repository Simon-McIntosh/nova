"""Export nova's bisected tie-island states as imas-efit version-2 trace records.

The tie-island states are the ones the merge-path admission probe bisects: the
carrier is built at a requested cell count, and the state exported is the moving
field evaluated at the bisected offset, where the read's boundary class or
saddle identity changes. Building the carrier is done exactly as
``saddle_admission_probe`` does it -- ``tie_probe.build(kind, requested, cache)``
-- against the same read-only cache, so the machine, field and bisected offset
are the probe's rather than a reconstruction of them.

The offset is **read out of a bisection receipt** produced by
``bisect_tie_island.py`` rather than passed on the command line, so a state
cannot be exported at an offset nobody bisected. A receipt whose bracket is not
adjacent binary64, or whose two ends agree, is refused: it names an interval
rather than a state. The provenance travels with the record as a sidecar JSON
carrying the bracket, the step count and the decision keys on either side.

What is written is a plain ``BOUNDARY_LEVEL_TRACE 2`` record: a rectilinear grid
over the machine, the moving field evaluated on it, the machine's wall as
segments and a polygon, and nova's own decision -- axis level and position, the
admitted saddle, the boundary level and the limited-or-diverted class -- in the
record's decision fields. So a nova tie-island state becomes something imas-efit
can open with its own reader.

Usage: export_tie_islands.py <cache> <output-dir> <receipt> <cells> [<receipt> <cells>]
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import json
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scripts.prototypes.efit_field_swap.trace_record import (
    TraceRecord,
    write_trace_record,
)
from scripts.prototypes.fast_map_production import tie_probe
from scripts.prototypes.fast_map_production.saddle_admission_probe import (
    measure,
    sampler_with_enclosure,
)

DIVERTED = 2
LIMITED = 1
GRID = 65


def moving_field(state, offset):
    """The probe's moving field at ``offset``, in the same units it bisects in."""
    shifted = jnp.asarray(offset * state["scale"])
    exterior = state["jets"].exterior + shifted * state["jet_slope"]
    return replace(state["jets"], exterior=exterior).field(state["reference"])


def wall_geometry(geometry):
    """The machine's wall as (nwall, 2) segments plus its ordered ring."""
    points = np.asarray(geometry.wall_points, dtype=np.float64)
    following = np.asarray(geometry.wall_following)
    r = np.column_stack((points[:, 0], points[following, 0]))
    z = np.column_stack((points[:, 1], points[following, 1]))
    return r, z, points


def render(field, geometry, grid=GRID):
    """Evaluate the field on a rectilinear grid covering the machine."""
    centre = np.asarray(geometry.centre)
    pitch = float(np.median(np.asarray(geometry.pitch)))
    pad = 3.0 * pitch
    r0, z0 = centre.min(axis=0) - pad
    r1, z1 = centre.max(axis=0) + pad
    rgrid = np.linspace(r0, r1, grid)
    zgrid = np.linspace(z0, z1, grid)
    points = np.stack(np.meshgrid(rgrid, zgrid, indexing="ij"), axis=-1)
    psi = np.asarray(jax.vmap(field.value)(jnp.asarray(points.reshape(-1, 2)))).reshape(
        grid, grid
    )
    return rgrid, zgrid, psi


def export(state, offset, out_dir, tag, sampled, expected_key, key_name):
    geometry = state["geometry"]
    field = moving_field(state, offset)
    convention = state["convention"]
    measured = measure(offset, state, sampled)
    actual_key = decision_key(measured, key_name)
    if actual_key != expected_key:
        raise SystemExit(
            f"export read at {offset!r} has key {actual_key}, "
            f"receipt records {expected_key}"
        )
    _, reading, *_ = sampled(jnp.asarray(offset * state["scale"]))
    jax.block_until_ready(reading.boundary_flux)
    sigma = float(convention.sigma)

    axis = np.asarray(reading.axis)
    saddle = np.asarray(reading.boundary)
    axis_psi = float(sigma * field.evaluate(jnp.asarray(axis)).value)
    saddle_psi = float(sigma * field.evaluate(jnp.asarray(saddle)).value)
    wall_level = float(
        np.max(
            sigma
            * jax.vmap(lambda p: field.evaluate(p).value)(
                jnp.asarray(geometry.wall_points)
            )
        )
    )
    diverted = bool(np.asarray(reading.boundary_class))
    rgrid, zgrid, psi = render(field, geometry)
    wall_r, wall_z, ring = wall_geometry(geometry)
    record = TraceRecord(
        slice=0,
        iteration=0,
        verdict=DIVERTED if diverted else LIMITED,
        saddle_source_node=0,
        wall_source_node=0,
        wall_source_segment=0,
        wall_axis_connected=True,
        handover_reverted=False,
        polarity=1,
        sweep_armed=False,
        axis_psi=axis_psi,
        axis_r=float(axis[0]),
        axis_z=float(axis[1]),
        saddle_contested=0.0,
        saddle_refined=saddle_psi,
        wall_level=wall_level,
        fed_back_psi=float(np.asarray(reading.boundary_flux)),
        rgrid=rgrid,
        zgrid=zgrid,
        wall_r=wall_r,
        wall_z=wall_z,
        wall_index=np.asarray([[1, len(wall_r)]], dtype=np.int32),
        cell_r=np.zeros(0),
        cell_z=np.zeros(0),
        polygon_r=ring[:, 0],
        polygon_z=ring[:, 1],
        psi=psi,
    )
    target = Path(out_dir) / f"{tag}-nova-{'diverted' if diverted else 'limited'}.txt"
    write_trace_record(target, record)
    print(
        "EXPORTED %s cells=%d offset=%r class=%d axis=(%.5f,%.5f) boundary=%.6f"
        % (
            target.name,
            len(centre_count(geometry)),
            offset,
            int(diverted),
            axis[0],
            axis[1],
            float(np.asarray(reading.boundary_flux)),
        ),
        flush=True,
    )
    print(
        "   psi range %.5f .. %.5f  wall_level=%.6f"
        % (psi.min(), psi.max(), wall_level)
    )
    return target


def centre_count(geometry):
    return geometry.centre


def bisected_brackets(receipt, path):
    """The bisected offset a receipt records, refusing anything but a state.

    The offset comes from the receipt rather than from a command-line argument,
    and the bracket must be adjacent binary64 and must change the decision key.
    A receipt failing either test names an interval or a non-transition, and
    neither is a tie-island state.
    """
    entries = receipt.get("bisected", [])
    if not entries:
        raise SystemExit(f"{path} records no bisected bracket")
    for entry in entries:
        bracket = entry.get("bracket", [])
        if len(bracket) != 2:
            raise SystemExit(f"{path} has a bracket without two endpoints")
        left, right = map(float, bracket)
        adjacent = bool(np.isfinite(left) and np.nextafter(left, np.inf) == right)
        left_key = decision_key(entry["left"], receipt["key"])
        right_key = decision_key(entry["right"], receipt["key"])
        changed = left_key != right_key
        if not adjacent or not changed:
            raise SystemExit(
                f"{path} is not an adjacent binary64 decision transition: "
                f"bracket={bracket!r}, keys={left_key!r}/{right_key!r}"
            )
        if entry.get("adjacent") is not True or entry.get("unchanged") is not False:
            raise SystemExit(f"{path} receipt flags contradict its endpoints or keys")
        yield left, right, entry


def bisected_offset(receipt, path):
    """The left endpoint of the first validated transition."""
    left, _, entry = next(bisected_brackets(receipt, path))
    return left, entry


def decision_key(row, key):
    """A bisection row's decision key as a tuple, whatever key was bisected on."""
    if key == "both":
        return (int(row["boundary_class"]), int(row["saddle_identity"]))
    return (row[key],)


def validate_carrier(receipt, requested, achieved):
    """Require both receipt cell counts to describe the built carrier."""
    if (receipt.get("requested_cells"), receipt.get("achieved_cells")) != (
        requested,
        achieved,
    ):
        raise SystemExit(
            f"carrier requested/achieved {requested}/{achieved} cells but "
            f"receipt records {receipt.get('requested_cells')}/"
            f"{receipt.get('achieved_cells')}"
        )


def main(cache, out_dir, jobs):
    """jobs: one (receipt-path, requested-cells) pair per exported state."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for receipt_path, requested in jobs:
        receipt = json.loads(Path(receipt_path).read_text())
        state = tie_probe.build(receipt["kind"], requested, cache)
        achieved = len(centre_count(state["geometry"]))
        validate_carrier(receipt, requested, achieved)
        tag = f"{receipt['kind']}-{achieved}cells"
        sampled = sampler_with_enclosure(state)
        for index, (left, right, entry) in enumerate(
            bisected_brackets(receipt, receipt_path), start=1
        ):
            for side, offset, row in (
                ("left", left, entry["left"]),
                ("right", right, entry["right"]),
            ):
                expected_key = decision_key(row, receipt["key"])
                target = export(
                    state,
                    offset,
                    out_dir,
                    f"{tag}-edge{index}-{side}",
                    sampled,
                    expected_key,
                    receipt["key"],
                )
                target.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "record": target.name,
                            "kind": receipt["kind"],
                            "requested_cells": requested,
                            "achieved_cells": achieved,
                            "offset": offset,
                            "side": side,
                            "key": receipt["key"],
                            "bracket": entry["bracket"],
                            "steps": entry["steps"],
                            "left_key": list(
                                decision_key(entry["left"], receipt["key"])
                            ),
                            "right_key": list(
                                decision_key(entry["right"], receipt["key"])
                            ),
                            "exported_key": list(expected_key),
                        },
                        indent=2,
                    )
                    + "\n"
                )
    return 0


if __name__ == "__main__":
    # Each carrier is named by its own receipt and its own requested cell count,
    # so a receipt for one cell count can never be applied to another's carrier.
    cache, out_dir = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    if len(rest) % 2:
        raise SystemExit(
            "usage: <cache> <out-dir> <receipt> <cells> [<receipt> <cells>]"
        )
    jobs = [(rest[index], int(rest[index + 1])) for index in range(0, len(rest), 2)]
    sys.exit(main(cache, out_dir, jobs))
