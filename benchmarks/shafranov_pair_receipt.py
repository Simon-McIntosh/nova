"""Impose the Shafranov row on the bank rows against the profiles' own moments.

Each bank row is read for the combination the extracted profiles imply,
``beta_p + l_i/2`` from the moment observation on the row's own reference
state and plasma current.  That value is then imposed as a constraint target
on the external-magnetics Shafranov row, whose compensating unknown is the
profile normalisation the source term already carries.  The receipt records,
per row, the combination the row achieved, the compensating fraction, the
outer step count, the terminal residual and the converged flag, so an unmet
target is reported rather than fitted away.  A row the solve refuses has no
terminal state at all, so its terminal combination is null: the target it was
refused against is not an outcome and is never recorded as one.

The per-row receipts can be rebuilt from a banked lane log, which carries one
``SHAFRANOV-ROW`` emission per row, with ``--emissions``: the replay takes
every value from the banked emissions and never enters the solver, so a
corrected field is republished without a new solve or a new job.

One poloidal panel per row is written under the project's plotting rules:
line contours only, no axes or grid, both the reference and the terminal
state's nulls drawn in their own styles, and the wall on every panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
from typing import Any
from dataclasses import replace

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from benchmarks import settled_mask_stall as settled
from nova.equilibrium.constraint import (
    ConstraintBinding,
    ConstraintContext,
    ConstraintPair,
    ExternalShafranovConstraint,
    ProfileAmplitudeUnknown,
)
from nova.equilibrium.source import (
    PolynomialFluxFunction,
    project_domain_profile,
)
from nova.equilibrium.topology import NoQualifiedAxisError, TopologyClass
from nova.equilibrium.wall_mask import WallUnit
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT / "docs/figures/constraint-augmented-newton-krylov/shafranov"
#: Row tolerance on the combination, stated in the row's own physical scale.
ROW_TOLERANCE = 1.0e-6
#: Display raster resolution for the per-row panels.
RASTER_SAMPLES = 181
#: Fields of one row receipt, in the order the document is written.  A receipt
#: replayed from a banked emission is assembled through this order, so a replay
#: and a fresh measurement agree on the document shape as well as the values.
ROW_FIELDS = (
    "identity",
    "status",
    "refusal_reason",
    "target_combination",
    "minor_radius_m",
    "plasma_current_a",
    "target_source",
    "observed_combination_at_reference",
    "reference_combination_gap",
    "target_error",
    "achieved_combination",
    "compensating_amplitude_fraction",
    "terminal_profile_combination",
    "outer_steps",
    "terminal_residual",
    "topology_consistent",
    "converged",
    "termination",
)
#: The prefix the lane prints one row emission under, which is what a replay
#: reads a banked log for.  ``ast.literal_eval`` is not needed: the emission is
#: the same JSON payload the receipt document carries.
EMISSION_PREFIX = "SHAFRANOV-ROW "


def _strict_float(value: Any) -> float | None:
    """Return a finite float or ``None`` so JSON carries no NaN."""
    if value is None:
        return None
    result = float(np.asarray(value))
    return result if np.isfinite(result) else None


def _terminal_combination(status: str, combination: Any) -> float | None:
    """Return the terminal combination a row receipt may carry.

    A refused row has no terminal state, so the target it was refused against
    is not an outcome of the solve and must not be recorded as one: the refusal
    is stated in ``status`` and ``refusal_reason``, and this field is null.  A
    row that did terminate reports the combination its own profiles read at the
    terminal state, which may differ from the target by the row's residual.
    """
    if status == "refused":
        return None
    return _strict_float(combination)


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _selection() -> dict[tuple[int, int], Any]:
    return {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in settled.select_slices_by_shot(
            settled.DECOMPOSITION_BANK
        )
    }


def _minor_radius(boundary: np.ndarray) -> float:
    """Return the reference boundary's geometric minor radius [m].

    The stored last closed flux surface is the geometry the row is stated
    against, so its own horizontal half-width is the radius that enters the
    Shafranov logarithm.  A figure of the plasma, not an operand: the row
    carries it in the payload and reports it beside every receipt.
    """
    points = np.asarray(boundary, dtype=float).reshape(-1, 2)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < 2:
        raise ValueError("the stored boundary carries too few finite nodes")
    return 0.5 * float(np.ptp(points[:, 0]))


def _combination(profile, flux, target_current) -> float:
    """Return ``beta_p + l_i/2`` the extracted profiles imply at one state."""
    observation = profile.integral_observation(flux, target_current)
    beta = float(np.asarray(observation.poloidal_beta))
    inductance = float(np.asarray(observation.internal_inductance))
    return beta + 0.5 * inductance


def _external_image(profile) -> np.ndarray:
    """Return the prescribed conductor flux on the lattice's own nodes [Wb]."""
    prescribed = profile.operator.prescribed_current_field
    if prescribed is None:
        raise RuntimeError("the row needs a prescribed conductor field")
    flux = np.asarray(prescribed.flux(), dtype=float)
    return flux[: profile.lattice.node_count]


def _pair(profile, *, target: float, minor_radius: float) -> tuple[Any, np.ndarray]:
    """Return the Shafranov row and the external image it is stated against."""
    external = _external_image(profile)
    functional = ExternalShafranovConstraint(
        minor_radius=jnp.asarray(minor_radius),
    )
    binding = ConstraintBinding(
        target=jnp.atleast_1d(jnp.asarray(target)),
        tolerance=jnp.asarray([ROW_TOLERANCE]),
        scale=jnp.asarray([1.0]),
        initial_unknown=jnp.asarray([0.0]),
        payload=(jnp.asarray(external), jnp.asarray(minor_radius)),
        policy="imposed",
    )
    unknown = ProfileAmplitudeUnknown("pressure_gradient", jnp.asarray([1.0]))
    return ConstraintPair(functional, unknown, binding), external


def _amplitude_availability(profile, component: str) -> str | None:
    """Return the refusal text when the amplitude is not a free unknown.

    The compensator moves the state through the source term's own scalar
    normalisation, and only a polynomial flux function carries one.  An
    extracted profile is an arbitrary callable, so the row states a constraint
    nothing on that row can move and must be refused rather than fitted.
    """
    field = "p_prime" if component == "pressure_gradient" else "ff_prime"
    function = getattr(profile.source.core, field, None)
    if isinstance(function, PolynomialFluxFunction):
        return None
    return (
        f"the source core's {field} is "
        f"{type(function).__name__}, not a PolynomialFluxFunction, so the "
        "profile amplitude is not a free unknown of this solve"
    )


def _observed_combination(profile, pair, flux) -> float:
    """Return the combination the row itself reads at one state."""
    context = ConstraintContext(jnp.asarray(flux), None, None, None)
    observed = pair.functional.observed(profile, context, pair.binding.payload)
    return float(np.asarray(jnp.atleast_1d(observed))[0])


def _topology(operator, state) -> dict[str, Any]:
    """Return the read nulls and boundary, or a recorded refusal."""
    try:
        _masks, topology = operator.read(jnp.asarray(state))
    except NoQualifiedAxisError as error:
        return {"read_status": "no_qualified_axis", "exception_text": str(error)}
    diverted = bool(np.asarray(topology.diverted))
    return {
        "read_status": "qualified",
        "class": str(TopologyClass.DIVERTED if diverted else TopologyClass.LIMITED),
        "boundary_rz_m": np.asarray(topology.boundary, dtype=float).tolist(),
        "axis_rz_m": np.asarray(topology.axis, dtype=float).reshape(-1)[:2].tolist(),
        "x_point_rz_m": np.asarray(topology.x_point, dtype=float)
        .reshape(-1, 2)
        .tolist(),
    }


def _wall_units(operator) -> tuple[Any, ...]:
    """Return the operator's wall as its own typed units.

    The wall is stored flat with unit offsets and per-unit closure and kind,
    so a panel can draw every unit on its own terms rather than one invented
    ring: an open material unit is dashed and is never joined to a neighbour.
    """
    coordinate = np.asarray(operator.wall.coordinate, dtype=float).reshape(-1, 2)
    offsets = np.asarray(operator.wall_unit_offsets, dtype=int)
    closed = np.asarray(operator.wall_unit_closed, dtype=bool)
    kinds = tuple(operator.wall_unit_kinds)
    return tuple(
        WallUnit(
            coordinate[start:stop, 0],
            coordinate[start:stop, 1],
            kind=kinds[index],
            closed=bool(closed[index]),
        )
        for index, (start, stop) in enumerate(
            zip(offsets[:-1], offsets[1:], strict=True)
        )
    )


def _raster(profile, state, units, *, samples: int = RASTER_SAMPLES):
    """Interpolate one state onto a display raster for line contours.

    The state vector is a stitched target: the plasma grid block first, then
    the wall nodes and any direct sample rows.  The lattice coordinate spans
    the grid block alone, so the grid prefix is the block to contour.  The
    raster is framed on the grid nodes and the wall together, so the vessel
    is not clipped out of its own panel.
    """
    points = np.asarray(profile.lattice.coordinate, dtype=float)
    field = np.asarray(state, dtype=float).reshape(-1)[: points.shape[0]]
    finite = np.all(np.isfinite(points), axis=1) & np.isfinite(field)
    points, field = points[finite], field[finite]
    if points.shape[0] < 3:
        raise ValueError("the state carries too few finite samples to contour")
    limits = np.vstack(
        (points, *[np.asarray(unit.vertices, dtype=float) for unit in units])
    )
    radial = np.linspace(
        float(np.min(limits[:, 0])), float(np.max(limits[:, 0])), samples
    )
    height = np.linspace(
        float(np.min(limits[:, 1])), float(np.max(limits[:, 1])), samples
    )
    radius_grid, height_grid = np.meshgrid(radial, height)
    raster = LinearNDInterpolator(points, field, fill_value=np.nan)(
        radius_grid, height_grid
    )
    return radial, height, np.asarray(raster, dtype=float)


def _render(
    profile, *, reference, terminal, units, path: Path, title: str, note: str
) -> dict:
    """Draw the reference and terminal states as shared-level line contours."""
    radial, height, reference_field = _raster(profile, reference, units)
    levels = poloidal.contour_levels(reference_field, count=12)
    reference_topology = _topology(profile.operator, reference)
    figure, axis = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    poloidal.draw_flux_contours(
        axis, radial, height, reference_field, levels, color="#3366cc"
    )
    drawn = [
        (
            reference_topology,
            DEFAULT_INK.variant(
                axis_color="#3366cc",
                xpoint_color="#3366cc",
                axis_marker="^",
                xpoint_marker="P",
            ),
        )
    ]
    if terminal is not None:
        _, _, terminal_field = _raster(profile, terminal, units)
        poloidal.draw_flux_contours(
            axis, radial, height, terminal_field, levels, color="#cc7722"
        )
        drawn.append(
            (
                _topology(profile.operator, terminal),
                DEFAULT_INK.variant(
                    axis_color="#cc7722",
                    xpoint_color="#cc7722",
                    axis_marker="^",
                    xpoint_marker="X",
                ),
            )
        )
    poloidal.draw_wall(axis, units=units)
    for topology, style in drawn:
        if topology.get("read_status") != "qualified":
            continue
        poloidal.draw_nulls(
            axis,
            magnetic_axis=topology["axis_rz_m"],
            x_points=np.asarray(topology["x_point_rz_m"], dtype=float),
            style=style,
            contain=units,
        )
    poloidal_axes(axis)
    axis.set_title(f"{title}\n{note}, shared levels", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return {
        "filesystem_path": str(path),
        "project_absolute_src": (
            f"/nova/figures/constraint-augmented-newton-krylov/shafranov/{path.name}"
        ),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _row_receipt(
    profile,
    *,
    identity: str,
    reference_state,
    target_current,
    minor_radius: float,
    requested,
    directory: Path,
) -> dict[str, Any]:
    """Impose the Shafranov row on one bank row and report the outcome.

    The row is stated against the reference boundary's own minor radius and
    the prescribed conductor image.  Where the profile amplitude is not a free
    unknown of the solve the row is refused, and the receipt records the
    refusal beside the gap between the combination the external magnetics
    imply and the combination the extracted profiles carry -- the number the
    row exists to close.
    """
    target = _combination(profile, reference_state, target_current)
    pair, _external = _pair(profile, target=target, minor_radius=minor_radius)
    refusal = _amplitude_availability(profile, "pressure_gradient")
    print(f"SHAFRANOV {identity} target={target!r} refusal={refusal!r}", flush=True)
    observed_at_reference = _observed_combination(profile, pair, reference_state)
    status = "refused" if refusal is not None else "imposed"
    entry: dict[str, Any] = {
        "identity": identity,
        "status": status,
        "refusal_reason": refusal,
        "target_combination": _strict_float(target),
        "minor_radius_m": _strict_float(minor_radius),
        "plasma_current_a": _strict_float(target_current),
        "target_source": (
            "poloidal_beta + internal_inductance/2 from "
            "ForwardProfile.integral_observation on the stored reference state"
        ),
        "observed_combination_at_reference": _strict_float(observed_at_reference),
        "reference_combination_gap": _strict_float(observed_at_reference - target),
        "target_error": None,
        "achieved_combination": None,
        "compensating_amplitude_fraction": None,
        "terminal_profile_combination": _terminal_combination(status, None),
        "outer_steps": 0,
        "terminal_residual": None,
        "topology_consistent": None,
        "converged": False,
        "termination": None,
    }
    terminal = None
    if refusal is None:
        branch = profile.solve_branch(
            jnp.asarray(reference_state),
            requested,
            target_current=target_current,
            constraint_pairs=(pair,),
        )
        equilibrium = branch.equilibrium
        flux = equilibrium.flux
        flux.block_until_ready()
        terminal = np.asarray(flux)
        records = list(equilibrium.constraints)
        record = records[0] if records else None
        terminal_combination = _combination(profile, flux, target_current)
        entry.update(
            {
                "target_error": (
                    None
                    if record is None
                    else _strict_float(abs(record.physical_residual[0]))
                ),
                "achieved_combination": (
                    None if record is None else _strict_float(record.observed[0])
                ),
                "compensating_amplitude_fraction": (
                    None
                    if record is None
                    else _strict_float(record.physical_unknown[0])
                ),
                "terminal_profile_combination": _terminal_combination(
                    status, terminal_combination
                ),
                "outer_steps": int(
                    np.asarray(equilibrium.fixed_point.active_set_iterations)
                ),
                "terminal_residual": _strict_float(branch.residual),
                "topology_consistent": bool(np.asarray(branch.topology_consistent)),
                "converged": bool(np.asarray(branch.converged)),
                "termination": settled._termination_name(
                    equilibrium.fixed_point.termination_reason
                ),
            }
        )
    units = _wall_units(profile.operator)
    slug = identity.replace("/", "-")
    note = (
        "reference state alone: the row is refused on this profile, so no "
        "terminal state exists"
        if refusal is not None
        else "reference blue / terminal orange"
    )
    entry["figure"] = _render(
        profile,
        reference=np.asarray(reference_state),
        terminal=terminal,
        units=units,
        path=directory / f"row-{slug}.png",
        title=f"MAST {identity}: beta_p + l_i/2 row",
        note=note,
    )
    return entry


def _row_path(directory: Path, identity: str) -> Path:
    """Return the receipt path one row identity writes to."""
    return directory / f"row-{identity.replace('/', '-')}.json"


def write_entry(directory: Path, entry: dict[str, Any]) -> None:
    """Write one row receipt beside its panel."""
    directory.mkdir(parents=True, exist_ok=True)
    _row_path(directory, entry["identity"]).write_text(
        json.dumps(entry, indent=2) + "\n", encoding="utf-8"
    )


def _emissions(path: Path) -> list[dict[str, Any]]:
    """Return the per-row emissions a banked lane log carries, in log order."""
    emissions = [
        json.loads(line[len(EMISSION_PREFIX) :])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(EMISSION_PREFIX)
    ]
    if not emissions:
        raise ValueError("the banked lane log carries no row emission")
    return emissions


def _figure_block(directory: Path, identity: str) -> dict[str, Any] | None:
    """Return the figure block of the row receipt already in place, if any.

    A replay does not re-render the panel, so it carries the committed block
    forward and the receipt keeps the provenance of the image it points at.
    """
    path = _row_path(directory, identity)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("figure")


def _emission_entry(
    emission: dict[str, Any], *, figure: dict[str, Any] | None
) -> dict[str, Any]:
    """Return one row receipt entry replayed from a banked row emission.

    Every value comes from the emission the lane printed; the terminal
    combination additionally passes the rule the measurement applies, so a
    refused row cannot carry the target it was refused against whatever the
    banked emission holds.
    """
    absent = [field for field in ROW_FIELDS if field not in emission]
    if absent:
        raise KeyError(f"the banked emission is missing {absent}")
    entry = {field: emission[field] for field in ROW_FIELDS}
    entry["terminal_profile_combination"] = _terminal_combination(
        entry["status"], entry["terminal_profile_combination"]
    )
    if figure is not None:
        entry["figure"] = figure
    return entry


def regenerate(*, emissions: Path, directory: Path) -> dict[str, Any]:
    """Rewrite the row receipts from a banked lane log, without a solve."""
    receipt_path = directory / "receipt.json"
    if not receipt_path.exists():
        raise FileNotFoundError(
            f"the aggregate receipt {receipt_path} must exist to be rewritten"
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    committed = [entry["identity"] for entry in receipt.get("rows_receipt", [])]
    entries = [
        _emission_entry(emission, figure=_figure_block(directory, emission["identity"]))
        for emission in _emissions(emissions)
    ]
    replayed = [entry["identity"] for entry in entries]
    if sorted(replayed) != sorted(committed):
        raise ValueError(
            "the banked log replays a different row set than the receipt holds: "
            f"{sorted(replayed)} against {sorted(committed)}"
        )
    for entry in entries:
        write_entry(directory, entry)
        print(
            "SHAFRANOV-STAMP "
            + json.dumps(
                {
                    key: entry[key]
                    for key in (
                        "identity",
                        "status",
                        "target_combination",
                        "terminal_profile_combination",
                        "observed_combination_at_reference",
                        "reference_combination_gap",
                        "refusal_reason",
                    )
                },
                sort_keys=True,
            ),
            flush=True,
        )
    receipt["rows"] = [
        [int(part) for part in entry["identity"].split("/")] for entry in entries
    ]
    receipt["rows_receipt"] = entries
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def measure(*, directory: Path, cache_root: Path | None = None) -> dict[str, Any]:
    """Impose the row on every row the decomposition bank qualifies."""
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
        if cache_root is None
        else cache_root
    )
    response_cache, carrier_evidence = settled._persisted_response_cache(
        settled.response_carrier.DEFAULT_CARRIER,
        settled.response_carrier.DEFAULT_RECEIPT,
    )
    selected = _selection()
    directory.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "receipt": "Shafranov beta_p + l_i/2 row imposed on the MAST bank rows",
        "row_set": "every row the decomposition bank qualifies; the receipt "
        "records the full selection rather than a stated count",
        "rows": [list(key) for key in sorted(selected)],
        "source": {
            "revision": _source_revision(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
        "configuration": {
            "route": "ForwardProfile.solve_branch public defaults",
            "constraint_policy": "imposed",
            "row": "ExternalShafranovConstraint: the external magnetics' "
            "large-aspect-ratio vertical-field identity inverted for "
            "beta_p + l_i/2 at the plasma's current centroid",
            "compensating_unknown": "ProfileAmplitudeUnknown on the "
            "pressure-gradient normalisation",
            "row_tolerance": ROW_TOLERANCE,
            "row_scale": 1.0,
            "prescribed_circuit_count": None,
            "persistent_compilation_cache": {
                "directory": str(cache.directory),
                "version": cache.version_key,
            },
        },
        "inputs": {"carrier_evidence": carrier_evidence},
        "rows_receipt": [],
    }
    for shot, row_index in sorted(selected):
        key = (shot, row_index)
        selected_row, qualification = selected[key]
        case, context = settled._mast_case_from_selection(
            settled.SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, policy = settled._passive_inclusive_case(
            case, context, response_cache
        )
        receipt["configuration"]["prescribed_circuit_count"] = int(
            getattr(profile.operator.prescribed_current_field, "circuit_count", 0)
        )
        reference_state = jnp.asarray(passive_case["state"])
        target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
        minor_radius = _minor_radius(np.asarray(case["boundary"], dtype=float))
        requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
        entry = _row_receipt(
            profile,
            identity=f"{shot}/{row_index}",
            reference_state=reference_state,
            target_current=target_current,
            minor_radius=minor_radius,
            requested=requested,
            directory=directory,
        )
        entry["plasma_current_a"] = target_current
        receipt["rows_receipt"].append(entry)
        (directory / "receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        write_entry(directory, entry)
        print(
            "SHAFRANOV-ROW "
            + json.dumps(
                {key: value for key, value in entry.items() if key != "figure"},
                sort_keys=True,
            ),
            flush=True,
        )
    (directory / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print("SHAFRANOV-DONE", flush=True)
    return receipt


PROJECTION_DIRECTORY = (
    ROOT / "docs/figures/constraint-augmented-newton-krylov/flux-function-fit"
)
#: The prefix the projection lane prints one row emission under.
PROJECTION_EMISSION_PREFIX = "FLUX-FIT-ROW "
#: Normalised-flux base the extracted gradients are sampled and fitted on.  It
#: is the bank's own declared extraction base, so the projection is stated
#: against the profiles the row is measured from rather than a re-grid.
PROJECTION_SAMPLES = 65
#: Highest polynomial order the projection may fit.
PROJECTION_MAXIMUM_ORDER = 6
#: Relative residual a fitted component must meet for its lowest order to stand.
PROJECTION_TOLERANCE = 1.0e-3
#: The compensator components whose scales are freed against the single scalar
#: row.  The row is one equation, so freeing both components in one solve is
#: singular; each is freed on its own solve and both are reported, which is
#: what tells a reader whether scale alone closes the gap and on which
#: component.
PROJECTION_COMPONENTS = ("pressure_gradient", "ff_prime")
#: Samples per component curve in the figure.
PROJECTION_REFERENCE_SAMPLES = 257


def _elongation(boundary) -> float:
    """Return the stored boundary's elongation, height over width."""
    points = np.asarray(boundary, dtype=float).reshape(-1, 2)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < 2:
        raise ValueError("the stored boundary carries too few finite nodes")
    width = float(np.ptp(points[:, 0]))
    if width <= 0.0:
        raise ValueError("the stored boundary has no radial extent")
    return float(np.ptp(points[:, 1])) / width


def _elongated_minor_radius(boundary) -> tuple[float, float, float]:
    """Return the elongation-corrected minor radius and its two operands.

    The vertical-field identity is stated on a circular model plasma, so the
    horizontal half-width alone understates the radius of the elongated
    boundary it is applied to; ``a sqrt(kappa)`` is the radius of the
    equivalent circular column carrying the same poloidal flux.
    """
    elongation = _elongation(boundary)
    geometric = _minor_radius(boundary)
    return geometric * float(np.sqrt(elongation)), geometric, elongation


def _projected_profile(profile):
    """Return one row's profile with its source core projected onto polynomials.

    The extracted gradients are sampled on the declared uniform base, fitted,
    and the fitted pair replaces the interpolant in the source while every
    other field of the source is carried across unchanged.  The projection is
    not allowed to move the observable state by itself, so the caller compares
    the combination before and after and reports the difference it finds.

    The callable representation is a static property of the compiled program,
    not a per-slice operand, so the source is bound by rebuilding the operator
    on the same mesh rather than through the per-slice source binding, which
    refuses a representation change outright.  Only the two flux functions
    move: the mesh, the prescribed conductor field, the sampling rows and the
    solve policy are carried across unchanged.
    """
    coordinate = np.linspace(0.0, 1.0, PROJECTION_SAMPLES)
    projection = project_domain_profile(
        profile.source.core,
        coordinate,
        maximum_order=PROJECTION_MAXIMUM_ORDER,
        tolerance=PROJECTION_TOLERANCE,
    )
    source = replace(profile.source, core=projection.core())
    return replace(
        profile, operator=replace(profile.operator, source=source)
    ), projection


def _projection_pair(profile, *, target: float, minor_radius: float, component: str):
    """Return the Shafranov row with one component's scale left free."""
    functional = ExternalShafranovConstraint(
        minor_radius=jnp.asarray(minor_radius),
    )
    binding = ConstraintBinding(
        target=jnp.atleast_1d(jnp.asarray(target)),
        tolerance=jnp.asarray([ROW_TOLERANCE]),
        scale=jnp.asarray([1.0]),
        initial_unknown=jnp.asarray([0.0]),
        payload=(jnp.asarray(_external_image(profile)), jnp.asarray(minor_radius)),
        policy="imposed",
    )
    unknown = ProfileAmplitudeUnknown(component, jnp.asarray([1.0]))
    return ConstraintPair(functional, unknown, binding)


def _component_curve(function, coordinate, scale: float) -> np.ndarray:
    """Return one component sampled on a coordinate under an amplitude scale."""
    values = np.asarray(function(jnp.asarray(coordinate)), dtype=float)
    return values * float(scale)


def _component_field(component: str) -> str:
    """Return the source-core field name one compensator component moves."""
    return "p_prime" if component == "pressure_gradient" else "ff_prime"


def _render_projection(
    profile,
    *,
    core,
    projection,
    scales,
    reference,
    terminal,
    units,
    identity: str,
    caption: str,
    path: Path,
) -> dict:
    """Draw one row's component curves beside its terminal poloidal state.

    The two component panels carry the extracted profile and the projected one
    the row was stated against, and on top of those each freed-scale variant
    at its own terminal amplitude, so a reader sees whether a uniform scale
    reaches the source curve's shape.  The third panel is the terminal flux as
    unfilled line contours on the reference's own levels with both null sets
    and the wall, per the project's plotting rules.
    """
    coordinate = np.linspace(0.0, 1.0, PROJECTION_REFERENCE_SAMPLES)
    figure, axes = plt.subplots(1, 3, figsize=(13.6, 4.4), constrained_layout=True)
    for axis, component in zip(axes[:2], PROJECTION_COMPONENTS):
        fit = (
            projection.p_prime
            if component == "pressure_gradient"
            else projection.ff_prime
        )
        extracted = getattr(core, _component_field(component))
        axis.plot(
            coordinate,
            _component_curve(extracted, coordinate, 1.0),
            color="#888888",
            linewidth=1.0,
            label="extracted",
        )
        axis.plot(
            coordinate,
            _component_curve(fit.function, coordinate, 1.0),
            color="#3366cc",
            linewidth=1.6,
            label="projected",
        )
        scale = scales.get(component)
        if scale is not None:
            axis.plot(
                coordinate,
                _component_curve(fit.function, coordinate, scale),
                color="#cc7722",
                linewidth=1.6,
                linestyle="--",
                label=f"freed scale {(scale - 1.0) * 100:+.2f} %",
            )
        axis.set_xlabel(r"$\psi_N$")
        axis.set_ylabel(
            r"$p'$ [Pa/Wb]" if component == "pressure_gradient" else r"$FF'$ [T m/Wb]"
        )
        axis.set_title(
            f"{component}: order {fit.order}, {fit.basis}, "
            f"cond {fit.condition_number:.3g}",
            fontsize=8,
        )
        axis.legend(fontsize=7)
    _, _, reference_field = _raster(profile, reference, units)
    radial, height, terminal_field = _raster(profile, terminal, units)
    levels = poloidal.contour_levels(reference_field, count=12)
    poloidal.draw_flux_contours(
        axes[2], radial, height, terminal_field, levels, color="#cc7722"
    )
    poloidal.draw_wall(axes[2], units=units)
    for state, style in (
        (
            reference,
            DEFAULT_INK.variant(
                axis_color="#3366cc",
                xpoint_color="#3366cc",
                axis_marker="^",
                xpoint_marker="P",
            ),
        ),
        (
            terminal,
            DEFAULT_INK.variant(
                axis_color="#cc7722",
                xpoint_color="#cc7722",
                axis_marker="^",
                xpoint_marker="X",
            ),
        ),
    ):
        topology = _topology(profile.operator, state)
        if topology.get("read_status") != "qualified":
            continue
        poloidal.draw_nulls(
            axes[2],
            magnetic_axis=topology["axis_rz_m"],
            x_points=np.asarray(topology["x_point_rz_m"], dtype=float),
            style=style,
            contain=units,
        )
    poloidal_axes(axes[2])
    axes[2].set_title(
        "terminal flux, line contours on shared levels\n"
        "reference blue (^ axis, P x-point) / terminal orange (^ axis, X x-point)",
        fontsize=8,
    )
    figure.suptitle(f"MAST {identity}: {caption}", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    # The vector companion carries the same panels in a form a text-only reader
    # can inspect, so a lane that cannot open the raster still sees the labels,
    # the fitted orders and the curves the record cites.
    svg_path = path.with_suffix(".svg")
    figure.savefig(svg_path)
    plt.close(figure)
    return {
        "filesystem_path": str(path),
        "project_absolute_src": (
            "/nova/figures/constraint-augmented-newton-krylov/"
            f"flux-function-fit/{path.name}"
        ),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "vector_filesystem_path": str(svg_path),
        "vector_project_absolute_src": (
            "/nova/figures/constraint-augmented-newton-krylov/"
            f"flux-function-fit/{svg_path.name}"
        ),
        "vector_sha256": hashlib.sha256(svg_path.read_bytes()).hexdigest(),
    }


def _projection_row_receipt(
    profile,
    *,
    identity: str,
    reference_state,
    target_current,
    minor_radius: float,
    geometric_minor_radius: float,
    elongation: float,
    requested,
    directory: Path,
) -> dict[str, Any]:
    """Project one bank row's profiles and free their scales under the row.

    The row is imposed with the single scalar compensator the solve already
    carries, freed one component at a time: the row is one equation, so a pair
    of free scales in one solve leaves the target underdetermined.  What each
    variant reports is whether a uniform scale on that component alone reaches
    the target, which is the question the record has to answer before a shape
    unknown is designed.
    """
    source_core = profile.source.core
    projected, projection = _projected_profile(profile)
    projection_only = _combination(profile, reference_state, target_current)
    target = _combination(projected, reference_state, target_current)
    print(
        f"PROJECTION {identity} order_p={projection.p_prime.order} "
        f"order_ff={projection.ff_prime.order} "
        f"projection_only_delta={target - projection_only!r}",
        flush=True,
    )
    variants: list[dict[str, Any]] = []
    scales: dict[str, float] = {}
    terminal = None
    for component in PROJECTION_COMPONENTS:
        pair = _projection_pair(
            projected,
            target=target,
            minor_radius=minor_radius,
            component=component,
        )
        branch = projected.solve_branch(
            jnp.asarray(reference_state),
            requested,
            target_current=target_current,
            constraint_pairs=(pair,),
        )
        equilibrium = branch.equilibrium
        flux = equilibrium.flux
        flux.block_until_ready()
        terminal = np.asarray(flux)
        records = list(equilibrium.constraints)
        record = records[0] if records else None
        fraction = (
            None if record is None else float(np.asarray(record.physical_unknown[0]))
        )
        achieved = None if record is None else float(np.asarray(record.observed[0]))
        if fraction is not None:
            scales[component] = 1.0 + fraction
        variant: dict[str, Any] = {
            "component": component,
            "achieved_combination": _strict_float(achieved),
            "combination_gap_after": _strict_float(
                None if achieved is None else achieved - target
            ),
            "compensating_amplitude_fraction": _strict_float(fraction),
            "terminal_residual": _strict_float(branch.residual),
            "outer_steps": int(
                np.asarray(equilibrium.fixed_point.active_set_iterations)
            ),
            "converged": bool(np.asarray(branch.converged)),
            "topology_consistent": bool(np.asarray(branch.topology_consistent)),
            "termination": settled._termination_name(
                equilibrium.fixed_point.termination_reason
            ),
        }
        variants.append(variant)
        print(
            PROJECTION_EMISSION_PREFIX
            + json.dumps(
                {"identity": identity, **variant},
                sort_keys=True,
            ),
            flush=True,
        )
    best = min(
        (
            variant
            for variant in variants
            if variant["combination_gap_after"] is not None
        ),
        key=lambda variant: abs(variant["combination_gap_after"]),
        default=None,
    )
    scale_alone_closes = (
        best is not None and abs(best["combination_gap_after"]) <= ROW_TOLERANCE
    )
    entry: dict[str, Any] = {
        "identity": identity,
        "status": "imposed" if best is not None else "no_terminal_state",
        "minor_radius_m": _strict_float(minor_radius),
        "geometric_minor_radius_m": _strict_float(geometric_minor_radius),
        "elongation": _strict_float(elongation),
        "plasma_current_a": _strict_float(target_current),
        "target_combination": _strict_float(target),
        "source_combination_at_reference": _strict_float(projection_only),
        "projected_combination_at_reference": _strict_float(target),
        "reference_combination_gap": _strict_float(target - projection_only),
        "projection": projection.receipt(),
        "scale_alone_closes_the_gap": scale_alone_closes,
        "variants": variants,
    }
    if terminal is not None:
        gap_after = ", ".join(
            f"{variant['component']} {variant['combination_gap_after']:+.3e}"
            if variant["combination_gap_after"] is not None
            else f"{variant['component']} n/a"
            for variant in variants
        )
        entry["figure"] = _render_projection(
            projected,
            core=source_core,
            projection=projection,
            scales=scales,
            reference=reference_state,
            terminal=terminal,
            units=_wall_units(projected.operator),
            identity=identity,
            caption=(
                f"gap on the combination {entry['reference_combination_gap']:+.3e}; "
                f"after freeing one scale at a time: {gap_after}; "
                + ("scale alone closes it" if scale_alone_closes else "shape needed")
            ),
            path=directory / f"row-{identity.replace('/', '-')}.png",
        )
    return entry


def project_rows(*, directory: Path, cache_root: Path | None = None) -> dict[str, Any]:
    """Project every qualified bank row and free its scales under the row."""
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
        if cache_root is None
        else cache_root
    )
    response_cache, carrier_evidence = settled._persisted_response_cache(
        settled.response_carrier.DEFAULT_CARRIER,
        settled.response_carrier.DEFAULT_RECEIPT,
    )
    selected = _selection()
    directory.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "receipt": "projected extracted flux functions with their scales freed "
        "under the Shafranov row",
        "row_set": "every row the decomposition bank qualifies",
        "rows": [list(key) for key in sorted(selected)],
        "source": {
            "revision": _source_revision(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
        "configuration": {
            "projection": "project_domain_profile on the declared uniform "
            f"{PROJECTION_SAMPLES}-point normalised-flux base",
            "projection_maximum_order": PROJECTION_MAXIMUM_ORDER,
            "projection_tolerance": PROJECTION_TOLERANCE,
            "row": "ExternalShafranovConstraint at the elongation-corrected "
            "minor radius a sqrt(kappa)",
            "compensating_unknown": "ProfileAmplitudeUnknown freed one "
            "component at a time: the row is one equation",
            "row_tolerance": ROW_TOLERANCE,
            "persistent_compilation_cache": {
                "directory": str(cache.directory),
                "version": cache.version_key,
            },
        },
        "inputs": {"carrier_evidence": carrier_evidence},
        "rows_receipt": [],
    }
    for shot, row_index in sorted(selected):
        key = (shot, row_index)
        selected_row, qualification = selected[key]
        case, context = settled._mast_case_from_selection(
            settled.SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, _policy = settled._passive_inclusive_case(
            case, context, response_cache
        )
        minor_radius, geometric, elongation = _elongated_minor_radius(
            np.asarray(case["boundary"], dtype=float)
        )
        entry = _projection_row_receipt(
            profile,
            identity=f"{shot}/{row_index}",
            reference_state=jnp.asarray(passive_case["state"]),
            target_current=abs(float(passive_case["reference"]["plasma_current_a"])),
            minor_radius=minor_radius,
            geometric_minor_radius=geometric,
            elongation=elongation,
            requested=jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8),
            directory=directory,
        )
        receipt["rows_receipt"].append(entry)
        (directory / "receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        print(
            PROJECTION_EMISSION_PREFIX
            + json.dumps(
                {key: value for key, value in entry.items() if key != "figure"},
                sort_keys=True,
            ),
            flush=True,
        )
    (directory / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print("FLUX-FIT-DONE", flush=True)
    return receipt


def main(argv=None):
    """Run the Shafranov-row receipt from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument(
        "--emissions",
        type=Path,
        default=None,
        help="rewrite the receipts from a banked lane log instead of solving",
    )
    parser.add_argument(
        "--projection-directory",
        type=Path,
        default=None,
        help="project each row's extracted profiles and free their scales "
        "under the Shafranov row, writing to this directory",
    )
    arguments = parser.parse_args(argv)
    if arguments.projection_directory is not None:
        project_rows(
            directory=arguments.projection_directory,
            cache_root=arguments.cache_root,
        )
    elif arguments.emissions is None:
        measure(directory=arguments.directory, cache_root=arguments.cache_root)
    else:
        regenerate(emissions=arguments.emissions, directory=arguments.directory)


if __name__ == "__main__":
    main()
