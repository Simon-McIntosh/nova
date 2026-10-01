"""The coil-edit panel writer's read-back guard on the persisted X-point pair.

The guard's job is to refuse an archive whose stored X-point coordinate and
stored X-point flux are not the same stored field, because the panel contours
the field and marks the pair together. A guard that compares a fit against its
own evaluation cannot see that, so these tests measure it where it matters:
against an archive known to disagree with itself, and against an archive the
writer itself produced.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

from benchmarks.coil_edit_latency import (
    SADDLE_FLUX_TOLERANCE,
    _persisted_saddle_residuals,
    _refuse_inconsistent_persisted_pairs,
    _write_panel_data,
)
from nova.equilibrium.wall_mask import WallUnit, pack_wall_units
from nova.jax.config import configure_dtypes

# The read-back refits the stored field, and an extended-precision refit is a
# different measurement from a single-precision one: the same consistent pair
# reads back at 1.9e-16 Wb with x64 on and at 1.1e-7 Wb with it off, which
# straddles the guard's tolerance and would report a sound archive as corrupt.
configure_dtypes()
assert jax.config.jax_enable_x64 is True, (
    "the read-back guard measures a refit, so the test process must resolve the "
    "same working precision the writer did"
)


ARCHIVED_FIXTURE = (
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-local/"
    "fsa-coil-edit-records/regenerated-fixture-evidence/panel-states.npz"
)

# The fixture this repository ships, drawn under the vertical current-centre
# row and persisted beside the sweep record it came from.  It is the positive
# arm of the pair: the retained archive above proves the guard fires, and this
# one proves it does not fire on the archive the panels are drawn from.
REFRESHED_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "figures"
    / "forward-solve-api"
    / "coil-edit-nonconvergence"
    / "panel-states.npz"
)


class _WallStub:
    """The vessel spans one panel persists, in the shape the writer reads them.

    ``operator.wall`` is where the writer resolves the drawn vessel and
    ``operator`` is where it reads the recorded spans; the real operator
    carries both, so the stand-in points its own ``wall`` at itself.
    """

    def __init__(self, units: tuple[WallUnit, ...]) -> None:
        self.wall = self
        self.wall_units = tuple(units)
        coordinate, offsets = pack_wall_units(list(units))
        self.coordinate = coordinate
        self.wall_unit_offsets = offsets
        self.wall_unit_closed = np.asarray([u.closed for u in units], dtype=bool)
        self.wall_unit_kinds = tuple(u.kind for u in units)


def _wall_stub() -> _WallStub:
    vessel = WallUnit(
        r=np.asarray([0.1, 1.7, 1.7, 0.1]),
        z=np.asarray([-1.2, -1.2, 1.2, 1.2]),
        kind="vessel",
        closed=True,
    )
    return _WallStub((vessel,))


RADIUS = np.linspace(0.2, 1.6, 17)
HEIGHT = np.linspace(-1.0, 1.0, 17)
# The stationary point sits off the raster nodes so the tracer brackets it in a
# cell rather than reading a node, which is the branch the guard must cover.
SADDLE_R = 0.93
SADDLE_Z = 0.05
# An archived lattice pair deliberately inconsistent with the raster field: the
# coordinate sits off the stationary point and the flux is not the field there.
ARCHIVED_OFFSET_R = 0.02
ARCHIVED_OFFSET_Z = 0.01
ARCHIVED_FLUX = -0.123


def _raster() -> np.ndarray:
    """Return the synthetic raster psi, ``(len(HEIGHT), len(RADIUS))``."""
    r = RADIUS[None, :]
    z = HEIGHT[:, None]
    return 0.5 * ((r - SADDLE_R) ** 2 - (z - SADDLE_Z) ** 2)


def _state(rows: int = 1) -> dict:
    """Return one synthetic terminal state carrying an archived lattice pair."""
    outer = _raster()
    x_points = np.asarray(
        [
            [
                SADDLE_R + ARCHIVED_OFFSET_R * (index + 1),
                SADDLE_Z + ARCHIVED_OFFSET_Z * (index + 1),
            ]
            for index in range(rows)
        ],
        dtype=float,
    )
    return {
        "psi": outer.T.ravel(),
        "shape": np.asarray(outer.shape, dtype=int),
        "separatrix": np.empty((0, 2), dtype=float),
        "nulls": {
            "axis": np.asarray([SADDLE_R, SADDLE_Z], dtype=float),
            "x_points": x_points,
            "x_point_flux": np.full(rows, ARCHIVED_FLUX, dtype=float),
            "saddle_index": 0,
        },
        "edit_index": 3,
        "edit_fraction": 0.02,
        "terminal_residual": 1.0e-6,
        "converged": True,
        "trip_count": 4,
        "termination": "settled",
        "class": "limited",
    }


def _reference_panel() -> dict:
    outer = _raster()
    return {
        "radius": RADIUS,
        "height": HEIGHT,
        "psi": outer.T.ravel(),
        "shape": np.asarray(outer.shape, dtype=int),
        "separatrix": np.empty((0, 2), dtype=float),
        "nulls": {
            "axis": np.asarray([SADDLE_R, SADDLE_Z], dtype=float),
            "x_points": np.asarray([[SADDLE_R, SADDLE_Z]], dtype=float),
            "x_point_flux": np.asarray([0.0], dtype=float),
            "saddle_index": 0,
        },
    }


def _write(path, *, rows: int = 1) -> dict:
    profile = SimpleNamespace(operator=_wall_stub())
    _write_panel_data(
        path,
        profile=profile,
        panel_states=[_state(rows=rows)],
        reference_panel=_reference_panel(),
    )
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def test_read_back_fires_on_the_archived_inconsistent_fixture() -> None:
    """The retained pre-repair fixture disagrees with itself at every state."""
    with np.load(ARCHIVED_FIXTURE, allow_pickle=False) as archive:
        residuals = _persisted_saddle_residuals(archive)

    assert len(residuals) == 20
    firing = [
        state for state, value in residuals.items() if value > SADDLE_FLUX_TOLERANCE
    ]
    assert firing == sorted(residuals), (
        "the read-back guard must fire on every archived state, fired on %d of %d"
        % (len(firing), len(residuals))
    )
    assert abs(residuals[0] - 2.249e-2) < 1.0e-4, (
        "state 0 carries the recorded 2.249e-2 Wb disagreement, read %.3e"
        % residuals[0]
    )
    with pytest.raises(ValueError, match="not its own field"):
        _refuse_inconsistent_persisted_pairs(np.load(ARCHIVED_FIXTURE))


def test_refreshed_fixture_reads_back_consistent_at_every_state() -> None:
    """The committed fixture is the positive control for the retained one.

    The retained pre-repair archive is deliberately inconsistent, so it proves
    only that the guard can fire; a guard that fires on everything guards
    nothing.  The archive this repository ships is what the guard must pass,
    and every one of its states is asserted here rather than a sample, because
    the state that disagrees is exactly the one a sample would miss.
    """
    if not REFRESHED_FIXTURE.exists():
        pytest.skip("the coil-edit panel states are absent from this checkout")
    with np.load(REFRESHED_FIXTURE, allow_pickle=False) as archive:
        residuals = _persisted_saddle_residuals(archive)

    assert len(residuals) == 20
    over = {
        state: value
        for state, value in residuals.items()
        if value > SADDLE_FLUX_TOLERANCE
    }
    assert not over, (
        "the committed fixture must read back as its own field at every state, "
        "over tolerance at %s" % sorted(over)
    )
    with np.load(REFRESHED_FIXTURE, allow_pickle=False) as archive:
        _refuse_inconsistent_persisted_pairs(archive)


def test_writer_persists_a_pair_its_own_read_back_accepts(tmp_path) -> None:
    """An archive the writer produced reads back as one terminal state."""
    stored = _write(tmp_path / "panel-states.npz")

    residuals = _persisted_saddle_residuals(stored)
    assert list(residuals) == [0]
    assert residuals[0] <= SADDLE_FLUX_TOLERANCE, (
        "the writer's own archive must read back consistent, got %.3e" % residuals[0]
    )
    _refuse_inconsistent_persisted_pairs(stored)

    coordinate = np.asarray(stored["xpoints_0"], dtype=float)[0]
    level = float(np.asarray(stored["xpoint_flux_0"], dtype=float)[0])
    assert abs(coordinate[0] - SADDLE_R) < 0.01
    assert abs(coordinate[1] - SADDLE_Z) < 0.01
    assert abs(level) < 0.01
    # The stored coordinate is the field's own stationary pair, not the
    # archived lattice locator the substitution replaced.
    assert abs(coordinate[0] - (SADDLE_R + ARCHIVED_OFFSET_R)) > 1.0e-6

    # The audit field carries the disagreement the raster substitution removed
    # rather than a zero: the archived pair's flux against the stored field.
    delta = float(np.asarray(stored["saddle_flux_archived_delta_0"]))
    assert abs(delta) > SADDLE_FLUX_TOLERANCE, (
        "the archived disagreement must survive as its own persisted field, "
        "got %.3e" % delta
    )
    assert "saddle_flux_residual_0" in stored


def test_writer_refuses_a_flux_row_shorter_than_its_coordinate_row(tmp_path) -> None:
    """One range guard: a short flux row cannot take the raster coordinate."""
    with pytest.raises(ValueError):
        _write(tmp_path / "short-flux.npz", rows=0)
