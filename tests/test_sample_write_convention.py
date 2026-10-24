"""The sample writer carries nova's COCOS 17 state into the DD entry it writes.

The sample holds the reader's convention, COCOS 17, and
``Sample.equilibrium_ids`` stores it into a Data Dictionary entry. DD3
publishes COCOS 11, so every psi-like and d/dpsi-like row moves by a sign on
the way out; a writer that skipped the move stored the source's signed value
in a target that means the opposite. These tests write a known state and read
the entry back, which is the only way to see a write factor at all: a factor
applied twice, applied the wrong way, or not applied changes the number in the
entry.

The writes are database-free: the entry is built in memory and read back from
the same object, so nothing is stored and no pulse is opened.

The DD3 arm is a whole-entry round trip. The DD4 arm cannot be: DD4 renamed
``global_quantities.beta_normal`` to ``beta_tor_norm`` and removed
``time_slice[0].boundary_separatrix``, so this writer's
node paths name fields DD4 does not have and the entry cannot be built there
at all. The DD4 arm therefore asserts the factor the writer resolves for a DD4
target, which is the identity, and the DD3 arm is what proves the writer
applies it.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray

from nova.imas.sample import DEFAULT_DD_VERSION, Sample
from nova.io.sourcemap import SourceMapError, dd_equilibrium_factor
from nova_cocos import DODPSI_LIKE, PSI_LIKE

#: nova's own convention, as the reader serves it.
PSI_BOUNDARY = -102.0
PSI1D = np.array([-102.0, -60.0, -1.0])
DPRESSURE_DPSI = np.array([-1.5, -0.9, -0.1])
F_DF_DPSI = np.array([-2.5, -1.5, -0.5])
IP = 1.0e6


def state() -> Sample:
    """Return a sample holding one known COCOS 17 state over three slices."""

    def per_time(value):
        return ("time", np.full(3, value))

    data = xarray.Dataset(
        {
            "psi_boundary": ("time", np.full(3, PSI_BOUNDARY)),
            "psi1d": (("time", "radius"), np.tile(PSI1D, (3, 1))),
            "dpressure_dpsi": (("time", "radius"), np.tile(DPRESSURE_DPSI, (3, 1))),
            "f_df_dpsi": (("time", "radius"), np.tile(F_DF_DPSI, (3, 1))),
            "ip": ("time", np.full(3, IP)),
            "li_3": per_time(0.9),
            "beta_normal": per_time(1.4),
            "boundary_type": ("time", np.zeros(3, int)),
            "minor_radius": per_time(1.0),
            "elongation": per_time(1.6),
            "triangularity_upper": per_time(0.1),
            "triangularity_lower": per_time(0.1),
            "triangularity_outer": per_time(0.2),
            "triangularity_inner": per_time(0.4),
            "squareness_upper_inner": per_time(0.01),
            "squareness_upper_outer": per_time(0.01),
            "squareness_lower_inner": per_time(0.01),
            "squareness_lower_outer": per_time(0.01),
            "geometric_axis": (("time", "index"), np.tile([6.0, 0.0], (3, 1))),
            "x_point_number": ("time", np.ones(3, int)),
            "x_point": (("time", "point"), np.tile([4.0, -3.2], (3, 1))),
            "strike_point": (
                ("time", "strike_point_index", "point"),
                np.tile([4.5, -3.9], (3, 2, 1)),
            ),
        },
        coords={
            "time": np.array([0.0, 1.0, 2.0]),
            "radius": np.array([0.1, 0.5, 1.0]),
            "strike_point_index": [0, 1],
        },
        attrs={"equilibrium": "130506,6,equilibrium,0,iter,public,hdf5"},
    )
    return Sample(data)


def test_dd3_write_carries_the_psi_rows_into_cocos_11():
    """A DD3 entry stores the source's convention, so every signed row flips."""
    assert dd_equilibrium_factor(DEFAULT_DD_VERSION, PSI_LIKE) == -1.0
    ids = state().equilibrium_ids(DEFAULT_DD_VERSION)

    for itime in range(3):
        time_slice = ids.time_slice[itime]
        assert np.isclose(float(time_slice.boundary_separatrix.psi), -PSI_BOUNDARY)
        assert np.allclose(np.asarray(time_slice.profiles_1d.psi, float), -PSI1D)
        assert np.allclose(
            np.asarray(time_slice.profiles_1d.dpressure_dpsi, float),
            -DPRESSURE_DPSI,
        )
        assert np.allclose(
            np.asarray(time_slice.profiles_1d.f_df_dpsi, float), -F_DF_DPSI
        )
        # a control: the current is no flux-like row and must not have moved
        assert np.isclose(float(time_slice.global_quantities.ip), IP)


def test_write_is_idempotent_in_the_convention_it_writes():
    """Reading a DD3 entry back at its own convention returns the source state.

    The written entry is what a DD3 reader would translate into COCOS 17, and
    this reader's factor is the inverse of the writer's, so the round trip lands
    on the state the sample held. A write factor applied backwards leaves every
    row negated here.
    """
    ids = state().equilibrium_ids(DEFAULT_DD_VERSION)
    assert dd_equilibrium_factor(DEFAULT_DD_VERSION, PSI_LIKE) == -1.0
    assert dd_equilibrium_factor(DEFAULT_DD_VERSION, DODPSI_LIKE) == -1.0

    time_slice = ids.time_slice[0]
    assert np.isclose(
        dd_equilibrium_factor(DEFAULT_DD_VERSION, PSI_LIKE)
        * float(time_slice.boundary_separatrix.psi),
        PSI_BOUNDARY,
    )
    assert np.allclose(
        dd_equilibrium_factor(DEFAULT_DD_VERSION, PSI_LIKE)
        * np.asarray(time_slice.profiles_1d.psi, float),
        PSI1D,
    )


@pytest.mark.parametrize("transformation", [PSI_LIKE, DODPSI_LIKE])
def test_dd4_target_passes_the_source_convention_through(transformation):
    """A DD4 target publishes COCOS 17, so the writer's rows do not move."""
    assert dd_equilibrium_factor("4.0.0", transformation) == 1.0


def test_a_version_with_no_declared_convention_is_refused():
    """The writer asks for a factor of a version it knows, never a guess."""
    with pytest.raises(SourceMapError):
        dd_equilibrium_factor("2.0.0", PSI_LIKE)
