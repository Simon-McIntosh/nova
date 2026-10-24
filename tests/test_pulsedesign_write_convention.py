"""The pulsedesign writer applies the DD convention factor at each of its sites.

``PulseDesign`` holds nova's own state, COCOS 17, and writes it into a Data
Dictionary entry at ``DEFAULT_DD_VERSION``, whose psi-like rows are COCOS 11.
Every psi-like row therefore moves by the factor
:func:`nova.io.sourcemap.dd_equilibrium_factor` returns for that version, and
nothing else does. A writer that skipped a site stored a COCOS 17 number in a
target that means the opposite.

The writer is exercised without a machine description or a database, which is
what the class's own entry point cannot do: ``PulseDesign`` reads the ITER
``pf_active`` description for its coil geometry, so every test in
``tests/test_pulsedesign.py`` skips without that database. Here the writer's
inputs are a dataset built in the test and an equilibrium IDS built in memory,
and the three factor sites are read back from the same IDS object. A factor
applied twice, applied the wrong way, or not applied changes the number in the
entry, so a read-back is the only way to see a write factor at all.
"""

from __future__ import annotations

import imas
import numpy as np
import xarray

from nova.imas.ids_entry import IdsEntry
from nova.imas.pulsedesign import PulseDesign
from nova.imas.sample import DEFAULT_DD_VERSION
from nova.io.sourcemap import dd_equilibrium_factor
from nova_cocos import PSI_LIKE

#: the known COCOS 17 state the writer is handed.
PSI_AXIS = -95.0
PSI_BOUNDARY = -102.0
PSI2D = np.array([[-101.0, -60.0, -1.0], [-50.0, -20.0, -0.5]])
LI_3 = 0.9

#: a shaping row the writer copies unchanged, and the value it is handed.
MINOR_RADIUS = 1.7

TIMES = np.array([0.0, 1.0, 2.0])

BOUNDARY_ATTRS = [
    "minor_radius",
    "elongation",
    "triangularity",
    "triangularity_upper",
    "triangularity_lower",
    "triangularity_inner",
    "triangularity_outer",
    "squareness_upper_inner",
    "squareness_upper_outer",
    "squareness_lower_inner",
    "squareness_lower_outer",
]


def waveform(value: float) -> tuple[str, np.ndarray]:
    """Return one per-time row of the known state."""
    return ("time", np.full(TIMES.size, value))


def dataset() -> xarray.Dataset:
    """Return the writer's input dataset holding one known COCOS 17 state."""
    data = xarray.Dataset(
        {
            "li_3": waveform(LI_3),
            "psi_axis": waveform(PSI_AXIS),
            "psi_boundary": waveform(PSI_BOUNDARY),
            "magnetic_axis": (("time", "point"), np.tile([6.2, 0.4], (TIMES.size, 1))),
            "geometric_axis": (("time", "point"), np.tile([6.0, 0.0], (TIMES.size, 1))),
            "boundary_type": ("time", np.ones(TIMES.size, int)),
            "boundary": (
                ("time", "boundary_index", "point"),
                np.tile([[4.0, -3.2], [5.0, 0.0], [4.0, 3.2], [3.0, 0.0]], (3, 1, 1)),
            ),
            "x_point_number": ("time", np.ones(TIMES.size, int)),
            "x_point": (("time", "point"), np.tile([4.0, -3.2], (TIMES.size, 1))),
            "strike_point_number": ("time", np.zeros(TIMES.size, int)),
            "strike_point": (
                ("time", "strike_point_index", "point"),
                np.zeros((TIMES.size, 2, 2)),
            ),
            "r": ("r", np.array([4.0, 0.4])),
            "z": ("z", np.array([-1.0, 0.0, 1.0])),
            "r2d": (("r", "z"), np.tile([[4.0], [0.4]], (1, 3))),
            "z2d": (("r", "z"), np.tile([-1.0, 0.0, 1.0], (2, 1))),
            "psi2d": (
                ("time", "r", "z"),
                np.stack([PSI2D] * TIMES.size),
            ),
        },
        coords={
            "time": TIMES,
            "point": ["r", "z"],
            "boundary_index": np.arange(4),
            "strike_point_index": np.arange(2),
        },
        attrs={
            "attrs_0d": ["li_3", "psi_axis", "psi_boundary"],
            "attrs_boundary_seperatrix": BOUNDARY_ATTRS,
        },
    )
    for attr in BOUNDARY_ATTRS:
        data[attr] = waveform(MINOR_RADIUS)
    return data


class WriterState:
    """Serve ``PulseDesign.equilibrium_ids`` its inputs without a machine.

    The writer reaches three attributes of its instance: the dataset it reads,
    the IDS to write into, and the metadata hook it calls. A machine
    description supplies the coil geometry that this writer never touches, and
    building one is what makes the class's own entry point need a database;
    supplying these three directly is the database-free seam.
    """

    tplasma = "hex"

    def __init__(self, data: xarray.Dataset):
        self._data = data
        self.data = data
        self.ids = imas.IDSFactory(DEFAULT_DD_VERSION).equilibrium()
        self.ids.ids_properties.homogeneous_time = 1
        self.ids.time = TIMES
        self.ids.time_slice.resize(TIMES.size)

    def update_metadata(self, ids_entry: IdsEntry):
        """Mark the entry as assembled by the writer under test."""
        ids_entry.ids.ids_properties.homogeneous_time = 1


def write(state: WriterState):
    """Run the writer's own body and return the IDS it filled."""
    return PulseDesign.equilibrium_ids.__get__(state)


def test_default_version_is_the_sample_writers():
    """The writer states no version of its own; the sample writer's is the one."""
    import nova.imas.pulsedesign as pulsedesign
    import nova.imas.sample as sample

    assert pulsedesign.DEFAULT_DD_VERSION == sample.DEFAULT_DD_VERSION


def test_every_psi_site_carries_the_dd3_factor():
    """A DD3 entry stores the source's convention, so every psi site flips."""
    factor = dd_equilibrium_factor(DEFAULT_DD_VERSION, PSI_LIKE)
    assert factor == -1.0, "the pinned default must name the opposite convention"

    ids = write(WriterState(dataset()))

    for itime in range(TIMES.size):
        time_slice = ids.time_slice[itime]
        assert np.isclose(
            float(time_slice.global_quantities.psi_axis), factor * PSI_AXIS
        ), "time_slice.global_quantities.psi_axis carries no factor"
        assert np.isclose(
            float(time_slice.global_quantities.psi_boundary), factor * PSI_BOUNDARY
        ), "time_slice.global_quantities.psi_boundary carries no factor"
        assert np.isclose(
            float(time_slice.boundary_separatrix.psi), factor * PSI_BOUNDARY
        ), "time_slice.boundary_separatrix.psi carries no factor"
        written = np.asarray(time_slice.profiles_2d[0].psi, float)
        assert written.shape == PSI2D.shape
        assert np.allclose(written, factor * PSI2D), (
            "time_slice.profiles_2d.psi carries no factor"
        )


def test_rows_that_are_not_psi_like_are_unchanged():
    """The shaping rows and the geometry are convention-free."""
    ids = write(WriterState(dataset()))
    for itime in range(TIMES.size):
        time_slice = ids.time_slice[itime]
        assert np.isclose(float(time_slice.global_quantities.li_3), LI_3)
        assert np.isclose(
            float(time_slice.boundary_separatrix.minor_radius), MINOR_RADIUS
        )
        assert time_slice.boundary_separatrix.type == 1


def test_a_dd4_default_passes_the_source_convention_through(monkeypatch):
    """The writer asks the DD owner for its factor, so a DD4 default is identity.

    A writer carrying a literal cannot reach this arm: the version it would ask
    about is the one it states, not the one the module names. So a site that
    stops reading the module's default and states its own factor fails here.
    """
    monkeypatch.setattr("nova.imas.pulsedesign.DEFAULT_DD_VERSION", "4.0.0")
    assert dd_equilibrium_factor("4.0.0", PSI_LIKE) == 1.0

    ids = write(WriterState(dataset()))
    for itime in range(TIMES.size):
        time_slice = ids.time_slice[itime]
        assert np.isclose(float(time_slice.global_quantities.psi_axis), PSI_AXIS)
        assert np.isclose(
            float(time_slice.global_quantities.psi_boundary), PSI_BOUNDARY
        )
        assert np.isclose(float(time_slice.boundary_separatrix.psi), PSI_BOUNDARY)
        assert np.allclose(np.asarray(time_slice.profiles_2d[0].psi, float), PSI2D)
