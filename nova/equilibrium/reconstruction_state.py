"""The one writer every reconstruction column is read as an equilibrium through.

A reduced reconstruction recovers a poloidal flux map, a boundary outline, the
flux extremes that outline bounds and a plasma current. Three such
reconstructions differ in how they find those quantities, not in how an
equilibrium IDS states them, so the assembly lives here once and every
conversion calls it. A change to the IDS convention is then one edit.

Two conventions are shared by every column and are therefore fixed here rather
than repeated per reconstruction:

- The stored flux map is the total poloidal flux on the reconstruction's own
  state grid, with ``dim1`` the radial coordinate and ``dim2`` the vertical one,
  and ``psi`` transposed into the IDS's ``(radial, vertical)`` order. A
  reconstruction that supplies no readable map withholds ``profiles_2d``
  entirely rather than publishing a map no flux-surface read could use.
- No reduced reconstruction supplies a toroidal-field function, so the stored
  profile carries the vacuum closure ``F dF/dpsi = 0``: the safety factor is
  read from the flux-surface geometry with the field held at its boundary
  value, identically for every column and for the reference.
"""

import numpy as np

__all__ = ["equilibrium_from_state", "write_equilibrium"]


def equilibrium_from_state(
    *,
    flux,
    ring,
    axis,
    flux_axis,
    flux_boundary,
    plasma_current,
    grid_r,
    grid_z,
    vacuum_b0,
    reference_radius,
    time=0.0,
):
    """Assemble one reconstructed state as a single-time equilibrium IDS.

    ``flux`` is the total flux map in ``(n_z, n_r)`` order, or ``None`` for a
    reconstruction that supplies no map a flux-surface read could use. ``ring``
    is the closed boundary outline as ``(n, 2)`` ``(r, z)`` points.
    """
    import imas
    from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

    grid_r = np.asarray(grid_r, dtype=float)
    grid_z = np.asarray(grid_z, dtype=float)
    ring = np.asarray(ring, dtype=float)
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])

    equilibrium = imas.IDSFactory().equilibrium()
    equilibrium.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    equilibrium.time = np.array([float(time)])
    equilibrium.time_slice.resize(1)
    frame = equilibrium.time_slice[0]
    frame.boundary.outline.r = ring[:, 0]
    frame.boundary.outline.z = ring[:, 1]
    frame.boundary.type = 1
    quantities = frame.global_quantities
    quantities.magnetic_axis.r = float(axis[0])
    quantities.magnetic_axis.z = float(axis[1])
    quantities.psi_axis = float(flux_axis)
    quantities.psi_boundary = float(flux_boundary)
    quantities.ip = float(plasma_current)
    span = float(flux_boundary) - float(flux_axis)
    norm = np.linspace(0.0, 1.0, grid_r.size)
    frame.profiles_1d.psi = float(flux_axis) + span * norm
    frame.profiles_1d.f_df_dpsi = np.zeros_like(norm)
    if flux is not None:
        frame.profiles_2d.resize(1)
        stored = frame.profiles_2d[0]
        stored.grid.dim1 = grid_r
        stored.grid.dim2 = grid_z
        stored.grid_type.index = 1
        stored.psi = np.asarray(flux, dtype=float).T
    equilibrium.vacuum_toroidal_field.b0 = np.array([-abs(float(vacuum_b0))])
    equilibrium.vacuum_toroidal_field.r0 = float(reference_radius)
    return equilibrium


def write_equilibrium(path, equilibrium, *, dd_version="4.1.0"):
    """Write one assembled equilibrium to an IMAS dataset and return the path."""
    import imas
    from pathlib import Path

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    writer = imas.DBEntry(f"imas:hdf5?path={out.resolve()}", "w", dd_version=dd_version)
    writer.put(equilibrium)
    writer.close()
    return out