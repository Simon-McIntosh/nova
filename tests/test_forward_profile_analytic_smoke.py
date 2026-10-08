"""The analytic validator case is a vacuum test: one coil, no plasma.

The pinned ``analytic`` case is a single filamentary coil at ``(103, 101)``
carrying 1 A, and its reference equilibrium carries only a 65x65 two-dimensional
flux map — ``global_quantities.ip`` is zero, ``psi_magnetic_axis`` holds the
empty sentinel, and ``profiles_1d`` and the boundary outline are empty.  There
is no plasma, so there is no plasma current and no magnetic axis to band; the
case validates the free-boundary flux assembly itself, not a confined
equilibrium.  The ITER trio rungs of the plan carry the Ip, axis and area bands.

The gate computes the vacuum poloidal flux of the input coil set on the
reference grid through the package's own coil Green function
(:func:`nova.biot.greens.greens_psi`, the kernel the forward operator carries),
with no fitted scale, and asserts the maximum absolute difference against the
reference peak ``|psi|``.  A forward profile solve is not run: the case is
vacuum with zero source current, so the Green function is the shipped route.
The COCOS transform is an IDENTITY — the reference ``psi`` is total poloidal
flux in the package's internal COCOS 17 convention, with neither a factor of
2*pi nor a sign between them — and that is asserted, not absorbed into a scale.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from nova.imas import validator_case as vc

_DD_VERSION = "4.1.0"

#: Relative bound on the map residual, as a fraction of the reference peak
#: ``|psi|``, written here before the comparison is scored.
MAP_BOUND = 1.0e-3

#: Environment seam that applies the declared negative-control mutation: the
#: input coil current scaled by this factor before comparison, which must fail
#: the map bound.  The gate the suite runs names the unscaled value.
CONTROL_SCALE_ENV = "NOVA_VALIDATOR_CONTROL_SCALE"
CONTROL_SCALE = 1.5


def _control_scale() -> float:
    return float(os.environ.get(CONTROL_SCALE_ENV, "1.0"))


def _resolve(tmp_path: Path) -> vc.ValidatorCase:
    try:
        return vc.resolve_validator_case("analytic", store_root=tmp_path / "store")
    except vc.LayerRegistryUnreachable as error:
        pytest.skip(f"validator registry unreachable: {error}")


def _vacuum_map_residual(case: vc.ValidatorCase, scale: float) -> dict:
    """Return the vacuum map residual of the input coil against the reference.

    The coil's placement and current are read from the input bundle's own
    ``pf_active``; the reference supplies the 65x65 grid and its flux map.  The
    node the coil sits on is singular and is excluded from the residual, and
    counted so the exclusion is visible rather than silent.
    """

    from nova.biot.greens import greens_psi

    entry = case.entry("input")
    try:
        coil = entry.get("pf_active").coil[0]
        rectangle = coil.element[0].geometry.rectangle
        coil_radius = float(rectangle.r)
        coil_height = float(rectangle.z)
        current = float(np.asarray(coil.current.data)[0])
    finally:
        entry.close()

    reference = case.entry("reference")
    try:
        slice_ = reference.get("equilibrium").time_slice[0]
        surface = slice_.profiles_2d[0]
        radius = np.asarray(surface.grid.dim1)
        height = np.asarray(surface.grid.dim2)
        reference_flux = np.asarray(surface.psi)
        plasma_current = float(slice_.global_quantities.ip)
    finally:
        reference.close()

    grid_radius, grid_height = np.meshgrid(radius, height, indexing="ij")
    vacuum = greens_psi(grid_radius, grid_height, coil_radius, coil_height) * (
        current * scale
    )
    finite = np.isfinite(vacuum)
    peak = float(np.abs(reference_flux).max())
    residual = float(np.abs(vacuum[finite] - reference_flux[finite]).max() / peak)
    return {
        "residual": residual,
        "peak": peak,
        "singular_nodes": int(finite.size - finite.sum()),
        "nodes": int(finite.sum()),
        "plasma_current": plasma_current,
        "coil": (coil_radius, coil_height, current),
    }


def test_analytic_reference_is_a_vacuum_map(tmp_path: Path) -> None:
    """The pinned analytic reference carries a vacuum flux map and no plasma."""

    case = _resolve(tmp_path)
    assert case.dd_version == _DD_VERSION
    assert {"input", "machine_description", "reference"} <= set(case.layers)

    entry = case.entry("reference")
    try:
        slice_ = entry.get("equilibrium").time_slice[0]
        assert float(slice_.global_quantities.ip) == 0.0
        assert len(np.asarray(slice_.profiles_1d.psi)) == 0
        surface = slice_.profiles_2d[0]
        assert np.asarray(surface.psi).shape == (65, 65)
    finally:
        entry.close()


def test_analytic_vacuum_flux_matches_the_reference(tmp_path: Path) -> None:
    """The coil's vacuum flux reproduces the reference map to 1e-3 of its peak.

    This is the analytic smoke gate.  It compares absolute total poloidal flux
    — the identity COCOS transform stated in the module docstring — so a sign
    flip or a 2*pi scale between the reference and the package would fail here
    rather than being absorbed by a fitted factor.
    """

    case = _resolve(tmp_path)
    result = _vacuum_map_residual(case, _control_scale())
    assert result["plasma_current"] == 0.0
    assert result["singular_nodes"] == 1
    assert result["residual"] < MAP_BOUND, result


def test_analytic_map_residual_is_reported(tmp_path: Path) -> None:
    """Emit the residual and the COCOS statement for the evidence record."""

    case = _resolve(tmp_path)
    result = _vacuum_map_residual(case, 1.0)
    radius, height, current = result["coil"]
    print(
        f"COCOS identity: reference psi is total poloidal flux (Wb), COCOS 17, "
        f"matches greens_psi with no 2*pi and no sign; "
        f"coil=({radius}, {height}) I={current} A; "
        f"map residual={result['residual']:.3e} of peak {result['peak']:.6e} Wb "
        f"over {result['nodes']} nodes ({result['singular_nodes']} singular excluded)"
    )
    assert result["residual"] < MAP_BOUND


def test_analytic_control_scaling_the_coil_current_fails_the_map_bound(
    tmp_path: Path,
) -> None:
    """The declared mutation: a 1.5x coil current must fail the map bound."""

    case = _resolve(tmp_path)
    result = _vacuum_map_residual(case, CONTROL_SCALE)
    assert result["residual"] > MAP_BOUND, result


def test_netcdf_to_hdf5_round_trips_a_layer(tmp_path: Path) -> None:
    """A written NetCDF equilibrium round-trips through the HDF5 converter."""

    import imas

    source = tmp_path / "case.nc"
    entry = imas.DBEntry(str(source), "w", dd_version=_DD_VERSION)
    equilibrium = imas.IDSFactory(version=_DD_VERSION).new("equilibrium")
    equilibrium.ids_properties.homogeneous_time = 1
    equilibrium.time = [1.0]
    equilibrium.time_slice.resize(1)
    equilibrium.time_slice[0].global_quantities.ip = 5.0e6
    entry.put(equilibrium)
    entry.close()

    target = tmp_path / "hdf5"
    vc.netcdf_to_hdf5(source, target, _DD_VERSION, ("equilibrium",))
    assert (target / "master.h5").is_file()

    before = (target / "master.h5").stat().st_mtime_ns
    vc.netcdf_to_hdf5(source, target, _DD_VERSION, ("equilibrium",))
    assert (target / "master.h5").stat().st_mtime_ns == before
