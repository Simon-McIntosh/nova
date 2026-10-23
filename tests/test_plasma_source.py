"""A reconstructed or simulated equilibrium paired with coil-current series."""

from dataclasses import replace
from pathlib import Path

import imas
import numpy as np
import pytest

from nova.imas.plasma_source import CoilCurrentSeries, PlasmaSource
from nova.imas.prescribed_plasma import plasma_flux_at_points
from scripts.iter_prescribed_plasma.measure import prescribe_slice
from scripts.reconstructed_plasma_source.measure import (
    FLUX_FRACTION_LIMIT,
    SHOT,
    mast_source,
    measure_linkage,
)


def test_coil_current_clock_is_independent_and_bounded():
    coils = CoilCurrentSeries(
        np.array([0.0, 0.5, 1.0]),
        ("upper", "lower"),
        np.array([[0.0, 4.0], [2.0, 6.0], [4.0, 8.0]]),
    )
    np.testing.assert_array_equal(coils.at(0.25), [1.0, 5.0])
    with pytest.raises(ValueError, match="does not cover"):
        coils.at(1.01)


@pytest.mark.slow
def test_mast_reconstruction_linkage_matches_own_psi():
    if not (
        Path("/work/projects/imas_gpu/mast/level1/shots") / f"{SHOT}.zarr"
    ).is_dir():
        pytest.skip("MAST level-1 mirror unavailable")
    source, pulse = mast_source()
    result = measure_linkage(source, pulse)
    assert len(source.coils.time) > len(source.slices)
    assert len(source.coils.names) == 13
    assert np.all(np.isfinite(source.coil_currents(0)))
    assert source.prescribe(0).boundary_source == "boundary_outline"
    assert result["limit_fraction"] == FLUX_FRACTION_LIMIT
    assert result["targets"]["coil"]["count"] > 0
    assert result["targets"]["loop"]["count"] > 0
    for kind in ("coil", "loop"):
        assert result["targets"][kind]["max_abs_wb"] <= result["limit_wb"]
    print(
        "MAST_LINKAGE "
        f"coils={result['targets']['coil']['count']} "
        f"coil_max_wb={result['targets']['coil']['max_abs_wb']:.9f} "
        f"loops={result['targets']['loop']['count']} "
        f"loop_max_wb={result['targets']['loop']['max_abs_wb']:.9f} "
        f"limit_wb={result['limit_wb']:.9f}"
    )


@pytest.mark.slow
def test_mast_flipped_psi_is_refused():
    if not (
        Path("/work/projects/imas_gpu/mast/level1/shots") / f"{SHOT}.zarr"
    ).is_dir():
        pytest.skip("MAST level-1 mirror unavailable")
    source, _ = mast_source()
    original = source.slices[0]
    flipped = replace(
        original,
        psi=-original.psi,
        psi_samples=-original.psi_samples,
        psi_axis=-original.psi_axis,
        psi_boundary=-original.psi_boundary,
    )
    with pytest.raises(ValueError, match="psi ordering disagree"):
        flipped.prescribe()
    assert source.prescribe(0).plasma_current > 0


@pytest.mark.slow
def test_dina_generic_source_reproduces_existing_prescription():
    """DINA 105069/3 read through the source reproduces its own prescription.

    The expectation changed only in the pulse: 105014/1 is refused by the
    translator, 105069/3 conforms.  The prescription itself is invariant under
    the DD3 to COCOS 17 translation, because the flux map, the boundary outline
    and both flux functions take the same -1 factor, which leaves the profile
    coordinate, the current density and the integrated current unchanged.
    """
    uri = f"imas:hdf5?path={IMASDB / '105069' / '3'}"
    with imas.DBEntry(uri, "r", dd_version="3.39.0") as entry:
        equilibrium = entry.get("equilibrium", lazy=False, autoconvert=False)
        active = entry.get("pf_active", lazy=False, autoconvert=False)
    source = PlasmaSource.from_imas(
        equilibrium, active, provenance="DINA 105069/3", indices=[200]
    )
    original = prescribe_slice(equilibrium.time_slice[200])
    prescribed = source.prescribe(0)
    np.testing.assert_array_equal(prescribed.cell_current, original.cell_current)
    np.testing.assert_array_equal(prescribed.boundary, original.boundary)
    # The one changed expectation: the DD3 to COCOS 17 translation flips the
    # flux orientation, so the raw profile current the scale is taken against
    # flips with it while the plasma current does not.  The scale is therefore
    # the negative of the raw route's, and the cell current it multiplies is
    # unchanged because the density it multiplies flipped too.
    assert prescribed.scale_factor == pytest.approx(-original.scale_factor)
    assert prescribed.plasma_current == original.plasma_current
    assert len(source.coil_currents(0)) == len(active.coil)
    assert plasma_flux_at_points(
        prescribed, np.array([7.0]), np.array([0.4])
    ) == pytest.approx(
        plasma_flux_at_points(original, np.array([7.0]), np.array([0.4]))
    )
    print(
        "DINA_IDENTITY "
        f"time_s={source.slices[0].time:.9f} "
        f"current_a={prescribed.plasma_current:.6f} "
        f"cells={prescribed.cell_current.size} "
        f"coil_rows={len(source.coils.time)}"
    )


#: The two ITER DINA pulses the translator refuses, each with the digit its
#: stored data contradicts.  Neither conforms to DD3's COCOS 11, so neither may
#: reach a slice.
NON_CONFORMING_DINA = {"105014/1": "sigma_Bp=-1", "105013/1": "sigma_rho_theta_phi=-1"}

IMASDB = Path("/work/imas/shared/imasdb/ITER/3")

CONFORMING_CORSICA = IMASDB / "130506" / "6"


@pytest.mark.parametrize("case", sorted(NON_CONFORMING_DINA))
@pytest.mark.slow
def test_from_imas_refuses_non_conforming_dina_by_name(case):
    """A read whose digits contradict its DD version never reaches a slice."""

    failed_digit = NON_CONFORMING_DINA[case]
    uri = f"imas:hdf5?path={IMASDB / case}"
    with imas.DBEntry(uri, "r") as entry:
        equilibrium = entry.get("equilibrium", lazy=False, autoconvert=False)
        active = entry.get("pf_active", lazy=False, autoconvert=False)
    with pytest.raises(ValueError) as refused:
        PlasmaSource.from_imas(
            equilibrium, active, provenance=f"DINA {case}", indices=[200]
        )
    message = str(refused.value)
    assert f"DINA {case}" in message
    assert failed_digit in message


@pytest.mark.slow
def test_from_imas_prescribes_corsica_130506_6():
    """The conforming DD3 CORSICA pulse prescribes in COCOS 17."""

    if not CONFORMING_CORSICA.is_dir():
        pytest.skip("imasdb pulse not present")
    entry = imas.DBEntry(
        f"imas:hdf5?path={CONFORMING_CORSICA}", "r", dd_version="3.41.0"
    )
    with entry:
        equilibrium = entry.get("equilibrium", lazy=False, autoconvert=False)
        active = entry.get("pf_active", lazy=False, autoconvert=False)
    source = PlasmaSource.from_imas(
        equilibrium, active, provenance="CORSICA 130506/6", indices=[0]
    )
    prescribed = source.prescribe(0)
    assert prescribed.plasma_current == pytest.approx(
        float(equilibrium.time_slice[0].global_quantities.ip)
    )
