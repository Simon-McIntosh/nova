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
    uri = "imas:hdf5?user=mcintos;pulse=105014;run=1;database=iter;version=3"
    with imas.DBEntry(uri, "r", dd_version="3.38.1") as entry:
        equilibrium = entry.get("equilibrium", lazy=False, autoconvert=False)
        active = entry.get("pf_active", lazy=False, autoconvert=False)
    source = PlasmaSource.from_imas(
        equilibrium, active, provenance="DINA 105014/1", indices=[200]
    )
    original = prescribe_slice(equilibrium.time_slice[200])
    prescribed = source.prescribe(0)
    np.testing.assert_array_equal(prescribed.cell_current, original.cell_current)
    np.testing.assert_array_equal(prescribed.boundary, original.boundary)
    assert prescribed.scale_factor == original.scale_factor
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
