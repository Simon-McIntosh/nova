"""Independent DINA current maps check the flux-function prescription."""

import imas
import numpy as np
import pytest

from scripts.iter_prescribed_plasma.measure import prescribe_slice


@pytest.mark.slow
def test_dina_current_profile_follows_both_stored_derivatives():
    uri = "imas:hdf5?user=mcintos;pulse=105052;run=1;database=iter;version=3"
    with imas.DBEntry(uri, "r", dd_version="3.38.1") as entry:
        equilibrium = entry.get("equilibrium", lazy=False, autoconvert=False)
    slice_ = equilibrium.time_slice[1462]
    stored = np.asarray(slice_.profiles_2d[0].j_tor)
    prescribed = prescribe_slice(slice_)
    inside = prescribed.current_density != 0
    assert np.count_nonzero(inside) > 1000
    assert np.sqrt(np.mean(stored[inside] ** 2)) > 1e5
    relative_error = np.sqrt(
        np.mean((prescribed.current_density[inside] - stored[inside]) ** 2)
    ) / np.sqrt(np.mean(stored[inside] ** 2))
    assert relative_error < 0.01
    assert prescribed.scale_factor == pytest.approx(-1.0, abs=0.01)
    assert (
        np.linalg.norm(
            np.asarray(prescribed.centroid)
            - np.asarray(
                (
                    float(slice_.global_quantities.current_centre.r),
                    float(slice_.global_quantities.current_centre.z),
                )
            )
        )
        < 0.09375
    )
