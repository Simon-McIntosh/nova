"""Show whether the existing current-density prescription detects reversed flux."""

from dataclasses import replace

import numpy as np

from nova.imas.plasma_source import CoilCurrentSeries, PlasmaSource
from nova.imas.profiles import prescribe_current_density
from nova.media.sources.mast_efit import read_pulse


frame = read_pulse(21978, times=[0.2], require_plasma=1e5).frames[0]
clock = CoilCurrentSeries(np.array([0.0, 0.4]), ("control",), np.array([[0.0], [1.0]]))
source = PlasmaSource.from_mast((frame,), clock, provenance="MAST 21978 EFIT")
slice_ = source.slices[0]
flipped = replace(
    slice_,
    psi=-slice_.psi,
    psi_samples=-slice_.psi_samples,
    psi_axis=-slice_.psi_axis,
    psi_boundary=-slice_.psi_boundary,
)
baseline = prescribe_current_density(
    flipped.radius,
    flipped.height,
    flipped.psi,
    flipped.psi_samples,
    flipped.p_prime,
    flipped.ff_prime,
    flipped.psi_axis,
    flipped.psi_boundary,
    flipped.plasma_current,
    flipped.outline,
    flipped.axis,
    flipped.flux_derivative_scale,
    x_points=flipped.x_points,
    boundary_outline=flipped.boundary_outline,
)
print(
    "BASELINE_DEFECT "
    f"flipped_psi_prescribed=True current_A={baseline.cell_current.sum():.6f} "
    f"scale={baseline.scale_factor:.9f}"
)
try:
    flipped.prescribe()
except ValueError as error:
    print(f"NEW_SOURCE_REFUSAL {error}")
else:
    raise AssertionError("flipped psi reached the current prescription")
