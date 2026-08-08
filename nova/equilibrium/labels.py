"""Fixed dimensions and poloidal angles for the plasma-geometry labels.

The connectivity/topology reads resolve a magnetic axis, an order-invariant
X-point null set of a fixed number of slots, and the last-closed-flux-surface
radii at a fixed set of poloidal angles. Those counts and angles are shared,
data-independent constants so every read is a fixed-shape reduction.
"""

from __future__ import annotations

import numpy as np

#: R (m) and |Z| (m) window a reconstructed X-point null must fall within to be
#: kept — a NaN / sentinel / wildly displaced artefact is dropped, a real
#: near-limiter null is not.
XPOINT_VESSEL_R_RANGE = (0.1, 2.0)
XPOINT_VESSEL_Z_ABS = 2.0

#: Number of X-point null-set candidate slots.
N_XPOINT_SLOTS = 2

#: Number of fixed poloidal angles the LCFS boundary is resampled onto.
N_LCFS_ANGLES = 8

#: Fixed target dimensionality: axis(2) + X-point null-set(2 slots x 2) + 8 LCFS
#: radii = 14.
TARGET_DIM = 2 + 2 * N_XPOINT_SLOTS + N_LCFS_ANGLES

#: Human-readable name per target component (length == TARGET_DIM). The X-point
#: is an ORDER-INVARIANT null SET of ``N_XPOINT_SLOTS`` unordered slots; the slot
#: index carries no ordering / topology meaning.
TARGET_NAMES: tuple[str, ...] = (
    "axis_R",
    "axis_Z",
    *tuple(f"xpt{s}_{c}" for s in range(N_XPOINT_SLOTS) for c in ("R", "Z")),
    *tuple(f"lcfs_r_{k}" for k in range(N_LCFS_ANGLES)),
)

#: Fixed poloidal angles (radians, CCW from the outboard +R midplane) the LCFS is
#: ray-cast onto. theta_k = 2*pi*k / N_LCFS_ANGLES.
LCFS_ANGLES = (2.0 * np.pi * np.arange(N_LCFS_ANGLES) / N_LCFS_ANGLES).astype(
    np.float64
)
