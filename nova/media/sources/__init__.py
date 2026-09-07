"""Adapters turning a machine's stored equilibrium into media frame records.

One record shape serves every machine, so a painter never learns which archive
a frame came from and a figure can carry MAST beside DIII-D. Each adapter owns
exactly one archive's conventions -- which axis is stored, what the flux unit
is, how absence is encoded -- and converts once, at the read.
"""

from nova.media.sources.frame import (
    EquilibriumFrame,
    MachineGeometry,
    Pulse,
    SurfaceFrame,
)
from nova.media.sources.mast_thomson import ThomsonString, read_thomson
from nova.media.sources.nova_labels import read_labels
from nova.media.sources.plasma_mesh import clip_to_boundary, hex_mesh

# The readers are re-exported so a caller can find them from the package
# rather than having to know which archive module owns which one: a reader
# named for what it returns is discoverable, one named for its archive is
# not.
__all__ = [
    "EquilibriumFrame",
    "MachineGeometry",
    "Pulse",
    "SurfaceFrame",
    "ThomsonString",
    "clip_to_boundary",
    "hex_mesh",
    "read_labels",
    "read_thomson",
]
