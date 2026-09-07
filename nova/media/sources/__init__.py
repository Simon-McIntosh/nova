"""Adapters turning a machine's stored equilibrium into media frame records.

One record shape serves every machine, so a painter never learns which archive
a frame came from and a figure can carry MAST beside DIII-D. Each adapter owns
exactly one archive's conventions -- which axis is stored, what the flux unit
is, how absence is encoded -- and converts once, at the read.
"""

from nova.media.sources.frame import EquilibriumFrame, MachineGeometry, Pulse

__all__ = ["EquilibriumFrame", "MachineGeometry", "Pulse"]
