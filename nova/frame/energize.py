"""Frame energization metamethod.

Records which of Ic / nturn are present so the frame can couple the turn
current It = Ic * nturn. The coupling itself lives in
:class:`~nova.frame.framelink.FrameLink`; this metamethod only sets the
metaframe energize group and availability flags.
"""

from dataclasses import dataclass, field

import nova.frame.metamethod as metamethod


@dataclass
class Energize(metamethod.Energize):
    """Manage dependant frame energization parameters."""

    frame: object = field(repr=False)

    def initialize(self):
        """Record availability of the coupled current columns."""
        for attr in self.available:
            self.available[attr] = attr in self.frame.columns
