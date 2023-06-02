"""Shape methods for BiotFrame."""
from dataclasses import dataclass, field

import nova.frame.metamethod as metamethod
from nova.frame.framelink import FrameLink


@dataclass
class Shape(metamethod.Shape):
    """Shape methods for BiotFrame."""

    name = "biotshape"

    frame: FrameLink = field(repr=False)
    region: str = field(init=False, default="")
    source: int = field(init=False, default=0)
    target: int = field(init=False, default=0)

    def initialize(self):
        """Initialize source and target number."""
        self.source = len(self.frame)
        self.target = len(self.frame)

    def set_source(self, number):
        """Set source number, define frame as target."""
        self.source = number
        self.region = "target"

    def set_target(self, number):
        """Set source number, define frame as source."""
        self.target = number
        self.region = "source"
