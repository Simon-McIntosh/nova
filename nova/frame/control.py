"""Manage control factroy methods."""
from dataclasses import dataclass
from functools import cached_property

from nova.frame.circuit import Circuit
from nova.frame.frameset import FrameSet


@dataclass
class Control(FrameSet):
    """Manage methods for frameset construction."""

    @cached_property
    def circuit(self):
        """Return coil constructor."""
        return Circuit(*self.frames, path=self.path, filename=self.filename)
