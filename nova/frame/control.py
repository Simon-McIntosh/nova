"""Manage control factroy methods."""
from dataclasses import dataclass

from nova.frame.frameset import FrameSet, frame_factory


@dataclass
class Control(FrameSet):
    """Manage methods for control instance initiation."""

    @frame_factory('nova.frame.circuit.Circuit')
    def circuit(self):
        """Return coil constructor."""
        return dict(path=self.path, filename=self.filename)
