"""Manage frameset opperations."""
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy.typing as npt

from nova.electromagnetic.frameset import FrameSet
from nova.electromagnetic.frame import Frame


@dataclass
class Switch:
    """Manage current selection logic."""

    frame: Frame = field(repr=False)
    subspace: bool = field(init=False)
    array: bool = field(init=False)


    @contextmanager
    def subset(self, index: str):
        """Set subset index."""
        if self.subspace:
            self.index = self.frame.subspace[index]
        else:
            self.index = self.frame[index]
        yield
        self.index = None

    @property
    def current(self):
        """Return line current."""
        if self.subspace:
            return self._get_current(self.frame.subspace)
        return self._get_current(self.frame)

    @current.setter
    def current(self, current):
        if self.subspace:
            return self._set_current(self.frame.subspace, current)
        return self._set_current(self.frame, current)

    def _get_current(self, frame):
        if self.index is None:
            return frame.Ic
        return frame.Ic[self.index]

    def _set_current(self, frame, current):
        if self.index is None:
            frame.Ic = current
            return
        if self.array:
            frame.Ic[self.index] = current
            return
        frame.loc[self.index, 'Ic'] = current


@dataclass
class FrameOps(FrameSet):
    """Manage current operations on frameset."""

    def __post_init__(self):
        """Init selection logic."""
        super().__post_init__()
        self.switch = Switch(self.subframe)

    @property
    def current(self):
        """Manage line current."""
        return self.switch.current

    @current.setter
    def current(self, current):
        self.switch.current = current
