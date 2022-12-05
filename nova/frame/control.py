"""Manage control factroy methods."""
from dataclasses import dataclass
from functools import cached_property
import inspect
from typing import ClassVar

from nova.frame.frameset import FrameSet


@dataclass
class Control(FrameSet):
    """Manage methods for frameset construction."""

    _controlmethods: ClassVar[dict[str, str]] = dict(
        circuit='.circuit.Circuit',
        )

    def _controlfactory(self, **kwargs):
        """Return nammed biot instance."""
        name = inspect.getframeinfo(inspect.currentframe().f_back, 0)[2]
        method = self.import_method(self._controlmethods[name], 'nova.frame')
        return method(*self.frames, path=self.path, filename=self.filename)

    @cached_property
    def circuit(self):
        """Return coil constructor."""
        return self._controlfactory()
