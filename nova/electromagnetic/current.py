"""Manage frame currents."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nova.electromagnetic.metamethod import MetaMethod

if TYPE_CHECKING:
    from nova.electromagnetic.frame import Frame


@dataclass
class Current(MetaMethod):
    """Manage frame currents."""

    frame: Frame = field(repr=False)
    attributes: list[str] = field(default_factory=lambda: [
        'Ic', 'It', 'Nt', 'active', 'plasma', 'optimize', 'feedback'])

    def initialize(self):
        """Init current attributes."""
        if self.enable:
            self.frame.metadata = {'lock': self.attributes}

    @property
    def Ic(self):
        """Manage line current."""
        return self.frame.Ic

    @Ic.setter
    def Ic(self, line_current):
        with self.frame.metaframe.unlock():
            self.frame.range.Ic = line_current
