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
    required: list[str] = field(default_factory=lambda: ['Ic', 'It'])
    additional: list[str] = field(default_factory=lambda: [
        'Ic', 'It', 'Nt', 'active', 'plasma', 'optimize', 'feedback'])
    current: list[str] = field(default_factory=lambda: ['Ic', 'It', 'Nt'])
    require_all: bool = False

    def initialize(self):
        """Init current attributes."""
        self.frame.metadata = {'current': self.current,
                               'additional': self.current}

    def in_current(self, col):
        """Return Ture if col in metaframe.current."""
        if isinstance(col, int):
            col = self.columns[col]
        if not isinstance(col, str):
            return False
        return col in self.frame.metaframe.current

    def update(self, col):
        if self.in_current(col):
            print(col, self.current)
    #def set_current(self):


    '''
    @property
    def Ic(self):
        """Manage line current."""
        return self.frame.Ic

    @Ic.setter
    def Ic(self, line_current):
        with self.frame.metaframe.unlock():
            self.frame.range.Ic = line_current
    '''
