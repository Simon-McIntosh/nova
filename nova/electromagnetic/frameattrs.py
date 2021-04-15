"""Manage grid attributes."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Union

from nova.electromagnetic.frame import Frame
from nova.electromagnetic.polygen import polyshape


@dataclass
class FrameArgs:
    """Prepend grid args."""

    frame: Frame = field(repr=False)
    subframe: Frame = field(repr=False)
    delta: float


@dataclass
class FrameAttrs(ABC, FrameArgs):
    """Manage frame attributes."""

    _attrs: dict = field(init=False, default_factory=dict, repr=False)
    default: dict = field(init=False, default_factory=lambda: {})
    grid: dict = field(init=False, default_factory=lambda: {})

    @abstractmethod
    def set_conditional_attributes(self):
        """
        Set conditional attributes.

        Example
        -------
        Set turn attribute to equal 'skin' iff delta == -1:
            self.ifthen('delta', -1, 'turn', 'skin')
        """

    @property
    def attrs(self):
        """Manage metagrid attrs."""
        return self._attrs

    @attrs.setter
    def attrs(self, attrs):
        self._attrs = self.default | attrs
        self.update_attrs()
        self._attrs['turn'] = polyshape[self._attrs['turn']]  # inflate turn
        self.set_conditional_attributes()
        self.grid = {attr: self._attrs.pop(attr) for attr in self.grid}

    def update_attrs(self):
        """Update missing attrs with instance values."""
        for attr in [attr.name for attr in fields(self)]:
            if isinstance(getattr(self, attr), (list, dict, Frame)):
                continue
            if attr not in self._attrs:
                self._attrs[attr] = getattr(self, attr)

    def ifthen(self, attr, cond, key, value):
        """Set _attrs[key] = value when _attrs[check] == cond."""
        if self._attrs.get(attr, getattr(self, attr)) == cond:
            self._attrs[key] = value
