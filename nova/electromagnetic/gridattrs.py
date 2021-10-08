"""Manage grid attributes."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields

from nova.electromagnetic.frameset import Frame
from nova.electromagnetic.polygen import polyshape


@dataclass
class GridAttrs(ABC, Frame):
    """Manage grid attributes."""

    _attrs: dict = field(init=False, default_factory=dict, repr=False)
    default: dict = field(init=False, default_factory=lambda: {})
    grid: dict = field(init=False, default_factory=lambda: {})
    link: bool = field(init=False, default=False)
    attributes: list[str] = field(init=False, default_factory=lambda: [
        'link', 'trim', 'fill', 'delta', 'turn', 'tile'])

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
        self.link = self._attrs.pop('link',
                                    self.frame.metaframe.default['link'])

    @property
    def subattrs(self):
        """Return subframe attrs."""
        return {attr: self._attrs[attr] for attr in self._attrs
                if attr not in self.frame
                and attr not in self.frame.metaframe.tag}

    def update_attrs(self):
        """Update missing attrs with instance values."""
        for attr in [attr.name for attr in fields(self)]:
            if attr in self.attributes and attr not in self._attrs:
                self._attrs[attr] = getattr(self, attr)

    def ifthen(self, attr, cond, key, value):
        """Set _attrs[key] = value when _attrs[check] == cond."""
        if self._attrs.get(attr, getattr(self, attr)) == cond:
            self._attrs[key] = value
