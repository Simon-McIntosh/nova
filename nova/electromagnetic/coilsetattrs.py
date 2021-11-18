"""Manage grid attributes."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields

from nova.electromagnetic.framesetloc import FrameSetLoc
from nova.electromagnetic.polygen import PolyGen


@dataclass
class CoilSetAttrs(ABC, FrameSetLoc):
    """CoilSetAttrs baseclass."""

    delta: float = -1
    _attrs: dict = field(init=False, default_factory=dict, repr=False)
    default: dict = field(init=False, default_factory=lambda: {})
    link: bool = field(init=False, default=False)
    attributes: list[str] = field(init=False, default_factory=lambda: [
        'link'])

    @abstractmethod
    def insert(self, *args, required=None, iloc=None, **additional):
        """
        Insert frame(s).

        Parameters
        ----------
        *args : Union[DataFrame, dict, list]
            Required input.
        required : list[str]
            Required attribute names (args). The default is None.
        iloc : int, optional
            Index before which coils are inserted. The default is None (-1).
        **additional : dict[str, Any]
            Additional input.

        Returns
        -------
        index : pandas.Index
            FrameSpace index.

        """

    @abstractmethod
    def set_conditional_attributes(self):
        """
        Set conditional attributes.

        Example
        -------
        Set turn attribute to equal 'skin' iff delta == -1:
            self.ifthen('delta', -1, 'turn', 'skin')
        """

    def ifthen(self, attr, cond, key, value):
        """Set _attrs[key] = value when _attrs[check] == cond."""
        if self._attrs.get(attr, getattr(self, attr)) == cond:
            self._attrs[key] = value

    @property
    def attrs(self):
        """Manage metagrid attrs."""
        return self._attrs

    @attrs.setter
    def attrs(self, attrs):
        self._attrs = self.default | attrs
        self.update_attrs()
        self.set_conditional_attributes()

    def update_attrs(self):
        """Update missing attrs with instance values."""
        for attr in [attr.name for attr in fields(self)]:
            if attr in self.attributes and attr not in self._attrs:
                self._attrs[attr] = getattr(self, attr)


@dataclass
class GridAttrs(CoilSetAttrs):
    """Manage grid attributes."""

    grid: dict = field(init=False, default_factory=lambda: {})
    attributes: list[str] = field(init=False, default_factory=lambda: [
        'link', 'trim', 'fill', 'delta', 'turn', 'tile'])

    @property
    def attrs(self):
        """Extend CoilSet attrs."""
        return CoilSetAttrs.attrs.fget(self)

    @attrs.setter
    def attrs(self, attrs):
        CoilSetAttrs.attrs.fset(self, attrs)
        self._attrs['turn'] = PolyGen.polyshape[self._attrs['turn']]
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
