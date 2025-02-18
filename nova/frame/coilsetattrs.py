"""Manage grid attributes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields

from nova.frame.framesetloc import FrameSetLoc
from nova.geometry.polyshape import PolyShape


@dataclass
class CoilSetAttrs(ABC, FrameSetLoc):
    """CoilSetAttrs baseclass."""

    delta: float = -1
    ifttt: bool = True
    name: str | None = None
    _attrs: dict = field(init=False, default_factory=dict, repr=False)
    default: dict = field(init=False, default_factory=lambda: {})
    link: bool = field(init=False, default=False)
    attributes: list[str] = field(
        init=False, default_factory=lambda: ["delta", "ifttt"]
    )

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

    @property
    def conditional_attributes(self):
        """Return conditional attributes flag."""
        return self.attrs.get("ifttt", True)

    @abstractmethod
    def set_conditional_attributes(self):
        """
        Set conditional attributes. Enable with ifttt (if this then that) boolean flag.

        Example
        -------
        Set turn attribute to equal 'skin' iff delta == -1:
            self.ifthen('delta', -1, 'turn', 'skin')
        """

    def _flatten(self, items):
        """Retrun flat list."""
        return [_item[0] if isinstance(_item, list) else _item for _item in items]

    def ifthen(self, attr, cond, key, value):
        """Update _attrs[key] = value when _attrs[check] == cond."""
        if isinstance(attr, str):
            attr, cond = [attr], [cond]
        attrs = self._flatten(
            [self._attrs.get(_attr, getattr(self, _attr)) for _attr in attr]
        )
        conds = self._flatten(cond)
        if all([_attr == _cond for _attr, _cond in zip(attrs, conds)]):
            self._attrs[key] = value

    @property
    def attrs(self):
        """Manage metagrid attrs."""
        return self._attrs

    @attrs.setter
    def attrs(self, attrs):
        self._attrs = self.default | attrs
        self.update_attrs()
        self.update_link()
        if self.conditional_attributes:
            self.set_conditional_attributes()

    def update_attrs(self):
        """Update missing attrs with instance values."""
        for attr in [attr.name for attr in fields(self)]:
            if attr in self.attributes and attr not in self._attrs:
                self._attrs[attr] = getattr(self, attr)
            if attr in ["section", "turn"]:
                if attr not in self._attrs:
                    continue
                self._attrs[attr] = PolyShape(self._attrs[attr]).shape

    def update_link(self):
        """Update link boolean."""
        _link = self._attrs.get("link", self.frame.metaframe.default["link"])
        if isinstance(_link, bool):
            self.link = _link


@dataclass
class GridAttrs(CoilSetAttrs):
    """Manage grid attributes."""

    gridattrs: dict = field(init=False, default_factory=lambda: {})
    attributes: list[str] = field(
        init=False,
        default_factory=lambda: [
            "trim",
            "fill",
            "delta",
            "turn",
            "section",
            "tile",
            "ifttt",
        ],
    )

    @property
    def attrs(self):
        """Extend CoilSet attrs."""
        return CoilSetAttrs.attrs.fget(self)

    @attrs.setter
    def attrs(self, attrs):
        CoilSetAttrs.attrs.fset(self, attrs)
        if self.conditional_attributes:
            self.set_conditional_attributes()
        self.gridattrs = {attr: self._attrs.pop(attr) for attr in self.gridattrs}

    @property
    def subattrs(self):
        """Return subframe attrs."""
        return {
            attr: self._attrs[attr]
            for attr in self._attrs
            if attr not in self.frame and attr not in self.frame.metaframe.tag
        }
