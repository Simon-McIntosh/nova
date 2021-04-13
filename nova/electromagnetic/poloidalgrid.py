"""Manage poloidal grids."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields

from nova.electromagnetic.frame import Frame
from nova.electromagnetic.polygrid import PolyGrid
from nova.electromagnetic.polygen import polyshape


@dataclass
class Frames:
    """Collect frame and subframe attrs."""

    frame: Frame = field(repr=False)
    subframe: Frame = field(repr=False)


@dataclass
class PoloidalGrid(ABC, Frames):
    """Generate subframe poloidal grid from frame."""

    delta: float
    tile: bool = False
    trim: bool = True
    fill: bool = False
    turn: str = 'rectangle'
    _attrs: dict = field(init=False, default_factory=dict, repr=False)
    _subattrs: dict[str, bool] = field(init=False,
                                       default_factory=dict, repr=False)

    @abstractmethod
    def update_conditionals(self):
        """
        Update conditional attributes.

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
        self._attrs = attrs
        self.update_attrs()
        self._attrs['turn'] = polyshape[self._attrs['turn']]  # inflate turn
        self.update_conditionals()
        self._subattrs = {attr: self._attrs.pop(attr)
                          for attr in ['tile', 'trim', 'fill']}

    @property
    def subattrs(self):
        """Return subframe attitional attributes."""
        return self._subattrs

    def update_attrs(self):
        """Update missing attrs with instance values."""
        for attr in [attr.name for attr in fields(self)]:
            if isinstance(getattr(self, attr), (dict, Frame)) or \
                    attr[0] == '_':
                continue
            if attr not in self._attrs:
                self._attrs[attr] = getattr(self, attr)

    def ifthen(self, attr, cond, key, value):
        """Set _attrs[key] = value when _attrs[check] == cond."""
        if self._attrs.get(attr, getattr(self, attr)) == cond:
            self._attrs[key] = value

    def insert(self, *required, iloc=None, **additional):
        """
        Insert frame(s).

        Parameters
        ----------
        *required : Union[DataFrame, dict, list]
            Required input.
        iloc : int, optional
            Index before which coils are inserted. The default is None (-1).
        **additional : dict[str, Any]
            Additional input.

        Returns
        -------
        None.

        """
        self.attrs = additional
        index = self.frame.insert(*required, iloc=iloc, **self.attrs)
        self.subgrid(index)

    def subgrid(self, index):
        """
        Grid frame.

        - Store filaments in subframe.
        - Link turns.

        """
        columns = ['poly', 'delta', 'turn', 'nturn']
        columns += [attr for attr in ['scale', 'skin'] if attr in self.frame]
        frame = self.frame.loc[index, columns]
        subframe = []
        for i, name in enumerate(index):
            data = PolyGrid(**frame.iloc[i].to_dict(), **self.subattrs).frame
            subframe.append(self.subframe.assemble(
                data, label=name, delim='_'))
        self.subframe.concatenate(*subframe)
