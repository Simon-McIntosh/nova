"""Extend pandas.DataFrame to manage coil and subcoil data."""

from dataclasses import dataclass, field
from typing import Union, Any

import numpy as np
import pandas
import shapely.geometry
import shapely.strtree
import scipy.interpolate

from nova.electromagnetic.frame import Frame
from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.pfcoil import PFcoil
from nova.electromagnetic.pfshell import PFshell
from nova.utilities import geom


@dataclass
class Mesh:
    """Manage mesh dimensions."""

    dpol: float = -1
    dplasma: float = 0.25
    dshell: float = 2.5
    dsubshell: float = 0.25
    dfield: float = 0.2
    frame: DataFrame = field(init=False, repr=False)
    subframe: DataFrame = field(init=False, repr=False)


@dataclass
class Section:
    """Manage sectional properties."""

    section: str = 'rectangle'
    turn: str = 'circle'
    turn_fraction: float = 1


@dataclass
class MetaData:
    """Manage frameset metadata."""

    required: list[str] = field(repr=False, default_factory=lambda: [
        'x', 'z', 'dl', 'dt'])
    additional: list[str] = field(repr=False, default_factory=lambda: [
        'section', 'turn'])
    available: list[str] = field(repr=False, default_factory=lambda: [
        'link', 'part', 'frame', 'dx', 'dz', 'dA', 'dl_x', 'dl_z',
        'delta', 'nx', 'nz', 'section', 'turn', 'turn_fraction',
        'Ic', 'It', 'Nt', 'Nf', 'Psi', 'Bx', 'Bz', 'B', 'acloss'])
    subspace: list[str] = field(repr=False, default_factory=lambda: [
        'Ic'])
    metadata: dict[str, Union[str, dict]] = field(repr=False,
                                                  default_factory=dict)


@dataclass
class FrameSet(Mesh, Section, MetaData):
    """
    Build frameset.

    - poloidal: add poloidal coils.
    - shell: add poloidal shells.
    - plasma: add plasma (poloidal).

    """

    frame: Frame = field(init=False, repr=False)
    subframe: Frame = field(init=False, repr=False)
    pfcoil: PFcoil = field(init=False, repr=False)
    pfshell: PFshell = field(init=False, repr=False)

    def __post_init__(self):
        """Init coil and subcoil."""
        metadata = {'section': self.section, 'turn': self.turn,
                    'turn_fraction': self.turn_fraction}
        metadata |= self.metadata
        self.frame = Frame(
            required=self.required, additional=self.additional,
            available=self.available, subspace=self.subspace,
            exclude=['dl_x', 'dl_z', 'frame'], **metadata)
        self.subframe = Frame(
            required=self.required, additional=self.additional,
            available=self.available,
            subspace=self.subspace+['It', 'Nt'],
            exclude=['turn', 'turn_fraction', 'Nf', 'delta'],
            delim='_', **metadata)
        self.pfcoil = PFcoil(self.frame, self.subframe, self.dpol)
        self.pfshell = PFshell(self.frame, self.subframe,
                               self.dshell, self.dsubshell)

    def drop(self, index=None):
        """
        Remove frame and subframe.

        Parameters
        ----------
        index : int or list or pandas.Index, optional
            Index of coils to be removed. The default is None (all coils).

        Returns
        -------
        loc : [int, int]
            Location index of first removed [frame, subframe].

        """
        if index is None:  # drop all coils
            index = self.coil.index
        if not pandas.api.types.is_list_like(index):
            index = [index]
        loc = self.get_loc(index)
        for name in index:
            if name in self.frame.index:
                self.subframe.drop(self.coil.loc[name, 'subindex'])
                self.frame.drop(name)
        return loc

    def _get_iloc(self, index):
        iloc = [None, None]
        for name in index:
            if name in self.coil.index:
                iloc[0] = self.coil.index.get_loc(index[0])
                subindex = self.coil.subindex[index[0]][0]
                iloc[1] = self.subcoil.index.get_loc(subindex)
                break
        return iloc

    def translate(self, index=None, dx=0, dz=0):
        """
        Translate coil in polidal plane.

        Parameters
        ----------
        index : int or array-like or Index, optional
            Coil index. The default is None (all coils).
        dx : float, optional
            x-coordinate translation. The default is 0.
        dz : float, optional
            z-coordinate translation. The default is 0.

        Returns
        -------
        None.

        """
        if index is None:
            index = self.coil.index
        elif not pandas.api.types.is_list_like(index):
            index = [index]
        self.coil.translate(index, dx, dz)
        for name in index:
            self.subcoil.translate(self.coil.loc[name, 'subindex'], dx, dz)


if __name__ == '__main__':

    frameset = FrameSet(dpol=0.05, section='circle')
    frameset.pfcoil.insert(range(3), 1, 0.75, 0.75, link=True, delta=0.2)

    frameset.pfshell.insert([1, 2, 3], [3, 4, 4], dt=0.1)
    frameset.subframe.polyplot()

    print(frameset.frame)
