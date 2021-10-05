"""Methods for ploting FrameSpace data."""
from dataclasses import dataclass, field
from typing import Union
import colorsys
from collections import Counter
import functools
import operator

import descartes
import numpy as np
import matplotlib
from matplotlib.collections import PatchCollection
import matplotlib.colors as mc
import pandas
import shapely.geometry

from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.error import ColumnError
from nova.utilities.IO import human_format
from nova.utilities.pyplot import plt


@dataclass
class BasePlot:
    """Plot baseclass for poly and vtk plot."""

    name = 'baseplot'

    frame: DataFrame = field(repr=False)

    def to_boolean(self, index):
        """Return boolean index."""
        try:
            if index.is_boolean():
                return index.to_numpy()
        except AttributeError:
            pass
        if index is None:
            return np.full(len(self.frame), True)
        if isinstance(index, str) and index in self.frame:
            index = self.frame.index[self.frame[index]]
        if isinstance(index, (str, int)):
            if isinstance(index, int):
                index = self.frame.index[index]
            index = [index]
        if isinstance(index, slice):
            index = self.frame.index[index]
        if np.array([isinstance(label, int) for label in index]).all():
            index = self.frame.index[index]
        return self.frame.index.isin(index)

    def get_index(self, index=None):
        """Return label based index for plot."""
        index = self.to_boolean(index)
        with self.frame.setlock(True, 'subspace'):
            try:
                if not self.zeroturn:  # exclude zeroturn (nturn == 0)
                    index &= self.frame.loc[:, 'nturn'] != 0
            except (AttributeError, KeyError, ColumnError):  # turns not set
                pass
            try:
                if not self.feedback:  # exclude stabilization coils
                    index &= ~self.frame.feedback
            except (AttributeError, KeyError, ColumnError):  # feedback not set
                pass
        index &= self.frame.segment == 'circle'
        return index


@dataclass
class Axes:
    """Manage plot axes."""

    _axes: plt.Axes = field(init=False, repr=False, default=None)

    def generate_axes(self):
        """Generate axes if unset."""
        if self._axes is None:
            self._axes = plt.gca()
            self._axes.set_aspect('equal')
            self._axes.axis('off')

    @property
    def axes(self):
        """Manage plot axes."""
        self.generate_axes()
        return self._axes

    @axes.setter
    def axes(self, axes):
        self._axes = axes


@dataclass
class Display:
    """Manage axes parameters."""

    patchwork: float = 0
    alpha: dict[str, float] = field(default_factory=lambda: {'plasma': 0.75})
    linewidth: float = 0.5
    edgecolor: str = 'white'
    facecolor: dict[str, str] = field(default_factory=lambda: {
        'vs3': 'C0', 'vs3j': 'gray', 'cs': 'C0', 'pf': 'C0',
        'trs': 'C3', 'dir': 'C3', 'vv': 'C3', 'vvin': 'C3',
        'vvout': 'C3', 'bb': 'C7', 'plasma': 'C4', 'cryo': 'C5'})
    zorder: dict[str, int] = field(default_factory=lambda: {
        'VS3': 1, 'VS3j': 0, 'CS': 3, 'PF': 2})

    def get_alpha(self, part):
        """Return patch alpha."""
        return self.alpha.get(part, 1)

    def get_facecolor(self, part):
        """Return patch facecolor."""
        return self.facecolor.get(part, 'C9')

    def get_zorder(self, part):
        """Return patch zorder."""
        return self.zorder.get(part, 0)

    def get_linewidth(self, part):
        """Return patch linewidth."""
        finesse_fraction = 0.01
        total_area = self.frame.area.sum()
        patch_area = self.frame.loc[self.frame.part == part, 'area'].mode()[0]
        area_fraction = patch_area/total_area
        if area_fraction < finesse_fraction:
            return self.linewidth * area_fraction/finesse_fraction
        return self.linewidth

    def patch_properties(self, parts):
        """Return unique dict of patch properties extracted from parts list."""
        return {part: {'alpha': self.get_alpha(part),
                       'facecolor': self.get_facecolor(part),
                       'zorder': self.get_zorder(part),
                       'linewidth': self.get_linewidth(part),
                       'edgecolor': self.edgecolor}
                for part in parts.unique()}

    def patch_number(self, parts):
        """Return patch number for each part."""
        return Counter(parts)


@dataclass
class Label:
    """Generate plot labels."""

    label: str = 'coil'
    current_unit: str = 'A'
    field_unit: bool = 'T'
    zeroturn: bool = False
    feedback: bool = False
    options: dict[str, Union[str, int, float]] = field(
        repr=False, default_factory=lambda: {'font_size': 'medium',
                                             'label_limit': 20})
    index: pandas.Index = field(init=False, repr=False)

    def __post_init__(self):
        """Update plot flags."""
        self.update_flags()
        super().__post_init__()

    def update_flags(self):
        """Update plot attribute flags."""
        if hasattr(self, 'biot'):
            if 'field' not in self.biot:
                self.field_unit = None
        if not 'energize' in self.frame.attrs:
            self.current_unit = None

    '''
    @property
        else:
            if not pandas.api.types.is_list_like(label):
                label = [label]
            parts = self.coil.part
            parts = [_part for _part in label if _part in parts]
    '''

    #def update_index(self):
    #    """Return update index from self.label boolean."""
    #    # if self.label#
    #
    #    with self.frame.setlock(True, 'subspace'):
    #        return self.frame.index[self.frame[self.label]]

    def add_label(self):
        """Add plot labels."""
        index = self.get_index()
        parts = self.frame.part[index]
        '''
        part_number = {p: sum(coil.part == p) for p in parts}
        # check for presence of field instance

        # referance vertical length scale
        referance_height = np.diff(self.axes.get_ylim())[0] / 100
        vertical_divisions = \
            np.sum(np.array([not parts.empty,
                             bool(self.current),
                             field]))
        if nz == 1:
            dz_ref = 0
        ztext = {name: 0 for name, value
                 in zip(['label', 'current', 'field'],
                        [label, current, field]) if value}
        for name, dz in zip(ztext, nz*dz_ref * np.linspace(1, -1, nz)):
            ztext[name] = dz
        for name, part in zip(coil.index, coil.part):
            if part in parts and N[part] < Nmax:
                x, z = coil.loc[name, 'x'], coil.loc[name, 'z']
                dx = coil.loc[name, 'dx']
                drs = 2/3*dx * np.array([-1, 1])
                if coil.part[name] == 'CS':
                    drs_index = 0
                    ha = 'right'
                else:
                    drs_index = 1
                    ha = 'left'
                # label coil
                ax.text(x + drs[drs_index], z + ztext['label'],
                        name, fontsize=fs, ha=ha, va='center',
                        color=0.2 * np.ones(3))
                if current:
                    if current == 'Ic' or current == 'A':  # line current
                        unit = 'A'
                        Ilabel = coil.loc[name, 'Ic']
                    elif current == 'It' or current == 'AT':  # turn current
                        unit = 'At'
                        Ilabel = coil.loc[name, 'It']
                    else:
                        raise IndexError(f'current {current} not in '
                                         '[Ic, A, It, AT]')
                    txt = f'{human_format(Ilabel, precision=1)}{unit}'
                    ax.text(x + drs[drs_index], z + ztext['current'], txt,
                            fontsize=fs, ha=ha, va='center',
                            color=0.2 * np.ones(3))
                if field:
                    self.update_field()
                    Blabel = coil.loc[name, 'B']
                    txt = f'{human_format(Blabel, precision=4)}T'
                    ax.text(x + drs[drs_index], z + ztext['field'], txt,
                            fontsize=fs, ha=ha, va='center',
                            color=0.2 * np.ones(3))
        '''
