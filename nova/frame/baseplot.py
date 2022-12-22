"""Methods for ploting FrameSpace data."""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import cached_property
from importlib import import_module
from typing import ClassVar, Optional, TYPE_CHECKING
from collections import Counter
from string import digits

import numpy as np
import statistics

if TYPE_CHECKING:
    import pandas

from nova.frame.dataframe import DataFrame
from nova.frame.error import ColumnError


@dataclass
class BasePlot:
    """Plot baseclass for poly and vtk plot."""

    name = 'baseplot'

    frame: DataFrame = field(repr=False)

    def to_boolean(self, index):
        """Return boolean index."""
        try:
            if index.dtype == bool:
                return index
        except AttributeError:
            pass
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

    def get_index(self, index=None, segment=None):
        """Return label based index for plot."""
        index = self.to_boolean(index)
        with self.frame.setlock(True, 'subspace'):
            try:
                if not self.zeroturn:  # exclude zeroturn (nturn == 0)
                    index &= self.frame.loc[:, 'nturn'] != 0
            except (AttributeError, KeyError, ColumnError):  # turns not set
                pass
        if segment:
            index &= self.frame.segment == segment
        return index


@dataclass
class Axes:
    """Manage plot axes."""

    style: str = '2d'

    def generate(self, style='2d'):
        """Generate new axis instance."""
        plt = import_module('matplotlib.pyplot')
        self.axes = plt.subplots(1, 1)[1]
        self.set_style(style)
        return self.axes

    def gca(self):
        """Link axes instance to current axes."""
        plt = import_module('matplotlib.pyplot')
        self._axes = plt.gca()

    def despine(self):
        """Remove spines from axes instance."""
        sns = import_module('seaborn')
        sns.despine(ax=self.axes)

    def set_style(self, style: Optional[str] = None):
        """Set axes style."""
        if style is None:
            style = self.style
        match style:
            case '1d':
                self.axes.set_aspect('auto')
                self.axes.axis('on')
                self.despine()
            case '2d':
                self.axes.set_aspect('equal')
                self.axes.axis('off')
            case _:
                raise NotImplementedError(f'style {style} not implemented')
        self.style = style

    @cached_property
    def _axes(self):
        """Cache axes instance."""
        return None

    @property
    def axes(self):
        """Manage plot axes."""
        if self._axes is None:
            self.gca()
            self.set_style()
        return self._axes

    @axes.setter
    def axes(self, axes):
        self._axes = axes


@dataclass
class MatPlotLib:
    """Manage matplotlib libaries."""

    def __getitem__(self, key: str):
        """Get item from matplotlib collections libary."""
        if 'Collection' in key:
            return getattr(self.collections, key)
        return import_module(f'matplotlib.{key}')

    @cached_property
    def collections(self):
        """Return matplotlib collections."""
        return import_module('matplotlib.collections')


@dataclass
class Plot:
    """Manage plot workflow."""

    def __post_init__(self):
        """Link matplotlib libaries."""
        self.mpl_axes = Axes()
        self.mpl = MatPlotLib()
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

    @property
    def axes(self):
        """Expose mpl axes instance."""
        return self.mpl_axes.axes

    @axes.setter
    def axes(self, axes):
        self.mpl_axes.axes = axes

    @property
    def axes_style(self):
        """Manage axes style."""
        return self.mpl_axes.style

    @axes_style.setter
    def axes_style(self, style: str):
        self.mpl_axes.set_style(style)

    def set_axes(self, axes, style: Optional[str] = None):
        """Set axes instance and style."""
        if axes is None:
            return self.mpl_axes.generate(style)
        return self.get_axes(axes, style)

    def get_axes(self, axes, style: Optional[str] = None):
        """Get current axes instance and set style."""
        self.axes = axes
        self.axes_style = style
        return self.axes

    def legend(self, *args, **Kwargs):
        """Expose axes legend."""
        self.mpl_axes.legend(*args, **Kwargs)


@dataclass
class Display:
    """Manage axes parameters."""

    patchwork: float = 0
    alpha: dict[str, float] = field(default_factory=lambda: {'plasma': 0.75})
    linewidth: float = 0.5
    edgecolor: str = 'white'
    facecolor: ClassVar[dict[str, str]] = {
        'vs3': 'C0', 'vs3j': 'gray', 'cs': 'C0', 'pf': 'C0',
        'trs': 'C3', 'dir': 'C3', 'vv': 'C3', 'vvin': 'C3',
        'vvout': 'C3', 'bb': 'C7', 'plasma': 'C4', 'cryo': 'C5',
        'fi': 'C2', 'tf': 'C7'}
    zorder: dict[str, int] = field(default_factory=lambda: {
        'VS3': 1, 'VS3j': 0, 'CS': 3, 'PF': 2})

    @staticmethod
    def get_part(part):
        """Return formated part name."""
        if part.rstrip(digits) == 'fi':
            return 'fi'
        return part

    def get_alpha(self, part):
        """Return patch alpha."""
        return self.alpha.get(part, 1)

    @classmethod
    def get_facecolor(cls, part):
        """Return patch facecolor."""
        return cls.facecolor.get(cls.get_part(part), 'C9')

    def get_zorder(self, part):
        """Return patch zorder."""
        return self.zorder.get(part, 0)

    def get_linewidth(self, part):
        """Return patch linewidth."""
        finesse_fraction = 0.01
        total_area = self.frame.area.sum()
        index = self.frame.part == part
        area = self.frame.loc[index, 'area']
        patch_area = statistics.mode(area)
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
    field_unit: bool | str = 'T'
    zeroturn: bool = False
    options: dict[str, str | int | float] = field(
        repr=False, default_factory=lambda: {'font_size': 'medium',
                                             'label_limit': 20})
    index: pandas.Index = field(init=False, repr=False)

    def __post_init__(self):
        """Update plot flags."""
        self.update_flags()
        if hasattr(super(), '__post_init__'):
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
