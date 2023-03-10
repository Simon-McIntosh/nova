"""Methods for ploting FrameSpace data."""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import cached_property
from importlib import import_module
import statistics
from string import digits
from typing import ClassVar, Optional, TYPE_CHECKING

import numpy as np

from nova.frame.dataframe import DataFrame
from nova.frame.error import ColumnError

if TYPE_CHECKING:
    import matplotlib


@dataclass
class Properties:
    """Manage plot properties."""

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

    def get_linewidth(self, unique_part, part, area, total_area):
        """Return patch linewidth."""
        finesse_fraction = 0.01
        patch_area = statistics.mode(area[part == unique_part])
        area_fraction = patch_area/total_area
        if area_fraction < finesse_fraction:
            return self.linewidth * area_fraction/finesse_fraction
        return self.linewidth

    def patch_kwargs(self, part: str):
        """Return single patch kwargs."""
        return dict(alpha=self.get_alpha(part),
                    facecolor=self.get_facecolor(part),
                    zorder=self.get_zorder(part),
                    linewidth=self.linewidth,
                    edgecolor=self.edgecolor)

    def patch_properties(self, part, area):
        """Return unique dict of patch properties extracted from parts list."""
        total_area = area.sum()
        return {unique_part: self.patch_kwargs(unique_part) | dict(
            linewidth=self.get_linewidth(unique_part, part, area, total_area))
                for unique_part in part.unique()}


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
    _fig: matplotlib.figure.Figure | None = field(init=False, repr=False)
    _axes: matplotlib.axes.Axes | None = field(init=False, repr=False)

    def generate(self, style='2d', nrows=1, ncols=1, **kwargs):
        """Generate new axis instance."""
        plt = import_module('matplotlib.pyplot')
        self.fig, self.axes = plt.subplots(nrows, ncols, **kwargs)
        self.set_style(style)
        return self.axes

    def gcf(self):
        """Link fig instance to current figure and return."""
        self._fig = import_module('matplotlib.pyplot').gcf()
        return self._fig

    def gca(self):
        """Link axes instance to current axes and return."""
        self._axes = import_module('matplotlib.pyplot').gca()
        return self._axes

    def despine(self, axes=None):
        """Remove spines from axes instance."""
        sns = import_module('seaborn')
        if axes is None:
            for axes in np.atleast_1d(self.axes):
                sns.despine(ax=axes)
        sns.despine(ax=axes)

    @staticmethod
    def _set_style(style, axes):
        """Set style on single axes instance."""
        match style:
            case '1d':
                axes.set_aspect('auto')
                axes.axis('on')
            case '2d':
                axes.set_aspect('equal')
                axes.axis('off')
            case _:
                raise NotImplementedError(f'style {style} not implemented')

    def set_style(self, style: Optional[str] = None):
        """Set axes style."""
        if style is None:
            style = self.style
        for axes in np.atleast_1d(self.axes):
            self._set_style(style, axes)
        if style == '1d':
            self.despine()
        self.style = style

    @property
    def fig(self):
        """Manage figure instance."""
        if self._fig is None:
            return self.gcf()
        return self._fig

    @fig.setter
    def fig(self, fig):
        self._fig = fig

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

    def legend(self, *args, **Kwargs):
        """Expose axes legend."""
        self.axes.legend(*args, **Kwargs)


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
class MoviePy:
    """Manage moviepy libaries."""

    @cached_property
    def editor(self):
        """Provide access to moviepy editor."""
        return import_module('moviepy.editor')

    @cached_property
    def bindings(self):
        """Provide access to moviepy video io bindings."""
        return import_module('moviepy.video.io.bindings')

    def mplfig_to_npimage(self, fig):
        """Return mplfig as npimage."""
        return self.bindings.mplfig_to_npimage(fig)


@dataclass
class Plot:
    """Manage plot workflow."""

    def __post_init__(self):
        """Link matplotlib libaries."""
        self.mpl_axes = Axes()
        self.mpl = MatPlotLib()
        self.plt = self.mpl['pylab']
        self.mpy = MoviePy()
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

    @property
    def fig(self):
        """Expose mpl figure instance."""
        return self.mpl_axes.fig

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

    def set_axes(self, style: Optional[str] = None, axes=None, **kwargs):
        """Set axes instance and style."""
        if axes is None:
            return self.mpl_axes.generate(style, **kwargs)
        return self.get_axes(style, axes=axes)

    def get_axes(self, style: Optional[str] = None, axes=None):
        """Get current axes instance and set style."""
        self.axes = axes
        self.axes_style = style
        return self.axes

    def legend(self, *args, **Kwargs):
        """Expose axes legend."""
        self.mpl_axes.legend(*args, **Kwargs)

    def savefig(self, *args, **kwargs):
        """Save figure to file."""
        self.plt.savefig(*args, **kwargs)
