"""Methods for ploting FrameSpace data."""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from importlib import import_module
import statistics
from string import digits
from typing import ClassVar, Optional, TYPE_CHECKING

import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Patch3DCollection
import xarray

from nova.frame.dataframe import DataFrame
from nova.frame.error import ColumnError

if TYPE_CHECKING:
    import matplotlib


@dataclass
class Properties:
    """Manage plot properties."""

    patchwork: float = 0
    linewidth: float = 0.5
    edgecolor: str = "white"
    alpha: ClassVar[dict[str, float]] = {"plasma": 0.25}
    facecolor: ClassVar[dict[str, str]] = {
        "vs": "C2",
        "vs3": "C2",
        "vf": "C2",
        "vs3j": "C2",
        "ivc": "C2",
        "cs": "C0",
        "oh": "C0",
        "pf": "C0",
        "trs": "C3",
        "dir": "C3",
        "vv": "C3",
        "passive": "C3",
        "vvin": "C3",
        "vvout": "C3",
        "bb": "C7",
        "plasma": "C4",
        "cryo": "C7",
        "fi": "C2",
        "tf": "C7",
        "elm": "C3",
        "cc": "C2",
    }
    zorder: dict[str, int] = field(
        default_factory=lambda: {"VS3": 1, "VS3j": 0, "CS": 3, "PF": 2}
    )

    @staticmethod
    def get_part(part):
        """Return formated part name."""
        if part.rstrip(digits) == "fi":
            return "fi"
        return part

    @classmethod
    def get_alpha(cls, part):
        """Return patch alpha."""
        return cls.alpha.get(cls.get_part(part), 1)

    @classmethod
    def get_facecolor(cls, part):
        """Return patch facecolor."""
        return cls.facecolor.get(cls.get_part(part), "C9")

    def get_zorder(self, part):
        """Return patch zorder."""
        return self.zorder.get(part, 0)

    def get_linewidth(self, unique_part, part, area, total_area):
        """Return patch linewidth."""
        finesse_fraction = 0.01
        patch_area = statistics.mode(area[part == unique_part])
        area_fraction = patch_area / total_area
        if area_fraction < finesse_fraction:
            return self.linewidth * area_fraction / finesse_fraction
        return self.linewidth

    def patch_kwargs(self, part: str):
        """Return single patch kwargs."""
        return {
            "alpha": self.get_alpha(part),
            "facecolor": self.get_facecolor(part),
            "zorder": self.get_zorder(part),
            "linewidth": self.linewidth,
            "edgecolor": self.edgecolor,
        }

    def patch_properties(self, part, area):
        """Return unique dict of patch properties extracted from parts list."""
        total_area = area.sum()
        return {
            unique_part: self.patch_kwargs(unique_part)
            | {"linewidth": self.get_linewidth(unique_part, part, area, total_area)}
            for unique_part in part.unique()
        }


@dataclass
class BasePlot:
    """Plot baseclass for poly and vtk plot."""

    name = "baseplot"

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
        if isinstance(index, str) and index in self.frame.part.values:
            index = self.frame.index[index == self.frame.part]
        if isinstance(index, (str, int)):
            if isinstance(index, int):
                index = self.frame.index[index]
            index = [index]
        if isinstance(index, slice):
            index = self.frame.index[index]
        if np.array([isinstance(label, int) for label in index]).all():
            index = self.frame.index[index]
        return self.frame.index.isin(index)

    def get_index(self, index=None, segment=None, zeroturn=False):
        """Return label based index for plot."""
        index = self.to_boolean(index)
        with self.frame.setlock(True, "subspace"):
            try:
                if not zeroturn:  # exclude zeroturn (nturn == 0)
                    index &= self.frame.loc[:, "nturn"] != 0
            except (AttributeError, KeyError, ColumnError):  # turns not set
                pass
        if segment:
            index &= self.frame.segment == segment
        return index


@dataclass
class Axes:
    """Manage plot axes."""

    style: str = "2d"
    nrows: int = 1
    ncols: int = 1
    _fig: matplotlib.figure.Figure | None = field(init=False, repr=False, default=None)
    _axes: matplotlib.axes.Axes | None = field(init=False, repr=False, default=None)

    @cached_property
    def gridspec(self):
        """Provide access to gridspec layout editor."""
        return import_module("matplotlib.gridspec").GridSpec

    @cached_property
    def plt(self):
        """Provied access to pyplot module."""
        return import_module("matplotlib.pyplot")

    def generate(self, style="2d", nrows=1, ncols=1, **kwargs):
        """Generate new axis instance."""
        if style == "triple":
            axes = []
            grid = self.gridspec(3, 2)
            grid.update(wspace=0.2, hspace=0.05)
            axes.append(self.plt.subplot(grid[0, 1]))
            axes.append(self.plt.subplot(grid[1, 1]))
            axes.append(self.plt.subplot(grid[2, 1]))
            axes.append(self.plt.subplot(grid[:, 0]))
            self.axes = axes
            self.fig = self.gcf()
            return self.axes
        if style == "3d":
            kwargs["subplot_kw"] = {"projection": "3d"}
        aspect = kwargs.pop("aspect", None)
        if aspect is not None:
            kwargs["figsize"] = import_module("matplotlib.figure").figaspect(aspect)
        self.fig, self.axes = self.plt.subplots(nrows, ncols, **kwargs)
        self.set_style(style)
        return self.axes

    def gcf(self):
        """Link fig instance to current figure and return."""
        self._fig = import_module("matplotlib.pyplot").gcf()
        return self._fig

    def gca(self):
        """Link axes instance to current axes and return."""
        self._axes = import_module("matplotlib.pyplot").gca()
        return self._axes

    def despine(self, axes=None):
        """Remove spines from axes instance."""
        sns = import_module("seaborn")
        if axes is None:
            axes = self.axes
        for _axes in np.atleast_1d(axes):
            sns.despine(ax=_axes)

    @staticmethod
    def _set_style(style, axes):
        """Set style on single axes instance."""
        match style:
            case "1d":
                axes.set_aspect("auto")
                axes.axis("on")
            case "2d" | "3d":
                axes.set_aspect("equal")
                axes.axis("off")
            case "plan":
                axes.set_aspect("equal")
                axes.axis("on")
                axes.set_xticks([])
                axes.set_yticks([])
            case _:
                raise NotImplementedError(f"style {style} not implemented")

    def set_style(self, style: Optional[str] = None):
        """Set axes style."""
        if style is None:
            style = self.style
        for axes in np.atleast_1d(self.axes):
            self._set_style(style, axes)
        if style in ["1d", "plan"]:
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

    @staticmethod
    def get_limit(axes):
        """Return axes limits."""
        return {"x": axes.get_xlim(), "y": axes.get_ylim()}

    @staticmethod
    def set_limit(axes, limit):
        """Set axes limits."""
        axes.set_xlim(limit["x"])
        axes.set_ylim(limit["y"])


@dataclass
class MatPlotLib:
    """Manage matplotlib libaries."""

    def __getitem__(self, key: str):
        """Get item from matplotlib collections libary."""
        if "Collection" in key:
            return getattr(self.collections, key)
        return import_module(f"matplotlib.{key}")

    @cached_property
    def collections(self):
        """Return matplotlib collections."""
        return import_module("matplotlib.collections")


@dataclass
class MoviePy:
    """Manage moviepy libaries."""

    @cached_property
    def editor(self):
        """Provide access to moviepy editor."""
        return import_module("moviepy.editor")

    @cached_property
    def videoclip(self):
        """Provide access to moviepy VideoClip class."""
        return self.editor.VideoClip

    @cached_property
    def bindings(self):
        """Provide access to moviepy video io bindings."""
        return import_module("moviepy.video.io.bindings")

    def mplfig_to_npimage(self, fig):
        """Return mplfig as npimage."""
        return self.bindings.mplfig_to_npimage(fig)


class FancyArrowPatch3D(FancyArrowPatch):
    """Draw 3D arrows https://stackoverflow.com/a/22867877/5025009."""

    def __init__(self, posA, posB, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._points = np.c_[posA, posB].T

    def draw(self, renderer):
        """Extend FancyArrowpatch for 3d renderer."""
        _points2d = proj3d.proj_transform(*self._points, renderer.M)
        self.set_positions(_points2d[0][:2], _points2d[1][:2])
        super().draw(renderer)


@dataclass
class Plot:
    """Manage plot workflow."""

    @cached_property
    def mpl_axes(self):
        """Return Axes instance."""
        return Axes()

    @cached_property
    def mpl(self):
        """Return matplotlib instance."""
        return MatPlotLib()

    @cached_property
    def plt(self):
        """Return pylab instance."""
        return self.mpl["pylab"]

    @contextmanager
    def test_plot(self):
        """Return plot, close plot, and reset interactive status."""
        _interactive = self.plt.isinteractive()
        self.plt.ioff()
        yield
        self.plt.close()
        del self.plt
        if _interactive:
            self.plt.ion()

    @cached_property
    def mpy(self):
        """Return moviepy instance."""
        return MoviePy()

    @cached_property
    def patch(self):
        """Provice acces to descartes PolygonPatch class."""
        return import_module("descartes").PolygonPatch

    def arrow(
        self,
        point: np.ndarray,
        vector: np.ndarray,
        scale=1,
        norm=None,
        axes=None,
        **kwargs,
    ):
        """Plot force vectors and intergration points."""
        style = kwargs.get("style", f"{vector.shape[1]}d")
        self.get_axes(style, axes)
        if norm is None:
            norm = np.max(np.linalg.norm(vector, axis=1))
        length = scale * vector / norm
        match style:
            case "2d":
                patch = self.mpl["patches"].FancyArrowPatch
                collection = self.mpl.collections.PatchCollection
            case "3d":
                patch = FancyArrowPatch3D
                collection = Patch3DCollection
            case _:
                raise NotImplementedError(f"arrow not implemented for {style} style")
        arrows = [
            patch(
                _point,
                _point + _length,
                mutation_scale=1,
                arrowstyle="simple,head_length=0.4, head_width=0.3," " tail_width=0.1",
                shrinkA=0,
                shrinkB=0,
            )
            for _point, _length in zip(point, length)
        ]
        collections = collection(arrows, facecolor="black", edgecolor="darkgray")
        self.axes.add_collection(collections)
        return norm

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

    def set_box_aspect(self):
        """Set equal aspect ratio for 3d axes."""
        self.axes.set_box_aspect(
            [np.ptp(getattr(self.axes, f"get_{attr}lim3d")()) for attr in "xyz"]
        )

    def legend(self, *args, **Kwargs):
        """Expose axes legend."""
        self.mpl_axes.legend(*args, **Kwargs)

    @property
    def axes_limit(self):
        """Mange axes limits."""
        if isinstance(self.axes, list | np.ndarray):
            limits = []
            for axes in self.axes:
                limits.append(Axes.get_limit(axes))
            return limits
        return Axes.get_limit(self.axes)

    @axes_limit.setter
    def axes_limit(self, limits):
        if isinstance(limits, list | np.ndarray):
            for axes, limit in zip(self.axes, limits):
                Axes.set_limit(axes, limit)
            return
        Axes.set_limit(self.axes, limits)

    def savefig(self, *args, **kwargs):
        """Save figure to file."""
        self.plt.savefig(*args, **kwargs)


class Plot1D(Plot):
    """Generate axes for 1d line objects."""

    @cached_property
    def mpl_axes(self):
        """Override Plot.Axes instance."""
        return Axes("1d")


class Plot2D(Plot):
    """Generate axes for 2d image objects."""

    @cached_property
    def mpl_axes(self):
        """Override Plot.Axes instance."""
        return Axes("2d")


class Plot3D(Plot):
    """Generate axes for 3d image objects."""

    @cached_property
    def mpl_axes(self):
        """Override Plot.Axes instance."""
        return Axes("3d")


@dataclass
class Animate(MoviePy, Plot):
    """Manage animation workflow."""

    fps: float = 10
    data: xarray.Dataset = field(default_factory=xarray.Dataset, repr=False)
    _segments: dict[str, np.ndarray] = field(
        init=False, default_factory=dict, repr=False
    )

    def __getitem__(self, attr: str):
        """Return item from data."""
        if hasattr(super(), "__getitem__"):
            return super().__getitem__(attr)
        try:
            return self.data[attr]
        except KeyError:
            return getattr(self, attr)

    @property
    def duration(self):
        """Manage animation duration."""
        if len(self._segments) == 0:
            return 0
        return np.max([value[-1, 0] for value in self._segments.values()])

    @duration.setter
    def duration(self, duration):
        if (current_duration := self.duration) == duration:
            return
        factor = duration / current_duration
        for value in self._segments.values():
            value[:, 0] *= factor

    def _animation_time(self, time, append: bool, num=50):
        """Return animation time vector."""
        match time:
            case int() | float() as duration:
                time = np.linspace(0, duration, num)
            case int() | float() as start, int() | float() as end:
                time = np.linspace(start, end, num)
        if append:
            time += self.duration
        return time

    def _animation_value(self, num: int, **kwargs):
        """Return animation value vector."""
        match kwargs:
            case {"amplitude": amplitude, **other}:
                repeat = other.get("repeat", 1)
                angle = repeat * np.linspace(0, 2 * np.pi, num)
                value = amplitude * np.sin(angle)
            case {"ramp": ramp}:
                transition = np.linspace(0, 1, num)
                value = ramp * transition
            case _:
                raise NotImplementedError(f"segment not implemented for {kwargs}")
        return value

    def get_itime(self, time):
        """Extend to return time index."""
        try:
            return super().get_itime(time)
        except AttributeError:
            raise NotImplementedError()

    def _animation_offset(self, attr: str, time: int | float) -> float:
        """Return attribute offset at current duration."""
        if "time" in self._segments:
            time = self.interp1d("time")(time)
            if attr == "time":
                return time
        try:
            itime = self.get_itime(time)
            return self.data[attr][itime].data.item()
        except (NotImplementedError, AttributeError):
            pass
        try:
            return self[attr]
        except AttributeError:
            return 0

    def add_animation(
        self, attr: str, time, append=True, offset: bool | float = True, **kwargs
    ):
        """Add moviepy animation segment."""
        num = kwargs.pop("num", 50)
        time = self._animation_time(time, append, num)
        value = self._animation_value(num, **kwargs)
        match offset:
            case True:
                value += self._animation_offset(attr, time[0])
            case int() | float():
                value += offset
        data = np.c_[time, value]
        if attr != "time":
            data = np.r_[
                [[time[0], np.nan]],
                data,
                [[time[-1] + 2.0 * np.finfo(np.float_).eps, np.nan]],
            ]
        if attr in self._segments:
            data = np.r_[self._segments[attr], data]
        self._segments[attr] = data

    def interp1d(self, attr: str):
        """Return animation segment interpolator."""
        waveform = self._segments[attr]
        return import_module("scipy.interpolate").interp1d(
            *waveform.T,
            bounds_error=False,
            fill_value=(waveform[0, 1], waveform[-1, 1]),
        )

    def scene(self, time: float):
        """Return animation scene at requested time-point."""
        return {
            attr: value
            for attr in self._segments
            if np.isfinite(value := float(self.interp1d(attr)(time)))
        }

    def plot_animation(self, skip_time=True):
        """Plot animation segments."""
        if self.duration == 0:
            return
        self.set_axes("1d")
        time = np.linspace(0, self.duration, 150)
        for attr in self._segments:
            if attr == "time" and skip_time:
                continue
            self.axes.plot(time, self.interp1d(attr)(time), label=attr)
        self.axes.legend()

    def make_frame(self, time):
        """Override method in child class."""
        raise NotImplementedError("animate method requies make_frame")

    def _make_frame(self, time):
        """Make frame for animation."""
        if isinstance(self.axes, list):
            for axes in self.axes:
                axes.clear()
        else:
            self.axes.clear()
        self.make_frame(time)
        self.fig.tight_layout(pad=0)
        return self.mplfig_to_npimage(self.fig)

    def animate(self, filename=None, duration=None):
        """Generate animiation."""
        if filename is None:
            filename = self.__class__.__name__.lower()
        if duration is None:
            duration = self.duration
        else:
            self.duration = duration
        if duration == 0:
            raise ValueError("no animation segments.")
        animation = self.videoclip(self._make_frame, duration=duration)
        animation.write_gif(f"{filename}.gif", fps=self.fps)
