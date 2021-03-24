"""Methods for ploting Frame data."""
from dataclasses import dataclass, field
import colorsys
import functools
import operator

from descartes import PolygonPatch
import numpy as np
import matplotlib
from matplotlib.collections import PatchCollection
import matplotlib.colors as mc
import pandas
import shapely.geometry

from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.polygon import Polygon
from nova.electromagnetic.polygen import polygen, polyframe, root_mean_square

from nova.electromagnetic.frame import FrameSet
from nova.utilities.IO import human_format
from nova.utilities.pyplot import plt

# pylint:disable=unsubscriptable-object


@dataclass
class Display:
    """Manage axes parameters."""

    axes: plt.Axes = field(repr=False, default=None)
    patchwork: float = 0.15
    alpha: dict[str, float] = field(default_factory=lambda: {'plasma': 0.75})
    linewidth: float = 0.25
    edgecolor: str = 'darkgrey'
    color: dict[str, str] = field(default_factory=lambda: {
        'VS3': 'C0', 'VS3j': 'gray', 'CS': 'C0', 'PF': 'C0',
        'trs': 'C3', 'dir': 'C3', 'vv': 'C3', 'vvin': 'C3',
        'vvout': 'C3', 'bb': 'C7', 'plasma': 'C4', 'Plasma': 'C4',
        'cryo': 'C5'})
    zorder: dict[str, int] = field(default_factory=lambda: {
        'VS3': 1, 'VS3j': 0, 'CS': 3, 'PF': 2})

    def get_alpha(self, part):
        """Return patch alpha."""
        return self.alpha.get(part, 1)

    def get_color(self, part):
        """Return patch facecolor."""
        return self.color.get(part, 'C9')

    def get_zorder(self, part):
        """Return patch zorder."""
        return self.zorder.get(part, 0)


@dataclass
class Flags:
    """Manage display flags."""

    overwrite: bool = False
    zeroturn: bool = False
    feedback: bool = False


@dataclass
class Labels:

    label: str = 'coil'
    current: str = 'A'
    field: bool = True
    font_size: str = 'medium'
    label_limit: int = 20




    def label_coil(self, ax, label='coil',  field=True,
               coil=None, Nmax=20):


        parts = parts.unique()
        parts = list(parts)
        N = {p: sum(coil.part == p) for p in parts}
        # check for presence of field instance
        field = False if 'field' not in self.biot_instances else field
        # referance vertical length scale
        dz_ref = np.diff(ax.get_ylim())[0] / 100
        nz = np.sum(np.array([parts is not False, current is not None,
                              field is not False]))
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

@dataclass
class PlotFrame(Display, Flags, MetaMethod):
    """Methods for ploting Frame data."""

    frame: FrameSet = field(repr=False)
    required: list[str] = field(default_factory=lambda: ['x', 'z', 'poly'])
    additional: list[str] = field(default_factory=lambda: [
        'part', 'patch', 'feedback', 'Nt'])

    def initialize(self):
        """Initialize metamethod."""
        self.frame.update_columns()
        if 'frame' not in self.frame:
            self.patchwork = 0

    def generate_axes(self):
        """Generate axes if unset."""
        if self._axes is None:
            self._axes = plt.gca()
            self._axes.set_aspect('equal')
            self._axes.axis('off')

    @property
    def axes(self):
        """Return plot axes."""
        self.generate_axes()
        return self._axes

    @axes.setter
    def axes(self, axes):
        self._axes = axes

    def __call__(self, axes=None, **kwargs):
        """Plot frame if not empty."""
        if not self.frame.empty:
            self.plot(axes, **kwargs)

    def plot(self, axes, **kwargs):
        """Plot frame."""
        self.axes = axes
        if self.update_patch():
            self.patch()  # patch on-demand
        index = np.full(len(self.frame), True)
        if not self.zeroturn:  # exclude zeroturn filaments (Nt == 0)
            index &= (self.frame.loc[:, 'Nt'] != 0)
        if not self.feedback:  # exclude stabilization coils
            index &= ~self.frame.feedback
        patch = self.frame.loc[index, 'patch']
        # form list of lists
        patch = [_patch if pandas.api.types.is_list_like(_patch)
                 else [_patch] for _patch in patch]
        if len(patch) > 0:  # flatten and sort
            patch = functools.reduce(operator.concat, patch)
            patch = np.array(patch)[np.argsort([p.zorder for p in patch])]
            patch_collection = PatchCollection(patch, match_original=True)
            self.axes.add_collection(patch_collection, autolim=True)
            self.axes.autoscale_view()

    def patch(self):
        """Update frame patch, call on-demand."""
        patch = [[] for __ in range(len(self.frame))]
        for i, (current_patch, poly, part) in enumerate(
                self.frame.loc[:, ['patch', 'poly', 'part']].values):
            if self.overwrite or self.update_patch(current_patch):
                if isinstance(poly, dict):
                    poly = shapely.geometry.shape(poly)
                if isinstance(poly, shapely.geometry.Polygon):
                    patch[i] = [PolygonPatch(poly)]
                else:
                    patch[i] = []
            else:
                patch[i] = [current_patch]
            for j in range(len(patch[i])):
                patch[i][j].set_edgecolor(self.edgecolor)
                patch[i][j].set_linewidth(self.linewidth)
                patch[i][j].set_antialiased(True)
                patch[i][j].set_facecolor(self.get_color(part))
                patch[i][j].set_zorder = self.get_zorder(part)
                patch[i][j].set_alpha(self.get_alpha(part))
                if self.patchwork != 0:
                    self.shuffle(patch[i][j])
        self.frame.loc[:, 'patch'] = np.asarray(patch, object)

    def update_patch(self, patch=None):
        """Return True if any patches are null else False."""
        if patch is None:
            patch = self.frame.patch
        return np.array(pandas.isnull(patch)).any()

    def shuffle(self, patch):
        """Update patch facecolor. Alternate lightness by +- factor."""
        factor = (1 - 2 * np.random.rand(1)[0]) * self.patchwork
        c = patch.get_facecolor()
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        c = colorsys.hls_to_rgb(
                c[0], max(0, min(1, (1 + factor) * c[1])), c[2])
        patch.set_facecolor(c)


if __name__ == '__main__':

    frame = FrameSet(Required=['x', 'z'], additional=['section', 'Ic'])
    frame.insert(range(2), 1, label='PF')
    frame.insert(range(2), 0)

    plot = PlotFrame(frame)
    plot.initialize()

    plot()
