"""Methods for ploting FrameSpace data."""
from dataclasses import dataclass, field
from collections import Counter
import colorsys

import numpy as np

from nova.frame import metamethod
from nova.frame.dataframe import DataFrame
from nova.graphics.plot import BasePlot, Plot, Properties

# pylint:disable=unsubscriptable-object


@dataclass
class Labels:
    """Manage polyplot labels."""

    current_unit: str = "A"
    field_unit: bool | str = "T"
    zeroturn: bool = False
    options: dict[str, str | int | float] = field(
        repr=False, default_factory=lambda: {"font_size": "medium", "label_limit": 20}
    )

    def __post_init__(self):
        """Update plot flags."""
        self.update_flags()
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def update_flags(self):
        """Update plot attribute flags."""
        if hasattr(self, "biot"):
            if "field" not in self.biot:
                self.field_unit = None
        if "energize" not in self.frame.attrs:
            self.current_unit = None

    def patch_number(self, parts):
        """Return patch number for each part."""
        return Counter(parts)

    def label(self):
        """Add plot labels."""
        index = self.get_index()
        self.frame.part[index]
        """
        print(parts)

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
        """


@dataclass
class PolyPlot(Plot, Properties, Labels, metamethod.PolyPlot, BasePlot):
    """Methods for ploting FrameSpace data."""

    frame: DataFrame = field(repr=False)
    additional: list[str] = field(default_factory=lambda: ["part"])
    rng: np.random.Generator = np.random.default_rng(2025)

    def initialize(self):
        """Initialize metamethod."""
        self.update_columns()
        if "frame" not in self.frame or len(self.frame) == 1:
            self.patchwork = 0

    def update_columns(self):
        """Update frame columns."""
        unset = [
            attr not in self.frame.columns for attr in self.required + self.additional
        ]
        if np.array(unset).any():
            self.frame.update_columns()

    def __call__(self, index=slice(None), axes=None, **kwargs):
        """Plot frame if not empty."""
        if not self.frame.empty:
            self.plot(index, axes, **kwargs)

    def plot(self, index=slice(None), axes=None, zeroturn=None, **kwargs):
        """
        Plot frame.

        Addapted from geoplot.PolygonPatch.
        """
        index = self.get_index(index, zeroturn=zeroturn)
        if sum(index) == 0:
            return
        self.get_axes("2d", axes)
        patch = []
        properties = self.patch_properties(self.frame.part, self.frame.area)
        basecolor = {part: properties[part]["facecolor"] for part in properties}
        for polyframe, part in self.frame.loc[index, ["poly", "part"]].to_numpy():
            patch_kwargs = properties[part].copy()
            if self.patchwork != 0:  # Shuffle basecolor
                patch_kwargs["facecolor"] = self.shuffle(basecolor[part])
            patch_kwargs |= kwargs
            try:  # MultiPolygon.
                for _poly in polyframe.poly:
                    assert False  # TODO remove branch if not triggered
                    patch.append(self.patch(_poly.__geo_interface__, **patch_kwargs))
            except (TypeError, AssertionError):  # Polygon.
                patch.append(
                    self.patch(polyframe.poly.__geo_interface__, **patch_kwargs)
                )
        patch_collection = self.mpl["PatchCollection"](patch, match_original=True)
        self.axes.add_collection(patch_collection)
        self.axes.autoscale_view()
        self.label()

    def shuffle(self, color):
        """Return shuffled facecolor. Alternate lightness by +-factor."""
        factor = (1 - 2 * self.rng.random(1)[0]) * self.patchwork
        color = colorsys.rgb_to_hls(*self.mpl["colors"].to_rgb(color))
        color = colorsys.hls_to_rgb(
            color[0], max(0, min(1, (1 + factor) * color[1])), color[2]
        )
        return color


if __name__ == "__main__":
    from nova.frame.coilset import CoilSet

    coilset = CoilSet()
    coilset.coil.insert([1, 2, 3], 0, 0.1, 0.3)
    coilset.plot()
