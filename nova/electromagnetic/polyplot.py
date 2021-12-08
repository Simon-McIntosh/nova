"""Methods for ploting FrameSpace data."""
from dataclasses import dataclass, field
import colorsys

import descartes
import numpy as np
from matplotlib.collections import PatchCollection
import matplotlib.colors as mc

from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.baseplot import Axes, Display, Label, BasePlot

# pylint:disable=unsubscriptable-object


@dataclass
class PolyPlot(Axes, Display, Label, MetaMethod, BasePlot):
    """Methods for ploting FrameSpace data."""

    name = 'polyplot'

    frame: DataFrame = field(repr=False)
    required: list[str] = field(default_factory=lambda: ['poly'])
    additional: list[str] = field(default_factory=lambda: ['part'])
    rng: np.random.Generator = np.random.default_rng(2025)

    def initialize(self):
        """Initialize metamethod."""
        self.update_columns()
        if 'frame' not in self.frame or len(self.frame) == 1:
            self.patchwork = 0

    def update_columns(self):
        """Update frame columns."""
        unset = [attr not in self.frame.columns
                 for attr in self.required + self.additional]
        if np.array(unset).any():
            self.frame.update_columns()

    def __call__(self, index=slice(None), axes=None, **kwargs):
        """Plot frame if not empty."""
        if not self.frame.empty:
            self.plot(index, axes, **kwargs)

    def plot(self, index=slice(None), axes=None, **kwargs):
        """
        Plot frame.

        Addapted from geoplot.PolygonPatch.
        """
        index = self.get_index(index)  # retrieve frame index
        if sum(index) == 0:
            return
        self.axes = axes  # set axes
        patch = []
        properties = self.patch_properties(self.frame.part)
        basecolor = {part: properties[part]['facecolor']
                     for part in properties}
        for polyframe, part in self.frame.loc[index,
                                              ['poly', 'part']].to_numpy():
            patch_kwargs = properties[part].copy()
            if self.patchwork != 0:  # Shuffle basecolor
                patch_kwargs['facecolor'] = self.shuffle(basecolor[part])
            patch_kwargs |= kwargs
            try:  # MultiPolygon.
                for _poly in polyframe.poly:
                    patch.append(descartes.PolygonPatch(
                        _poly.__geo_interface__, **patch_kwargs))
            except (TypeError, AssertionError):  # Polygon.
                patch.append(descartes.PolygonPatch(
                    polyframe.poly.__geo_interface__, **patch_kwargs))
        patch_collection = PatchCollection(patch, match_original=True)
        self.axes.add_collection(patch_collection)
        self.axes.autoscale_view()
        if self.label:
            self.add_label()

    def shuffle(self, color):
        """Return shuffled facecolor. Alternate lightness by +-factor."""
        factor = (1 - 2 * self.rng.random(1)[0]) * self.patchwork
        color = colorsys.rgb_to_hls(*mc.to_rgb(color))
        color = colorsys.hls_to_rgb(
                color[0], max(0, min(1, (1 + factor) * color[1])), color[2])
        return color
