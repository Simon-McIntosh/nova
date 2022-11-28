"""Manage plotting objects."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import matplotlib.pyplot as plt


@dataclass
class BaseAxes(ABC):
    """Manage plot axes."""

    _axes: plt.Axes = field(init=False, repr=False, default=None)
    _figure: plt.Figure = field(init=False, repr=False, default=None)

    @abstractmethod
    def generate_axes(self):
        """Generate axes if unset."""
        if self._axes is None or self._figure is None:
            self._figure, self._axes = plt.subplots(1, 1)
            self._axes.set_aspect('equal')
            self._axes.axis('off')

    @property
    def figure(self):
        """Manage figure handle."""
        self.generate_axes()
        #return self._figure
        return plt.gcf()

    @property
    def axes(self):
        """Manage plot axes."""
        self.generate_axes()
        return self._axes

    @axes.setter
    def axes(self, axes):
        self._axes = axes


class LinePlot(BaseAxes):
    """Generate axes for 2d line objects."""

    def generate_axes(self):
        """Generate axes if unset."""
        if self._axes is None:
            self._axes = plt.gca()
            self._axes.axis('on')
            plt.despine()


class ImagePlot(BaseAxes):
    """Generate axes for 2d image objects."""

    def generate_axes(self):
        """Generate axes if unset."""
        if self._axes is None:
            self._axes = plt.gca()
            self._axes.set_aspect('equal')
            self._axes.axis('off')
