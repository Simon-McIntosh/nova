"""Create assembly graphics."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import descartes
from matplotlib.collections import PatchCollection
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

import numpy as np
import numpy.typing as npt
import xarray

from nova.geometry.polygon import Polygon
from nova.graphics.plot import Plot2D


@dataclass
class Wedge(Plot2D):
    """Generate wedge polygons."""

    radius: float
    phi: float
    delta_radius: float
    delta_phi: float
    resolution: int = 15
    data: xarray.DataArray = field(default_factory=xarray.DataArray)

    def __post_init__(self):
        """Generate wedge polygon."""
        phi_space = np.linspace(self.phi - self.delta_phi/2,
                                self.phi + self.delta_phi/2, self.resolution)
        radius_space = self.radius * np.ones(self.resolution)
        phi = np.append(phi_space, phi_space[::-1])
        radius = np.append(radius_space - self.delta_radius/2,
                           radius_space + self.delta_radius/2)
        phi = np.append(phi, phi[0])
        radius = np.append(radius, radius[0])
        self.data = xarray.Dataset(attrs=self.attrs)
        self.data['boundary_index'] = range(len(phi))
        self.data['boundary_coord'] = ['x', 'y', 'radius', 'phi']
        self.data['boundary'] = ('boundary_coord', 'boundary_index'), \
            [radius*np.cos(phi), radius*np.sin(phi), radius, phi]
        self.data['centroid'] = 'boundary_coord', \
            [self.radius*np.cos(self.phi), self.radius*np.sin(self.phi),
             self.radius, self.phi]

    @property
    def attrs(self):
        """Return wedge attributes."""
        return {attr: getattr(self, attr) for attr in
                ['radius', 'phi', 'delta_radius', 'delta_phi', 'resolution']}

    def plot(self, color='gray', linestyle='-',
             marker='X', markerlinewidth=0.3,
             patch_color='gray', patch_alpha=0.2):
        """Plot single polygon."""
        if linestyle is not None:
            self.axes.plot(*self.data.boundary[:2],
                           color=color, linestyle=linestyle)
        if marker is not None:
            self.axes.plot(*self.data.centroid[:2],
                           color=color, marker=marker)
        if patch_color is not None:
            patch = [descartes.PolygonPatch(
                Polygon(self.data.boundary[:2]).poly.__geo_interface__,
                color=patch_color, alpha=patch_alpha)]
            patch_collection = PatchCollection(patch, match_original=True)
            self.axes.add_collection(patch_collection)
            self.axes.autoscale_view()


@dataclass
class BaseVault:
    """Generate poly representation of TF coil vault."""

    radius: npt.ArrayLike
    phi: npt.ArrayLike
    delta_radius: npt.ArrayLike
    delta_phi: npt.ArrayLike
    ncoil: int = 18
    resolution: int = 15
    coil_index: list[int] = None
    data: xarray.Dataset = field(default_factory=xarray.Dataset)

    def __post_init__(self):
        """Built TF coil vault."""
        self.data = xarray.Dataset(
            dict(coil_index=range(self.ncoil),
                 coil_coord=['radius', 'phi', 'delta_radius', 'delta_phi']))
        self.data['coil'] = ('coil_index', 'coil_coord'), \
            np.array([self.radius, self.phi,
                      self.delta_radius, self.delta_phi]).T
        wedge = []
        for coil in self.data.coil:
            wedge.append(Wedge(*coil.data).data)
        vault = xarray.concat(wedge, self.data.coil_index)
        self.data = xarray.merge([self.data, vault])

    @property
    def plot_kwargs(self):
        """Return default plot kwargs."""
        return dict(color='grey', linestyle='-', marker='.',
                    patch_color='grey')

    def plot(self, **kwargs):
        """Plot vault."""
        kwargs = self.plot_kwargs | kwargs
        coil_index = kwargs.pop('coil_index', self.coil_index)
        if self.coil_index is None:
            coil_index = self.data.coil_index
        for index in coil_index:
            Wedge(*self.data.coil[index].data).plot(**kwargs)


@dataclass
class UniformVault(BaseVault):
    """Generate poly representation of TF coil vault."""

    radius: float
    delta_radius: float
    delta_phi: float
    phase: float = np.pi/18
    phi: float = field(init=False)

    def __post_init__(self):
        """Elevate float input to uniform spaced arrays."""
        self.radius *= np.ones(self.ncoil)
        self.delta_radius *= np.ones(self.ncoil)
        self.phi = np.linspace(0, 2*np.pi, self.ncoil, endpoint=False)
        self.phi += self.phase
        self.delta_phi *= np.ones(self.ncoil)
        super().__post_init__()

    @property
    def plot_kwargs(self):
        """Extend BaseVaut default plot kwargs."""
        plot_kwargs = dict(color='grey', linestyle=':', marker='.')
        return super().plot_kwargs | plot_kwargs


@dataclass
class BaseAssembly:
    """Plot Uniform vault with placement windows."""

    radius: float = 0.824
    delta_radius: float = 0.3
    gap: float = 0.05
    radial_window: float = 0.1
    toroidal_window: float = 0.075
    coil_index: list[int] = None
    ncoil: int = 18

    def __post_init__(self):
        """Translate gap to delta_phi."""
        self.delta_phi = 2*np.pi/self.ncoil - self.gap/self.radius
        self.build()

    def build(self):
        """Place coils and allignment windows."""
        self.vault = UniformVault(self.radius, self.delta_radius,
                                  self.delta_phi, ncoil=self.ncoil,
                                  coil_index=self.coil_index)
        self.window = UniformVault(self.radius, self.radial_window,
                                   self.toroidal_window/self.radius,
                                   ncoil=self.ncoil,
                                   coil_index=self.coil_index)

    def plot(self):
        """Plot referance vault."""
        self.plot_vault()
        self.plot_window()

    def plot_vault(self):
        """Plot vault."""
        self.vault.plot(marker='.')

    def plot_window(self):
        """Plot placement window."""
        self.window.plot(patch_color='C3', marker=None, linestyle=None)


@dataclass
class Animate(ImagePlot):
    """Animation base class."""

    duration: float = 5.
    fps: float = 10.
    samples: int = None

    @abstractmethod
    def sample(self, index=0):
        """Draw sample."""

    @abstractmethod
    def plot(self):
        """Generate plot."""

    def make_frame(self, time: float):
        """Return single frame."""
        self.axes.clear()
        index = int(self.samples * time/self.duration)
        self.sample(index)
        self.plot()
        self.figure.tight_layout(pad=-0.5)
        return mplfig_to_npimage(self.figure)

    def movie(self, filename='tf_assembly'):
        """Make movie."""
        animation = VideoClip(self.make_frame, duration=self.duration)
        animation.write_gif(f'{filename}.gif', fps=self.fps)


@dataclass
class BaseSample(BaseAssembly, Animate):
    """Sample vault assembly error."""

    error: dict[str, float] = field(default_factory=lambda: dict(
        radius=0.02, rphi=0.02, delta_rphi=0.02))
    sead: int = 2025
    samples: int = 20
    expand: float = 0.1
    data: xarray.Dataset = field(default_factory=xarray.Dataset)

    def __post_init__(self):
        """Adjust polygon by error dict."""
        super().__post_init__()
        self.reference_vault = self.vault
        boundary = self.reference_vault.data.boundary[self.coil_index]
        self._axis = [boundary[:, 0].min(), boundary[:, 0].max(),
                      boundary[:, 1].min(), boundary[:, 1].max()]
        delta = [np.diff(self._axis[:2])[0], np.diff(self._axis[2:])[0]]
        self._axis += self.expand * np.array(
            [-delta[0], delta[0], -delta[1], delta[1]])
        self.phi = self.vault.phi
        rng = np.random.default_rng(self.sead)
        self.data = xarray.Dataset(
            dict(sample_index=range(self.samples),
                 coord=['radius', 'phi', 'delta_radius', 'delta_phi'],
                 coil_index=range(self.ncoil)),
            attrs=dict(sead=self.sead))
        self.data['sample'] = xarray.DataArray(0., self.data.coords)
        self.error |= {attr.replace('rphi', 'phi'):
                       self.error[attr]/self.radius for attr in self.error
                       if 'rphi' in attr}
        for i, attr in enumerate(self.data.coord.values):
            self.data['sample'][:, i, :] = getattr(self, attr)
            if attr in self.error:
                error = self.error[attr]
                self.data.sample[:, i, :] += \
                    rng.uniform(-error, error, (self.samples, self.ncoil))
        self.sample()

    def sample(self, index=0):
        """Draw sample."""
        self.vault = BaseVault(*self.data.sample[index], ncoil=self.ncoil,
                               coil_index=self.coil_index)

    def plot(self):
        """Plot sample."""
        self.reference_vault.plot(coil_index=self.coil_index,
                                  patch_color=None, marker=None)
        super().plot()
        self.axes.axis(self._axis)


@dataclass
class Coil(BaseSample):
    """Visulize coil samples."""

    coil_index: list[int] = field(default_factory=lambda: [0])


@dataclass
class SSAT(BaseSample):
    """Visulize a pair of coils on the SSAT."""

    coil_index: list[int] = field(default_factory=lambda: [0, 1])


@dataclass
class ReferenceSSAT(BaseSample):
    """Visulize SSAT assembly with dataum coil."""

    coil_index: list[int] = field(default_factory=lambda: [0, 1])

    def __post_init__(self):
        """Update window for datum coil case."""
        super().__post_init__()
        self.window = BaseVault(self.radius * np.ones(self.ncoil),
                                self.phi,
                                self.radial_window * np.ones(self.ncoil),
                                [2*self.toroidal_window/self.radius
                                 if i % 2 == 1
                                 else 0 for i in range(self.ncoil)],
                                ncoil=self.ncoil, coil_index=self.coil_index)

    def sample(self, index=0):
        """Draw sample."""
        data = self.data.sample[index].copy()
        data[1, ::2] = self.phi[::2]
        data[1, 1::2] += (data[1, 1::2] - self.phi[1::2])
        self.vault = BaseVault(*data, ncoil=self.ncoil,
                               coil_index=self.coil_index)


@dataclass
class Sector(BaseSample):
    """Visulize placment of a single sector in-pit."""

    ssat_sample: int = 0
    phase: float = np.pi/9
    sector_index: list[int] = field(default_factory=lambda: [0])

    def __post_init__(self):
        """Apply sector displacments."""
        if self.sector_index is None:
            self.sector_index = range(self.ncoil // 2)
        sectors = [[2*sector, 2*sector+1] for sector in self.sector_index]
        self.coil_index = [coil for pair in sectors for coil in pair]
        super().__post_init__()
        self.coil_window = self.window
        self.window = UniformVault(
            self.radius, 1/np.sqrt(2) * self.radial_window,
            1/np.sqrt(2) * self.toroidal_window/self.radius,
            ncoil=self.ncoil // 2, phase=self.phase,
            coil_index=self.sector_index)
        for attr in ['radius', 'phi', 'delta_radius', 'delta_phi']:
            setattr(self, attr,
                    self.data.sample[self.ssat_sample].sel(coord=attr).data)
        rng = np.random.default_rng(self.sead)
        for i, attr in enumerate(self.data.coord.values):
            self.data['sample'][:, i, :] = getattr(self, attr)
            if attr in self.error:
                error = self.error[attr]
                sample = rng.uniform(-error, error,
                                     (self.samples, self.ncoil // 2))
                self.data.sample[:, i, ::2] += sample
                self.data.sample[:, i, 1::2] += sample

    def plot_vault(self):
        """Extend plot vault to disabe coil centroid markers."""
        super().plot_vault()
        coil_centroid = self.vault.data.centroid.values
        sector_centroid = (coil_centroid[::2] + coil_centroid[1::2]) / 2
        self.axes.plot(sector_centroid[self.sector_index, 0],
                       sector_centroid[self.sector_index, 1], '.',
                       color='gray')

    def plot_window(self):
        """Extend plot window."""
        self.coil_window.plot(patch_color='gray', marker=None)
        super().plot_window()


@dataclass
class Assembly(Sector):
    """Plot full vault assembly."""

    sector_index: list[int] = field(default_factory=lambda: range(9))


if __name__ == '__main__':

    assembly = Sector()
    assembly.plot()
    #assembly.movie()
    #wedge = ErrorWedge(0.824, 0, 0.2, np.pi/9, error=dict(delta_radius=-0.1))
    #wedge.plot()
