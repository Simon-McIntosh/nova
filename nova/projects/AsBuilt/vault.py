"""Create assembly graphics."""
from dataclasses import dataclass, field
from typing import Union

import descartes
import numpy as np
import numpy.typing as npt
from matplotlib.collections import PatchCollection
import xarray

from nova.geometry.polygon import Polygon

from nova.utilities.plotter import ImagePlot


@dataclass
class Wedge(ImagePlot):
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

    def plot(self, color='gray', linestyle='-', marker='X', patch=True):
        """Plot single polygon."""
        self.axes.plot(*self.data.boundary[:2],
                       color=color, linestyle=linestyle)
        self.axes.plot(*self.data.centroid[:2],
                       color=color, marker=marker)

        if patch:
            patch = [descartes.PolygonPatch(
                Polygon(self.data.boundary[:2]).poly.__geo_interface__,
                color=color, alpha=0.2)]
            patch_collection = PatchCollection(patch, match_original=True)
            self.axes.add_collection(patch_collection)
            self.axes.autoscale_view()


@dataclass
class ErrorWedge(Wedge):
    """Adjust wedge by placement error."""

    error: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Adjust polygon by error dict."""
        for attr in self.error:
            setattr(self, attr, getattr(self, attr) + self.error[attr])
        super().__post_init__()
        self.data.attrs |= {f'{attr}_error': self.error[attr]
                            for attr in self.error}


@dataclass
class BaseVault:
    """Generate poly representation of TF coil vault."""

    radius: npt.ArrayLike
    phi: npt.ArrayLike
    delta_radius: npt.ArrayLike
    delta_phi: npt.ArrayLike
    ncoil: int = 18
    resolution: int = 15
    data: xarray.Dataset = field(default_factory=xarray.Dataset)

    def __post_init__(self):
        """Built TF coil vault."""
        self.data = xarray.Dataset(
            dict(coil_index=range(self.ncoil),
                 coil_coord=['radius', 'phi', 'delta_radius', 'delta_phi']))
        self.data['coil'] = ('coil_index', 'coil_coord'), \
            [self.radius, self.phi, self.delta_radius, self.delta_phi]
        '''
        self.data['coil'] = xarray.DataArray(0., self.data.coords)
        self.data['coil'][:, 0] = self.radius
        self.data['coil'][:, 1] = self.phi
        self.data['coil'][:, 2] = self.delta_radius
        self.data['coil'][:, 3] = self.delta_phi
        '''
        wedge = []
        for coil in self.data.coil:
            wedge.append(Wedge(*coil.data).data)
        vault = xarray.concat(wedge, self.data.coil_index)
        self.data = xarray.merge([self.data, vault])

    def plot(self):
        """Plot vault."""
        for coil in self.data.coil:
            print(coil.data)
            Wedge(*coil.data).plot()


@dataclass
class UniformVault(BaseVault):
    """Generate poly representation of TF coil vault."""

    radius: float
    phi: float
    delta_radius: float
    delta_phi: float

@dataclass
class ErrorVault:
    """Generate poly representation of TF coil vault."""

    radius: float = 0.824
    delta_radius: float = 0.3
    delta_phi: float = np.pi/9
    referance_gap: float = 2.0
    gap_factor: float = 10
    ncoil: int = 18
    data: xarray.Dataset = field(default_factory=xarray.Dataset)

    def __post_init__(self):
        """Built TF coil vault."""
        data = []
        for coil in range(self.ncoil):
            data.append()



if __name__ == '__main__':

    vault = Vault(0.824, 0, 0.3, np.pi/9)
    vault.plot()
    #wedge = ErrorWedge(0.824, 0, 0.2, np.pi/9, error=dict(delta_radius=-0.1))
    #wedge.plot()
