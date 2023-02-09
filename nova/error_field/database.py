"""Manage error field database."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas
import scipy

from nova.database.datafile import Datafile
from nova.database.filepath import FilePath
from nova.frame.baseplot import Plot


@dataclass
class Database(Plot, Datafile):
    """Interpolate field dataset to first wall and decompose."""

    filename: str
    dirname: str = '.error_field'
    datadir: str = '/mnt/data/error_field'

    library: ClassVar[dict[str, str]] = {
        '86T4WW': 'Database of magnetic field produced by Ferromagnetic '
                  'Inserts magnetized by TF, CS and PF coils in 5MA/1.8T '
                  'H-mode scenario'}

    def reshape(self, vector, shape, value=None):
        """Return vector reshaped as fortran array with axes 1, 2 swapped."""
        return vector.values.reshape(
            (shape[0],) + shape[-2:][::-1], order='F').swapaxes(2, 1)
        '''
        if value is None:
            return np.append(array, array[:, :1, :], axis=1)
        value *= np.ones_like(array)
        return np.append(array, value[:, :1, :], axis=1)
        '''

    def build(self):
        """Build database from source datafile."""
        self.read_datafile()
        self.store()

    def read_datafile(self):
        """Read source datafile."""
        datafile = FilePath(dirname=self.datadir,
                            filename=f'{self.filename}.txt')
        with open(datafile.filepath, 'r') as file:
            header = file.readline()
            data = pandas.read_csv(
                file, header=None, delim_whitespace=True,
                names=['r', 'phi', 'z', 'Br', 'Bphi', 'Bz'])
        shape = tuple(int(dim) for dim in header.split()[:3])
        self.data.attrs['uid'] = self.filename
        self.data.attrs['title'] = self.library[self.filename]
        self.data.attrs['Io'] = float(header.split()[-1])
        self.data.coords['radius'] = self.reshape(data.r, shape)[:, 0, 0]
        self.data.coords['phi'] = self.reshape(data.phi, shape, 360)[0, :, 0]
        self.data.coords['height'] = self.reshape(data.z, shape)[0, 0, :]
        for attr in ['Br', 'Bphi', 'Bz']:
            self.data[f'grid_{attr}'] = ('radius', 'phi', 'height'), \
                self.reshape(data[attr], shape)

    def build_surface(self):
        """Build control surface."""
        datafile = FilePath(dirname=self.datadir, filename='surface.txt')
        with open(datafile.filepath, 'r') as file:
            data = pandas.read_csv(file, header=None, delim_whitespace=True,
                                   names=['radius', 'height'])
        delta = np.zeros((3, len(data)))
        delta[0] = np.roll(data.radius, -1) - np.roll(data.radius, 1)
        delta[2] = np.roll(data.height, -1) - np.roll(data.height, 1)
        normal = np.cross(np.array([0, 1, 0])[np.newaxis, :], delta, axisb=0)
        normal /= np.linalg.norm(normal, axis=1)[:, np.newaxis]
        self.data.coords['coord'] = ['radius', 'height']
        self.data.coords['loop'] = \
            ('index', 'coord'), np.c_[data.radius, data.height]
        self.data.coords['normal'] = \
            ('index', 'coord'), np.c_[normal[:, 0], normal[:, 2]]

    def compose(self):
        """Interpolate field components to control surface."""
        for attr in ['Br', 'Bphi', 'Bz']:
            self.data[f'loop_{attr}'] = \
                ('index', 'phi'), self.data[f'grid_{attr}'].interp(
                    dict(radius=self.data.loop[:, 0],
                         height=self.data.loop[:, 1])).data
        field = np.stack([self.data.loop_Br, self.data.loop_Bz], axis=-1)
        self.data['loop_Bn'] = \
            ('index', 'phi'), np.einsum('ik,ijk->ij', self.data.normal, field)

    def decompose(self):
        """Perform Fourier decomposition."""

    def plot_normal(self, skip=5):
        """Plot loop surface and loop normals."""
        self.set_axes('2d')
        self.axes.plot(self.data.loop[:, 0], self.data.loop[:, 1], 'C0-')
        patch = self.mpl['patches'].FancyArrowPatch
        tail = self.data.loop[::skip]
        length = self.data.normal[::skip]
        arrows = [patch((x, z), (x+dx, z+dz), mutation_scale=0.5,
                        arrowstyle='simple,head_length=0.4, head_width=0.3,'
                        ' tail_width=0.1', shrinkA=0, shrinkB=0)
                  for x, z, dx, dz in
                  zip(tail[:, 0], tail[:, 1], length[:, 0], length[:, 1])]
        collections = self.mpl.collections.PatchCollection(
            arrows, facecolor='black', edgecolor='darkgray')
        self.axes.add_collection(collections)
        self.axes.autoscale_view()

    def plot_trace(self, attr='loop_Bn', index=250):
        """Plot poloidal trace."""

        coef = scipy.fft.rfft(self.data[attr].data)
        coef[:, 36:] = 0
        data = scipy.fft.irfft(coef)

        print(data.shape, self.data[attr].shape)

        self.set_axes('1d')
        self.axes.plot(self.data[attr].phi, self.data[attr][index], 'C0-')
        self.axes.plot(self.data[attr].phi, data[index], 'C1--')
        self.axes.set_xlabel(r'$\phi$ deg')
        self.axes.set_ylabel(attr)



if __name__ == '__main__':

    database = Database('86T4WW')
    database.build_surface()
    database.compose()

    #database.plot_normal()

    database.plot_trace()
