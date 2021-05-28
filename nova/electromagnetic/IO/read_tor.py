"""Read toroidal currents from DINA disruption simulations."""
from dataclasses import dataclass
from os import path

import numpy as np

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import pandas as pd
import scipy
import shapely
import xarray

from nova.electromagnetic.coilset import CoilSet
from nova.electromagnetic.IO.read_waveform import read_dina
from nova.electromagnetic.machinedata import MachineData
from nova.electromagnetic.polyplot import Axes

from nova.utilities.pyplot import plt
from nova.utilities.IO import readtxt


@dataclass
class VDE(Axes, CoilSet):  # read_dina,
    """
    Read tor_cur_data*.dat file from DINA simulation.

    Temperal listing of toroidal currents.

    """

    def load_file(self, folder, read_txt=False):
        """Load disruption data."""
        dina = read_dina('disruptions')
        filepath = dina.locate_file('tor_cur', folder=folder)
        filepath = '.'.join(filepath.split('.')[:-1])
        self.name = filepath.split(path.sep)[-2]
        if read_txt or not path.isfile(filepath + '.h5'):
            self.frame.drop()
            self.subframe.drop()
            self.read_file(filepath)  # read txt file
            self.grid.solve(2e4, 0.05)
            self.store(f'{filepath}.h5')
        else:
            self.load(f'{filepath}.h5')

    def load(self, file):
        """Load data from hdf5."""
        super().load(file)  # load coilset
        with xarray.open_dataset(file, group='vde_data') as data:
            data.load()
            self.data = data  # load vde data

    def store(self, file):
        """store data to hdf5."""
        super().store(file)  # store coilset
        self.data.to_netcdf(file, mode='a', group='vde_data')

    def read_file(self, filepath):  # called by load_file
        """Read txt data."""
        with readtxt(filepath + '.dat') as self.rt:
            self.read_coils()
            self.read_frames()
        self.read_data()

    def read_coils(self):
        """Load poloidal field filaments."""
        self.rt.skiplines(5)  # skip header
        self.insert_coils()  # insert poloidal field coils
        self.insert_shells()
        self.insert_plasma()

    def insert_coils(self):
        """Update coilset with poloidal field coils."""
        self.rt.checkline('1.')
        self.rt.skiplines(1)
        index = self.rt.readline(True, string=True)
        index.extend(self.rt.readline(True, string=True))
        geom = np.zeros((4, len(index)))
        for i, var in enumerate(['x', 'z', 'dx', 'dz']):
            self.rt.skiplines(1)
            geom[i, :] = self.rt.readblock()
        geom *= 1e-2  # cm to meters
        part = ['cs' if 'CS' in name else 'pf' for name in index]
        turns = dict(CS3U=554, CS2U=554, CS1U=554,
                     CS1L=554, CS2L=554.0, CS3L=554.0,
                     PF1=248.64, PF2=115.2, PF3=185.92,
                     PF4=169.92, PF5=216.8, PF6=459.36)
        nturn = [turns[name] for name in index]
        self.coil.insert(*geom, name=index, part=part, turn='hex', nturn=nturn)

    def insert_shells(self, dt=60e-3):
        """Insert vessel and blanket shells."""
        self.rt.checkline('2.')
        self.rt.skiplines(1)
        self.nf = self.rt.readnumber()
        self.rt.skiplines(3)
        self.filaments = \
            np.zeros(self.nf, dtype=[('x1', '2float'), ('z1', '2float'),
                                     ('x2', '2float'), ('z2', '2float'),
                                     ('turn', '2float'), ('n_turn', 'int')])
        # read from file
        for i in range(self.nf):
            n_turn = self.rt.readnumber()
            self.filaments[i]['n_turn'] = n_turn
            for j in range(n_turn):
                x1, z1, x2, z2, turn = [float(d) for d in
                                        self.rt.readline(True)]
                for var, value in zip(['x1', 'z1', 'x2', 'z2', 'turn'],
                                      [x1, z1, x2, z2, turn]):
                    self.filaments[i][var][j] = value
        # add shells
        for i, filament in enumerate(self.filaments):
            for j in range(filament['n_turn']):
                x1, x2 = 1e-2*filament['x1'][j], 1e-2*filament['x2'][j]
                z1, z2 = 1e-2*filament['z1'][j], 1e-2*filament['z2'][j]
                part = 'vv' if filament['n_turn'] == 1 else 'bb'
                self.shell.insert([x1, x2], [z1, z2], 0, dt,
                                  label=part, part=part)
                if j == 1:  # link blanket pairs
                    self.link(self.frame.index[-2:], factor=-1)

    def insert_plasma(self):
        """Insert plasma filaments."""
        machine = MachineData()
        machine.load_data()
        fw = pd.concat((machine.data['firstwall'],
                        machine.data['divertor']))
        self.plasma.insert(fw.to_numpy(), name='Plasma', part='plasma')

    def read_frames(self):
        """Read transient frame data."""
        self.frames = []
        self.rt.skiplines(6)
        while True:
            try:
                self.frames.append(self.get_current())
            except ValueError:
                break

    def read_data(self):
        """Read frame data into xarray dataset."""
        time = [1e-3*frame[0] for frame in self.frames]
        plasma = self.loc['plasma']
        self.data = xarray.Dataset(
            coords=dict(index=self.subframe.subspace.index.to_list(),
                        plasma=self.subframe.index[plasma].to_list(),
                        time=time))
        self.data['It'] = xarray.DataArray(
                0., dims=['time', 'index'],
                coords=[self.data.time, self.data.index])
        self.data['Ic'] = xarray.DataArray(
                0., dims=['time', 'index'],
                coords=[self.data.time, self.data.index])
        self.data['nturn'] = xarray.DataArray(
                0., dims=['time', 'plasma'],
                coords=[self.data.time, self.data.plasma])
        nturn = self.frame.loc[
            self.frame.subspace.index, 'nturn']
        for i, frame in enumerate(self.frames):
            self.data['It'][i, :12] = -1e3*np.array(frame[3][:12])
            self.data['It'][i, 12:-1] = -1e3*np.array(frame[1])
            self.data['It'][i, -1] = -1e3*np.sum(frame[2][2::3])
            self.data['Ic'][i] = self.data['It'][i] / nturn
            # extract plasma coordinates and filament current
            xp = 1e-2*np.array(frame[2][0::3])
            zp = 1e-2*np.array(frame[2][1::3])
            Ip = -1e3*np.array(frame[2][2::3])  # -kA to A
            if len(xp) == 0:
                self.data['nturn'][i, :] = 0
                continue
            # estimate DINA grid spacing
            delta = xp[1:] - xp[:-1]
            try:
                delta = scipy.stats.mode(delta[delta != 0])[0][0]
            except IndexError:
                delta = self.dcoil
            # calculte convex hull and inflate by delta/2
            sep = shapely.geometry.MultiPoint(np.array([xp, zp]).T).convex_hull
            # update plasma separatrix
            self.plasma.update(sep.buffer(delta/2))
            # calculate plasma filament turns
            turns = scipy.interpolate.griddata(
                (xp, zp), Ip, self.loc['ionize', ['x', 'z']],
                method='nearest')
            # normalize turn number
            self.loc['ionize', 'nturn'] = turns / np.sum(turns)
            self.data['nturn'][i] = self.loc['plasma', 'nturn']

    def get_current(self):
        """Read current vector from txt file."""
        self.rt.skiplines(1)
        t = self.rt.readnumber()
        self.rt.skiplines(1)
        filament = self.rt.readblock()  # vessel / blanket current vector
        self.rt.skiplines(3)
        plasma = self.rt.readblock()  # plasma current
        self.rt.skiplines(1)
        coil = self.rt.readblock()  # poloidal field coil current.
        return (t, filament, plasma, coil)

    def update(self, index):
        """Update coil currents and plasma positon to match index."""
        self.sloc['Ic'] = self.data['Ic'][index].values
        # self.sloc['coil', 'Ic'] = 0
        # self.sloc['plasma', 'Ic'] = 0
        self.loc['plasma', 'nturn'] = self.data['nturn'][index].values
        ionize = self.data['nturn'][index].values != 0
        self.loc['plasma', 'ionize'] = ionize
        self.grid.update_turns()

    def plot(self):
        """plot coilset."""
        super().plot(axes=self.axes)
        self.plasma.plot(axes=self.axes)
        self.grid.plot(axes=self.axes)

    def make_frame(self, position, axes=None):
        """Make frame for make_movie."""
        self.axes = axes  # set axes
        self.axes.clear()
        start_time = self.data.time.values[0]
        end_time = self.data.time.values[-1]
        delta_time = end_time - start_time
        shift = 0.6*delta_time
        start_time += shift
        delta_time -= shift
        time = start_time + delta_time*position / self.duration
        index = scipy.interpolate.interp1d(
            self.data.time, range(self.data.dims['time']))(time)
        index = int(np.round(index))
        self.update(index)
        self.plot()
        return mplfig_to_npimage(plt.gcf())

    def make_movie(self, file, duration, fps=40):
        """Make movie."""
        self.duration = duration
        animation = VideoClip(self.make_frame, duration=self.duration)
        animation.write_videofile(f'{file}.mp4', fps=fps, audio=False)

if __name__ == '__main__':

    vde = VDE(dcoil=0.5, dplasma=0.35, dshell=2.5)

    vde.load_file(-3, read_txt=False)

    plt.set_aspect(1.1)
    vde.duration = 2
    #vde.make_frame(0.5)
    vde.make_movie('tmp', 10, 25)
