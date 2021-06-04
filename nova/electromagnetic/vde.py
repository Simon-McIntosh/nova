"""Read toroidal currents from DINA disruption simulations."""
from dataclasses import dataclass, field
from typing import Union
import os

import numpy as np

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import pandas as pd
import scipy
import shapely
import xarray

from nova.electromagnetic.coilset import CoilSet
from nova.definitions import root_dir
from nova.electromagnetic.biotpoint import PointInverse
from nova.electromagnetic.IO.read_waveform import read_dina
from nova.electromagnetic.machinedata import MachineData
from nova.electromagnetic.polyplot import Axes

from nova.utilities.pyplot import plt
from nova.utilities.IO import readtxt
from nova.utilities.time import clock


@dataclass
class VDE(Axes, CoilSet):  # read_dina,
    """
    Read tor_cur_data*.dat file from DINA simulation.

    Temperal listing of toroidal currents.

    """

    folder: Union[str, int] = None
    dcoil: float = 0.25
    dplasma: float = 0.25
    dshell: float = 0.5
    read_txt: bool = False
    file: str = field(init=False, repr=False)
    dina_file: str = field(init=False, repr=False)

    duration: float = 5
    fps: int = 15

    def __post_init__(self):
        """Build file paths."""
        self.directory = os.path.join(root_dir, 'data/DINA/hdf5')
        if self.folder is not None:
            self.set_path(self.folder)
        super().__post_init__()

    def set_path(self, folder):
        """Set full filepath."""
        try:
            self.dina_file = \
                read_dina('disruptions').locate_file('tor_cur', folder=folder)
            self.folder = self.dina_file.split(os.path.sep)[-2]
        except IndexError:
            self.dina_file = None
            self.folder = folder
        self.file = os.path.join(self.directory, f'{self.folder}.h5')

    def set_file(self, file):
        """Set file paths."""
        if isinstance(file, int):
            return self.set_path(file)
        if file[-3:] == '.h5':
            self.file = file
            folder = file.split(os.path.sep)[-1].removesuffix('.h5')
            return self.set_path(folder)
        return self.set_path(file)

    def load_folders(self):
        """Load all disruption data."""
        folders = read_dina('disruptions').folders
        tick = clock(len(folders), header='Reading DINA disruption data.')
        for folder in folders:
            self.load(folder)
            tick.tock()

    def load(self, file):
        """Load disruption data."""
        self.set_file(file)
        if self.read_txt or not os.path.isfile(self.file):
            self.frame.drop()
            self.subframe.drop()
            self.read_file(self.dina_file)  # read txt file
            self.grid.solve(2e4, 0.05)
            self.store(self.file)
        else:
            super().load(self.file)  # load coilset
            with xarray.open_dataset(self.file, group='vde_data') as data:
                data.load()
                self.data = data  # load vde data

    def store(self, file):
        """Store data to hdf5."""
        if file[-3:] != '.h5':
            file = os.path.join(self.directory, f'{file}_{self.folder}.h5')
        super().store(file)  # store coilset
        self.data.to_netcdf(file, mode='a', group='vde_data')

    def read_file(self, filepath):  # called by load_file
        """Read txt data."""
        with readtxt(filepath) as self.rt:
            self.read_coils()
            transient = self.read_transient()
        self.read_data(transient)

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

    def read_transient(self):
        """Return transient current data."""
        transient = []
        self.rt.skiplines(6)
        while True:
            try:
                transient.append(self.get_current())
            except ValueError:
                break
        return transient

    def read_data(self, transient):
        """Read frame data into xarray dataset."""
        transient = transient[1:-1]
        time = [1e-3*frame[0] for frame in transient]
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
        for i, frame in enumerate(transient):
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

    def update(self, index, solve_grid=True):
        """Update coil currents and plasma positon to match index."""
        self.sloc['Ic'] = self.data['Ic'][index].values
        # self.sloc['coil', 'Ic'] = 0
        # self.sloc['plasma', 'Ic'] = 0
        self.loc['plasma', 'nturn'] = self.data['nturn'][index].values
        ionize = self.data['nturn'][index].values != 0
        self.loc['plasma', 'ionize'] = ionize
        if solve_grid:
            self.grid.update_turns()
        self.point.update_turns()
        self.probe.update_turns()

    def plot(self, axes=None):
        """Plot coilset."""
        super().plot(axes=axes)
        if self.loc['plasma', 'nturn'].sum() != 0:
            self.plasma.plot(axes=self.axes)
        self.grid.plot(axes=self.axes, levels=30)

    def get_index(self, position):
        """Return frame index."""
        start_time = self.data.time.values[0]
        end_time = self.data.time.values[-1]
        delta_time = end_time - start_time
        time = start_time + delta_time*position / self.duration
        index = scipy.interpolate.interp1d(
            self.data.time, range(self.data.dims['time']))(time)
        return int(np.round(index))

    def make_frame(self, position, axes=None):
        """Make frame for make_movie."""
        self.axes = axes  # set axes
        self.axes.clear()
        index = self.get_index(position)
        self.update(index)
        self.plot()
        return mplfig_to_npimage(plt.gcf())

    def make_movie(self, prefix=None, make_frame=None):
        """Make movie."""
        if make_frame is None:
            make_frame = self.make_frame
        filename = f'{self.folder}.mp4'
        if prefix:
            filename = f'{prefix}_{filename}'
        file = os.path.join(self.directory, '../animations', filename)
        plt.figure(figsize=(5, 7), facecolor='white')
        make_frame(0)
        animation = VideoClip(make_frame, duration=self.duration)
        animation.write_videofile(file, fps=self.fps)

    def extract_waveform(self):
        """Extract probe waveforms."""
        self.probe.data['time'] = self.data.time
        for attr in ['psi', 'br', 'bz', 'dbdt_r', 'dbdt_z']:
            self.probe.data[attr] = xarray.DataArray(
                0., dims=['time', 'target'],
                coords=[self.probe.data.time, self.probe.data.target])
        tick = clock(self.data.dims['time'], header='Extracting waveform')
        for i in range(self.data.dims['time']):
            self.update(i, solve_grid=False)
            self.probe.data['psi'][i, :] = \
                self.probe.data.Psi.values @ self.subframe.subspace['Ic']
            self.probe.data['br'][i, :] = \
                self.probe.data.Br.values @ self.subframe.subspace['Ic']
            self.probe.data['bz'][i, :] = \
                self.probe.data.Bz.values @ self.subframe.subspace['Ic']
            tick.tock()
        self.probe.data['dbdt_r'][:] = np.gradient(
            self.probe.data.br, self.probe.data.time, axis=0)
        self.probe.data['dbdt_z'][:] = np.gradient(
            self.probe.data.bz, self.probe.data.time, axis=0)
        self.probe.store(self.file)  # save data to file

    def plot_waveform(self, index=slice(1, -10), axes=None, filt=True,
                      **kwargs):
        """Plot probe waveforms."""
        if axes is None:
            axes = plt.gca()
        if filt:
            time, bz = self.filt(self.probe.data.time[index],
                                 self.probe.data.bz[index, 0], 25)

            dbdt_z = np.gradient(bz, time, axis=0)
            plt.plot(time[:-200], dbdt_z[:-200], **kwargs)
            return dbdt_z
        plt.plot(self.probe.data.time[index],
                 self.probe.data.dbdt_z[index, 0], **kwargs)
        return self.probe.data.dbdt_z[index, 0]

    def butter_lowpass(self, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = scipy.signal.butter(order, normal_cutoff,
                                   btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=2):
        b, a = self.butter_lowpass(cutoff, fs, order)
        y = scipy.signal.filtfilt(b, a, data)
        return y

    def filt(self, time, value, cutoff):
        dt_min = np.min(np.diff(time))
        _time = np.linspace(time[0], time[-1],
                            int((time[-1]-time[0]) / dt_min))
        fs = 1 / (_time[1]-_time[0])
        _value = scipy.interpolate.interp1d(time, value)(_time)
        return _time, self.butter_lowpass_filter(_value, cutoff, fs)


# Filter requirements.
order = 6
fs = 30.0       # sample rate, Hz
cutoff = 3.667  # desired cutoff frequency of the filter, Hz


if __name__ == '__main__':

    folder = 'VDE_UP_slow'
    #folder = 'MD_DW_exp16'
    vde = VDE(folder=folder, read_txt=False)
    '''
    points = ((9.8, 3), (5, 6.2),
              (9.3, -3), (4.7, -6.2),
              (2.75, 2.5), (2.75, -1.5)
              )
    vde.point.solve(points)
    vde.probe.solve(((10.5, 0.5),))
    vde.extract_waveform()
    vde.store(vde.file)
    vde.store('duck')
    '''

    duck = VDE(folder=f'duck_{vde.folder}')
    svv_index = duck.sloc[duck.loc['part'] == 'vv', :].index.to_list()
    svv_iloc = svv_index.index('vv114')
    duck.sloc['vv0':svv_index[svv_iloc-1], 'free'] = True
    duck.sloc['fix'] = ~duck.sloc['free']
    duck.loc[duck.loc['part'] == 'plasma', 'nturn'] = 0
    duck.loc[duck.loc['part'] == 'bb', 'nturn'] = 0

    vv_index = duck.loc[duck.loc['part'] == 'vv', :].index.to_list()
    vv_iloc = vv_index.index('vv114')
    duck.loc[vv_index[vv_iloc]:vv_index[-1], 'nturn'] = 0
    duck.data.nturn[:] = 0  # plasma turns
    duck.data.It[:, ~duck.sloc['coil'] & ~duck.sloc['free']] = 0
    duck.data.Ic[:, ~duck.sloc['coil'] & ~duck.sloc['free']] = 0
    #duck.store(duck.file)

    inverse = PointInverse(duck.frame, duck.subframe, duck.point.data)

    # fit duck
    def fit():
        tick = clock(vde.data.dims['time'], header='Extracting duck')
        for index in range(vde.data.dims['time']):
            vde.update(index, solve_grid=False)
            duck.update(index, solve_grid=False)
            duck.sloc[~duck.sloc['coil'] & ~duck.sloc['free'], 'Ic'] = 0
            # solve
            Psi = vde.point.data.Psi.values @ vde.sloc['Ic']
            inverse.solve_lstsq(Psi)
            # save solution to file
            duck.data.Ic[index, duck.sloc['free']] = duck.sloc['free', 'Ic']
            tick.tock()
        duck.extract_waveform()
        duck.store(duck.file)

    def make_frame(position, axes=None):
        vde.axes = axes  # set axes
        vde.axes.clear()
        index = vde.get_index(position)
        vde.update(index)
        vde.subframe.polyplot()
        vde.plasma.plot(axes=vde.axes)
        vde.grid.plot(axes=vde.axes, levels=20, colors='darkgray',
                      linewidths=2)
        # plot duck
        duck.update(index)
        duck.grid.plot(axes=vde.axes, colors='C2', levels=vde.grid.levels,
                       linestyles='dashed', linewidths=2)
        duck.point.plot(marker='*', color='C1', ms=10)
        return mplfig_to_npimage(plt.gcf())

    def animate():
        plt.set_aspect(0.9)
        vde.make_movie()
        vde.make_movie('duck', make_frame)

    def plot_single(patchwork=0.25):
        plt.figure()
        vde.subframe.polyplot.patchwork = patchwork
        vde.subframe.polyplot()
        vde.plasma.plot()
        vde.grid.plot()
        vde.subframe.polyplot.patchwork = 0

    def plot_single_duck(patchwork=0.25):
        plt.set_aspect(0.9)
        plt.figure()
        duck.subframe.polyplot.patchwork = patchwork
        duck.subframe.polyplot()
        duck.subframe.polyplot.patchwork = 0
        duck.point.plot(marker='*', color='C1', ms=25)
        duck.probe.plot(marker='o', color='C2', ms=20)

    def plot_frame(time):
        make_frame(time)

    def plot_waveform():
        plt.figure()
        vde_dbdt = vde.plot_waveform(label='DINA')
        duck_dbdt = duck.plot_waveform(label='Duck')
        plt.legend()
        plt.xlabel(r'$t$ s')
        plt.ylabel(r'$\dot{B_z}$ Ts$^{-1}$')
        plt.despine()
        rel_err = np.max(abs(vde_dbdt-duck_dbdt)) / np.max(abs(vde_dbdt))
        plt.title(f'{vde.folder.replace("_", " ")} {1e2*rel_err:1.1f}%')

    def plot_currents():
        plot_frame(0.362)

        plt.figure()
        plt.bar(range(duck.loc['free'].sum()),
                1e-3*vde.loc[duck.loc['free'], 'It'], label='DINA')
        plt.bar(range(duck.loc['free'].sum()), 1e-3*duck.loc['free', 'It'],
                alpha=0.5, label='Duck')
        plt.despine()
        plt.legend()
        plt.xlabel('vessel filament index')
        plt.ylabel(r'$I$ kA')

    #fit()
    #animate()
    plot_waveform()
    #plot_single()
    plot_single_duck()
    #plot_frame(2)
    #duck.plot()

    #vde.make_movie('duck', make_frame)





    '''
    vde.update(1)
    duck.update(1)
    Psi = vde.point.data.Psi.values @ vde.sloc['Ic']
    duck.sloc['plasma', 'Ic'] = 0
    inverse.solve_lstsq(Psi)


    vde.plot()
    duck.grid.plot(axes=vde.axes, colors='C0', levels=vde.grid.levels,
                   linestyles='dashed')

    vde.point.plot(marker='X', color='C3')
    vde.probe.plot(marker='o', color='C7')

    plt.figure()
    plt.bar(range(vde.sloc['passive'].sum()), vde.sloc['passive', 'Ic'])
    plt.bar(range(duck.sloc['passive'].sum()), duck.sloc['passive', 'Ic'],
            alpha=0.5)
    '''




    '''
    vde.update(-1800)
    plt.set_aspect(0.9)
    vde.plot()
    #vde.make_movie()

    vde.point.plot(marker='X', color='C3')
    vde.probe.plot(marker='o', color='C7')


    axes = vde.axes
    levels = vde.grid.levels

    vde = VDE(folder='VDE_DW_slow')
    vde.update(-1800)
    #vde.sloc['plasma', 'Ic'] = 0
    vde.grid.plot(axes=axes, colors='C0', levels=levels,
                  linestyles='dashed')
    '''
