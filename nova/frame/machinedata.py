from os import path
from copy import deepcopy

import pandas as pd
import numpy as np
import shapely.geometry
import shapely.algorithms

from nova.definitions import root_dir
from nova.plot import plt
from nova.utilities.IO import pythonIO
from nova.frame.coilset import CoilSet


class MachineData(CoilSet, pythonIO):
    """
    load ITER data and geometry.

    Data_for_study_of_ITER_plasma_magnetic_c_33NHXN_v3_15.xlsx
    Models_for_calculation_of_axisymmetric_c_XBQF5H_v2_2.xlsx
    """

    def __init__(self, read_txt=False, **kwargs):
        self.read_txt = read_txt
        self.directory = path.join(root_dir, 'input/ITER')
        super().__init__(**kwargs)

    @staticmethod
    def append(data, x, z, rho, dt):
        """Append attributes to data."""
        for key, value in zip(['x', 'z', 'rho', 'dt'], [x, z, rho, dt]):
            data[key].append(value)

    @staticmethod
    def orient(frame):
        """
        Orient dataframe ccw.

        Parameters
        ----------
        frame : DataFrame
            pandas dataframe with x and z coordinates.

        Returns
        -------
        frame : DataFrame
            DataFrame with ccw orientation.

        """
        if frame.shape[0] > 2 and 'x' in frame and 'z' in frame:
            ring = shapely.geometry.LinearRing(
                frame.loc[:, ['x', 'z']].values)
            if not ring.is_ccw:
                frame = frame.iloc[::-1, :]
        return frame

    def read_sheet(self, sheetname, skiprows, usecols, columns={}, nrows=None):
        """Read excel worksheet."""
        sheet = pd.read_excel(self.f, sheetname, skiprows=skiprows,
                              usecols=usecols, nrows=nrows).dropna()
        columns = {**{'R, m': 'x', 'Z, m': 'z'}, **columns}
        if np.array([True for c in sheet.columns
                     if '.1' in c or '.2' in c or 'eff' in c]).any():
            strip = {c: c.replace('.1', '').replace('.2', '').
                     replace('eff', '') for c in sheet.columns}
            sheet.rename(columns=strip, inplace=True)
        if columns:
            sheet.rename(columns=columns, inplace=True)
        if 'h, mm' in sheet.columns:
            sheet.loc[:, 'h, mm'] *= 1e3
            sheet.rename(columns={'h, mm': 'h, m'}, inplace=True)
        sheet = self.orient(sheet)
        return sheet

    def read_model(self, name, sheetname, skiprows, usecols, nrows=None,
                   dt=0.06, ring=False, rho=None):
        """Read geometric model."""
        columns = {'R1(m)': 'x1', 'R2(m)': 'x2', 'Z1(m)': 'z1',
                   'Z2(m)': 'z2', 'Ω(Ohm)': 'R'}
        model = self.read_sheet(sheetname, skiprows, usecols, nrows=nrows,
                                columns=columns)
        if ring:  # triangular support and divertor rail
            ring = model.copy()
            model = pd.DataFrame(index=range(1), columns=columns.values())
            model.loc[:, 'x1':'x2'] = ring.x.values
            model.loc[:, 'z1':'z2'] = ring.z.values
            model.R = rho
        data = {}
        _data = {var: [] for var in ['x', 'z', 'rho', 'dt']}
        _xz, index = np.array([0, 0]), 0
        for i in range(model.shape[0]):
            segment = model.iloc[i]
            x = segment.loc[['x1', 'x2']].to_numpy()
            z = segment.loc[['z1', 'z2']].to_numpy()
            x_mean = x.mean()  # mean radius
            # segment length
            dL = np.linalg.norm(np.diff(np.array([x, z]), axis=1))
            # resistivity / thickness
            rho = dL * segment.loc['R'] / (2 * np.pi * x_mean)
            xz = np.array([x[0], z[0]])
            if i == 0 or not np.equal(_xz, xz).all():
                segment_name = f'S{index}{name}'
                if name == 'cryo' and index in [35, 36]:
                    dt = 0.01
                else:
                    dt = 0.06
                data[segment_name] = deepcopy(_data)
                self.append(data[segment_name], x[0], z[0], dt * rho, dt)
                index += 1
            _xz = np.array([x[1], z[1]])
            self.append(data[segment_name], x[1], z[1], dt * rho, dt)

        if index == 1:  # drop sector index
            data = {name: data[segment_name]}
        if name == 'cryo':
            for index, level in zip([35, 36], ['U', 'L']):
                data[f'{level}CTS'] = data.pop(f'S{index}{name}')
        for frame in data:
            data[frame] = pd.DataFrame(data[frame])
            data[frame] = self.orient(data[frame])
        self.models[name] = data

    def load_models(self, **kwargs):
        """Load models from .pk file."""
        read_txt = kwargs.get('read_txt', self.read_txt)
        filepath = path.join(self.directory, 'ITER_machine_models')
        if read_txt or not path.isfile(filepath + '.pk'):
            self.read_models()
            self.save_pickle(filepath, ['models'])
        else:
            self.load_pickle(filepath)

    def read_models(self):
        """Read model set."""
        self.models = {}
        self.filename = \
            'Models_for_calculation_of_axisymmetric_c_XBQF5H_v2_2.xlsx'
        with pd.ExcelFile(path.join(self.directory, self.filename),
                          engine='openpyxl') as self.f:
            self.read_model('vvin', 'Conducting structures', 10,
                            np.arange(1, 6), nrows=100)
            self.read_model('vvout', 'Conducting structures', 112,
                            np.arange(1, 6), nrows=100)
            self.read_model('cryo', 'Conducting structures', 215,
                            np.arange(1, 6), nrows=249)
            self.read_model('trs', 'Conducting structures', 55,
                            np.arange(10, 12), nrows=2, ring=True, rho=0.8)
            self.read_model('dir', 'Conducting structures', 67,
                            np.arange(10, 12), nrows=2, ring=True, rho=0.9)

    def plot_models(self):
        """Plot geometrical models."""
        plt.set_aspect(1.1)
        for i, part in enumerate(self.models):
            for segment in self.models[part]:
                data = self.models[part][segment]
                plt.plot(data['x'], data['z'], f'C{i}')
        plt.axis('equal')
        plt.axis('off')

    def load_data(self, **kwargs):
        """Load machine data."""
        read_txt = kwargs.get('read_txt', self.read_txt)
        filepath = path.join(self.directory, 'ITER_machine_data')
        if read_txt or not path.isfile(filepath + '.pk'):
            self.read_data()
            self.save_pickle(filepath, ['data'])
        else:
            self.load_pickle(filepath)

    def read_data(self):
        """Read geometric data."""
        self.data = {}
        self.filename = \
            'Data_for_study_of_ITER_plasma_magnetic_c_33NHXN_v3_15.xlsx'
        with pd.ExcelFile(path.join(self.directory, self.filename),
                          engine='openpyxl') as self.f:

            self.data['separatrix'] = \
                self.read_sheet('Target separatrix', 7, [2, 3])

            self.data['firstwall'] = \
                self.read_sheet('FW & Divertor', 7, [1, 2])

            self.data['divertor'] = \
                self.read_sheet('FW & Divertor', 7, [4, 5])

            self.data['SSring'] = \
                self.read_sheet('VV & TS & DIR', 171, [1, 2], nrows=2)

            self.data['DIR'] = \
                self.read_sheet('VV & TS & DIR', 183, [1, 2], nrows=2)

            self.data['VVin'] = \
                self.read_sheet('VV & TS & DIR', 10, [1, 2], nrows=135)

            self.data['VVout'] = \
                self.read_sheet('VV & TS & DIR', 10, [4, 5])

            self.data['cryostat'] = \
                self.read_sheet('Cryostat & CST', 8, np.arange(1, 5))[:14]

            self.data['cryostatCTS'] = \
                self.read_sheet('Cryostat & CST', 8, np.arange(1, 5))[13:]

            for rib in range(4):
                self.data[f'cryostatR{rib+1}'] = self.read_sheet(
                        'Cryostat & CST', 8+rib*5, np.arange(7, 11), nrows=2)

            self.data['upperCTS'] = self.read_sheet(
                    'Cryostat & CST', 8, np.arange(12, 17))

    def plot_data(self, keys=None, ax=None, legend=False, **kwargs):
        """Plot geometric data."""
        if ax is None:
            ax = plt.gca()
        if keys is not None:
            if not pd.api.types.is_list_like(keys):
                keys = [keys]
        else:
            keys = self.data.keys()
        for key in keys:
            try:
                ax.plot(self.data[key]['x'], self.data[key]['z'], label=key,
                        **kwargs)
            except KeyError:
                raise KeyError(key, self.data[key].columns)
        ax.axis('equal')
        ax.axis('off')
        if legend:
            ax.legend()

    def select_coilset(self, part_list=None):
        if not hasattr(self, 'models'):
            self.load_models()
        if part_list is None:
            part_list = self.models.keys()
        if not pd.api.types.is_list_like(part_list):
            part_list = part_list.replace('_', ' ')
            part_list = part_list.split()
        part_list = [pl for pl in part_list if pl in self.models]
        part_list = list(np.unique(np.sort(part_list)))
        return part_list

    def load_coilset(self, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        filename = 'ITER_coilset_'
        part_list = self.select_coilset(kwargs.get('part_list', None))
        filename += '_'.join(part_list)
        filepath = path.join(self.directory, filename)
        if read_txt or not path.isfile(filepath + '.pk'):
            self.build_coilset(part_list)
            self.save_coilset(filename, directory=self.directory)
        else:
            CoilSet.load_coilset(self, filename, directory=self.directory)
        return self.coilset

    def build_coilset(self, part_list):
        for part in part_list:
            for segment in self.models[part]:
                frame = self.models[part][segment]
                print(frame.dt)
                self.shell.insert(frame.x, frame.z, self.dshell,
                                  frame.dt, rho=frame.rho,
                                  part=part, name=segment)


if __name__ == '__main__':

    machine = MachineData(dcoil=0.2, dshell=0, read_txt=False)

    #machine.load_coilset(part_list='vvin vvout')

    #machine.load_models(read_txt=True)
    #machine.plot_models()

    #machine.load_data(read_txt=True)
    #machine.plot_data()

    '''
    machine.load_coilset(part_list='trs dir', read_txt=True)

    machine.Ic = 20e3
    machine.plot(subcoil=True)

    machine.grid.generate_grid()
    machine.grid.plot_flux()

    machine.read_data()
    machine.plot_data()
    '''


    '''
    line = [(xz)
            for xz in machine.data['firstwall'].loc[:, ['x', 'z']].to_numpy()]
    polygon = shapely.geometry.LineString(line).buffer(0.03, cap_style=2,
                                                       join_style=2)

    patch = PolygonPatch(polygon)

    ax = plt.gca()
    ax.add_patch(patch)
    '''
