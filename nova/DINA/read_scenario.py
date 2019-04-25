from nep.DINA.read_dina import read_dina
from amigo.qdafile import QDAfile
import numpy as np
from scipy.interpolate import interp1d
from os import sep
from os.path import isfile
import pandas as pd
from amigo.pyplot import plt
from amigo.geom import rdp_extract


class operate:

    def __init__(self, t, frame):
        self.t = t
        self.frame = frame
        self.feature_extract()

    def feature_extract(self, eps=1e-2, dx=1.25):
        t, Ip = rdp_extract(self.t, self.frame.Ip.values, dx=dx, eps=eps)
        self.waveform = pd.DataFrame({'t': t, 'Ip': Ip})
        self.extract_flattop()


    def extract_flattop(self, threshold=0.8):
        slope = np.diff(self.waveform.Ip) / np.diff(self.waveform.t)
        self.waveform.loc[:, 'slope'] = np.append(slope, None)

        threshold_index = abs(self.waveform.Ip) > threshold


        #slope = self.waveform.slope[threshold_index]
        #flattop_index = np.argmin(abs())

        print(flattop_index)
        print(self.waveform.loc[flattop_index, :])
        # print(min_slope)
        '''
        n = len(t) - 1

        self.scnario =

        def operate(self, plot=False, dt_window=10, nstd=3):
        # extract scenario data
        Ipl, t, dt = self.d2.frame['Ip'].values, self.d2.teq, self.d2.dt
        # identify operating modes
        trim = np.argmax(Ipl[::-1] < -1e-3)
        ind = len(Ipl)-trim
        # low-pass filter current
        Ipl_lp = lowpass(Ipl, dt, dt_window=dt_window)
        dIpldt_lp = np.gradient(Ipl_lp, t)  # slope
        self.hip = histopeaks(t[:ind], dIpldt_lp[:ind], nstd=nstd, nlim=9,
                              nbins=300)  # modes
        opp_index = self.hip.timeseries(Ip=Ipl[:ind], plot=plot)
        self.flattop = {}  # plasma current flattop
        self.flattop['index'] = opp_index[0]
        self.flattop['t'] = t[opp_index[0]]
        self.flattop['dt'] = np.diff(self.flattop['t'])[0]
        psi_flattop = self.d2.frame['PSI(axis)'].values[self.flattop['index']]
        self.flattop['dpsi'] = np.diff(psi_flattop)[0]
        '''



class interpolate:

    def __init__(self, **kwargs):
        self._to = None
        self._extrapolate = kwargs.get('extrapolate', False)

    def set_index(self, index=None):
        if index is None:
            self.index = self.data.columns
        else:
            # remove columns absent from data
            self.index = [var for var in index if var in self.data]
        self._vector = pd.Series(index=self.index)  # initalise data slice

    @property
    def to(self):
        '''
        interpolation time instance (float)
        '''
        self.check_folder_set()
        return self._to

    @to.setter
    def to(self, to):
        '''
        update vector to input time, to
        '''
        self._to = to
        self.check_extrapolate()
        self._vector.loc[:] = self.interpolator(self.to)  # update vector

    @property
    def vector(self):
        self.check_folder_set()
        return self._vector

    def check_folder_set(self):
        if self.folder is None:
            raise NameError('scenario folder unset: '
                            'use self.load_file(folder) to load')
        if self._to is None:
            self.to = self.t[0]  # reset to default

    @property
    def extrapolate(self):
        return self._extrapolate

    @extrapolate.setter
    def extrapolate(self, extrapolate):
        if not isinstance(extrapolate, bool):
            raise ValueError(f'type(extrapolate) bool != type({extrapolate}) '
                             f' {type(extrapolate)})')
        if extrapolate != self._extrapolate:
            self.update(self.to)
        self._extrapolate = extrapolate

    def check_extrapolate(self):
        if not self._extrapolate and (self._to < self.t[0] or
                                      self._to > self.t[-1]):
            raise IndexError(f'requested time: {self._to}s '
                             'outside data range '
                             f'{self.t[0]} to {self.t[-1]}')

    def space(self):
        t, unique_index = np.unique(self.data['t'], return_index=True)
        dt = np.mean(np.diff(t))
        tmax = np.nanmax(t)
        tmin = np.nanmin(t)
        nt = int(tmax/dt)
        self.t = np.linspace(tmin, tmax, nt)
        self.dt = (tmax - tmin) / (nt - 1)
        return unique_index


class scenario_data(read_dina, interpolate):

    '''
    Attributes:
        data (pd.DataFrame): DINA raw data (load using read_scenario)
        t (np.array): time vector with equidistant spacing
        interpolator (): q 1d interpolator (vectorised) for data2 input
        to (float): instance time (default, start of file)
        instance (pd.Series): interpolated data at time instance to
        frame (pd.DataFrame): interpolated data across time vector t
        index (list): column names extracted from data
        coil_index (list): coil names
        Ic (pd.Series): coil current vector
    '''

    def __init__(self, database_folder='operations', folder=None,
                 read_txt=False, extrapolate=False):
        '''
        Attributes:
            database_folder (str): database folder
            folder (str): scenario folder
            read_txt (bool): read / reread source text files
        '''
        super().__init__(database_folder=database_folder,
                         read_txt=read_txt,
                         extrapolate=extrapolate)
        self.load_file(folder)

    def load_file(self, folder, verbose=True, **kwargs):
        self.folder = folder
        if self.folder is not None:
            read_txt = kwargs.get('read_txt', self.read_txt)
            filename = self.locate_folder('data2', folder)[0]
            filename += '_scenario_data'
            if read_txt or not isfile(filename + '.pk'):
                self.read_file(folder)
                self.save_pickle(
                        filename, ['t', 'interpolator', 'frame', '_vector',
                                   '_Ic', 'index', 'coil_index'])
            else:
                self.load_pickle(filename)

    def read_file(self, folder):
        scn = read_scenario(self.database_folder, read_txt=False)
        scn.load_file(folder)
        self.interpolate(scn.data2)

    def correct_time(self):
        if 'time' in self.data:  # rename time field
            self.data['t'] = self.data['time']
            self.data.pop('time')

    def correct_coordinates(self):
        '''
        correct coodrinate system for DINA data created before self.date_switch
        '''
        if self.date > self.date_switch:
            coordinate_switch = 1
        else:  # old file - correct coordinates
            coordinate_switch = -1
        for var in self.data:
            if ('I' in var and len(var) <= 5) or ('V' in var):
                self.data[var] *= coordinate_switch

    def correct_units(self):
        for var in self.data:
            if var == 'Ip':  # plasma
                self.data.loc[:, var] *= 1e6  # MA to A
            elif var[0] == 'I' and len(var) <= 5:
                self.data.loc[:, var] *= 1e3  # kAturn to Aturn

    def set_index(self, *args):
        # current columns
        index = [var for var in self.data if ('I' in var and len(var) <= 5)]
        # additional baseline data
        index += ['Rcur', 'Zcur', 'ap', 'kp', 'Rp', 'Zp', 'a',
                  'Ksep', 'BETAp', 'li(3)', 't', 'PSI(axis)', 'Emag', 'Lp',
                  'q(95)', 'q(axis)', 'Vloop', 'D(PSI)res', 'Cejima',
                  '<PSIext>', '<PSIcoils>']
        for arg in args:  # append aditional columns
            if arg not in index:
                index.append(arg)
        interpolate.set_index(self, index)
        self.coil_index = [var for var in self.index
                           if ('I' in var and len(var) <= 5 and var != 'Ip')]
        self.coil_index.insert(self.coil_index.index('Ics1'), 'Ics1')
        cs1_loc = self.coil_index.index('Ics1')
        Ic_index = [var[1:].upper() for var in self.coil_index]
        Ic_index[cs1_loc:cs1_loc+2] = ['CS1U', 'CS1L']  # CS central pair
        self._Ic = pd.Series(index=Ic_index)

    def interpolate(self, data, *args):
        '''
        build 1d vectorized interpolator, initalize self.data
        '''
        self.data = data.copy()  # create local copy
        self.correct_time()
        self.correct_coordinates()
        self.correct_units()
        self.set_index(*args)
        unique_index = self.space()  # generate equidistant time vector
        data_block = np.zeros((len(unique_index), len(self.index)))
        for i, index in enumerate(self.index):  # build input data
            data_block[:, i] = self.data[index][unique_index]
        self.interpolator = interp1d(self.data['t'][unique_index],
                                     data_block, axis=0,
                                     fill_value='extrapolate')
        self.data = None  # unlink raw data (reload from source)
        self.frame = pd.DataFrame(  # generate dataframe
                self.interpolator(self.t), index=self.t, columns=self.index)

    @property
    def Ip(self):
        '''
        return plasma current at time to
        '''
        self.check_folder_set()
        return self.vector['Ip']

    @property
    def Ic(self):
        '''
        return pd.Series of coil currents
        '''
        self.check_folder_set()
        self._Ic.loc[:] = self.vector.loc[self.coil_index].values
        return self._Ic


class force_data(read_dina):

    '''
    Attributes:
        data (pd.DataFrame): DINA raw data (load using read_scenario)
        t (np.array): time vector with equidistant spacing
        interpolator (): q 1d interpolator (vectorised) for data2 input
        to (float): instance time
        instance (pd.Series): interpolated data at time instance to
        frame (pd.DataFrame): interpolated data across time vector t
        index (list): column names extracted from data
        coil_index (list): coil names
        Ic (pd.Series): coil current vector
    '''

    def __init__(self, database_folder='operations', folder=None,
                 read_txt=False):
        '''
        Attributes:
            database_folder (str): database folder
            folder (str): scenario folder
            read_txt (bool): read / reread source text files
        '''
        super().__init__(database_folder, read_txt)  # dina read utilities
        if folder is not None:
            self.load_file(folder)

    def load_file(self, folder, verbose=True, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        filename = self.locate_folder('data2', folder) + '_scenario_data'
        if read_txt or not isfile(filename + '.pk'):
            self.read_file(folder)
            self.save_pickle(filename, ['t', 'interpolator', 'frame', 'vector',
                                        'Ic', 'index', 'coil_index'])
        else:
            self.load_pickle(filename)

    def read_file(self, folder, file_type='txt'):
        scn = read_scenario(self.database_folder, read_txt=self.read_txt)
        scn.load_file(folder)
        self.interpolate(scn.data2)

    def set_index(self):
        a=1

        '''
        coils = self.coilset.coil.index
        CSname = self.coilset.coil.index[self.coilset.coil.part == 'CS']
        self.post = {'DINA': {}, 'Nova': {}}
        # DINA
        self.post['DINA']['t'] = pd.Series(self.data3['time'])
        nC, nt = self.coilset.coil.nC, len(self.post['DINA']['t'])
        Fx, Fz, B = np.zeros((nt, nC)), np.zeros((nt, nC)), np.zeros((nt, nC))
        for i, name in enumerate(self.coilset.coil.index):
            B[:, i] = self.data3[f'B_{name.lower()}']
            Fx[:, i] = self.data3[f'Fr_{name.lower()}']
            Fz[:, i] = self.data3[f'Fz_{name.lower()}']
        self.post['DINA']['B'] = pd.DataFrame(B, columns=coils)
        self.post['DINA']['Fx'] = pd.DataFrame(Fx, columns=coils)
        self.post['DINA']['Fz'] = pd.DataFrame(Fz, columns=coils)
        self.post['DINA']['Fsep'] = self.calculate_Fsep(
                self.post['DINA']['Fz'].loc[:, CSname])
        '''


class read_scenario(read_dina):

    def __init__(self, database_folder='operations', folder=None,
                 read_txt=False, file_type='txt'):
        super().__init__(database_folder, read_txt)  # dina read utilities
        if folder is not None:
            self.load_file(folder, file_type=file_type)

    def load_file(self, folder, verbose=True, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        file_type = kwargs.get('file_type', 'txt')
        filename, file_type = self.locate_folder('data2', folder, file_type)
        if read_txt or not isfile(filename + '.pk'):
            self.read_file(folder, file_type=file_type, verbose=verbose)
            self.save_pickle(filename, ['data2', 'data3'])
        else:
            self.load_pickle(filename)

    def read_file(self, folder, file_type='txt', verbose=True):
        if verbose:
            print(f'reading {self.filename}.{file_type}')
        if file_type == 'txt':
            self.read_txt_file(folder)
        elif file_type == 'qda':
            self.read_qda_file(folder)

    def read_txt_file(self, folder, dropnan=True, force=False):
        filename = self.locate_file('data2.txt', folder=folder)
        self.data2 = self.read_csv(filename, dropnan=True, split=',',
                                   dataframe=True)
        filename = self.locate_file('data3.txt', folder=folder)
        self.data3 = self.read_csv(filename, dropnan=True, split=',',
                                   dataframe=True)

    def read_qda_file(self, folder):
        filename = self.locate_file('data2.qda', folder=folder)
        self.name = filename.split(sep)[-3]
        self.qdafile = QDAfile(filename)
        self.data2 = pd.DataFrame()
        columns = {}
        for i, (var, nrow) in enumerate(zip(self.qdafile.headers,
                                            self.qdafile.rows)):
            var = var.decode()
            if nrow > 0:
                columns[var] = var.split(',')[0]
                self.data2[columns[var]] = np.array(self.qdafile.data[i, :])
        filename = self.locate_file('data3.qda', folder=folder)
        self.name = filename.split(sep)[-3]
        self.qdafile = QDAfile(filename)
        self.data3 = pd.DataFrame()
        columns = {}
        for i, (var, nrow) in enumerate(zip(self.qdafile.headers,
                                            self.qdafile.rows)):
            var = var.decode()
            if nrow > 0:
                columns[var] = var.split(',')[0]
                self.data3[columns[var]] = np.array(self.qdafile.data[i, :])





if __name__ is '__main__':

    # scn = read_scenario(read_txt=False)
    # scn.load_file(6, read_txt=False)  # read / load single file
    # scn.load_folder()

    d2 = scenario_data(read_txt=False)
    # d2.load_folder()
    d2.load_file(6)

    opp = operate(d2.t, d2.frame)


    #rdp_extract(d2.t, d2.frame.Ip, 15e6/70, dx=0.25, eps=0.5, plot=True)

    #plt.plot(scn.d2.teq, scn.)
    # scn.load_folder()  # load / reload full project dir

