from nep.DINA.read_dina import read_dina
import numpy as np
from scipy.interpolate import interp1d
from os.path import isfile
import pandas as pd
from amigo.pyplot import plt
from amigo.geom import rdp_extract
from astropy import units


class operate:
    '''
    extract features from DINA waveform using rdp algorithum
    '''
    def __init__(self, feature_name='Ip'):
        self._feature = {}
        self.feature_name = feature_name

    @property
    def feature_name(self):
        return self._feature_name

    @feature_name.setter
    def feature_name(self, feature_name):
        self._feature_name = feature_name

    @property
    def feature(self):
        return self._feature[self.feature_name]

    @feature.setter
    def feature(self, feature):
        self._feature[self.feature_name] = feature

    def feature_extract(self, feature_name, eps=1e-3, dt=0.1, dt_window=None,
                        threshold=0.95):
        '''
        Attributes
            feature_name (str): frame column used to perform feature extract
            eps (float): acceptible deviation expressed as fraction of y extent
            dt (float): step width for subsampling
            dt_window (float): filter window
        '''
        self.feature_name = feature_name
        feature = {'eps': eps, 'dt': dt, 'dt_window': dt_window}
        y = self.frame.loc[:, self.feature_name].values[:, 0]
        xrdp = rdp_extract(self.t, y, dx=dt, eps=eps,
                           dx_window=dt_window)[0]
        # interpolate data frame to rdp nodes
        with np.errstate(invalid='ignore'):
            data = self.interpolator(xrdp)
        frame = pd.DataFrame(data, columns=self.frame.columns)
        # initalise node and segment dataframes
        feature['node'] = pd.DataFrame(columns=self.frame.columns)
        feature['segment'] = pd.DataFrame(columns=self.frame.columns)
        for col in self.frame.columns:
            feature['node'].loc[:, col] = frame.loc[:, col].values
            feature['segment'].loc[:, col] = \
                np.diff(feature['node'].loc[:, col])
        feature['node'].reset_index(inplace=True)
        self.feature_label(feature, threshold=threshold)
        self.feature = feature  # store feature

    @property
    def feature_tag(self):
        if self.feature_name == 'Zx':
            if not self.feature_switch:
                tag = 'XPF'
            else:
                tag = 'XPD'
        elif self.feature_name == 'Ti':
            if not self.feature_switch:
                tag = 'SOH'
            else:
                tag = 'EOC'
        else:
            if not self.feature_switch:
                tag = 'SO'
            else:
                tag = 'EO'
            if self.feature_name == 'Pfus':
                tag += 'B'  # burn
            elif self.feature_name == 'Ip':
                tag += 'F'  # flat-top
            else:
                tag += f'_{self.feature_name}'
        self.feature_switch = not self.feature_switch
        return tag

    def feature_label(self, feature, threshold=0.95):
        self.feature_switch = False
        value = abs(feature['node'].loc[:, self.feature_name])
        max_value = np.max(value.values)
        for index in feature['node'].index:
            if value.loc[index].values[0] > threshold*max_value:
                if not self.feature_switch:
                    tag = self.feature_tag
                    feature['node'].rename(index={index: tag},
                                           inplace=True)
            elif value.loc[index].values[0] < threshold*max_value:
                if self.feature_switch:
                    tag = self.feature_tag
                    feature['node'].rename(index={index-1: tag},
                                           inplace=True)
                    break
        if self.feature_name == 'Ip':
            feature['node'].rename(index={0: 'SOD'}, inplace=True)
            if value.loc[value.index[1]].values[0] > 0.01*max_value:
                feature['node'].rename(index={'SOD': 'SOP'}, inplace=True)
            else:
                for index in value.index[1:]:
                    if value.loc[index].values[0] < 0.01*max_value:
                        feature['node'].rename(index={index: 'SOP'},
                                               inplace=True)
                        break
            if 'EOF' in feature['node'].index:
                EOF_index = int(feature['node'].loc['EOF', 'index'].values[0])
                for index in value.index[EOF_index+1:]:
                    if value.loc[index].values[0] < 0.01*max_value:
                        feature['node'].rename(index={index: 'EOP'},
                                               inplace=True)
                        break

    def extract_feature_keypoints(self):
        self.feature_keypoints = pd.DataFrame(
                columns=self.feature['node'].columns)
        self.feature_keypoints.rename(columns={'index': 'frame_index'},
                                      inplace=True)
        for feature_name in self._feature:
            self.feature_name = feature_name
            for index in self.feature['node'].index:
                if isinstance(index, str):
                    vector = pd.Series(self.feature['node'].loc[index, :])
                    vector.drop('index', level=0, inplace=True)
                    vector.loc['frame_index'] = \
                        np.argmin(abs(self.t - vector.t.values[0]))
                    self.feature_keypoints = self.feature_keypoints.append(
                            vector)
        self.feature_keypoints.sort_values(('t', 's'), inplace=True)

    def extract_feature_segments(self):
        self.feature_segments = pd.DataFrame(
                columns=['frame_index', 't', 'dt', 'dpsi'])
        for feature_name, label, keypoints in \
                zip(['Ip', 'Pfus', 'Ti', 'Zx'],
                    ['flattop', 'burn', 'hot', 'Xpoint'],
                    [['SOF', 'EOF'], ['SOB', 'EOB'], ['SOH', 'EOC'],
                     ['XPF', 'XPD']]):
            self.feature_name = feature_name  # load feature
            # ensure that both keypoints are present
            if np.array([key in self.feature['node'].index
                         for key in keypoints]).all():
                t_index = [[] for __ in range(len(keypoints))]
                for i, key in enumerate(keypoints):
                    t_index[i] = self.feature['node'].loc[key, 't'][0]
                index = [np.argmin(abs(self.t-t)) for t in t_index]
                vector = pd.Series(name=label)
                vector['frame_index'] = index
                vector['t'] = t_index
                vector['dt'] = np.diff(t_index)[0]
                psi = self.frame.loc[index, 'PSI(axis)'].values[:, 0]
                vector['dpsi'] = np.diff(psi)[0]
                self.feature_segments = self.feature_segments.append(vector)

    def extract_features(self):
        for variable, threshold in zip(['Ip', 'Pfus', 'Ti', 'Zx'],
                                       [0.95, 0.85, 0.25, 0.95]):
            self.feature_extract(variable, threshold=threshold)
        self.extract_feature_segments()
        self.extract_feature_keypoints()

    def feature_plot(self, feature_name=None):
        if feature_name is None:
            feature_name = ['Ip', 'Pfus', 'Ti', 'Zx']
        ax = plt.subplots(len(feature_name), 1, sharex=True)[1]
        if len(feature_name) == 1:
            ax = [ax]
        for i, name in enumerate(feature_name):
            ax[i].plot(self.t, self.frame.loc[:, name],
                       alpha=0.75, color='gray')
            self.feature_name = name  # load feature
            for index in self.feature['node'].index:
                if isinstance(index, str):
                    if index[0] == 'S' or index == 'XPF':
                        ha = 'left'
                    else:
                        ha = 'right'
                    if index[-1] == 'B':
                        va = 'top'
                    else:
                        va = 'bottom'
                    if index[-1] == 'P':  # flip
                        ha = 'left' if ha == 'right' else 'right'
                    ax[i].plot(self.feature['node'].t.loc[index],
                               self.feature['node'].loc[index, name],
                               'o', color='gray')
                    ax[i].text(self.feature['node'].t.loc[index],
                               self.feature['node'].loc[index, name],
                               index, ha=ha, va=va, color='gray')
            ax[i].set_ylabel(f'${name}$, {self.vector[name].index[0]}')
            ax[i].plot(self.feature['node'].t,
                       self.feature['node'].loc[:, name], '.-', color=f'C{i}')
        plt.despine()
        plt.detick(ax)
        ax[-1].set_xlabel(f'$t$, {self.vector["t"].index[0]}')


class interpolate:

    def __init__(self, extrapolate):
        self._to = None  # time instance (float)
        self._ko = None  # keypoint instance (SOF, EOF, etc.)
        self._extrapolate = extrapolate

    def set_index(self, index=None):
        if index is None:
            self.index = self.data.columns
        else:
            # remove columns absent from data
            self.index = [var for var in index if var in self.data]
        self.unit = dict(self.index)
        # initalise data slice
        index = pd.MultiIndex.from_tuples(self.index, names=['name', 'unit'])
        self._vector = pd.Series(index=index)

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
        Attributes:
            to (float): input time
            to (str): feature_keypoint
        '''
        if isinstance(to, str):  # feature keypoint
            self._ko = to
            if to in self.feature_keypoints.index:
                to = self.feature_keypoints.loc[to, 't'].values[0]
            else:
                self.feature_plot()
                raise IndexError(f'{to} not in {self.feature_keypoints.index}')
        else:
            self._ko = None  # unset keypoint
        self._to = to
        self.check_extrapolate()
        self._vector.loc[:] = self.interpolator(self._to)  # update vector

    @property
    def ko(self):
        return self._ko

    @ko.setter
    def ko(self, ko):
        '''
        Attributes:
            ko (str): feature_keypoint
        '''
        self.to = ko  # set via to instance

    @property
    def vector(self):
        self.check_folder_set()
        return self._vector

    def check_folder_set(self):
        if self.folder is None:
            raise NameError('scenario folder unset: '
                            'use self.load_file(folder) to load')
        if self._to is None:
            self._to = self.t[0]  # reset to default

    @property
    def extrapolate(self):
        return self._extrapolate

    @extrapolate.setter
    def extrapolate(self, extrapolate):
        if not isinstance(extrapolate, bool):
            raise ValueError(f'type(extrapolate) bool != type({extrapolate}) '
                             f' {type(extrapolate)})')
        if extrapolate != self._extrapolate:
            self.to = self._to  # update
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


class scenario_data(read_dina, interpolate, operate):

    '''
    Attributes:
        data (pd.DataFrame): DINA raw data (load using read_scenario)
        t (np.array): time vector with equidistant spacing
        dt (float): time delta
        interpolator (): q 1d interpolator (vectorised) for data2 input
        to (float): instance time (default, start of file)
        vector (pd.Series): interpolated data at time instance to
        frame (pd.DataFrame): interpolated data across time vector t
        index (list): column names extracted from data
        coil_index (list): coil names
        Ic (pd.Series): coil current vector
    '''

    def __init__(self, folder=None, database_folder='operations',
                 read_txt=False, extrapolate=False):
        '''
        Attributes:
            database_folder (str): database folder
            folder (str): scenario folder
            read_txt (bool): read / reread source text files
        '''
        read_dina.__init__(self, database_folder=database_folder,
                           read_txt=read_txt)
        interpolate.__init__(self, extrapolate)
        operate.__init__(self)  # initalise operation instance
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
                        filename, ['t', 'dt', '_to', 'interpolator',
                                   'frame', '_vector',
                                   '_Ic', 'index', 'unit', 'coil_iloc',
                                   '_feature', 'feature_keypoints',
                                   'feature_segments'])
            else:
                self.load_pickle(filename)

    def read_file(self, folder):
        scn = read_scenario(folder, self.database_folder, read_txt=False)
        self.interpolate(scn.data2)
        operate.__init__(self)  # re-initalise operation instance
        self.extract_features()  # extract operational keypoints / keysegments

    def rename_columns(self):
        self.data.rename(columns={'time': 't'}, level=0, inplace=True)
        self.data.rename(columns={'Rsep_or_Rlim': 'Rx_or_Rtouch',
                                  'Zsep_or_Zlim': 'Zx_or_Ztouch'}, level=0,
                         inplace=True)
        self.data.rename(columns={'Rx_or_Rtouch': 'Rx',
                                  'Zx_or_Ztouch': 'Zx'}, level=0, inplace=True)
        self.data.rename(columns={'<Ti>': 'Ti'}, level=0, inplace=True)

    def correct_coordinates(self):
        '''
        correct coodrinate system for DINA data created before self.date_switch
        '''
        if self.date > self.date_switch:
            coordinate_switch = 1
        else:  # old file - correct coordinates
            coordinate_switch = -1
        for var in self.data.columns:
            if ('I' in var[0] and len(var[0]) <= 5) or ('V' in var[0]):
                self.data[var] *= coordinate_switch

    def subindex(self, *args):
        '''
        return reduced subindex
        aditional paramters specified in *args
        pass dummy *args = [-1] to use default subindex
        '''
        # current columns
        columns = self.data.columns
        index = [var for var in columns if ('I' in var and len(var) <= 5)]
        # additional baseline data
        index += ['Rcur', 'Zcur', 'ap', 'kp', 'Rp', 'Zp', 'a',
                  'Ksep', 'BETAp', 'li(3)', 't', 'PSI(axis)', 'Emag', 'Lp',
                  'q(95)', 'q(axis)', 'Vloop', 'D(PSI)res', 'Cejima',
                  '<PSIext>', '<PSIcoils>']
        for arg in args:  # append aditional columns
            if arg not in index:
                index.append(arg)
        return index

    def build_index(self, *args):
        if len(args) == 0:
            index = self.data.columns
        else:
            index = self.subindex(*args)
        self.set_index(index)  # interpolate
        coil_names, self.coil_iloc = np.array(
                [[v[0], i] for i, v in enumerate(self.index)
                 if ('I' in v[0] and len(v[0]) <= 5 and v[0] != 'Ip')]).T

        cs1_loc = coil_names.tolist().index('Ics1')
        coil_names = np.insert(coil_names, cs1_loc, 'Ics1')
        self.coil_iloc = np.insert(self.coil_iloc, cs1_loc,
                                   self.coil_iloc[cs1_loc])
        Ic_index = [name[1:].upper() for name in coil_names]
        Ic_index[cs1_loc:cs1_loc+2] = ['CS1U', 'CS1L']  # CS central pair
        Ic_index = [(name, 'A') for name in Ic_index]
        self._Ic = pd.Series(index=pd.MultiIndex.from_tuples(
                Ic_index, names=['name', 'unit']))

    def interpolate(self, data, *args):
        '''
        build 1d vectorized interpolator, initalize self.data
        '''
        self.data = data.copy()  # create local copy
        self.rename_columns()
        self.correct_coordinates()
        self.build_index(*args)
        unique_index = self.space()  # generate equidistant time vector
        data_block = np.zeros((len(unique_index), len(self.index)))
        for i, index in enumerate(self.index):  # build input data
            data_block[:, i] = self.data[index][unique_index]
        t = self.data['t'].values[unique_index].flatten()
        self.interpolator = interp1d(t, data_block, axis=0,
                                     fill_value='extrapolate')
        self.data = None  # unlink raw data (reload from source)
        # generate dataframe
        self.frame = pd.DataFrame(self.interpolator(self.t))
        self.frame.columns = pd.MultiIndex.from_tuples(
                self.index, names=['name', 'unit'])

    @property
    def Ip(self):
        '''
        return plasma current [A] at time to
        '''
        self.check_folder_set()
        read_unit = self.vector['Ip'].index[0]
        conversion_factor = units.Unit(read_unit).to('A')
        return conversion_factor*self.vector['Ip'][0]

    @property
    def Ic(self):
        '''
        return pd.Series of coil currents [A] at time to
        '''
        self.check_folder_set()
        Ic_vector = self.vector.iloc[self.coil_iloc]
        read_unit = Ic_vector.index.get_level_values('unit')[0]
        write_unit = self._Ic.index.get_level_values('unit')[0]
        if read_unit != write_unit:
            conversion_factor = units.Unit(read_unit).to(write_unit)
        else:
            conversion_factor = 1
        self._Ic.loc[:] = conversion_factor * Ic_vector.values
        return self._Ic.droplevel(1)  # single index pandas Series [A]


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

    def build_index(self):
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

    def __init__(self, folder=None, database_folder='operations',
                 read_txt=False, file_type='txt'):
        read_dina.__init__(self, database_folder, read_txt)  # read utilities
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

    def read_txt_file(self, folder, dropnan=True):
        filename = self.locate_file('data2.txt', folder=folder)
        self.data2 = self.read_csv(filename, dropnan=dropnan, split=',',
                                   dataframe=True)
        filename = self.locate_file('data3.txt', folder=folder)
        self.data3 = self.read_csv(filename, dropnan=dropnan, split=',',
                                   dataframe=True)

    def read_qda_file(self, folder, dropnan=True):
        filename = self.locate_file('data2.qda', folder=folder)
        self.data2 = self.read_qda(filename, dropnan=dropnan, split=',',
                                   dataframe=True)
        filename = self.locate_file('data3.qda', folder=folder)
        self.data3 = self.read_qda(filename, dropnan=dropnan, split=',',
                                   dataframe=True)


if __name__ is '__main__':

    # scn = read_scenario(read_txt=True)
    # scn.load_file(46)  # read / load single file
    # scn.load_folder()

    d2 = scenario_data(read_txt=False)
    # d2.load_folder()
    d2.load_file(24)

    # d2.to = 100
    # print(d2.Ic)
    # print(d2.Ip)






