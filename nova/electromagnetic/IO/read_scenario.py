from os.path import isfile, join
import re
import subprocess

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from astropy import units
#import docx

from nova.utilities.pyplot import plt
from nova.utilities.geom import rdp_extract
from nova.utilities.geom import vector_lowpass
from nova.electromagnetic.IO.read_waveform import read_dina


class operate:
    '''
    extract features from waveforms using rdp algorithum
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
        y = self.frame.loc[:, self.feature_name].values
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
            if value.loc[index] > threshold*max_value:
                if not self.feature_switch:
                    tag = self.feature_tag
                    feature['node'].rename(index={index: tag},
                                           inplace=True)
            elif value.loc[index] < threshold*max_value:
                if self.feature_switch:
                    tag = self.feature_tag
                    feature['node'].rename(index={index-1: tag},
                                           inplace=True)
                    break
        if self.feature_name == 'Ip':
            feature['node'].rename(index={0: 'IM'}, inplace=True)
            if value.loc[value.index[1]] > 0.01 * max_value:
                feature['node'].rename(index={'IM': 'SOP'}, inplace=True)
            else:
                for index in value.index[1:]:
                    if value.loc[index] < 0.01 * max_value:
                        feature['node'].rename(index={index: 'SOP'},
                                               inplace=True)
                        break
            if 'EOF' in feature['node'].index:
                EOF_index = int(feature['node'].loc['EOF', 'index'])
                for index in value.index[EOF_index+1:]:
                    if value.loc[index] < 0.01 * max_value:
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
                    vector = pd.Series(self.feature['node'].loc[index, :],
                                       dtype=float)
                    vector.loc['frame_index'] = \
                        np.argmin(abs(self.t - vector.t))
                    self.feature_keypoints = self.feature_keypoints.append(
                            vector)
        self.feature_keypoints.sort_values('t', inplace=True)

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
                    t_index[i] = self.feature['node'].loc[key, 't']
                index = [np.argmin(abs(self.t-t)) for t in t_index]
                vector = pd.Series(name=label, dtype=float)
                vector['frame_index'] = index
                vector['t'] = t_index
                vector['dt'] = np.diff(t_index)[0]
                psi = self.frame.loc[index, 'PSI(axis)'].values
                vector['dpsi'] = np.diff(psi)[0]
                self.feature_segments = self.feature_segments.append(vector)

    def extract_features(self):
        for variable, threshold in zip(['Ip', 'Pfus', 'Ti', 'Zx',
                                        '<PSIext>', '<PSIcoils>', 'PSI(axis)'],
                                       [0.95, 0.85, 0.25, 0.95, 1, 1, 1]):
            try:
                self.feature_extract(variable, threshold=threshold)
            except KeyError:
                pass
        self.extract_feature_segments()
        self.extract_feature_keypoints()

    def plot_features(self, feature_name=None, ax=None, ls=None):
        if feature_name is None:
            feature_name = ['Ip', 'Pfus', 'Ti', 'Zx']
        if ax is None:
            ax = plt.subplots(len(feature_name), 1, sharex=True)[1]
            if len(feature_name) == 1:
                ax = [ax]
        if ls is None:
            ls = '.-'
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
            label = name.replace('(', '').replace(')', '')
            label = label.replace('PSI', r'$\psi$')
            ax[i].set_ylabel(f'${name}$, {self.units["read"][name]}')
            ax[i].plot(self.feature['node'].t,
                       self.feature['node'].loc[:, name], ls, color=f'C{i}')
        plt.despine()
        plt.detick(ax)
        ax[-1].set_xlabel(f'$t$, {self.units["read"]["t"]}')


class interpolate:

    def __init__(self, extrapolate, sort=False):
        self._to = None  # time instance (float)
        self._ko = None  # keypoint instance (SOF, EOF, etc.)
        self._extrapolate = extrapolate
        self.sort = sort  # sort index

    def set_index(self, index=None):
        if index is None:
            index = self.data.columns
        else:
            index = [var for var in index if var in self.data]
        name = [var[0] for var in index]
        unit = np.array([var[1] for var in index])
        # initalise data structures ( + sort index)
        self.index = pd.Index(name)
        if self.sort:
            self.index, indexer = self.index.sort_values(return_indexer=True)
        else:
            indexer = np.arange(self.index.size)
        self._vector = pd.Series(index=self.index, dtype=float)
        self.units = pd.DataFrame(index=self.index,
                                  columns=['read', 'write', 'factor'])
        self.units['read'] = unit[indexer]
        self.units['write'] = unit[indexer]  # initalize as 1-1
        self.units['factor'] = 1

    def update_units(self, read, write):
        '''
        set / update conversion factors for selected units / index
        Attributes:
            read (str or [str]): read unit
            write (str): write unit
        '''
        read_units = self.units['read'].unique()
        if not pd.api.types.is_list_like(read):  # ensure iterable
            read = [read]
        for _read in read:
            if _read in read_units:
                index = self.units['read'] == _read
                self.units.loc[index, 'write'] = write
                self.units.loc[index, 'factor'] = units.Unit(_read).to(write)

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
        self.check_folder_set()
        if isinstance(to, str):  # feature keypoint
            self._ko = to
            if to in self.feature_keypoints.index:
                to = self.feature_keypoints.loc[to, 't']
            else:
                self.plot_features()
                raise IndexError(f'{to} not in {self.feature_keypoints.index}')
        else:
            self._ko = None  # unset keypoint
        self._to = to
        self.check_extrapolate()
        vector = self.interpolator(self._to)  # interpolate
        if hasattr(self, 'units'):
            self._vector.iloc[:] = vector * self.units['factor'].values
        else:
            self._vector.iloc[:] = vector

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


class scenario_limits:

    def __init__(self, folder=None, t='d3'):
        self.initialize_limits()
        self.load_data(folder, t=t)

    def reset_limits(self):
        self.limit = {'I': {}, 'F': {}, 'B': {}}

    def initialize_limits(self):
        'default limits for ITER coil-set'
        self.reset_limits()  # reset
        # current limits kA
        self.set_limit(ICS=45)
        self.set_limit(IPF1=48, IPF2=55, IPF3=55,
                       IPF4=55, IPF5=52, IPF6=52)
        # force limits
        self.set_limit(FCSsep=120, side='upper')
        self.set_limit(FCSsum=60, side='both')
        self.set_limit(FPF1=-150, FPF2=-75, FPF3=-90, FPF4=-40,
                           FPF5=-10, FPF6=-190, side='lower')
        self.set_limit(FPF1=110, FPF2=15, FPF3=40, FPF4=90,
                           FPF5=160, FPF6=170, side='upper')
        # field limits
        self.set_limit(BCS=[[45, 40], [12.6, 13]], side='abs')
        self.set_limit(BPF1=[[48, 41], [6.4, 6.5]], side='abs')
        self.set_limit(BPF2=[[55, 50], [4.8, 5]], side='abs')
        self.set_limit(BPF3=[[55, 50], [4.8, 5]], side='abs')
        self.set_limit(BPF4=[[55, 50], [4.8, 5]], side='abs')
        self.set_limit(BPF5=[[52, 33], [5.7, 6]], side='abs')
        self.set_limit(BPF6=[[48, 41], [6.4, 6.5]], side='abs')

    def set_limit(self, side='both', eps=1e-2, **kwargs):
        # set as ICSsum for [I][CSsum] etc...
        if side == 'both' or side == 'equal':
            index = [0, 1]
        elif side == 'lower':
            index = [0]
        elif side == 'upper':
            index = [1]
        elif side == 'abs':
            index = [0]
        else:
            errtxt = 'invalid side parameter [both, lower, upper]'
            errtxt += ': {}'.format(side)
            raise IndexError(errtxt)
        for key in kwargs:
            variable = key[0]
            if key[1:] not in self.limit[variable]:  # initalize limit
                if side == 'abs':
                    self.limit[variable][key[1:]] = [1e36]
                else:
                    self.limit[variable][key[1:]] = [-1e36, 1e36]
            if kwargs[key] is None:
                for i in index:
                    sign = self.sign_limit(i, side)
                    self.limit[variable][key[1:]][i] = sign * 1e36
            else:  # set limit(s)
                if side == 'abs':
                    self.limit[variable][key[1:]] = kwargs[key]
                else:
                    for i in index:
                        sign = self.sign_limit(i, side)
                        if side == 'equal':
                            value = sign * eps + kwargs[key]
                        else:
                            value = sign * kwargs[key]
                        self.limit[variable][key[1:]][i] = value

    def sign_limit(self, index, side):
        # only apply sign to symetric limits (side==both)
        if side in ['both', 'equal']:
            sign = 2*index-1
        else:
            sign = 1
        return sign

    def load_data(self, folder, t='d3'):
        self.d2 = scenario_data(folder)
        self.d3 = forcefield_data(folder)
        self.t = t
        self.normalize()

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, t):
        if isinstance(t, str):
            self._t = getattr(self, t).t
        else:
            self._t = t
            t = ''
        frame = []
        for dataset in ['d2', 'd3']:
            if t == dataset:
                frame.append(getattr(self, dataset).frame)
            else:
                frame.append(pd.DataFrame(
                        getattr(self, dataset).interpolator(self.t),
                        columns=getattr(self, dataset).index))
        self.frame = pd.concat(frame, axis=1)
        self.frame.rename(columns={'Fz_1': 'Fz_cssep', 'Fz_2': 'Fz_cssum'},
                          inplace=True)

    def extract_index(self):
        self.index = {}
        for var in self.limit:
            index = []
            for name in self.limit[var]:
                if var == 'F':
                    prefix = 'Fz'
                else:
                    prefix = var
                if var in ['F', 'B']:
                    prefix += '_'
                name = f'{prefix}{name.lower()}'
                if name in self.frame:
                    index.append(name)
                else:
                    index.extend([col for col in self.frame if name in col])
            self.index[var] = index

    def normalize(self):
        self.norm = pd.DataFrame()
        self.extract_index()
        for var in self.index:
            for name in self.index[var]:
                if var == 'I':
                    label = name[1:]
                else:
                    label = name.split('_')[-1]
                label = f'{label[:2].upper()}{label[2:]}'
                if label[-1] in ['l', 'u']:
                    label = label.upper()
                if label not in self.limit[var]:
                    label = label[:2]
                limit = self.limit[var][label]
                self.norm.loc[:, name] = self.frame.loc[:, name]
                if len(np.shape(limit)) == 1:  # static limit
                    value = self.norm.loc[:, name].copy()
                    for i, sign in enumerate([-1, 1]):
                        index = sign * value > 0
                        self.norm.loc[index, name] /= limit[i]
                else:  # current dependance
                    coil = name.split('_')[-1]
                    if 'cs1' in coil:
                        coil = 'cs1'
                    I = abs(self.frame.loc[:, f'I{coil}'])
                    limit = interp1d(*limit, fill_value='extrapolate')(I)
                    self.norm.loc[:, name] /= limit

    def plot(self, multi=True, strID='', ax=None):
        if ax is None:
            ax = plt.subplots(len(self.index), 1, sharex=True, sharey=True)[1]
        ylabel = {'I': '$I^*$', 'F': '$F^*$', 'B': '$B^*$'}
        for i, index in enumerate(self.index):
            norm = self.norm.loc[:, self.index[index]]
            max_norm = norm.max(axis=1)
            iC = 0
            waveform = np.zeros(len(self.t))
            for limit in self.index[index]:
                _waveform = norm.loc[:, limit].copy()
                max_index = np.isclose(_waveform, max_norm)
                if np.sum(max_index) > 0:
                    waveform[max_index] = _waveform[max_index]
                    _waveform[~max_index] = np.nan
                    _label = limit
                    if index == 'I':
                        _label = _label[1:]
                    else:
                        _label = _label.split('_')[-1]
                    _label = _label.upper()
                    _label = _label.replace('SEP', 'sep').replace('SUM', 'sum')
                    if np.nanmax(_waveform) > 0.75:
                        _color = f'C{iC}'
                        iC += 1
                    else:
                        _color = 'darkgray'
                        _label = None
                    if multi:
                        ax[i].plot(self.t, _waveform, label=_label,
                                   color=_color)
            if multi:
                # plot keypoints
                for keypoint, marker in zip(['SOF', 'SOB', 'EOB'],
                                        ['*', 'P', 'X']):
                    try:
                        t = self.d2.feature_keypoints.loc[keypoint, 't']
                        value = interp1d(self.t, waveform)(t)
                        ax[i].plot(t, value, marker=marker, color='k', ms=8)
                    except:
                        pass
                # multi axes legend
                ax[i].legend(ncol=1, loc='center right',
                         bbox_to_anchor=(1.15, 0.5), frameon=False,
                         fontsize='xx-small')
            else:
                strID = strID.replace(' ', '_')
                label = self.d2.filename.replace(strID, '')
                ax[i].plot(self.t, waveform, label=label)
            ax[i].plot([self.t[0], self.t[-1]], np.ones(2), '--',
                       color='gray')
            ax[i].set_ylabel(ylabel[index])
            ax[i].set_ylim(0.5)
        if multi:
            plt.suptitle(self.d2.filename)
        else:
            ax[1].legend(ncol=1, loc='center right',
                         bbox_to_anchor=(1.3, 0.5), frameon=False,
                         fontsize='xx-small')

        ax[-1].set_xlabel('$t$ s')
        plt.despine()

'''
class cosica_data(read_dina):

    def __init__(self):
        read_dina.__init__(self, database_folder=database_folder,
                           read_txt=read_txt)
'''

class scenario_data(read_dina, interpolate, operate):

    '''
    read DINA scenario data

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

    def __init__(self, folder=None, database_folder='scenarios',
                 read_txt=False, extrapolate=False, sort=False,
                 additional_columns=[]):
        '''
        Attributes:
            database_folder (str): database folder
            folder (str): scenario folder
            read_txt (bool): read / reread source text files
            extrapolate (bool): extrapolate beyond DINA data set
            additional_columns (list): additional column names
                None: load full dataset
                []: use default columns defined in subset
        '''
        read_dina.__init__(self, database_folder=database_folder,
                           read_txt=read_txt)
        interpolate.__init__(self, extrapolate, sort)
        operate.__init__(self)  # initalise operation instance
        self.additional_columns = additional_columns
        self.load_file(folder)

    def load_file(self, folder, verbose=True, **kwargs):
        self.folder = folder
        additional_columns = kwargs.get('additional_columns',
                                        self.additional_columns)
        if self.folder is not None:
            read_txt = kwargs.get('read_txt', self.read_txt)
            filename = self.locate_folder('*data2', folder)[0]
            filename += '_scenario_data'
            if additional_columns is None:
                filename += '_fullset'
            attributes = ['t', 'dt', '_to', 'interpolator', 'frame', '_vector',
                          '_Ic', 'index', 'units', 'Ic_iloc',
                          'additional_columns',
                          '_feature', 'feature_keypoints', 'feature_segments']
            if read_txt or not isfile(filename + '.pk'):
                self.read_file(folder, additional_columns)
                self.save_pickle(filename, attributes)
            else:
                if self.load_pickle(filename) or \
                        (sorted(additional_columns) !=
                         self.additional_columns):
                    self.read_file(folder, additional_columns)
                    self.save_pickle(filename, attributes)
            self.update_units(['kA', 'MA'], 'A')  # scale current units
            self.update_units(['mm', 'cm'], 'm')

    def read_file(self, folder, additional_columns=[]):
        try:
            scn = read_scenario(folder, self.database_folder, read_txt=False)
        except (AttributeError, ValueError):
            scn = read_scenario(folder, self.database_folder, read_txt=True)
        self.interpolate(scn.data2, additional_columns)
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
            if ('I' in var[0] and len(var[0]) <= 5) or ('V' in var[0]) \
                or ('PSI' in var[0]):
                self.data[var] *= coordinate_switch

    def subindex(self, additional_columns):
        '''
        return reduced subindex
        aditional paramters specified in additional_columns
        set additional_columns=[] to use default subindex
        '''
        # current columns (I)
        index = [var for var in self.data.columns.droplevel(1)
                 if ('I' in var and len(var) <= 5)]
        # additional baseline data
        index += ['Rcur', 'Zcur', 'ap', 'kp', 'Rp', 'Zp', 'a',
                  'Ksep', 'BETAp', 'li(3)', 't', 'PSI(axis)', 'Emag', 'Lp',
                  'q(95)', 'q(axis)', 'Vloop', 'D(PSI)res', 'Cejima',
                  '<PSIext>', '<PSIcoils>', 'Pfus', 'Ti', 'Zx']
        for column in additional_columns:  # append aditional columns
            if column not in index:
                index.append(column)
        # remove absent
        index = [idx for idx in index if idx in self.data.columns.droplevel(1)]
        index = self.data.columns.to_frame().loc[index]  # extract subset
        index = pd.MultiIndex.from_frame(index)  # generate multi-index
        return index

    def build_index(self, additional_columns):
        self.additional_columns = sorted(additional_columns)
        if additional_columns is None:  # load full dataset
            index = self.data.columns
        else:
            index = self.subindex(additional_columns)
        self.set_index(index)  # interpolate
        # extract coil names
        coil_names, self.Ic_iloc = np.array(
                [[v, i] for i, v in enumerate(self.index)
                 if ('I' in v and len(v) <= 5)]).T  # and v != 'Ip'
        self.Ic_iloc = self.Ic_iloc.astype(int)
        cs1_loc = coil_names.tolist().index('Ics1')
        coil_names = np.insert(coil_names, cs1_loc, ['Ics1', 'Ics1'])
        self.Ic_iloc = np.insert(
                self.Ic_iloc, cs1_loc,
                [self.Ic_iloc[cs1_loc], self.Ic_iloc[cs1_loc]])
        plasma_loc = coil_names.tolist().index('Ip')
        Ic_index = [name[1:].upper() for name in coil_names]
        Ic_index[plasma_loc] = 'Plasma'
        # CS central pair
        Ic_index[cs1_loc:cs1_loc+3] = ['CS1U', 'CS1', 'CS1L']
        self._Ic = pd.Series(index=Ic_index, dtype=float)
        if self.sort:
            self._Ic.sort_index(inplace=True)

    def interpolate(self, data, additional_columns=[]):
        '''
        build 1d vectorized interpolator, initalize self.data
        '''
        self.data = data.copy()  # create local copy
        self.rename_columns()
        self.correct_coordinates()
        self.build_index(additional_columns)  # build and sort
        unique_index = self.space()  # generate equidistant time vector
        data_block = np.zeros((len(unique_index), len(self.index)))
        self.data = self.data.droplevel(1, axis=1)  # drop units from column
        for i, index in enumerate(self.index):  # build input data
            data_block[:, i] = self.data[index][unique_index]
        t = self.data['t'].values[unique_index].flatten()
        self.interpolator = interp1d(t, data_block, axis=0,
                                     fill_value='extrapolate')
        self.data = None  # unlink raw data (reload from source)
        self.frame = pd.DataFrame(self.interpolator(self.t),
                                  columns=self.index)

    @property
    def Ip(self):
        '''
        return plasma current at time to
        '''
        return self._Ip

    @property
    def Ic(self):
        '''
        return pd.Series of coil currents at time to
        '''
        return self._Ic

    @property
    def to(self):
        return interpolate.to.fget(self)

    @to.setter
    def to(self, to):
        interpolate.to.fset(self, to)  # update interpolation instance
        self._Ic.iloc[:] = self.vector.iloc[self.Ic_iloc].values  # coils
        self._Ip = self.vector['Ip']  # plasma

    def plot(self, x='t', y='Ip', xslice=None, dt_filt=0, dt_min=0,
             strID='', ax=None, **kwargs):
        # slice
        if xslice is None:
            xslice = [self.frame.index[0], self.frame.index[-1]]
        else:
            if not pd.api.types.is_list_like(xslice):
                xslice = [xslice]
            for i in range(len(xslice)):
                if isinstance(xslice[i], str):
                    xslice[i] = self.feature_keypoints.loc[xslice[i],
                                                           'frame_index']
        if len(xslice) == 1:
            dt = 0
            xslice = xslice[0]
        else:
            dt = self.frame.loc[xslice[1], 't'] - \
                self.frame.loc[xslice[0], 't']
            xslice = slice(*xslice)
        x = self.frame.loc[xslice, x]
        y = self.frame.loc[xslice, y]
        # filter
        if dt_filt != 0:
            n_filt = int(dt_filt/self.dt)
            if n_filt % 2 == 0:
                n_filt += 1
            x = vector_lowpass(x.values, n_filt)
            y = vector_lowpass(y.values, n_filt)
        if ax is None:
            ax = plt.gca()

        if isinstance(x, float):
            ax.plot(x, y, **kwargs)
        else:
            if dt > dt_min:
                strID = strID.replace(' ', '_')
                label = kwargs.pop('label',
                                   self.filename.replace(strID, ''))
                ax.plot(x, y, label=label, **kwargs)
        return dt

    def plot_current(self, **kwargs):
        """Plot coil current vectors."""
        for name in self.index[self.Ic_iloc]:
            if name != 'Ip' and 'tf' not in name:
                label = name[1:].upper()
                self.plot(y=name, label=name, **kwargs)


class forcefield_data(read_dina, interpolate):

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
        B (pd.DataFrame): coil field vector
        F (pd.DataFrame): coil force vector
    '''

    def __init__(self, folder=None, database_folder='scenarios',
                 read_txt=False, extrapolate=False):
        '''
        Attributes:
            database_folder (str): database folder
            folder (str): scenario folder
            read_txt (bool): read / reread source text files
            extrapolate (bool): extrapolate beyond DINA data set
            additional_columns (list): additional column names
                None: load full dataset
                []: use default columns defined in subset
        '''
        read_dina.__init__(self, database_folder=database_folder,
                           read_txt=read_txt)
        interpolate.__init__(self, extrapolate)
        self.load_file(folder)

    def load_file(self, folder, verbose=True, **kwargs):
        self.folder = folder
        if self.folder is not None:
            read_txt = kwargs.get('read_txt', self.read_txt)
            filename = self.locate_folder('*data3', folder)[0]
            filename += '_field_data'
            attributes = ['t', 'dt', '_to', 'interpolator', 'frame', '_vector',
                          'index']
            if read_txt or not isfile(filename + '.pk'):
                self.read_file(folder)
                self.save_pickle(filename, attributes)
            else:
                if self.load_pickle(filename):
                    self.read_file(folder)
                    self.save_pickle(filename, attributes)

    def read_file(self, folder):
        scn = read_scenario(folder, self.database_folder, read_txt=False)
        self.interpolate(scn.data3)

    def rename_columns(self):
        self.data.rename(columns={'time': 't'}, level=0, inplace=True)
        columns = {}
        for var in self.data.columns:
            if var[0][:3] == 'Fr_':
                columns[var[0]] = var[0].replace('Fr_', 'Fx_')
        self.data.rename(columns=columns, level=0, inplace=True)

    def correct_coordinates(self):
        '''
        correct coodrinate system for DINA data created before self.date_switch
        '''
        if self.date > self.date_switch:
            coordinate_switch = 1
        else:  # old file - correct coordinates
            coordinate_switch = -1
        for var in self.data.columns:
            if var[0][:2] == 'Ip':  # field given as L2norm
                self.data[var] *= coordinate_switch

    def interpolate(self, data):
        '''
        build 1d vectorized interpolator, initalize self.data
        '''
        self.data = data.copy()  # create local copy
        self.rename_columns()
        self.correct_coordinates()
        self.set_index(self.data.columns)  # interpolate
        unique_index = self.space()  # generate equidistant time vector
        data_block = np.zeros((len(unique_index), len(self.index)))
        self.data = self.data.droplevel(1, axis=1)  # drop units from column
        for i, index in enumerate(self.index):  # build input data
            data_block[:, i] = self.data[index][unique_index]
        t = self.data['t'].values[unique_index].flatten()
        self.interpolator = interp1d(t, data_block, axis=0,
                                     fill_value='extrapolate')
        self.data = None  # unlink raw data (reload from source)
        self.frame = pd.DataFrame(self.interpolator(self.t),
                                  columns=self.index)

    @property
    def to(self):
        return interpolate.to.fget(self)

    @to.setter
    def to(self, to):
        interpolate.to.fset(self, to)  # update interpolation instance


class read_scenario(read_dina):

    def __init__(self, folder=None, database_folder='scenarios',
                 read_txt=False, file_type='txt'):
        read_dina.__init__(self, database_folder, read_txt)  # read utilities
        if folder is not None:
            self.load_file(folder, file_type=file_type)

    def load_file(self, folder, verbose=True, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        file_type = kwargs.get('file_type', 'txt')
        filename, file_type = self.locate_folder('*data2', folder, file_type)
        if read_txt or not isfile(filename + '.pk'):
            self.read_file(folder, file_type=file_type, verbose=verbose)
            self.save_pickle(filename, ['data2', 'data3'])
        else:
            if self.load_pickle(filename):
                self.read_file(folder, file_type=file_type, verbose=verbose)
                self.save_pickle(filename, ['data2', 'data3'])

    def read_file(self, folder, file_type='txt', verbose=True):
        if verbose:
            print(f'reading {self.filename}.{file_type}')
        if file_type == 'txt':
            self.read_txt_file(folder)
        elif file_type == 'qda':
            self.read_qda_file(folder)

    def read_txt_file(self, folder, dropnan=True):
        filename = self.locate_file('*data2.txt', folder=folder)
        self.data2 = self.read_csv(filename, dropnan=dropnan, split=',',
                                   dataframe=True)
        filename = self.locate_file('*data3.txt', folder=folder)
        self.data3 = self.read_csv(filename, dropnan=dropnan, split=',',
                                   dataframe=True)

    def read_qda_file(self, folder, dropnan=True):
        filename = self.locate_file('*data2.qda', folder=folder)
        self.data2 = self.read_qda(filename, dropnan=dropnan, split=',',
                                   dataframe=True)
        filename = self.locate_file('*data3.qda', folder=folder)
        self.data3 = self.read_qda(filename, dropnan=dropnan, split=',',
                                   dataframe=True)

    def upload(self, folder, **kwargs):
        'upload DINA data to ITER cluster'
        self.load_file(folder, **kwargs)  # load local file
        metadata = {
            'title': 'DINA scenario data',
            'IDM referance': self.read_IDM(folder),
            'folder': self.filename,
            'date': self.date.strftime('%m/%Y')}
        netCDFfile = join(self.filepath, 'scenario_data')
        self.save_netCDF(netCDFfile, ['data2', 'data3'], **metadata)
        with open(netCDFfile+'.nc', 'w') as f:
            f.write(f'{np.random.random(9)}')
        print(netCDFfile+'.nc')
        secure_hash = self.hash_file(netCDFfile+'.nc', algorithm='md5')
        print(secure_hash)
        # upload file to ITER cluster'
        subprocess.run(['scp', netCDFfile+'.nc',
                        'hpc-login.iter.org:/work/imas/shared/external/'
                        f'assets/nova/MD5/{secure_hash}'])
        #self.load_netCDF(netCDFfile+'.nc')


    def read_IDM(self, folder):
        'read IDM referance from word file'
        file = self.locate_file_type('Parameters*Data2*', 'docx', folder)[0]
        doc = docx.Document(file)
        for paragraph in doc.paragraphs:
            IDM = re.search(r'\[(.*)\]', paragraph.text)
            if IDM:
                IDM = IDM.groups(1)[0]
                break
        return IDM

if __name__ == '__main__':

    # d3 = field_data(read_txt=False)
    # d3.load_folder()
    # d3.load_file('15MA DT-DINA2016-01_v1.1')

    scn = read_scenario(database_folder='scenarios', read_txt=True)
    #scn.load_folder()
    scn.load_file(-1)

    #scn.load_file('15MA DT-DINA2020-04', read_txt=True)

    #scn.read_IDM(-1)

    '''
    d2 = scenario_data(read_txt=False)
    #d2.load_folder()
    d2.load_file(-1)  # read / load single file
    #d2.load_file(-1, read_txt=True)  # read / load single file

    #plt.set_context('talk')
    #d2.plot_features(feature_name=['Ip', '<PSIcoils>'])

    d3 = forcefield_data(read_txt=False)
    #d3.load_folder()
    d3.load_file(-1)

    d3.to = 'IM'
    '''

    '''
    d2.ko = 100
    print(d2.Ic)
    # print(d2.Ip)
    '''
