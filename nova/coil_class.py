from nova.coil_set import CoilSet, CoilFrame
from matplotlib.collections import PatchCollection
from amigo.pyplot import plt
import numpy as np
import pandas as pd
import matplotlib


class CoilClass:
    '''
    CoilClass:
        - implements methods to manage input and
            output of data to/from the CoilSet class
        - provides interface to eqdsk files containing coil data
    '''
    def __init__(self, *args, eqdsk=None, dCoil=0):
        self.dCoil = dCoil
        self.coil = CoilFrame(
                additional_columns=['Ic', 'It', 'Nt', 'Nf', 'dCoil',
                                    'subcoil_index', 'mpc'],
                default_attributes={'dCoil': dCoil})
        self.subcoil = CoilFrame(
                additional_columns=['Ic', 'It',  'Nt', 'coil'])
        self.plasma = CoilFrame(additional_columns=['Ic', 'It',  'Nt'])
        self.add_coilset(*args)  # add list of coilset instances
        self.add_eqdsk(eqdsk)
        self.initalize_matrix()  # coil flux and force interaction matrices

    def initalize_matrix(self):
        '''
        matrix: a dictionary of force interaction matrices stored as dataframes
        '''
        self.matrix = {}
        self.matrix['inductance'] = {
                'Mc': None,  # line-current inductance matrix
                'Mt': None}  # amp-turn inductance matrix
        self.matrix['coil'] = {
                'Fx': None,  # radial force
                'Fz': None,  # vertical force
                'xFx': None,  # first radial moment of radial force
                'xFz': None,  # first radial moment of vertical force
                'zFx': None,  # first vertical moment of radial force
                'zFz': None,  # first vertical moment of vertical force
                'My': None}  # torque
        self.matrix['subcoil'] = {
                'Fx': None, 'Fz': None, 'xFx': None, 'xFz': None,
                'zFx': None, 'zFz': None, 'My': None}
        self.matrix['plasma'] = {'Fx': None, 'Fz': None}

    @property
    def coilset(self):
        return CoilSet(self.coil, self.subcoil, self.plasma, self.matrix)

    @coilset.setter
    def coilset(self, coilset):
        self.coil = coilset.coil
        self.subcoil = coilset.subcoil
        self.plasma = coilset.plasma
        self.matrix = coilset.matrix

    def add_coilset(self, *args):
        for coilset in args:
            for attribute in ['coil', 'subcoil', 'plasma']:
                coil = getattr(coilset, attribute)
                if not coil.empty:
                    getattr(self, attribute).add_coil(coil)

    @staticmethod
    def check_current_label(label):
        if label not in ['It', 'Ic']:  # turn-current, conductor current
            raise AttributeError(f'\ncurrent label: {label}\n'
                                 'not present in [It, Ic]\n')

    @property
    def current(self, label, index=None):
        self.check_current_label(label)
        if index is None:  # return full set
            index = self.coil.index
        value = self.coil.loc[index, 'It']  # turn current
        if label == 'It':  # filament current
            value /= self.coil.loc[index, 'Nf']
        elif label == 'Ic':  # conductor current
            value /= self.coil.loc[index, 'Nf']
        return value

    @current.setter
    def current(self, value, label, index=None):
        self.check_current_label(label)
        if index is None:  # return full set
            index = self.coil.index

    def add_eqdsk(self, eqdsk):
        if eqdsk:
            frame = self.coil.get_frame(eqdsk['xc'], eqdsk['zc'],
                                        eqdsk['dxc'], eqdsk['dzc'],
                                        It=eqdsk['It'], name='eqdsk',
                                        delim='')
            frame = self.categorize_coilset(frame)
            self.coil.concatenate(frame)
            self.add_subcoil(index=frame.index)

    @staticmethod
    def categorize_coilset(frame, xo=None, rename=True):
        '''
        categorize coils in frame as CS or PF
        categorization split based on coils minimum radius
        CS coils ordered by x then z
        PF coils ordered by theta taken about coilset centroid
        '''
        if xo is None:
            xo = (frame['x'].mean(), frame['z'].mean())
        # sort CS coils ['x', 'z']
        CSo = frame['x'].idxmin()
        xCS = frame.loc[CSo, 'x'] + frame.loc[CSo, 'dx']
        CS = frame.loc[frame['x'] <= xCS, :]
        CS = CS.sort_values(['x', 'z'])
        CS = CS.assign(part='CS')
        # sort PF coils ['theta']
        PF = frame.loc[frame['x'] > xCS, :]
        PF = PF.assign(theta=np.arctan2(PF['z'], PF['x']))
        PF = PF.sort_values('theta')
        PF.drop(columns='theta', inplace=True)
        PF = PF.assign(part='PF')
        if rename:
            CS.index = [f'CS{i}' for i in range(CS.nC)]
            PF.index = [f'PF{i}' for i in range(PF.nC)]
        frame = pd.concat([PF, CS])
        return frame

    def update_metadata(self, coil, **kwargs):
        # update coilset metadata, coil in ['coil', 'subcoil', 'plasma']
        getattr(self, coil)._update_metadata(**kwargs)

    def add_coil(self, *args, subcoil=True, **kwargs):
        index = self.coil.add_coil(*args, **kwargs)  # add primary coil
        if subcoil:
            self.add_subcoil(index=index)

    def add_subcoil(self, index=None, remesh=False):
        if index is None:  # re-mesh all coils
            remesh = True
            index = self.coil.index
        if remesh:
            self.subcoil.drop(self.subcoil.index, inplace=True)
        frame = [[] for __ in range(len(index))]
        for i, name in enumerate(index):
            frame[i] = self.mesh_coil(name, dCoil=self.coil.at[name, 'dCoil'],
                                      part=self.coil.at[name, 'part'])
        self.subcoil.concatenate(*frame)

    def mesh_coil(self, name, dCoil=None, part=None):
        if dCoil is None:
            dCoil = self.coil.at[name, 'dCoil']
        It = self.coil.at[name, 'It']  # coil turn-current
        cross_section = self.coil.at[name, 'cross_section']
        x, z, dx, dz = self.coil.loc[name, ['x', 'z', 'dx', 'dz']]
        if dCoil is None or dCoil == 0:
            dCoil = np.max([dx, dz])
        elif dCoil == -1:  # mesh per-turn (inductance calculation)
            Nt = self.coil.at[name, 'Nt']
            dCoil = (dx * dz / Nt)**0.5
        nx = int(np.ceil(dx / dCoil))
        nz = int(np.ceil(dz / dCoil))
        if nx < 1:
            nx = 1
        if nz < 1:
            nz = 1
        dx_, dz_ = dx / nx, dz / nz  # subcoil dimensions
        x_ = x + np.linspace(dx_ / 2, dx - dx_ / 2, nx) - dx / 2
        z_ = z + np.linspace(dz_ / 2, dz - dz_ / 2, nz) - dz / 2
        xm_, zm_ = np.meshgrid(x_, z_, indexing='ij')
        xm_ = np.reshape(xm_, (-1, 1))[:, 0]
        zm_ = np.reshape(zm_, (-1, 1))[:, 0]
        Nf = len(xm_)  # filament number
        self.coil.at[name, 'Nf'] = Nf
        frame = self.subcoil.get_frame(xm_, zm_, dx_, dz_,
                                       It=It/Nf, name=name,
                                       cross_section=cross_section, coil=name,
                                       part=part)
        self.coil.at[name, 'subcoil_index'] = list(frame.index)
        frame.loc[:, 'Nt'] = self.coil.at[name, 'Nt'] / Nf
        return frame

    def add_mpc(self, name, factor=1):
        '''
        define multi-point constraint linking a set of coils (Ic)
        name: list of coil names (present in self.coil.index)
        factor: inter-coil coupling factor
        '''
        if not pd.api.types.is_list_like(name):
            raise IndexError(f'name: {name} must be list like')
        elif len(name) == 1:
            raise IndexError(f'len({name}) must be > 1')
        if not pd.api.types.is_list_like(factor):
            factor = np.ones(len(name))
        elif len(factor) != len(name):
            raise IndexError(f'len(factor={factor}) must == 1 '
                             f'or == len(name={name})')
        for n, f in zip(name, factor):
            self.coil.at[n, 'mpc'] = (name[0], f)

    def plot_coil(self, ax, coil):
        if not coil.empty:
            patch = coil.loc[:, 'patch']  # sort patch based on zorder
            patch = patch.iloc[np.argsort([p.zorder for p in patch])]
            pc = PatchCollection(patch, edgecolor='k',
                                 match_original=True)
            ax.add_collection(pc)

    def plot(self, subcoil=True, plasma=True, label=False, current=False,
             ax=None):
        if ax is None:
            ax = plt.gca()
        if subcoil:
            self.plot_coil(ax, self.subcoil)
        else:
            self.plot_coil(ax, self.coil)
        if plasma:
            self.plot_coil(ax, self.plasma)
        if label or current:
            self.label_coil(ax, label, current)
        ax.axis('equal')
        ax.axis('off')

    def label_coil(self, ax, label, current, unit='A', coil=None, fs=None):
        if fs is None:
            fs = matplotlib.rcParams['legend.fontsize']
        if coil is None:
            coil = self.coil
        parts = np.unique(coil.part)
        if label is True:
            label = parts
        elif label is False:
            label = []
        if current is True:
            current = parts
        elif current is False:
            current = []
        for name, part in zip(coil.index, coil.part):
            x, z = coil.at[name, 'x'], coil.at[name, 'z']
            dx, dz = coil.at[name, 'dx'], coil.at[name, 'dz']
            if coil.part[name] == 'CS':
                drs = -2.5 / 3 * dx
                ha = 'right'
            else:
                drs = 2.5 / 3 * dx
                ha = 'left'
            if part in label and part in current:
                zshift = max([dz / 10, 0.5])
            else:
                zshift = 0
            if part in label:
                ax.text(x + drs, z + zshift, name, fontsize=fs,
                        ha=ha, va='center', color=0.2 * np.ones(3))
            if part in current:
                if unit == 'A':  # amps
                    Ic = coil.at[name, 'Ic']
                    txt = '{:1.1f}kA'.format(Ic * 1e-3)
                else:  # amp turns
                    It = coil.at[name, 'It']
                    if abs(It) < 0.1e6:  # display as kA.t
                        txt = '{:1.1f}kAT'.format(It * 1e-3)
                    else:  # MA.t
                        txt = '{:1.1f}MAT'.format(It * 1e-6)
                ax.text(x + drs, z - zshift, txt,
                        fontsize=fs, ha=ha, va='center',
                        color=0.2 * np.ones(3))

if __name__ is '__main__':

    cc = CoilClass(dCoil=0.05)
    cc.update_metadata('coil', additional_columns=['R'])
    cc.add_coil(1, 0, 0.1, 0.1, name='PF6', part='CS')
    cc.add_coil([1, 3], 1, 0.3, 0.3, name='PF', part='PF', delim='')
    cc.add_coil(1, 2, 0.1, 0.1, name='PF4', part='VS3')
    cc.add_coil(1.6, 1.5, 0.2, 0.2, name='PF7', part='vvin', dCoil=0.01)

    cc.plot()







