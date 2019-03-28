from nova.coil_set import CoilSet, CoilFrame
from matplotlib.collections import PatchCollection
from amigo.pyplot import plt
import numpy as np
import pandas as pd


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
                current_column='It',
                additional_columns=['Ic', 'It', 'Nt', 'Nf', 'dCoil',
                                    'subcoil_index', 'mpc'],
                default_attributes={'dCoil': dCoil})
        self.subcoil = CoilFrame(
                current_column='If', additional_columns=['If', 'Nt', 'coil'])
        self.plasma = CoilFrame(
                current_column='If', additional_columns=['If'])
        self.add_coilset(*args)  # add list of coilset instances
        self.add_eqdsk(eqdsk)
        self.matrix = None  # force interaction matrices

    @property
    def coilset(self):
        return CoilSet(self.coil, self.subcoil, self.plasma, self.matrix)

    @coilset.setter
    def coilset(self, coilset):
        self.coil = coilset.coil
        self.subcoil = coilset.subcoil
        self.plasma = coilset.plasma
        self.matrix = coilset.matrix

    @staticmethod
    def check_current_label(label):
        if label not in ['It', 'Ic', 'If']:  # turn, conductor, filament
            raise AttributeError(f'\ncurrent label: {label}\n'
                                 'not present in [It, Ic, If]\n')

    @property
    def current(self, label, index=None):
        self.check_current_label(label)
        if index is None:  # return full set
            index = self.coil.index
        value = self.coil.loc[index, 'It']  # turn current
        if label == 'If':  # filament current
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

    @staticmethod
    def sort_coilset(coilset):  # order coil dict for use by inverse.py
        index = np.append(coilset['index']['PF']['name'],
                          coilset['index']['CS']['name'])
        coilset['coil'] = coilset['coil'].reindex(index)
        nPF = coilset['index']['PF']['n']
        nCS = coilset['index']['CS']['n']
        coilset['index']['PF']['index'] = np.arange(0, nPF)
        coilset['index']['CS']['index'] = np.arange(nPF, nPF+nCS)

    def add_coilset(self, *args):
        for coilset in args:
            for attribute in ['coil', 'subcoil', 'plasma']:
                coil = getattr(coilset, attribute)
                if not coil.empty:
                    getattr(self, attribute).add_coil(coil)

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
        It = self.coil.at[name, 'It']
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
        If = It / Nf  # filament current
        self.coil.at[name, 'Nf'] = Nf
        frame = self.subcoil.get_frame(xm_, zm_, dx_, dz_, If=If, name=name,
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


    def inductance(self):
        cc.green(Xfix, Zfix, Xc, Zc, dX=dXc, dZ=dZc,
                                           cross_section=cross_section)
        '''
        fix_o = copy.deepcopy(self.fix)  # store current BCs
        if hasattr(self, 'G'):  # store coupling matrix
            self.G_ = self.G.copy()
        self.initalize_fix()  # reinitalize BC vector
        for i, name in enumerate(self.adjust_coils):
            coil = self.coil['active'][name]
            for x, z in zip(coil['x'], coil['z']):
                self.add_psi(1, point=(x, z))
        self.set_foreground()
        Gi = np.zeros((self.nC, self.nC))  # inductance coupling matrix
        Ncount = 0
        for i, name in enumerate(self.adjust_coils):
            Nf = self.coilset['coil'][name]['Nf']
            Gi[i, :] = np.sum(self.G[Ncount:Ncount+Nf, :], axis=0)
            Ncount += Nf
        Gi /= self.Iscale  # coil currents [A]
        turns = np.array([self.coilset['coil'][name]['Nt']
                          for name in self.coil['active']])
        turns = np.dot(turns.reshape(-1, 1), turns.reshape(1, -1))
        fillaments = np.array([self.coilset['coil'][name]['Nf']
                               for name in self.coil['active']])
        fillaments = np.dot(fillaments.reshape(-1, 1),
                            fillaments.reshape(1, -1))
        # PF/CS inductance matrix
        self.Mc = Gi / fillaments  # inductance [H]
        self.Mt = self.Mc * turns  # inductance [H]
        self.fix = fix_o  # reset BC vector
        if hasattr(self, 'Go'):  # reset coupling matrix
            self.G = self.G_
            del self.G_
        '''

    def plot_coil(self, ax, coil):
        if not coil.empty:
            patch = coil.loc[:, 'patch']  # sort patch based on zorder
            patch = patch.iloc[np.argsort([p.zorder for p in patch])]
            pc = PatchCollection(patch, edgecolor='k',
                                 match_original=True)
            ax.add_collection(pc)

    def plot(self, subcoil=True, plasma=True, ax=None):
        if ax is None:
            ax = plt.gca()
        if subcoil:
            self.plot_coil(ax, self.subcoil)
        else:
            self.plot_coil(ax, self.coil)
        if plasma:
            self.plot_coil(ax, self.plasma)

        ax.axis('equal')
        ax.axis('off')


if __name__ is '__main__':

    cc = CoilClass(dCoil=0.05)
    cc.update_metadata('coil', additional_columns=['R'])
    cc.add_coil(1, 0, 0.1, 0.1, name='PF6', part='CS')
    cc.add_coil([1, 3], 1, 0.3, 0.3, name='PF', part='PF', delim='')
    cc.add_coil(1, 2, 0.1, 0.1, name='PF4', part='VS3')
    cc.add_coil(1.6, 1.5, 0.2, 0.2, name='PF7', part='vvin', dCoil=0.01)

    cc.plot()







