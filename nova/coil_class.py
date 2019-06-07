from nova.coil_set import CoilSet, CoilFrame
from matplotlib.collections import PatchCollection
from amigo.pyplot import plt
import numpy as np
import pandas as pd
import matplotlib
from nova.biot_savart import biot_savart, self_inductance
from nep.DINA.read_scenario import scenario_data
from astropy import units


class CoilClass:
    '''
    CoilClass:
        - implements methods to manage input and
            output of data to/from the CoilSet class
        - provides interface to eqdsk files containing coil data
        - provides interface to DINA scenaria data
    '''
    def __init__(self, *args, eqdsk=None, dCoil=0, turn_fraction=1,
                 scenario_filename=None):
        self.dCoil = dCoil
        self.coil = CoilFrame(
                additional_columns=['Ic', 'It', 'Nt', 'Nf', 'dCoil',
                                    'subindex', 'mpc', 'turn_fraction'],
                default_attributes={'dCoil': dCoil,
                                    'turn_fraction': turn_fraction})
        self.subcoil = CoilFrame(
                additional_columns=['Ic', 'It',  'Nt', 'coil'])
        self.matrix = CoilSet.empty_matrix()
        self.add_coilset(*args)  # add list of coilset instances
        self.add_eqdsk(eqdsk)
        self.initalize_functions()  # initalise functions
        self._scenario_filename = None
        self.scenario_filename = scenario_filename

    def initalize_functions(self):
        self.bs = biot_savart()
        self.d2 = scenario_data()

    @property
    def coilset(self):
        return CoilSet(self.coil, self.subcoil, self.matrix)

    @coilset.setter
    def coilset(self, coilset):
        self.coil = coilset.coil
        self.subcoil = coilset.subcoil
        self.matrix = coilset.matrix

    def subset(self, coil_index, invert=False):
        if not pd.api.types.is_list_like(coil_index):
            coil_index = [coil_index]
        if invert:
            coil_index = self.coil.loc[~self.coil.index.isin(coil_index)].index
        subindex = []
        for index in coil_index:
            subindex.extend(self.coil.loc[index, 'subindex'])
        return CoilSet(self.coil.loc[coil_index], self.subcoil.loc[subindex])

    def add_coilset(self, *args):
        if len(args) == 1:  # single coilset
            self.coilset = args[0]
        else:  # build from multiple coilsets
            for coilset in args:
                for attribute in ['coil', 'subcoil']:
                    coil = getattr(coilset, attribute)
                    if not coil.empty:
                        getattr(self, attribute).add_coil(coil)

    @property
    def Ic(self):
        '''
        Returns:
            self.coil.Ic (pd.Series): coil instance line current [A]
        '''
        return self.coil.Ic

    @Ic.setter
    def Ic(self, value):
        self.set_current(value, 'Ic')

    @property
    def It(self):
        '''
        Returns:
            self.coil.It (pd.Series): coil instance turn current [A.turns]
        '''
        return self.coil.It

    @It.setter
    def It(self, value):
        self.set_current(value, 'It')

    @property
    def Ip(self):
        # return total plasma current
        return self.coil.loc['Plasma', 'Ic']

    @Ip.setter
    def Ip(self, Ip):
        self.coil.Ic = pd.Series({'Plasma': Ip})

    @property
    def scenario_filename(self):
        return self._scenario_filename

    @scenario_filename.setter
    def scenario_filename(self, filename):
        '''
        Attributes:
            filename (str) DINA filename
            filename (int) DINA fileindex
        '''
        if filename != self._scenario_filename:
            self.d2.load_file(filename)
            self._scenario_filename = self.d2.filename

    @property
    def scenario(self):
        '''
        return scenario metadata
        '''
        return pd.Series({'filename': self.scenario_filename,
                          'to': self.d2.to, 'ko': self.d2.ko})

    @scenario.setter
    def scenario(self, to):
        '''
        Attributes:
            to (float): input time
            to (str): feature_keypoint
        '''
        self.d2.to = to  # update scenario data (time or keypoint)
        self.Ic = self.d2.Ic  # update coil currents
        self.update_plasma()  # update plasma based on d2 data

    def set_current(self, value, current_column):
        '''
        update current in coil and subcoil instances (Ic and It)
        index built as union of value.index and coil.index
        mpc constraints applied
        Args:
            value (pd.Series): current update
            current_column (str):
                'Ic' == line current [A]
                'It' == turn current [A.turns]
        '''
        if current_column not in ['Ic', 'It']:
            raise IndexError(f'current column: {current_column} '
                             'not in [Ic, It]')
        index = [n for n in value.index if n in self.coil.index]  # union index
        setattr(self.coil, current_column, value.loc[index])  # coil
        coil_current = getattr(self.coil, current_column)
        subcoil_current = pd.Series(index=self.subcoil.index)
        for name, subindex in zip(self.coil.index, self.coil.subindex):
            subcoil_current.loc[subindex] = coil_current[name]
        setattr(self.subcoil, current_column, subcoil_current)  # subcoil

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
        x, z, dx, dz = self.coil.loc[name, ['x', 'z', 'dx', 'dz']]
        cross_section = self.coil.at[name, 'cross_section']
        turn_fraction = self.coil.at[name, 'turn_fraction']
        if dCoil is None or dCoil == 0:
            dCoil = np.mean([dx, dz])
        if dCoil == -1:  # mesh per-turn (inductance calculation)
            cross_section = 'circle'
            Nt = self.coil.at[name, 'Nt']
            dCoil = (dx * dz / Nt)**0.5
        else:
            turn_fraction = 1
        nx = int(np.round(dx / dCoil))
        nz = int(np.round(dz / dCoil))
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
        frame = self.subcoil.get_frame(xm_, zm_,
                                       turn_fraction*dx_,
                                       turn_fraction*dz_,
                                       It=It/Nf, name=name,
                                       cross_section=cross_section, coil=name,
                                       part=part)
        self.coil.at[name, 'subindex'] = list(frame.index)
        frame.loc[:, 'Nt'] = self.coil.at[name, 'Nt'] / Nf
        return frame

    def drop_coil(self, index=None):
        if index is None:
            index = self.coil.index
        if not pd.api.types.is_list_like(index):
            index = [index]
        for name in index:
            if name in self.coil.index:
                self.subcoil.drop_coil(self.coil.loc[name, 'subindex'])
                self.coil.drop_coil(index)

    def add_plasma(self, *args, **kwargs):
        label = kwargs.pop('label', 'Pl')  # filament prefix
        name = kwargs.pop('name', 'Pl_0')
        part = kwargs.pop('part', 'plasma')
        coil = kwargs.pop('coil', 'Plasma')
        cross_section = kwargs.pop('cross_section', 'square')
        self.drop_coil('Plasma')
        # add plasma filaments to subcoil
        subindex = self.subcoil.add_coil(
                *args, label=label, part=part, coil=coil, name=name,
                cross_section=cross_section, **kwargs)
        Ip = self.subcoil.It[subindex]  # filament currents
        Ip_net = Ip.sum()  # net plasma current
        if not np.isclose(Ip.sum(), 0):
            Nt = Ip / Ip_net  # filament turn number
        else:
            Nt = np.ones(Ip.size)
        self.subcoil.loc[subindex, 'Nt'] = Nt
        xpl = self.subcoil.x[subindex]  # filament x-location
        zpl = self.subcoil.z[subindex]  # filament z-location
        dx = dz = np.sqrt(np.sum(self.subcoil.dx[subindex] *
                                 self.subcoil.dz[subindex]))
        # add plasma to coil (x_gmd, z_amd)
        Nf = Ip.size
        self.coil.add_coil(biot_savart.gmd(xpl, Nt), biot_savart.amd(zpl, Nt),
                           dz, dx, Nf=Nf, dCoil=None, cross_section='circle',
                           name='Plasma', part='plasma', turn_fraction=1)
        self.coil.at['Plasma', 'subindex'] = list(subindex)
        #if Nf > 1:
        #    self.inductance('Plasma', update=True)  # re-size plasma coil
        self.Ic = pd.Series({'Plasma': Ip_net})  # update plasma net current

    def inductance(self, name, update=False):
        '''
        calculate self-inductance and geometric mean of single coil

        Attributes:
            name (str): coil name (present in self.coil.index)
            update (bool): apply update to self.coil.loc[name]
        '''
        coilset = self.subset(name)  # create single coil coilset
        biot_savart(coilset).inductance()  # calculate self-inductance
        L = coilset.matrix['inductance']['Mt'].loc[name, name]
        dr = self_inductance(coilset.coil.x[name]).minor_radius(L)
        # calculate geometric and arithmetic means
        Nt = coilset.subcoil.Nt
        x_gmd = biot_savart.gmd(coilset.subcoil.x, Nt)
        z_amd = biot_savart.amd(coilset.subcoil.z, Nt)
        if update:  # apply update
            coilset.coil.loc[name, ['x', 'z']] = x_gmd, z_amd
            coilset.coil.loc[name, ['dx', 'dz']] = 2*dr, 2*dr
            CoilFrame.patch_coil(coilset.coil)  # re-generate coil patch
            self.coil.loc[name] = coilset.coil.loc[name]
        coilset = None  # remove coilset
        return L

    def update_plasma(self):
        coordinates = ['Rcur', 'Zcur']
        if not np.array([c in self.d2.unit for c in coordinates]).all():
            coordinates = ['Rp', 'Zp']
        v2 = self.d2.vector.loc[['Lp'] + coordinates].droplevel(1)
        if 'Lp' not in self.d2.unit:
            v2['Lp'] = 1.1e-5  # default plasma self-inductance H
        scale = units.Unit(self.d2.unit[coordinates[0]]).to('m')
        if not np.isclose(scale, 1):
            for c in coordinates:
                v2.loc[c] *= scale  # convert coordinates
        Xp, Zp, Lp = v2.loc[coordinates + ['Lp']]
        dr = self_inductance(Xp).minor_radius(Lp)

        if 'Plasma' not in self.coil.index:  # create plasma coilset
            self.add_plasma(Xp, Zp, 2*dr, 2*dr, cross_section='circle')
        else:  # update plasma coilset
            subindex = self.coil.at['Plasma', 'subindex']
            # TODO update multi-filament plasma model
        self.Ip = self.d2.Ip  # update plasma current

    def calculate_inductance(self, mutual=False):
        self.bs.mutual = mutual
        self.bs.load_coilset(self.coilset)
        self.bs.inductance()  # values written in self.coilset.matrix

    def add_mpc(self, name, factor=1):
        '''
        define multi-point constraint linking a set of coils
        name: list of coil names (present in self.coil.index)
        factor: inter-coil coupling factor
        '''
        if not pd.api.types.is_list_like(name):
            raise IndexError(f'name: {name} must be list like')
        elif len(name) == 1:
            raise IndexError(f'len({name}) must be > 1')
        if not pd.api.types.is_list_like(factor):
            factor = factor * np.ones(len(name)-1)
        elif len(factor) != len(name)-1:
            raise IndexError(f'len(factor={factor}) must == 1 '
                             f'or == len(name={name})-1')
        for n, f in zip(name[1:], factor):
            self.coil.at[n, 'mpc'] = (name[0], f)

    def plot_coil(self, ax, coil, alpha=1):
        if not coil.empty:
            patch = coil.loc[:, 'patch']  # sort patch based on zorder
            no_patch = pd.isnull(coil.loc[:, 'patch'])
            if no_patch.any():
                coil.patch_coil(coil)
            patch = patch.iloc[np.argsort([p.zorder for p in patch])]
            pc = PatchCollection(patch, edgecolor='k',
                                 match_original=True, alpha=alpha)
            ax.add_collection(pc)

    def plot(self, subcoil=True, plasma=True, label=False, current=False,
             ax=None, unit='A'):
        if ax is None:
            ax = plt.gca()
        if subcoil:
            self.plot_coil(ax, self.subcoil)
        else:
            self.plot_coil(ax, self.coil, alpha=0.5)
        if 'Plasma' in self.coil.index and plasma:
            self.label_plasma(ax)
        if label or current:
            self.label_coil(ax, label, current, unit)
        ax.axis('equal')
        ax.axis('off')

    def label_plasma(self, ax, fs=None):
        if fs is None:
            fs = matplotlib.rcParams['legend.fontsize']
        x = self.coil.x['Plasma']
        z = self.coil.z['Plasma']
        ax.text(x, z, f'{1e-6*self.Ip:1.1f}MA', fontsize=fs,
                ha='center', va='center', color=0.9 * np.ones(3))

    def label_coil(self, ax, label, current, unit, coil=None, fs=None):
        if fs is None:
            fs = matplotlib.rcParams['legend.fontsize']
        if coil is None:
            coil = self.coil
        parts = np.unique(coil.part)
        parts = [p for p in parts if p != 'plasma']
        if label is True:
            label = parts
        elif label is False:
            label = []
        if current is True:
            current = parts
        elif current is False:
            current = []
        ylim = np.diff(ax.get_ylim())[0]
        for name, part in zip(coil.index, coil.part):
            x, z = coil.at[name, 'x'], coil.at[name, 'z']
            dx, dz = coil.at[name, 'dx'], coil.at[name, 'dz']
            if coil.part[name] == 'CS':
                drs = -2.0 / 3 * dx
                ha = 'right'
            else:
                drs = 2.0 / 3 * dx
                ha = 'left'
            if part in label and part in current:
                zshift = max([dz / 10, ylim / 5])
            else:
                zshift = 0
            if part in label:
                ax.text(x + drs, z + zshift, name, fontsize=fs,
                        ha=ha, va='center', color=0.2 * np.ones(3))
            if part in current:
                if unit == 'A':  # amps
                    Ic = coil.at[name, 'Ic']
                    txt = '{:1.1f}kA'.format(Ic * 1e-3)
                elif unit == 'AT':  # amp turns
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
    cc.add_coil(1, 0, 0.1, 0.1, name='PF6', part='CS', cross_section='circle')
    cc.add_coil([2, 2, 3, 3.5], [1, 0, -1, -3], 0.3, 0.3,
                name='PF', part='PF', delim='', Nt=12)
    cc.add_coil(3, 2, 0.1, 0.1, name='PF4', part='VS3', turn_fraction=0.75,
                Nt=15, dCoil=-1)
    cc.add_coil(5.6, 3.5, 0.2, 0.2, name='PF7', part='vvin', dCoil=0.01)

    #cc.add_plasma(1, [1.5, 2, 2.5], 0.5, 0.2, It=-15e6/3)
    #cc.add_plasma(6, [1.5, 2, 2.5], 0.5, 0.2, It=-15e6/3)

    cc.scenario_filename = 48
    cc.scenario = 'EOF'

    cc.plot(label=True, current=True)
    cc.plot(subcoil=False)



    #cc.calculate_inductance()
