import io
import os

import numpy as np
import pandas as pd

import amigo
from amigo.IO import pythonIO
from amigo.geom import rotate2D
from amigo.pyplot import plt
from amigo.png_tools import data_load
from amigo.IO import class_dir
import nep_data.geom
from nova.electromagnetic.coilclass import CoilClass
from nova.electromagnetic.coilset import CoilSet
from nova.electromagnetic.machinedata import MachineData

resistivity_ss = 0.815e-6  # steel electrical resistivity at 100C
resistivity_cu = 0.029e-6  # copper electrical resistivity at 100C

density_ss = 8050
density_cu = 8940


class ITERcoilset(CoilClass):

    def __init__(self, read_txt=False, **kwargs):
        self.read_txt = read_txt
        CoilClass.__init__(self, **kwargs)
        #self.update_coilframe_metadata(
        #    'coil', _additional_columns=['m', 'material', 'R'])
        self.load_coilset(**kwargs)

    def load_coilset(self, **kwargs):
        read_txt = kwargs.pop('read_txt', self.read_txt)
        filename, coils, kwargs = self.select_coils(**kwargs)
        filepath = self.filepath(filename)
        if not os.path.isfile(filepath + '.pk') or read_txt:
            self.build_coilset(coils, **kwargs)
            self.save_coilset(filename)
        else:
            CoilClass.load_coilset(self, filename)
            #if self.grid.generate_grid(**kwargs):  
            #    self.save_coilset(filename)  # save on-demand update

    def select_coils(self, **kwargs):
        coils = kwargs.pop('coils', ['pf', 'vsj', 'vv'])  # default set
        if not pd.api.types.is_list_like(coils):
            coils = coils.replace('_', ' ')
            coils = coils.split()
        coils = [c for c in coils 
                 if c in ['pf', 'vs', 'vsj', 'vv', 'trs', 'dir']]
        coils = list(np.unique(np.sort(coils)))
        if 'vs' in coils and 'vsj' in coils:
            coils.remove('vs')  # remove vs if selection is over-defined
        filename = '_'.join(coils)
        filename = filename.replace('pf', f'pf{self.dCoil*1e3:.0f}')
        return filename, coils, kwargs

    def build_coilset(self, coils, **kwargs):
        dCoil = self.dCoil  # PF subcoil dimension
        # targets = kwargs.get('targets', {})  # data extraction targets
        if 'pf' in coils:  # pf coilset
            self.append_coilset(PFgeom(VS=False, dCoil=dCoil).coilset)
        if 'vsj' in coils or 'vs' in coils:  # vs coils with/without ss jacket
            jacket = True if 'vsj' in coils else False
            self.append_coilset(VSgeom(jacket=jacket).coilset)  
        if 'vv' in coils:  # vv coilset (inner, outer, and trs)
            self.append_coilset(
                MachineData().load_coilset(part_list='vvin vvout'))
            #self.append_coilset(VVcoils(model='full', read_txt=True).coilset)
        if 'trs' in coils:
            self.append_coilset(
                MachineData().load_coilset(part_list='trs'))
        if 'dir' in coils:
            self.append_coilset(
                MachineData().load_coilset(part_list='dir'))
        #self.mutual.solve_interaction()  # compute mutual interaction
        #self.grid.generate_grid(**kwargs, regen=True)
        #self.add_targets(targets=targets)
        #self.update_interaction()
        
        
class ITERdata(pythonIO):

    def __init__(self, dCoil=0.2, plasma=True, source='PCR', read_txt=False):
        pythonIO.__init__(self)  # python read/write
        self.plasma = plasma
        self.source = source
        self.dCoil = dCoil
        self.read_txt = read_txt
        self.path = os.path.join(class_dir(nep_data.geom) + '/')
        self.cc = CoilClass(dCoil=dCoil)
        self.load_coils()

    def load_coils(self, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        self.source = kwargs.get('source', self.source)
        filename = os.path.join(self.path, f'IO_{self.source}')
        if self.dCoil != -1:
            filename += f'_{self.dCoil}'
        if self.plasma:
            filename += '_plasma'
        if read_txt or not os.path.isfile(filename + '.pk'):
            self.read_coils()
            self.save_pickle(filename, ['coilset', 'Mipp', 'Mddd', 'names'])
        else:
            self.load_pickle(filename)
            self.cc.append_coilset(self.coilset)

    def read_coils(self):
        self.initalize_coilclass(source=self.source)
        self.load_Mddd()
        self.load_Mipp()

    def initalize_coilclass(self, source='PCR'):
        pf = PFgeom(dCoil=self.dCoil, source=source).cs
        self.cc.append_coilset(pf)
        if self.plasma:
            self.cc.add_coil(6.2, 0.5, 0.1, 0.1, name='Plasma', part='Plasma')
        self.names = list(self.cc.coil.index)
        self.coilset = self.cc.coilset

    def build_matrix(self, tab):
        M = pd.DataFrame(index=self.names, columns=self.names)
        for n0 in self.names:
            for n1 in self.names:
                if tab[n0][n1] is not None or tab[n1][n0] is not None:
                    m = tab[n0][n1] if tab[n0][n1] is not None else tab[n1][n0]
                    M.loc[n0, n1] = m
                    if n0 != n1:
                        M.loc[n1, n0] = m
                else:
                    print(f'inductance {n0}-{n1} not found')
        return M

    def load_Mipp(self):
        # tabular data from Inductances of ITER magnets 2MFSSS v1.1
        ipp = dict([(n0, dict([(n1, None) for n1 in self.names]))
                    for n0 in self.names])

        ipp['PF1'].update(PF1=7.00E-01)
        ipp['PF2'].update(PF1=1.13E-01, PF2=4.72E-01)
        ipp['PF3'].update(PF1=9.98E-02, PF2=2.38E-01, PF3=1.85E+00)
        ipp['PF4'].update(PF1=4.90E-02, PF2=1.03E-01, PF3=4.52E-01,
                          PF4=1.55E+00)
        ipp['PF5'].update(PF1=2.42E-02, PF2=4.71E-02, PF3=1.83E-01,
                          PF4=3.58E-01, PF5=1.55E+00)
        ipp['PF6'].update(PF1=1.56E-02, PF2=2.90E-02, PF3=1.06E-01,
                          PF4=1.88E-01, PF5=4.89E-01, PF6=2.38E+00)
        ipp['CS3U'].update(PF1=1.34E-01, PF2=4.43E-02, PF3=4.89E-02,
                           PF4=2.81E-02, PF5=1.57E-02, PF6=1.14E-02,
                           CS3U=7.91E-01)
        ipp['CS2U'].update(PF1=6.00E-02, PF2=3.64E-02, PF3=5.11E-02,
                           PF4=3.52E-02, PF5=2.26E-02, PF6=1.83E-02,
                           CS3U=2.47E-01, CS2U=7.91E-01)
        ipp['CS1U'].update(PF1=2.80E-02, PF2=2.64E-02, PF3=4.86E-02,
                           PF4=4.18E-02, PF5=3.31E-02, PF6=3.16E-02,
                           CS3U=5.39E-02, CS2U=2.47E-01, CS1U=7.91E-01)
        ipp['CS1L'].update(PF1=1.45E-02, PF2=1.82E-02, PF3=4.24E-02,
                           PF4=4.61E-02, PF5=4.77E-02, PF6=5.93E-02,
                           CS3U=1.85E-02, CS2U=5.39E-02, CS1U=2.46E-01,
                           CS1L=7.91E-01)
        ipp['CS2L'].update(PF1=8.31E-03, PF2=1.24E-02, PF3=3.47E-02,
                           PF4=4.64E-02, PF5=6.62E-02, PF6=1.21E-01,
                           CS3U=8.25E-03, CS2U=1.86E-02, CS1U=5.39E-02,
                           CS1L=2.47E-01, CS2L=7.91E-01)
        ipp['CS3L'].update(PF1=5.14E-03, PF2=8.53E-03, PF3=2.73E-02,
                           PF4=4.25E-02, PF5=8.17E-02, PF6=2.48E-01,
                           CS3U=4.34E-03, CS2U=8.26E-03, CS1U=1.85E-02,
                           CS1L=5.39E-02, CS2L=2.46E-01, CS3L=7.91E-01)
        if self.plasma:
            ipp['Plasma'].update(PF1=3.15E-04, PF2=4.91E-04, PF3=1.21E-03,
                                 PF4=1.11E-03, PF5=7.37E-04, PF6=5.57E-04,
                                 CS3U=2.59E-04, CS2U=4.06E-04, CS1U=5.21E-04,
                                 CS1L=4.80E-04, CS2L=3.33E-04, CS3L=2.04E-04,
                                 Plasma=2.40E-05)
        self.Mipp = self.build_matrix(ipp)

    def load_Mddd(self):
        # tabular data from DDD11-4: PF Coils and Supports 2LGJUP v3.0
        ddd = dict([(n0, dict([(n1, None) for n1 in self.names]))
                    for n0 in self.names])
        ddd['PF1'].update(PF1=7.076E-01)
        ddd['PF2'].update(PF1=1.132E-01, PF2=4.740E-01)
        ddd['PF3'].update(PF1=1.008E-01, PF2=2.393E-01, PF3=1.860E+00)
        ddd['PF4'].update(PF1=4.917E-02, PF2=1.027E-01, PF3=4.552E-01,
                          PF4=1.565E+00)
        ddd['PF5'].update(PF1=2.431E-02, PF2=4.710E-02, PF3=1.838E-01,
                          PF4=3.606E-01, PF5=1.562E+00)
        ddd['PF6'].update(PF1=1.599E-02, PF2=2.927E-02, PF3=1.064E-01,
                          PF4=1.891E-01, PF5=4.910E-01, PF6=2.394E+00)
        ddd['CS3L'].update(CS3L=7.954E-01)
        ddd['CS2L'].update(CS3L=2.471E-01, CS2L=7.954E-01)
        ddd['CS1L'].update(CS3L=5.396E-02, CS2L=2.471E-01, CS1L=7.954E-01)
        ddd['CS1U'].update(CS3L=1.854E-02, CS2L=5.396E-02, CS1L=2.471E-01,
                           CS1U=7.954E-01)
        ddd['CS2U'].update(CS3L=8.267E-03, CS2L=1.854E-02, CS1L=5.396E-02,
                           CS1U=2.471E-01, CS2U=7.954E-01)
        ddd['CS3U'].update(CS3L=4.344E-03, CS2L=8.267E-03, CS1L=1.854E-02,
                           CS1U=5.396E-02, CS2U=2.471E-01, CS3U=7.954E-01)
        ddd['PF1'].update(CS3L=5.156E-03, CS2L=8.348E-03, CS1L=1.460E-02,
                          CS1U=2.812E-02, CS2U=6.021E-02, CS3U=1.348E-01)
        ddd['PF2'].update(CS3L=8.519E-03, CS2L=1.235E-02, CS1L=1.813E-02,
                          CS1U=2.640E-02, CS2U=3.641E-02, CS3U=4.445E-02)
        ddd['PF3'].update(CS3L=2.733E-02, CS2L=3.473E-02, CS1L=4.241E-02,
                          CS1U=4.858E-02, CS2U=5.109E-02, CS3U=4.890E-02)
        ddd['PF4'].update(CS3L=4.248E-02, CS2L=4.637E-02, CS1L=4.612E-02,
                          CS1U=4.185E-02, CS2U=3.521E-02, CS3U=2.814E-02)
        ddd['PF5'].update(CS3L=8.188E-02, CS2L=6.640E-02, CS1L=4.807E-02,
                          CS1U=3.314E-02, CS2U=2.268E-02, CS3U=1.573E-02)
        ddd['PF6'].update(CS3L=2.485E-01, CS2L=1.212E-01, CS1L=5.936E-02,
                          CS1U=3.161E-02, CS2U=1.833E-02, CS3U=1.142E-02)
        if self.plasma:
            ddd['Plasma'].update(PF6=0.000550, PF5=0.000722, PF4=0.001075,
                                 PF3=0.001169, PF2=0.000479, PF1=0.000311,
                                 CS3L=0.000189, CS2L=0.000317, CS1L=0.000469,
                                 CS1U=0.000510, CS2U=0.000388, CS3U=0.000240,
                                 Plasma=0.000033)
        self.Mddd = self.build_matrix(ddd)

    def calculate_Minv(self, dCoil=None, plasma=True):
        self.cc.update_inductance()
        self.Mnova = self.cc.inductance['Mc']  # link

    def compare(self):
        if not hasattr(self, 'Mnova'):
            self.calculate_Minv()
        self.plot()

    def plot(self):
        ax = plt.subplots(2, 1, figsize=(8, 8), sharey=True)[1]
        X = range(len(self.Mnova.values.flatten()))
        nC = self.cc.coilset.coil.nC
        nC2 = int(nC / 2)
        idx = [0, nC2, nC]
        for j in range(2):
            index = slice(idx[j]*(nC+1), idx[j+1]*(nC+1))
            ax[j].bar(X[index], self.Mddd.values.flatten()[index],
                      width=0.95, label='DDD')
            # ax[j].bar(X[index], self.Mipp.values.flatten()[index],
            #           width=0.75, label='IPP')
            ax[j].bar(X[index], self.Mnova.values.flatten()[index],
                      width=0.5, label='Nova')
            ax[j].set_xticks([j*nC2*(nC+1) + i*(nC+1)
                              for i in range(idx[j+1]-idx[j])])
            ax[j].set_xticklabels(self.names[slice(idx[j], idx[j+1])])
            ax[j].set_ylabel('$M$ H')
        ax[0].legend()
        ax[1].set_xlabel('coil')
        plt.despine()

    def getM(self, method='ddd'):
        if method == 'ddd':
            M = self.Mddd
        elif method == 'ipp':
            M = self.Mipp
        elif method == 'Nova':
            if not hasattr(self, 'Mnova'):
                self.calculate_Minv()
            M = self.Mnova
        return M


class PFgeom(CoilSet):  # PF/CS coilset

    def __init__(self, VS=False, dCoil=0.25, source='PCR'):
        CoilSet.__init__(self, dCoil=dCoil, turn_fraction=0.665,
                         turn_section='skin', skin_fraction=0.75)
        self.update_coilframe_metadata(
                'coil', additional_columns=['m', 'material', 'R'])
        self.load(VS=VS, source=source)

    def load(self, VS=False, source='PCR'):
        # Ro: referance FDU resistance at 0C, m: FDU mass
        if source == 'PCR':  # update
            f = io.StringIO('''
                	    X, m	Z, m	DX, m	DZ, m	N,	R, ohm	m, Kg
                CS3U	1.6870	5.4640	0.7400	2.093	554	0.102	9.0e3
                CS2U	1.6870	3.2780	0.7400	2.093 	554	0.113	10.0e3
                CS1U	1.6870	1.0920	0.7400	2.093 	554	0.124	11.0e3
                CS1L	1.6870	-1.0720	0.7400	2.093 	554	0.124	11.0e3
                CS2L	1.6870	-3.2580	0.7400	2.093 	554	0.113	10.0e3
                CS3L	1.6870	-5.4440	0.7400	2.093 	554	0.090	8.0e3
                PF1	3.9431	7.5641	0.9590	0.9841	248.64	0.0377	7.5e3
                PF2	8.2851	6.5298	0.5801	0.7146	115.20	0.0283	10.0e3
                PF3	11.9919	3.2652	0.6963	0.9538	185.92	0.0961	34.0e3
                PF4	11.9630	-2.2336	0.6382	0.9538	169.92	0.0791	28.0e3
                PF5	8.3908	-6.7369	0.8125	0.9538	216.80	0.0791	28.0e3
                PF6	4.3340	-7.4765	1.5590	1.1075	459.36	0.120	24.0e3
                ''')
        elif source == 'baseline':  # old
            f = io.StringIO('''
                	    X, m	Z, m	DX, m	DZ, m	N,	R, ohm	m, Kg
                CS3U	1.722	5.313	0.719	2.075	554	0.102	9.0e3
                CS2U	1.722	3.188	0.719	2.075	554	0.113	10.0e3
                CS1U	1.722	1.063	0.719	2.075	554	0.124	11.0e3
                CS1L	1.722	-1.063	0.719	2.075	554	0.124	11.0e3
                CS2L	1.722	-3.188	0.719	2.075	554	0.113	10.0e3
                CS3L	1.722	-5.313	0.719	2.075	554	0.090	8.0e3
                PF1	3.9431	7.5641	0.9590	0.9841	248.64	0.0377	7.5e3
                PF2	8.2851	6.5298	0.5801	0.7146	115.20	0.0283	10.0e3
                PF3	11.9919	3.2652	0.6963	0.9538	185.92	0.0961	34.0e3
                PF4	11.9630	-2.2336	0.6382	0.9538	169.92	0.0791	28.0e3
                PF5	8.3908	-6.7369	0.8125	0.9538	216.80	0.0791	28.0e3
                PF6	4.3340	-7.4765	1.5590	1.1075	459.36	0.120	24.0e3
                ''')
        data = pd.read_csv(f, delimiter='\t', skiprows=1, index_col=0,
                           skipinitialspace=True)
        columns = {}
        for c in list(data):
            columns[c] = c.split(',')[0].lower()
        columns['R, ohm'] = 'R'
        columns['N,'] = 'Nt'
        data = data.rename(columns=columns)
        part = ['CS' if 'CS' in name else 'PF' for name in data.index]
        data.rename(columns={'dx': 'dl', 'dz': 'dt'}, inplace=True)        
        coil = self.coil.get_coil(data, material='steel',
                                  cross_section='rectangle', part=part)
        #coil = self.cc.categorize_coilset(coil, rename=True)
        self.coil.concatenate(coil)
        self.meshcoil(index=coil.index)
        self.add_mpc(['CS1L', 'CS1U'], 1)  # link CS1 modules
        if VS:  # add vs3 system after primary coils
            self.append_coilset(VSgeom(jacket=True).cs.coilset)


class VSgeom(CoilSet):  # VS coil class

    def __init__(self, invessel=True, jacket=True):
        CoilSet.__init__(self)
        self.add_conductor(invessel=invessel)
        
        reindex = {'LVS0-LVS3': 'LVS', 'UVS0-UVS3': 'UVS'}
        if jacket:
            self.add_jacket()  # add steel jacket
            reindex = {**reindex, 'LVSj0-LVSj3': 'LVSj', 'UVSj0-UVSj3': 'UVSj'}
        # cluster upper and lower coils and jackets
        self.cluster(4, merge_pairs=False)  
        self.rename(index=reindex)
        self.coil.add_mpc(['LVS', 'UVS'], -1)  # link vs turns
        self.extract()

    def add_conductor(self, invessel=True):  # VS coil geometory
        co = 0.1065  # inner
        c1 = 0.14451  # outer
        rcs = np.array([co, c1]) / (2 * np.pi)
        acs_turn = np.pi * (rcs[1]**2 - rcs[0]**2)  # single turn cross-section
        d = 2 * rcs[1]  # turn diameter
        dt = 0.3  # skin_fraction
        dx_wp = 0.068  # winding pack width
        dz_wp = 0.064  # widning pack height
        self.geom = {}
        self.geom['LVS'] = {'x': 7.504, 'z': -2.495, 'dx': dx_wp, 'dz': dz_wp,
                            'theta': -37.8*np.pi/180, 'sign': 1,
                            'Nt': 4}
        self.geom['UVS'] = {'x': 5.81, 'z': 4.904, 'dx': dx_wp, 'dz': dz_wp,
                            'theta': 25.9*np.pi/180 + np.pi, 'sign': -1,
                            'Nt': -4}
        if not invessel:  # offest coils
            self.geom['UVS']['x'] += 1.7
            self.geom['UVS']['z'] -= 0
            self.geom['LVS']['x'] = self.geom['UVS']['x']
            self.geom['LVS']['z'] += -2.4

        xp = np.zeros((4, 2))  # coil pattern
        for i, (ix, iz) in enumerate(zip([1, 1, -1, -1], [-1, 1, 1, -1])):
            xp[i, 0] = ix*dx_wp/2
            xp[i, 1] = iz*dz_wp/2
        self.xc = {}  # coil centers
        for name in self.geom:  # add subcoils
            xc = np.dot(xp, rotate2D(self.geom[name]['theta'])[0])
            xc[:, 0] += self.geom[name]['x']
            xc[:, 1] += self.geom[name]['z']
            self.xc[name] = xc
            for i, x in enumerate(xc):
                subname = f'{name}{i}'
                self.add_coil(
                    x[0], x[1], d, dt, dCoil=0, name=subname,
                    cross_section='skin',turn_section='skin', 
                    skin_fraction=dt, material='copper',part='VS3',
                    Nt=1)
                R = resistivity_cu * 2 * np.pi * x[0] / acs_turn
                m = density_cu * 2 * np.pi * x[0] * acs_turn
                self.coil.at[subname, 'R'] = R
                self.coil.at[subname, 'm'] = m

    def add_jacket(self, rcs=[0.0265, 0.0295]):
        acs_turn = np.pi * (rcs[1]**2 - rcs[0]**2)  # single turn cross-section
        d = 2*rcs[1]
        dt = 0.15  # turn_fraction
        Nf = 4
        for name in self.geom:
            for isub in range(Nf):
                subname = name+'{}'.format(isub)
                x_sub = self.subcoil.at[subname, 'x']
                z_sub = self.subcoil.at[subname, 'z']
                R = resistivity_ss * 2 * np.pi * x_sub / acs_turn
                m = density_ss * 2 * np.pi * x_sub * acs_turn
                jacket_name = f'{name}j{isub}'
                self.add_coil(x_sub, z_sub, d, dt, R=R, name=jacket_name,
                              cross_section='skin', turn_section='skin',
                              skin_fraction=dt,
                              m=m, material='steel', dCoil=0, part='VS3j',
                              power=False)

    def plot_centers(self, coil='LVS', ax=None):
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        ax.plot(self.xc[coil][:, 0], self.xc[coil][:, 1], '.C5')

    def extract(self):
        self.nP = len(self.geom)
        self.points = np.zeros((self.nP, 2))  # xz coordinates
        self.theta = np.zeros(self.nP)
        self.theta_coil = {}
        for i, coil in enumerate(self.geom):  # order dw, up
            self.points[i, 0] = self.geom[coil]['x']
            self.points[i, 1] = self.geom[coil]['z']
            self.theta[i] = self.geom[coil]['theta']
            self.theta_coil[coil] = self.geom[coil]['theta']


class VVcoils(CoilSet):

    # path = os.path.join(class_dir(nep), '../Data/geom/')
    # data_mine(path, 'upper_vv_extended', [5.0, 7.5], [4.4, 6.0])
    # data_mine(path, 'lower_triangular_support_extended',
    #           [6.5, 9.5], [-4.0, -1.75])

    def __init__(self, model='local', invessel=True, read_txt=False):
        CoilSet.__init__(self, dCoil=0)  # python read/write
        self.model = model
        self.invessel = invessel
        self.read_txt = read_txt
        self.data_path = os.path.join(class_dir(nep_data.geom) + '/')
        self.load_coils()

    def load_coils(self, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        invessel = kwargs.get('invessel', self.invessel)
        filename = f'vv_coils_{self.model}'
        if invessel:
            filename += '_invessel'
        if read_txt or not os.path.isfile(filename + '.pk'):
            self.read_coils()
            self.save_coilset(filename)
            #self.save_pickle(filename, ['centerlines', '_coilset'])
        else:
            self.load_coilset(filename)
            #self.load_pickle(filename)
            #self.cc.append_coilset(self._coilset)

    def read_coils(self):
        self.dt = 60e-3  # vv thickness
        self.load_centerlines(model=self.model)
        self.add_vv_coils()

    def load_centerlines(self, model='local', plot=False, ax=None):
        # model = 'local', 'full'
        self.centerlines = {}
        if model == 'local':
            for location, filename, date in \
                    zip(['lower', 'upper'],
                        ['lower_triangular_support_extended',
                         'upper_vv_extended'],
                        ['2018_07_17', '2018_07_17']):
                centerlines = data_load(self.data_path, filename, date=date)[0]
                self.centerlines[location] = {}
                for cl, part in zip(centerlines, ['vvin', 'vvout', 'trs']):
                    self.centerlines[location][part] = cl
        elif model == 'full':
            vv_shell = data_load(self.data_path, 'vv_full', date='2018_08_22')[0]
            self.centerlines['vv'] = {}
            for cl, part in zip(vv_shell[1:], ['vvin', 'vvout']):
                self.centerlines['vv'][part] = cl
                self.close_centerline(self.centerlines['vv'][part])
            trs = data_load(self.data_path, 'lower_triangular_support_extended',
                            date='2018_07_17')[0]
            self.centerlines['trs'] = {}
            for cl, part in zip(trs[-1:], ['trs']):
                self.centerlines['trs'][part] = cl
        if plot:
            for location in self.centerlines:
                self.plot_centerlines(location, ax=ax)

    @staticmethod
    def close_centerline(centerline):
        for x in ['x', 'y']:
            centerline[x] = centerline[x][:-1]
            centerline[x] = np.append(centerline[x], centerline[x][0])

    def plot_centerlines(self, location, ax=None, **kwargs):
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        for part in self.centerlines[location]:
            cl = self.centerlines[location][part]
            ax.plot(cl['x'], cl['y'], **kwargs)
        plt.despine()
        ax.axis('equal')

    def plot(self, ax=None, plot_centerlines=False, plot_coils=True):
        if ax is None:
            ax = plt.gca()
        for i, location in enumerate(self.centerlines):
            if plot_centerlines:
                self.plot_centerlines(location, ax=ax, zorder=10, color='C0')
        if plot_coils:
            self.plot(subcoil=True, ax=ax)

    def subplot(self, plot_centerlines=False):
        ax = plt.subplots(1, 2, figsize=(9, 4))[1]
        xlim, ylim = [], []
        for i, location in enumerate(self.centerlines):
            self.plot_centerlines(location, ax=ax[i], zorder=10, color='C1')
            xlim.append(ax[i].get_xlim())
            ylim.append(ax[i].get_ylim())
            if not plot_centerlines:
                ax[i].cla()
            ax[i].set_xlim(xlim[i])
            ax[i].set_ylim(ylim[i])
        ax[0].set_title('lower VS coil')
        ax[1].set_title('upper VS coil')
        return ax, xlim, ylim

    def add_vv_coils(self):
        for location in self.centerlines:
            for part in self.centerlines[location]:
                cl = self.centerlines[location][part]
                L = amigo.geom.length(cl['x'], cl['y'], norm=False)[-1]
                xfun, zfun = amigo.geom.xzfun(cl['x'], cl['y'])
                nc = int(np.ceil(L / self.dt))
                dx = L / nc  # coil width
                dz = self.dt  # coil height
                x, z = np.zeros(nc), np.zeros(nc)
                R, m = np.zeros(nc), np.zeros(nc)
                name = [{} for __ in range(nc)]
                if not self.coil.empty:
                    io = self.coil.index[self.coil.part == part].size
                else:
                    io = 0
                for i in range(nc):
                    lnorm = dx*(i + 0.5) / L  # normalised length
                    x[i] = float(xfun(lnorm))
                    z[i] = float(zfun(lnorm))
                    R[i] = resistivity_ss * 2 * np.pi * x[i] / (dx * dz)
                    m[i] = density_ss * 2 * np.pi * x[i] * dx * dz
                    name[i] = f'{part}{i+io}'
                self.add_coil(x, z, dx, dz, R=R, name=name, part=part,
                                 cross_section='square', 
                                 turn_section='square',
                                 m=m, material='steel', power=False)
        #self.cluster(20) 

class elm_coils:

    def __init__(self):
        self.initalize_coil()
        self.initalize_geometry()
        self.load()

    #def initalize_coil(self):
    #    self.pf = PF()  # primary coil object

    def initalize_geometry(self):
        co = 0.1065  # inner
        c1 = 0.14451  # outer
        rcs = np.array([co, c1]) / (2 * np.pi)
        self.d = 2 * rcs[1]  # turn diameter
        dx = 0.0689  # winding pack width
        dz = 0.122  # widning pack height
        self.pattern = np.zeros((6, 2))  # coil pattern
        for i, (ix, iz) in enumerate(zip([1, 1, 1, -1, -1, -1],
                                         [-1, 0, 1, 1, 0, -1])):
            self.pattern[i, 0] = ix*dx/2
            self.pattern[i, 1] = iz*dz/2
        self.geom = {}
        self.xc = {}

    def add_coil(self, name, x, z, theta, tf):
        self.add_geom(name, x, z, theta)  # append geometry
        x, z, dx, dz = [self.geom[name][var] for var in ['x', 'z', 'dx', 'dz']]
        self.pf.add_coil(x, z, dx, dz, 0, name=name, Nt=6,
                         categorize=False, tf=tf)
        xc = np.dot(self.pattern, rotate2D(self.geom[name]['theta'])[0])
        self.xc[name] = xc.copy()
        self.xc[name][:, 0] += self.geom[name]['x']
        self.xc[name][:, 1] += self.geom[name]['z']
        self.pf.coilset['coil'][name]['Nf'] = 0
        for i, x in enumerate(self.xc[name]):  # add subcoils
            self.pf.coilset['coil'][name]['Nf'] += 1
            subcoil = {'x': x[0], 'z': x[1],
                       'dx': self.d, 'dz': self.d, 'It': 0}
            #self.pf.coilset['subcoil'][f'{name}_{i}'] = \
            #    PF.mesh_coil(subcoil, dCoil=None)[0]
            self.pf.coilset['subcoil'][f'{name}_{i}']['tf'] = tf

    def add_geom(self, name, x, z, theta):
        self.geom[name] = {'x': x, 'z': z, 'theta': theta,
                           'dx': self.d, 'dz': self.d}

    def load(self):
        self.add_coil('lower_elm_feed',
                      8.23013, -1.54604, -65.4*np.pi/180, 30/360)
        self.add_coil('lower_elm_return',
                      7.77119, -2.38146, -54.4*np.pi/180, 30/360)
        self.add_coil('middle_elm_feed',
                      8.618, 1.79037, -108.6*np.pi/180, 20/360)
        self.add_coil('middle_elm_return',
                      8.661, -0.549, -68.8*np.pi/180, 20/360)
        self.add_coil('upper_elm_feed',
                      7.73486, 3.38015, -130.3*np.pi/180, 28.5/360)
        self.add_coil('upper_elm_return',
                      8.26235, 2.62623, -119.6*np.pi/180, 28.5/360)
        self.pf.set_geometric_mean()

    def plot_poloidal(self, subcoil=False):
        names = np.unique([name.split('_')[0]
                          for name in self.pf.coilset['coil']])
        x, z = np.zeros(2), np.zeros(2)
        for name in names:
            feed_return = [f'{name}_elm_feed', f'{name}_elm_return']
            if subcoil:
                Nf = self.pf.coilset['coil'][feed_return[0]]['Nf']
                for i in range(Nf):
                    for j in range(2):
                        subcoil = f'{feed_return[j]}_{i}'
                        x[j] = self.pf.coilset['subcoil'][subcoil]['x']
                        z[j] = self.pf.coilset['subcoil'][subcoil]['z']
                    plt.plot(x, z, 'C0-')
            else:
                for j in range(2):
                    x[j] = self.pf.coilset['coil'][feed_return[j]]['x']
                    z[j] = self.pf.coilset['coil'][feed_return[j]]['z']
                plt.plot(x, z, '-', color='gray', zorder=-20)

    def plot(self):
        self.pf.plot(subcoil=True)
        self.plot_poloidal()


if __name__ == '__main__':

    # elm = elm_coils()
    # elm.plot()

    #IOdata = ITERdata(plasma=True, dCoil=-1, source='baseline', read_txt=False)
    #IOdata.compare()
    #IOdata.cc.plot(label=True, ax=plt.subplots(1, 1)[1])
    
    ITER = ITERcoilset(coils='pf', dCoil=0.2, n=2e3, 
                       limit=[4, 8.5, -3, 3], read_txt=False)
    """
    cc = ITER.cc
    cc.scenario_filename = -2
   
    #cs.add_coil(6, -3, 1.5, 1.5, name='PF16', part='PF', Nt=600, It=5e5,
    #            turn_section='circle', turn_fraction=0.7, dCoil=0.75)
    cc.scenario = 'IM'
    #cc.scenario = 'SOP'
    
    
    
    #cs.current_update = 'passive'
    #cc.Ic = {f'PF{i}': 0 for i in [2, 3, 4]}
    
    plt.set_aspect(1.2)
    cc.plot(label=['PF', 'CS'])
    
    #cc.grid.plot_flux()
    
    #cc.grid.plot_field()
    
    '''
    from nova.streamfunction import SF
    
    sf = SF(eqdsk={'x': cc.grid.x, 'z': cc.grid.z, 
                   'psi': cc.grid.Psi})
    sf.contour(ax=plt.gca(), separatrix=True, plot_points=True)
    '''
    
    #vs = VSgeom(jacket=True).cs
    #vs.plot()
    
    #vv = VVcoils(model='full', read_txt=True).cs
    #vv.plot()

    #pf = PFgeom(VS=True, dCoil=0.15).cs
    #pf.plot(label=['PF', 'CS'], current='Ic')

    '''
    vs = VSgeom(jacket=True).cc
    pf = PFgeom(VS=False).cc
    vv = VVcoils(model='full', invessel=False).cc
    cc = CoilClass(pf, vs, vv)
    cc.plot()
    '''
    """
