from nova.coil_object import CoilObject
from nova.coil_set import CoilSet, CoilFrame
from amigo.pyplot import plt
import numpy as np
import pandas as pd
from nova.biot_savart import biot_savart, self_inductance
from nova.mesh_grid import MeshGrid
from nep.DINA.read_scenario import scenario_data
from astropy import units


class CoilClass(CoilSet):
    '''
    CoilClass:
        - implements methods to manage input and
            output of data to/from the CoilSet class
        - provides interface to eqdsk files containing coil data
        - provides interface to DINA scenaria data
    '''
    def __init__(self, *args, eqdsk=None, scenario_filename=None, **kwargs):
        CoilSet.__init__(self, *args, **kwargs)  # inherent from CoilSet
        self.add_eqdsk(eqdsk)
        self.initalise_data()
        self.initalize_functions()  # initalise functions
        self._scenario_filename = None
        self.scenario_filename = scenario_filename

        self._levels = None
        self._limit = None

    @property
    def coilset(self):
        return CoilObject(self.coil, self.subcoil, inductance=self.inductance,
                          force=self.force, subforce=self.subforce,
                          grid=self.grid)

    @coilset.setter
    def coilset(self, coilset):
        for attr in ['coil', 'subcoil', 'inductance',
                     'force', 'subforce', 'grid']:
            setattr(self, attr, getattr(coilset, attr))

    def initalise_data(self):
        self.inductance = self.initalize_inductance()
        self.force = self.initialize_force()
        self.subforce = self.initialize_force()
        self.grid = self.initialize_grid()

    def initalize_functions(self):
        self.d2 = scenario_data()

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
        self.update_plasma()  # update plasma based on d2 data
        self.Ic = self.d2.Ic  # update coil currents

    def add_eqdsk(self, eqdsk):
        if eqdsk:
            frame = self.coil.get_frame(eqdsk['xc'], eqdsk['zc'],
                                        eqdsk['dxc'], eqdsk['dzc'],
                                        It=eqdsk['It'], name='eqdsk',
                                        delim='')
            frame = self.categorize_coilset(frame)
            self.coil.concatenate(frame)
            self.add_subcoil(index=frame.index)

    def self_inductance(self, name, update=False):
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
        #coordinates = ['Rp', 'Zp']
        if not np.array([c in self.d2.unit for c in coordinates]).all():
            coordinates = ['Rp', 'Zp']
        v2 = self.d2.vector.loc[['Lp', 'kp', 'ap'] + coordinates].droplevel(1)
        if 'Lp' not in self.d2.unit:
            v2['Lp'] = 1.1e-5  # default plasma self-inductance H
        scale = units.Unit(self.d2.unit[coordinates[0]]).to('m')
        if not np.isclose(scale, 1):
            for c in coordinates:
                v2.loc[c] *= scale  # convert coordinates
        Xp, Zp, Lp = v2.loc[coordinates + ['Lp']]
        dr = self_inductance(Xp).minor_radius(Lp)
        #dx, dz = 2*dr, 2*dr
        dx = 1.518*v2.ap
        dz = v2.kp * dx
        if 'Plasma' not in self.coil.index:  # create plasma coilset
            self.add_plasma(Xp, Zp, dx, dz, dCoil=0.25)
        else:  # update plasma coilset
            # TODO update multi-filament plasma model
            # subindex = self.coil.at['Plasma', 'subindex']
            self.add_plasma(Xp, Zp, 2*dr, 2*dr)
        self.calculate_inductance(source_index=['Plasma'])
        self.Ip = self.d2.Ip  # update plasma current

    def calculate_inductance(self, mutual=True,
                             source_index=None, invert_source=False,
                             target_index=None, invert_target=False):
        '''
        calculate / update inductance matrix

            Attributes:
                mutual (bool): include gmr correction for adjacent turns
                coil_index (list): update inductance for coil subest
                invert (bool): invert coil_index selection
        '''
        if self.inductance['Mc'].empty:
            source_index = None
            target_index = None
        if source_index is not None:
            source = self.subset(source_index, invert=invert_source)
        else:
            source = self.coilset
        if target_index is not None:
            target = self.subset(target_index, invert=invert_target)
        else:
            target = self.coilset
        bs = biot_savart(source=source, target=target, mutual=mutual)
        Mc = bs.calculate_inductance()
        if source_index is None and target_index is None:
            self.inductance['Mc'] = Mc  # line-current
        else:
            index = np.append(source.coil.index, target.coil.index)
            expand = [name for name in np.unique(index)
                      if name not in self.inductance['Mc'].index]
            for name in expand:
                self.inductance['Mc'].loc[:, name] = None
                self.inductance['Mc'].loc[name, :] = None
            self.inductance['Mc'].loc[target.coil.index,
                                      source.coil.index] = Mc
            self.inductance['Mc'].loc[source.coil.index,
                                      target.coil.index] = Mc.T
        Nt = self.coilset.coil['Nt'].values
        Nt = Nt.reshape(-1, 1) * Nt.reshape(1, -1)
        self.inductance['Mt'] = self.inductance['Mc'] / Nt  # amp-turn

    def update_grid(self, n=1e4, limit=None, expand=0.05):
        self.grid = CoilSet.initialize_grid()
        if limit is None:
            if self._limit is None:
                x, z = self.subcoil.loc[:, ['x', 'z']].to_numpy().T
                dx, dz = self.subcoil.loc[:, ['dx', 'dz']].to_numpy().T
                limit = np.array([(x - dx/2).min(), (x + dx/2).max(),
                                  (z - dz/2).min(), (z + dz/2).max()])
                dx, dz = np.diff(limit[:2])[0], np.diff(limit[2:])[0]
                delta = np.mean([dx, dz])
                limit += expand * delta * np.array([-1, 1, -1, 1])
                self._limit = limit
            else:
                limit = self._limit
        mg = MeshGrid(n, limit)  # set mesh
        self.grid['n'] = [mg.nx, mg.nz]
        self.grid['dx'] = np.diff(limit[:2])[0] / (mg.nx - 1)
        self.grid['dz'] = np.diff(limit[2:])[0] / (mg.nz - 1)
        self.grid['limit'] = limit
        self.grid['x2d'] = mg.x2d
        self.grid['z2d'] = mg.z2d
        bs = biot_savart(source=self.coilset, mutual=False)
        Psi = bs.calculate_interaction(grid=self.grid)
        self.grid['Psi'] = Psi
        # self.grid['Bx'] = Bx
        # self.grid['Bz'] = Bz

    def solve_grid(self, n=1e4, limit=None, nlevels=31,
                   plot=False, update=False, expand=0.05, color='k'):
        if self.grid['Psi'] is None or update:
            self.update_grid(n=n, limit=limit, expand=expand)
        for var in ['Psi']:  # 'Bx', 'Bz'
            value = np.dot(self.grid[var], self.Ic).reshape(self.grid['n'])
            self.grid[var.lower()] = value

        '''
        psi_x, psi_z = np.gradient(self.grid['psi'],
                                   self.grid['dx'], self.grid['dz'])
        bx = -psi_z / self.grid['x2d']
        bz = psi_x / self.grid['x2d']
        '''

        if plot:
            if self._levels is None:
                levels = nlevels
            else:
                levels = self._levels
            QuadContourSet = plt.contour(
                    self.grid['x2d'], self.grid['z2d'], self.grid['psi'],
                    levels, colors=color, linestyles='-', linewidths=1.0,
                    alpha=0.5, zorder=50)
            if self._levels is None:
                self._levels = QuadContourSet.levels
            plt.axis('equal')
            '''
            scale = 20
            plt.quiver(self.grid['x2d'], self.grid['z2d'],
                       self.grid['bx'], self.grid['bz'], scale=scale,
                       color='C0')
            plt.quiver(self.grid['x2d'], self.grid['z2d'],
                       bx, bz, scale=scale, color='C3')
            '''


if __name__ is '__main__':

    cc = CoilClass(dCoil=0.15)
    # cc.update_metadata('coil', additional_columns=['R'])

    x, z, dx = 5.5, -5, 4.2
    dz = 2*dx
    cc.add_coil(x, z, dx, dz, name='PF6', part='PF', Ic=50e3,
                cross_section='ellipse',
                turn_section='square', turn_fraction=1)

    #plt.plot(*cc.coil.at['PF6', 'polygon'].exterior.xy, 'C3')

    '''
    cc.add_coil([2, 2, 3, 3.5], [1, 0, -1, -3], 0.3, 0.3,
                name='PF', part='PF', delim='', Nt=30)
    cc.add_coil(3, 2, 0.5, 0.8, name='PF4', part='VS3', turn_fraction=0.75,
                Nt=15, dCoil=-1)
    cc.add_coil(5.6, 3.5, 0.5, 0.8, name='PF7', part='vvin', dCoil=0.01)
    '''
    # cc.add_plasma(1, [1.5, 2, 2.5], 0.5, 0.2, It=-15e6/3)
    # cc.add_plasma(6, [1.5, 2, 2.5], 0.5, 0.2, It=-15e6/3)

    #cc.plot(label=True)
    #cc.calculate_inductance()

    # cc.scenario_filename = -2
    # cc.scenario = 'EOF'
    # cc.calculate_inductance(source_index=['Plasma'])

    #cc.solve_grid(n=2e3, plot=True, update=True, expand=0.25,
    #              nlevels=31, color='k')
    cc.plot(subcoil=False)
    cc.plot(label=True)



    '''
    cc.drop_coil('PF6')


    cc.add_coil(x, -5, dx, dx, name='PF7', part='CS', Ic=50e3, dCoil=1.5)
    cc.plot(label=True)
    '''
    cc.solve_grid(n=1e3, plot=True, update=True, expand=0.15,
                  nlevels=61, color='C3')



