from nova.coil_set import CoilSet
from nep.DINA.read_scenario import scenario_data
from nova.biot_savart import biot_savart, self_inductance
from nova.mesh_grid import MeshGrid
from amigo.pyplot import plt
import numpy as np
import pandas as pd
from astropy import units
import amigo.geom


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
        self._scenario_filename = scenario_filename

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
        self.update_plasma_coil()  # update plasma location based on d2 data
        self.update_plasma_current()  # update plasma current
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
        Mc = biot_savart(coilset).calculate_inductance()  # self-inductance
        L = Mc.at[name, name]
        dr = self_inductance(coilset.coil.x[name]).minor_radius(L)
        # calculate geometric and arithmetic means
        Nt = coilset.subcoil.Nt
        x_gmd = amigo.geom.gmd(coilset.subcoil.x, Nt)
        z_amd = amigo.geom.amd(coilset.subcoil.z, Nt)
        if update:  # apply update
            coilset.coil.loc[name, ['x', 'z']] = x_gmd, z_amd
            coilset.coil.loc[name, ['dx', 'dz']] = 2*dr, 2*dr
            CoilSet.patch_coil(coilset.coil)  # re-generate coil patch
            self.coil.loc[name] = coilset.coil.loc[name]
        coilset = None  # remove coilset
        return L

    def update_plasma_coil(self):
        coordinates = ['Rcur', 'Zcur']
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
        dx, dz = 2*dr, 2*dr
        # dx = 1.518*v2.ap
        # dz = v2.kp * dx
        if (np.array([Xp, dx, dz]) != 0).all():
            if 'Plasma' not in self.coil.index:  # create plasma coilset
                self.add_plasma(Xp, Zp, dx, dz)
            else:  # update plasma coilset
                # TODO update multi-filament plasma model
                # subindex = self.coil.at['Plasma', 'subindex']
                self.add_plasma(Xp, Zp, 2*dr, 2*dr)
            self.calculate_inductance(source_index=['Plasma'])
            self.calculate_interaction(coil_index=['Plasma'])
        elif 'Plasma' in self.coil.index:
            self.drop_coil('Plasma')

    def update_plasma_current(self):
        if 'Plasma' in self.coil.index:
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

    def generate_grid(self, **kwargs):
        self.grid = self.initialize_grid(**kwargs)  # reset / update defaults
        if self.grid['limit'] is None:
            self.grid['limit'] = self._get_grid_limit(self.grid['expand'])
        mg = MeshGrid(self.grid['n'], self.grid['limit'])  # set mesh
        self.grid['n2d'] = [mg.nx, mg.nz]
        self.grid['dx'] = np.diff(self.grid['limit'][:2])[0] / (mg.nx - 1)
        self.grid['dz'] = np.diff(self.grid['limit'][2:])[0] / (mg.nz - 1)
        self.grid['x2d'] = mg.x2d
        self.grid['z2d'] = mg.z2d

    def _get_grid_limit(self, expand):
        if expand is None:
            expand = self.grid['expand']  # use coil_object default
        x, z = self.subcoil.loc[:, ['x', 'z']].to_numpy().T
        dx, dz = self.subcoil.loc[:, ['dx', 'dz']].to_numpy().T
        limit = np.array([(x - dx/2).min(), (x + dx/2).max(),
                          (z - dz/2).min(), (z + dz/2).max()])
        dx, dz = np.diff(limit[:2])[0], np.diff(limit[2:])[0]
        delta = np.mean([dx, dz])
        limit += expand * delta * np.array([-1, 1, -1, 1])
        return limit

    def _regenerate_grid(self, **kwargs):
        '''
        compare kwargs to current grid settings, update as required
        '''
        regen = False
        grid = {}  # grid parameters
        for key in ['n', 'limit', 'expand', 'levels', 'nlevels']:
            grid[key] = kwargs.pop(key, self.grid[key])
        if grid['limit'] is None:  # update grid limit
            grid['limit'] = self._get_grid_limit(grid['expand'])
        regen = not np.array_equal(grid['limit'], self.grid['limit']) or \
            grid['n'] != self.grid['n']
        if regen:  # update grid to match kwargs
            self.generate_grid(**grid)
        return regen

    def calculate_interaction(self, coil_index=None, **kwargs):
        '''
        kwargs:
            n (int): grid node number
            limit (np.array): [xmin, xmax, zmin, zmax] grid limits
            expand (float): expansion beyond coil limits (when limit not set)
            nlevels (int)
            levels ()
        '''
        kwargs = self._set_levels(**kwargs)  # update contour levels
        regen = self._regenerate_grid(**kwargs)  # regenerate grid on demand
        if regen or self.grid['Psi'].empty or coil_index is not None:
            if coil_index is None:
                coilset = self.coilset
            else:
                coilset = self.subset(coil_index)
            bs = biot_savart(source=coilset, mutual=False)
            Psi = bs.calculate_interaction(grid=self.grid)
            if self.grid['Psi'].empty:
                self.grid['Psi'] = Psi
            else:  # append
                for name in coilset.coil.index:
                    self.grid['Psi'].loc[:, name] = Psi.loc[:, name]
        return regen

    def _set_levels(self, **kwargs):
        '''
        kwargs:
            nlevels (int): number of contour levels
            levels: contour levels
        '''
        self.grid['nlevels'] = kwargs.pop('nlevels', self.grid['nlevels'])
        self.grid['levels'] = kwargs.pop('levels', self.grid['levels'])
        return kwargs

    def solve_grid(self, plot=False, color='gray', **kwargs):
        self.calculate_interaction(**kwargs)
        for var in ['Psi']:  # 'Bx', 'Bz'
            value = np.dot(self.grid[var], self.Ic).reshape(self.grid['n2d'])
            self.grid[var.lower()] = value
        '''
        psi_x, psi_z = np.gradient(self.grid['psi'],
                                   self.grid['dx'], self.grid['dz'])
        bx = -psi_z / self.grid['x2d']
        bz = psi_x / self.grid['x2d']
        '''
        if plot:
            if self.grid['levels'] is None:
                levels = self.grid['nlevels']
            else:
                levels = self.grid['levels']
            QuadContourSet = plt.contour(
                    self.grid['x2d'], self.grid['z2d'], self.grid['psi'],
                    levels, colors=color, linestyles='-', linewidths=1.0,
                    alpha=0.5, zorder=5)
            self.grid['levels'] = QuadContourSet.levels
            plt.axis('equal')


if __name__ == '__main__':

    cc = CoilClass(dCoil=0.15)
    cc.update_metadata('coil', additional_columns=['R'])
    cc.scenario_filename = '15MA DT-DINA2016-01_v1.1'

    x, z, dx = 5.5, -5, 4.2
    dz = 2*dx
    cc.add_coil(x, z, dx, dz, name='Plasma', part='plasma', Ic=-15e6,
                cross_section='circle',
                turn_section='square', turn_fraction=1, Nt=1)

    # plt.plot(*cc.coil.at['Plasma', 'polygon'].exterior.xy, 'C3')
    # cc.add_plasma(1, [1.5, 2, 2.5], 0.5, 0.2, It=-15e6/3)
    # cc.add_plasma(6, [1.5, 2, 2.5], 0.5, 0.2, It=-15e6/3)

    cc.solve_grid(n=2e3, plot=True, expand=0.25,
                  nlevels=51)
    cc.plot()

    '''
    #cc.plot(label=True)
    #cc.calculate_inductance()

    cc.scenario_filename = -2
    cc.scenario = 'EOF'
    # cc.calculate_inductance(source_index=['Plasma'])

    #cc.solve_grid(n=2e3, plot=True, update=True, expand=0.25,
    #              nlevels=31, color='k')
    cc.plot(subcoil=False)
    cc.plot(label=True)
    '''






