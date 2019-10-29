from nova.coil_set import CoilSet
from nep.DINA.read_scenario import scenario_data
from nova.biot_savart import biot_savart, self_inductance
from nova.mesh_grid import MeshGrid
from amigo.pyplot import plt
import numpy as np
import pandas as pd
import amigo.geom


class CoilClass(CoilSet):
    '''
    CoilClass:
        - implements methods to manage input and
            output of data to/from the CoilSet class
        - provides inductance caluculation methods (biot_savart)
        - provides interface to eqdsk files containing coil data
        - provides interface to DINA scenaria data
    '''
    def __init__(self, *args, eqdsk=None, filename=None, **kwargs):
        CoilSet.__init__(self, *args, **kwargs)  # inherent from CoilSet
        self.add_eqdsk(eqdsk)
        self.initalize_functions()
        self.initalize_metadata()
        self.filename = filename

    def initalize_functions(self):
        self.t = None  # scenario time instance (d2.to)
        self.d2 = scenario_data()

    def initalize_metadata(self):
        self._scenario_filename = ''
        self._plasma_metadata = pd.Series(
                index=['filename', 'to', 'ko', 't', 'Ip', 'Rp', 'Zp', 'Lp',
                       'Rcur', 'Zcur',
                       'x', 'z', 'dx', 'dz', 'cross_section', 'turn_section'])

    @property
    def filename(self):
        return pd.Series({'scenario': self.scenario_filename,
                          'plasma': self.plasma_filename})

    @filename.setter
    def filename(self, filename):
        self.scenario_filename = filename
        self.plasma_filename = filename

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
        if filename != self._scenario_filename and filename is not None:
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
        self.t = self.d2.to  # time instance
        #self.update_plasma()
        #Ic = self.d2.Ic.reindex(self.Ic.index)
        self.frame.data.Ic = self.d2.Ic.to_dict()
        #self.frame.Ic = self.d2.Ic.to_dict()

    @property
    def plasma(self):
        return self.plasma_metadata

    @plasma.setter
    def plasma(self, metadata):
        '''
        Attributes:
            metadata (None or dict): None: self.d2, kwargs: overide
        Metadata kwargs:
            filename (str | int): DINA filename | DINA fileindex
            to (float | str): input time | feature_keypoint
            x (float): current center
            z (float): current center
            dx (float): bounding box
            dz (float): bounding box
            Ip (float): current
            Lp (float): inductance
            cross_section (str): cross-section [circle, elipse, square, skin]
            turn_section (str): turn-section [circle, elipse, square, skin]
        '''
        if pd.isnull(metadata):  # release all plasma parameters
            self._plasma_metadata.loc[:] = np.nan
        else:
            self._plasma_metadata.update(pd.Series(metadata))  # fix parameters
        scenario = self.scenario  # extract scenario
        scenario.update(self._plasma_metadata)  # update
        if pd.isnull(self._plasma_metadata[['to', 'ko']]).all():
            for key in ['ko', 'to']:  # unset
                scenario[key] = np.nan
        else:
            if pd.notnull(self._plasma_metadata['to']):
                self.d2.to = scenario['to']  # time | keypoint
            elif pd.notnull(self._plasma_metadata['ko']):
                self.d2.ko = scenario['ko']  # keypoint
            for key in ['ko', 'to']:  # re-set
                scenario[key] = getattr(self.d2, key)
        for key in scenario.index:  # propogate changes
            self._plasma_metadata[key] = scenario[key]
        if pd.notnull(self._plasma_metadata['to']):
            plasma_metadata = self.extract_plasma_metadata()
            plasma_metadata.update(self._plasma_metadata)  # overwrite
            self._plasma_metadata.update(plasma_metadata)  # update
            for key in metadata:
                if pd.isnull(metadata[key]):
                    self._plasma_metadata[key] = np.nan  # release specified

    def extract_plasma_metadata(self, cross_section='ellipse',
                                turn_section='square', Lp=1.1e-5):
        '''
        extract plasma metadata from self.d2 instance
        '''
        d2 = {'t': 1.2, 'Ip': 1.2, 'Rp': 1.2, 'Lp': 1.2, 'Rcur': 1.2, 'Zcur': 1.2}
        plasma_metadata = {}
        for key in ['t', 'Ip', 'Rp', 'Lp', 'Rcur', 'Zcur']:
            #plasma_metadata[key] = self.d2.vector.at[key]
            plasma_metadata[key] = d2[key]

        #self.d2.vector.reindex(
        #        ['t', 'Ip', 'Rp', 'Zp', 'Lp', 'Rcur', 'Zcur'])
        #xp, zp, Lp = plasma_metadata.loc[['Rcur', 'Zcur', 'Lp']]
        #if not pd.isnull(plasma_metadata.loc[['Rcur'])
        '''
        coordinates = ['Rcur', 'Zcur']
        if not np.array([c in self.d2.index for c in coordinates]).all():
            coordinates = ['Rp', 'Zp']
        v2 = self.d2.vector.reindex(coordinates + ['Lp', 'kp', 'ap'])
        if pd.notnull(self._plasma_metadata['Lp']):  # fixed self-inductance
            v2['Lp'] = self._plasma_metadata['Lp']
        elif 'Lp' not in self.d2.index:
            v2['Lp'] = Lp  # default plasma self-inductance H
        '''
        '''
        xp, zp, Lp = v2.loc[coordinates + ['Lp']]  # current center, inductance
        # xp, zp, Lp = v2.iloc[:3]
        dr = self_inductance(xp).minor_radius(Lp)
        dx, dz = 2*dr, 2*dr  # bounding box
        Ip = self.d2.Ip  # current
        plasma_metadata = pd.Series(
                {'Ip': Ip, 'Lp': Lp, 'x': xp, 'z': zp, 'dx': dx, 'dz': dz,
                 'cross_section': cross_section, 'turn_section': turn_section})
        '''
        #return plasma_metadata

    def update_plasma(self):
        self.plasma_metadata = self.extract_plasma_metadata()  # extract
        '''
        self.plasma_metadata.update(self._plasma_metadata)  # overwrite
        self.update_plasma_coil()  # update coil position
        self.update_plasma_current()
        '''

    def update_plasma_coil(self):
        pl = self.plasma_metadata.loc[['x', 'z', 'dx', 'dz']]
        if (pl != 0).all():  # plasma position valid
            if 'Plasma' in self.coil.index:
                update = (pl != self.coil.loc['Plasma', pl.index]).any()
            else:
                update = True
            if update:  # update plasma coils, inductance and interaction
                self.add_plasma(*pl)  # create / update plasma
                self.update_inductance(source_index=['Plasma'])
                self.update_interaction(coil_index=['Plasma'])
        elif 'Plasma' in self.coil.index:  # remove plasma
            self.drop_coil('Plasma')

    def update_plasma_current(self):
        if 'Plasma' in self.coil.index:
            self.Ip = self.plasma_metadata['Ip']  # update plasma current

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

    def update_inductance(self, mutual=True,
                          source_index=None, invert_source=False,
                          target_index=None, invert_target=False):
        '''
        calculate / update inductance matrix

            Attributes:
                mutual (bool): include gmr correction for adjacent turns
                coil_index (list): update inductance for coil subest
                invert_coil (bool): invert coil_index selection
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
        '''
        compare kwargs to current grid settings, update grid on-demand

        kwargs:
            n (int): grid node number
            limit (np.array): [xmin, xmax, zmin, zmax] grid limits
            expand (float): expansion beyond coil limits (when limit not set)
            nlevels (int)
            levels ()
        '''
        update = kwargs.get('update', False)
        grid = {}  # grid parameters
        for key in ['n', 'limit', 'expand', 'levels', 'nlevels']:
            grid[key] = kwargs.pop(key, self.grid[key])
        if grid['limit'] is None:  # update grid limit
            grid['limit'] = self._get_grid_limit(grid['expand'])
        update = not np.array_equal(grid['limit'], self.grid['limit']) or \
            grid['n'] != self.grid['n'] or update
        if update:
            self._generate_grid(**grid)
        return update

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

    def _generate_grid(self, **kwargs):
        self.grid = self.initialize_grid(**kwargs)
        if self.grid['n'] > 0:
            if self.grid['limit'] is None:
                self.grid['limit'] = self._get_grid_limit(self.grid['expand'])
            mg = MeshGrid(self.grid['n'], self.grid['limit'])  # set mesh
            self.grid['n2d'] = [mg.nx, mg.nz]
            self.grid['dx'] = np.diff(self.grid['limit'][:2])[0] / (mg.nx - 1)
            self.grid['dz'] = np.diff(self.grid['limit'][2:])[0] / (mg.nz - 1)
            self.grid['x2d'] = mg.x2d
            self.grid['z2d'] = mg.z2d
            self.grid['update'] = True

    def add_targets(self, targets=None, **kwargs):
        '''
        Kwargs:
            targets (): target coordinates
                (dict, pd.DataFrame() or list like):
                    x (np.array): x-coordinates
                    z (np.array): z-coordinates
                    update (np.array): update flag
            append (bool): create new list | append
            update (bool): update interaction matricies
            drop_duplicates (bool): drop duplicates
        '''
        update = kwargs.get('update', False)  # update all
        if targets is not None:
            append = kwargs.get('append', True)
            drop_duplicates = kwargs.get('drop_duplicates', True)
            if not isinstance(targets, pd.DataFrame):
                if pd.api.types.is_dict_like(targets):
                    targets = pd.DataFrame(targets)
                elif pd.api.types.is_list_like(targets):
                    x, z = targets
                    if not pd.api.types.is_list_like(x):
                        x = [x]
                    if not pd.api.types.is_list_like(z):
                        z = [z]
                    targets = pd.DataFrame({'x': x, 'z': z})
                if append:
                    io = self.target['points'].shape[0]
                else:
                    io = 0
                targets['index'] = [f'P{i+io}'
                                    for i in range(targets.shape[0])]
                targets.set_index('index', inplace=True)
                targets = targets.astype('float')
            if not targets.empty:
                if self.target['points'].equals(targets):  # duplicate set
                    self.target['points']['update'] = False
                else:
                    targets['update'] = True
                    if append:
                        self.target['points'] = pd.concat(
                                (self.target['points'], targets))
                        if drop_duplicates:
                            self.target['points'].drop_duplicates(
                                    subset=['x', 'z'], inplace=True)
                    else:  # overwrite
                        self.target['points'] = targets
            if update:  # update all
                self.target['points']['update'] = True
            else:
                update = self.target['points']['update'].any()
        else:
            if update:  # update all
                self.target['points']['update'] = True
        self.target['update'] = update
        return update

    def update_interaction(self, coil_index=None):
        if coil_index is not None:  # full update
            self.grid['update'] = True and self.grid['n'] > 0
            self.target['update'] = True
            self.target['points']['update'] = True
        update_targets = self.grid['update'] or self.target['update']
        if update_targets or coil_index is not None:
            if coil_index is None:
                coilset = self.coilset  # full coilset
            else:
                coilset = self.subset(coil_index)  # extract subset
            bs = biot_savart(source=coilset, mutual=False)  # load coilset
            if self.grid['update'] and self.grid['n'] > 0:
                bs.load_target(self.grid['x2d'].flatten(),
                               self.grid['z2d'].flatten(),
                               label='G', delim='', part='grid')
                self.grid['update'] = False  # reset update status
            if self.target['update']:
                update = self.target['points']['update']  # new points only
                points = self.target['points'].loc[update, :]  # subset
                bs.load_target(points['x'], points['z'], name=points.index,
                               part='target')
                self.target['points'].loc[update, 'update'] = False
            M = bs.calculate_interaction()
            for matrix in M:
                if self.interaction[matrix].empty:
                    self.interaction[matrix] = M[matrix]
                elif coil_index is None:
                    drop = self.interaction[matrix].index.unique(level=1)
                    for part in M[matrix].index.unique(level=1):
                        if part in drop:  # clear prior to concat
                            if part == 'target':
                                self.interaction[matrix].drop(
                                        points.index, level=0,
                                        inplace=True, errors='ignore')
                            else:
                                self.interaction[matrix].drop(
                                        part, level=1, inplace=True)
                    self.interaction[matrix] = pd.concat(
                            [self.interaction[matrix], M[matrix]])
                else:  # selective coil_index overwrite
                    for name in coilset.coil.index:
                        self.interaction[matrix].loc[:, name] = \
                            M[matrix].loc[:, name]

    def solve_interaction(self, plot=False, color='gray', *args, **kwargs):
        self.add_targets(**kwargs)  # add | append data targets
        self.generate_grid(**kwargs)  # re-generate grid on demand
        self.update_interaction()  # update on demand
        for matrix in self.interaction:
            if not self.interaction[matrix].empty:
                variable = matrix.lower()
                index = self.interaction[matrix].index
                value = np.dot(self.interaction[matrix].loc[:, self.Ic.index],
                               self.Ic)
                frame = pd.DataFrame(value, index=index)
                for part in frame.index.unique(level=1):
                    part_data = frame.xs(part, level=1)
                    part_dict = getattr(self, part)
                    if 'n2d' in part_dict:  # reshape data to n2d
                        part_data = part_data.to_numpy()
                        part_data = part_data.reshape(part_dict['n2d'])
                        part_dict[variable] = part_data
                    else:
                        part_data = pd.concat(
                                (pd.Series({'t': self.t}), part_data))
                        part_dict[variable] = pd.concat(
                                (part_dict[variable], part_data.T),
                                ignore_index=True)
        '''
        psi_x, psi_z = np.gradient(self.grid['psi'],
                                   self.grid['dx'], self.grid['dz'])
        bx = -psi_z / self.grid['x2d']
        bz = psi_x / self.grid['x2d']
        '''
        if plot and self.grid['n'] > 0:
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

    cc = CoilClass(dCoil=0.15, n=1e3, expand=0.25, nlevels=51)
    cc.update_metadata('frame', additional_columns=['R'])
    cc.scenario_filename = '15MA DT-DINA2016-01_v1.1'

    x, z, dx = 5.5, -5, 4.2
    dz = 2*dx
    cc.add_coil(x, z, dx, dz, name='PF1', part='PF', Ic=-15e6,
                cross_section='circle',
                turn_section='square', turn_fraction=1, Nt=1)

    plt.plot(*cc.frame.at['PF1', 'polygon'].exterior.xy, 'C3')
    cc.add_plasma(1, [1.5, 2, 2.5], 0.5, 0.2, It=-15e6/3)
    cc.plot()
    # cc.add_plasma(6, [1.5, 2, 2.5], 0.5, 0.2, It=-15e6/3)

    cc.scenario = 100

    '''

    # cc.generate_grid(n=0)
    cc.add_targets(([1.0, 2], [4, 5]))
    cc.update_interaction()

    cc.add_targets(([1, 2, 3], [4, 5, 3]), append=True)
    cc.update_interaction()

    cc.add_targets((1, 4), append=True, update=True)
    cc.add_targets(([1, 2, 3], [4, 5, 3.1]), append=True)


    cc.plot(label=True)
    #cc.solve_interaction(plot=True)
    '''

    '''
    #cc.plot(label=True)
    #cc.update_inductance()

    cc.scenario_filename = -2
    cc.scenario = 'EOF'
    # cc.update_inductance(source_index=['Plasma'])

    #cc.solve_grid(n=2e3, plot=True, update=True, expand=0.25,
    #              nlevels=31, color='k')
    cc.plot(subcoil=False)
    cc.plot(label=True)
    '''






