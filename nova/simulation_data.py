import numpy as np
from pandas import Series, DataFrame, concat
from pandas.api.types import is_list_like, is_dict_like
from amigo.pyplot import plt
from nova.mesh_grid import MeshGrid
#from nova.biot_savart import biot_savart


class SimulationData:
    '''
    container for sumulation data
    
        target (dict): poloidal target coordinates and data
            target['targets'] (DataFrame):  target xz-coordinates
            target['Psi'] (DataFrame): poloidal flux
            target['Bx'] (DataFrame): radial field
            target['Bz'] (DataFrame): vertical field
            target['update'] (bool): update flag

        grid (dict): poloidal grid coordinates
            grid['n'] ([2int]): grid dimensions
            grid['limit'] ([4float]): grid limits
            grid['x2d'] (np.array): x-coordinates (radial)
            grid['z2d'] (np.array): z-coordinates
            grid['Psi'] (np.array): poloidal flux
            grid['Bx'] (np.array): radial field
            grid['Bz'] (np.array): vertical field
            grid['update'] (bool): update flag
            
        interaction (dict): coil grid / target interaction matrices (DataFrame)
            interaction['Psi']: poloidal flux interaction matrix
            interaction['Bx']: radial field interaction matrix
            interaction['Bz']: vertical field interaction matrix
    '''
    
    # main class attributes
    _simulation_attributes = ['target', 'grid', 'interaction']

    def __init__(self, target=None, grid=None, interaction=None, **kwargs):
        self._attributes += self._simulation_attributes
        self.target = self._initialize_target(target)
        self.grid = self._initialize_grid(grid, **kwargs)
        self.interaction = self._initialize_interaction(interaction)
        
    @staticmethod        
    def _initialize_interaction(interaction=None):
        if interaction is None:
            interaction = {'Psi': DataFrame(),
                           'Bx': DataFrame(), 
                           'Bz': DataFrame()}
        return interaction
        
    @staticmethod
    def _initialize_grid(grid=None, **kwargs):
        if grid is None:  # initalize
            grid = {'n': 1e4,  # default grid dimensions
                    'n2d': None,  # ([int, int]) as meshed dimensions
                    'limit': None,  # (np.array) grid limits
                    'expand': 0.05,  # (float) grid expansion beyond coils
                    'nlevels': 31,  # (int) number of contour levels
                    'levels': None,  # contour levels
                    'x2d': None,  # (np.array) x-coordinates
                    'z2d': None,  # (np.array) z-coordinates
                    'Psi': None,  # (np.array) poloidal flux
                    'Bx': None,  # (np.array) radial field
                    'Bz': None,  # (np.array) vertical field
                    'update': False}
            for key in kwargs:
                grid[key] = kwargs[key]  # overwrite defaults
        return grid

    @staticmethod
    def _initialize_target(target=None):
        if target is None:  # initalize
            target = {'targets': DataFrame(columns=['x', 'z', 'update']),
                      'Psi': DataFrame(),  # poloidal flux
                      'Bx': DataFrame(),  # radial field
                      'Bz': DataFrame(),  # vertical field
                      'update': False}
        return target

    def clear_target_data(self):
        for variable in ['Psi', 'Bx', 'Bz']:
            self.target[variable] = DataFrame()
            
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
        self.grid = self._initialize_grid(**kwargs)
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
            
    def plot_grid(self, ax=None, **kwargs):
        self.generate_grid(**kwargs)
        if ax is None:
            ax = plt.gca()
        MeshGrid._plot(self.grid['x2d'], self.grid['z2d'], self.grid['limit'],
                       ax=ax, zorder=-500, **kwargs)  # plot grid  
        
    def add_targets(self, targets=None, **kwargs):
        '''
        Kwargs:
            targets (): target coordinates
                (dict, DataFrame() or list like):
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
            if not isinstance(targets, DataFrame):
                if is_dict_like(targets):
                    targets = DataFrame(targets)
                elif is_list_like(targets):
                    x, z = targets
                    if not is_list_like(x):
                        x = [x]
                    if not is_list_like(z):
                        z = [z]
                    targets = DataFrame({'x': x, 'z': z})
                if append:
                    io = self.target['targets'].shape[0]
                else:
                    io = 0
                targets['index'] = [f'P{i+io}'
                                    for i in range(targets.shape[0])]
                targets.set_index('index', inplace=True)
                targets = targets.astype('float')
            if not targets.empty:
                if self.target['targets'].equals(targets):  # duplicate set
                    self.target['targets']['update'] = False
                else:
                    targets['update'] = True
                    if append:
                        self.target['targets'] = concat(
                                (self.target['targets'], targets))
                        if drop_duplicates:
                            self.target['targets'].drop_duplicates(
                                    subset=['x', 'z'], inplace=True)
                    else:  # overwrite
                        self.target['targets'] = targets
            if update:  # update all
                self.target['targets']['update'] = True
            else:
                update = self.target['targets']['update'].any()
        else:
            if update:  # update all
                self.target['targets']['update'] = True
            else:
                update = self.target['update'] 
        self.target['update'] = update
        return update
    
    def update_interaction(self, coil_index=None, **kwargs):
        self.generate_grid(**kwargs)  # add | append data targets
        self.add_targets(**kwargs)  # re-generate grid on demand
        if coil_index is not None:  # full update
            self.grid['update'] = True and self.grid['n'] > 0
            self.target['update'] = True
            self.target['targets']['update'] = True
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
                update = self.target['targets']['update']  # new points only
                targets = self.target['targets'].loc[update, :]  # subset
                bs.load_target(targets['x'], targets['z'], name=targets.index,
                               part='target')
                self.target['targets'].loc[update, 'update'] = False
                self.target['update'] = False
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
                    self.interaction[matrix] = concat(
                            [self.interaction[matrix], M[matrix]])
                else:  # selective coil_index overwrite
                    for name in coilset.coil.index:
                        self.interaction[matrix].loc[:, name] = \
                            M[matrix].loc[:, name]
                            
    def solve_interaction(self, plot=False, color='gray', *args, **kwargs):
        'generate grid / target interaction matrices'
        self.update_interaction(**kwargs)  # update on demand
        for matrix in self.interaction:  # Psi, Bx, Bz
            if not self.interaction[matrix].empty:
                # variable = matrix.lower()
                #index = self.interaction[matrix].index
                #value = np.dot(
                #        self.interaction[matrix].loc[:, self.coil.data.index],
                #        self.coil.data.Ic)
                #value = self.interaction[matrix].dot(self.Ic)
                value = np.dot(self.interaction[matrix].to_numpy(), self.Ic)
                #coil = DataFrame(value, index=index)  # grid, target
                '''
                for part in coil.index.unique(level=1):
                    part_data = coil.xs(part, level=1)
                    part_dict = getattr(self, part)
                    if 'n2d' in part_dict:  # reshape data to n2d
                        part_data = part_data.to_numpy()
                        part_data = part_data.reshape(part_dict['n2d'])
                        part_dict[matrix] = part_data
                    else:
                        part_data = concat(
                                (Series({'t': self.t}), part_data),
                                sort=False)
                        part_dict[matrix] = concat(
                                (part_dict[matrix], part_data.T),
                                ignore_index=True, sort=False)
                '''
        if plot and self.grid['n'] > 0:
            if self.grid['levels'] is None:
                levels = self.grid['nlevels']
            else:
                levels = self.grid['levels']
            QuadContourSet = plt.contour(
                    self.grid['x2d'], self.grid['z2d'], self.grid['Psi'],
                    levels, colors=color, linestyles='-', linewidths=1.0,
                    alpha=0.5, zorder=5)
            self.grid['levels'] = QuadContourSet.levels
            plt.axis('equal')
            #plt.quiver(self.grid['x2d'], self.grid['z2d'], 
            #           self.grid['Bx'], self.grid['Bz'])

