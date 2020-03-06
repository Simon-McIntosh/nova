import pandas as pd
import numpy as np
from nova.mesh_grid import MeshGrid


class SimulationData:
    '''
    container for sumulation data
    
        interaction (dict): coil grid / target interaction matrices (DataFrame)
            interaction['Psi']: poloidal flux interaction matrix
            interaction['Bx']: radial field interaction matrix
            interaction['Bz']: vertical field interaction matrix
            
        inductance (dict): coil colocation inductance matirces (DataFrame)
            inductance['Mc']: line-current inductance matrix
            inductance['Mt']: amp-turn inductance matrix

        grid (dict): poloidal grid coordinates
            grid['n'] ([2int]): grid dimensions
            grid['limit'] ([4float]): grid limits
            grid['x2d'] (np.array): x-coordinates (radial)
            grid['z2d'] (np.array): z-coordinates
            grid['psi'] (np.array): poloidal flux
            grid['bx'] (np.array): radial field
            grid['bz'] (np.array): vertical field
            grid['update'] (bool): update flag

        target (dict): poloidal target coordinates and data
            target['points'] (DataFrame):  point name, xz-coordinates
            target['psi'] (DataFrame): poloidal flux
            target['bx'] (DataFrame): radial field
            target['bz'] (DataFrame): vertical field
            target['update'] (bool): update flag
    '''

    def __init__(self, grid=None, target=None, **kwargs):
        self.interaction = self._initialize_interaction()
        self.grid = self._initialize_grid(grid, **kwargs)
        self.target = self._initialize_target(target)
        
    @staticmethod        
    def _initialize_interaction():
        return {'Psi': pd.DataFrame(),
                'Bx': pd.DataFrame(), 'Bz': pd.DataFrame()}
        
    @staticmethod        
    def _initialize_inductance():
        return {'Mc': DataFrame(), 'Mt': DataFrame()}

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
                    'psi': None,  # (np.array) poloidal flux
                    'bx': None,  # (np.array) radial field
                    'bz': None,  # (np.array) vertical field
                    'update': False}
            for key in kwargs:
                grid[key] = kwargs[key]  # overwrite defaults
        return grid

    @staticmethod
    def _initialize_target(target=None):
        if target is None:  # initalize
            target = {'points': pd.DataFrame(columns=['x', 'z', 'update']),
                      'psi': pd.DataFrame(),  # poloidal flux
                      'bx': pd.DataFrame(),  # radial field
                      'bz': pd.DataFrame(),  # vertical field
                      'update': False}
        return target

    def clear_target_data(self):
        for variable in ['psi', 'bx', 'bz']:
            self.target[variable] = pd.DataFrame()
            
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
            else:
                update = self.target['update'] 
        self.target['update'] = update
        return update

