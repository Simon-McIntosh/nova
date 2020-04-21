import numpy as np
from pandas import DataFrame, concat
from pandas.api.types import is_list_like, is_dict_like
from amigo.pyplot import plt
from nova.mesh_grid import MeshGrid
from nova.biot_savart import BiotSavart, BiotAttributes
                

class Mutual(BiotSavart, BiotAttributes):
    
    _biot_attributes = []
    
    def __init__(self, **mutual_attributes):
        BiotSavart.__init__(self)
        BiotAttributes.__init__(self, **mutual_attributes)
        self.load_source(self.subcoil)  # link source
        self.load_target(self.subcoil)  # link target
        

class Grid(BiotSavart, BiotAttributes):
    ''' 
    grid interaction methods and data
    
    Key Attributes:
        n (int): grid dimension
        limit ([4float]): grid limits
        expand (float): grid expansion beyond coils
        nlevels (int): number of contour levels
        
    Derived Attributes:
        n2d: None,  # ([int, int]) as meshed dimensions
        x2d (np.array): 2D x-coordinates (radial)
        z2d (np.array): 2D z-coordinates
        levels (np.array): contour levels
        Psi (np.array):  poloidal flux
        Bx (np.array): radial field
        Bz (np.array): vertical field
    '''
    
    _biot_attributes = ['n', 'n2d', 'limit', 'coilset_limit', 'expand',
                   'nlevels', 'levels', 'x2d', 'z2d']
    
    _default_biot_attributes = {'n': 1e4, 'expand': 0.05, 'nlevels': 31}
    
    
    def __init__(self, subcoil, **grid_attributes):
        BiotSavart.__init__(self)
        BiotAttributes.__init__(self, **grid_attributes)
        self.load_source(subcoil)  # link source
        
    def solve_interaction(self):
        self.load_target(x=self.x2d, z=self.z2d)
        BiotSavart.solve_interaction(self)        
        
    def _generate_grid(self, **grid_attributes):
        self.biot_attributes = grid_attributes  # update attributes
        if self.n > 0:
            mg = MeshGrid(self.n, self._limit)  # set mesh
            self.n2d = [mg.nx, mg.nz]
            self.x, self.z = mg.x, mg.z
            self.dx = np.diff(self._limit[:2])[0] / (mg.nx - 1)
            self.dz = np.diff(self._limit[2:])[0] / (mg.nz - 1)
            self.x2d = mg.x2d
            self.z2d = mg.z2d
            self.solve_interaction()
    
    def generate_grid(self, **kwargs):
        '''
        compare kwargs to current grid settings, update grid on-demand

        kwargs:
            n (int): grid node number
            grid_limit (np.array): [xmin, xmax, zmin, zmax] fixed grid limits
            expand (float): expansion beyond coil limits (when limit not set)
            nlevels (int)
            levels ()
        '''
        grid_attributes = {}  # grid attributes
        for key in ['n', 'limit', 'coilset_limit',
                    'expand', 'levels', 'nlevels']:
            grid_attributes[key] = kwargs.pop(key, getattr(self, key))
        if grid_attributes['limit'] is None: # calculate coilset limit
            grid_attributes['coilset_limit'] = \
                self._get_coil_limit(grid_attributes['expand'])
        regenerate_grid = \
            not np.array_equal(grid_attributes['limit'], self.limit) or \
            not np.array_equal(grid_attributes['coilset_limit'], 
                               self.coilset_limit) or \
            grid_attributes['n'] != self.n
        if regenerate_grid:
            self._generate_grid(**grid_attributes)
        return regenerate_grid

    def _get_coil_limit(self, expand, xmin=1e-3):
        if expand is None:
            expand = self.expand  # use default
        if self.source.empty:
            raise IndexError('source coilframe empty')
        x, z, = self.source.x, self.source.z
        dx, dz = self.source.dx, self.source.dz
        limit = np.array([(x - dx/2).min(), (x + dx/2).max(),
                          (z - dz/2).min(), (z + dz/2).max()])
        dx, dz = np.diff(limit[:2])[0], np.diff(limit[2:])[0]
        delta = np.mean([dx, dz])
        limit += expand * delta * np.array([-1, 1, -1, 1])
        if limit[0] < xmin:
            limit[0] = xmin
        return limit
    
    @property
    def _limit(self):
        if self.limit is not None:
            return self.limit
        else:
            return self.coilset_limit
            
    def plot_grid(self, ax=None, **kwargs):
        self.generate_grid(**kwargs)
        if ax is None:
            ax = plt.gca()  
        MeshGrid._plot(self.x2d, self.z2d, self._limit[:2], self._limit[2:],
                       ax=ax, zorder=-500, **kwargs)  # plot grid 
        
    def plot_flux(self):
        if self.n > 0:
            if self.levels is None:
                levels = self.nlevels
            else:
                levels = self.levels
            QuadContourSet = plt.contour(
                    self.x2d, self.z2d, self.Psi,
                    levels, colors='lightgrey', linestyles='-', 
                    linewidths=1.0,
                    alpha=0.9, zorder=4)  # default zorder = 2
            if self.levels is None:
                self.levels = QuadContourSet.levels
            plt.axis('equal')
    
class Interaction:
    
    '''
    Formulae:

        F[*] = [Ic]'[force[*]][Ic] (N, Nm)

        force (dict): coil force interaction matrices (DataFrame) 
        _force[*] (np.array): force[*] = reduce([Nt][Nc]*[_force])
        force['Fx'] (np.array):  net radial force
        force['Fz'] (np.array):  net vertical force
        force['xFx'] (np.array): first radial moment of radial force
        force['xFz'] (np.array): first radial moment of vertical force
        force['zFx'] (np.array): first vertical moment of radial force
        force['zFz'] (np.array): first vertical moment of vertical force
        force['My'] (np.array):  in-plane torque}
    '''
    @staticmethod        
    def _initialize_force():
        return {'Fx': np.array([]), 'Fz': np.array([]),
                'xFx': np.array([]), 'xFz': np.array([]),
                'zFx': np.array([]), 'zFz': np.array([]),
                'My': np.array([])}
    '''
    def extend_frame(self, frame, index, columns):
        index = [idx for idx in index if idx not in frame.index]
        columns = [c for c in columns if c not in frame.columns]
        frame = concat((frame, DataFrame(index=index, columns=columns)),
                       sort=False)
        return frame
    
    def concatenate_matrix(self):
        index = self.index
        if 'coil' in self.columns:  # subcoil
            columns = np.unique(self.coil)
        else:
            columns = index
        for attribute in self._matrix_attributes:
            frame = getattr(self, attribute)
            if isinstance(frame, dict):
                for key in frame:
                    frame[key] = self.extend_frame(frame[key], index, columns)
            else:
                frame = self.extend_frame(frame, index, columns)
            setattr(self, attribute, frame)
    '''
                
    
class SimulationData:
    '''
    container for simulation data
    
        target (dict): poloidal target coordinates and data
            target['targets'] (DataFrame):  target xz-coordinates
            target['Psi'] (DataFrame): poloidal flux
            target['Bx'] (DataFrame): radial field
            target['Bz'] (DataFrame): vertical field
            target['update'] (bool): update flag

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


if __name__ is '__main__':
    
    #data = SimulationData()
    #data.generate_grid()
    
    grid = Grid(coilset={'x': [1, 2, 3], 'z': [3, 2, 5]}, limit=[-1, 1, -2, 2])
    grid.plot_grid(color='C3')
    grid.plot_grid(limit=[-2, 2, 0,5])

