import numpy as np
from pandas import DataFrame, concat
from pandas.api.types import is_list_like, is_dict_like

from amigo.pyplot import plt
from nova.electromagnetic.meshgrid import MeshGrid
from nova.electromagnetic.biotsavart import BiotSavart, BiotAttributes
                

class Mutual(BiotSavart, BiotAttributes):
    
    _biot_attributes = []
    
    def __init__(self, subcoil, **mutual_attributes):
        BiotSavart.__init__(self)
        BiotAttributes.__init__(self, **mutual_attributes)
        self.load_source(subcoil)  # link source
        self.load_target(subcoil)  # link target
        

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
                        'nlevels', 'levels', 'x', 'z', 'x2d', 'z2d']
    
    _default_biot_attributes = {'n': 1e4, 'expand': 0.05, 'nlevels': 31}
    
    def __init__(self, subcoil, **grid_attributes):
        BiotSavart.__init__(self)
        BiotAttributes.__init__(self, **grid_attributes)
        self.load_source(subcoil)  # link source coilset
        
    def solve_interaction(self):
        if not hasattr(self, 'target'):
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
            self.load_target(x=self.x2d, z=self.z2d)
            self.solve_interaction()
    
    def generate_grid(self, **kwargs):
        '''
        compare kwargs to current grid settings, update grid on-demand

        kwargs:
            n (int): grid node number
            limit (np.array): [xmin, xmax, zmin, zmax] fixed grid limits
            expand (float): expansion beyond coil limits (when limit not set)
            nlevels (int)
            levels ()
            regen (bool): force grid regeneration
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
            grid_attributes['n'] != self.n or kwargs.get('regen', False)
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
        delta = np.max([dx, dz])
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
        
    def plot_flux(self, ax=None, lw=1, color='lightgrey',
                  **kwargs):
        if self.n > 0:
            if ax is None:
                ax = plt.gca() 
            levels = kwargs.get('levels', self.levels)
            if levels is None:
                levels = self.nlevels
            QuadContourSet = ax.contour(
                    self.x2d, self.z2d, self.Psi,
                    levels, colors=color, linestyles='-', 
                    linewidths=lw,
                    alpha=0.9, zorder=4)
            if self.levels is None:
                self.levels = QuadContourSet.levels
            plt.axis('equal')
            
    def plot_field(self):
        if self.n > 0:
            plt.quiver(self.x2d, self.z2d, self.Bx, self.Bz)
    

class Target(BiotSavart, BiotAttributes):
    
    ''' 
    traget interaction methods and data
    
    Key Attributes:
        targets (DataFrame): grid dimension
        
    Derived Attributes:
        Psi (np.array):  poloidal flux
        Bx (np.array): radial field
        Bz (np.array): vertical field
    '''
    _biot_attributes = ['targets']
    _default_biot_attributes = {}
    _target_attributes = ['x', 'z']
    
    def __init__(self, subcoil, **target_attributes):
        BiotSavart.__init__(self)
        BiotAttributes.__init__(self, **target_attributes)
        self.initialize_targets()
        self.load_source(subcoil)  # link source coilset
        
    def __repr__(self):
        return self.targets.to_string(max_cols=8, max_colwidth=10)
        
    def initialize_targets(self):
        self.targets = DataFrame(columns=self._target_attributes)

    def add_targets(self, targets, append=True, drop_duplicates=True):
        '''
        Attributes:
            targets (): target coordinates
                (dict, DataFrame() or list like):
                    x (np.array): x-coordinates
                    z (np.array): z-coordinates
            append (bool): create new list | append
            drop_duplicates (bool): drop duplicates
        '''
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
                io = self.targets.shape[0]
            else:
                io = 0
            targets['index'] = [f'P{i+io}'
                                for i in range(targets.shape[0])]
            targets.set_index('index', inplace=True)
            targets = targets.astype('float')        
        if not targets.empty:
            if append:
                self.targets = concat((self.targets, targets),
                                      ignore_index=True, sort=False)
                if drop_duplicates:
                    self.targets.drop_duplicates(
                            subset=['x', 'z'], inplace=True)
            else:  # overwrite
                self.targets = targets
                
    @property
    def n(self):
        return self.targets.shape[0]
                
    #@property
    #def n2d(self):
    #    return self.targets.shape[0]
    
    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca() 
        ls = kwargs.pop('ls', '.')
        color = kwargs.pop('color', 'C3')
        ax.plot(self.targets['x'], self.targets['z'],
                ls, color=color, **kwargs)
        
    def solve_interaction(self):
        self.load_target(x=self.targets['x'], z=self.targets['z'])
        BiotSavart.solve_interaction(self)   
      
        
class Colocate(Target):
    
    _target_attributes = ['label', 'x', 'z', 'value',
                          'nx', 'nz', 'd_dx', 'd_dz',  
                          'factor', 'weight']
        
    def __init__(self, subcoil, **colocate_attributes):
        Target.__init__(self, subcoil, **colocate_attributes)
        
    def add_targets(self, *args, **kwargs):
        '''
        add colocation points 
        
        len(args) == 1 (DataFrame or dict): colocation points as frame
        len(args) == fix.ncol (float or array): colocation points as args
        len(args) == 0
        
            (DataFrame or fix.columns): fix data points
        kwargs:
            (float): alternate input method
        '''
        default = {'label': 'Psi', 'x': 0., 'z': 0., 
                   'value': 0., 'd_dx': 0., 'd_dz': 0., 
                   'nx': 0., 'nz': 0., 'factor': 1., 'weight': 1.}
        if len(args) == 1:  # as frame
            target = args[0]   
        else:  # as args and kwargs
            target = {key: value 
                      for key, value in zip(self._target_attributes, args)}
            for key in kwargs:
                target[key] = kwargs[key]
        # populate missing entries with defaults
        if len(target) != self.targets.shape[1]:  # fill defaults
            for key in default:
                if key not in target:
                    target[key] = default[key]
        nrow = np.max([len(arg) if is_list_like(arg) else 1 for arg in args])
        target = DataFrame(target, index=range(nrow))
        norm = np.linalg.norm([target['nx'], target['nz']], axis=0)
        for nhat in ['nx', 'nz']:
            target.loc[target.index[norm != 0], nhat] /= norm[norm != 0]
        Target.add_targets(self, target)  # append Biot colocation targets
        
    def update_targets(self):
        'update targets.value from Psi and/or field'
        psi_index = ['Psi' in l for l in self.targets.label]
        self.targets.loc[psi_index, 'value'] = self.Psi[psi_index]
        
    def set_weight(self, index, gradient):
        index &= (gradient > 0)  # ensure gradient > 0
        self.targets.loc[index, 'weight'] = 1 / gradient[index]
            
    def update_weight(self):        
        'update colocation weight based on inverse of absolute gradient'
        gradient = self.targets.loc[:, ['d_dx', 'd_dz']].to_numpy()
        normal = self.targets.loc[:, ['nx', 'nz']].to_numpy()
        d_dx, d_dz = gradient.T
        # compute gradient magnitudes
        gradient_L2 = np.linalg.norm(gradient, axis=1)  # L2norm
        field_index = np.array(['B' in l for l in self.targets.label])
        self.set_weight(field_index, gradient_L2)
        gradient_dot = abs(np.array([g @ n for g, n in zip(gradient, normal)]))
        bndry_index = ['bndry' in l for l in self.targets.label]
        self.set_weight(bndry_index, gradient_dot)
        # calculate mean weight
        if sum(bndry_index) > 0:
            mean_index = bndry_index
        elif sum(field_index) > 0:
            mean_index = field_index
        else:
            mean_index = slice(None)
        mean_weight = self.targets.weight[mean_index].mean()
        # not field or Psi_bndry (separatrix)
        mean_index = [not field and not bndry for field, bndry in zip(
                    field_index, bndry_index)]
        
        self.targets.loc[mean_index, 'weight'] = mean_weight
        self.wsqrt = np.sqrt(self.targets.factor * 
                             self.targets.weight)
        self.wsqrt /= np.mean(self.wsqrt)  # normalize weight
        
        
    def plot_colocate(self, tails=True):
        self.update_weight()
        
        style = DataFrame(index=['color', 'marker', 'markersize',
                                 'markeredgewidth'])
        
        plt.plot(self.colocate)
        '''
            psi, Bdir, Bxz = [], [], []
            tail_length = 0.75
            for bc, w in zip(self.fix['BC'], weight):
                if 'psi' in bc:
                    psi.append(w)
                elif bc == 'Bdir':
                    Bdir.append(w)
                elif bc == 'Bx' or bc == 'Bz':
                    Bxz.append(w)
            if len(psi) > 0:
                psi_norm = tail_length / np.mean(psi)
            if len(Bdir) > 0:
                Bdir_norm = tail_length / np.mean(Bdir)
            if len(Bxz) > 0:
                Bxz_norm = tail_length / np.mean(Bxz)
            for x, z, bc, bdir, w in zip(self.fix['x'], self.fix['z'],
                                         self.fix['BC'], self.fix['Bdir'],
                                         weight):
                if bdir[0]**2 + bdir[1]**2 == 0:  # tails
                    direction = [0, -1]
                else:
                    direction = bdir
                # else:
                #    d_dx,d_dz = self.get_gradients(bc,x,z)
                #    direction = np.array([d_dx,d_dz])/np.sqrt(d_dx**2+d_dz**2)
                if 'psi' in bc:
                    norm = psi_norm
                    marker, size, color = 'o', 7.5, 'C0'
                    plt.plot(x, z, marker, color=color, markersize=size)
                    plt.plot(x, z, marker, color=[1, 1, 1],
                             markersize=0.3 * size)
                else:
                    if bc == 'Bdir':
                        norm = Bdir_norm
                        marker, size, color, mew = 'o', 4, 'C1', 0.0
                    elif bc == 'null':
                        norm = Bxz_norm
                        marker, size, color, mew = 'o', 2, 'C2', 0.0
                    elif bc == 'Bx':
                        norm = Bxz_norm
                        marker, size, color, mew = '_', 10, 'C2', 1.75
                    elif bc == 'Bz':
                        norm = Bxz_norm
                        marker, size, color, mew = '|', 10, 'C2', 1.75
                    plt.plot(x, z, marker, color=color, markersize=size,
                             markeredgewidth=mew)
                if tails:
                    plt.plot([x, x + direction[0] * norm * w],
                             [z, z + direction[1] * norm * w],
                             color=color, linewidth=2)
            plt.axis('equal')
        '''
        
    def plot(self):
        plt.plot(self.targets.x, self.targets.z, 'o')

        
class BiotMethods:
    
    _biot_methods = {'grid': Grid, 'mutual': Mutual, 'target': Target,
                     'colocate': Colocate}

    def initialize_biot_method(self, instance):
        'link biot instance to method'
        method = self._biot_instances[instance]
        setattr(self, instance, self._biot_methods[method](self.subcoil))   
        
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


if __name__ == '__main__':
    
    #data = SimulationData()
    #data.generate_grid()
    
    grid = Grid(coilset={'x': [1, 2, 3], 'z': [3, 2, 5]}, limit=[-1, 1, -2, 2])
    grid.plot_grid(color='C3')
    grid.plot_grid(limit=[-2, 2, 0,5])

