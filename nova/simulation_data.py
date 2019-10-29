class SimulationData:
    '''
    container for sumulation data

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
            target['points'] (pd.DataFrame):  point name, xz-coordinates
            target['psi'] (pd.DataFrame): poloidal flux
            target['bx'] (pd.DataFrame): radial field
            target['bz'] (pd.DataFrame): vertical field
            target['update'] (bool): update flag
    '''

    def __init__(self, grid=None, target=None):
        self.grid = self.initialize_grid(grid)
        self.target = self.initialize_target(target)

    @staticmethod
    def initialize_grid(grid=None, **kwargs):
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
    def initialize_target(target=None):
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
