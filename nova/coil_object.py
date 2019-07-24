from nova.coil_frame import CoilFrame
import pandas as pd


class CoilObject:
    '''
    instance wrapper for coilset data

    Attributes:

        coil (pd.DataFrame): coil data
        subcoil (pd.DataFrame): subcoil data

        inductance (dict): dictionary of inductance interaction matirces
            inductance['Mc'] (pd.DataFrame): line-current inductance matrix
            inductance['Mt'] (pd.DataFrame): amp-turn inductance matrix

        force (dict): coil force interaction matrices (pd.DataFrame)
            force['Fx']:  net radial force
            force['Fz']:  net vertical force
            force['xFx']: first radial moment of radial force
            force['xFz']: first radial moment of vertical force
            force['zFx']: first vertical moment of radial force
            force['zFz']: first vertical moment of vertical force
            force['My']:  net torque}
        subforce (dict): filament force interaction matrices (pd.DataFrame)
            subforce = {Fx, Fz, xFx, xFz, zFx, zFz, My}

        grid (dict): poloidal grid coordinates and interaction matrices
            grid['n'] ([2int]): grid dimensions
            grid['limit'] ([4float]): grid limits
            grid['x2d'] (np.array): x-coordinates (radial)
            grid['z2d'] (np.array): z-coordinates
            grid['Psi'] (pd.DataFrame): poloidal flux interaction matrix
            grid['psi'] (pd.DataFrame): poloidal flux
            grid['Bx'] (pd.DataFrame): radial field interaction matrix
            grid['Bz'] (pd.DataFrame): vertical field interaction matrix
            grid['bx'] (pd.DataFrame): radial field
            grid['bz'] (pd.DataFrame): vertical field
    '''
    _attributes = ['coil', 'subcoil', 'inductance', 'force', 'subforce',
                   'grid']

    def __init__(self, coil=None, subcoil=None, inductance=None, force=None,
                 subforce=None, grid=None, **kwargs):
        self.dCoil = kwargs.pop('dCoil', 0)
        self.dPlasma = kwargs.pop('dPlasma', 0.25)
        self.turn_fraction = kwargs.pop('turn_fraction', 1)
        self.initialize_coils(coil, subcoil)
        metadata = kwargs.pop('metadata', {})
        self.unpack_metadata(metadata)
        self.inductance = self.initalize_inductance(inductance)
        self.force = self.initialize_force(force)
        self.subforce = self.initialize_force(subforce)
        self.grid = self.initialize_grid(grid)

    def initialize_coils(self, coil, subcoil):
        if coil is None:
            self.coil = CoilFrame(
                    additional_columns=['Ic', 'It', 'Nt', 'Nf', 'dCoil',
                                        'subindex', 'mpc', 'cross_section',
                                        'turn_section', 'turn_fraction',
                                        'patch', 'polygon', 'part'],
                    default_attributes={'dCoil': self.dCoil,
                                        'turn_fraction': self.turn_fraction})
        else:
            self.coil = coil
        if subcoil is None:
            self.subcoil = CoilFrame(
                    additional_columns=['Ic', 'It', 'Nt', 'coil',
                                        'cross_section',
                                        'patch', 'polygon', 'part'],
                    default_attributes={})
        else:
            self.subcoil = subcoil

    @staticmethod
    def initalize_inductance(inductance=None):
        '''
        inductance interaction matrix, H
        '''
        if inductance is None:
            inductance = {'Mc': pd.DataFrame(),  # line-current
                          'Mt': pd.DataFrame()}  # amp-turn
        return inductance

    @staticmethod
    def initialize_force(force=None):
        '''
        force: a dictionary of force interaction matrices stored as dataframes
        '''
        if force is None:
            force = {
                    'Fx': None,  # radial force
                    'Fz': None,  # vertical force
                    'xFx': None,  # first radial moment of radial force
                    'xFz': None,  # first radial moment of vertical force
                    'zFx': None,  # first vertical moment of radial force
                    'zFz': None,  # first vertical moment of vertical force
                    'My': None}  # in-plane torque
        return force

    @staticmethod
    def initialize_grid(grid=None, **kwargs):
        if grid is None:  # initalize
            grid = {'n': 1e4,  # default grid dimensions
                    'n2d': None,  # ([int, int]) as meshed dimensions
                    'limit': None,  # (np.array) grid limits
                    'expand': 0.05,  # (float) grid expansion
                    'nlevels': 31,  # (int) number of contour levels
                    'levels': None,  # contour levels
                    'x2d': None,  # (np.array) x-coordinates
                    'z2d': None,  # (np.array) z-coordinates
                    'Psi': pd.DataFrame(),  # flux interaction matrix
                    'psi': pd.DataFrame(),  # poloidal flux
                    'Bx': pd.DataFrame(),  # radial field interaction matrix
                    'Bz': pd.DataFrame(),  # radial field interaction matrix
                    'bx': pd.DataFrame(),  # radial field
                    'bz': pd.DataFrame()}  # vertical field
            for key in kwargs:
                grid[key] = kwargs[key]  # overwrite defaults
        return grid

    def unpack_metadata(self, metadata):
        '''
        update coilset metadata for frame in ['coil', 'subcoil']
        '''
        for frame in metadata:
            self.update_metadata(frame, **metadata[frame])

    def update_metadata(self, frame, **metadata):
        '''
        update frame metadata
        '''
        getattr(self, frame)._update_metadata(**metadata)


if __name__ == '__main__':

    print('\nusage examples given in nova.coil_class')
