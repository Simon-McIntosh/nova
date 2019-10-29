
class CoilMatrix:
    '''
    container for coil_matrix and subcoil_matrix (filament) data

    Attributes:

        force (dict): coil force interaction matrices (np.2darray)
            force['Fx']:  net radial force
            force['Fz']:  net vertical force
            force['xFx']: first radial moment of radial force
            force['xFz']: first radial moment of vertical force
            force['zFx']: first vertical moment of radial force
            force['zFz']: first vertical moment of vertical force
            force['My']:  net torque}

        inductance (dict): dictionary of inductance matirces (np.2darray)
            inductance['Mc']: line-current inductance matrix
            inductance['Mt']: amp-turn inductance matrix

        interaction (dict): dictionary of interaction matrices (np.2darray)
            interaction['Psi']: poloidal flux interaction matrix
            interaction['Bx']: radial field interaction matrix
            interaction['Bz']: vertical field interaction matrix
    '''

    def __init__(self):
        self.index = list(index)
        self.inductance = self.initialize_inductance(inductance)
        self.interaction = self.initialize_interaction(interaction)
        self.force = self.initialize_force(force)

    @staticmethod
    def initialize_inductance(inductance=None):
        '''
        inductance interaction matrix, H
        '''
        if inductance is None:
            inductance = {'Mc': pd.DataFrame(),  # line-current
                          'Mt': pd.DataFrame()}  # amp-turn
        return inductance

    @staticmethod
    def initialize_interaction(interaction=None):
        if interaction is None:  # initalize
            interaction = {
                    'Psi': pd.DataFrame(),  # flux interaction matrix
                    'Bx': pd.DataFrame(),  # radial field interaction matrix
                    'Bz': pd.DataFrame()}  # radial field interaction matrix
        return interaction

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
