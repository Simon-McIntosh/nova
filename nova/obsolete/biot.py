"""Build biotset."""
from dataclasses import dataclass, field

from nova.electromagnetic.frame import Frame


@dataclass
class Biot:
    """
    Manage biot methods for CoilSet.

    Examples
    --------
    Initialize biot methods

    >>> cs = CoilSet()
    >>> cs.biot_instances = 'field'
    >>> cs.biot_instances = ['probe', 'colocate']
    >>> cs.biot_instances = {'local_grid': 'grid'}
    >>> print(cs.biot_instances)
    {'plasmagrid': 'plasmagrid',
     'plasmafilament': 'plasmafilament',
     'local_grid': 'grid'}

    """

    frame: Frame
    subframe: Frame

    '''
    _biot_methods = {'mutual': Mutual,
                     'forcefield': ForceField,
                     'acloss': ACLoss,
                     'passive': Passive,
                     'background': BackGround,
                     'probe': Probe,
                     'field': Field,
                     'colocate': Colocate,
                     'grid': Grid,
                     'plasmagrid': PlasmaGrid,
                     'plasmafilament': PlasmaFilament}

    def __init__(self):
        """Initialize biot instances."""
        self._biot_instances = {}

    @property
    def biot_instances(self):
        """
        Initialize biot methods.

        Maintain dictionary of initialized biot methods

        Parameters
        ----------
        biot_instances : str or list or dict
            Collection of biot instances to initialize.

            - str or list : biot name and method assumed equal
            - dict : biot name and method diffrent {biot_name: biot_method}
            - dict : biot name, method and attributes
              {biot_name: [biot_method, biot_attributes], ...}

        Raises
        ------
        IndexError
            biot_method not given in self._biot_methods.

        Returns
        -------
        biot_instances : dict
            Dict of initialized biot methods.

        """
        return self._biot_instances

    @biot_instances.setter
    def biot_instances(self, biot_instances):
        if not is_list_like(biot_instances):
            biot_instances = [biot_instances]
        for biot_name in biot_instances:
            if isinstance(biot_instances, dict):
                if is_list_like(biot_instances[biot_name]):
                    biot_method, biot_attributes = biot_instances[biot_name]
                else:
                    biot_method = biot_instances[biot_name]
                    biot_attributes = {}
            else:
                biot_method = biot_name
                biot_attributes = {}
            if biot_method in self._biot_methods:
                if biot_name not in self._biot_instances:
                    self._biot_instances.update({biot_name: biot_method})
                if not hasattr(self, biot_name):  # initialize method
                    self._initialize_biot_method(biot_name, biot_method,
                                                 **biot_attributes)
            else:
                raise IndexError(f'method {biot_method} not found in '
                                 f'{self._biot_methods}\n'
                                 'unable to initialize method')

    def clear_biot(self):
        """Remove all biot instances."""
        for biot_name in self.biot_instances:
            delattr(self, biot_name)
        self._biot_instances = {}

    def _initialize_biot_method(self, name, method, **attributes):
        """Create biot instance and link to method."""
        setattr(self, name,
                self._biot_methods[method](self.subcoil, **attributes))

    @property
    def biot_attributes(self):
        """
        Manage attributes for all biot_instances.

        Parameters
        ----------
        biot_attributes : dict
            Set biot_attributes, default {}.

        Returns
        -------
        _biot_attributes : dict
            biot_attributes for all biot_instances.

        """
        _biot_attributes = {}
        for instance in self._biot_instances:
            biot_attribute = '_'.join([instance, 'biot_attributes'])
            _biot_attributes[biot_attribute] = \
                getattr(getattr(self, instance), 'biot_attributes')
        return _biot_attributes

    @biot_attributes.setter
    def biot_attributes(self, biot_attributes):
        for instance in self._biot_instances:
            biot_attribute = '_'.join([instance, 'biot_attributes'])
            setattr(getattr(self, instance), 'biot_attributes',
                    biot_attributes.get(biot_attribute, {}))
            getattr(self, instance).assemble_biotset()

    def _get_instance_attributes(self, attribute):
        return {instance: getattr(getattr(self, instance), attribute)
                for instance in self._biot_instances}

    def _set_instance_attributes(self, attribute, status):
        if not isinstance(status, bool):
            raise ValueError(f'flag type {type(status)} must be bool')
        else:
            for instance in self._biot_instances:
                setattr(getattr(self, instance), attribute, status)

    @property
    def update_plasma_turns(self):
        r"""
        Manage biot instance plasma_turn flags.

        Parameters
        ----------
        status : bool
            Set update flag for all biot_instances.
            Set flag to True following a change to plasma interaction
            matrix (plasma turns).
            Setting flag to True ensures that interaction matrix
            :math:`\_m\_` is re-evaluated

        Returns
        -------
        status : nested dict
            plasma_turn flag status for all biot instances.

        """
        return self._get_instance_attributes('update_plasma_turns')

    @update_plasma_turns.setter
    def update_plasma_turns(self, status):
        self._set_instance_attributes('update_plasma_turns', status)

    @property
    def update_coil_current(self):
        r"""
        Manage biot instance coil_current flags.

        Parameters
        ----------
        status : bool
            Set update flag for all biot_instances.
            Set flag to True following a change to coil currents.
            Setting flag to True ensures that interaction matrix
            dot product is re-evaluated

            .. math::
                \_M = \_m \cdot I_c

        Returns
        -------
        status : nested dict
            coil_current flag status for all biot instances.

        """
        return self._get_instance_attributes('update_coil_current')

    @update_coil_current.setter
    def update_coil_current(self, status):
        self._set_instance_attributes('update_coil_current', status)

    @property
    def update_plasma_current(self):
        r"""
        Manage biot instance plasma_current flags.

        Parameters
        ----------
        status : bool
            Set update flag for all biot_instances.
            Set flag to True following a change to plasma current or plasma
            interaction matrix (plasma turns).
            Setting flag to True ensures that interaction matrix
            dot product is re-evaluated

            .. math::
                \_M\_ = \_m\_ \cdot I_p

        Returns
        -------
        status : nested dict
            plasma_current flag status for all biot instances.

        """
        return self._get_instance_attributes('update_plasma_current')

    @update_plasma_current.setter
    def update_plasma_current(self, status):
        self._set_instance_attributes('update_plasma_current', status)

    def solve_biot(self):
        """
        Calculate interactions for all biot instances.

        Returns
        -------
        None.

        """
        for instance in self._biot_instances:
            getattr(self, instance).solve_biot()

    @property
    def dField(self):
        """
        Field probe resolution.

        Parameters
        ----------
        dField : float
            Resoultion of field probes spaced around the perimiters of
            specified coils.

            - 0: No interpolation - probes plaed at polygon boundary points
            - -1: dField set equal to each coils' dCoil parameter

        Returns
        -------
        dField: float
            Field probe resolution.

        """
        self._check_default('dField')
        return self._dField

    @dField.setter
    def dField(self, dField):
        self._dField = dField

    def update_field(self):
        """
        Update field biot instance.

        Calculate maximum L2 norm of magnetic field around the perimiters of
        specified coils. Probe resolution specified via dField property

        Returns
        -------
        None.

        """
        self.coil.refresh_dataframe()  # flush updates
        if self.field.nT > 0:  # maximum of coil boundary values
            frame = self.field.frame
            self.coil.loc[frame.index, frame.columns] = self.field.frame

    def update_forcefield(self, subcoil=False):
        if subcoil and self.forcefield.reduce_target:
            self.forcefield.solve_interaction(reduce_target=False)

        for variable in ['Psi', 'Bx', 'Bz']:
            setattr(self.subcoil, variable,
                    getattr(self.forcefield, variable))
        self.subcoil.B = \
            np.linalg.norm([self.subcoil.Bx, self.subcoil.Bz], axis=0)
        # set coil variables to maximum of subcoil bundles

        for variable in ['Psi', 'Bx', 'Bz', 'B']:
            setattr(self.coil, variable,
                    np.maximum.reduceat(getattr(self.subcoil, variable),
                                        self.subcoil._reduction_index))
    '''
