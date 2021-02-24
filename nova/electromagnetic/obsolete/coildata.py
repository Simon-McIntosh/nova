# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 20:33:42 2021

@author: mcintos
"""
    """
    Key Attributes
    --------------
    Ic : float, array-like
        Coil line current [A]
    It : float, array-like
        Coil turn curent [A.turns]
    Nt : float, array-like
        Coil turn number.
    active : bool, array-like
        Coil current control status.
    optimize : bool, array-like
        Optimization flag.
    plasma : bool, array-like
        Plasma flag.
    feedback : bool, array-like
        Feedback stabilization flag
    """

    '''
        def add_column(self, label):
        """Add column to Frame initializing values to default."""
        if label not in self.metaframe.columns:
            self.metadata = {'additional': [label]}
            if len(self) > 0:  # initialize with default value
                print(label, self.metaframe.default[label])
                self[label] = self.metaframe.default[label]
    '''

    '''
    def __init__(self):
        """Build fast access data."""
        #
        #for attribute in self.metaarray.array:
        #    self.data[attribute] = self[attribute].to_numpy()
        # extract properties
        #self.validate_array()
        self.metaarray.properties = [p for p, __ in inspect.getmembers(
            Array, lambda o: isinstance(o, property))]
    '''

    '''
        def reduce_multipoint(self, matrix):
        """Apply multipoint constraints to coupling matrix."""
        _matrix = matrix[:, self._mpc_iloc]  # extract primary coils
        if len(self._mpl_index) > 0:  # add multi-point links
            _matrix[:, self._mpl_index[:, 0]] += \
                matrix[:, self._mpl_index[:, 1]] * \
                np.ones((len(matrix), 1)) @ self._mpl_factor.reshape(-1, 1)
        return _matrix
    '''

name change:
    _dataframe_attributes -> metaarray.data, dict
    _coildata_attributes -> metaarray.frame
    _update_dataframe -> metaarray.update
    _coildata_properties -> metaarray.properties

    frame: dict[str, str] = field(
        repr=False, default_factory=lambda: {'current_update': 'full'})




    # current update attributes
    _coilcurrent_attributes = []

    # CoilData indices
    _coildata_indices = ['reduction_index',
                         'plasma_reduction_index',
                         'plasma_iloc',
                         'ionize_index',
                         'current_index']

    # compact mpc attributes - subset of coilframe and coildata attributes
    _mpc_attributes = ['Ic', 'active', 'plasma', 'optimize', 'feedback',
                       'current_index']

    # multi-point constraints (shared line-current)
    _mpc_constraints = ['mpc_index', 'mpc_iloc', 'mpc_referance',
                        'mpc_factor', 'mpl_index', 'mpl_factor']

    # class properties (inspect.getmembers) exclude from setattr
    _coildata_properties = []

    # update flags
    _coildata_flags = {'update_dataframe': False,
                       'update_coilframe': True,
                       'update_biotsavart': True,
                       'current_update': 'full',
                       'relink_mpc': True}

    def __init__(self):
        self._initialize_coildata_flags()
        self._initialize_coildata_attributes()
        self._initialize_dataframe_attributes()
        self._initialize_coilcurrent_attributes()
        self._unlink_coildata_attributes()



    def _initialize_coildata_flags(self):
        for flag in self._coildata_flags:  # update read/write
            setattr(self, f'_{flag}', None)  # unlink from DataFrame
            setattr(self, f'_{flag}', self._coildata_flags[flag])
        self.update_dataframe = False

    def _initialize_dataframe_attributes(self):
        self._dataframe_attributes = self._mpc_attributes.copy()

    def _initialize_coildata_attributes(self):
        self._coildata_attributes = {}

    def _initialize_coilcurrent_attributes(self):
        self._coilcurrent_attributes = [attribute for attribute in
                                        self._mpc_attributes if attribute
                                        not in ['Ic', 'current_index']]

    def _unlink_coildata_attributes(self):
        # list attributes
        for attribute in self._dataframe_attributes +\
                         self._coildata_indices + \
                         self._mpc_attributes + \
                         self._mpc_constraints:
            setattr(self, f'_{attribute}', None)
        # dict attributes
        for attribute in self._coildata_attributes:
            setattr(self, f'_{attribute}', None)
        self.coildata_attributes = self._coildata_attributes

    @property
    def dataframe_attributes(self):
        """Return list of fast access dataframe attributes."""
        return self._dataframe_attributes

    @dataframe_attributes.setter
    def dataframe_attributes(self, dataframe_attributes):
        """Append coilframe attributes (fast access)."""
        for attribute in dataframe_attributes:
            if attribute not in self._dataframe_attributes:
                setattr(self, f'_{attribute}', None)
                self._dataframe_attributes.append(attribute)
                if attribute in self.columns:
                    self.refresh_coilframe(attribute)

    @property
    def coildata_attributes(self):
        """Extract coildata attributes."""
        self._coildata_attributes = {
                attribute: getattr(self, f'_{attribute}')
                for attribute in self._coildata_attributes}
        return self._coildata_attributes

    @coildata_attributes.setter
    def coildata_attributes(self, coildata_attributes):
        'set coildata attributes'
        update = {attribute: coildata_attributes[attribute]
                  for attribute in coildata_attributes
                  if attribute not in self._coildata_attributes}
        if len(update) > 0:
            self._coildata_attributes.update(update)
            for attribute in update:
                setattr(self, f'_{attribute}', None)
        for attribute in coildata_attributes:
            setattr(self, f'_{attribute}', coildata_attributes[attribute])

    @property
    def update_dataframe(self):
        return np.fromiter(self._update_dataframe.values(), dtype=bool).any()

    @update_dataframe.setter
    def update_dataframe(self, value):
        if type(value) == bool:
            self._update_dataframe = {
                attribute: value for attribute in self._dataframe_attributes}
        elif isinstance(value, dict):
            self._update_dataframe.update(value)
        else:
            self._update_dataframe.update({
                attribute: True for attribute in value})

    def _update_flags(self, **kwargs):
        for flag in self._coildata_flags:
            if flag in kwargs:
                setattr(self, f'_{flag}', kwargs[flag])

    def rebuild_coildata(self):
        if self.coil_number > 0:
            self._extract_mpc()  # extract multi-point constraints
            self._extract_data_attributes()  # extract from DataFrame columns
            self._extract_reduction_index()
            self.current_update = self._current_update  # set flag
            self.refresh_dataframe()  # transfer from coilframe to dataframe


    def _extract_data_attributes(self):
        self.update_dataframe = False
        for attribute in self._dataframe_attributes + self._coildata_indices:
            if attribute in ['active', 'plasma', 'optimize', 'feedback']:
                dtype = bool
            else:
                dtype = float
            if attribute in self:  # read from DataFrame column
                value = self[attribute].to_numpy(dtype=dtype)
            elif attribute in self._default_attributes:  # default
                value = np.array([self._default_attributes[attribute]
                                  for __ in range(self.coil_number)],
                                 dtype=dtype)
            else:
                value = np.zeros(self.coil_number, dtype=dtype)
            if attribute in self._mpc_attributes:  # mpc compaction
                value = value[self._mpc_iloc]
            setattr(self, f'_{attribute}', value)
        #self._ionize_index = self._plasma[self._mpc_referance]


