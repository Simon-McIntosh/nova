from contextlib import contextmanager
import numpy as np
from pandas import DataFrame, Index
from pandas.api.types import is_list_like, is_dict_like
import inspect


class CoilData():
    '''
    provides fast access to dynamic coil and subcoil data

    Key Attributes:
        Ic (np.array, float): coil line current [A]
        It (np.array, float): coil turn curent [A.turns]
        Nt (np.array, float): coil turn number
        power (np.array, bool): coil power supply status 
        optimize (np.array, bool): optimization flag  
        plasma (np.array, bool): plasma flag
        
    Derived Attributes:
        Psi: magnetic flux [Wb]
        Bx: radial field [T]
        Bz: vertical field [T]
        B: field magnitude [T]
        Fx:  net radial force [N]
        Fz:  net vertical force [N]
        xFx: first radial moment of radial force [Nm]
        xFz: first radial moment of vertical force [Nm]
        zFx: first vertical moment of radial force [Nm]
        zFz: first vertical moment of vertical force [Nm]
        My:  in-plane torque [Nm]
    
    Formulae (CoilMatrix): 
        Psi = [flux][Ic] (Wb)
        B[*] = [field[*]][Ic] (T)
        F[*] = [Ic]'[force[*]][Ic] (N, Nm)
    '''

    # list of fast access np.array variables linked to CataFrame
    _coilframe_attributes = []
    
    # metadata attributes
    _coildata_attributes = {}
    
    _coilcurrent_attributes = []
        
    # CoilData indices
    _coildata_indices = ['reduction_index',
                         'plasma_reduction_index',
                         'plasma_iloc',
                         'plasma_index', 
                         'current_index']
    
    # compact mpc attributes - subset of coilframe and coildata attributes
    _mpc_attributes = ['Ic', 'power', 'plasma', 'optimize', 'current_index']
    
    # multi-point constraints (shared line-current)
    _mpc_constraints = ['mpc_index', 'mpc_iloc', 'mpc_referance', 
                        'mpc_factor', 'mpl_index', 'mpl_factor']
    
    # class properties (inspect.getmembers) exclude from setattr
    _coildata_properties = []
    
    # update flags
    _coildata_flags = {'update_dataframe': False,
                       'update_coilframe': True,
                       'update_biotsavart': True,
                       'relink_mpc': True}
        
    def __init__(self):
        self._extract_coildata_properties()
        self._initialize_coildata_flags()
        self._initialize_coilframe_attributes()
        self._initialize_coildata_attributes()
        self._initialize_coilcurrent_attributes()
        self._unlink_coildata_attributes()
        
    def _extract_coildata_properties(self):
        self._coildata_properties = [p for p, __ in inspect.getmembers(
            CoilData, lambda o: isinstance(o, property))]
        
    def _initialize_coildata_flags(self):
        for flag in self._coildata_flags:  # update read/write
            setattr(self, f'_{flag}', None)  # unlink from DataFrame
            setattr(self, f'_{flag}', self._coildata_flags[flag]) 
        self.update_dataframe = False
        
    def _initialize_coilframe_attributes(self, **kwargs):
        self.coilframe_attributes = self._mpc_attributes
    
    def _initialize_coildata_attributes(self):
        self._coildata_attributes = {'current_update': 'full'}
                
    def _initialize_coilcurrent_attributes(self):
        self._coilcurrent_attributes = [attribute for attribute in 
                                        self._mpc_attributes if attribute 
                                        not in ['Ic', 'current_index']]
        
    def _unlink_coildata_attributes(self):
        # list attributes
        for attribute in self._coilframe_attributes +\
                         self._coildata_indices + \
                         self._mpc_attributes + \
                         self._mpc_constraints:
            setattr(self, f'_{attribute}', None)
        # dict attributes
        for attribute in self._coildata_attributes:
            setattr(self, f'_{attribute}', None)
        self.coildata_attributes = self._coildata_attributes
        
    @property 
    def coilframe_attributes(self):
        return self._coilframe_attributes
    
    @coilframe_attributes.setter 
    def coilframe_attributes(self, coilframe_attributes):
        'append coilframe attributes (fast access)'
        for attribute in coilframe_attributes:
            if attribute not in self._coilframe_attributes:
                setattr(self, f'_{attribute}', None)
                self._coilframe_attributes.append(attribute)
                if attribute in self.columns:
                    self.refresh_coilframe(attribute)
                    
    @property
    def coildata_attributes(self):
        'extract coildata attributes'
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
        self._update_dataframe = {attribute: value 
                                  for attribute in self._coilframe_attributes}
                             
    def _update_flags(self, **kwargs):
        for flag in self._coildata_flags:
            if flag in kwargs:
                setattr(self, f'_{flag}', kwargs[flag])
            
    def rebuild_coildata(self):
        if self.nC > 0:
            self._extract_mpc()  # extract multi-point constraints
            self._extract_data_attributes()  # extract from DataFrame columns
            self._extract_reduction_index()
            self.current_update = self._current_update  # set flag
            self.refresh_dataframe()
               
    def _extract_data_attributes(self):
        self.update_dataframe = False
        for attribute in self._coilframe_attributes + \
                         self._coildata_indices:
                if attribute in ['power', 'plasma', 'optimize']:
                    dtype = bool
                else:
                    dtype = float             
                if attribute in self:  # read from DataFrame column
                    value = self[attribute].to_numpy(dtype=dtype)
                elif attribute in self._default_attributes:  # default
                    value = np.array([self._default_attributes[attribute] 
                                      for __ in range(self.nC)], dtype=dtype)
                else:
                    value = np.zeros(self.nC, dtype=dtype)
                if attribute in self._mpc_attributes:  # mpc compaction
                    value = value[self._mpc_iloc]
                setattr(self, f'_{attribute}', value)
        self._plasma_index = self._plasma[self._mpc_referance]
                    
    def _extract_mpc(self):  # extract mpc interger index and factor
        mpc = self.get('mpc', [None for __ in range(self.nC)])
        self._mpc_iloc = [i for i, _mpc in enumerate(mpc) if not _mpc]
        self._mpc_index = self.index[self._mpc_iloc]
        self._mpc_referance = np.zeros(self.nC, dtype=int)
        self._mpc_factor = np.ones(self.nC, dtype=float)
        for i, (index, _mpc) in enumerate(zip(self.index, mpc)):
            if not _mpc:
                self._mpc_referance[i] = list(self._mpc_index).index(index)
            else:
                self._mpc_referance[i] = list(self._mpc_index).index(_mpc[0])
                self._mpc_factor[i] = _mpc[1]
        if 'coil' in self:  # link subcoil to coil referance
            mpc_index = self._mpc_index.to_numpy().copy()
            for i, index in enumerate(self._mpc_index):
                mpc_index[i] = self.at[index, 'coil']
            self._mpc_index = Index(mpc_index) 
        # construct multi-point link ()
        mpl = np.array([
            [referance, couple, factor] for couple, (referance, _mpc, factor) 
            in enumerate(zip(self._mpc_referance, mpc, self._mpc_factor))
            if _mpc])
        if len(mpl) > 0:
            self._mpl_index = mpl[:, :2].astype(int)  # (refernace, couple)
            self._mpl_factor = mpl[:, 2]  # coupling factor
        else:
            self._mpl_index = []
            self._mpl_factor = []
        self._relink_mpc = True      

    def _extract_reduction_index(self):  # extract reduction incices (reduceat)
        if 'coil' in self:  # subcoil
            coil = self.coil.to_numpy()
            _name = coil[0]
            _reduction_index = [0]
            for i, name in enumerate(coil):
                if name != _name:
                    _reduction_index.append(i)
                    _name = name
            self._reduction_index = np.array(_reduction_index)
            self._plasma_iloc = np.arange(self._nC)[
            self._plasma_index[self._reduction_index]]
            filament_indices = np.append(self._reduction_index, self.nC)
            plasma_filaments = filament_indices[self._plasma_iloc+ 1] - \
                    filament_indices[self._plasma_iloc]
            self._plasma_reduction_index = \
                    np.append(0, np.cumsum(plasma_filaments)[:-1])
        else:  # coil, reduction only applied to subfilaments
            self._reduction_index = None
            self._plasma_iloc = None
            self._plasma_reduction_index = None
    
    @property
    def current_update(self):
        return self._current_update
    
    @current_update.setter
    def current_update(self, update_flag):
        '''
        set update current_index via current flag for coil current update
            flag == 'full': update full current vector
            flag == 'active': update active coils (power & ~plasma)
            flag == 'passive': update passive coils (~power & ~plasma)
            flag == 'free': update free coils (optimize & ~plasma)
            flag == 'fix': update fix coils (~optimize & ~plasma)
            flag == 'plasma': update plasma (plasma)
            flag == 'coil': update all coils (~plasma)
        '''
        self._current_update = update_flag
        if self.nC > 0 and self._mpc_iloc is not None:
            if update_flag == 'full':
                self._current_index = np.full(self._nC, True)  # full
            elif update_flag == 'active':
                self._current_index = self._power & ~self._plasma  # active 
            elif update_flag == 'passive':
                self._current_index = ~self._power & ~self._plasma  # passive
            elif update_flag == 'free':
                self._current_index = self._optimize & ~self._plasma  # free
            elif update_flag == 'fix':
                self._current_index = ~self._optimize & ~self._plasma  # fix                
            elif update_flag == 'plasma':
                self._current_index = self._plasma  # plasma
            elif update_flag == 'coil':
                self._current_index = ~self._plasma  # all coils        
            else:
                raise IndexError(f'flag {update_flag} not in '
                                 '[full, actitve, passive, plasma, coil]')
    
    @property
    def current_index(self):
        'display power, optimize, plasma and current update status'
        if self.nC > 0:
            return DataFrame(
                    {'power': self._power, 
                     'optimize': self._optimize, 
                     'plasma': self._plasma,
                     self.current_update: self._current_index},
                    index=self._mpc_index)
        else:
            return DataFrame(columns=['power', 'optimize', 'plasma', 
                                      self.current_update])
                
    def _set_current(self, value, current_column='Ic', update_dataframe=True):
        '''
        update line-current in variable _Ic
        index built as union of value.index and coil.index
        Args:
            value (dict or itterable): current update vector
            current_column (str):
                'Ic' == line current [A]
                'It' == turn current [A.turns]
        '''
        self._update_dataframe['Ic'] = update_dataframe  # update dataframe
        self._update_dataframe['It'] = update_dataframe
        nU = sum(self._current_index)  # length of update vector
        current = getattr(self, '_Ic')
        if current_column == 'It':  # convert to turn current
            current * self._Nt[self._mpc_iloc]
        if is_dict_like(value):
            for i, (index, update) in enumerate(zip(self.index[self._mpc_iloc],
                                                    self._current_index)):
                if index in value and update:
                    current[i] = value[index]  # overwrite
        else:  # itterable
            if not is_list_like(value):
                value = value * np.ones(nU)
            if len(value) == nU:  # cross-check input length
                current[self._current_index] = value
            else:
                raise IndexError(
                        'length of input does not match '
                        f'"{self.current_update}" coilset\n'
                        'coilset.index: '
                        f'{self._mpc_index[self._current_index]}\n'
                        f'value: {value}\n\n'
                        f'{self.current_update}\n')
        if current_column == 'Ic':
            self._Ic = current
        elif current_column == 'It':
            self._Ic = current / self._Nt[self._mpc_iloc]
        else:
            raise AttributeError(f'current column {current_column} '
                                 'not in [Ic, It]')
         
    @property
    def _nC(self):  # mpc coil number 
        return len(self._mpc_iloc)
    
    @property
    def Ic(self):
        '''
        Returns:
            self.Ic (np.array): coil instance line subindex current [A]
        '''
        return self._Ic[self._mpc_referance] * self._mpc_factor

    @Ic.setter
    def Ic(self, value):
        self._set_current(value, 'Ic')
        
    @property
    def It(self):
        '''
        Returns:
            self.coil.It (np.array): coil instance turn current [A.turns]
        '''
        return self._Ic[self._mpc_referance] * self._mpc_factor * self._Nt

    @It.setter
    def It(self, value):
        self._set_current(value, 'It')
    
    @property
    def Np(self):  # plasma filament turn number
        return self._Nt[self._plasma_index]
    
    @Np.setter
    def Np(self, value):  # set plasma fillament number
        self._Nt[self._plasma_index] = value
        self._Nt[self._plasma_index] /= np.sum(self._Nt[self._plasma_index])
        self._update_dataframe['Nt'] = True
        
    @property
    def nP(self):  # number of plasma filaments
        return np.sum(self._plasma_index)
    
    @property
    def nPlasma(self):  # number of active plasma fillaments
        return len(self.Np[self.Np > 0])
        
    @property
    def Ip(self):
        ''''
        Returns:
            sum(It) (float): plasma line current [A]'
        '''
        return self.It[self._plasma_index]
        
    @Ip.setter
    def Ip(self, value):
        self._Ic[self._plasma] = value
        self._update_dataframe['Ic'] = True
        
    @property
    def Ip_sum(self):  # net plasma current
        return self.Ip.sum()
    
    @staticmethod 
    @contextmanager
    def _write_to_dataframe(self):
        'prevent local attribute write via __setitem__ during dataframe update'
        self._update_coilframe = False
        yield # with self._write_to_dataframe(self):
        self._update_coilframe = True
                
    def refresh_dataframe(self):
        'transfer data from coilframe attributes to dataframe'
        if self.update_dataframe:
            _update_dataframe = self._update_dataframe.copy()
            self.update_dataframe = False
            with self._write_to_dataframe(self):
                for attribute in _update_dataframe:
                    if _update_dataframe[attribute]:
                        self.loc[:, attribute] = getattr(self, attribute)
                        if attribute in ['Ic', 'It']:
                            _attr = next(attr for attr in ['Ic', 'It'] 
                                         if attr != attribute)
                            self._update_dataframe[_attr] = False
            
    def refresh_coilframe(self, key):
        'transfer data from dataframe to coilframe attributes'
        if self._update_coilframe:  # protect against regressive update
            if key in ['Ic', 'It']:
                _current_update = self.current_update
                self.current_update = 'full'
                self._set_current(self.loc[self.index[self._mpc_iloc], key],
                                  current_column=key, update_dataframe=False)
                self.current_update = _current_update
            else:
                value = self.loc[:, key].to_numpy()
                if key in self._mpc_attributes:
                    value = value[self._mpc_iloc]
                setattr(self, f'_{key}', value)
            if key in self._update_dataframe:
                self._update_dataframe[key] = False