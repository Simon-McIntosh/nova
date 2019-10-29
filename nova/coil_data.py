import numpy as np
import pandas as pd


class CoilData:
    '''
    provides fast access to dynamic coil and subcoil data

    Attributes:
        index (list): coil index
        nC (int): coil number
        Ic (np.array, float): coil line current [A]
        It (np.array, float): coil turn curent [A.turns]
        Nt (np.array, float): coil turn number
        control (np.array, bool): coil power supply status
    '''

    _attributes = {'Ic': 0, 'It': 0, 'Nt': 1, 'mpc': '', 'control': True}

    def __init__(self, frame, control=True):
        self.initalize_frame(frame)
        self.control = control

    def initalize_frame(self, frame):
        self.frame = {}  # extract attributes from frame
        self.frame['update'] = True  # full frame update flag
        self.frame['index'] = np.array(frame.index)  # frame index
        self.frame['nC'] = frame.nC  # frame coil number
        for attribute in self._attributes:
            if attribute in frame.columns:  # extract value from frame
                self.frame[attribute] = frame[attribute].to_numpy()
            else:  # set default from self._attributes
                self.frame[attribute] = \
                    np.array([self._attributes[attribute]
                              for __ in range(self.frame['nC'])])
        # extract mpc interger index
        self.mpc_index = [i for i, mpc in
                          enumerate(self.frame['mpc']) if not mpc]
        self.index = self.frame['index'][self.mpc_index]
        self.frame['mpc_referance'] = np.zeros(frame.nC, dtype=int)
        self.frame['mpc_factor'] = np.ones(frame.nC, dtype=int)
        for i, (index, mpc) in \
                enumerate(zip(self.frame['index'], self.frame['mpc'])):
            if not mpc:
                self.frame['mpc_referance'][i] = list(self.index).index(index)
            else:
                self.frame['mpc_referance'][i] = list(self.index).index(mpc[0])
                self.frame['mpc_factor'][i] = mpc[1]
        # subframe - link referance id to primary coil
        if 'coil' in frame.columns:
            for i, index in enumerate(self.index):
                self.index[i] = frame.loc[index, 'coil']
        self.nC = len(self.index)
        self._update_index = np.full(self.nC, True)
        self._control = self.frame['control'][self.mpc_index]
        # initalize current vectors
        self._Ic = self.frame['Ic'][self.mpc_index]
        self._It = self.frame['It'][self.mpc_index]
        # initalize coil turn number
        self.Nt = self.frame['Nt'][self.mpc_index]

    @property
    def control(self):
        return pd.DataFrame({'control': self._control,
                             'update': self._update_index},
                            index=self.index)

    @control.setter
    def control(self, flag):
        '''
        set update_index via control flag for coil current update
            flag == True: update control coils (self._control=True)
            flag == False: update solution coils (self._control=False)
            flag == None: update full current vector
        '''
        if flag is None:
            self._update_index = np.full(self.nC, True)  # full vector update
        elif flag:
            self._update_index = self._control  # control coil currents
        else:
            self._update_index = ~self._control  # solution coil currents

    def set_current(self, value, current_column='Ic'):
        '''
        update current in subindex variables (Ic and It)
        index built as union of value.index and coil.index
        Args:
            value (dict or itterable): current update vector
            current_column (str):
                'Ic' == line current [A]
                'It' == turn current [A.turns]
        '''
        self.frame['update'] = True
        nC = sum(self._update_index)  # length of update vector
        if isinstance(value, dict):  # dict
            current = np.zeros(nC, dtype=float)
            for i, index in enumerate(self.index[self._update_index]):
                if index in value:
                    current[i] = value[index]
                else:
                    raise KeyError(f'coil {index} not specified in input dict')
        else:  # itterable
            if not pd.api.types.is_list_like(value):
                value = [value]
            current = np.array(value, dtype=float)
            if len(current) != nC:  # cross-check input lenght
                raise IndexError('length if input does not match '
                                 'primary coil number\n'
                                 f'len(value): {len(value)}\n'
                                 f'self.nC: {self.nC}\n'
                                 f'self.index: {self.index}')
            else:
                current = current
        if current_column == 'Ic':
            self._Ic = current
            self._It = current * self.Nt
        elif current_column == 'It':
            self._Ic = current / self.Nt
            self._It = current
        else:
            raise AttributeError(f'current column {current_column} '
                                 'not in [Ic, It]')

    @property
    def Ic(self):
        '''
        Returns:
            self.Ic (np.array): coil instance line subindex current [A]
        '''
        return self._Ic

    @Ic.setter
    def Ic(self, value):
        self.set_current(value, 'Ic')

    @property
    def It(self):
        '''
        Returns:
            self.coil.It (np.array): coil instance turn current [A.turns]
        '''
        return self._It

    @It.setter
    def It(self, value):
        self.set_current(value, 'It')

    @property
    def Ip(self):
        # return total plasma current
        return self._Ip

    @Ip.setter
    def Ip(self, Ip):
        self.Ip = Ip

    def update_frame(self):
        if self.frame['update']:
            self.frame['Ic'] = self.Ic[self.frame['mpc_referance']] *\
                self.frame['mpc_factor']
            self.frame['It'] = self.frame['Ic'] * self.frame['Nt']
            self.frame['update'] = False
