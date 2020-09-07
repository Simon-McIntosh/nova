# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 13:22:00 2020

@author: mcintos
"""
    #def assemble(self):
        
        
    '''
    def assemble_source(self):
        'load source filaments into points structured array'
        for label, column in zip(
                ['rs', 'rs_rms', 'zs', 'Ns', 'dl', 'dt', 'dx', 'dz'],
                ['x', 'rms', 'z', 'Nt', 'dl', 'dt', 'dx', 'dz']):
            self.points[label] = np.dot(
                    np.ones((self.nT, 1)), 
                    getattr(self.source, column).reshape(1, -1)).flatten()
        csID = np.array([self._source_cross_section.index(cs)  # cross-section 
                         for cs in self.source.cross_section])
        csID = np.dot(np.ones((self.nT, 1)), csID.reshape(1, -1)).flatten()
        for i, cs in enumerate(self._source_cross_section):
            self.points['cs'][csID == i] = cs
        self.points['dr'] = np.linalg.norm(
            [self.points['dx'], self.points['dz']], axis=0) / 2

    def assemble_target(self):
        'load target points into points structured array'
        for label, column in zip(['r', 'z', 'N'], ['x', 'z', 'Nt']):
            self.points[label] = np.dot(
                    getattr(self.target, column).reshape(-1, 1), 
                    np.ones((1, self.nS))).flatten()
    '''
        
        
    '''
    frame = self._initialize_frame(*args, **kwargs)
    self.coilset = frame.coilset
    #self.frame = frame
    CoilFrame.__init__(self, frame)
    #self._emulate(**kwargs)
    '''
    
    '''
    def _emulate(self, **kwargs):
        self._emulate_nC(**kwargs)
        self._emulate_plasma()
        self._setattr(**kwargs)
        self._emulate_coil_index()
        self._check_attribute_length()
        
    @staticmethod
    def _initialize_frame(*args, **kwargs):
        nargs = len(args)
        if nargs == 0:  # key-word input
            frame = kwargs
        elif len(args) == 1:  # CoilFrame or dict
            frame = args[0]
        else:  # arguments ordered as BiotFrame._frame_attributes
            frame = {key: args[i] 
                     for i, key in enumerate(BiotFrame._frame_attributes)}
        return frame
        
    def _emulate_nC(self, **kwargs):
        'emulate CoilFrame.nC: calculate maximum element number in kwars' 
        n2d = [np.shape(kwargs[key]) 
               for key in kwargs if is_list_like(kwargs[key])]
        nC = [np.prod(n) for n in n2d]
        arg_nC = np.argmax(nC)
        self.nC = nC[arg_nC]  # filament number
        self._nC = self.nC  # collapsed filament number
        self.n2d = n2d[arg_nC]  # 2d shape

    def _emulate_plasma(self):
        self._plasma_index = np.zeros(self.nC, dtype=bool)
        self.Np = np.array([]) # plasma filament turn number
        self.nP = 0  # number of plasma filaments
    
    def _emulate_coil_index(self):
        'emulate CoilFrame._reduction_index'
        if not hasattr(self, '_reduction_index'):
            self._reduction_index = np.arange(self._nC)
    
    def _setattr(self, **kwargs):
        'set data attributes'
        for key in self._frame_attributes:
            if key in kwargs:
                value = kwargs[key]
            elif key in self._default_frame_attributes:
                value = self._default_frame_attributes[key]
            else:
                raise KeyError(f'required attribute {key} not found')
            if not is_list_like(value):
                value = [value]
            value = np.array(value).flatten()  # ensure 1D input
            if len(value) == 1:
                value = np.array([value[0] for __ in range(self.nC)])
            setattr(self, key, value)
            
    def _check_attribute_length(self):
        nC = np.zeros(len(self._frame_attributes), dtype=int)
        for i, attribute in enumerate(self._frame_attributes):
            nC[i] = np.size(getattr(self, attribute))
        if not np.all(nC == self.nC):
            err = Series(nC, index=self._frame_attributes, name='nC')
            raise IndexError(f'miss-matched data input: \n{err}')
    '''
