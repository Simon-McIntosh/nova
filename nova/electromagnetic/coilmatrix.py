import numpy as np

from nova.electromagnetic.biotelements import Filament


class CoilMatrix:
    '''
    container for coil_matrix and subcoil_matrix (filament) data
    
    Formulae:
        Psi = [psi][Ic] (Wb)
        Bx = [bx][Ic] (T)
        Bz = [bz][Ic] (T)
        
        
    Attributes:
        psi (np.array): flux matrix [nT, nC] 
        _psi (np.array): plasma unit filaments [:, nP]
        
        bx (np.array): radial field matrix [nT, nC] 
        bz (np.array): vertical field matrix [nT, nC] 
        _b* (np.array): plasma unit filaments [:, nP]
            
    '''
    
    # main class attribures
    _coilmatrix_attributes = ['psi', '_psi', 
                              'bx', '_bx', 'bz', '_bz']

    def __init__(self):
        self._initialize_coilmatrix_attributes()
             
    def _initialize_coilmatrix_attributes(self):
        for attribute in self._coilmatrix_attributes:  
            setattr(self, f'{attribute}', np.array([]))
            
    def flux_matrix(self, method):
        'calculate filament flux (inductance) matrix'
        psi = self.calculate(method, 'scalar_potential')
        self.psi , self._psi = self.save_matrix(psi)
        
    def field_matrix(self, method):
        'calculate subcoil field matrix'
        field = {'x': 'radial_field', 'z': 'vertical_field'}
        for xz in field:  # save field matricies
            b, _b = self.save_matrix(self.calculate(method, field[xz])) 
            setattr(self, f'b{xz}', b)
            setattr(self, f'_b{xz}', _b)
                
    def calculate(self, method, attribute):
        'calculate biot attributes (flux, radial_field, vertical_field)'
        return getattr(method, attribute)()  

    def solve(self, **biot_attributes):
        self.biot_attributes = biot_attributes  # update attributes
        self.update_biotset()  # assemble geometory matrices
        filament = Filament(self.source, self.target)
        self.flux_matrix(filament)  # assemble flux interaction matrix
        self.field_matrix(filament)  # assemble field interaction matricies 
        self._solve = False
        
    def save_matrix(self, M):
        M = M.reshape(self.nT, self.nS)  # source-target reshape (matrix)
        _M = M[:, self.source._plasma_index]  # extract plasma unit filaments  
        if self.source_turns:
            M *= self.source._Nt_.reshape(self.nT, self.nS) 
        if self.target_turns:  
            M *= self.target._Nt_.reshape(self.nT, self.nS)
        # reduce
        if self.reduce_source and len(self.source._reduction_index) < self.nS:
            M = np.add.reduceat(M, self.source._reduction_index, axis=1)
        if self.reduce_target and len(self.target._reduction_index) < self.nT:
            M = np.add.reduceat(M, self.target._reduction_index, axis=0)
        return M, _M  # turn-turn interaction, unit plasma interaction
    
    @property
    def Fx(self):
        return np.add.reduceat(2*np.pi*self.source.coilframe.x*self.source.coilframe.It*self.Bz, 
                      self.source._reduction_index)


    '''
    def _update_plasma(self, M, _M_):
        'update plasma turns'
        if self.source.nP > 0:  # source plasma filaments 
            _m = _M * self.source.Np
            M[:, self.source._plasma_iloc] = np.add.reduceat(
                _m, self.source._plasma_reduction_index, axis=1)
        if _M_.size > 0:  # update target plasma filaments
            M[self.source._plasma_iloc, :] = M[:, self.source._plasma_iloc].T
            _m_ = np.add.reduceat(_M_ * self.source.Np,
                                  self.source._plasma_reduction_index, axis=1)
            _m_ = np.add.reduceat(_m_.T * self.target.Np,
                                  self.target._plasma_reduction_index, axis=1)
            M[self.target._plasma_iloc][:, self.source._plasma_iloc] = _m_.T
             
    def update_flux(self):
        self._update_plasma(self.flux, self._flux)
        
    def update_field(self):
        for xz in self.field:
            self._update_plasma(self.field[xz], self._field[xz])
    '''
            
    def _reshape(self, M):
        if hasattr(self, 'n2d'):
            M = M.reshape(self.n2d)
        return M
    
    def _dot(self, variable):
        if self._solve:
            self.solve()
        matrix = getattr(self, variable.lower())
        return self._reshape(np.dot(matrix, self.source.coilframe._Ic))

    @property
    def Psi(self):
        return self._dot('Psi')
    
    @property
    def Bx(self):
        return self._dot('Bx')
    
    @property
    def Bz(self):
        return self._dot('Bz')
    
    @property 
    def B(self):
        return np.linalg.norm([self._dot('Bx'), self._dot('Bz')], axis=0)

    
if __name__ == '__main__':
    cm = CoilMatrix()
    
    
    
    