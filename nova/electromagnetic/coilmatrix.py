import numpy as np


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

    
if __name__ == '__main__':
    cm = CoilMatrix()
    
    
    
    