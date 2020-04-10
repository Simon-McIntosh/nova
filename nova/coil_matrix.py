import numpy as np


class CoilMatrix:
    '''
    container for coil_matrix and subcoil_matrix (filament) data
    
    Formulae:
        Psi = [flux][Ic] (Wb)
        B[*] = [field[*]][Ic] (T)
        flux = reduce([Nt][Nc]*[_flux])
        field[*] = reduce([Nt][Nc]*[_field])
        
    Attributes:
        _flux (np.array): unit filaments
        flux (np.array): coil colocation flux matrix 
        field (2D np.array): coil colocation field matrices
            _field[*] (np.array): unit filaments
            field['x'] (np.array): radial field
            field['z'] (np.array): vertical field 
    '''
    
    # main class attribures
    _coilmatrix_attributes = ['_flux', 'flux', '_field', 'field']

    def __init__(self, **coilmatrix_attributes):
        self._initialilze_coilmatrix_attributes()
        self.coilmatrix = coilmatrix_attributes  # exchange attributes
             
    def _initialilze_coilmatrix_attributes(self):
        for attribute in self._coilmatrix_attributes:  
            setattr(self, f'{attribute}', None)  # unlink from DataFrame
            if 'flux' in attribute:
                setattr(self, f'{attribute}', np.array([]))
            else:
                setattr(self, f'{attribute}', 
                        {var: np.array([]) for var in ['x', 'z']})
            
    @property
    def coilmatrix(self):
        'extract coilmatrix attributes'
        return {attribute: getattr(self, attribute)
                for attribute in self._coilmatrix_attributes}

    @coilmatrix.setter
    def coilmatrix(self, coilmatrix_attributes):
        'set coilmatrix attributes'
        for attribute in self._coilmatrix_attributes:
            value = coilmatrix_attributes.get(attribute, None)
            if value is not None:
                setattr(self, attribute, value)
    
    
if __name__ == '__main__':
    cm = CoilMatrix()
    
    '''

    print(cm.subfield['x'])
    '''
    
    
    