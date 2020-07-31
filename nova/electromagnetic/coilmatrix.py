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
        _flux (np.array): plasma source unit filaments [:, nP]
        _flux_ (np.array): plasma mutual unit filaments [nP, nP]
        flux (np.array): coil colocation flux matrix 
        field (dict): coil colocation field matrices
            _field[*] (np.array): plasma unit filaments
            field['x'] (np.array): radial field
            field['z'] (np.array): vertical field 
    '''
    
    # main class attribures
    _coilmatrix_attributes = ['_flux', '_flux_', 'flux', 
                              '_field', '_field_', 'field']

    def __init__(self):
        self._initialize_coilmatrix_attributes()
        #self.coilmatrix = coilmatrix_attributes  # exchange attributes
             
    def _initialize_coilmatrix_attributes(self):
        for attribute in self._coilmatrix_attributes:  
            if 'flux' in attribute:
                setattr(self, f'{attribute}', np.array([]))
            else:
                setattr(self, f'{attribute}', 
                        {var: np.array([]) for var in ['x', 'z']})
    '''
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
    '''
    
    
if __name__ == '__main__':
    cm = CoilMatrix()
    
    
    
    