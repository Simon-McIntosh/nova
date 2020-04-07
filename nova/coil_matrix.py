import numpy as np


class CoilMatrix:
    '''
    container for coil_matrix and subcoil_matrix (filament) data
    
    Formulae:
        Psi = [flux][Ic] (Wb)
        B[*] = [field[*]][Ic] (T)
        F[*] = [Ic]'[force[*]][Ic] (N, Nm)

    Attributes:
        _flux (np.array): flux = reduce([Nt][Nc]*[_flux])
        flux (np.array): coil colocation flux matrix 
        
        field (dict): coil colocation field matrices (DataFrame) 
            _field[*] (np.array): field[*] = reduce([Nt][Nc]*[_field])
            field['x'] (np.array): radial field
            field['z'] (np.array): vertical field
                    
        force (dict): coil force interaction matrices (DataFrame) 
            _force[*] (np.array): force[*] = reduce([Nt][Nc]*[_force])
            force['Fx'] (np.array):  net radial force
            force['Fz'] (np.array):  net vertical force
            force['xFx'] (np.array): first radial moment of radial force
            force['xFz'] (np.array): first radial moment of vertical force
            force['zFx'] (np.array): first vertical moment of radial force
            force['zFz'] (np.array): first vertical moment of vertical force
            force['My'] (np.array):  in-plane torque}
    '''
    
    # main class attribures
    _coilmatrix_attributes = ['flux', 'field', 'force']

    def __init__(self, **kwargs):
        self._unlink_coilmatrix_attributes()
        self.set_coilmatrix_attributes(**kwargs)  # set attributes from kwargs
     
    def _unlink_coilmatrix_attributes(self):
        for attribute in self._coilmatrix_attributes:  # unlink from DataFrame
            for prefix in ['_', '']:
                setattr(self, f'{prefix}{attribute}', None)  
            
    def set_coilmatrix_attributes(self, **kwargs):
        for attribute in self._coilmatrix_attributes:
            _name = f'_initialize_{attribute}'
            _initialize_function = getattr(self, _name)
            for prefix in ['_', '']:
                value = kwargs.get(f'{prefix}{attribute}', None)
                if value is None:
                    value = _initialize_function()
                setattr(self, f'{prefix}{attribute}', value)
            
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
    
    @staticmethod        
    def _initialize_flux():
        return np.array([])
    
    @staticmethod        
    def _initialize_field():
        return {'x': np.array([]), 'z': np.array([])}
    
    @staticmethod        
    def _initialize_force():
        return {'Fx': np.array([]), 'Fz': np.array([]),
                'xFx': np.array([]), 'xFz': np.array([]),
                'zFx': np.array([]), 'zFz': np.array([]),
                'My': np.array([])}
    '''
    def extend_frame(self, frame, index, columns):
        index = [idx for idx in index if idx not in frame.index]
        columns = [c for c in columns if c not in frame.columns]
        frame = concat((frame, DataFrame(index=index, columns=columns)),
                       sort=False)
        return frame
    
    def concatenate_matrix(self):
        index = self.index
        if 'coil' in self.columns:  # subcoil
            columns = np.unique(self.coil)
        else:
            columns = index
        for attribute in self._matrix_attributes:
            frame = getattr(self, attribute)
            if isinstance(frame, dict):
                for key in frame:
                    frame[key] = self.extend_frame(frame[key], index, columns)
            else:
                frame = self.extend_frame(frame, index, columns)
            setattr(self, attribute, frame)
    '''
                
    
if __name__ == '__main__':
    cm = CoilMatrix()
    
    '''
    cm.add_matrix(['PF1', 'PF2'], 
                  subindex=['PF1_0', 'PF1_1', 'PF2'])
    cm.add_matrix(['PF3', 'PF2'])
    print(cm.subfield['x'])
    '''
    
    
    