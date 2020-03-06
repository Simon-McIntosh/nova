from pandas import DataFrame, concat
import numpy as np


class CoilMatrix:
    '''
    container for coil_matrix and subcoil_matrix (filament) data
    
    Formulae:
        Psi = [flux][Ic] (Wb)
        B[*] = [field[*]][Ic] (T)
        F[*] = [Ic]'[force[*]][Ic] (N, Nm)
        Mc = inductance['Mc'][Ic] (H) line-current
        Mt = inductance['Mt'][It] (H) turn-current

    Attributes:
        flux (DataFrame): coil colocation flux matrix 
        
        field (dict): coil colocation field matrices (DataFrame) 
            field['x']: radial field
            field['z']: vertical field
                    
        force (dict): coil force interaction matrices (DataFrame) 
            force['Fx']:  net radial force
            force['Fz']:  net vertical force
            force['xFx']: first radial moment of radial force
            force['xFz']: first radial moment of vertical force
            force['zFx']: first vertical moment of radial force
            force['zFz']: first vertical moment of vertical force
            force['My']:  in-plane torque}
    '''
    
    # main class attribures
    _matrix_attributes = ['flux', 'field', 'force']

    def __init__(self, **kwargs):
        self.set_matrix_attributes(**kwargs)  # set attributes from kwargs
     
    def set_matrix_attributes(self, **kwargs):
        for attribute in self._matrix_attributes:
            _name = f'_initialize_{attribute}'
            _initialize_function = getattr(self, _name)
            value = kwargs.get(attribute, None)
            if value is None:
                value = _initialize_function()
            setattr(self, attribute, value)
            
    @property
    def matrix(self):
        kwargs = {attribute: getattr(self, attribute)
                  for attribute in self._matrix_attributes}
        return CoilMatrix(**kwargs)

    @matrix.setter
    def matrix(self, matrix):
        for attribute in self._matrix_attributes:
            setattr(self, attribute, getattr(matrix, attribute))
    
    @staticmethod        
    def _initialize_flux():
        return DataFrame()
    
    @staticmethod        
    def _initialize_field():
        return {'x': DataFrame(), 'z': DataFrame()}
    
    @staticmethod        
    def _initialize_force():
        return {'Fx': DataFrame(), 'Fz': DataFrame(),
                'xFx': DataFrame(), 'xFz': DataFrame(),
                'zFx': DataFrame(), 'zFz': DataFrame(),
                'My': DataFrame()}
    
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
                
    
if __name__ == '__main__':
    cm = CoilMatrix()
    
    cm.add_matrix(['PF1', 'PF2'], 
                  subindex=['PF1_0', 'PF1_1', 'PF2'])
    cm.add_matrix(['PF3', 'PF2'])
    print(cm.subfield['x'])
    
    
    