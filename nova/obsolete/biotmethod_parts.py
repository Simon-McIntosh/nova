class Interaction:
    
    '''
    Formulae:

        F[*] = [Ic]'[force[*]][Ic] (N, Nm)

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
