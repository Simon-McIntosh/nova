# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:56:04 2020

@author: mcintos
"""

    """
    @property
    def plasma(self):
        return self.plasma_metadata

    @plasma.setter
    def plasma(self, metadata):
        '''
        Attributes:
            metadata (None or dict): None: self.d2, kwargs: overide
        Metadata kwargs:
            filename (str | int): DINA filename | DINA fileindex
            to (float | str): input time | feature_keypoint
            x (float): current center
            z (float): current center
            dx (float): bounding box
            dz (float): bounding box
            Ip (float): current
            Lp (float): inductance
            cross_section (str): cross-section [circle, elipse, square, skin]
            turn_section (str): turn-section [circle, elipse, square, skin]
        '''
        if pd.isnull(metadata):  # release all plasma parameters
            self._plasma_metadata.loc[:] = np.nan
        else:
            self._plasma_metadata.update(pd.Series(metadata))  # fix parameters
        scenario = self.scenario  # extract scenario
        scenario.update(self._plasma_metadata)  # update
        if pd.isnull(self._plasma_metadata[['to', 'ko']]).all():
            for key in ['ko', 'to']:  # unset
                scenario[key] = np.nan
        else:
            if pd.notnull(self._plasma_metadata['to']):
                self.d2.to = scenario['to']  # time | keypoint
            elif pd.notnull(self._plasma_metadata['ko']):
                self.d2.ko = scenario['ko']  # keypoint
            for key in ['ko', 'to']:  # re-set
                scenario[key] = getattr(self.d2, key)
        for key in scenario.index:  # propogate changes
            self._plasma_metadata[key] = scenario[key]
        if pd.notnull(self._plasma_metadata['to']):
            plasma_metadata = self.extract_plasma_metadata()
            plasma_metadata.update(self._plasma_metadata)  # overwrite
            self._plasma_metadata.update(plasma_metadata)  # update
            for key in metadata:
                if pd.isnull(metadata[key]):
                    self._plasma_metadata[key] = np.nan  # release specified

    def extract_plasma_metadata(self, cross_section='ellipse',
                                turn_section='square', Lp=1.1e-5):
        'extract plasma metadata from self.d2 instance'
        '''
        d2 = {'t': 1.2, 'Ip': 1.2, 'Rp': 1.2, 'Lp': 1.2, 'Rcur': 1.2, 'Zcur': 1.2}
        plasma_metadata = {}
        for key in ['t', 'Ip', 'Rp', 'Lp', 'Rcur', 'Zcur']:
            #plasma_metadata[key] = self.d2.vector.at[key]
            plasma_metadata[key] = d2[key]

        #self.d2.vector.reindex(
        #        ['t', 'Ip', 'Rp', 'Zp', 'Lp', 'Rcur', 'Zcur'])
        #xp, zp, Lp = plasma_metadata.loc[['Rcur', 'Zcur', 'Lp']]
        #if not pd.isnull(plasma_metadata.loc[['Rcur'])
        '''
        coordinates = ['Rcur', 'Zcur']
        if not np.array([c in self.d2.index for c in coordinates]).all():
            coordinates = ['Rp', 'Zp']
        v2 = self.d2.vector.reindex(coordinates + ['Lp', 'kp', 'ap'])
        if pd.notnull(self._plasma_metadata['Lp']):  # fixed self-inductance
            v2['Lp'] = self._plasma_metadata['Lp']
        elif 'Lp' not in self.d2.index:
            v2['Lp'] = Lp  # default plasma self-inductance H
        xp, zp, Lp = v2.loc[coordinates + ['Lp']]  # current center, inductance
        # xp, zp, Lp = v2.iloc[:3]
        dr = self_inductance(xp).minor_radius(Lp)
        dx, dz = 2*dr, 2*dr  # bounding box
        Ip = self.d2.Ip  # current
        plasma_metadata = pd.Series(
                {'Ip': Ip, 'Lp': Lp, 'x': xp, 'z': zp, 'dx': dx, 'dz': dz,
                 'cross_section': cross_section, 'turn_section': turn_section})
        return plasma_metadata

    def update_plasma(self):
        self.plasma_metadata = self.extract_plasma_metadata()  # extract
        self.plasma_metadata.update(self._plasma_metadata)  # overwrite
        self.update_plasma_coil()  # update coil position
        self.update_plasma_current()

    def update_plasma_coil(self):
        pl = self.plasma_metadata.loc[['x', 'z', 'dx', 'dz']]
        if (pl != 0).all():  # plasma position valid
            if 'Plasma' in self.coil.index:
                update = (pl != self.coil.loc['Plasma', pl.index]).any()
            else:
                update = True
            if update:  # update plasma coils, inductance and interaction
                self.add_plasma(*pl)  # create / update plasma
                #self.update_inductance(source_index=['Plasma'])
                #self.update_interaction(coil_index=['Plasma'])
        elif 'Plasma' in self.coil.index:  # remove plasma
            self.drop_coil('Plasma')

    def update_plasma_current(self):
        if 'Plasma' in self.coil.index:
            self.Ip = self.plasma_metadata['Ip']  # update plasma current
    """

    """
    def self_inductance(self, name, update=False):
        '''
        calculate self-inductance and geometric mean of single coil

        Attributes:
            name (str): coil name (present in self.coil.index)
            update (bool): apply update to self.coil.loc[name]
        '''
        coilset = self.subset(name)  # create single coil coilset
        Mc = biot_savart(coilset).calculate_inductance()  # self-inductance
        L = Mc.at[name, name]
        dr = self_inductance(coilset.coil.x[name]).minor_radius(L)
        # calculate geometric and arithmetic means
        Nt = coilset.subcoil.Nt
        x_gmd = amigo.geom.gmd(coilset.subcoil.x, Nt)
        z_amd = amigo.geom.amd(coilset.subcoil.z, Nt)
        if update:  # apply update
            coilset.coil.loc[name, ['x', 'z']] = x_gmd, z_amd
            coilset.coil.loc[name, ['dx', 'dz']] = 2*dr, 2*dr
            CoilSet.patch_coil(coilset.coil)  # re-generate coil patch
            self.coil.loc[name] = coilset.coil.loc[name]
        coilset = None  # remove coilset
        return L
    
    def solve_colocation(self):
        bs = biot_savart(self.coilset, mutual=True)
        bs.colocate(subcoil=False)  # values at coil centroid 
        
        self.flux = bs.flux_matrix()
        
        
        #B = bs.field_matrix()
        
        #Psi = bs.reduce(Psi)
        #print(Psi)
        '''
        bs.colocate()
        Psi = bs.flux_matrix()
        B = bs.field_matrix()
        # flux
        self.flux, self.subflux = bs.reduce(Psi, retcol=True)
        # field
        for i, var in enumerate(['x', 'z']):
            self.field[var], self.subfield[var] = bs.reduce(B[i], retcol=True)        
        # force
        x = self.subcoil.x
        print(np.shape(B), np.shape(x))
        #self.subforce['Fx'].loc[:] = 2 * np.pi * bs.mu_o * x * B[1]
        #self.subforce['Fz'].loc[:] = -2 * np.pi * bs.mu_o * x * B[0]
        '''

        '''
            force['Fx']:  net radial force
            force['Fz']:  net vertical force
            force['xFx']: first radial moment of radial force
            force['xFz']: first radial moment of vertical force
            force['zFx']: first vertical moment of radial force
            force['zFz']: first vertical moment of vertical force
            force['My']:  in-plane torque}
        '''
  

    def update_inductance(self, mutual=True,
                          source_index=None, invert_source=False,
                          target_index=None, invert_target=False):
        '''
        calculate / update inductance matrix

            Attributes:
                mutual (bool): include gmr correction for adjacent turns
                coil_index (list): update inductance for coil subset
                invert_coil (bool): invert coil_index selection
        '''
        if self.inductance['Mc'].empty:
            source_index = None
            target_index = None
        if source_index is not None:
            source = self.subset(source_index, invert=invert_source)
        else:
            source = self.coilset
        if target_index is not None:
            target = self.subset(target_index, invert=invert_target)
        else:
            target = self.coilset
        bs = biot_savart(source=source, target=target, mutual=mutual)
        Mc = bs.calculate_inductance()
        
        if source_index is None and target_index is None:  # full update
            self.inductance['Mc'] = Mc  # line-current
        else:  # partial update
            index = np.append(source.coil.index, target.coil.index)
            expand = [name for name in np.unique(index)
                      if name not in self.inductance['Mc'].index]
            for name in expand:
                self.inductance['Mc'].loc[:, name] = None
                self.inductance['Mc'].loc[name, :] = None
            self.inductance['Mc'].loc[target.coil.index,
                                      source.coil.index] = Mc
            self.inductance['Mc'].loc[source.coil.index,
                                      target.coil.index] = Mc.T
        Nt = self.coilset.coil['Nt'].values
        Nt = Nt.reshape(-1, 1) * Nt.reshape(1, -1)
        self.inductance['Mt'] = self.inductance['Mc'] / Nt  # amp-turn
    """


            