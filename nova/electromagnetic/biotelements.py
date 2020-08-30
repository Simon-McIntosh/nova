import numpy as np
from scipy.special import ellipk, ellipe

class Points:
    
    'store source and target points in structured array'
    
    mu_o = 4e-7*np.pi  # magnetic constant [Vs/Am]
    
    _source_cross_section = ['square', 'rectangle', 'circle', 'ellipse', 
                             'skin', 'shell', 'polygon']
    
    _points_dtype = [('rs', float),  # source radius (centroid)
                     ('rs_rms', float),  # source radius (rms)
                     ('r', float),  # target radius
                     ('zs', float),  # source height
                     ('z', float),  # target height
                     ('Ns', float),  # source turn number
                     ('N', float),  # target turn number
                     ('dL', float),  # source-target seperation
                     ('dl', float),  # primary shape delta 
                     ('dt', float),  # secondary shape delta 
                     ('dx', float),  # radial bounding box delta 
                     ('dz', float),  # vertical bounding box delta
                     ('dr', float),  # maximum filament dimension
                     ('cs', 'U10'),  # source cross section 
                     ('instance', 'U10'),  # interaction method
                     ('far_field', bool)]  # far field flag
    
    def __init__(self, rms=False, **kwargs):
        self.initialize_points()
    
    def initialize_point_deltas(self):
        self.point_delta = {f'd{var}': 0 for var in ['r', 'rs', 'z', 'zs']}
          
    def initialize_point_array(self, Npoints):
        self.Npoints = Npoints
        self._points = np.zeros(Npoints, dtype=self._points_dtype)
        
    def initialize_points(self):
        self.initialize_point_deltas()
        self.vector = {}

    @property
    def points(self):
        return self._points
    
    @points.setter
    @profile
    def points(self, points):
        self.Npoints = len(points)  # interaction number
        self._points = points  # store point subset
        self.initialize_points()
        if hasattr(self, 'set_point_position'):
            self.set_point_position()  # Vectors.set_points
        if hasattr(self, 'offset'):    
            self.offset()  # Filament.offset
        
    def update_point_seperation(self):
        self.points['dL'] = np.linalg.norm(
            np.array([self.points['rs'] - self.points['r'],
                      self.points['zs'] - self.points['z']]), axis=0)
        
        
class Vectors(Points):
    
    def __init__(self, rms=False):
        self.rms = rms
        Points.__init__(self)
        
    def set_point_position(self, **kwargs):
        '''
        (re)position source filaments and target points [dr, drs, dz, dzs]
        set vector attributes: r, rs, z, zs
        '''
        self.rms = kwargs.pop('rms', self.rms)  # update rms flag
        for dvar in self.point_delta:
            delta = kwargs.pop(dvar, self.point_delta[dvar])
            var = pvar = dvar[1:]
            if not hasattr(self, var) or not \
                    np.isclose(delta, self.point_delta[dvar]).all():
                self.point_delta[dvar] = delta
                if var == 'rs' and self.rms:
                    pvar += '_rms'
                self.vector[var] = self.points[pvar] + self.point_delta[dvar]
                self.update_flag = True
        
    def update(self):
        if self.update_flag:
            self.b = self.vector['rs'] + self.vector['r']
            self.gamma = self.vector['zs'] -self.vector['z']
            self.a2 = self.gamma**2 + (self.vector['r'] + self.vector['rs'])**2
            self.a = np.sqrt(self.a2)
            self.k2 = 4 * self.vector['r'] * self.vector['rs'] / self.a2  
            self.ck2 = 1 - self.k2  # complementary modulus
            self.K = ellipk(self.k2)  # first complete elliptic integral
            self.E = ellipe(self.k2)  # second complete elliptic integral 
            self.update_flag = False
            
            
class Filament(Vectors):
    
    'compute interaction using complete circular filaments'
    
    _cross_section_factor = {'circle': np.exp(-0.25),  # circle-circle
                             'square': 2*0.447049,  # square-square
                             'skin': 1}  # skin-skin
    
    _cross_section = 'filament'  # applicable cross section type
    
    def __init__(self, rms=True):
        Vectors.__init__(self, rms=rms)

    @profile
    def offset(self):
        'offset source and target points '
        self.dL = np.array([self.vector['r']-self.vector['rs'],
                            self.vector['z']-self.vector['zs']])
        self.dL_mag = np.linalg.norm(self.dL, axis=0)
        self.dL_norm = np.zeros((2, self.Npoints))
        self.index = np.isclose(self.dL_mag, 0)  # self index
        self.dL_norm[0, self.index] = 1  # radial offset
        self.dL_norm[:, ~self.index] = \
            self.dL[:, ~self.index] / self.dL_mag[~self.index]
        idx = self.dL_mag < self.points['dr'] # seperation < L2 norm radius
        ro = self.points['dr'][idx] * np.array(
            [self._cross_section_factor['square'] 
             if cs not in self._cross_section_factor 
             else self._cross_section_factor[cs] 
             for cs in self.points['cs'][idx]])
        factor = (1 - self.dL_mag[idx] / self.points['dr'][idx]) / 2
        deltas = {}
        for i, var in enumerate(['r', 'z']):
            offset = np.zeros(self.Npoints)
            offset[idx] = factor * ro * self.dL_norm[i][idx]
            deltas.update({f'd{var}': offset, f'd{var}s': -offset})
        self.set_point_position(**deltas)
        
    def flux(self): 
        'vector and scalar potential'
        self.update()  # on-demand coefficent update
        Aphi = 1 / (2*np.pi) * self.a/self.vector['r'] * \
            ((1 - self.k2/2) * self.K - self.E)  # Wb/Amp-turn-turn
        psi = 2 * np.pi * self.mu_o * self.vector['r'] * Aphi  # scalar potential
        return psi
    
    def radial_field(self):  
        self.update()  # on-demand coefficent update
        Br = self.mu_o / (2*np.pi) * self.gamma * (
            self.K - (2-self.k2) / (2*self.ck2) * self.E) / (self.a*self.vector['r'])
        return Br  # T / Amp-turn-turn

    def vertical_field(self):  # T / Amp-turn-turn
        self.update()  # on-demand coefficent update
        Bz = self.mu_o / (2*np.pi) * (self.vector['r']*self.K - \
            (2*self.vector['r'] - self.b*self.k2) / 
            (2*self.ck2) * self.E) / (self.a*self.vector['r'])
        return Bz  # T / Amp-turn-turn


class BiotPoints(Points):
    
    def __init__(self, ndr=0):
        self.ndr = ndr  # far field interaction radius, filament dr multiple
        self.initialize_biot_instance()
        self.add_default_biot_instance()
        
    def initialize_biot_instance(self):
        self.biot_instance = {}  # interaction instance
        self.cross_section = {}  # aplicable cross sections
        
    def add_default_biot_instance(self):
        self.add_biot_instance('filament', Filament(), far_field=True)
        # insert additional biot methods as required ...
        #   self.add_biot_instance(... )
           
    def add_biot_instance(self, name, biot_instance, cross_section=None, 
                     far_field=False):
        '''
        Attributes:
            name (str): instance label
            biot_instance (instance): Biot instance
            
        Kwargs:
            cross_section (str or [str]): applicable cross sections
            far_field (bool): promote instance to far field
        '''
        if cross_section is None:
            cross_section = biot_instance._cross_section
        if isinstance(cross_section, str):
            if cross_section == 'all':
                cross_section = self._cross_sections
            else:
                cross_section = [cross_section]
        if far_field:
            self.far_field = name
        self.biot_instance[name] = biot_instance
        self.cross_section[name] = cross_section
        
    @property
    def far_field(self):
        return self._far_field
    
    @far_field.setter
    def far_field(self, name):
        'set far field instance'
        self._far_field = name
        if hasattr(self, '_points'):
            self.update_far_field()
        
    def update_far_field(self):
        self.update_point_seperation()
        index = self.points['dL'] >= self.ndr * self.points['dr']
        self.points['instance'][index] = self.far_field
        self.points['far_field'] = False
        self.points['far_field'][index] = True
        
    @profile
    def set_biot_instance(self):
        'set instance attribute in points structured array'
        self.points['instance'] = ''
        self.update_far_field()
        nearfield = self.points['instance'] == ''
        for cs in np.unique(self.points['cs'][nearfield]):
            cs_index = (self.points['cs'] == cs) & nearfield
            found = False
            for name in self.cross_section:
                if cs in self.cross_section[name]:
                    self.points['instance'][cs_index] = name
                    found = True
                    break 
            if not found:
                self.points['instance'][cs_index] = self.far_field
        self.set_biot_index()
                
    @profile
    def set_biot_index(self):
        self.index = {}
        for name in self.biot_instance:
            if name in self.points['instance']:
                self.index[name] = self.points['instance'] == name
                # upload points to instance
                self.biot_instance[name].points = \
                    self.points[self.index[name]] 
            '''
            index = self.points['instance'] == name
            if sum(index) == 0:
                index = None
            self.index[name] = index
            if index is not None:  # upload points to instance
                self.biot_instance[name].points = self.points[index]
            '''
      
    def calculate(self, attribute):
        'calculate biot attributes (flux, radial_field, vertical_field)'
        variable = np.zeros(self.Npoints)
        for name in self.index:
            if self.index[name] is not None:
                variable[self.index[name]] = \
                        getattr(self.biot_instance[name], attribute)()
        return variable.reshape(self.nT, self.nS)  # source-target reshape (matrix)


if __name__ == '__main__':
    
    bp = BiotPoints(ndr=3)
    
    bp.initialize_point_array(4)
    
    bp.points['r'] = [5, 4, 4, 3]
    bp.points['rs'] = [5, 4.2, 6, 12]
    bp.points['dr'] = 0.3
    bp.points['cs'] = 'rectangle'
    bp.points['cs'][0] = 'polygon'
    bp.points['cs'][1] = 'square'
    
    bp.update_point_seperation()
    
    bp.add_biot_instance('filament', Filament(), far_field=True)
    bp.add_biot_instance('rec', Filament(), cross_section='square')
    bp.add_biot_instance('skin', Filament(), cross_section='skin')
    bp.add_biot_instance('poly', Filament(), cross_section='polygon')
    
    bp.set_biot_instance()
    print(bp.points)

    
    