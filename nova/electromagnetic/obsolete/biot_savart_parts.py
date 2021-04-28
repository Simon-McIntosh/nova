# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 13:22:00 2020

@author: mcintos
"""

    def _offset_filaments(self):
        'offset source and target points'
        # point seperation
        dL = np.array([(self.r-self.rs), (self.z-self.zs)])
        dL_mag = np.linalg.norm(dL, axis=0)
        dr = self.dl/2  # filament characteristic radius
        ro = dr*self.cross_section_factor  # self seperation

        # zero-seperation
        index = np.isclose(dL_mag, 0)
        dL_norm = np.zeros((2, self.nI))
        dL_norm[0, index] = 1  # radial offset
        dL_norm[:, ~index] = dL[:, ~index] / dL_mag[~index]
        # initalize offsets
        dr, dz = np.zeros(self.nI), np.zeros(self.nI)

        # mutual offset
        nx = dL[0] / self.drs
        nz = dL[1] / self.dzs
        mutual_index = np.where((nx <= 5) & (nz <= 5))  # mutual index
        mutual_factor = self.gmr.evaluate(nx[mutual_index], nz[mutual_index])
        dr[mutual_index] = (mutual_factor-1) * dL[0, mutual_index]
        dz[mutual_index] = (mutual_factor-1) * dL[1, mutual_index]

        # self inductance index
        self_index = np.where(dL_mag <= ro)  # seperation < dl/2
        # self_dr = self.dl[self_index]/2  # filament characteristic radius
        # self_ro = self_dr*self.cross_section_factor[self_index]  # seperation
        self_ro = ro[self_index]
        self_factor = 1 - dL_mag[self_index]/self_ro
        dr[self_index] = self_factor*self_ro*dL_norm[0, self_index]  # radial
        dz[self_index] = self_factor*self_ro*dL_norm[1, self_index]  # vertical

        # rms offset
        drms = -(self.r+self.rs)/4 + np.sqrt((self.r+self.rs)**2 -
                                             8*dr*(self.r - self.rs + 2*dr))/4
        self.rs += drms
        self.r += drms
        # offset source filaments
        self.rs -= dr/2
        self.zs -= dz/2
        # offset target filaments
        self.r += dr/2
        self.z += dz/2



class Points:

    'store source and target points in structured array'

    mu_o = 4e-7*np.pi  # magnetic constant [Vs/Am]

    _source_cross_section = ['square', 'rectangle', 'circle', 'ellipse',
                             'skin', 'shell', 'polygon']

    '''
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
    '''

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

    def set_biot_index(self):
        self.index = {}
        for name in self.biot_instance:
            if name in self.points['instance']:
                self.index[name] = self.points['instance'] == name
                # upload points to instance
                self.biot_instance[name].points = \
                    self.points[self.index[name]]

    def calculate(self, attribute):
        'calculate biot attributes (flux, radial_field, vertical_field)'
        variable = np.zeros(self.Npoints)
        for name in self.index:
            if self.index[name] is not None:
                variable[self.index[name]] = \
                        getattr(self.biot_instance[name], attribute)()
        # source-target reshape (matrix)
        return variable.reshape(self.nT, self.nS)



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
