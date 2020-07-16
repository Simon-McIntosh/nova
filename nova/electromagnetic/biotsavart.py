import numpy as np
from pandas.api.types import is_list_like
from pandas import Series
from scipy.special import ellipk, ellipe

from nova.electromagnetic.coilframe import CoilFrame
from nova.electromagnetic.coilmatrix import CoilMatrix
from amigo.pyplot import plt
import quadpy


class BiotFrame:
    
    _frame_attributes = ['x', 'z', 'dx', 'dz', 'Nt', 'cross_section']
    _default_frame_attributes = {
            'dx': 0, 'dz': 0, 'Nt': 1, 'cross_section': 'circle'}
    
    def __init__(self, **kwargs):
        self._load_data(**kwargs)
        
    def _load_data(self, **kwargs):
        self._emulate_nC(**kwargs)
        self._emulate_plasma()
        self._setattr(**kwargs)
        self._emulate_coil_index()
        self._check_attribute_length()
        
    @staticmethod
    def load(*args, **kwargs):
        'static load'
        nargs = len(args)
        if nargs == 0:  # key-word input
            frame = kwargs
        elif len(args) == 1:  # CoilFrame or dict
            frame = args[0]
        else:  # arguments ordered as BiotFrame._frame_attributes
            frame = {key: args[i] 
                     for i, key in enumerate(BiotFrame._frame_attributes)}
        if not isinstance(frame, CoilFrame):
            frame = BiotFrame(**frame)   # emulate dict as CoilFrame
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


class BiotAttributes:
    
    'manage attributes to and from Biot derived classes'
    _biot_attributes = []
    _default_biot_attributes = {}
    
    def __init__(self, **biot_attributes):
        self._biot_attributes += self._biotsavart_attributes
        self._biot_attributes += self._coilmatrix_attributes
        self._default_biot_attributes = {**self._default_biot_attributes, 
                                         **self._biotsavart_attributes}
        self.biot_attributes = biot_attributes
    
    @property
    def biot_attributes(self):
        return {attribute: getattr(self, attribute) for attribute in 
                self._biot_attributes}
        
    @biot_attributes.setter
    def biot_attributes(self, _biot_attributes):
        for attribute in self._biot_attributes:
            default = self._default_biot_attributes.get(attribute, None)
            value = _biot_attributes.get(attribute, None)
            if value is not None:
                setattr(self, attribute, value)  # set value 
            elif not hasattr(self, attribute):
                setattr(self, attribute, default)  # set default
                

class BiotArray:
    
    _cross_section_ID = {'square': 0, 'rectangle': 1, 
                         'circle': 2, 'ellipse': 3, 'skin': 4,
                         'shell': 5, 'polygon': 6}
    
    def __init__(self, source=None):
        if source is not None:
            self.load_source(source)
        
    def load_source(self, *args, **kwargs):
        self.source = BiotFrame.load(*args, **kwargs)
        
    def load_target(self, *args, **kwargs):
        self.target = BiotFrame.load(*args, **kwargs)
        if hasattr(self.target, 'n2d'):
            self.n2d = self.target.n2d  # target grid shape
        else:
            self.n2d = self.target.nC
            
    def assemble_source(self):
        for label, column in zip(
                ['rs', 'rs_rms', 'zs', 'Ns', 'dl', 'dt', 'dx', 'dz'],
                ['x', 'rms', 'z', 'Nt', 'dl', 'dt', 'dx', 'dz']):
            self.points[label] = np.dot(
                    np.ones((self.nT, 1)), 
                    getattr(self.source, column).reshape(1, -1)).flatten()
        # cross-section ID
        csID = np.array([self._cross_section_ID[cs] 
                        for cs in self.source.cross_section])
        self.points['csID'] = np.dot(np.ones((self.nT, 1)), 
                                     csID.reshape(1, -1)).flatten()
        self.points['dr'] = np.linalg.norm([self.points['dx'], 
                                            self.points['dz']], axis=0) / 2

    def assemble_target(self):
        for label, column in zip(['r', 'z', 'N'], ['x', 'z', 'Nt']):
            self.points[label] = np.dot(
                    getattr(self.target, column).reshape(-1, 1), 
                    np.ones((1, self.nS))).flatten()

    def assemble(self):
        'assemble interaction strucutred array'
        self.nS = self.source.nC  # source filament number
        self.nT = self.target.nC  # target point number
        self.nI = self.nS*self.nT  # total number of interactions
        self.points = np.zeros(
                self.nI, dtype=[('rs', float),  # source radius (centroid)
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
                                ('csID', int)])  # cross section ID
        self.assemble_source()
        self.assemble_target()
        self.points['dL'] = np.linalg.norm(
                np.array([self.points['rs'] - self.points['r'],
                          self.points['zs'] - self.points['z']]), axis=0)
            
    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.points['rs'], self.points['zs'], 'C1o', label='source')
        ax.plot(self.points['r'], self.points['z'], 'C2.', label='target')
        plt.legend()
        

class Vectors(object):
    
    mu_o = 4e-7*np.pi  # magnetic constant [Vs/Am]
    
    def __init__(self, points, rms=False, **kwargs):
        self.initialize_delta()
        self.points = points
        self.rms = rms
        self.position(**kwargs)  # initialize source and target points
        
    def initialize_delta(self):
        self.delta = {f'd{var}': 0 for var in ['r', 'rs', 'z', 'zs']}
          
    @property
    def points(self):
        return self._points
    
    @points.setter
    def points(self, points):
        self.nP = len(points)  # interaction number
        self._points = points  # store point subset 
        
    def position(self, **kwargs):
        '(re)position source filaments and target points [dr, drs, dz, dzs]'
        self.rms = kwargs.pop('rms', self.rms)  # update rms flag
        for dvar in self.delta:
            delta = kwargs.pop(dvar, self.delta[dvar])
            var = pvar = dvar[1:]
            if not hasattr(self, var) or not \
                    np.isclose(delta, self.delta[dvar]).all():
                self.delta[dvar] = delta
                if var == 'rs' and self.rms:
                    pvar += '_rms'
                setattr(self, var, self.points[pvar] + self.delta[dvar])
                self.update_flag = True
        
    def update(self):
        if self.update_flag:
            self.gamma = self.zs - self.z
            self.a2 = self.gamma**2 + (self.r + self.rs)**2
            self.a = np.sqrt(self.a2)
            self.k2 = 4 * self.r * self.rs / self.a2  # modulus
            self.ck2 = 1 - self.k2  # complementary modulus
            self.K = ellipk(self.k2)  # first complete elliptic integral
            self.E = ellipe(self.k2)  # second complete elliptic integral 
            self.update_flag = False
        

    
    ''' 
    @property
    def U(self):
        if self._U is None:
            self._U = self.k2 * (4*self.gamma**2 + 3*self.rs**2 - 
                                 5*self.r**2) / (4*self.r)
    '''
    

class Filament(Vectors):
    'complete circular filaments'
    def __init__(self, points, rms=True):
        Vectors.__init__(self, points, rms=rms)
        self.factor = {'circle': np.exp(-0.25),  # circle-circle
                       'square': 2*0.447049,  # square-square
                       'skin': 1}  # skin-skin
        self.csID_lookup = {0: 'square', 1: 'square',
                            2: 'circle', 3: 'circle', 
                            4: 'skin', 
                            5: 'square', 6: 'square'}
        self.offset()
        
    def offset(self):
        'offset source and target points '
        self.dL = np.array([self.r-self.rs, self.z-self.zs])
        self.dL_mag = np.linalg.norm(self.dL, axis=0)
        self.dL_norm = np.zeros((2, self.nP))
        self.index = np.isclose(self.dL_mag, 0)  # self index
        self.dL_norm[0, self.index] = 1  # radial offset
        self.dL_norm[:, ~self.index] = \
            self.dL[:, ~self.index] / self.dL_mag[~self.index]
        idx = self.dL_mag < self.points['dr'] # seperation < L2 norm radius
        ro = self.points['dr'] * np.array([self.factor[self.csID_lookup[csID]] 
                                           for csID in self.points['csID']])
        factor = (1 - self.dL_mag[idx] / self.points['dr'][idx]) / 2
        deltas = {}
        for i, var in enumerate(['r', 'z']):
            offset = np.zeros(self.nP)
            offset[idx] = factor * ro[idx] * self.dL_norm[i][idx]
            deltas.update({f'd{var}': offset, f'd{var}s': -offset})
        self.position(**deltas)
        
    def flux(self):
        'vector and scalar potential'
        self.update()  # update coefficents
        Aphi = 1 / (2*np.pi) * self.a/self.r * \
            ((1 - self.k2/2) * self.K - self.E)  # 
        psi = 2 * np.pi * self.mu_o * self.r * Aphi  # scalar potential
        return psi
    

class Rectangle(Vectors):
    
    def __init__(self, points):
        Vectors.__init__(self, points)
        
    def B(self, phi):
        return np.sqrt(self.rs**2 + self.r**2 - 2*self.r*self.rs*np.cos(phi))
    
    def D(self, phi):
        return np.sqrt(self.gamma**2 + self.B(phi)**2)
    
    def G(self, phi):
        return np.sqrt(self.gamma**2 + self.r**2 * np.sin(phi)**2)
    
    def b1(self, phi):
        'beta 1'
        return (self.rs - self.r*np.cos(phi)) / self.G(phi)
    
    def b2(self, phi):
        'beta 2'
        return self.gamma / self.B(phi)
    
    def b3(self, phi):
        'beta 3'
        return self.gamma * (self.rs - self.r*np.cos(phi)) \
                / (self.r*np.sin(phi)*self.D(phi))
          
    def Jf(self, phi):
        'compute J intergrand'
        f = np.zeros(np.shape(phi))
        for i in range(f.shape[1]):
            f[:, i] = np.arcsinh(self.b1(phi[:, i]))
        return f
        
    def J(self, alpha, index=2):
        scheme = quadpy.line_segment.gauss_patterson(index)
        bounds = np.dot(np.array([[self.phi(0)], [self.phi(alpha)]]), 
                        np.ones((1, self.nI)))
        return scheme.integrate(self.Jf, bounds)
    
    def Cphi(self, alpha):
        return 0.5*self.gamma*self.a * (1 - self.k2*np.sin(alpha)**2)**0.5 *\
                    -np.sin(2*alpha) \
                -1/6 * np.arcsinh(self.b2(alpha)) *\
                    np.sin(2*alpha) * (2*self.r**2*np.sin(2*alpha)**2 + 
                                       3 * (self.rs**2 - self.r**2)) \
                -1/4 * self.gamma*self.r*np.arcsinh(self.b1(alpha)) *\
                    -np.sin(4*alpha) \
                -1/3 * self.r**2*np.arctan(self.b3(alpha)) - np.cos(2*alpha)**3
        
        
    def flux(self):
        'calculate flux for rectangular coil section'
        Aphi = self.Cphi(np.pi/2) + self.gamma*self.r*self.J(np.pi/2) \
                + self.gamma*self.a / (6*self.r) * (self.U*self.K - 
                                                    2*self.rs*self.E)
        for p in range(3):
            Aphi += self.gamma / (6 * self.a * self.r)
        return np.zeros(len(self.r))
        
        
class BiotSavart(CoilMatrix, BiotArray):

    mu_o = 4 * np.pi * 1e-7  # magnetic constant [Vs/Am]
    
    _biotsavart_attributes = {}  
    
    def __init__(self, source=None):
        CoilMatrix.__init__(self)
        BiotArray.__init__(self, source)
        

    '''
    def _extract_data(self, frame):
        data = {}
        for key in ['x', 'z', 'dx', 'dz', 'Nt']:
            data[key] = getattr(frame, key)
        data['ro'] = self.gmr.calculate_self(
                data['dx'], data['dz'], frame.cross_section)
        return data
    '''
    
    # structured array
    #fields; x, z, rms, turn_section,  
            
    '''
    def assemble_source(self):
        self.nT = self.target.nC  # target number
        data = self._extract_data(self.source)
        self.source_m = {}
        for key in data:
            self.source_m[key] = \
                np.dot(np.ones((self.nT, 1)), data[key].reshape(1, -1))
        
    def assemble_target(self):
        self.nS = self.source.nC  # source filament number
        data = self._extract_data(self.target)
        self.target_m = {}
        for key in data:
            self.target_m[key] = \
                np.dot(data[key].reshape(-1, 1), np.ones((1, self.nS)))
                
    def assemble(self):
        self.assemble_source()
        self.assemble_target()
        #self.offset()  # transform turn-trun offset to geometric mean
        
    def offset(self):
        'transform turn-trun offset to geometric mean'
        self.dL = np.array([self.target_m['x'] - self.source_m['x'],
                            self.target_m['z'] - self.source_m['z']])
        self.Ro = np.exp((np.log(self.source_m['x']) +
                          np.log(self.target_m['x'])) / 2)
        self.dL_mag = np.linalg.norm(self.dL, axis=0)
        iszero = np.isclose(self.dL_mag, 0)  # self index
        self.dL_norm = np.zeros((2, self.nT, self.nS))
        self.dL_norm[:, ~iszero] = self.dL[:, ~iszero] / self.dL_mag[~iszero]
        self.dL_norm[0, iszero] = 1
        # self inductance index
        dr = (self.source_m['dx'] + self.source_m['dz']) / 4  # mean radius
        idx = self.dL_mag < dr  # seperation < mean radius
        # mutual inductance offset
        if self.mutual_offset:  # mutual inductance offset
            nx = abs(self.dL_mag / self.source_m['dx'])
            nz = abs(self.dL_mag / self.source_m['dz'])
            mutual_factor = self.gmr.evaluate(nx, nz)
            mutual_adjust = (mutual_factor - 1) / 2
            for i, key in enumerate(['x', 'z']):
                offset = mutual_adjust[~idx] * self.dL[i][~idx]
                self._apply_offset(key, offset, ~idx)
        # self-inductance offset
        factor = (1 - self.dL_mag[idx] / dr[idx]) / 2
        ro = np.max([self.source_m['ro'][idx],
                     self.target_m['ro'][idx]], axis=0)
        for i, key in enumerate(['x', 'z']):
            offset = factor * ro * self.dL_norm[i][idx]
            self._apply_offset(key, offset, idx)

    def _apply_offset(self, key, offset, index):
        if key == 'r':
            Ro_offset = np.exp(
                    (np.log(self.source_m[key][index] - offset) +
                     np.log(self.target_m[key][index] + offset)) / 2)
            shift = self.Ro[index] - Ro_offset  # gmr shift
        else:
            shift = np.zeros(np.shape(offset))
        self.source_m[key][index] -= offset + shift
        self.target_m[key][index] += offset - shift
        return shift

    def locate(self):
        xt, zt = self.target_m['x'], self.target_m['z']
        xs, zs = self.source_m['x'], self.source_m['z']
        return xt, zt, xs, zs
    '''

    def flux_matrix(self, ndr=0):
        'calculate filament flux (inductance) matrix'
        '''
        xt, zt, xs, zs = self.locate()
        m = 4 * xt * xs / ((xt + xs)**2 + (zt - zs)**2)
        flux = np.array((xt * xs)**0.5 * ((2 * m**-0.5 - m**0.5) *
                        ellipk(m) - 2 * m**-0.5 * ellipe(m)))
        flux *= self.mu_o  # unit filaments, Wb/Amp-turn-turn
        '''
        flux = np.zeros(self.nI)  # initalize vector
        
        index = self.points['dL'] > ndr/2 * self.points['dr']
        self.filament = Filament(self.points[index], rms=True)
        
        #self.filament.position(dr=1.5, dz=-0.5)
        #self.filament.update()
        
        flux[index] = self.filament.flux()
        
        flux = flux.reshape(self.nT, self.nS)  # source-target reshape (matrix)
        self.flux , self._flux, self._flux_ = self.save_matrix(flux)
        
    def field_matrix(self):
        'calculate subcoil field matrix'
        xt, zt, xs, zs, = self.locate()
        a = np.sqrt((xt + xs)**2 + (zt - zs)**2)
        m = 4 * xt * xs / a**2
        I1 = 4 / a * ellipk(m)
        I2 = 4 / a**3 * ellipe(m) / (1 - m)
        A = (zt - zs)**2 + xt**2 + xs**2
        B = -2 * xt * xs
        field = {}
        field['x'] = xs / 2 * (zt - zs) / B * (I1 - A * I2)
        field['z'] = xs / 2 * ((xs + xt * A / B) * I2 - xt / B * I1)
        for xz in field:  # save field matricies
            self.field[xz], self._field[xz], self._field_[xz] = \
                self.save_matrix(self.mu_o / (2 * np.pi) * field[xz])  # T / Amp-turn-turn
    
    def solve_interaction(self):
        self.assemble()  # assemble geometory matrices
        self.flux_matrix()  # assemble flux interaction matrix
        #self.field_matrix()  # assemble field interaction matricies 
        
    def save_matrix(self, M):
        # extract plasma unit filaments
        _M_ = M[self.target._plasma_index][:, self.source._plasma_index]  
        # reduce
        M *= self.points['N'].reshape(self.nT, self.nS)  # target turns
        _M = M[:, self.source._plasma_index]  # unit source filament
        M *= self.points['Ns'].reshape(self.nT, self.nS)  # source turns
        #if len(self.target._reduction_index) < self.nT:  # sum sub-target
        #    M = np.add.reduceat(M, self.target._reduction_index, axis=0)
        #    _M = np.add.reduceat(_M, self.target._reduction_index, axis=0)
        if len(self.source._reduction_index) < self.nS:  # sum sub-source
            M = np.add.reduceat(M, self.source._reduction_index, axis=1)
        return M, _M, _M_  # turn-turn interaction, source unit, mutual unit

    def _update_plasma(self, M, _M, _M_):
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
        self._update_plasma(self.flux, self._flux, self._flux_)
        
    def update_field(self):
        for xz in self.field:
            self._update_plasma(self.field[xz], self._field[xz], 
                                self._field_[xz])
        
    @property
    def Psi(self):
        self._Psi = np.dot(self.flux, self.source._Ic)
        if self.n2d != 0:
            self._Psi = self._Psi.reshape(self.n2d)
        return self._Psi
    
    @property
    def Bx(self):
        self._Bx = np.dot(self.field['x'], self.source._Ic)
        if self.n2d != 0:
            self._Bx = self._Bx.reshape(self.n2d)
        return self._Bx
    
    @property
    def Bz(self):
        self._Bz = np.dot(self.field['z'], self.source._Ic)
        if self.n2d != 0:
            self._Bz = self._Bz.reshape(self.n2d)
        return self._Bz


if __name__ == '__main__':
    
    from nova.electromagnetic.coilset import CoilSet
    cs = CoilSet(dCoil=0.2, dPlasma=0.05, turn_fraction=0.5)
    #cs.add_coil(3.943, 7.564, 0.959, 0.984, Nt=248.64, name='PF1', part='PF')
    #cs.add_coil(1.6870, 5.4640, 0.7400, 2.093, Nt=554, name='CS3U', part='CS')
    #cs.add_coil(1.6870, 3.2780, 0.7400, 2.093, Nt=554, name='CS2U', part='CS')
    #cs.add_plasma(3.5, 4.5, 1.5, 2.5, It=-15e6, cross_section='ellipse')
    
    cs.add_plasma(3.5, 4.5, 1.5, 2.5, dPlasma=0.5, 
                  It=-15e6, cross_section='circle')

    cs.current_update = 'coil'
    
    
    plt.set_aspect(1.2)
    
    cs.grid.generate_grid(expand=1, n=5e3)
    #cs.grid.plot_grid()
    
    cs.Ic = -40e3
            
    cs.plot(current='A')
    cs.grid.plot_flux()
    
    cs.add_plasma(3.5, 4.5, 1.5, 2.5, dPlasma=0.05, 
                  It=-15e6, cross_section='circle')
    cs.plot()
    cs.grid.generate_grid(regen=True)
    cs.grid.plot_flux(color='C0')
    
    '''
    
    
    bs = BiotSavart(cs.subcoil)

    bs.load_target(cs.subcoil)
    bs.assemble()
    bs.flux_matrix() 
    bs.plot()
    '''
    
    

    #scheme = quadpy.disk.lether(2)
    #scheme.show()
    #val = scheme.integrate(lambda x: np.exp(x[0]), [0.0, 0.0], 1.0)
    #bs = biot_savart(cs.coilset, mutual=True)

    #bs.colocate(subcoil=True)
    #_B = bs.field_matrix()
    #_Bx = bs.reduce(_B[0])
    
    '''
    bs.colocate(subcoil=False)
    
    B = bs.field_matrix()
    print(B['x'])

    Mc = bs.calculate_inductance()
    '''
    
    #bs.target.plot(label=True)

    # plt.title(cc.coilset.matrix['inductance']['Mc'].CS3U)