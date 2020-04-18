from nova.coil_frame import CoilFrame
from nova.coil_matrix import CoilMatrix
from scipy.special import ellipk, ellipe
from nova.inductance.geometric_mean_radius import geometric_mean_radius
import numpy as np
from scipy.optimize import minimize_scalar
from pandas.api.types import is_list_like
from pandas import Series

class BiotFrame:
    
    _frame_attributes = ['x', 'z', 'dx', 'dz', 'Nt', 'cross_section']
    _default_attributes = {'dx': 0, 'dz': 0, 'Nt': 1, 
                           'cross_section': 'circle'}
    
    def __init__(self, dx=0, dz=0, cross_section='circle', **kwargs):
        self.load_data(**kwargs)
        
    def load_data(self, **kwargs):
        self._emulate_nC(**kwargs)
        self._emulate_plasma()
        self._setattr(**kwargs)
        self._emulate_coil_index()
        self._check_attribute_length()
        
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
            elif key in self._default_attributes:
                value = self._default_attributes[key]
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
    _attributes = []
    _default_attributes = {}
    
    def __init__(self, **attributes):
        self._attributes += self._biot_attributes
        self._attributes += self._coilmatrix_attributes
        self._default_attributes = {**self._default_attributes, 
                                    **self._biot_attributes}
        self.attributes = attributes
    
    @property
    def attributes(self):
        return {attribute: getattr(self, attribute) for attribute in 
                self._attributes}
        
    @attributes.setter
    def attributes(self, _attributes):
        for attribute in self._attributes:
            default = self._default_attributes.get(attribute, None)
            value = _attributes.get(attribute, None)
            if value is not None:
                setattr(self, attribute, value)  # set value 
            elif not hasattr(self, attribute):
                setattr(self, attribute, default)  # set default
                
        
class BiotSavart(CoilMatrix):

    mu_o = 4 * np.pi * 1e-7  # magnetic constant [Vs/Am]
    
    _biot_attributes = {'mutual_offset': True}  # include mutual inductance offset

    def __init__(self):
        self.gmr = geometric_mean_radius()  # load mutual gmr factors
        CoilMatrix.__init__(self)
        
    def _load_frame(self, *args, **kwargs):
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
        
    def load_source(self, *args, **kwargs):
        self.source = self._load_frame(*args, **kwargs)
        
    def load_target(self, *args, **kwargs):
        self.target = self._load_frame(*args, **kwargs)
        if hasattr(self.target, 'n2d'):
            self.n2d = self.target.n2d  # target shape
        else:
            self.n2d = self.target.nC

    def _extract_data(self, frame):
        data = {}
        for key in ['x', 'z', 'dx', 'dz', 'Nt']:
            data[key] = getattr(frame, key)
        data['ro'] = self.gmr.calculate_self(
                data['dx'], data['dz'], frame.cross_section)
        return data
            
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
        self.offset()  # transform turn-trun offset to geometric mean
        
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
        self.dL_norm[1, iszero] = 1
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

    def flux_matrix(self):
        'calculate filament flux (inductance) matrix'
        xt, zt, xs, zs = self.locate()
        m = 4 * xt * xs / ((xt + xs)**2 + (zt - zs)**2)
        flux = np.array((xt * xs)**0.5 * ((2 * m**-0.5 - m**0.5) *
                        ellipk(m) - 2 * m**-0.5 * ellipe(m)))
        flux *= self.mu_o  # unit filaments, Wb/Amp-turn-turn
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
                self.save_matrix(self.mu_o * field[xz])  # T / Amp-turn-turn
    
    def solve_interaction(self):
        self.assemble()  # assemble geometory matrices
        self.flux_matrix()  # assemble flux interaction matrix
        self.field_matrix()  # assemble field interaction matricies 
        
    def save_matrix(self, M):
        # extract plasma unit filaments
        _M_ = M[self.target._plasma_index][:, self.source._plasma_index]  
        # reduce
        M *= self.target_m['Nt']  # target turns
        _M = M[:, self.source._plasma_index]  # unit source filament
        M *= self.source_m['Nt']  # source turns
        if len(self.target._reduction_index) < self.nT:  # sum sub-target
            M = np.add.reduceat(M, self.target._reduction_index, axis=0)
            _M = np.add.reduceat(_M, self.target._reduction_index, axis=0)
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
        #if self.source._update_biotsavart:
        self._Psi = np.dot(self.flux, self.source._Ic).reshape(self.n2d)
        #    self.source._update_biotsavart = False
        return self._Psi

    """
    from simulation data
    def update_interaction(self, coil_index=None, **kwargs):
        self.generate_grid(**kwargs)  # add | append data targets
        self.add_targets(**kwargs)  # re-generate grid on demand
        if coil_index is not None:  # full update
            self.grid['update'] = True and self.grid['n'] > 0
            self.target['update'] = True
            self.target['targets']['update'] = True
        update_targets = self.grid['update'] or self.target['update']
        if update_targets or coil_index is not None:
            if coil_index is None:
                coilset = self.coilset  # full coilset
            else:
                coilset = self.subset(coil_index)  # extract subset
            bs = biot_savart(source=coilset, mutual=False)  # load coilset
            if self.grid['update'] and self.grid['n'] > 0:
                bs.load_target(self.grid['x2d'].flatten(),
                               self.grid['z2d'].flatten(),
                               label='G', delim='', part='grid')
                self.grid['update'] = False  # reset update status
            if self.target['update']:
                update = self.target['targets']['update']  # new points only
                targets = self.target['targets'].loc[update, :]  # subset
                bs.load_target(targets['x'], targets['z'], name=targets.index,
                               part='target')
                self.target['targets'].loc[update, 'update'] = False
                self.target['update'] = False
            M = bs.calculate_interaction()
            for matrix in M:
                if self.interaction[matrix].empty:
                    self.interaction[matrix] = M[matrix]
                elif coil_index is None:
                    drop = self.interaction[matrix].index.unique(level=1)
                    for part in M[matrix].index.unique(level=1):
                        if part in drop:  # clear prior to concat
                            if part == 'target':
                                self.interaction[matrix].drop(
                                        points.index, level=0,
                                        inplace=True, errors='ignore')
                            else:
                                self.interaction[matrix].drop(
                                        part, level=1, inplace=True)
                    self.interaction[matrix] = concat(
                            [self.interaction[matrix], M[matrix]])
                else:  # selective coil_index overwrite
                    for name in coilset.coil.index:
                        self.interaction[matrix].loc[:, name] = \
                            M[matrix].loc[:, name]
                            
    def solve_interaction(self, plot=False, color='gray', *args, **kwargs):
        'generate grid / target interaction matrices'
        self.update_interaction(**kwargs)  # update on demand
        for matrix in self.interaction:  # Psi, Bx, Bz
            if not self.interaction[matrix].empty:
                # variable = matrix.lower()
                #index = self.interaction[matrix].index
                #value = np.dot(
                #        self.interaction[matrix].loc[:, self.coil.data.index],
                #        self.coil.data.Ic)
                #value = self.interaction[matrix].dot(self.Ic)
                value = np.dot(self.interaction[matrix].to_numpy(), self.Ic)
                #coil = DataFrame(value, index=index)  # grid, target
                '''
                for part in coil.index.unique(level=1):
                    part_data = coil.xs(part, level=1)
                    part_dict = getattr(self, part)
                    if 'n2d' in part_dict:  # reshape data to n2d
                        part_data = part_data.to_numpy()
                        part_data = part_data.reshape(part_dict['n2d'])
                        part_dict[matrix] = part_data
                    else:
                        part_data = concat(
                                (Series({'t': self.t}), part_data),
                                sort=False)
                        part_dict[matrix] = concat(
                                (part_dict[matrix], part_data.T),
                                ignore_index=True, sort=False)
                '''
        if plot and self.grid['n'] > 0:
            if self.grid['levels'] is None:
                levels = self.grid['nlevels']
            else:
                levels = self.grid['levels']
            QuadContourSet = plt.contour(
                    self.grid['x2d'], self.grid['z2d'], self.grid['Psi'],
                    levels, colors=color, linestyles='-', linewidths=1.0,
                    alpha=0.5, zorder=5)
            self.grid['levels'] = QuadContourSet.levels
            plt.axis('equal')
            #plt.quiver(self.grid['x2d'], self.grid['z2d'], 
            #           self.grid['Bx'], self.grid['Bz'])
    """
    
    
    
    '''
    def index_part(self, M):
        M.loc[:, 'part'] = self.target.coil['part']
        M.set_index('part', append=True, inplace=True)
        return M
    
    def column_reduce(self, Mo):
        Mo = pd.DataFrame(Mo, index=self.target.subcoil.index,
                          columns=self.source.subcoil.index, dtype=float)
        Mcol = pd.DataFrame(index=self.target.subcoil.index,
                            columns=self.source.coil.index, dtype=float)
        for name in self.source.coil.index:  # column reduction
            index = self.source.coil.subindex[name]
            Mcol.loc[:, name] = Mo.loc[:, index].sum(axis=1)
        return Mcol

    def row_reduce(self, Mcol):
        Mrow = pd.DataFrame(columns=self.source.coil.index, dtype=float)
        if 'subindex' in self.target.coil.columns:
            #part = self.target.coil['part']
            for name in self.target.coil.index:  # row reduction
                index = self.target.coil.subindex[name]
                Mrow.loc[name, :] = Mcol.loc[index, :].sum(axis=0)
        else:
            #part = self.target.subcoil['part']
            Mrow = Mcol
        #Mrow['part'] = part
        #Mrow.set_index('part', append=True, inplace=True)
        return Mrow

    def calculate_inductance(self):
        self.colocate()  # set targets
        Mc = self.row_reduce(self.flux_matrix())  # line-current
        return Mc

    def calculate_interaction(self):
        self.assemble(offset=True)  # build interaction matrices
        M = {}
        M['Psi'] = self.flux_matrix()  # line-current interaction
        return M
    '''


class self_inductance:
    '''
    self-inductance methods for a single turn circular coil
    '''
    def __init__(self, x, cross_section='circle'):
        self.x = x  # coil major radius
        self.cross_section = cross_section  # coil cross_section
        self.cross_section_factor = \
            geometric_mean_radius.gmr_factor[self.cross_section]

    def minor_radius(self, L, bounds=(0, 1)):
        '''
        inverse method, solve coil minor radius for given inductance

        Attributes:
            L (float): target inductance Wb
            bounds (tuple of floats): bounds fraction of major radius

        Returns:
            dr (float): coil minor radius
        '''
        self.Lo = L
        r = minimize_scalar(self.flux_err, method='bounded',
                            bounds=bounds, args=(self.Lo),
                            options={'xatol': 1e-12}).x
        gmr = self.x * r
        dr = gmr / self.cross_section_factor
        return dr

    def flux_err(self, r, *args):
        gmr = r * self.x
        L_target = args[0]
        L = self.flux(gmr)
        return (L-L_target)**2

    def flux(self, gmr):
        '''
        calculate self-induced flux though a single-turn coil

        Attributes:
            a (float): coil major radius
            gmr (float): coil cross-section geometric mean radius

        Retuns:
            L (float): self inductance of coil
        '''
        if self.x > 0:
            L = self.x * ((1 + 3 * gmr**2 / (16 * self.x**2)) *
                          np.log(8 * self.x / gmr) -
                          (2 + gmr**2 / (16 * self.x**2)))
        else:
            L = 0
        return biot_savart.mu_o * L  # Wb


if __name__ == '__main__':
    
    from nova.coil_set import CoilSet
    cs = CoilSet(dCoil=-1, turn_fraction=0.7, mutual=False)
    cs.add_coil(3.943, 7.564, 0.959, 0.984, Nt=248.64, name='PF1', part='PF')
    cs.add_coil(1.6870, 5.4640, 0.7400, 2.093, Nt=554, name='CS3U', part='CS',
                cross_section='circle')
    cs.add_coil(1.6870, 3.2780, 0.7400, 2.093, Nt=554, name='CS2U', part='CS')
    cs.add_plasma(5, 2.5, 1.5, 1.5, It=5e6, cross_section='circle')

    print(cs.biot_attributes)
    #bs = BiotSavart()
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