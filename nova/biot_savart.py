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
        self._setattr(**kwargs)
        self._emulate_coil_index()
        self._check_attribute_length()
        
    def _emulate_nC(self, **kwargs):
        'emulate CoilFrame.nC: calculate maximum element number in kwars' 
        self.nC = np.max([np.size(kwargs[key]) 
                         for key in kwargs if is_list_like(kwargs[key])])
    
    def _emulate_coil_index(self):
        'emulate CoilFrame._coil_index'
        if not hasattr(self, '_coil_index'):
            self._coil_index = np.arange(self.nC)
    
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
                value = np.array([value for __ in range(self.nC)])
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

        
class BiotSavart(CoilMatrix):

    mu_o = 4 * np.pi * 1e-7  # magnetic constant [Vs/Am]
    
    _biot_attributes = {'mutual': True}  # include mutual inductance offset

    def __init__(self, **biot_attributes):
        self._initialize_biot_attributes()  # initialize unset biot attributes
        self.biot_attributes = biot_attributes
        self.gmr = geometric_mean_radius()  # mutual gmr factors
        
    def _initialize_biot_attributes(self):
        for attribute in self._biot_attributes:
            if not hasattr(self, attribute):
                setattr(self, attribute, self._biot_attributes[attribute])
        
    @property
    def biot_attributes(self):
        'extract biot-savart attributes'
        self._biot_attributes = {
                attribute: getattr(self, attribute)
                for attribute in self._biot_attributes}
        return self._biot_attributes
        
    @biot_attributes.setter
    def biot_attributes(self, biot_attributes):
        'set biot-savart attributes'
        for attribute in self._biot_attributes:
            value = biot_attributes.get(attribute, None)
            if value is not None:
                setattr(self, attribute, value)
        CoilMatrix.__init__(self, **biot_attributes)  # initalize biot-savart 

        
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
        if self.mutual:  # mutual inductance offset
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
        '''
        calculate filament flux (inductance) matrix
        '''
        xt, zt, xs, zs = self.locate()
        m = 4 * xt * xs / ((xt + xs)**2 + (zt - zs)**2)
        flux = np.array((xt * xs)**0.5 * ((2 * m**-0.5 - m**0.5) *
                        ellipk(m) - 2 * m**-0.5 * ellipe(m)))
        flux *= self.mu_o  # Wb / Amp-turn-turn
        # flux *= Nt * Nc  # turn-turn interaction, line-current
        return flux

    def field_matrix(self):
        '''
        calculate subcoil field matrix
        '''
        field = np.zeros((2, self.nT, self.nS))
        xt, zt, xs, zs, = self.locate()
        a = np.sqrt((xt + xs)**2 + (zt - zs)**2)
        m = 4 * xt * xs / a**2
        I1 = 4 / a * ellipk(m)
        I2 = 4 / a**3 * ellipe(m) / (1 - m)
        A = (zt - zs)**2 + xt**2 + xs**2
        B = -2 * xt * xs
        # xs / (2 * np.pi)
        field[0] = xs / 2 * (zt - zs) / B * (I1 - A * I2)
        field[1] = xs / 2 * ((xs + xt * A / B) * I2 - xt / B * I1)
        field *= self.mu_o  # T / Amp-turn-turn
        return field
    
    def _reduce(self, matrix):
        matrix *= self.source_m['Nt'] * self.target_m['Nt']  # line-current
        #if len(self.target._coil_index) < self.nT:
        #    matrix = np.add.reduceat(matrix, self.target._coil_index, axis=0)
        if len(self.source._coil_index) < self.nS:
            matrix = np.add.reduceat(matrix, self.source._coil_index, axis=1)
        return matrix
    
    def reduce(self):
        self.flux = self._reduce(self._flux)
        for i in range(2):
            self.field[i] = self._reduce(self._field[i])

    def solve_interaction(self):
        self.assemble()
        self._flux = self.flux_matrix()
        self._field = self.field_matrix()
        self.reduce()
        
    @property
    def Psi(self):
        return np.dot(self.flux, self.source._Ic)
        
    #def update_interaction(self):
  
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