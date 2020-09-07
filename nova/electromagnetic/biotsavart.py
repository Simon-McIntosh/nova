import numpy as np
from pandas.api.types import is_list_like
from pandas import Series

from nova.electromagnetic.coilframe import CoilFrame
from nova.electromagnetic.coilmatrix import CoilMatrix
from nova.electromagnetic.biotelements import Points, BiotPoints
from amigo.pyplot import plt


class BiotFrame(CoilFrame):
    
    def __init__(self, *args, **kwargs):
        CoilFrame.__init__(self, coilframe_metadata={
            '_required_columns': ['x', 'z'],
            '_additional_columns': ['rms', 'dx', 'dz', 'Nt', 'cross_section'],
            '_default_attributes': {'dx': 0, 'dz': 0, 'Nt': 1, 
                                    'cross_section': 'circle'},
            '_coilframe_attributes': ['x', 'z', 'dx', 'dz', 'Nt'],
            '_coildata_attributes': {'region': None, 'nS': None, 'nT': None}})
        self.add_coil(*args, **kwargs)
        
    def add_coil(self, *args, **kwargs):
        self.link_coilframe(*args)  # store referance to CoilFrame
        CoilFrame.add_coil(self, *args, **kwargs)
        
    def link_coilframe(self, *args):
        'set link to coilframe instance to permit future coilframe updates'
        self.coilframe = None
        if self._is_coilframe(*args, accept_dataframe=False):
            self.coilframe = args[0]

    def update_coilframe(self, force_update=False):
        if self.coilframe is not None:
            if self.coilframe.nC != self.nC or force_update:
                self.drop_coil() 
                CoilFrame.add_coil(self, self.coilframe)
        
    @property
    def region(self):
        'source / target region - implicit - set via self.nT or self.nS'
        return self._region
        
    @property
    def nS(self):
        'source filament number'
        return self._nS
    
    @nS.setter
    def nS(self, value):
        'set source filament number for target region'
        self._region = 'target'
        self._nT = self.nC
        self._nS = value 
        
    @property
    def nT(self):
        'target filament number'
        return self._nT
    
    @nT.setter
    def nT(self, value):
        'set target filament number for source region'
        self._region = 'source'
        self._nS = self.nC
        self._nT = value        
        
    def __getattr__(self, key):
        'subclass coilframe getattr'
        if key in self._coilframe_attributes:
            # get coilframe vector
            value = getattr(self, f'_{key}')
            if key in self._mpc_attributes:  # inflate
                value = value[self._mpc_referance]
            if self.nS is None or self.nT is None or self.region is None:
                err_txt = 'complementary source (self.nS) or target (self.nT) '
                err_txt += 'number not set'
                raise IndexError(err_txt)
            if self.region == 'source':  # assemble source
                value = np.dot(np.ones((self.nT, 1)), 
                               value.reshape(1, -1)).flatten()
            elif self.region == 'target':  # assemble target
                value = np.dot(value.reshape(-1, 1), 
                               np.ones((1, self.nS))).flatten()
            return value
        else:
            return CoilFrame.__getattr__(self, key)
        

class BiotAttributes:
    
    'manage attributes to and from Biot derived classes'
    _biot_attributes = []
    _default_biot_attributes = {}
    
    def __init__(self, **biot_attributes):
        self._append_biot_attributes(self._biotsavart_attributes)
        self._append_biot_attributes(self._coilmatrix_attributes)
        self._default_biot_attributes = {**self._default_biot_attributes, 
                                         **self._biotsavart_attributes}
        self.biot_attributes = biot_attributes
        
    def _append_biot_attributes(self, attributes):
        self._biot_attributes += [attr for attr in attributes 
                                  if attr not in self._biot_attributes]
    
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
                
                
class BiotArray(Points):
    
    def __init__(self, source=None):
        if source is not None:
            self.load_source(source)
            
    def load_source(self, *args, **kwargs):
        self.source = BiotFrame(*args, **kwargs)
        
    def load_target(self, *args, **kwargs):
        self.target = BiotFrame(*args, **kwargs)
  
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

    def assemble(self):
        'assemble interaction'
        self.nS = self.source.nC  # source filament number
        self.nT = self.target.nC  # target point number
        self.nI = self.nS*self.nT  # total number of interactions
        
        
        #self.initialize_point_array(self.nI)
        #self.assemble_source()
        #self.assemble_target()
        self.set_biot_instance()
        
    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.points['rs'], self.points['zs'], 
                'C1o', label='source')
        ax.plot(self.points['r'], self.points['z'], 
                'C2.', label='target')
        plt.legend()
    
        
class BiotSavart(CoilMatrix, BiotArray, BiotPoints):
    
    _biotsavart_attributes = {'_solve_interaction': True}
    
    def __init__(self, source=None, ndr=3, mutual=False):
        CoilMatrix.__init__(self)
        BiotArray.__init__(self, source)
        #BiotPoints.__init__(self, ndr=ndr)
        #self.mutual = mutual
        
    def flux_matrix(self):
        'calculate filament flux (inductance) matrix'
        flux = self.calculate('flux')
        self.flux , self._flux, self._flux_ = self.save_matrix(flux)
        
    def field_matrix(self):
        'calculate subcoil field matrix'
        field = {}
        field['x'] = self.calculate('radial_field')
        field['z'] = self.calculate('vertical_field')
        for xz in field:  # save field matricies
            self.field[xz], self._field[xz], self._field_[xz] = \
                self.save_matrix(field[xz])  # T / Amp-turn-turn

    def solve_interaction(self):
        self.assemble()  # assemble geometory matrices
        self.flux_matrix()  # assemble flux interaction matrix
        self.field_matrix()  # assemble field interaction matricies 
        self._solve_interaction = False
        
    def save_matrix(self, M):
        # extract plasma unit filaments
        _M_ = M[self.target._plasma_index][:, self.source._plasma_index]  
        # reduce
        if self.mutual:
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
            
    def _reshape(self, M):
        if hasattr(self, 'n2d'):
            M = M.reshape(self.n2d)
        return M
    
    def _dot(self, variable):
        if self._solve_interaction:
            self.solve_interaction()
        if variable == 'Psi':
            matrix = self.flux 
        elif variable in ['Bx', 'Bz']:
            matrix = self.field[variable[-1]]
        else:
            raise IndexError(f'variable {variable} not in [Psi, Bx, Bz]')
        return self._reshape(np.dot(matrix, self.source._Ic))

    @property
    def Psi(self):
        return self._dot('Psi')
    
    @property
    def Bx(self):
        return self._dot('Bx')
    
    @property
    def Bz(self):
        return self._dot('Bz')

if __name__ == '__main__':
    
    from nova.electromagnetic.coilset import CoilSet
    cs = CoilSet(dCoil=0.2, dPlasma=0.05, turn_fraction=0.5)
    cs.add_coil(3.943, 7.564, 0.959, 0.984, Nt=248.64, name='PF1', part='PF')
    #cs.add_coil(1.6870, 5.4640, 0.7400, 2.093, Nt=554, name='CS3U', part='CS')
    #cs.add_coil(1.6870, 3.2780, 0.7400, 2.093, Nt=554, name='CS2U', part='CS')
    #cs.add_plasma(3.5, 4.5, 1.5, 2.5, It=-15e6, cross_section='ellipse')
    
    cs.add_plasma(3.5, 4.5, 1.5, 2.5, dPlasma=0.5, 
                  It=-15e6, cross_section='circle')

    cs.plot()
    """
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
    """