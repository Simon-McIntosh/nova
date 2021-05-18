import dolfin as df
import ufl
from amigo.pyplot import plt
from nep.coil_geom import PFgeom
from nova.coil_class import CoilClass
from nova.mesh_grid import MeshGrid
from scipy.interpolate import interp1d
import numpy as np
from nova.coil_frame import CoilFrame
import pandas as pd
from scipy.interpolate import RectBivariateSpline, NearestNDInterpolator
from scipy.optimize import minimize
import mshr


class Material_Model:
    
    _models = ['isotropic', 'transversal', 'orthotropic']
    _modulus_factor = 5e11  # normalization factor for minimise
    
    def __init__(self, material_model=None):
        self._material_data = pd.Series(dtype=float)
        self._fenics_data = {}
        self.material_model = material_model
        
    def _set_attributes(self, key_attributes, derived_attributes):
        self._key_attributes = key_attributes
        self._ismodulus = np.zeros(len(key_attributes), dtype=bool)
        self._active = np.ones(len(key_attributes), dtype=bool)
        self._derived_attributes = derived_attributes
        self._model_attributes = key_attributes + derived_attributes
        
    @property
    def material_model_name(self):
        return self._material_model_name
    
    @material_model_name.setter
    def material_model_name(self, name):
        if len(name) == 1 and isinstance(name, str):
            _m = [m[0] for m in self._models]
            if name in _m:
                name = self._models[_m.index(name)]
            else:
                raise IndexError(f'quick index {name} ' +\
                                 f'not found in _models {self._models}')
        elif name not in self._models and name is not None:
            raise IndexError(f'{name} not found in _models {self._models}')
        self._material_model_name = name
        
    @property
    def material_model(self):
        return self._material_model_name
        
    @material_model.setter
    def material_model(self, material_model):
        self.material_model_name = material_model
        if self.material_model is not None:
            if self.material_model == 'isotropic':  # isotropic coefficients
                '2 free variables'
                self._set_attributes(['E', 'nu'], ['mu', 'lmbda'])
                self.update_derived_attributes = self._update_isotropic
                self.material_data = {'E': 54e9, 'nu': 0.3}  # default values
                self._ismodulus[0] = True
            elif self.material_model == 'transversal':
                '4 free variables'
                self._set_attributes(['Ep', 'Et', 'nu_pp', 'nu_tp'], ['C'])
                self.update_derived_attributes = self._update_transversal
                self.material_data = {
                         'Ep': 32e9,#52e9,#54e9,  # poloidal tensile modulus
                         'Et': 111e9, #130e9,  # toroidal tensile modulus
                         'nu_pp': 0.29, # poloidal-poloidal Poisson's ratio
                         'nu_tp': 0.29}  # toroidal-poloidal Poisson's ratio
                self._ismodulus[:2] = True
            elif self.material_model == 'orthotropic':
                '6 free variables'
                self._set_attributes(['Ex', 'Et', 'Ez', 'Gxz',
                                      'nu_xz', 'nu_tx', 'nu_tz'], 
                                     ['C'])
                self.update_derived_attributes = self._update_orthotropic
                self.material_data = {
                         'Ex': 48.5e9,#51e9,  # x tensile modulus
                         'Et': 120.4e9,#111e9,  # toroidal tensile modulus
                         'Ez': 46.6e9,#53e9,  # x tensile modulus
                         'Gxz': 6.43e9,#13.7e9,
                         'nu_xz': 0.2998,#0.29,  # poloidal Poisson's ratio
                         'nu_tx': 0.1207*120.4/48.5,#0.3,#0.3*112/52,  # Et/Ex * nu_xt
                         'nu_tz': 0.1154*120.4/46.6,#0.3,#0.143,  # toroidal Poisson's ratio
                         }
                self._ismodulus[:4] = True
            else:
                raise NotImplementedError(
                        f'material_model {self.material_model} ' +\
                        'not implemented')
                
    def _update_isotropic(self):
        E, nu = self._material_data.loc[self._key_attributes]
        # lame's parameters
        mu = E/2/(1+nu)
        lmbda = E*nu/(1+nu)/(1-2*nu)
        self.fenics_data = {'mu': df.Constant(mu), 
                            'lmbda': df.Constant(lmbda)}
        
    def _update_transversal(self):
        Ep, Et, nu_pp, nu_tp = self._material_data.loc[self._key_attributes]
        Ex = Ez = Ep
        nu_xz = nu_pp  # poloidal Poisson's ratio
        nu_tx = nu_tz = nu_tp  # toroidal Poisson's ratio
        Gxz = Ex / (2 * (1 + nu_xz))  # polodal shear modulus
        self._update_stiffness(Ex, Et, Ez, Gxz, nu_xz, nu_tx, nu_tz)
        
    def _update_orthotropic(self):
        self._update_stiffness(*self._material_data.loc[self._key_attributes])
         
    def _update_stiffness(self, Ex, Et, Ez, Gxz, nu_xz, nu_tx, nu_tz):
        'build compliance matrix eps = S.sigma [xx, tt, zz, xz]' 
        self.S = np.array([[    1/Ex, -nu_tx/Et, -nu_xz/Ex,     0],
                           [-nu_tx/Et,     1/Et, -nu_tz/Et,     0],
                           [-nu_xz/Ex, -nu_tz/Et,     1/Ez,     0],
                           [       0,        0,          0, 1/Gxz]])
        C = np.linalg.inv(self.S)  # invert to from stiffness matrix
        self.fenics_data = {'C': df.as_matrix(C)}
        
    @property
    def material_data(self):
        return self._material_data
    
    @material_data.setter
    def material_data(self, material_data):
        for attribute in material_data:
            if attribute not in self._key_attributes:
                self._raise_data_error(attribute, 'key')
            self._update_derived_attributes = True
            self._update_internal_work = True  # fenics build flag
            self._material_data[attribute] = material_data[attribute]
            
    @property
    def fenics_data(self):
        if self._update_derived_attributes:
            self.update_derived_attributes()  # update derived attributes
            self._update_derived_attributes = False
        return self._fenics_data
    
    @fenics_data.setter
    def fenics_data(self, material_data):
        for attribute in material_data:
            if attribute not in self._derived_attributes:
                self._raise_data_error(attribute, 'derived')
            self._fenics_data[attribute] = material_data[attribute]
                
    def _raise_data_error(self, attribute, attribute_type):
        _attributes = getattr(self, f'_{attribute_type}_attributes')
        raise IndexError(f'attribute {attribute} incompatable ' +\
                         f'with {self.material_model} ' +\
                         f'material {attribute_type} attributes {_attributes}')
    

class BC(df.SubDomain):
    def __init__(self, limit):
        self.limit = limit
        df.SubDomain.__init__(self)


class Edge(BC):
    def inside(self, x, on_boundary):
        'select coil base outer corner (edge)'
        return df.near(x[0], self.limit[1]) and df.near(x[1], self.limit[2])


class Base(BC):
    def inside(self, x, on_boundary):
        'select coil base'
        return df.near(x[1], self.limit[2]) and on_boundary

  
class JxB(df.UserExpression):
    'apply JxB force to fenics model'
    
    def __init__(self, fx, fz):
        self.fx = fx  # radial force interpolator
        self.fz = fz  # vertical force interpolator
        df.UserExpression.__init__(self)
        
    def eval(self, value, x):
        value[0] = self.fx(*x)
        value[1] = self.fz(*x)
        
    def value_shape(self):
        return (2,)
    
    
class CSmodulue(Material_Model):
    
    def __init__(self, read_txt=False, material_model='isotropic', N=None):
        self.read_txt = read_txt
        self._extract = False
        Material_Model.__init__(self, material_model)
        self.load_coilset()
        self.mesh_structure(N=N)
        
    def load_coilset(self, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        if read_txt:
            #cs = PFgeom(VS=False, dCoil=-1).cs  # load PF coilset
            #cc = CoilClass(**cs.subset(['CS3U']).coilset)  # select single coil
            #cc.coil.Nt = 568  # increase turn number
            #cc.subcoil.Nt = cc.coil.Nt[0] / cc.coil.Nf[0]
            cc = CoilClass(dCoil=-1, 
                           turn_fraction=0.688,#0.688, 
                           mutual_offset=True,
                           turn_section='circle')  # 
            #cc.add_coil(1.682, 0, 0.738, 2.0965, Nt=485,#Nt=544, 
            #            name='CSM1', part='CS')  # RT
            cc.add_coil(1.6870, 0, 0.7405, 2.1, Nt=544,#Nt=544, 
                        name='CSM1', part='CS')  # RT
            #cc.plot()
            cc.solve_interaction()
            cc.grid.generate_grid(expand=0)
            
            cc.save_coilset('CSM1')
        else:
            cc = CoilClass()
            cc.load_coilset('CSM1')
        self.cc = cc
        
    def mesh_structure(self, N=None):
        if N is None:
            N = self.cc.coil.Nt[0]
        self.limit = self.cc.coil.limit(['CSM1'])  # coil bounding box
        self.mesh = df.RectangleMesh(df.Point(self.limit[::2]), 
                                     df.Point(self.limit[1::2]),
                                     *MeshGrid(N, self.limit).n2d)
        '''
        coil = mshr.Rectangle(df.Point(self.limit[0]-10e-3, self.limit[2]), 
                              df.Point(self.limit[1]+10e-3+1.6e-3, 
                                       self.limit[3]))
        wp = mshr.Rectangle(df.Point(self.limit[::2]), 
                            df.Point(self.limit[1::2]))
        insID = mshr.Rectangle(df.Point(self.limit[0]-10e-3, self.limit[2]), 
                               df.Point(self.limit[0], self.limit[3]))
        insOD = mshr.Rectangle(df.Point(self.limit[1], self.limit[2]), 
                               df.Point(self.limit[1]+10e-3, self.limit[3]))
        belt = mshr.Rectangle(df.Point(self.limit[1]+10e-3, self.limit[2]), 
                               df.Point(self.limit[1]+10e-3+1.6e-3, 
                                        self.limit[3]))
        coil.set_subdomain(1, wp)  # winding pack
        coil.set_subdomain(2, insID)  # ID ground insulation
        coil.set_subdomain(3, insOD)  # OD ground insulation
        coil.set_subdomain(4, belt)  # belt
        self.mesh = mshr.generate_mesh(coil, 10)
        '''

        self.x = df.SpatialCoordinate(self.mesh)
        self.dx = df.Measure('dx')
        self._assemble_function_space()
        
    def _assemble_function_space(self):
        self.V = df.VectorFunctionSpace(self.mesh, 'CG', degree=2)
        self.u = df.Function(self.V, name="Displacement")
        self.du = df.TrialFunction(self.V)
        self.u_ = df.TestFunction(self.V)
        self.eps_u_ = self.eps(self.u_)        
        self._assemble_internal_work()
        self.T = df.TensorFunctionSpace(self.mesh, 'CG', degree=2,
                                        shape=(3, 3))
        self.f_sigma = df.Function(self.T, name='Stress')
        self.f_eps = df.Function(self.T, name='Strain')
        
    def _assemble_forcing(self):
        self.l = df.inner(self.f, self.u_) * self.x[0] * self.dx
        
    def _assemble_internal_work(self):
        self.a = df.inner(self.sigma(self.du), 
                          self.eps_u_) * self.x[0] * self.dx
    
    def constrain(self):
        return [df.DirichletBC(self.V.sub(1), df.Constant(0), 
                               Edge(self.limit), method='pointwise')]
                    
    def solve(self):
        if self._update_internal_work:
            self._assemble_internal_work()
            self._update_internal_work = False
        df.solve(self.a == self.l, self.u, bcs=self.constrain())
            
    @property
    def Ic(self):
        return self.cc.Ic
    
    @Ic.setter
    def Ic(self, Ic):
        self.cc.Ic = Ic
        self._update_force_interpolators()
        
    def _update_force_interpolators(self):
        J = self.cc.coil.It / (self.cc.coil.dx * self.cc.coil.dz)
        centers = (self.cc.subcoil.x, self.cc.subcoil.z)
        
        psi_x, psi_z = np.gradient(self.cc.grid.Psi / (2 * np.pi), 
                                   self.cc.grid.dx, self.cc.grid.dz)
        xm = self.cc.grid.x2d
        xm[xm == 0] = 1e-34
        Bx = -psi_z / xm
        Bz = psi_x / xm
        '''
        Bx = RectBivariateSpline(self.cc.grid.x, self.cc.grid.z, Bx).ev(*centers)
        Bz = RectBivariateSpline(self.cc.grid.x, self.cc.grid.z, Bz).ev(*centers)
        self.fx = NearestNDInterpolator(centers, J * Bz)
        self.fz = NearestNDInterpolator(centers, -J * Bx)
        '''
        self.fx = RectBivariateSpline(self.cc.grid.x, self.cc.grid.z, 
                                      J * Bz)
        self.fz = RectBivariateSpline(self.cc.grid.x, self.cc.grid.z,
                                      -J * Bx)
        self.f = JxB(self.fx, self.fz)
        self._assemble_forcing()
        
    def plot_flux(self, ax=None):
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        plt.set_aspect(1.1)
        self.cc.plot(ax=ax)
        self.cc.grid.plot_flux(ax=ax)

    def eps(self, v):  # axisymetric strain tensor
        return df.sym(
                df.as_tensor([[v[0].dx(0), 0, (v[0].dx(1)+v[1].dx(0))/2],
                              [0, v[0]/self.x[0], 0],
                              [(v[0].dx(1)+v[1].dx(0))/2, 0, v[1].dx(1)]]))
                    
    @staticmethod
    def strain2voigt(e):
        'e is a 2nd-order tensor, returns its Voigt vectorial representation'
        return df.as_vector([e[0, 0], e[1, 1], e[2, 2], 2*e[0, 2]])
    
    @staticmethod
    def voigt2stress(s):
        """
        s is a stress-like vector (no 2 factor on last component)
        returns its tensorial representation
        s = [sx, st, sz, sxz]
        """
        return df.as_tensor([[s[0],    0, s[3]],
                             [   0, s[1],    0],
                             [s[3],    0, s[2]]])

    def sigma_isotropic(self, v):
        lmbda = self.fenics_data['lmbda']
        mu = self.fenics_data['mu']
        return lmbda*df.tr(self.eps(v))*df.Identity(3) + 2.0*mu*self.eps(v)

    def sigma_orthotropic(self, v):
        return self.voigt2stress(df.dot(self.fenics_data['C'], 
                                        self.strain2voigt(self.eps(v))))
    
    def sigma(self, v):
        if self.material_model == 'isotropic':
            return self.sigma_isotropic(v)
        elif self.material_model in ['transversal', 'orthotropic']:
            return self.sigma_orthotropic(v)
        
    def vM(self, u):
        i, j = ufl.indices(2)
        s = df.dev(self.sigma(u))  # deviatoric stress 
        return df.sqrt(3 / 2 * s[i, j] * s[j, i])  # von Mises stress
    
    def extract_displacments(self, verbose=True):
        displace = {}
        displace[('ID', 'mm')] = 1e3 * (self.u(self.limit[::3])[1] - 
                                        self.u(self.limit[::2])[1])
        displace[('OD', 'mm')] = 1e3 * (self.u(self.limit[1::2])[1] - 
                                        self.u(self.limit[1:3])[1])
        displace[('U', 'mm')] = 1e3 * (self.u(self.limit[1::2])[0] - 
                                        self.u(self.limit[::3])[0])
        displace[('L', 'mm')] = 1e3 * (self.u(self.limit[1:3])[0] - 
                                        self.u(self.limit[::2])[0])        
        midplane = {'ID': (self.limit[0], np.mean(self.limit[-2:])),
                    'OD': (self.limit[1], np.mean(self.limit[-2:])),
                    'UID': (self.limit[0], self.limit[-1]),
                    'UOD': (self.limit[1], self.limit[-1]),
                    'LID': (self.limit[0], self.limit[-2]),
                    'LOD': (self.limit[1], self.limit[-2])}
        self.f_eps.assign(df.project(self.eps(self.u), self.T))
        self.f_sigma.assign(df.project(self.sigma(self.u), self.T))
        for ID in midplane:
            #sigma_xx = self.f_sigma(midplane[ID])[0]
            #sigma_zz = self.f_sigma(midplane[ID])[8]
            eps_tt = self.f_eps(midplane[ID])[4]  # hoop strain
            displace[(ID, 'ppm')] = 1e6 * eps_tt
        self.displace = pd.Series(displace)
        self.displace.index = pd.MultiIndex.from_tuples(self.displace.index)
        if verbose:
            print(self.displace)
            
    def material_data_text(self, material_data=None, full=False, tex=False):
        if material_data is None:
            material_data = self.material_data
        if isinstance(material_data, pd.Series):
            material_data = material_data.to_numpy()
        attributes = np.array(self._key_attributes)    
        if full:
            _ismodulus = self._ismodulus
            join_str = '\n'
        else:
            _ismodulus = self._ismodulus[self._active]
            material_data = material_data[self._active]
            attributes = attributes[self._active]
            join_str = ', '
        if tex:
            tex_attributes = []
            for i in range(len(attributes)):
                var = attributes[i]
                if var[0] == 'E':
                    var = '_'.join([var[0], var[1:]])
                elif var[:2] == 'nu':
                    vsplit = var.split('_')
                    var = f'\{vsplit[0]}_{{{vsplit[1]}}}'
                if attributes[i] in attributes[self._active] and self._extract:
                   var = rf'$\mathbf{{{var}}}$'
                else:
                   var = rf'${var}$'
                tex_attributes.append(f'{var}')
            attributes = np.array(tex_attributes)
        modulus_text = ', '.join((rf'{attr}={1e-9*M:1.1f}' for 
                          attr, M in zip(attributes[_ismodulus],
                                         material_data[_ismodulus])))
        nu_text = ', '.join((rf'{attr}={nu:1.3f}' for 
                          attr, nu in zip(attributes[~_ismodulus],
                                          material_data[~_ismodulus])))
        if len(modulus_text) == 0:
            join_str = ''
        return join_str.join((modulus_text, nu_text))
    
    def match_shape(self, x, *args):
        _x = x.copy()
        _x[self._ismodulus[self._active]] *= self._modulus_factor
        attributes = np.array(self._key_attributes)[self._active]
        self.material_data = {a: _x[i] for i, a in enumerate(attributes)}
        displace_ct = args  # cold test data
        self.solve()
        self.extract_displacments(verbose=False)
        displace = [self.displace[('ID', 'mm')], self.displace[('OD', 'mm')],
                    self.displace[('ID', 'ppm')], self.displace[('OD', 'ppm')]]
        err = displace_ct - np.array(displace)
        err[2:] /= 1000  # normalize strain error
        print(self.material_data_text(full=False), 
              self.displace.to_numpy(), np.linalg.norm(err))
        return np.linalg.norm(err)
    
    def extract_properties(self):
        xo = self.material_data.to_numpy()[self._active]
        _ismodulus = self._ismodulus[self._active]
        xo[_ismodulus] /= self._modulus_factor  # normalize moduli
        Mlim = 1e9 * np.array([10, 160]) / self._modulus_factor
        Nulim = (0.05, 0.9)
        nM = np.sum(_ismodulus)
        nNu = len(xo) - nM
        #data = (-3.02, -1.77)  # 48.5 kA
        data = (-2.05, -1.20, 838, 457)  # 40kA
        x = minimize(self.match_shape, xo, args=data, #, 323, 455
                     method='SLSQP', options={'ftol': 1e-4},
                     bounds=(*[Mlim for __ in range(nM)], 
                             *[Nulim for __ in range(nNu)]))
        print(x)
        self.match_shape(x.x, *data)  # propogate solution
        self._extract = True
        
    def plot(self, scale=100, full=False, twin=False):
        plt.set_aspect(1.0)
        
        if twin:
            fig, ax = plt.subplots(1, 2, sharey=True)
            plt.sca(ax[0])
            df.plot(self.mesh, alpha=0.2, zorder=50)
            df.plot(self.f, mesh=self.mesh, scale_units='height', 
                    scale=2.5e9, width=0.0075, zorder=60)
            #self.cc.plot(ax=ax[0])
            self.cc.grid.plot_flux(ax=ax[0], lw=1.5, color='gray')
            plt.despine()
            plt.axis('off')
            ax_disp = ax[1]
        else:
            fig, ax_disp = plt.subplots(1, 1)
        plt.sca(ax_disp)
        df.plot(scale*self.u, mode="displacement")
        plt.despine()
        plt.axis('off')
        
        pID = [self.limit[0], np.mean(self.limit[-2:])]
        pID += scale * self.u(*pID)
        pOD = (self.limit[1], np.mean(self.limit[-2:]))
        pOD += scale * self.u(*pOD)
        
        pUID = [self.limit[0], self.limit[-1]]
        pUID += scale * self.u(*pUID)
        pUOD = [self.limit[1], self.limit[-1]]
        pUOD += scale * self.u(*pUOD)
        
        pU = [np.mean(self.limit[:2]), self.limit[-1]]
        pU += scale * self.u(*pU)
        pL = [np.mean(self.limit[:2]), self.limit[-2]]
        pL += scale * self.u(*pL)
        
        ax_disp.text(*pID, f'{csm.displace[("ID", "mm")]:1.2f}mm\n',
                   ha='left', va='bottom')
        ax_disp.text(*pOD, f'\n{csm.displace[("OD", "mm")]:1.2f}mm',
                   ha='right', va='top')
        ax_disp.text(*pU, f'{csm.displace[("U", "mm")]:1.2f}mm',
                   ha='center', va='top', color='C3')  
        #ax_disp.text(*pL, f'{csm.displace[("L", "mm")]:1.2f}mm',
        #           ha='center', va='bottom')  
        ax_disp.text(*pID, f'{csm.displace[("ID", "ppm")]:1.0f}ppm',
                   ha='right', va='center', rotation=90)
        ax_disp.text(*pUID, f'{csm.displace[("UID", "ppm")]:1.0f}ppm',
                   ha='right', va='top', rotation=90, color='C3')
        ax_disp.text(*pUOD, f'{csm.displace[("UOD", "ppm")]:1.0f}ppm',
                   ha='left', va='top', rotation=-90, color='C3')
        ax_disp.text(*pOD, f'{csm.displace[("OD", "ppm")]:1.0f}ppm',
                   ha='left', va='center', rotation=-90)
        fig.suptitle(
                #f'{self.material_model}\n' + \
                f'{self.material_data_text(full=full, tex=True)}', 
                y=0.95, fontsize='small')
    
    
if __name__ == '__main__':
    
    csm = CSmodulue(material_model='t', read_txt=True,
                    N=None)
    
    
    #csm.Ic = 48.5e3# * 1.28
    
    csm.Ic = 40e3
    csm.material_data = {'Ep': 22.5e9}
    
    # t: 'Ep', 'Et', 'nu_pp', 'nu_pt'
    csm._active[:] = False
    csm._active[1] = True
    csm._active[2] = True
    csm._active[3] = True
    #csm._active[-1] = True
    #csm._active[-1:] = True
    
    #csm._active[-2:] = True
    csm.extract_properties()
    
    csm.solve()
    csm.extract_displacments()

    plt.set_context('talk')
    csm.plot(full=True)

    
    
