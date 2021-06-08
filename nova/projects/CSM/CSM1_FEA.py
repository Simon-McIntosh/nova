
import itertools
import sys

import numpy as np
import pandas as pd
import pandas
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize

import dolfin as df
import ufl

from nova.electromagnetic.biotgrid import BiotGrid, Grid
from nova.electromagnetic.coilset import CoilSet
from nova.utilities.pyplot import plt
#from nova.mesh_grid import MeshGrid


class Material_Model:
    
    _models = ['isotropic', 'transversal', 'orthotropic']
    _modulus_factor = 5e10  # normalization factor for minimise
    
    def __init__(self, material_model=None):
        self._material_data = pd.Series(dtype=float)
        self._fenics_data = {}
        self.material_model = material_model
        
    def set_data(self):
        """Set experimental data."""
        self.data = dict(IDdv=-2.05, ODdv=-1.19, 
                         IDh0=788, ODh0=437)  # 40kA CSM1
        '''
        self.data = dict(IDdv=-2.04, ODdv=-1.15,
                         IDr=1.31/2, ODr=1.22/2, 
                         IDh0=896, IDh1=832, IDh2=604, 
                         ODh0=533, ODh1=497, ODh2=390, 
                         IDv0=-1025, ODv0=-888) # 40kA CSM2 
        '''
        
    def set_labels(self, labels: list[str]):
        """Set optimization labels."""
        if labels is None:
            labels = []
        self.labels = labels
        
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
                         'nu_pp': 0.4,#0.29, # poloidal-poloidal Poisson's ratio
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
                         'Ez': 46.6e9,#53e9,  # z tensile modulus
                         'Gxz': 13.7e9,#6.43e9,#13.7e9,
                         'nu_xz': 0.3,#0.2998,#0.29,  # poloidal Poisson's ratio
                         'nu_tx': 0.3,#0.1207*120.4/48.5,#0.3,#0.3*112/52,  # Et/Ex * nu_xt
                         'nu_tz': 0.3,#0.1154*120.4/46.6,#0.3,#0.143,  # toroidal Poisson's ratio
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
        #self.fenics_data = {'C': df.as_matrix(C)}
        self.fenics_data = {'C': C}
        
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
    
    def __init__(self, read_txt=False, material_model='isotropic', num=None,
                 labels=None):
        self.read_txt = read_txt
        self._extract = False
        Material_Model.__init__(self, material_model)
        self.load_coilset()
        self.mesh_structure(num=num) 
        self.set_data()
        self.set_labels(labels)
        self.iter = itertools.count(0)

    def load_coilset(self, **kwargs):
        # read_txt = kwargs.get('read_txt', self.read_txt)

        coilset = CoilSet(dcoil=0)

        coilset.coil.insert(1.6870, 0, 0.7405, 2.1, nturn=544, scale=0.688, 
                            section='rectangle', name='CSM1', part='cs')  # RT
        biotgrid = BiotGrid(*coilset.frames)
        biotgrid.solve(1e3, 0)
        self.coilset = coilset
        self.biotgrid = biotgrid
        
    def mesh_structure(self, num=None):
        bounds = self.coilset.frame.poly[0].bounds  # coil bounding box
        self.limit = bounds[::2] + bounds[1::2]
        if num is None:
            num = self.coilset.frame.nturn[0]
            grid = Grid(number=num, limit=self.limit)
            num = (grid.data.dims['x'] + 1) * (grid.data.dims['z'] + 1)
        grid = Grid(number=num, limit=self.limit)
        self.mesh = df.RectangleMesh(df.Point(bounds[:2]),
                                     df.Point(bounds[2:]),
                                     grid.data.dims['x'], grid.data.dims['z'])
        
        self.x = df.SpatialCoordinate(self.mesh)
        self.dx = df.Measure('dx')
        self._assemble_function_space()
        
    def _assemble_function_space(self):
        self.V = df.VectorFunctionSpace(self.mesh, 'CG', degree=2)
        self.u = df.Function(self.V, name="Displacement")
        self.du = df.TrialFunction(self.V)
        self.u_ = df.TestFunction(self.V)
        self.eps_u_ = self.eps(self.u_)   
        self.C = df.Constant(self.fenics_data['C'])
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
                    
    def solve(self, log_level=30):
        df.set_log_level(log_level)
        if self._update_internal_work:
            self.C.assign(df.Constant(self.fenics_data['C']))
            self._assemble_internal_work()
            self._update_internal_work = False
        df.solve(self.a == self.l, self.u, bcs=self.constrain())
            
    @property
    def Ic(self):
        return self.coilset.sloc['Ic']
    
    @Ic.setter
    def Ic(self, Ic):
        self.coilset.sloc['Ic'] = Ic
        self._update_force_interpolators()
        
    def _update_force_interpolators(self):
        J = self.coilset.subframe.It.sum() / (self.coilset.frame.dx[0] * 
                                     self.coilset.frame.dz[0])

        Psi = np.dot(self.biotgrid.data.Psi, self.coilset.sloc['Ic'])
        Psi.shape = self.biotgrid.shape
        psi_x, psi_z = np.gradient(Psi / (2 * np.pi), 
                                   self.biotgrid.data.x, self.biotgrid.data.z)
        xm = self.biotgrid.data.x2d.values
        xm[xm == 0] = 1e-34
        Bx = -psi_z / xm
        Bz = psi_x / xm
        self.fx = RectBivariateSpline(self.biotgrid.data.x, 
                                      self.biotgrid.data.z, 
                                      J * Bz)
        self.fz = RectBivariateSpline(self.biotgrid.data.x, 
                                      self.biotgrid.data.z,
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
        return self.voigt2stress(df.dot(self.C, 
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
        midplane = np.mean(self.limit[-2:])
        turn_height = (self.limit[-1] - self.limit[-2]) / 40
        self.position = dict(IDdv=(self.limit[::3], self.limit[::2]),
                             ODdv=(self.limit[1::2], self.limit[1:3]),
                             # Udr=(self.limit[1::2], self.limit[::3]),
                             IDr=self.limit[::3], 
                             ODr=self.limit[1::2],
                             IDh0=(self.limit[0], midplane),
                             IDh1=(self.limit[0], midplane + 10*turn_height),
                             IDh2=(self.limit[0], midplane + 18*turn_height),
                             ODh0=(self.limit[1], midplane),
                             ODh1=(self.limit[1], midplane + 10*turn_height),
                             ODh2=(self.limit[1], midplane + 18*turn_height),
                             IDv0=(self.limit[0], midplane + 0.5*turn_height),
                             ODv0=(self.limit[1], midplane + 0.5*turn_height))
                
        displace = {}
        displace[('IDdv', 'mm')] = 1e3 * (self.u(self.position['IDdv'][0])[1] - 
                                         self.u(self.position['IDdv'][1])[1])
        displace[('ODdv', 'mm')] = 1e3 * (self.u(self.position['ODdv'][0])[1] - 
                                         self.u(self.position['ODdv'][1])[1])
        # displace[('Udr', 'mm')] = 1e3 * (
        #     self.u(self.position['Udr'][0])[0] - 
        #     self.u(self.position['Udr'][1])[0])
        displace[('IDr', 'mm')] = 1e3 * self.u(self.position['IDr'])[0]
        displace[('ODr', 'mm')] = 1e3 * self.u(self.position['ODr'])[0]
        
        self.f_eps.assign(df.project(self.eps(self.u), self.T))
        self.f_sigma.assign(df.project(self.sigma(self.u), self.T))
        # extract data
        for label in ['IDh0', 'IDh1', 'IDh2', 'ODh0', 'ODh1', 'ODh2']:
            eps_tt = self.f_eps(self.position[label])[4]  # hoop strain
            displace[(label, 'ppm')] = 1e6 * eps_tt
        for label in ['IDv0', 'ODv0']:
            eps_zz = self.f_eps(self.position[label])[8]  # vertical strain
            displace[(label, 'ppm')] = 1e6 * eps_zz
        self.displace = pd.Series(displace)
        self.displace.index = pd.MultiIndex.from_tuples(self.displace.index)
        if verbose:
            displace = self.displace.reset_index(level=1, name='FEA')
            displace.rename(columns=dict(level_1='unit'), inplace=True)
            
            dataframe = pandas.DataFrame(
                index=displace.index,
                columns=['unit', 'value', 'FEA', 
                         'fit', 'error %'])
            dataframe.loc[:, displace.columns] = displace
            dataframe.loc[self.data.keys(), 'value'] = self.data
            dataframe['fit'] = [name in self.labels 
                                for name in dataframe.index]
            dataframe['error %'] = 1e2 * (dataframe.FEA - 
                                        dataframe.value) / dataframe.value
            self.dataframe = dataframe
            if dataframe.fit.any():
                print('')
                print(dataframe)
                print('')
                print(dataframe.loc[dataframe['fit'], 'error %'].abs().mean())
                print(dataframe['error %'].abs().mean())
            
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
    
    def match_shape(self, x, *args, verbose=True):
        _x = x.copy()
        _x[self._ismodulus[self._active]] *= self._modulus_factor
        attributes = np.array(self._key_attributes)[self._active]
        self.material_data = {a: _x[i] for i, a in enumerate(attributes)}
        displace_ct = np.array(args[0])  # cold test data
        self.solve()
        self.extract_displacments(verbose=False)
        displace = self.displace.droplevel(1)
        displace = displace.loc[self.labels].to_numpy()
        err = 1e2 * (displace - displace_ct) / displace_ct

        if verbose:
            txt = f'\r{next(self.iter)} {self.material_data_text(full=False)} '
            txt += f'{np.mean(np.abs(err)):1.3f}\t\t\t\t\t'
            sys.stdout.write(txt)
            sys.stdout.flush()
        return np.mean(np.abs(err))
    
    def extract_properties(self, labels=None):
        if labels is not None:
            self.set_labels(labels)
        xo = self.material_data.to_numpy()[self._active]
        _ismodulus = self._ismodulus[self._active]
        xo[_ismodulus] /= self._modulus_factor  # normalize moduli

        Mlim = 1e9 * np.array([0.05, 180]) / self._modulus_factor
        Nulim = (-1+1e-3, 1.5)
        Nulim = (0.05, 0.35)
        nM = np.sum(_ismodulus)
        nNu = len(xo) - nM
        #data = (-3.02, -1.77)  # 48.5 kA
        #data = (-2.05, -1.20, -0.34, 838, 457)  # 40kA
        data = [self.data[label] for label in self.labels]
        self.iter = itertools.count(0)
        x = minimize(self.match_shape, xo, args=data, #, 323, 455
                     method='SLSQP', options={'ftol': 1e-5},  # 1e-4
                     bounds=(*[Mlim for __ in range(nM)], 
                             *[Nulim for __ in range(nNu)]))
        print(x)
        self.match_shape(x.x, data)  # propogate solution
        self._extract = True
        
    def plot(self, scale=100, full=False, twin=False):
        plt.set_aspect(0.9)
        
        if twin:
            fig, ax = plt.subplots(1, 2, sharey=True)
            plt.sca(ax[0])
            df.plot(self.mesh, alpha=0.2, zorder=50)
            df.plot(self.f, mesh=self.mesh, scale_units='height', 
                    scale=2.5e9, width=0.01, zorder=60)
            #self.cc.plot(ax=ax[0])
            self.biotgrid.plot(axes=ax[0], colors='gray', linewidths=1.5, 
                               zorder=-50)
            plt.despine()
            plt.axis('off')
            ax_disp = ax[1]
        else:
            fig, ax_disp = plt.subplots(1, 1)
        plt.sca(ax_disp)
        df.plot(scale*self.u, mode="displacement", edgecolor='lightgray',
                linewidth=1.5, vmin=0, vmax=0, cmap='gray_r')
        plt.despine()
        plt.axis('off')
        

        color_index = itertools.count(0)      
        self.extract_displacments()
        for label in ['IDdv', 'ODdv', 'Udr']:
            color = f'C{next(color_index)%10}'
            if label not in self.labels:
                continue
            coord = np.zeros((2, 2))
            for i in range(2):
                coord[:, i] = self.position[label][i]
                coord[:, i] += scale*self.u(*coord[:, i])
            label = f'{label} {self.dataframe.loc[label, "error %"]:1.2f}%'
            marker = '>:' if label == 'Udr' else '^:'
            ax_disp.plot(*coord, marker, ms=6, mew=4, label=label, color=color)

        for label in ['IDr', 'ODr']:
            color = f'C{next(color_index)%10}'
            if label not in self.labels:
                continue
            coord = self.position[label]
            coord += scale*self.u(*coord)
            label = f'{label} {self.dataframe.loc[label, "error %"]:1.2f}%'
            ax_disp.plot(*coord, '>', ms=6, mew=4, label=label, color=color)
        for label in ['IDh0', 'IDh1', 'IDh2', 'ODh0', 'ODh1', 'ODh2']:
            color = f'C{next(color_index)%10}'
            if label not in self.labels:
                continue
            coord = self.position[label]
            coord += scale*self.u(*coord)
            label = f'{label} {self.dataframe.loc[label, "error %"]:1.2f}%'
            ax_disp.plot(*coord, '_', ms=18, mew=4, label=label, color=color)
        for label in ['IDv0', 'ODv0']:
            color = f'C{next(color_index)%10}'
            if label not in self.labels:
                continue
            coord = self.position[label]
            coord += scale*self.u(*coord)
            label = f'{label} {self.dataframe.loc[label, "error %"]:1.2f}%'
            ax_disp.plot(*coord, '|', ms=18, mew=4, label=label, color=color)
        plt.legend(ncol=1, loc='center left', 
                   bbox_transform=ax_disp.transAxes,
                   bbox_to_anchor=(1.25, 0.5))
        
        fig.suptitle(
                #f'{self.material_model}\n' + \
                f'{self.material_data_text(full=full, tex=True)}', 
                y=0.95, fontsize='small')
            
    def plot_biot(self):
        plt.set_aspect(0.9)
        
        fig, ax = plt.subplots(1, 3, sharey=True)
        self.biotgrid.plot(axes=ax[0])
        self.coilset.plot(axes=ax[0])
        ax[0].axis('equal')
        ax[0].axis('off')
        
        plt.sca(ax[1])
        df.plot(self.mesh, alpha=0.2, zorder=50)
        ax[1].axis('equal')
        ax[1].axis('off')
        
        plt.sca(ax[2])
        df.plot(self.f, mesh=self.mesh, scale_units='height', 
                scale=2.5e9, width=0.01, zorder=60)
        ax[2].axis('equal')
        ax[2].axis('off')
    
    
if __name__ == '__main__':
    
    csm = CSmodulue(material_model='t', num=120)
    
    csm.Ic = 40e3


    # t: 'Ep', 'Et', 'nu_pp', 'nu_pt'
    
    csm._active[:] = True
    #csm._active[-2:] = False
    #csm._active[2:] = False

    #csm._active[-1] = False
    
    #csm._active[1] = True
    #csm._active[2] = True
    #csm._active[3] = True
    
    #csm._active[-1] = True
    #csm._active[-1:] = True
    
    #csm._active[-2:] = True
    
    #csm.material_data = dict(Ez=30e9, Ex=50e9)
    #labels = ['IDh0', 'IDh1', 'IDh2', 'ODh0', 'ODh1', 'ODh2']
    #labels = ['IDdv', 'ODdv', 'IDh0']
    #labels = ['IDh0', 'IDh1', 'ODh0', 'ODh1', 'ODh2', 'IDv0']
    labels = ['IDdv', 'ODdv', 'IDh0', 'ODh0']
    #labels = list(csm.data)
    
    csm.extract_properties(labels=labels)
    #csm.labels = labels
    #csm.solve()
    #csm.plot(scale=250, full=True, twin=False)
    
    csm_hf = CSmodulue(material_model=csm.material_model, num=None)
    csm_hf.material_data = csm.material_data.to_dict()
    csm_hf.Ic = csm.Ic
    #csm_hf.extract_properties(labels=labels)
    csm_hf.solve()

    plt.set_context('talk')
    csm_hf.labels = labels
    csm_hf.plot(scale=250, full=True, twin=False)

    #csm_hf.plot_biot()
    
    
