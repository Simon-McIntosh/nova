import itertools
from itertools import count, cycle
from warnings import warn

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D

from nova.structural.properties import secondmoment
from nova.utilities.pyplot import plt
from nova.utilities.addtext import linelabel
from nova.utilities import geom


def delete_row_csr(mat, i):
    if not isinstance(mat, csr_matrix):
        raise ValueError('works only for CSR format -- use .tocsr() first')
    n = mat.indptr[i + 1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i + 1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i + 1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i + 1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0] - 1, mat._shape[1])


class finiteframe(secondmoment):

    def __init__(self, frame='1D', nShape=11, scale_factor=1):
        super().__init__()
        self.coordinate = ['x', 'y', 'z', 'tx', 'ty', 'tz']
        self.scale_factor = scale_factor  # displacment scale factor (plotting)
        self.nShape = nShape  # element shape function resolution
        self.set_shape_coefficents()  # element shape coefficients
        self.frame = frame
        self.set_stiffness()  # set stiffness matrix + problem dofs
        self.define_materials()
        self.initalize_mat()  # materials array
        self.initalize_BC()  # boundary conditions
        self.initalise_mesh()  # FE mesh
        self.initalise_couple()  # node coupling

    def clfe(self):  # clear all (mesh, BCs, constraints and loads)
        self.initalize_BC()  # clear boundary conditions
        self.initalise_mesh()  # clear FE mesh - also deletes loads
        self.initalise_couple()  # remove node coupling
        try:
            del self.T3  # clear local - global roation matrix to force update
        except AttributeError:
            pass  # T3 not defined

    def clf(self):  # clear loads retaining mesh, BCs and constraints
        self.Fo = np.zeros(self.nnd * self.ndof)

    def initalise_couple(self):
        self.ncp = 0  # number of constraints
        self.mpc = {}  # multi-point constraint

    def initalize_mat(self, nmat_max=50, nsec_max=2):
        self.nmat_max = nmat_max
        self.nsec_max = nsec_max
        self.nmat = 0
        self.matID = -1
        self.mat_index = {}  # dict of lists (matIDs)
        self.pntID = -1
        self.Iindex = {'y': 1, 'z': 2}
        self.pnt = []  # list of list - outer points of each section for stress
        Itype = np.dtype({'names': ['xx', 'yy', 'zz'],
                          'formats': ['float', 'float', 'float']})
        Ctype = np.dtype({'names': ['y', 'z'], 'formats': ['float', 'float']})
        self.mtype = np.dtype({'names': ['name', 'E', 'G', 'rho',
                                         'J', 'A', 'I', 'v', 'C', 'pntID'],
                               'formats': ['|U16', 'float', 'float', 'float',
                                           'float', 'float', Itype, 'float',
                                           Ctype, 'int']})
        self.mtype_arrray = np.dtype(
                {'names': ['ID', 'name', 'nsection', 'mat_o', 'mat_array'],
                 'formats': ['int', '|U16', 'int',
                             self.mtype, (self.mtype, self.nsec_max)]})
        self.mat = np.zeros((nmat_max), dtype=self.mtype_arrray)

    def extend_mat(self, nblock=50):  # extend mat structured array
        self.matID += 1  # increment material number
        self.nmat += 1  # increment material ID
        if self.nmat > self.nmat_max:
            self.nmat_max += nblock
            self.mat = np.append(
                    self.mat, np.zeros((nblock), dtype=self.mtype_arrray))

    def add_mat(self, name, materials, sections):
        '''
        add material + sectional properties
        name == user defined label
        materials == [list of material names]
        sections == [list of sections]
        add mat combines material + section lists as addition (sliding)
        EI = EI_1 + EI_2 + ...
        '''
        self.extend_mat()  # extend structured array in blocks
        area_weight = ['E', 'v', 'rho', 'J']  # list of area weighted terms
        self.mat[self.matID]['ID'] = self.matID
        self.mat[self.matID]['name'] = name
        for i, (material, section) in enumerate(zip(materials, sections)):
            mat, pnt = self.set_mat(material, section)
            self.mat[self.matID]['mat_array'][i] = self.add_pnt(mat, pnt)
            self.mat[self.matID]['nsection'] += 1

        mat_o = np.zeros(1, dtype=self.mtype)[0]  # default mat data structure
        for i in range(self.mat[self.matID]['nsection']):
            mat_instance = self.mat[self.matID]['mat_array'][i]
            mat_o['A'] += mat_instance['A']  # sum areas
            for var in area_weight:  # sum area weighted terms
                mat_o[var] += mat_instance['A']*mat_instance[var]
            for var in mat_instance['I'].dtype.names:  # sum second moments
                mat_o['I'][var] += mat_instance['E']*mat_instance['I'][var]
        for var in area_weight:  # normalise area weighted terms
            mat_o[var] /= mat_o['A']
        mat_o['G'] = self.update_G(mat_o['E'], mat_o['v'])
        for var in mat_instance['I'].dtype.names:  # normalise second moments
            mat_o['I'][var] /= mat_o['E']
        mat_o['name'] = 'mean'
        self.mat[self.matID]['mat_o'] = mat_o
        self.mat_index[name] = [self.matID]
        return self.matID

    def define_materials(self):
        # forged == inner leg, cast == outer + supports
        self.mat_data = {}
        self.mat_data['wp'] = {'E': 95e9, 'rho': 8940, 'v': 0.33}
        self.mat_data['steel_forged'] = {'E': 205e9, 'rho': 7850, 'v': 0.29}
        self.mat_data['steel_cast'] = {'E': 190e9, 'rho': 7850, 'v': 0.29}
        self.update_shear_modulus()

    def add_material(self, name, data):
        #  data = dict{'E':, 'rho':, 'v': }
        self.mat_data[name] = data
        self.update_shear_modulus()

    def update_shear_modulus(self):
        for name in self.mat_data:
            E, v = self.mat_data[name]['E'], self.mat_data[name]['v']
            self.mat_data[name]['G'] = self.update_G(E, v)

    def update_G(self, E, v):
        return E/(2*(1+v))

    def set_mat(self, material_name, section):
        mat = np.zeros(1, dtype=self.mtype)[0]  # default mat data structure
        mat['name'] = material_name
        material = self.mat_data[material_name]
        for var in material:
            mat[var] = material[var]
        pnt = section.pop('pnt', [[0], [0]])
        for var in section:
            try:  # treat as scalar
                mat[var] = section[var]
            except TypeError:  # extract dict
                for subvar in section[var]:
                    mat[var][subvar] = section[var][subvar]
        return mat, pnt

    def add_pnt(self, mat, pnt):
        self.pntID += 1  # advance pntID index
        self.pnt.append(pnt)
        mat['pntID'] = self.pntID
        return mat

    def get_mat_o(self, nmat):  # get averaged sectional properties
        if self.frame == 'spring':
            mat = ['E', 'A']
        elif self.frame == '1D':
            mat = ['E', 'Iz']
        elif self.frame == '2D':
            mat = ['E', 'A', 'Iz']
        else:
            mat = ['E', 'A', 'G', 'J', 'Iy', 'Iz']
        values = []
        for m in mat:
            if m[0] == 'I':  # 'Iy, Iz'
                val = self.mat['mat_o'][nmat]['I'][self.Iindex.get(m[1])]
            else:
                val = self.mat['mat_o'][nmat][m]
            values.append(val)
        return values

    def initalise_mesh(self):
        self.X = []
        self.npart = 0  # part number
        self.part = {}  # part ordered dict
        self.nnd = 0  # node number
        self.nel = 0  # element number
        self.el = {}

    def add_nodes(self, X):
        if len(np.shape(X)) == 1:  # 1D node input - expand
            X = np.array(X).reshape(-1, 3)
        if np.size(X) == 3:
            X = np.reshape(X, (1, 3))  # single node
        else:
            if np.linalg.norm(X[0, :] - X[-1, :]) == 0:  # loop
                X = X[:-1, :]  # remove repeated node
                # set close_loop=True in add_elements()
        nX = len(X)
        self.nndo = self.nnd  # start index
        self.nnd += nX  # increment node index
        if len(self.X) == 0:  # initalise node arrays
            self.X = X
            self.nd_topo = np.zeros(nX)
            self.Fo = np.zeros(nX * self.ndof)
            self.D = {}
            for disp in self.coordinate:
                self.D[disp] = np.zeros(nX)
            self.Fr = {}  # reaction forces
            for load in self.load:
                self.Fr[load] = np.zeros(nX)
        else:  # append
            self.X = np.append(self.X, X, axis=0)
            self.nd_topo = np.append(self.nd_topo, np.zeros(nX))
            self.Fo = np.append(self.Fo, np.zeros(nX * self.ndof))
            for disp in self.D:
                self.D[disp] = np.append(self.D[disp], np.zeros(nX))
            for load in self.Fr:
                self.Fr[load] = np.append(self.Fr[load], np.zeros(nX))

    def list_nodes(self):
        n = np.zeros((self.nnd - self.nndo - 1, 2), dtype='int')
        n[:, 1] = np.arange(self.nndo + 1, self.nnd)
        n[:, 0] = n[:, 1] - 1
        return n

    def add_elements(self, n=[], nmat='', part_name='', close_loop=False,
                     el_dy=[0, 1, 0]):
        # list of node pairs shape==[n,2]
        if len(part_name) == 0:
            part_name = 'part_{:1d}'.format(self.npart)
        n = np.array(n)
        if len(n) == 0:  # construct from last input node set
            n = self.list_nodes()
        elif len(np.shape(n)) == 1:  # single dimension node list
            nl = np.copy(n)  # node list
            n = np.zeros((len(n) - 1, 2), dtype='int')
            n[:, 0], n[:, 1] = nl[:-1], nl[1:]
        elif np.size(n) == 2:
            n = np.reshape(n, (1, 2))
        if len(n) == 0:
            err_txt = 'zero length node pair array'
            raise ValueError(err_txt)
        if close_loop:
            n = np.append(n, np.array([n[-1, 1], n[0, 0]], dtype=int, ndmin=2),
                          axis=0)
        self.nelo = self.nel  # start index
        self.nel += len(n)  # element number
        # element aligned coordinate
        dx = self.X[n[:, 1], :] - self.X[n[:, 0], :]
        X = self.X[n[:, 0], :] + dx/2  # element midpoint

        self.check_frame(dx)
        dl = np.linalg.norm(dx, axis=1)  # element length
        norm = np.dot(np.array(dl, ndmin=2).T, np.ones((1, 3)))
        dx = dx / norm  # unit length
        el_dy = np.array(el_dy, ndmin=2)  # element orentation
        if np.shape(el_dy) != np.shape(dx):
            if np.shape(el_dy)[0] == 1:  # duplicate el_dy
                el_dy = np.dot(el_dy.T, np.ones((1, len(n)))).T
            else:
                errtxt = 'malformed element orientation vector, el_dy\n'
                errtxt += 'el_dy vector set greater than one'
                errtxt += 'and el_dy shape {} '.format(np.shape(el_dy))
                errtxt += 'not equal to dx shape {}'.format(np.shape(dx))
                raise ValueError(errtxt)
        dy = np.zeros(np.shape(dx))
        for i, (dx_, dy_) in enumerate(zip(dx, el_dy)):
            dy[i, :] = dy_ - np.dot(dy_, dx_.T) * dx_
        norm = np.dot(np.linalg.norm(dy, axis=1).reshape(-1, 1), 
                      np.ones((1, 3)))
        dy /= norm
        dz = np.cross(dx, dy)  # right hand coordinates

        # set material properties
        if isinstance(nmat, int):  # constant sectional properties
            mat_array = nmat * np.ones(len(n), dtype=int)
        elif isinstance(nmat, str):
            try:
                nmat = self.mat_index[nmat]  # extract mat_index
            except KeyError:
                if len(nmat) == 0:
                    errtxt = 'nmat variable not set in add_elements\n'
                    errtxt += 'specify ether nmat index (int) or '
                    errtxt += 'sectional data name (str)'
                else:
                    errtxt = 'sectional properties for '
                    errtxt += '\'{}\' not present'.format(nmat)
                raise ValueError(errtxt)
            nm = len(nmat)
            if nm == 1:
                mat_array = nmat * np.ones(len(n), dtype=int)
            else:  # variation with normalized length
                l_sum = np.cumsum(dl)
                l_norm = l_sum/l_sum[-1]
                l_norm -= dl/(2*l_sum[-1])
                mat_interp = interp1d(np.linspace(0, 1, nm),
                                      nmat, kind='nearest')
                mat_array = np.array(mat_interp(l_norm), dtype=int)
        elif len(nmat) == len(n):  # nmat 2D list - linear variation
            mat_array = nmat
        else:
            errtxt = 'incorrect input of nmat variable'
            errtxt += 'int or str'
            raise ValueError(errtxt)
        self.el_append('dx', dx)  # store elements
        self.el_append('dy', dy)
        self.el_append('dz', dz)
        self.el_append('dl', dl)
        self.el_append('X', X)
        self.el_append('mat', mat_array)
        self.el_append('n', n)
        self.el_append('contact', np.ones(len(n)))
        self.nd_connect(n)
        self.add_part(part_name)

    def nd_connect(self, n):
        for nd in np.ravel(n):
            self.nd_topo[nd] += 1

    def el_append(self, key, value):
        if key in self.el:
            self.el[key] = np.append(self.el[key], value, axis=0)
        else:
            self.el[key] = value

    def check_name(self, name):  # ensure unique part name
        if name in self.part:
            count = 1
            name = name + '_{:02d}'.format(count)
            while name in self.part:
                count += 1
                name = list(name)
                name[-2:] = '{:02d}'.format(count)
                name = ''.join(name)
                if count == 99:
                    raise ValueError('too many duplicate part names')
        return name

    def add_part(self, name, iel=[]):
        self.npart += 1
        name = self.check_name(name)
        self.part[name] = {}  # initalise dict
        if len(iel) == 0:  # construct from last input element set
            iel = np.arange(self.nelo, self.nel)
        self.part[name]['el'] = iel  # construct from input element list
        self.part[name]['nel'] = len(iel)
        self.part[name]['Lnd'] = np.append(0, np.cumsum(self.el['dl'][iel]))
        n = 2 * (len(iel))
        L, nd = np.zeros(n), np.zeros(n)
        L[::2] = self.part[name]['Lnd'][:-1]
        L[1::2] = self.part[name]['Lnd'][1:]
        L /= self.part[name]['Lnd'][-1]  # normalise
        nd[::2] = self.el['n'][iel, 0]
        nd[1::2] = self.el['n'][iel, 1]
        self.part[name]['nd_fun'] = interp1d(L, nd)  # node interpolator
        self.part[name]['Lel'] = (self.part[name]['Lnd'][1:] +
                                  self.part[name]['Lnd'][:-1]) / 2

    def initalize_BC(self):
        self.BCdtype = [('nd', int), ('d', float)]
        self.BC = {}
        self.BC['fix'] = np.array([], dtype=self.BCdtype)  # type:[node,...]
        self.BC['pin'] = np.array([], dtype=self.BCdtype)  # type:[node,...]
        for key in self.dof:
            self.BC[key] = np.array([], dtype=self.BCdtype)  # type:[node,...]

    def set_shape_coefficents(self):
        self.S = {}
        self.S['s'] = np.linspace(0, 1, self.nShape)  # inter-element spacing
        # Hermite shape functions (disp)
        self.S['Nv'] = np.zeros((self.nShape, 4))
        self.S['Nv'][:, 0] = 1 - 3 * self.S['s']**2 + 2 * self.S['s']**3
        self.S['Nv'][:, 1] = self.S['s'] - 2 * self.S['s']**2 + self.S['s']**3
        self.S['Nv'][:, 2] = 3 * self.S['s']**2 - 2 * self.S['s']**3
        self.S['Nv'][:, 3] = -self.S['s']**2 + self.S['s']**3
        self.S['Nv_dL'] = np.zeros((self.nShape, 2))
        self.S['Nv_dL'][:, 0] = np.copy(self.S['Nv'][:, 1])
        self.S['Nv_dL'][:, 1] = np.copy(self.S['Nv'][:, 3])
        # Hermite functions (curvature)
        self.S['Nd2v'] = np.zeros((self.nShape, 4))
        self.S['Nd2v'][:, 0] = -6 + 12 * self.S['s']
        self.S['Nd2v'][:, 1] = -4 + 6 * self.S['s']
        self.S['Nd2v'][:, 2] = 6 - 12 * self.S['s']
        self.S['Nd2v'][:, 3] = -2 + 6 * self.S['s']
        self.S['Nd2v_dL'] = np.zeros((self.nShape, 2))
        self.S['Nd2v_dL'][:, 0] = np.copy(self.S['Nd2v'][:, 1])
        self.S['Nd2v_dL'][:, 1] = np.copy(self.S['Nd2v'][:, 3])
        # Hermite functions (shear)
        self.S['Nd3v'] = np.zeros((self.nShape, 4))
        self.S['Nd3v'][:, 0] = 12 * np.ones(self.nShape)
        self.S['Nd3v'][:, 1] = 6 * np.ones(self.nShape)
        self.S['Nd3v'][:, 2] = -12 * np.ones(self.nShape)
        self.S['Nd3v'][:, 3] = 6 * np.ones(self.nShape)
        self.S['Nd3v_dL'] = np.zeros((self.nShape, 2))
        self.S['Nd3v_dL'][:, 0] = np.copy(self.S['Nd3v'][:, 1])
        self.S['Nd3v_dL'][:, 1] = np.copy(self.S['Nd3v'][:, 3])

    def interpolate(self):
        for el in range(self.nel):
            d = self.displace(el)  # local 12 dof nodal displacment
            u = np.zeros(2)
            for i in range(2):  # axial displacment
                u[i] = d[6 * i]
            for i, j in enumerate([1, 3]):  # adjust for element length
                self.S['Nv'][:, j] = self.S['Nv_dL'][:, i] * \
                    self.el['dl'][el]
                self.S['Nd2v'][:, j] = self.S['Nd2v_dL'][:, i] * \
                    self.el['dl'][el]
                self.S['Nd3v'][:, j] = self.S['Nd3v_dL'][:, i] * \
                    self.el['dl'][el]
            v = np.zeros((self.nShape, 2))
            d2v = np.zeros((self.nShape, 2))
            d3v = np.zeros((self.nShape, 2))
            for i, label in enumerate([['y', 'tz'], ['z', 'ty']]):
                if label[0] in self.disp:
                    d1D = self.displace_1D(d, label)
                    v[:, i] = np.dot(self.S['Nv'], d1D)
                    d2v[:, i] = np.dot(self.S['Nd2v'], d1D) / \
                        self.el['dl'][el]**2
                    d3v[:, i] = np.dot(self.S['Nd3v'], d1D) / \
                        self.el['dl'][el]**3
            self.store_shape(el, u, v, d2v, d3v)
            self.store_stress(el, u, d2v)  # store stress components along el
        self.shape_part(labels=['u', 'U', 'd2u', 'd3u'])  # set part shape
        self.stress_part()  # set part stress
        self.deform()

    def displace(self, el):
        d = np.zeros(12)
        for i, n in enumerate(self.el['n'][el]):  # each node in element
            for label in self.disp:
                index = self.coordinate.index(label)
                d[i * 6 + index] = self.D[label][n]
        for i in range(4):  # transform to local coordinates
            d[i * 3:i * 3 + 3] = np.dot(self.T3[:, :, el], d[i * 3:i * 3 + 3])
        return d

    def displace_1D(self, d, label):
        d1D = np.zeros(4)
        for i in range(2):  # node
            for j in range(2):  # label
                index = self.coordinate.index(label[j])
                d1D[2 * i + j] = d[6 * i + index]
                if label[j] == 'ty':  # displacment in z
                    d1D[2 * i + j] *= -1
        return d1D

    def set_shape(self):
        self.shape = {}
        for label in ['u', 'd2u', 'd3u', 'U', 'D']:
            self.shape[label] = np.zeros((self.nel, 3, self.nShape))

    def store_shape(self, el, u, v, d2v, d3v):
        self.shape['u'][el, 0] = np.linspace(u[0], u[1], self.nShape)
        self.shape['u'][el, 1] = v[:, 0]  # displacment
        self.shape['u'][el, 2] = v[:, 1]
        self.shape['U'][el] = np.dot(self.T3[:, :, el].T,  # to global
                                     self.shape['u'][el, :, :])
        self.shape['d2u'][el, 1] = d2v[:, 0]  # curvature
        self.shape['d2u'][el, 2] = d2v[:, 1]
        self.shape['d3u'][el, 1] = d3v[:, 0]  # shear, dM/dx
        self.shape['d3u'][el, 2] = d3v[:, 1]

    def set_stress(self):
        nsection = self.mat['nsection'].max()  # maximum section number
        self.stress = {}  # axial, curve_y, curve_z
        for label in ['axial', 'axial_load',
                      'cy_max', 'cy_min', 'cy',
                      'cz_max', 'cz_min', 'cz',
                      's_max', 's_min', 's']:
            self.stress[label] = np.zeros((self.nel, nsection))

    def store_stress(self, el, u, d2v):
        # store maximum stress components for each element
        contact = self.el['contact'][el]
        L = self.el['dl'][el]  # element length
        du = u[1] - u[0]  # piecewise constant
        nmat = self.el['mat'][el]
        mat_array = self.mat['mat_array'][nmat]
        for nsec in range(self.mat[nmat]['nsection']):
            mat = mat_array[nsec]
            E = mat['E'] * contact  # adjust stiffness
            axial = E * du / L  # axial stress
            axial_load = axial * mat['A']
            C = mat['C']  # centroid
            pntID = mat['pntID']
            pnt = self.pnt[pntID]  # section outline
            shape = np.shape(pnt)
            if len(shape) == 3:  # multi part
                npart = shape[1]
            else:
                npart = 1
                pnt = np.expand_dims(pnt, 1)
            y, z = np.array([]), np.array([])
            for i in range(npart):
                y = np.append(y, pnt[0][i])
                z = np.append(z, pnt[1][i])
            y -= C[0]  # distance from centroid
            z -= C[1]
            nyz = len(y)
            y = np.dot(np.ones((self.nShape, 1)), y.reshape(1, -1))
            z = np.dot(np.ones((self.nShape, 1)), z.reshape(1, -1))
            cy = E * z * np.dot(d2v[:, 1].reshape(-1, 1), np.ones((1, nyz)))
            cz = E * y * np.dot(d2v[:, 0].reshape(-1, 1), np.ones((1, nyz)))
            s = cy + cz + axial
            self.stress['axial'][el, nsec] = axial
            self.stress['axial_load'][el, nsec] = axial_load
            self.stress['cy_max'][el, nsec] = np.max(cy)
            self.stress['cy_min'][el, nsec] = np.min(cy)
            cy_index = np.argmax(abs(cy))
            self.stress['cy'][el, nsec] = cy.flatten()[cy_index]
            self.stress['cz_max'][el, nsec] = np.max(cz)
            self.stress['cz_min'][el, nsec] = np.min(cz)
            cz_index = np.argmax(abs(cz))
            self.stress['cz'][el, nsec] = cz.flatten()[cz_index]
            self.stress['s_max'][el, nsec] = np.max(s)
            self.stress['s_min'][el, nsec] = np.min(s)
            s_index = np.argmax(abs(s))
            self.stress['s'][el, nsec] = s.flatten()[s_index]

    def stress_part(self):
        for part in self.part:
            nel = self.part[part]['nel']
            nmat = self.el['mat'][self.part[part]['el'][0]]
            nsec = self.mat[nmat]['nsection']
            self.part[part]['nsection'] = nsec
            self.part[part]['name'] = self.mat[nmat]['mat_array']['name']
            self.part[part]['name'] = self.part[part]['name'][:nsec]
            for label in self.stress:  # initalize
                self.part[part][label] = np.zeros((nel, nsec))
            for iel, el in enumerate(self.part[part]['el']):
                nsection = self.mat[self.el['mat'][el]]['nsection']
                for label in self.stress:
                    self.part[part][label][iel, :] = \
                        self.stress[label][el, :nsection]
            #self.part[part]['Lel'] = (self.part[part]['Lnd'][1:] +
            #                          self.part[part]['Lnd'][:-1]) / 2

    def shape_part(self, labels=['u', 'U', 'D', 'd2u', 'd3u']):
        for part in self.part:
            nS = self.part[part]['nel'] * self.nShape
            self.part[part]['Lshp'] = \
                np.linspace(0, self.part[part]['Lnd'][-1], nS)  # sub-divided
            for label in labels:
                self.part[part][label] = np.zeros((nS, 3))
            for iel, el in enumerate(self.part[part]['el']):
                i = iel * self.nShape
                for label in labels:
                    self.part[part][label][i:i + self.nShape, :] = \
                        self.shape[label][el, :, :].T

    def set_stiffness(self):  # set element stiffness matrix
        if self.frame == 'spring':
            dof = ['u', 'v', 'w']
            disp = ['x', 'y', 'z']
            load = ['fx', 'fy', 'fz']
            stiffness = self.stiffness_spring
        elif self.frame == '1D':
            dof = ['v', 'rz']
            disp = ['y', 'tz']
            load = ['fy', 'mz']
            stiffness = self.stiffness_1D
        elif self.frame == '2D':
            dof = ['u', 'v', 'rz']
            disp = ['x', 'y', 'tz']
            load = ['fx', 'fy', 'mz']
            stiffness = self.stiffness_2D
        elif self.frame == '3D':
            dof = ['u', 'v', 'w', 'rx', 'ry', 'rz']
            disp = ['x', 'y', 'z', 'tx', 'ty', 'tz']
            load = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']
            stiffness = self.stiffness_3D
        else:
            raise IndexError(f'frame not found, check setting: {self.frame}')
        self.dof = dof
        self.disp = disp
        self.load = load
        self.ndof = len(dof)  # degrees of fredom per node
        self.stiffness = stiffness

    def stiffness_spring(self, el):  # dof [u]
        L = self.el['dl'][el]
        E, A = self.get_mat_o(self.el['mat'][el])
        k = E * A / L * np.array([[0, 0,  0, 0, 0,  0],
                                   [0, 0,  0, 0, 0,  0],
                                   [0, 0,  1, 0, 0, -1],
                                   [0, 0,  0, 0, 0,  0],
                                   [0, 0,  0, 0, 0,  0],
                                   [0, 0, -1, 0, 0,  1]])
        return k

    def stiffness_1D(self, el):  # dof [v,rz]
        a = self.el['dl'][el] / 2
        E, Iz = self.get_mat_o(self.el['mat'][el])
        k = E * Iz / (2 * a**3) * np.array([[3,   3*a,   -3,  3*a],
                                             [3*a, 4*a**2, -3*a, 2*a**2],
                                             [-3, -3*a,    3, -3*a],
                                             [3*a, 2*a**2, -3*a, 4*a**2]])
        return k

    def stiffness_2D(self, el):  # dof [u,v,rz]
        L = self.el['dl'][el]
        E, A, Iz = self.get_mat_o(self.el['mat'][el])
        k = np.array([[A*E/L, 0, 0,
                       -A*E/L, 0, 0],
                       [0,  12*E*Iz/L**3, 6*E*Iz/L**2,
                        0, -12*E*Iz/L**3, 6*E*Iz/L**2],
                       [0,  6*E*Iz/L**2, 4*E*Iz/L,
                        0, -6*E*Iz/L**2, 2*E*Iz/L],
                       [-A*E/L, 0, 0,
                        A*E/L, 0, 0],
                       [0, -12*E*Iz/L**3, -6*E*Iz/L**2,
                        0,  12*E*Iz/L**3, -6*E*Iz/L**2],
                       [0,  6*E*Iz/L**2, 2*E*Iz/L,
                        0, -6*E*Iz/L**2, 4*E*Iz/L]])
        k = self.rotate_matrix(k, el)  # transfer to global coordinates
        return k

    def stiffness_3D(self, el):  # dof [u,v,w,rx,ry,rz]
        L = self.el['dl'][el]
        E, A, G, J, Iy, Iz = self.get_mat_o(self.el['mat'][el])
        k = np.array(
            [[A*E/L, 0, 0, 0, 0, 0,
             -A*E/L, 0, 0, 0, 0, 0],
             [0,  12*E*Iz/L**3, 0, 0, 0, 6*E*Iz/L**2,
              0, -12*E*Iz/L**3, 0, 0, 0, 6*E*Iz/L**2],
             [0, 0,  12*E*Iy/L**3, 0, -6*E*Iy/L**2, 0,
              0, 0, -12*E*Iy/L**3, 0, -6*E*Iy/L**2, 0],
             [0, 0, 0,  G*J/L, 0, 0,
              0, 0, 0, -G*J/L, 0, 0],
             [0, 0, -6*E*Iy/L**2, 0, 4*E*Iy/L, 0,
              0, 0,  6*E*Iy/L**2, 0, 2*E*Iy/L, 0],
             [0,  6*E*Iz/L**2, 0, 0, 0, 4*E*Iz/L,
              0, -6*E*Iz/L**2, 0, 0, 0, 2*E*Iz/L],
             [-A*E/L, 0, 0, 0, 0, 0,
              A*E/L, 0, 0, 0, 0, 0],
             [0, -12*E*Iz/L**3, 0, 0, 0, -6*E*Iz/L*2,
              0,  12*E*Iz/L**3, 0, 0, 0, -6*E*Iz/L**2],
             [0, 0, -12*E*Iy/L**3, 0, 6*E*Iy/L**2, 0,
              0, 0,  12*E*Iy/L**3, 0, 6*E*Iy/L**2, 0],
             [0, 0, 0, -G*J/L, 0, 0,
              0, 0, 0,  G*J/L, 0, 0],
             [0, 0, -6*E*Iy/L**2, 0, 2*E*Iy/L, 0,
              0, 0,  6*E*Iy/L**2, 0, 4*E*Iy/L, 0],
             [0,  6*E*Iz/L**2, 0, 0, 0, 2*E*Iz/L,
              0, -6*E*Iz/L**2, 0, 0, 0, 4*E*Iz/L]])
        k = self.rotate_matrix(k, el)  # transfer to global coordinates
        return k

    def check_frame(self, dx):
        if self.frame == '1D':
            if np.sum(abs(np.diff(dx[:, 1]))) > 0:
                err_txt = 'error: rotation in z for 1D element'
                err_txt += ' - select frame=2D'
                raise ValueError(err_txt)
        elif self.frame == '2D':
            if np.sum(abs(np.diff(dx[:, 2]))) > 0:
                err_txt = 'error: rotation in y for 2D element'
                err_txt += ' - select frame=3D'
                raise ValueError(err_txt)

    def update_rotation(self):
        if not hasattr(self, 'T3'):  # delete T3 matrix in self.clfe()
            self.get_rotation()
        # elements added after first solve
        elif np.shape(self.T3)[2] != self.nel:
            self.get_rotation()

    def get_rotation(self):
        dx_, dy_, dz_ = self.el['dx'], self.el['dy'], self.el['dz']
        self.T3 = np.zeros((3, 3, self.nel))
        for i, (dx, dy, dz) in enumerate(zip(dx_, dy_, dz_)):
            # direction cosines
            self.T3[:, :, i] = np.array([dx, dy, dz])

    def rotate_matrix(self, k, el):
        T = np.zeros((2 * self.ndof, 2 * self.ndof))
        for i in range(int(2 / 3 * self.ndof)):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = self.T3[:, :, el]
        k = T.T @ k @ T
        return k

    # 'dof','disp','load'
    def check_input(self, vector, label, terminate=True):
        attributes = getattr(self, vector)
        error_code = 0
        if label not in attributes and not \
                (vector == 'dof' and (label == 'fix' or label == 'pin')):
            error_code = 1
            if terminate:
                err_txt = 'attribute \'' + label + '\' not present'
                err_txt += 'in frame=' + self.frame
                err_txt += ' [' + ', '.join(attributes) + ']'
                raise ValueError(err_txt)
        return error_code

    def add_load(self, **kwargs):  # distribute load to adjacent nodes
        csys = ''  # coodrdinate system unset
        if 'f' in kwargs or 'F' in kwargs:  # point load
            load_type = 'point'
            if 'L' in kwargs and 'part' in kwargs:  # identify element
                L, part = kwargs.get('L'), kwargs.get('part')
                Ln = self.part[part]['nd_fun'](L)  # length along part
                # TODO fix fractional element loading - nd_fun / el_fun
                el = int(np.floor(Ln))  # element index
                s = Ln - el  # fraction along element
                if L == 1:
                    el -= 1
                    s = 1
            elif 'el' in kwargs:
                el = kwargs.get('el')
                s = kwargs.get('s', 0.5)
            else:
                raise ValueError('define element index or length and part')
            if 'F' in kwargs:
                csys = 'global'
                f = kwargs.get('F')
            elif 'f' in kwargs:
                csys = 'local'
                f = kwargs.get('f')
        elif 'w' in kwargs or 'W' in kwargs:  # distributed load
            load_type = 'dist'
            if 'el' in kwargs:
                el = kwargs.get('el')
            else:
                raise ValueError('define element index')
            s = kwargs.get('s', 0.5)
            if 'W' in kwargs:
                csys = 'global'
                w = np.array(kwargs.get('W'))
            elif 'w' in kwargs:
                csys = 'local'
                w = np.array(kwargs.get('w'))
            f = w * self.el['dl'][el]
        if len(csys) == 0:
            raise ValueError('load vector unset')
        elif csys == 'global':
            f = np.dot(self.T3[:, :, el], f)  # rotate to force to local csys
        fn = np.zeros((6, 2))  # 6 dof local nodal load vector
        # split point load to F,M
        for i, label in enumerate(['fx', 'fy', 'fz']):
            if label in self.load:
                if label == 'fx':
                    fn[i, 0] = (1 - s) * f[i]
                    fn[i, 1] = s * f[i]
                else:  # fy,fz
                    fn[i, 0] = (1 - s)**2 * (1 + 2 * s) * f[i]
                    fn[i, 1] = s**2 * (3 - 2 * s) * f[i]
                    mi = 5 if label == 'fy' else 4  # moment index
                    # fn[mi, 0] = f[i] * (1 - s)**2 * s * self.el['dl'][el]
                    # fn[mi, 1] = -f[i] * s**2 * (1 - s) * self.el['dl'][el]
                    if load_type == 'dist':
                        # reduce moment for distributed load
                        fn[mi, :] *= 8 / 12
            else:
                if abs(f[i]) > 1e-12:
                    err_txt = 'non zero load \'' + label + '\''
                    err_txt += ' not present in frame=' + self.frame
                    err_txt += ' [' + ', '.join(self.load) + ']\n'
                    err_txt += 'increase frame element dimension\n'
                    raise ValueError(err_txt)

        for i, node in enumerate(self.el['n'][el]):  # each node
            F = np.zeros((6))
            F[:3] = np.dot(self.T3[:, :, el].T, fn[:3, i])  # force
            F[3:] = np.dot(self.T3[:, :, el].T, fn[3:, i])  # moment
            for index, label in enumerate(['fx', 'fy', 'fz',
                                           'mx', 'my', 'mz']):
                if label in self.load:
                    self.add_nodal_load(node, label, F[index])

    def add_nodal_load(self, node, label, load):
        self.check_input('load', label)
        index = self.load.index(label)
        self.Fo[node * self.ndof + index] += load  # accumulate

    def update_mass(self):
        for part in self.part:
            mass = 0
            for el, Lel in zip(self.part[part]['el'], self.part[part]['Lel']):
                nm = self.el['mat'][el]  # material index
                mass += self.mat['mat_o']['rho'][nm] *\
                    self.mat['mat_o']['A'][nm] * Lel
            self.part[part]['mass'] = mass

    def add_weight(self, g=[0, 0, -1]):
        self.update_rotation()  # check / update rotation matrix
        for part in self.part:
            for el in self.part[part]['el']:
                nm = self.el['mat'][el]  # material index
                w = 9.81 * self.mat['mat_o']['rho'][nm] *\
                    self.mat['mat_o']['A'][nm]
                W = w*np.array(g)
                self.add_load(el=el, W=W)  # self weight

    def add_tf_load(self, sf, ff, tf, Bpoint, parts=['loop', 'nose'],
                    method='function'):  # method='function'
        cage = coil_cage(nTF=tf.profile.nTF, rc=2*tf.rc,
                         plasma={'sf': sf}, coil=tf.p['cl'], ny=1, nr=1)
        self.update_rotation()  # check / update rotation matrix
        ff.set_bm(cage)  # set tf magnetic moment (for method==function)
        for part in parts:
            for el in self.part[part]['el']:
                point = self.el['X'][el]  # element midpoint
                w = ff.topple(point, cage.Iturn*self.el['dx'][el],
                              cage, Bpoint, method=method)[0]  # body force
                self.add_load(el=el, W=w)  # bursting/toppling load

    def check_part(self, part):
        if part not in self.part and part != 'all':
            err_txt = part + \
                ' not present in [' + ', '.join(self.part.keys()) + ']'
            raise ValueError(err_txt)

    def part_nodes(self, index, part, ends=2):
        # el ends, 0==start,1==end,2==both
        if len(part) > 0:  # element index relitive to part
            index_type = 'element'
        else:  # node index
            index_type = 'node'
        if index_type == 'element':
            if part == 'all':
                nodes = range(self.nnd)
            else:
                if index == 'all':
                    if len(part) > 0:  # node index relitive to part
                        elements = list(range(self.part[part]['nel']))
                else:
                    elements = index
                    if not isinstance(elements, list):  # convert to list
                        elements = [elements]
                nodes = -1 * np.ones(2 * len(elements))  # node array
                for i, element in enumerate(elements):
                    el = self.part[part]['el'][element]
                    if ends == 0 or ends == 2:  # start or both
                        nodes[2 * i] = self.el['n'][el, 0]
                    if ends == 1 or ends == 2:  # end or both
                        nodes[2 * i + 1] = self.el['n'][el, 1]
                nodes = np.unique(nodes)
                nodes = nodes[nodes > -1]
        elif index_type == 'node':
            nodes = index
            if not isinstance(nodes, list):  # convert to list
                nodes = [nodes]
        return nodes

    def extract_dof(self, dof):
        if isinstance(dof, str):  # convert to list
            dof = [dof]
        if dof[0] == 'fix':  # fix all dofs
            dof = self.dof
        elif dof[0] == 'pin':  # fix translational dofs
            dof = [dof for dof in self.dof if 'r' not in dof]
        elif 'n' in dof[0]:  # free constraint 'nu','nrx',...
            for d in dof:
                if 'n' not in d:
                    errtxt = 'mixed constraints in negated cp '
                    errtxt += 'dof: {} \n'.format(dof)
                    errtxt += 'place\'n\' type constraints in exclusive set'
                    raise ValueError(errtxt)
                dof_free = [dof for dof in self.dof if d[1:] in dof]
            dof = [dof for dof in self.dof if dof not in dof_free]
        else:
            dof = dof
        return dof

    def add_bc(self, dof, index, part=None, d=0, ends=2, terminate=True):
        if part:  # part=part then index=elements
            self.check_part(part)
            nodes = self.part_nodes(index, part, ends=ends)  # select nodes
        else:  # part=None then index=nodes,
            nodes = index
        if isinstance(nodes, str) or not hasattr(nodes, '__len__'):
            nnd = 1
        else:
            nnd = len(nodes)
        dof = self.extract_dof(dof)
        for constrn in dof:  # constrain nodes
            error_code = self.check_input('dof', constrn, terminate=terminate)
            if not error_code:  # append structured array
                BC = np.ones(nnd, dtype=self.BCdtype)
                BC['nd'], BC['d'] = nodes, d
                index = [np.array(
                        [nd in self.BC[constrn]['nd'] for nd in BC['nd']]),
                         np.array(
                        [nd in BC['nd'] for nd in self.BC[constrn]['nd']])]
                if sum(index[1]) > 0:
                    self.BC[constrn]['d'][index[1]] = BC[index[0]]['d']
                self.BC[constrn] = np.append(self.BC[constrn], BC[~index[0]])

    def extractBC(self):  # extract matrix indicies for constraints
        self.BCindex = np.array([], dtype=self.BCdtype)
        for j, constrn in enumerate(self.BC.keys()):
            ui = self.ndof * self.BC[constrn]['nd']  # node dof
            BC = np.zeros(len(ui), dtype=self.BCdtype)
            BC['d'] = self.BC[constrn]['d']
            if len(ui) > 0:
                if constrn == 'fix':  # fix all dofs
                    for i in range(self.ndof):
                        BC['nd'] = ui + i
                        self.BCindex = np.append(self.BCindex, BC)
                elif constrn == 'pin':  # pin node (u,v,w)
                    for i in range(int(self.ndof / 2)):
                        BC['nd'] = ui + i
                        self.BCindex = np.append(self.BCindex, BC)
                else:
                    # skip-fix,pin
                    BC['nd'] = ui + j - 2
                    self.BCindex = np.append(self.BCindex, BC)
        # self.BCindex = list(map(int, set(self.BCindex)))

    def extractND(self):
        BC = np.zeros(self.ndof, dtype=self.BCdtype)
        for nd in np.where(self.nd_topo == 0)[0]:  # remove unconnected nodes
            BC['nd'] = nd * self.ndof + np.arange(0, self.ndof)
            self.BCindex = np.append(self.BCindex, BC)

    def assemble(self):  # assemble global stiffness matrix
        self.Ko = np.zeros((self.nK, self.nK))  # matrix without constraints
        for el in range(self.nel):
            ke = self.stiffness(el) * self.el['contact'][el]
            ke_index = itertools.product([0, self.ndof], repeat=2)
            ko_index = itertools.product(self.ndof * self.el['n'][el],
                                         repeat=2)
            for i, j in zip(ko_index, ke_index):
                self.Ko[i[0]:i[0]+self.ndof, i[1]:i[1]+self.ndof] += \
                    ke[j[0]:j[0]+self.ndof, j[1]:j[1]+self.ndof]

    def add_cp(self, nodes, dof='fix', nset='next', axis='z', rotate=False):
        nodes = np.copy(nodes)  # make local copy
        if nset == 'next':  # (default)
            self.ncp += 1
        elif isinstance(nset, int):
            self.ncp = nset
        else:
            errtxt = 'nset must be interger or string, \'high\' or \'next\''
            raise ValueError(errtxt)
        if self.ncp == 0:
            self.ncp = 1
        name, dof = self.initalise_cp_set(dof)
        self.mpc[name]['nodes'] = np.append(self.mpc[name]['nodes'], nodes)
        self.mpc[name]['nc'] += len(nodes) - 1
        self.mpc[name]['neq'] = self.mpc[name]['ndof'] * self.mpc[name]['nc']
        self.mpc[name]['Cc'] = -np.identity(self.mpc[name]['neq'])
        if rotate:  # populate Cr with rotational constraint
            axis = self.check_axis(axis)
            mask = ~np.in1d(['x', 'y', 'z'], axis)
            theta = np.arcsin(np.cross(self.X[nodes[0], mask],
                                       self.X[nodes[1], mask]) /
                              (np.linalg.norm(self.X[nodes[0], mask]) *
                               np.linalg.norm(self.X[nodes[1], mask])))
            if np.dot(self.X[nodes[0], mask], self.X[nodes[1], mask]) < 0:
                theta = np.pi / 2 + (np.pi / 2 - theta)
            self.rotate_cp(name, dof, theta, axis)
        else:
            self.mpc[name]['Cr'] = np.identity(self.mpc[name]['neq'])
        # self.check_cp_nodes()
        self.extract_cp_nodes(self.mpc[name])

    def check_axis(self, axis):
        if self.ndof == 3:  # 2D elements
            if axis != 'z':
                warn('rotation switched to z-axis for 2D element')
                axis = 'z'
        return axis

    def rotate_cp(self, name, dof, theta, axis):
        self.check_cp_rotation(dof, self.mpc[name]['neq'])
        R = geom.rotate(theta, axis=axis)  # 3x3 rotation matrix
        self.mpc[name]['Cr'] = np.zeros((self.mpc[name]['neq'],
                                         self.mpc[name]['neq']))
        self.mpc[name]['Cr'][:3, :3] = R
        if self.mpc[name]['neq'] == 6:
            self.mpc[name]['Cr'][3:, 3:] = R

    def check_cp_rotation(self, dof, neq):
        if neq != self.ndof and neq != int(self.ndof / 2):
            errtxt = 'constraint dof \'{}\'\n'.format(dof)
            errtxt += 'incompatable with rotation constraint\n'
            errtxt += 'ndof {}, neq {}'.format(self.ndof, neq)
            raise ValueError(errtxt)
        elif neq == self.ndof:
            if dof != self.dof:
                errtxt = 'constraint dofs in wrong order\n'
                errtxt += 'specified as :{}\n'.format(dof)
                errtxt += 'required order: {}'.format(self.dof)
                raise ValueError(errtxt)
        elif neq == int(self.ndof / 2):
            if dof != self.dof[:int(self.ndof / 2)] or \
               dof != self.dof[int(self.ndof / 2):]:
                errtxt = 'constraint dofs in wrong order\n'
                errtxt += 'specified as :{}\n'.format(dof)
                errtxt += 'required order'
                errtxt += ': {}\n'.format(self.dof[:int(self.ndof / 2)])
                errtxt += 'or'
                errtxt += ': {}\n'.format(self.dof[int(self.ndof / 2):])
                raise ValueError(errtxt)
        if self.ndof == 2:  # exclude 1D elements
            errtxt = 'unable to apply rotational '
            errtxt += 'boundary conditions to 1D model\n'
            errtxt += 'set \'theta=0\' in cp.add'
            raise ValueError(errtxt)

    def initalise_cp_set(self, dof):
        name = 'cp{:d}'.format(self.ncp)  # cp set name
        dof = self.extract_dof(dof)
        if name not in self.mpc:  # create dof set
            self.mpc[name] = {'dof': dof, 'nodes': np.array([], dtype=int)}
            self.mpc[name]['ndof'] = len(dof)
            self.mpc[name]['dr'] = np.array([], dtype=int)
            self.mpc[name]['dc'] = np.array([], dtype=int)
            self.mpc[name]['nc'] = 0
        elif dof != self.mpc[name]['dof']:
            raise ValueError('inconsistent dofs in repeated cp')
        return name, dof

    def extract_cp_nodes(self, mpc):
        dof = mpc['dof']
        row = mpc['nodes'] * self.ndof
        for j in range(mpc['nc']):
            for k, mdof in enumerate(self.dof):
                if mdof in dof:  # retain first node
                    mpc['dr'] = np.append(mpc['dr'], row[0] + k)
                    mpc['dc'] = np.append(mpc['dc'], row[1 + j] + k)

    def assemble_cp_nodes(self):
        self.cp_nd = {'dr': np.array([], dtype=int),
                      'dc': np.array([], dtype=int), 'n': 0}
        self.couple = {}
        ieq = count(0)
        for name in self.mpc:
            for node in ['dr', 'dc']:
                self.cp_nd[node] = np.append(self.cp_nd[node],
                                             self.mpc[name][node])
            self.cp_nd['n'] += self.mpc[name]['neq']

            for i, (Cr, Cc) in enumerate(zip(self.mpc[name]['Cr'],
                                             self.mpc[name]['Cc'])):
                eqname = 'eq{:d}'.format(next(ieq))
                self.couple[eqname] = {'dr': self.mpc[name]['dr'], 'Cr': Cr,
                                       'dc': self.mpc[name]['dc'], 'Cc': Cc,
                                       'dro': self.mpc[name]['dr'][i]}

    def check_cp_nodes(self):  # check for node repeats in constraints
        nodes = np.array([])
        for label in self.mpc:
            nodes = np.append(nodes, self.mpc[label]['nodes'])
        if len(nodes) != len(np.unique(nodes)):
            errtxt = 'repeated node in cp defnition:\n'
            for label in self.mpc:
                errtxt += label + ' ' + str(self.mpc[label]['nodes']) + '\n'
            raise ValueError(errtxt)

    def apply_displacments(self):
        self.d = np.zeros(self.nK)  # apply displacemnt constraints (non-hom)
        self.d[self.BCindex['nd']] = self.BCindex['d']
        self.Fd = np.dot(self.Ko, self.d)

    def constrain(self):  # apply and colapse constraints
        self.extractBC()  # remove BC dofs
        self.extractND()  # remove unconnected nodes
        self.assemble_cp_nodes()
        self.apply_displacments()  # apply displacment constraints
        self.K = np.copy(self.Ko)
        self.F = np.copy(self.Fo - self.Fd)  # subtract displacment forces
        self.nd = {}  # node index
        self.nd['do'] = np.arange(0, self.nK, dtype=int)  # all nodes
        self.nd['mask'] = np.zeros(self.nK, dtype=bool)  # all nodes
        self.nd['mask'][self.BCindex['nd']] = True  # remove
        self.nd['mask'][self.cp_nd['dc']] = True  # condense
        self.nd['dc'] = self.nd['do'][self.nd['mask']]  # condensed
        self.nd['dr'] = self.nd['do'][~self.nd['mask']]  # retained
        self.nd['nc'] = np.sum(self.nd['mask'])
        self.nd['nr'] = np.sum(~self.nd['mask'])
        self.Cc = np.zeros((self.cp_nd['n'], self.cp_nd['n']))
        self.Cr = np.zeros((self.cp_nd['n'], self.nd['nr']))
        for i, name in enumerate(self.couple):
            couple = self.couple[name]
            self.Cr[i, np.in1d(self.nd['dr'], couple['dr'])] = couple['Cr']
            self.Cc[i, np.in1d(self.cp_nd['dc'], couple['dc'])] = couple['Cc']

        # build transformation matrix
        self.Tc = np.zeros((self.nd['nc'], self.nd['nr']))  # initalise
        index = np.in1d(self.nd['dc'], self.cp_nd['dc'])
        self.Tc[index, :] = np.dot(-np.linalg.inv(self.Cc), self.Cr)
        self.T = np.append(np.identity(self.nd['nr']), self.Tc, axis=0)

        # sort and couple K and F
        self.K = np.append(self.K[self.nd['dr'], :],
                           self.K[self.nd['dc'], :], axis=0)
        self.K = np.append(self.K[:, self.nd['dr']],
                           self.K[:, self.nd['dc']], axis=1)
        self.K = np.dot(np.dot(self.T.T, self.K), self.T)
        self.F = np.append(self.F[self.nd['dr']],
                           self.F[self.nd['dc']], axis=0)
        self.F = np.dot(self.T.T, self.F)

    def switch_units(self):
        # local units kg	mm	ms	kN	GPa	kN-mm
        for i in range(3):
            self.Ko[:, i::self.ndof] *= 1e-6  # N/m - KN/mm
            self.Ko[:, i+3::self.ndof] *= 1e-3  # N/rad - KN/rad
            self.Fo[i::self.ndof] *= 1e-3  # N - KN
            self.Fo[i+3::self.ndof] *= 1  # Nm - KNmm
        for i in range(3):
            self.Dn[i::self.ndof] *= 1e-3  # mm - m
            self.Dn[i+3::self.ndof] *= 1  # rad - rad
            self.Fn[i::self.ndof] *= 1e3  # KN - N
            self.Fo[i+3::self.ndof] *= 1  # KNm - Nm

    def solve(self):
        self.nK = int(self.nnd * self.ndof)  # stiffness matrix
        self.set_shape()  # dict of displacments
        self.set_stress()
        self.update_mass()
        self.update_rotation()  # evaluate/update rotation matricies
        self.assemble()  # assemble stiffness matrix
        self.constrain()  # apply and colapse constraints
        self.bounding_box()  # calulate problem length scale
        self.Dn = np.zeros(self.nK)  # initalise displacment matrix
        self.check_condition(10)  # check stiffness matrix condition number
        self.Dn[self.nd['dr']] = np.linalg.solve(self.K, self.F)  # global
        self.Dn[self.nd['dc']] = np.dot(
            self.Tc, self.Dn[self.nd['dr']])  # patch constrained DOFs
        self.Dn[self.BCindex['nd']] = self.BCindex['d']  # patch displacments
        self.Fn = np.dot(self.Ko, self.Dn)  # reaction forces
        for i, disp in enumerate(self.disp):
            self.D[disp] = self.Dn[i::self.ndof]
        for i, load in enumerate(self.load):
            self.Fr[load] = self.Fn[i::self.ndof]
        self.interpolate()

    def preload_err(self, d, *args):
        dof, displacement_node, reaction_node, load = args
        self.add_bc(dof, displacement_node, d=d)  # apply nodal displacment
        self.solve()
        err = abs(self.Fr[self.load[self.dof.index(dof)]][reaction_node] -
                  load)
        return err

    def preload(self, dof, displacement_node, reaction_node, load):
        self.bounding_box()  # calulate problem length scale
        do = 0.5e-3*self.bb  # fraction of problem's bounding box
        minimize(self.preload_err, do,
                 args=(dof, displacement_node, reaction_node, load),
                 method='Nelder-Mead', options={'fatol': 1e1})

    def check_condition(self, digits):
        self.condition_number = np.linalg.cond(self.K)
        self.digit_loss = np.log10(self.condition_number)
        if self.digit_loss > digits:
            warntxt = '\n\nIll-conditioned stiffness matrix\n'
            warntxt += 'Check model boundary conditions\n'
            warntxt += 'Accuracy loss of upto '
            warntxt += '{:d} digits\n'.format(int(np.ceil(self.digit_loss)))
            warn(warntxt)

    def bounding_box(self):
        dx = np.zeros(3)
        self.xo = np.zeros(3)
        for i in range(3):
            dx[i] = np.max(self.X[:, i]) - np.min(self.X[:, i])
            self.xo[i] = np.mean([np.max(self.X[:, i]), np.min(self.X[:, i])])
        self.bb = 1.05 * np.max(dx)

    def deform(self, *args):
        if len(args) == 1:
            self.scale_factor = args[0]
        if self.scale_factor < 0:  # relitive displacment
            Umax = np.max(np.sqrt(self.shape['U'][:, 0, :]**2 +
                                  self.shape['U'][:, 1, :]**2 +
                                  self.shape['U'][:, 2, :]**2))
            if Umax > 0:
                self.scale = -self.scale_factor * self.bb / Umax
            else:
                self.scale = 1
        else:  # absolute displacment
            self.scale = self.scale_factor
        for el in range(self.nel):
            for i in range(3):
                n = self.el['n'][el]
                self.shape['D'][el, i, :] = self.scale * \
                    self.shape['U'][el, i, :] + \
                    np.linspace(self.X[n[0], i], self.X[n[1], i], self.nShape)
        self.shape_part(labels=['D'])

    def get_index(self, projection):
        index = ('xyz'.index(projection[0]), 'xyz'.index(projection[1]))
        return index

    def select_parts(self, select=[]):
        if isinstance(select, str):
            select = [select]
        if not select:
            part = self.part
        else:
            part = [p for p in self.part if p in select]
        return part

    def plot_nodes(self, projection='xz', ms=5, select=[], **kwargs):
        ax = kwargs.get('ax', plt.gca())
        index = self.get_index(projection)
        ax.plot(self.X[:, index[0]], self.X[:, index[1]], 'o', markersize=ms,
                color=0.75 * np.ones(3))
        for part in self.select_parts(select):
            for el in self.part[part]['el']:
                nd = self.el['n'][el]
                ax.plot([self.X[nd[0], index[0]], self.X[nd[1], index[0]]],
                        [self.X[nd[0], index[1]], self.X[nd[1], index[1]]],
                        color='C0', alpha=0.5)
        plt.axis('equal')
        mpc = kwargs.get('mpc', False)
        if mpc:
            for cp in self.mpc:
                nodes = self.mpc[cp]['nodes']
                ax.plot(self.X[nodes, index[0]],
                        self.X[nodes, index[1]], 'C3o-', ms=4)

    def plot_vectors(self, projection='xz', **kwargs):
        ax = kwargs.get('ax', plt.gca())
        index = self.get_index(projection)
        factor = 0.1 * self.bb
        for X, dX in zip(self.el['X'],
                         zip(self.el['dx'], self.el['dy'], self.el['dz'])):
            for i in index:
                ax.arrow(X[index[0]], X[index[1]],
                         factor * dX[i][0, index[0]],
                         factor * dX[i][0, index[1]],
                         head_width=0.15 * factor,
                         head_length=0.2 * factor,
                         color='C{}'.format(i+3))

    def plot_F(self, projection='xz', factor=0.25, **kwargs):
        ax = kwargs.get('ax', plt.gca())
        index = self.get_index(projection)
        nF = sum([1 for var in self.load if 'f' in var])
        F = np.zeros((self.nnd, nF))
        for i in range(nF):
            F[:, i] = self.Fo[i::self.ndof]
        if nF > 1:
            Fmag = np.max(np.linalg.norm(F, axis=1))
        else:
            Fmag = np.max(F)
        if Fmag == 0:
            factor = 1
        else:
            factor *= self.bb / Fmag
        for i, (X, dX) in enumerate(
                zip(self.X, zip(self.D['x'], self.D['y'], self.D['z']))):
            j = i * self.ndof
            if self.frame == '1D':
                F = [0, self.Fo[j]]
            else:
                F = [self.Fo[j], self.Fo[j + 1], self.Fo[j + 2]]
            nF = np.linalg.norm(F)
            if nF != 0:
                ax.arrow(X[index[0]] + self.scale * dX[index[0]],
                         X[index[1]] + self.scale * dX[index[1]],
                         factor * F[index[0]], factor * F[index[1]],
                         head_width=factor * 0.15 * nF,
                         head_length=factor * 0.2 * nF,
                         color='C1', zorder=10, lw=1)
                ax.plot(X[index[0]] + self.scale * dX[index[0]]
                        + factor * F[index[0]],
                        X[index[1]] + self.scale * dX[index[1]]
                        + factor * F[index[1]], '.', alpha=0.5, zorder=10,
                        lw=6)

    def plot_displacment(self, projection='xz', select=[], **kwargs):
        ax = kwargs.get('ax', plt.gca())
        index = self.get_index(projection)
        for ic, part in enumerate(self.select_parts(select)):
            for iel in range(self.part[part]['nel']):  #
                i = iel * self.nShape
                ax.plot(self.part[part]['D'][i:i+self.nShape, index[0]],
                        self.part[part]['D'][i:i+self.nShape, index[1]],
                        color=f'C{ic}')
        for el in range(self.nel):
            for end in [0, -1]:
                ax.plot(self.shape['D'][el, index[0], end],
                        self.shape['D'][el, index[1], end],
                        'o', color='gray', ms=5)

    def plot_moment(self):
        plt.figure(figsize=plt.figaspect(0.75))
        text = linelabel(value='', postfix='', Ndiv=25)
        part = self.part
        # part = ['loop', 'trans_lower', 'trans_upper']
        color = cycle(range(10))
        for i, part in enumerate(part):
            ci = next(color)
            Lnorm = self.part[part]['Lshp'][-1]
            plt.plot(self.part[part]['Lshp']/Lnorm,
                     self.part[part]['d2u'][:, 1],
                     '--', color='C{}'.format(ci))
            plt.plot(self.part[part]['Lshp']/Lnorm,
                     self.part[part]['d2u'][:, 2],
                     '-', color='C{}'.format(ci))
            text.add(part)
        text.plot()
        plt.despine()
        plt.xlabel('part length')
        plt.ylabel('part curvature')

    def plot_stress(self):
        part = self.part
        # part = ['loop', 'trans_lower', 'trans_upper']
        color = cycle(range(10))
        axes = plt.subplots(3, 1, sharex=True, sharey=True,
                            figsize=plt.figaspect(0.75))[1]
        stress = ['cy', 'cz', 'axial']
        for i, part in enumerate(part):
            ci = next(color)
            # Lnorm = self.part[part]['Lshp'][-1]
            for label, ax in zip(stress, axes):  # , 's'
                for j, ls in zip(range(self.part[part]['nsection']),
                                 ['-', '--']):
                    section = self.part[part]['name'][j]
                    ax.plot(self.part[part]['Lel'],  # /Lnorm
                            1e-6*self.part[part][label][:, j],
                            ls, color='C{}'.format(ci),
                            label=part+'_'+section)
        for ax, label in zip(axes, stress):
            ax.set_ylabel(f'{label} MPa')
            plt.despine()
        for i in range(len(axes) - 1):
            plt.setp(axes[i].get_xticklabels(), visible=False)
        plt.xlabel('part length')
        # plt.sca(axes[1])
        # plt.legend()

    def plot_nodal(self):
        fig, ax = plt.subplots(2, 1, sharex=True, squeeze=True)
        text = linelabel(ax=ax[0], value='', postfix='', Ndiv=5)
        for i, var in enumerate(['x', 'y', 'z']):
            ax[0].plot(self.D[var], 'C{}'.format(i))
            text.add(var)
        text.plot()

        text = linelabel(ax=ax[1], value='', postfix='', Ndiv=5)
        for i, var in enumerate(['tx', 'ty', 'tz']):
            ax[1].plot(self.D[var], 'C{}'.format(i))
            text.add(var)
        text.plot()

    def plot_sections(self, ax=None, theta=0):
        if ax is None:
            ax = Axes3D(plt.figure(figsize=plt.figaspect(1.5)))
        R = geom.rotate(theta, axis='z')  # rotation matrix
        plt.axis('equal')
        plt.axis('off')
        dx_ref = np.array([1, 0, 0], ndmin=2, dtype=float)
        dy_ref = np.array([0, 1, 0], ndmin=2, dtype=float)
        dz_ref = np.array([0, 0, 1], ndmin=2, dtype=float)
        for n in range(self.nel):  # for all elements
            X = np.copy(self.el['X'][n])  # element midpoint
            dx = self.el['dx'][n]  # element tangent
            dy = self.el['dy'][n]  # element normal
            pivot = np.cross(dx_ref, dx)[0]
            pnorm = np.linalg.norm(pivot)
            theta_a = np.arccos(np.dot(dx_ref, np.array(dx.T))[0][0])
            if pnorm == 0:
                pivot = dz_ref
            else:
                pivot /= pnorm
            nodes = self.el['n'][n]
            D = np.zeros(3)
            for i, var in enumerate(['x', 'y', 'z']):
                D[i] = self.D[var][nodes].mean()
            X += self.scale*D  # add dissplacment
            matID = self.el['mat'][n]
            nsection = self.mat[matID]['nsection']
            mat_array = self.mat[matID]['mat_array']
            for nsec in range(nsection):
                pntID = mat_array['pntID'][nsec]
                pnt = self.pnt[pntID]  # section outline
                shape = np.shape(pnt)
                if len(shape) == 3:  # multi part
                    nshape = shape[1]
                else:
                    nshape = 1
                    pnt = np.expand_dims(pnt, 1)
                for ns in range(nshape):
                    y = pnt[0][ns]
                    z = pnt[1][ns]
                    x = np.zeros(np.shape(y))
                    points = geom.qrotate(np.array([x, y, z]).T,
                                          theta_a, xo=[0, 0, 0],
                                          dx=pivot)
                    dy_ref_o = geom.qrotate(dy_ref, theta_a, xo=[0, 0, 0],
                                            dx=pivot)

                    dy_dot = np.dot(dy_ref_o, dy.T).tolist()[0][0]
                    dy_pivot = np.cross(dy_ref_o, dy)[0]
                    if dy_dot > 1:
                        dy_dot = 1
                    if np.linalg.norm(dy_pivot) == 0:
                        dy_pivot = np.array(dx)[0]
                    theta_b = np.arccos(dy_dot)
                    points = geom.qrotate(points, theta_b, xo=[0, 0, 0],
                                          dx=dy_pivot)
                    x, y, z = points[:, 0], points[:, 1], points[:, 2]
                    x += X[0]  # translate
                    y += X[1]
                    z += X[2]
                    Xr = np.dot(np.array([x, y, z]).T, R).T  # rotate patch
                    geom.polyfill3D(Xr[0], Xr[1], Xr[2], ax=ax, alpha=0.5)
        for i, part in enumerate(self.part):
            D = self.part[part]['D']
            D = np.dot(D, R)
            # plot element centre lines
            ax.plot(D[:, 0], D[:, 1], D[:, 2], color=0.5*np.ones(3))
        plt.axis('equal')

    def plot_matrix(self, M):
        plt.figure(figsize=plt.figaspect(1))
        M = np.array(M)
        edgecolors = 'r' if len(M) <= 50 else None
        plt.pcolormesh(abs(M), cmap=plt.cm.gray, vmin=0, vmax=1,
                       edgecolors=edgecolors)
        ax = plt.gca()
        ax.invert_yaxis()
        plt.axis('equal')

    def plot(self, projection='xz', scale_factor=-0.2, select=[]):
        with scale(self.deform, scale_factor):
            plt.figure()
            self.plot_nodes(projection=projection, select=select)
            self.plot_displacment(projection=projection, select=select)
            # self.plot_F(projection=projection)
            plt.axis('off')
            plt.axis('equal')

    def plot_twin(self, scale_factor=10):
        ax = plt.subplots(1, 2, sharex=True, sharey=True,
                          figsize=plt.figaspect(0.75))[1]
        with scale(self.deform, scale_factor):
            for i, projection in enumerate(['xz', 'yz']):
                self.plot_nodes(projection=projection, ax=ax[i])
                self.plot_displacment(projection=projection, ax=ax[i])
                self.plot_F(projection=projection, ax=ax[i])
                ax[i].axis('equal')
                ax[i].axis('off')

    def plot3D(self, ax=None, nR=1, scale_factor=-0.2):
        if ax is None:
            fig = plt.figure(figsize=plt.figaspect(1))
            ax = fig.add_subplot(111, projection='3d')
        Theta = np.linspace(0, 2*np.pi, nR, endpoint=False)
        with scale(self.deform, scale_factor):
            for t in Theta:
                self.plot_sections(ax=ax, theta=t)
        ax.set_xlim(self.xo[0] - self.bb/2, self.xo[0] + self.bb/2)
        ax.set_ylim(self.xo[1] - self.bb/2, self.xo[1] + self.bb/2)
        ax.set_zlim(self.xo[2] - self.bb/2, self.xo[2] + self.bb/2)
        ax.axis('off')
        ax.axis('equal')


class scale:  # tempary deformation of structure for plots

    def __init__(self, deform, scale_factor):
        self.deform = deform
        self.scale_factor = scale_factor

    def __enter__(self):
        self.deform(self.scale_factor)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.deform(1)
