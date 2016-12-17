import numpy as np
import pylab as pl
from scipy.sparse import lil_matrix,csr_matrix
from scipy.sparse.linalg import spsolve
from collections import OrderedDict
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import itertools 
from amigo.geom import vector_length
from scipy.interpolate import interp1d
from amigo.addtext import linelabel
from mpl_toolkits.mplot3d import Axes3D
from nova.coil_cage import coil_cage
from amigo import geom
color = sns.color_palette('Set2',12)

def delete_row_csr(mat, i):
    if not isinstance(mat, csr_matrix):
        raise ValueError('works only for CSR format -- use .tocsr() first')
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])

class FE(object):
    
    def __init__(self,frame='1D',nShape=11):
        self.nShape = nShape  # element shape function resolution
        self.coordinate = ['x','y','z','tx','ty','tz']
        self.initalise_mat()
        self.frame = frame 
        self.set_stiffness()  # set stiffness matrix + problem dofs
        self.initalize_BC()  # boundary conditions
        self.initalise_grid()  # grid
        self.cp = CP(self.dof)  # node coupling

    def initalise_mat(self,nmat_max=10):
        self.nmat_max = nmat_max
        self.nmat = 0
        self.mat = np.zeros((nmat_max),dtype=[('E','float'),('G','float'),
                                              ('J','float'),('Iy','float'),
                                              ('Iz','float'),('A','float'),
                                              ('rho','float')])
        
    def add_mat(self,nmat,**kwargs):
        self.nmat = nmat  # set material number
        if self.nmat >= self.nmat_max:
            err_txt = 'nmat=={:d}>=nmat_max=={:1d}'.format(self.nmat,
                                                           self.nmat_max)
            err_txt += ' increase size of material array (initalise_mat)'
            raise ValueError(err_txt)
        if 'mat' in kwargs:
            mat = kwargs.get('mat')
            for p in mat:
                self.mat[self.nmat][p] = mat[p]
        else:
            for p in self.mat.dtype.names:
                if p in kwargs:
                    self.mat[self.nmat][p] = kwargs.get(p)
            if 'I' in kwargs:  # isotropic second moment
                for I in ['Iy','Iz']:
                    self.mat[self.nmat][I] = kwargs.get('I')
    
    def get_mat(self,nmat=0):
        if self.frame=='1D':
            mat = ['E','Iz']
        elif self.frame=='2D':
            mat = ['E','A','Iz']
        else:
            mat = ['E','A','G','J','Iy','Iz']
        values = []
        for m in mat:
            values.append(self.mat[nmat][m])
        return values
        
    def initalise_grid(self,npart_max=10):
        self.X = []
        self.npart = 0  # part number
        self.part = OrderedDict()  # part ordered dict
        self.nnd = 0  # node number
        self.nel = 0  # element number
        self.el = {}
           
    def add_nodes(self,X):
        self.close_loop = False
        if np.size(X) == 3:
            X = np.reshape(X,(1,3))  # single node
        else:
            if np.linalg.norm(X[0,:]-X[-1,:]) == 0:  # loop
                X = X[:-1,:]  # remove repeated node
                self.close_loop = True
        nX = len(X)        
        self.nndo = self.nnd  # start index
        self.nnd += nX  # increment node index
        if len(self.X) == 0:  # initalise
            self.X = X
            self.nd_topo = np.zeros(nX)
            self.F = np.zeros(nX*self.ndof)
            self.D = {}
            for disp in ['x','y','z','tx','ty','tz']:
                self.D[disp] = np.zeros(nX) 
        else:
            self.X = np.append(self.X,X,axis=0)
            self.nd_topo = np.append(self.nd_topo,np.zeros(nX))
            self.F = np.append(self.F,np.zeros(nX*self.ndof))
            for disp in self.D:
                self.D[disp] = np.append(self.D[disp],np.zeros(nX))
         
    def get_nodes(self):
        n = np.zeros((self.nnd-self.nndo-1,2),dtype='int')
        n[:,1] = np.arange(self.nndo+1,self.nnd)
        n[:,0] = n[:,1]-1
        return n
        
    def add_elements(self,n=[],nmat=0,part_name=''):  
        # list of node pairs shape==[n,2]
        if len(part_name) == 0:
            part_name = 'part_{:1d}'.format(self.npart)
        n = np.array(n)
        if len(n) == 0:  # construct from last input node set
            n = self.get_nodes()
        elif len(np.shape(n)) == 1:  # single dimension node list
            nl = np.copy(n)  # node list
            n = np.zeros((len(n)-1,2),dtype='int')
            n[:,0],n[:,1] = nl[:-1],nl[1:]
        elif np.size(n) == 2:
            n = np.reshape(n,(1,2))
        if len(n) == 0:
            err_txt = 'zero length node pair array'
            raise ValueError(err_txt)
        if self.close_loop:
            n = np.append(n,np.array([n[-1,1],n[0,0]],dtype=int,ndmin=2),
                                     axis=0)
        self.nelo = self.nel  # start index    
        self.nel += len(n)  # element number
        dx = self.X[n[:,1],:]-self.X[n[:,0],:] # element aligned coordinate
        self.check_frame(dx)
        dl = np.linalg.norm(dx,axis=1)  # element length
        norm = np.dot(np.matrix(dl).T,np.ones((1,3)))
        print(dx,dl,norm)
        dx = dx/norm  # unit length
        dz = np.zeros(np.shape(dx))
        dz[:,2] = 1  # points lie in z-plane
        dy = np.cross(dz,dx)  # right hand coordinates
        self.el_append('dx',dx)  # store elements
        self.el_append('dy',dy)
        self.el_append('dz',dz)
        self.el_append('dl',dl)
        self.el_append('mat',nmat*np.ones(len(n),dtype=int))
        self.el_append('n',n)
        self.nd_connect(n)
        self.add_part(part_name)
        
    def nd_connect(self,n):
        for nd in np.ravel(n):
            self.nd_topo[nd] += 1
        
    def el_append(self,key,value):
        if key in self.el:
            self.el[key] = np.append(self.el[key],value,axis=0)
        else:
            self.el[key] = value
            
    def check_name(self,name):  # ensure unique part name
        if name in self.part:
            count = 1
            name = name+'_{:02d}'.format(count)
            while name in self.part:
                count += 1
                name = list(name)
                name[-2:] = '{:02d}'.format(count)
                name = ''.join(name)
                if count == 99:
                    raise ValueError('too many duplicate part names')
        return name
        
    def add_part(self,name,iel=[]):
        self.npart += 1
        name = self.check_name(name)
        self.part[name] = {}  # initalise dict
        if len(iel) == 0:  # construct from last input element set
            iel = np.arange(self.nelo,self.nel)
        self.part[name]['el'] = iel  # construct from input element list
        self.part[name]['nel'] = len(iel)
        self.part[name]['L'] = np.append(0,np.cumsum(self.el['dl'][iel]))
        n = 2*(len(iel))
        L,nd = np.zeros(n),np.zeros(n)
        L[::2] = self.part[name]['L'][:-1]
        L[1::2] = self.part[name]['L'][1:]
        L /= self.part[name]['L'][-1]  # normalise
        nd[::2] = self.el['n'][iel,0]
        nd[1::2] = self.el['n'][iel,1]
        self.part[name]['Ln'] = interp1d(L,nd)  # node interpolator

    def initalize_BC(self):
        self.BC = OrderedDict()
        self.BC['fix'] = np.array([]) # type:[node,...] 
        self.BC['pin'] = np.array([]) # type:[node,...] 
        for key in self.dof:
            self.BC[key] = np.array([])# type:[node,...] 
            
    def initalize_shape_coefficents(self,nShape=21):
        self.nShape = nShape
        nsh = self.nel
        self.shape = {}
        self.shape['x'] = np.zeros(nsh)
        for label in ['u','d2u','U']:
            self.shape[label] = np.zeros((nsh,3,self.nShape))
        self.S = {}
        self.S['s'] = np.linspace(0,1,nShape)  # inter-element spacing
        self.S['Nv'] = np.zeros((nShape,4))  #  Hermite shape functions (disp)
        self.S['Nv'][:,0] = 1-3*self.S['s']**2+2*self.S['s']**3
        self.S['Nv'][:,1] = self.S['s']-2*self.S['s']**2+self.S['s']**3
        self.S['Nv'][:,2] = 3*self.S['s']**2-2*self.S['s']**3
        self.S['Nv'][:,3] = -self.S['s']**2+self.S['s']**3
        self.S['Nv_dL'] = np.zeros((nShape,2)) 
        self.S['Nv_dL'][:,0] = np.copy(self.S['Nv'][:,1])
        self.S['Nv_dL'][:,1] = np.copy(self.S['Nv'][:,3])
        self.S['Nd2v'] = np.zeros((nShape,4))  #  Hermite functions (curve)
        self.S['Nd2v'][:,0] = -6+12*self.S['s']
        self.S['Nd2v'][:,1] = -4+6*self.S['s']
        self.S['Nd2v'][:,2] = 6-12*self.S['s']
        self.S['Nd2v'][:,3] = -2+6*self.S['s']
        self.S['Nd2v_dL'] = np.zeros((nShape,2)) 
        self.S['Nd2v_dL'][:,0] = np.copy(self.S['Nd2v'][:,1])
        self.S['Nd2v_dL'][:,1] = np.copy(self.S['Nd2v'][:,3])

    def interpolate(self):
        for el in range(self.nel):
            d = self.displace(el)  # local 12 dof nodal displacment 
            u = np.zeros(2)
            for i in range(2):  # axial displacment
                u[i] = d[6*i]
            for i,j in enumerate([1,3]):  # adjust for element length
                self.S['Nv'][:,j] = self.S['Nv_dL'][:,i]*self.el['dl'][el]
                self.S['Nd2v'][:,j] = self.S['Nd2v_dL'][:,i]*self.el['dl'][el]
            v = np.zeros((self.nShape,2))
            d2v = np.zeros((self.nShape,2))
            for i,label in enumerate([['y','tz'],['z','ty']]):
                if label[0] in self.disp:
                    d1D = self.displace_1D(d,label)
                    v[:,i] = np.dot(self.S['Nv'],d1D)
                    d2v[:,i] = np.dot(self.S['Nd2v'],d1D)/self.el['dl'][el]**2
            self.store_shape(el,u,v,d2v)
        self.shape_part()  # set deformed part shape

    def store_shape(self,el,u,v,d2v):
        nS = self.nShape
        self.shape['u'][el,0] = np.linspace(u[0],u[1],nS)
        self.shape['u'][el,1] = v[:,0]  # displacment
        self.shape['u'][el,2] = v[:,1]
        self.shape['U'][el] = np.dot(self.T3[:,:,el].T,  #  to global
                                           self.shape['u'][el,:,:])
        for j in range(3):
            n = self.el['n'][el]
            self.shape['U'][el,j,:] += np.linspace(self.X[n[0],j],
                                                   self.X[n[1],j],nS)
        self.shape['d2u'][el,1] = d2v[:,0]  # curvature
        self.shape['d2u'][el,2] = d2v[:,1]
        
    def shape_part(self):
        for part in self.part:
            nS = self.part[part]['nel']*(self.nShape-1)+1
            self.part[part]['l'] = np.linspace(0,1,nS)
            for label in ['u','U','d2u']:
                self.part[part][label] = np.zeros((nS,3))
            for nel,el in enumerate(self.part[part]['el']):
                i = nel*(self.nShape-1)
                for label in ['u','U','d2u']:
                    self.part[part][label][i:i+self.nShape,:] = \
                    self.shape[label][el,:,:].T
                
    def displace(self,el):
        d = np.zeros(12)
        for i,n in enumerate(self.el['n'][el]):  # each node in element
            for label in self.disp:
                index = self.coordinate.index(label)
                d[i*6+index] = self.D[label][n]
        for i in range(4):  # transform to local coordinates
            d[i*3:i*3+3] = np.linalg.solve(self.T3[:,:,el].T,d[i*3:i*3+3]) 
        return d
            
    def displace_1D(self,d,label):
        d1D = np.zeros(4)
        for i in range(2):  # node
            for j in range(2):  # label
                index = self.coordinate.index(label[j])
                d1D[2*i+j] = d[6*i+index]
                if label[j] == 'ty':  # displacment in z
                    d1D[2*i+j] *= -1
        return d1D
        
    def set_stiffness(self):  # set element stiffness matrix
        if self.frame=='1D':
            dof = ['v','rz']
            disp = ['y','tz']
            load = ['fy','mz']
            stiffness = self.stiffness_1D
        elif self.frame=='2D':
            dof = ['u','v','rz']
            disp = ['x','y','tz']
            load = ['fx','fy','mz']
            stiffness = self.stiffness_2D
        else:
            dof = ['u','v','w','rx','ry','rz']
            disp = ['x','y','z','tx','ty','tz']
            load = ['fx','fy','fz','mx','my','mz']
            stiffness = self.stiffness_3D  
        self.dof = dof
        self.disp = disp
        self.load = load
        self.ndof = len(dof)  # degrees of fredom per node
        self.stiffness = stiffness
        
    def stiffness_1D(self,el):  # dof [v,rz]
        a = self.el['dl'][el]/2
        E,Iz = self.get_mat()
        k = E*Iz/(2*a**3)*np.matrix([[3,  3*a,   -3,  3*a],
                                     [3*a,4*a**2,-3*a,2*a**2],
                                     [-3,-3*a,    3, -3*a],
                                     [3*a,2*a**2,-3*a,4*a**2]])    
        return k

    def stiffness_2D(self,el):  # dof [u,v,rz]
        a = self.el['dl'][el]/2
        E,A,Iz = self.get_mat()
        k = np.matrix([[A*E/(2*a), 0,               0,
                       -A*E/(2*a), 0,               0],
                       [0,         3*E*Iz/(2*a**3), 3*E*Iz/(2*a**2),
                        0,        -3*E*Iz/(2*a**3), 3*E*Iz/(2*a**2)],
                       [0,         3*E*Iz/(2*a**2), 2*E*Iz/a,
                        0,        -3*E*Iz/(2*a**2), E*Iz/a],
                       [-A*E/(2*a),0,               0,
                         A*E/(2*a),0,               0],
                       [0,        -3*E*Iz/(2*a**3),-3*E*Iz/(2*a**2),
                        0,         3*E*Iz/(2*a**3),-3*E*Iz/(2*a**2)],
                       [0,         3*E*Iz/(2*a**2), E*Iz/a,
                        0,        -3*E*Iz/(2*a**2), 2*E*Iz/a]])
        k = self.rotate_matrix(k,el)  # transfer to global coordinates
        return k
        
    def stiffness_3D(self,el):  # dof [u,v,w,rx,ry,rz]
        a = self.el['dl'][el]/2
        E,A,G,J,Iy,Iz = self.get_mat()
        k = np.matrix([[A*E/(2*a), 0,               0,
                        0,         0,               0,
                        -A*E/(2*a),0,               0,
                        0,         0,               0],
                       [0,         3*E*Iz/(2*a**3), 0,
                        0,         0,               3*E*Iz/(2*a**2),
                        0,        -3*E*Iz/(2*a**3), 0,
                        0,         0,               3*E*Iz/(2*a**2)],
                       [0,         0,               3*E*Iy/(2*a**3),
                        0,        -3*E*Iy/(2*a**2), 0,
                        0,         0,              -3*E*Iy/(2*a**3),
                        0,        -3*E*Iy/(2*a**2), 0],
                       [0,         0,               0,
                        G*J/(2*a), 0,               0,
                        0,         0,               0,
                        -G*J/(2*a),0,               0],
                       [0,         0,              -3*E*Iy/(2*a**2),
                        0,         2*E*Iy/a,        0,
                        0,         0,               3*E*Iy/(2*a**2),
                        0,         E*Iy/a,          0],
                       [0,         3*E*Iz/(2*a**2), 0,
                        0,         0,               2*E*Iz/a,
                        0,        -3*E*Iz/(2*a**2), 0,
                        0,         0,               E*Iz/a],
                       [-A*E/(2*a),0,               0,
                        0,         0,               0,
                        A*E/(2*a), 0,               0,
                        0,         0,               0],
                       [0,        -3*E*Iz/(2*a**3), 0,
                        0,         0,              -3*E*Iz/(2*a**2),
                        0,         3*E*Iz/(2*a**3), 0,
                        0,         0,              -3*E*Iz/(2*a**2)],
                       [0,         0,              -3*E*Iy/(2*a**3),
                        0,         3*E*Iy/(2*a**2), 0,
                        0,         0,               3*E*Iy/(2*a**3),
                        0,         3*E*Iy/(2*a**2), 0],
                       [0,         0,               0,
                       -G*J/(2*a), 0,               0,
                        0,         0,               0,
                        G*J/(2*a), 0,               0],
                       [0,         0,              -3*E*Iy/(2*a**2),
                        0,         E*Iy/a,          0,
                        0,         0,               3*E*Iy/(2*a**2),
                        0,         2*E*Iy/a,        0],
                       [0,         3*E*Iz/(2*a**2), 0,
                        0,         0,               E*Iz/a,
                        0,        -3*E*Iz/(2*a**2), 0,
                        0,         0,               2*E*Iz/a]])
        k = self.rotate_matrix(k,el)  # transfer to global coordinates
        return k

    def check_frame(self,dx):
        if self.frame=='1D':
            if np.sum(abs(np.diff(dx[:,1]))) > 0:
                err_txt = 'error: rotation in z for 1D element'
                err_txt += ' - select frame=2D'
                raise ValueError(err_txt)
        elif self.frame=='2D':
            if np.sum(abs(np.diff(dx[:,2]))) > 0:
                err_txt = 'error: rotation in y for 2D element'
                err_txt += ' - select frame=3D'
                raise ValueError(err_txt)
                
    def update_rotation(self):
        if not hasattr(self,'T3'):
            self.get_rotation()
        elif np.shape(self.T3)[2] != self.nel:
            self.get_rotation()
        
    def get_rotation(self):
        dx_,dy_,dz_ = self.el['dx'],self.el['dy'],self.el['dz']
        self.T3 = np.zeros((3,3,self.nel))
        X = np.array([1,0,0]).T
        Y = np.array([0,1,0]).T
        Z = np.array([0,0,1]).T
        for i,(dx,dy,dz) in enumerate(zip(dx_,dy_,dz_)):
            self.T3[:,:,i] = \
            np.matrix([[np.dot(dx,X),np.dot(dx,Y),np.dot(dx,Z)],
                       [np.dot(dy,X),np.dot(dy,Y),np.dot(dy,Z)],
                       [np.dot(dz,X),np.dot(dz,Y),np.dot(dz,Z)]])
        
    def rotate_matrix(self,M,el):
        T = np.zeros((2*self.ndof,2*self.ndof))                
        for i in range(int(2/3*self.ndof)):
            T[3*i:3*i+3,3*i:3*i+3] = self.T3[:,:,el]
        k = T.T*M*T
        return k
        
    def check_input(self,vector,label,terminate=True):  # 'dof','disp','load'
        attributes = getattr(self,vector)
        error_code = 0
        if label not in attributes and not \
        (vector == 'dof' and (label == 'fix' or label == 'pin')):
            error_code = 1
            if terminate:
                err_txt = 'attribute \''+label+'\' not present'
                err_txt += 'in frame='+self.frame
                err_txt +=' ['+', '.join(attributes)+']'
                raise ValueError(err_txt)
        return error_code
        
    def add_load(self,**kwargs):  # distribute load to adjacent nodes
        csys = ''  # coodrdinate system unset 
        if 'f' in kwargs or 'F' in kwargs:  # point load
            load_type = 'point'
            if 'L' in kwargs and 'part' in kwargs:  # identify element
                L,part = kwargs.get('L'),kwargs.get('part')
                Ln = self.part[part]['Ln'](L)  # length along part
                el = int(np.floor(Ln))  # element index
                s = Ln-el  # fraction along element
                if L == 1:
                    el -= 1
                    s = 1
            elif 'el' in kwargs:
                el = kwargs.get('el')
                s = kwargs.get('s',0.5)
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
            s = kwargs.get('s',0.5)
            if 'W' in kwargs:
                csys = 'global'
                w = np.array(kwargs.get('W'))
            elif 'w' in kwargs:                
                csys = 'local'
                w = np.array(kwargs.get('w'))
            f = w*self.el['dl'][el]
        if len(csys) == 0:
            raise ValueError('load vector unset')
        elif csys == 'global':   
            f = np.linalg.solve(self.T3[:,:,el].T,f)  # rotate to local csys
        fn = np.zeros((6,2))  # 6 dof local nodal load vector
        for i,label in enumerate(['fx','fy','fz']):  # split point load to F,M
            if label in self.load:
                if label == 'fx':
                    fn[i,0] = (1-s)*f[i]
                    fn[i,1] = s*f[i]
                else:  # fy,fz
                    fn[i,0] = (1-s)**2*(1+2*s)*f[i]
                    fn[i,1] = s**2*(3-2*s)*f[i]
                    mi = 5 if label == 'fy' else 4  # moment index
                    fn[mi,0] = f[i]*(1-s)**2*s*self.el['dl'][el]
                    fn[mi,1] = -f[i]*s**2*(1-s)*self.el['dl'][el]
                    if load_type == 'dist':
                        fn[mi,:] *= 8/12  # reduce moment for distributed load
            else:
                if abs(f[i]) > 1e-12:            
                    err_txt = 'non zero load \''+label+'\''
                    err_txt += ' not present in frame='+self.frame
                    err_txt +=' ['+', '.join(self.load)+']'
                    raise ValueError(err_txt)
        for i,node in enumerate(self.el['n'][el]):  # each node
            if csys == 'global':
                F = np.zeros((6))
                for j in range(2):  # force,moment
                    F[j*3:j*3+3] = np.dot(self.T3[:,:,el].T,fn[j*3:j*3+3,i])
            else:
                F = fn[:,i]
            for index,label in enumerate(['fx','fy','fz','mx','my','mz']):
                if label in self.load:
                    self.add_nodal_load(node,label,F[index])
        
    def add_nodal_load(self,node,label,load):
        self.check_input('load',label)
        index = self.load.index(label)
        self.F[node*self.ndof+index] += load
        
    def add_weight(self):
        self.update_rotation()  # check / update rotation matrix
        for part in self.part:
            for el in self.part[part]['el']:
                nm = self.el['mat'][el]  # material index
                w = -9.81*self.mat['rho'][nm]*self.mat['A'][nm]
                self.add_load(el=el,W=[0,w,0])  # self weight
                
    def add_tf_load(self,config,tf,Bpoint,parts=['loop','nose'],
                    method='function'):
        cage = coil_cage(nTF=tf.profile.nTF,rc=tf.rc,
                         plasma={'config':config},coil={'cl':tf.x['cl']})
        # eq.Bpoint == point calculated method (slow)
        # sf.Bpoint == spline interpolated method (fast)
        if 'streamfunction' in Bpoint.__str__():  
            topright = Bpoint((np.max(tf.x['cl']['r']),
                               np.max(tf.x['cl']['z'])),
                              check_bounds=True)    
            bottomleft = Bpoint((np.max(tf.x['cl']['r']),
                                 np.max(tf.x['cl']['z'])),
                                check_bounds=True)
            if not(topright and bottomleft):
                errtxt = 'TF coil extends outside Bpoint interpolation grid\n'
                errtxt = 'extend sf grid\n'
                raise ValueError(errtxt)
                
        if method == 'function':  # calculate tf feild as fitted 1/r function
            i = np.argmax(tf.x['cl']['z'])
            ro,zo = tf.x['cl']['r'][i],tf.x['cl']['z'][i]
            bm = -ro*cage.point((ro,0,zo),variable='feild')[1]  # TF moment
        elif method == 'BS':  # calculate tf feild with full Biot-Savart
            Rp = geom.rotate(np.pi/2,'x')
            Rm = geom.rotate(-np.pi/2,'x')
        else:
            errtxt = 'invalid tf feild method {}\n'.format(method)
            errtxt += 'select method from \'function\' or \'BS\'\n'
            raise ValueError(errtxt)
        self.update_rotation()  # check / update rotation matrix
        for part in parts:
            for el in self.part[part]['el']:
                n = self.el['n'][el]  # node index pair
                point = np.zeros(3)
                for i in range(3):  # calculate load at element mid-point
                    point[i] = np.mean(self.X[n,i])  
                if method == 'function':
                    b = np.zeros(3)
                    b[2] = bm/point[0]  # TF feild (fast version)
                else:
                    b = np.dot(Rm,cage.point(np.dot(Rp,point),  # (slow)
                                             variable='feild'))
                b[:2] += Bpoint((point[:2]))  # PF feild (sf-fast, eq-slow)
                w = np.cross(self.el['dx'][el],b)
                self.add_load(el=el,W=w)  # bursting/toppling load

    def check_part(self,part):
        if part not in self.part:
            err_txt = part+' not present in ['+', '.join(self.part.keys())+']'
            raise ValueError(err_txt)
        
    def part_nodes(self,index,part,ends=2):  # el ends, 0==start,1==end,2==both
        if len(part) > 0:  # element index relitive to part
            index_type = 'element'
        else:  # node index
            index_type = 'node'
        if index_type == 'element':
            if index == 'all':
                if len(part) > 0:  # node index relitive to part
                    elements = list(range(self.part[part]['nel']))
            else:
                elements = index
                if not isinstance(elements,list):  # convert to list
                    elements = [elements]
            nodes = -1*np.ones(2*len(elements))  # node array
            for i,element in enumerate(elements):
                el = self.part[part]['el'][element]
                if ends==0 or ends==2:  # start or both
                    nodes[2*i] = self.el['n'][el,0]
                if ends==1 or ends==2:  # end or both
                    nodes[2*i+1] = self.el['n'][el,1]
            nodes = np.unique(nodes)
            nodes = nodes[nodes>-1]
        elif index_type == 'node':
            nodes = index
            if not isinstance(nodes,list):  # convert to list
                nodes = [nodes]
        return nodes
        
    def addBC(self,dof,index,part='',ends=2,terminate=True):
        # part='' then index=nodes, part=part then index=elements
        self.check_part(part)
        nodes = self.part_nodes(index,part,ends=ends)  # select nodes
        if isinstance(dof,str):  # convert to list
            dof = [dof]
        for constrn in dof:  # constrain nodes
            error_code = self.check_input('dof',constrn,terminate=terminate)
            if not error_code:
                self.BC[constrn] = np.append(self.BC[constrn],np.array(nodes))

    def extractBC(self):  # extract matrix indicies for constraints
        self.BCindex = np.array([])
        for j,key in enumerate(self.BC.keys()):
            ui = self.ndof*self.BC[key]  # node dof
            if len(ui) > 0:
                if key == 'fix':  # fix all dofs
                    for i in range(self.ndof):
                        self.BCindex = np.append(self.BCindex,ui+i)
                elif key == 'pin':  # pin node (u,v,w)
                    for i in range(int(self.ndof/2)):
                        self.BCindex = np.append(self.BCindex,ui+i)
                else:
                    # skip-fix,pin
                    self.BCindex = np.append(self.BCindex,ui+j-2)  
        self.BCindex = list(map(int,set(self.BCindex)))
        
    def extractND(self):
        for nd in np.where(self.nd_topo==0)[0]:  # remove unconnected nodes
            self.BCindex = np.append(self.BCindex,
                                     nd*self.ndof+np.arange(0,self.ndof))
        
    def assemble(self):  # assemble global stiffness matrix
        self.Ko = np.zeros((self.nK,self.nK))  # matrix without constraints
        for el in range(self.nel):
            ke = self.stiffness(el)            
            ke_index = itertools.product([0,self.ndof],repeat=2)
            ko_index = itertools.product(self.ndof*self.el['n'][el],repeat=2)
            for i,j in zip(ko_index,ke_index):
                self.Ko[i[0]:i[0]+self.ndof,i[1]:i[1]+self.ndof] += \
                ke[j[0]:j[0]+self.ndof,j[1]:j[1]+self.ndof]
        self.K = np.copy(self.Ko)
        
    def setBC(self):  # colapse constrained dof
        self.extractBC()  # remove BC dofs
        self.extractND()  # remove unconnected nodes
        self.Fo = np.copy(self.F)
        self.Dindex = np.arange(0,self.nK)  # all nodes (include unconnected)
        if len(self.BCindex) > 0:
            self.F = np.delete(self.F,self.BCindex)
            for i in range(2):  # rows and cols
                self.K = np.delete(self.K,self.BCindex,i)
            self.Dindex = np.delete(self.Dindex,self.BCindex)

    def solve(self):
        self.nK = int(self.nnd*self.ndof)  # stiffness matrix 
        self.initalize_shape_coefficents(nShape=self.nShape)   
        self.update_rotation()  # evaluate/update rotation matricies
        self.assemble()  # assemble stiffness matrix
        
        ###
        self.extractBC()  # remove BC dofs
        self.extractND()  # remove unconnected nodes
        
        self.nd = {}  # node index
        self.nd['do'] = np.arange(0,self.nK)  # all nodes (include unconnected)
        self.nd['mask'] = np.in1d(self.nd['do'],self.BCindex)  # condense mask
        
        # add to condensed mask

        self.cp.extract_nodes()
        
        print(self.nd['mask'])
        self.nd['mask'] = self.nd['mask'] | np.in1d(self.nd['do'],
                                                     self.cp.nodes['dc'])
        print(self.nd['mask'])
        ###

        self.nd['dc'] = self.nd['do'][self.nd['mask']]  # condensed
        self.nd['dr'] = self.nd['do'][~self.nd['mask']]  # retained

        self.nd['nc'] = np.sum(self.nd['mask'])
        self.nd['nr'] = np.sum(~self.nd['mask'])



        T = np.append(np.identity(self.nd['nr']),
                      np.zeros((self.nd['nc'],self.nd['nr'])),axis=0)
        
        print(np.shape(T))
        print(T)
        
        # sort K
        K = np.append(self.K[self.nd['dr'],:],self.K[self.nd['dc'],:],axis=0)  
        K = np.append(K[:,self.nd['dr']],self.K[:,self.nd['dc']],axis=1)
        
        # sort F
        F = np.append(self.F[self.nd['dr']],self.F[self.nd['dc']],axis=0)  
        
        F = np.dot(T.T,F)
        
        K = np.dot(np.dot(T.T,K),T)
        print('F',F)
        print(self.F)
        ###
        
        
        self.setBC()  # remove constrained equations from stiffness + load 
        
        print('K-',K)
        print('K',self.K)
        self.Dn = np.zeros(self.nK)
        
        self.Dn[self.Dindex] = np.linalg.solve(self.K,self.F) # global
        for i,disp in enumerate(self.disp):
            self.D[disp] = self.Dn[i::self.ndof]
        self.interpolate()
        
    def plot_3D(self,ms=5):
        fig = pl.figure()
        ax = fig.gca(projection='3d')
        ax.plot(self.X[:,0],self.X[:,2],self.X[:,1],'o',markersize=ms,
                color=0.75*np.ones(3))
        for i,(part,c) in enumerate(zip(self.part,color)):
            ax.plot(self.part[part]['U'][:,0],
                    self.part[part]['U'][:,2],
                    self.part[part]['U'][:,1],color=c)
                
        #ax.axis('off')
                
    def plot_twin(self,scale=5e-1,ms=5):
        pl.figure(figsize=(10,5))
        ax1 = pl.subplot(121)
        ax2 = pl.subplot(122, sharey=ax1)
        ax1.set_aspect('equal')
        ax1.set_axis_off()
        ax2.set_aspect('equal')
        ax2.set_axis_off()
        ax1.plot(self.X[:,0],self.X[:,1],'o',markersize=ms,
                color=0.75*np.ones(3))
        ax2.plot(self.X[:,2],self.X[:,1],'o',markersize=ms,
                color=0.75*np.ones(3))
        for i,(part,c) in enumerate(zip(self.part,color)):
            ax1.plot(self.part[part]['U'][:,0],
                     self.part[part]['U'][:,1],color=c)
            ax2.plot(self.part[part]['U'][:,2],
                     self.part[part]['U'][:,1],color=c)
        for i,(X,dx,dy,dz) in enumerate(zip(self.X,self.D['x'],
                                          self.D['y'],self.D['z'])):
            j = i*self.ndof
            if np.linalg.norm([self.Fo[j],self.Fo[j+1]]) != 0:
                ax1.arrow(X[0]+dx,X[1]+dy,
                          scale*self.Fo[j],scale*self.Fo[j+1],
                          head_width=0.15,head_length=0.3) 
            if np.linalg.norm([self.Fo[j+2],self.Fo[j+1]]) != 0:              
                ax2.arrow(X[2]+dz,X[1]+dy,
                          scale*self.Fo[j+2],scale*self.Fo[j+1],
                          head_width=0.15,head_length=0.3) 

    def plot_nodes(self,ms=5):
        pl.plot(self.X[:,0],self.X[:,1],'o',markersize=ms,
                color=0.75*np.ones(3))
        for part,c in zip(self.part,color):
            for el in self.part[part]['el']:
                nd = self.el['n'][el]
                pl.plot([self.X[nd[0],0],self.X[nd[1],0]],
                        [self.X[nd[0],1],self.X[nd[1],1]],color=c,alpha=0.5)
        sns.despine()
        
    def plot_F(self,scale=1):
        for i,(X,dx,dy) in enumerate(zip(self.X,self.D['x'],self.D['y'])):
            j = i*self.ndof
            if self.frame == '1D':
                F = [0,self.Fo[j]]
            else:
                F = [self.Fo[j],self.Fo[j+1]]
            nF = np.linalg.norm(F)
            if nF != 0:
                pl.arrow(X[0]+dx,X[1]+dy,scale*F[0],scale*F[1],
                         head_width=scale*0.2*nF,head_length=scale*0.3*nF)        
        
    def plot_displacment(self):
        for i,(part,c) in enumerate(zip(self.part,color)):
            pl.plot(self.part[part]['U'][:,0],
                    self.part[part]['U'][:,1],color=c)
        pl.axis('equal')
        
    def plot_curvature(self):
        pl.figure(figsize=([4,3*12/16]))
        text = linelabel(value='',postfix='',Ndiv=5) 
        part = self.part #  ['loop','nose']  # 
        for i,(part,c) in enumerate(zip(part,color)):
            pl.plot(self.part[part]['l'],
                    self.part[part]['d2u'][:,2],'--',color=c)
            pl.plot(self.part[part]['l'],
                    self.part[part]['d2u'][:,1],color=c)
            text.add(part)
        text.plot()
        sns.despine()
        pl.xlabel('part length')
        pl.ylabel('part curvature')
        
class CP(object):  # couple node dofs
    
    def __init__(self,model_dof):
        self.n = 0
        self.mpc = OrderedDict()  # multi-point constraint
        self.model_dof = model_dof  # model dof list
        self.ndof = len(model_dof)
        
    def add(self,nodes,dof='fix',nset='next'):  # first node retained
        if nset == 'next':  # (default)
            self.n += 1
        elif isinstance(nset,int):
            self.n = nset
        else:
            errtxt = 'nset must be interger or string, \'high\' or \'next\''
            raise ValueError(errtxt)
        if self.n == 0:
            self.n = 1
        self.label = 'cp{:d}'.format(self.n)  # cp set label
        if self.label not in self.mpc:  # create dof set
            dof = self.extract_dof(dof)
            self.mpc[self.label] = {'dof':dof,'nodes':np.array([])}
        self.mpc[self.label]['nodes'] = \
        np.append(self.mpc[self.label]['nodes'],nodes)
        self.check()
            
    def extract_dof(self,dof):
        if isinstance(dof,str):  # convert to list
                dof = [dof]
        if dof[0] == 'fix':  # fix all dofs
            dof = self.model_dof
        elif dof[0] == 'pin':  # fix translational dofs
            dof = [dof for dof in self.model_dof if 'r' not in dof]
        elif dof[0][0] == 'n':  # free rotation 'nrz'
            mask = np.in1d(self.model_dof,dof[0])
            dof = self.model_dof[~mask]
        else:
            dof = dof
        return dof
        
    def extract_nodes(self):
        self.nodes = {'dr':np.array([],dtype=int),'dc':np.array([],dtype=int)}
        for i,name in enumerate(self.mpc):
            dof = self.mpc[name]['dof']
            row = self.mpc[name]['nodes']*self.ndof  # retain first node
            for j,mdof in enumerate(self.model_dof):
                if mdof in dof:
                    self.nodes['dr'] = np.append(self.nodes['dr'],row[0]+j)
                    self.nodes['dc'] = np.append(self.nodes['dc'],row[1:]+j)
        print(self.nodes)
                    
                
            
    def check(self):  # check for node repeats in constraints
        nodes = np.array([])
        for label in self.mpc:
            nodes = np.append(nodes,self.mpc[label]['nodes'])
        if len(nodes) != len(np.unique(nodes)):
            errtxt = 'repeated node in cp defnition:\n'
            for label in self.mpc:
                errtxt += label+' '+str(self.mpc[label]['nodes'])+'\n'
            raise ValueError(errtxt)
            
            