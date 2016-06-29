import numpy as np
import pylab as pl
from scipy.sparse import lil_matrix,csr_matrix
from scipy.sparse.linalg import spsolve
from collections import OrderedDict
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

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

# element stiffness matrix
class FE(object):
    
    def __init__(self,frame='1D',*args):
        self.coordinate = ['x','y','z','tx','ty','tz']
        self.set_mat(args)
        self.frame = frame 
        self.set_stiffness()  # set stiffness matrix + problem dofs
        self.initalize_BC()  # boundary conditions

    def initalise(self,nShape=21):
        self.initalize_F()  # load
        self.initalize_D()  # displacment
        self.initalize_shape_coefficents(nShape=nShape)  # shape function coefficients
        
    def set_mat(self,*args):
        properties = ['E','G','J','Iy','Iz','A']
        if 'mat' in args:
            self.mat = args.get('mat')
        else:
            self.mat = {}
            for p in properties:
                self.mat[p] = 1
        for p in properties:
            if p in args:
                self.mat[p] = args.get(p)
                
    def get_mat(self):
        if self.frame=='1D':
            mat = ['E','Iz']
        elif self.frame=='2D':
            mat = ['E','A','Iz']
        else:
            mat = ['E','A','G','J','Iy','Iz']
        values = []
        for m in mat:
            values.append(self.mat[m])
        return values
        
    def grid(self,X):
        from amigo.geom import vector_length
        from scipy.interpolate import interp1d
        self.X = X
        self.nel = np.shape(self.X)[0]-1  # element number
        self.nK = int((self.nel-1)*self.ndof+2*self.ndof)  # stiffness matrix
        
        self.L = vector_length(self.X,norm=False)  # loop length
        self.Ln = interp1d(self.L/self.L[-1],range(self.nel+1))  # normalised
        dx = np.array(np.diff(self.X,axis=0))  # element aligned coordinate
        self.check_frame(dx)
        dl = np.linalg.norm(dx,axis=1)  # element length
        norm = np.dot(np.matrix(dl).T,np.ones((1,3)))
        dx /= norm  # unit length

        dz = np.zeros(np.shape(dx))
        dz[:,2] = 1  # points lie in z-plane
        dy = np.cross(dz,dx)  # right hand coordinates
        self.get_rotation(dx,dy,dz)  # evaluate rotation matricies
        self.dX,self.dY,self.dZ,self.dL = dx,dy,dz,dl
        
    def initalize_D(self):  # initalize displacment vector [u,v,w,rx,ry,rz]
        self.D = {}
        for disp in ['x','y','z','tx','ty','tz']:
            self.D[disp] = np.zeros(self.nel+1)
            
    def initalize_F(self):
        self.F = np.zeros(self.nK)  # loading vector

    def initalize_BC(self):
        self.BC = OrderedDict()
        self.BC['fix'] = np.array([])# type:[node,...] 
        for key in self.dof:
            self.BC[key] = np.array([])# type:[node,...] 
            
    def initalize_shape_coefficents(self,nShape=21):
        self.nShape = nShape
        nsh = 1+(self.nShape-1)*self.nel
        self.shape = {}
        self.shape['x'] = np.zeros(nsh)
        for label in ['u','du','d2u','U']:
            self.shape[label] = np.zeros((nsh,3))
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
        self.S['Nd2v'] = np.zeros((nShape,4))  #  Hermite shape functions (curve)
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
                self.S['Nv'][:,j] = self.S['Nv_dL'][:,i]*self.dL[el]
                self.S['Nd2v'][:,j] = self.S['Nd2v_dL'][:,i]*self.dL[el]
            v = np.zeros((self.nShape,2))
            d2v = np.zeros((self.nShape,2))
            for i,label in enumerate([['y','tz'],['z','ty']]):
                if label[0] in self.disp:
                    d1D = self.displace_1D(d,label)
                    v[:,i] = np.dot(self.S['Nv'],d1D)
                    d2v[:,i] = np.dot(self.S['Nd2v'],d1D)/self.dL[el]**2
            self.store_shape(el,u,v,d2v)

    def store_shape(self,el,u,v,d2v):
        i = el*(self.nShape-1)
        nS = self.nShape
        self.shape['x'][i:i+nS] = np.linspace(self.L[el],self.L[el+1],nS)
        self.shape['u'][i:i+nS,0] = np.linspace(u[0],u[1],nS)
        self.shape['u'][i:i+nS,1] = v[:,0]  # displacment
        self.shape['u'][i:i+nS,2] = v[:,1]
        self.shape['U'][i:i+nS,:] = np.dot(self.T3[:,:,el].T,\
        self.shape['u'][i:i+nS,:].T).T #  transform to global
        for j in range(3):
            self.shape['U'][i:i+nS,j] += np.linspace(self.X[el,j],
                                                     self.X[el+1,j],nS)
        self.shape['d2u'][i:i+nS,1] = d2v[:,0]  # curvature
        self.shape['d2u'][i:i+nS,2] = d2v[:,1]
    
    def displace(self,el):
        d = np.zeros(12)
        for i in range(2):  # each node in element
            for label in self.disp:
                index = self.coordinate.index(label)
                d[i*6+index] = self.D[label][el+i]
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
        a = self.dL[el]/2
        E,Iz = self.get_mat()
        k = E*Iz/(2*a**3)*np.matrix([[3,  3*a,   -3,  3*a],
                                     [3*a,4*a**2,-3*a,2*a**2],
                                     [-3,-3*a,    3, -3*a],
                                     [3*a,2*a**2,-3*a,4*a**2]])    
        return k

    def stiffness_2D(self,el):  # dof [u,v,rz]
        a = self.dL[el]/2
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
        a = self.dL[el]/2
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
        
    def get_rotation(self,dX,dY,dZ):
        self.T3 = np.zeros((3,3,self.nel))
        X = np.array([1,0,0]).T
        Y = np.array([0,1,0]).T
        Z = np.array([0,0,1]).T
        for i,(dx,dy,dz) in enumerate(zip(dX,dY,dZ)):
            self.T3[:,:,i] = np.matrix([[np.dot(dx,X),np.dot(dx,Y),np.dot(dx,Z)],
                                       [np.dot(dy,X),np.dot(dy,Y),np.dot(dy,Z)],
                                       [np.dot(dz,X),np.dot(dz,Y),np.dot(dz,Z)]])
        
    def rotate_matrix(self,M,el):
        T = np.zeros((2*self.ndof,2*self.ndof))                
        for i in range(int(2/3*self.ndof)):
            T[3*i:3*i+3,3*i:3*i+3] = self.T3[:,:,el]
        k = T.T*M*T
        return k
        
    def assemble(self):  # assemble global stiffness matrix
        self.Ko = np.zeros((self.nK,self.nK))  # matrix without constraints
        for el in range(self.nel):
            ke = self.stiffness(el)
            j = el*self.ndof  # node index here
            #####    ######
            self.Ko[j:j+2*self.ndof,j:j+2*self.ndof] += ke
        self.K = np.copy(self.Ko)
        
    def check_input(self,vector,label,terminate=True):  # 'dof','disp','load'
        attributes = getattr(self,vector)
        error_code = 0
        if label not in attributes and not (vector == 'dof' and label == 'fix'):
            error_code = 1
            if terminate:
                err_txt = 'attribute \''+label+'\' not present'
                err_txt += 'in frame='+self.frame
                err_txt +=' ['+', '.join(attributes)+']'
                raise ValueError(err_txt)
        return error_code
            
    def add_force(self,L,F):  # l, [Fx,Fy,Fz], global force vector, split to pair
        L = self.Ln(L)
        el = int(np.floor(L))
        s = L-el  # fraction along element
        f = np.linalg.solve(self.T3[:,:,el].T,F)  # to local cord
        fn = np.zeros((6,2))  # 6 dof local nodal load vector
        
        for i,label in enumerate(['fx','fy','fz']):  # split point load into F,M
            if label in self.load:
                if label == 'fx':
                    fn[i,0] = (1-s)*f[i]
                    fn[i,1] = s*f[i]
                else:  # fy,fz
                    fn[i,0] = (1-s)**2*(1+2*s)*f[i]
                    fn[i,1] = s**2*(3-2*s)*f[i]
                    mi = 5 if label == 'fy' else 4
                    fn[mi,0] = f[i]*(1-s)**2*s*self.dL[el]
                    fn[mi,1] = -f[i]*s**2*(1-s)*self.dL[el]
            else:
                if f[i] != 0:            
                    err_txt = 'non zero load \''+label+'\''
                    err_txt += ' not present in frame='+self.frame
                    err_txt +=' ['+', '.join(self.load)+']'
                    raise ValueError(err_txt)
        for i in range(2):  # each node
            node = el+i
            F = np.zeros((6))
            for j in range(2):  # force,moment
                F[j*3:j*3+3] = np.dot(self.T3[:,:,el].T,fn[j*3:j*3+3,i])  # global
            for index,label in enumerate(['fx','fy','fz','mx','my','mz']):
                if label in self.load:
                    self.add_nodal_load(node,label,F[index])

    def add_nodal_load(self,node,label,load):
        self.check_input('load',label)
        index = self.load.index(label)
        self.F[node*self.ndof+index] += load

        '''
        distributed
        point
        gravity
        '''
        
    '''
    def add_distributed_load(self):  # global
        W = np.zeros((self.nel,3))
        W[:,1] = -1
        for i,(w,dl) in enumerate(zip(W,self.dL)):
            for dof in self.load
            self.add_nodal_load()
            print(w)
    '''    
        
    def addBC(self,dof,nodes,terminate=True):
        error_code = self.check_input('dof',dof,terminate=terminate)
        if not error_code:
            self.BC[dof] = np.append(self.BC[dof],np.array(nodes))
        
    def setBC(self):  # colapse constrained dof
        self.extractBC()
        if len(self.BCindex) > 0:
            self.F = np.delete(self.F,self.BCindex)
            for i in range(2):  # rows and cols
                self.K = np.delete(self.K,self.BCindex,i)
        
    def extractBC(self):  # extract matrix indicies for constraints
        self.BCindex = np.array([])
        for j,key in enumerate(self.BC.keys()):
            ui = self.ndof*self.BC[key]  # node dof
            if len(ui) > 0:
                if key == 'fix':  # fix all dofs
                    for i in range(self.ndof):
                        self.BCindex = np.append(self.BCindex,ui+i)
                else:
                    self.BCindex = np.append(self.BCindex,ui+j-1)
        self.BCindex = list(map(int,set(self.BCindex)))
        self.Dindex = np.arange(0,self.nK)
        self.Dindex = np.delete(self.Dindex,self.BCindex)
        
    def solve(self):
        self.assemble()  # assemble stiffness matrix
        self.setBC()  # remove constrained equations from stiffness + load 
        self.Dn = np.zeros(self.nK)
        self.Dn[self.Dindex] = np.linalg.solve(self.K,self.F)  # global displacment
        for i,disp in enumerate(self.disp):
            self.D[disp] = self.Dn[i::self.ndof]

    def plot_nodes(self):
        ms = 5
        pl.plot(self.X[:,0],self.X[:,1],'o',markersize=ms)
        #pl.plot(self.X[:,0]+self.D['x'],
        #        self.X[:,1]+self.D['y'],'o',markersize=ms) 
        sns.despine()