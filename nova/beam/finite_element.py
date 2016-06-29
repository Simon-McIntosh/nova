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
    
    def __init__(self,X,frame='1D',nShape=21,*args):
        self.coordinate = ['x','y','z','tx','ty','tz']
        self.set_mat(args)
        self.X = X
        self.frame = frame 
        self.grid()
        self.set_stiffness()
        self.initalize_BC()  # boundary conditions
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
        
    def grid(self):
        from amigo.geom import vector_length
        from scipy.interpolate import interp1d
        self.nel = np.shape(self.X)[0]-1  # element number
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
        self.S = {}
        self.S['s'] = np.linspace(0,1,nShape)  # inter-element spacing
        self.S['Nv'] = np.zeros((nShape,4))  #  Hermite shape functions
        self.S['Nv'][:,0] = 1-3*self.S['s']**2+2*self.S['s']**3
        self.S['Nv'][:,0] = self.S['s']-2*self.S['s']**2+self.S['s']**3
        self.S['Nv'][:,0] = 3*self.S['s']**2-2*self.S['s']**3
        self.S['Nv'][:,0] = -self.S['s']**2+self.S['s']**3
        
        
        self.S['a'] = np.zeros((4,2,self.nel))
        self.S['mo'] = np.matrix([[1,0,0,0],
                                  [0,1,0,0],
                                  [1,1,1,1],
                                  [0,1,2,3]])
        self.S['m'] = np.copy(self.S['mo'])
        
        
        
        self.S['mv'] = np.zeros((nShape,4))
        for i in range(4):
            self.S['mv'][:,i] = self.S['s']**i
        self.S['mdv'] = np.zeros((nShape,3))
        for i,c in enumerate([1,2,3]):
            self.S['mdv'][:,i] = c*self.S['s']**i
        self.S['md2v'] = np.zeros((nShape,2))    
        for i,c in enumerate([2,6]):
            self.S['md2v'][:,i] = c*self.S['s']**i    
        self.S['u'] = np.zeros((2,self.nel))
            
    def shape_coefficents(self):
        for el in range(self.nel):
            d = self.displace(el)  # local 12 dof nodal displacment 
            for i in range(2):
                self.S['u'][i,el] = d[6*i]  # axial displacment
            for i in [1,3]:
                self.S['m'][i,:] = self.S['mo'][i,:]/self.dL[el]
            for i,label in enumerate([['y','tz'],['z','ty']]):
                if label[0] in self.disp:
                    d1D = self.displace_1D(d,label)
                    self.S['a'][:,i,el] = np.linalg.solve(self.S['m'],d1D)

    def shapes(self):
        nsh = 1+(self.nShape-1)*self.nel
        self.shape = {}
        self.shape['x'] = np.zeros(nsh)
        for label in ['u','du','d2u','U']:
            self.shape[label] = np.zeros((nsh,3))
        for el in range(self.nel):
            self.get_shapes(el)

    def get_shapes(self,el):
        i = el*(self.nShape-1)
        nS = self.nShape
        self.shape['x'][i:i+nS] = np.linspace(self.L[el],self.L[el+1],nS)
        self.shape['u'][i:i+nS,0] = np.linspace(self.S['u'][0,el],
                                                self.S['u'][1,el],nS)
        self.shape['u'][i:i+nS,1] = np.dot(self.S['mv'],self.S['a'][:,0,el])
        self.shape['u'][i:i+nS,2] = np.dot(self.S['mv'],self.S['a'][:,1,el])
        self.shape['U'][i:i+nS,:] = np.dot(self.T3[:,:,el].T,\
        self.shape['u'][i:i+nS,:].T).T #+self.X[el,:]
        for j in range(3):
            self.shape['U'][i:i+nS,j] += np.linspace(self.X[el,j],
                                                     self.X[el+1,j],nS)
        self.shape['d2u'][i:i+nS,1] = np.dot(self.S['md2v'],self.S['a'][2:,0,el])
    
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
        for i in range(2):
            for j in range(2):
                index = self.coordinate.index(label[j])
                d1D[2*i+j] = d[6*i+index]
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
            disp = ['x','y','z','tz','ty','tz']
            load = ['fx','fy','fz','mx','my','mz']
            stiffness = self.stiffness_3D  
        self.dof = dof
        self.disp = disp
        self.load = load
        self.ndof = len(dof)  # degrees of fredom per node
        self.nK = int((self.nel-1)*self.ndof+2*self.ndof)
        self.stiffness = stiffness
        return dof
        
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
        k = E*np.matrix([[A/(2*a), 0,             0,
                         -A/(2*a), 0,             0],
                         [0,       3*Iz/(2*a**3), 3*Iz/(2*a**2),
                          0,      -3*Iz/(2*a**3), 3*Iz/(2*a**2)],
                         [0,       3*Iz/(2*a**2), 2*Iz/a,
                          0,      -3*Iz/(2*a**2), Iz/a],
                         [-A/(2*a),0,             0,
                          A/(2*a), 0,             0],
                         [0,      -3*Iz/(2*a**3),-3*Iz/(2*a**2),
                          0,       3*Iz/(2*a**3),-3*Iz/(2*a**2)],
                         [0,       3*Iz/(2*a**2), Iz/a,
                          0,      -3*Iz/(2*a**2), 2*Iz/a]])
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
        self.Ko = np.zeros((self.nK,self.nK))
        for el in range(self.nel):
            ke = self.stiffness(el)
            j = el*self.ndof
            self.Ko[j:j+2*self.ndof,j:j+2*self.ndof] += ke
        self.K = np.copy(self.Ko)
        
    def check_input(self,vector,label):  # 'dof','disp','load'
        attributes = getattr(self,vector)
        if label not in attributes and not (vector == 'dof' and label == 'fix'):
            err_txt = 'attribute \''+label+'\' not present in frame='+self.frame
            err_txt +=' ['+', '.join(attributes)+']'
            raise ValueError(err_txt)
            
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
        
    def addBC(self,dof,nodes):
        self.check_input('dof',dof)
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

    def plot(self):
        #fig = pl.figure()
        #ax = fig.add_subplot(111)  #, projection='3d'
        pl.plot(self.X[:,0],self.X[:,1],'o')
        pl.plot(self.X[:,0]+self.D['x'],
                self.X[:,1]+self.D['y'],'o')
        #ax.auto_scale_xyz([4,18], [-11,11], [-11,11])
        #ax.axis('equal')
        sns.despine()