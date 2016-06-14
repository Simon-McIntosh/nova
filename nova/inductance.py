import numpy as np
from itertools import combinations_with_replacement as cr

mu_o = 4*np.pi*1e-7  # magnetic constant [Vs/Am]

def dloop(X):
    dX = np.diff(np.append(X,X[0,:,None].T,axis=0),axis=0)
    return dX
    
def rotate(theta):
    Rz = np.array([[np.cos(theta),-np.sin(theta),0],
                   [np.sin(theta),-np.cos(theta),0],
                   [0,0,1]])
    return Rz
    
class neumann(object):
    def __init__(self,r,**kwargs):
        self.r = r
        if 'X' in kwargs:
            self.setX(kwargs.get('X'))
        if 'X_' in kwargs:
            self.setX_(kwargs.get('X_'))
            
    def setX(self,X):
        self.X = X
        self.dX = dloop(X)
        
    def setX_(self,X_):
        self.X_ = X_
        self.dX_ = dloop(X_)
        
    def setr(self,r):
        self.r = r
        
    def intergrate(self,ij):
        dr = np.linalg.norm(self.X[ij[0],:]-self.X_[ij[1],:])
        if dr > self.r/2:  # mutual inductance
            L = np.dot(self.dX[ij[0],:],self.dX_[ij[1],:])/dr
        else:  # self inductance
            L = 0
        if ij[0] != ij[1]:
            L *= 2  # upper + lower
        return L
        
    def calculate(self):
        loop = cr(range(len(self.X)),2)
        L = 0
        for ij in loop:
            L += self.intergrate(ij)
        if np.linalg.norm(self.X-self.X_) == 0:  # self inductance correction
            L += np.sum(np.linalg.norm(self.dX,axis=1))/2
        L *= mu_o/(4*np.pi) 
        return L
    