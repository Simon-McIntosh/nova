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
    
def induce(X,X_,r):
    dX,dX_,L = dloop(X),dloop(X_),0
    for i in range(len(X)):
        for j in range(len(X_)):
            dr = np.linalg.norm(X[i,:]-X_[j,:])
            if dr != 0:  # mutual inductance
                L += np.dot(dX[i,:],dX_[j,:])/dr
            else:  # self inductance
                dl = np.linalg.norm(dX[i,:])
                L += 2*dl*(np.log(2*dl/r)-1)
    L *= mu_o/(4*np.pi)   
    return(L)