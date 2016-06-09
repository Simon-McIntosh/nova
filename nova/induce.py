import numpy as np
mu_o = 4*np.pi*1e-7  # magnetic constant [Vs/Am]

def dloop(X):
    dX = np.diff(np.append(X,X[0,:,None].T,axis=0),axis=0)
    return dX
    
def rotate(theta):
    Rz = np.array([[np.cos(theta),-np.sin(theta),0],
                   [np.sin(theta),-np.cos(theta),0],
                   [0,0,1]])
    return Rz
    
def inductance(X,X_):
    dX,dX_,L = dloop(X),dloop(X_),0
    for i in range(len(X)):
        for j in range(len(X_)):
            if np.linalg.norm(X[i,:]-X_[j,:]) != 0:
                L += np.dot(dX[i,:],dX_[j,:])/np.linalg.norm(X[i,:]-X_[j,:])
            else:
                L += np.linalg.norm(dX[i,:])/2
    L *= mu_o/(4*np.pi)   
    return(L)