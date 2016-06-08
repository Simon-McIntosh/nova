from numpy.core.umath_tests import matrix_multiply
import numpy as np
from pymls import bounded_lsq

#setup the prob:
A=np.array([[1,-3],[5,7]])
b=np.array([[-50],[50]])
ll=np.array(([[-10],[-10]]))
ul=np.array(([[10],[10]]))    
#Solve it  
x0=bounded_lsq(A,b,ll,ul)


#Now let's display the problem and its solutions:
import matplotlib.pyplot as plt
x=y=np.linspace(-30,30,500)
X,Y=np.meshgrid(x,y)
S=np.dstack((X,Y))
SN=matrix_multiply(S,A.T)
plt.clf()
plt.contourf(x,y,np.sqrt(((SN-b.T)**2).sum(-1)),30,cmap=plt.cm.PuBu)
plt.colorbar()
rect=np.vstack((ll,ul-ll))
patch=plt.Rectangle(ll,*(ul-ll),facecolor=(0.0,0.,0.,0))
plt.gca().add_patch(patch)
plt.annotate("Bounded Min",
                xy=x0, xycoords='data',
                xytext=(-5, 5), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"),
                )
    
plt.annotate("Lsq Min",
                xy=np.linalg.lstsq(A,b)[0], xycoords='data',
                xytext=(20, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"),
                )
                
plt.scatter(*x0)
plt.scatter(*np.linalg.lstsq(A,b)[0])
plt.show()
