import numpy as np
import pylab as pl
from scipy.interpolate import interp1d as interp1
from matplotlib.collections import LineCollection

def cline(x,y,z,clim,cmap):  # color line
    norm = pl.Normalize(clim[0],clim[1])
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)    
    lc = LineCollection(segments,cmap=cmap,norm=norm)
    if not hasattr(z, "__iter__"):  # single value check
        z = np.array([z])
    lc.set_array(z)
    lc.set_linewidth(1)
    pl.gca().add_collection(lc)
    
class FC(object):
    
    def __init__(self,sf,flip=1,targets=None):
        self.sf = sf
        self.flip = flip
        self.targets = targets
'''        
    def arc3(self,points):
        A = np.array(points[0])
        B = np.array(points[1])
        C = np.array(points[2])
        a = np.linalg.norm(C - B)
        b = np.linalg.norm(C - A)
        c = np.linalg.norm(B - A)
        s = (a + b + c) / 2
        R = a*b*c/(4*np.sqrt(s*(s-a)*(s-b)*(s-c)))
        b1 = a**2*(b**2+c**2-a**2)
        b2 = b**2*(a**2+c**2-b**2)
        b3 = c**2*(a*a+b**2-c**2)
        P = np.column_stack((A,B,C)).dot(np.hstack((b1,b2,b3)))
        P /= (b1+b2+b3)
        return (R,P)
        
    def length(self,R,Z,norm=True):
        L = np.append(0,np.cumsum(np.sqrt(np.diff(R)**2+np.diff(Z)**2)))
        if norm: L = L/L[-1]
        return L
        
    def strike_arc(self,config,plot=False):
        if config is 'DD1':
            Roffset = 3.0
            p3arc = [[6.7,-4.4],[7.3,-4.7],[8,-4.85]]  # three-point strike arc
        elif config is 'DD3':
            p3arc = [[10,-9.53],[12,-7.8],[13,-5.8]]  # three-point strike arc
            Roffset = -0.4
        arc_radius,arc_centre = self.arc3(p3arc)  # generate 3point arc
        arc_radius -= Roffset  # offset from TF coil
        da = np.sqrt((p3arc[0][0]-p3arc[2][0])**2+(p3arc[0][1]-p3arc[2][1])**2)
        dtheta = 2*np.arctan2(da/2,arc_radius)
        
        theta = [np.arctan2(p3arc[0][1]-arc_centre[1],
                            p3arc[0][0]-arc_centre[0])-dtheta/4,
                  np.arctan2(p3arc[2][1]-arc_centre[1],
                             p3arc[2][0]-arc_centre[0])+dtheta/4]
        theta_arc = np.linspace(theta[0],theta[1],100)
        self.Rtarget = arc_centre[0]+arc_radius*np.cos(theta_arc)
        self.Ztarget = arc_centre[1]+arc_radius*np.sin(theta_arc)
        
        if plot:
            for i in range(3):
                pl.plot(self.flip*p3arc[i][0],p3arc[i][1],'mo')
            pl.plot(self.flip*self.Rtarget,self.Ztarget,'k:')
            
    def strike_point(self,Xi,graze):
        ratio = np.sin(graze)*np.sqrt(Xi[-1]**2+1)
        if np.abs(ratio) > 1:
            theta = np.sign(ratio)*np.pi
        else:
            theta = np.arcsin(ratio)
        return theta
                 
    def cross_legs(self,leg,layer_index):
        Rsol = self.sf.legs[leg]['R'][layer_index]
        Zsol = self.sf.legs[leg]['Z'][layer_index] 
        return  self.Xtrim(Rsol,Zsol)

    def legs(self,leg,layer_index=0,L2D=0):
        if not hasattr(self.sf,'Rsol') or L2D != 0:
            self.sf.sol()
        if L2D == 0:
            L2D = self.targets[leg]['L2D'][-1]
        Rsol,Zsol = self.cross_legs(leg,layer_index)
        Lsol = self.length(Rsol,Zsol,norm=False)
        if layer_index != 0:
            Rsolo,Zsolo = self.cross_legs(leg,0)
            Lsolo = self.length(Rsolo,Zsolo,norm=False)
            indexo = np.argmin(np.abs(Lsolo-L2D))
            index = np.argmin((Rsol-Rsolo[indexo])**2+
                          (Zsol-Zsolo[indexo])**2)
            L2D = Lsol[index]
        else:
            index = np.argmin(np.abs(Lsol-L2D))
        Rend,Zend = interp1(Lsol,Rsol)(L2D),interp1(Lsol,Zsol)(L2D)
        Rsol,Zsol = Rsol[:index],Zsol[:index]  # trim to strike point
        Rsol,Zsol = np.append(Rsol,Rend),np.append(Zsol,Zend)
        return (Rsol,Zsol)
        
    def trim_legs(self):
        for leg in self.sf.legs.keys():
            for layer_index in range(self.sf.legs[leg]['i']):
                if 'core' not in leg:
                    R,Z = self.legs(leg,layer_index)
                    self.sf.legs[leg]['R'][layer_index] = R
                    self.sf.legs[leg]['Z'][layer_index] = Z
        
    def Xtrim(self,Rsol,Zsol):
        Xindex = np.argmin((self.sf.Xpoint[0]-Rsol)**2+
                           (self.sf.Xpoint[1]-Zsol)**2)
        if (Rsol[-1]-Rsol[Xindex])**2+(Zsol[-1]-Zsol[Xindex])**2 <\
        (Rsol[0]-Rsol[Xindex])**2+(Zsol[0]-Zsol[Xindex])**2:
            Rsol = Rsol[:Xindex]  # trim to Xpoints
            Zsol = Zsol[:Xindex]
            Rsol = Rsol[::-1]
            Zsol = Zsol[::-1]
        else:
            Rsol = Rsol[Xindex:]  # trim to Xpoints
            Zsol = Zsol[Xindex:]
        return (Rsol,Zsol)
        
    def get_graze(self,R,Z):
        T = np.array([R[-1]-R[-2],Z[-1]-Z[-2]])
        T /= np.sqrt(T[0]**2+T[1]**2)  # normal target vector
        B = self.sf.Bcoil([R[-1],Z[-1]])
        B /= np.sqrt(B[0]**2+B[1]**2)  # normal poloidal feild line vector
        theta = np.arccos(np.dot(B,T))
        if theta > np.pi/2: theta = np.pi-theta
        Xi = self.expansion([R[-1]],[Z[-1]])
        graze = np.arcsin(np.sin(theta)*(Xi[-1]**2+1)**-0.5)
        return graze
            
    def expansion(self,Rsol,Zsol):
        Xi = np.array([])
        Bm = np.abs(self.sf.bcentr*self.sf.rcentr)
        for r,z in zip(Rsol,Zsol):
            B = self.sf.Bcoil([r,z])
            Bp = np.sqrt(B[0]**2+B[1]**2)  # polodial feild
            Bphi = Bm/r  # torodal field
            Xi = np.append(Xi,Bphi/Bp)  # feild expansion
        return Xi
        
    def connection(self,leg,layer_index):
        Rsol = self.sf.legs[leg]['R'][layer_index]
        Zsol = self.sf.legs[leg]['Z'][layer_index]
        if len(Rsol) < 2:
            L2D,L3D = [0],[0]
        else:
            dRsol = np.diff(Rsol)
            dZsol = np.diff(Zsol)
            L2D = np.append(0,np.cumsum(np.sqrt(dRsol**2+dZsol**2)))
            dTsol = np.array([])
            Xi = self.expansion(Rsol,Zsol)
            for r,dr,dz,xi in zip(Rsol[1:],dRsol,dZsol,Xi):
                dLp = np.sqrt(dr**2+dz**2)
                dLphi = xi*dLp
                dTsol = np.append(dTsol,dLphi/(r+dr/2))
            L3D = np.append(0,np.cumsum(dTsol*np.sqrt((dRsol/dTsol)**2+
                                        (dZsol/dTsol)**2+(Rsol[:-1])**2)))
        return L2D,L3D,Rsol,Zsol
                                    
'''