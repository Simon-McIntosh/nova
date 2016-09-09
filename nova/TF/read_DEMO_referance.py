from openpyxl import load_workbook
import numpy as np
import pylab as pl
from amigo import geom
from scipy.interpolate import interp1d
from collections import OrderedDict
import seaborn as sns
from itertools import cycle,count
import scipy as sp

def get_label(label,label_array,force=False,part=None):
    if label != None:
        label_array.append(label.replace(' ','_'))
        flag = True
    else:
        if force:
            label_array.append(part)
            if len(label_array) == 1:
                flag = True
            else:
                if label_array[-2] != label_array[-1]:
                    flag = True
                else:
                    flag = False
        else:
            flag = False
    return flag
    
def read_loop(part,loop,npoints=100):
    r,z = part[loop]['r'],part[loop]['z']
    if len(r) > 0:
        r,z = geom.theta_sort(r,z)  # sort azimuth
        #r,z = np.append(r,r[0]),np.append(z,z[0])  # close loop
        #r,z = geom.rzInterp(r,z,ends=True,npoints=npoints)
    return r,z
            
def set_figure():
    pl.axis('off')
    pl.axis('equal')
    
class DEMO(object):
    
    def __init__(self):
        filename = 'DEMO1_Reference_Design_-_2015_April_(_EU_2MG46D_v1_0'
        self.read(filename)
        
    def read(self,filename):
        wb = load_workbook(filename='./referance/'+filename+
                           '.xlsx',read_only=True,
                   data_only=True)
        ws = wb[wb.get_sheet_names()[0]]
        self.parts = OrderedDict()  # component parts    
        part,loop = [],[]
        for row in ws.columns:
            new_part = get_label(row[0].value,part)
            if new_part:
                self.parts[part[-1]] = OrderedDict()
            if len(part) == 0:
                continue
            new_loop = get_label(row[1].value,loop,force=new_part,part=part[-1])
            p,l = part[-1],loop[-1]
            if new_loop:
                self.parts[p][l] = OrderedDict()
            
            comp,start = row[2].value,2
            while comp == None:
                start += 1
                comp = row[start].value
            if comp == 'Rad':  # replace confusing 'Rad' label
                comp = 'r'
                
            self.parts[p][l][comp] = np.zeros(len(row)-1)
            for i,r in enumerate(row[start+1:]):
                try:
                    self.parts[p][l][comp][i] = 1e-3*float(r.value)  # m
                except:
                    break
            self.parts[p][l][comp] = self.parts[p][l][comp][:i]
        
    def process(self):
        for part in self.parts:
            for loop in self.parts[part]:
                r,z = read_loop(self.parts[part],loop)
                
                if part == 'Blanket':
                    r,z = geom.theta_sort(r,z,xo=(np.mean(r)-1,np.mean(z)-3),
                                          origin='top')
                    
                    n = len(r)
                    r_,z_ = np.zeros(n),np.zeros(n)
                    r_[0],z_[0] = r[0],z[0]
                    for i in range(n-1):
                        dr = sp.linalg.norm([r-r_[i],z-z_[i]],axis=0)
                        j = np.argmin(dr)
                        r_[i+1],z_[i+1] = r[j],z[j]
                        r,z = np.delete(r,j),np.delete(z,j)
                    r,z = r_,z_
                    pl.plot(r,z)
                    set_figure()
                    
                    
                    n = len(r)
                    ro,zo,jo = np.zeros(n),np.zeros(n),count(0)
                    r1,z1,j1 = np.zeros(n),np.zeros(n),count(0)
                    dro,dr1 = np.array([1,0]),np.array([1,0])
                    
                    xo,x1 = [r[0],z[0]],[r[0],z[0]]
                    for i in range(n-1):
                        dro_ = np.array([r[i+1]-xo[0],z[i+1]-xo[1]])
                        dr1_ = np.array([r[i+1]-x1[0],z[i+1]-x1[1]])
                        
                        dro_ /= sp.linalg.norm(dro_)
                        dr1_ /= sp.linalg.norm(dr1_)
                        
                        x = (r[i+1],z[i+1])
                        print(np.dot(dro,dro_),np.dot(dr1,dr1_))
                        if np.dot(dro,dro_) > 0.5:#np.dot(dr1,dr1_):
                        #if sp.linalg.norm(dro_) < sp.linalg.norm(dr1_):
                            j,dro = next(jo),dro_
                            xo = x
                            ro[j],zo[j] = x
                        else:
                            j,dr1 = next(j1),dr1_
                            x1 = x
                            r1[j],z1[j] = x
                            
                    j = next(jo)
                    ro,zo = ro[:j],zo[:j]
                    j = next(j1)
                    r1,z1 = r1[:j],z1[:j]
                    
                    pl.figure()
                    pl.plot(ro,zo,'.-')
                    pl.plot(ro[0],zo[0],'o')
                    pl.plot(r1,z1,'.-')
                    set_figure()
                    

        
            
    def plot(self):
        for part in self.parts:
            for loop in self.parts[part]:
                pl.plot(self.parts[part][loop]['r'],
                        self.parts[part][loop]['z'],'.',markersize=2.5)

            
            
if __name__ is '__main__':          
        demo = DEMO()
        
        '''
        color = cycle(sns.color_palette('Set2',5))
        for item in ['TF_Coil','Vessel']:
            part = demo.parts[item]
            geom.fill_loop(part,color=next(color))
        demo.plot()
        '''
        demo.process()
        #
