from openpyxl import load_workbook
import numpy as np
import pylab as pl
from amigo import geom
from scipy.interpolate import interp1d
from collections import OrderedDict
import seaborn as sns
from itertools import cycle,count
import scipy as sp
from matplotlib.collections import PolyCollection

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
            
def add_segment(lines,k):
    segment = '{:1.0f}'.format(next(k))
    lines[segment] = {'r':np.array([]),'z':np.array([])}
    return segment
    
def append_value(lines,segment,r,z):
    lines[segment]['r'] = np.append(lines[segment]['r'],r)
    lines[segment]['z'] = np.append(lines[segment]['z'],z)
                    
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
                r,z = geom.read_loop(self.parts[part],loop)
                
                if part in ['Plasma']:
                    pl.plot(r,z)
                    #r,z = geom.point_loop(r,z)
                    
                    '''
                    verts = np.array([r,z])
                    verts = [np.swapaxes(verts,0,1)]
                    coll = PolyCollection(verts,edgecolors='none')
                    ax = pl.gca()
                    ax.add_collection(coll)
                    ax.autoscale_view()
                    '''                    
                    
                    '''
                    n = len(r)-1
                    dR = np.array([r[1:]-r[:-1],z[1:]-z[:-1]])
                    dR = np.array([np.gradient(r),np.gradient(z)])
                    dL = sp.linalg.norm(dR,axis=0)
                    
                    k,kflag = count(0),False
                    lines = {}
                    segment = add_segment(lines,k)
                    for i in range(n):
                        kink = np.arccos(np.dot(dR[:,i]/dL[i],
                                                dR[:,i+1]/dL[i+1]))*180/np.pi
                        append_value(lines,segment,r[i+1],z[i+1])
                        if kink > 30 and kflag == False:  # angle, deg
                            segment = add_segment(lines,k)
                            append_value(lines,segment,r[i+1],z[i+1])
                            kflag = True
                        else:
                            kflag = False


                        
                    segments = list(lines.keys())
                    l = np.zeros(len(segments))
                    for i,seg in enumerate(segments):
                        l[i] = len(lines[seg]['r'])
                        
                    seg = np.argsort(l)[-2:]  # select loops (in/out)
                    rmax = np.zeros(2)
                    for i,s in enumerate(seg):
                        rmax[i] = np.max(lines[segments[s]]['r'])
                    seg = seg[np.argsort(rmax)]
                    
                    
                    r = lines[segments[seg[0]]]['r']
                    z = lines[segments[seg[0]]]['z']


                    #pl.plot(r,z)
                    '''
                        
                    '''
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
                    '''

        
            
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
        '''
        from nova.config import Setup
        from nova.streamfunction import SF
        from nova.radial_build import RB
        from nova.elliptic import EQ
        from nova.coils import PF
        from nova.inverse import INV
        
        import seaborn as sns
        rc = {'figure.figsize':[8*12/16,8],'savefig.dpi':120, # 
              'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
              'lines.linewidth':2}
        sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
                font_scale=7/8,rc=rc)
        
        setup = Setup('DEMO_SN',eqdir='../../eqdsk/')
        sf = SF(setup.filename)
        
        rb = RB(setup,sf)
        pf = PF(sf.eqdsk)
        eq = EQ(sf,pf,limit=[5,14,-7,7],n=5e3)  
        
        eq.get_plasma_coil()
        #inv = INV(sf,pf,rb,eq,configTF='SN',config='SN')
        
        pf.plot(coils=pf.coil,label=True,plasma=True,current=False) 
        sf.contour()
        '''
