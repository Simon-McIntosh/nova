import numpy as np
from amigo import geom
import pylab as pl
import seaborn as sns
from itertools import cycle
color = cycle(sns.color_palette('Set2',12))
sns.set_context('talk')
sns.set_style(style='white')

import seaborn as sns
rc = {'figure.figsize':[5,5*10/16],'savefig.dpi':150, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
Color = cycle(sns.color_palette('Set2'))

class build(object):
    
    def __init__(self):
        self.patch = []
        self.hole = []
        self.C = {'x':0,'y':0}
        self.I = {'xx':0,'yy':0,'zz':0}
        self.A = 0
        
        
    def translate(self,C,I,A,poly,dx,dy,append=True):  # parrallel axis
        C['x'] += dx 
        C['y'] += dy
        I['xx'] += A*C['y']**2
        I['yy'] += A*C['x']**2
        I['zz'] += A*(C['y']**2+C['x']**2)
        if append:
            self.add_patch(poly,dx,dy)
        else:
            self.add_hole(poly,dx,dy)
        return C,I
    
    def circ(self,r=1,ro=0):
        C = {'x':0,'y':0}  # centroid
        A = np.pi*(r**2-ro**2)  # area
        Ixx = np.pi/4*(r**4-ro**4)
        Iyy = Ixx
        Izz = 2*Ixx
        t = np.linspace(0,2*np.pi,50)  # draw
        x = np.append(r*np.cos(t),ro*np.cos(t[::-1]))
        y = np.append(r*np.sin(t),ro*np.sin(t[::-1]))
        patch = {'x':x,'y':y}
        return C,{'xx':Ixx,'yy':Iyy,'zz':Izz},A,patch
    
    def rect(self,b=1,h=1):  # width,height
        C = {'x':0,'y':0}  # centroid
        A = b*h  # area
        Ixx = b*h**3/12
        Iyy = b**3*h/12
        Izz = Ixx+Iyy
        x = np.array([-b/2,b/2,b/2,-b/2])
        y = np.array([-h/2,-h/2,h/2,h/2])
        patch = {'x':x,'y':y}
        return C,{'xx':Ixx,'yy':Iyy,'zz':Izz},A,patch
    
    def tri(self,b=1,h=1):  # base,half-height
        C = {'x':-b/3,'y':0}  # centroid
        A = 0.5*b*h  # area
        Ixx = b*h**3/6
        Iyy = b**3*h/6
        Izz = Ixx+Iyy
        x = np.array([0,-b,0],dtype='float')
        y = np.array([h,0,-h],dtype='float')
        patch = {'x':x,'y':y}
        return C,{'xx':Ixx,'yy':Iyy,'zz':Izz},A,patch
            
    def add_patch(self,patch,dx,dy):
        patch['x'] += dx
        patch['y'] += dy
        self.patch.append(patch)
        
    def add_hole(self,hole,dx,dy):
        hole['x'] += dx
        hole['y'] += dy
        self.hole.append(hole)
        
    def update(self,C,I,A):
        for ax in self.C:  # adjust centroid
            self.C[ax] = (self.C[ax]*self.A+C[ax]*A)/(self.A+A)
        self.A += A  # increment area
        for ax in self.I:  # increment second moments
            self.I[ax] += I[ax]
            
    def downdate(self,C,I,A):
        for ax in self.C:  # adjust centroid
            self.C[ax] = (self.C[ax]*self.A-C[ax]*A)/(self.A+A)
        self.A -= A  # increment area
        for ax in self.I:  # increment second moments
            self.I[ax] -= I[ax]
            
    def get_shape(self,shape):
        try:
            gen = getattr(self,shape) 
        except:
            raise ValueError('shape {} not found'.format(shape))
        return gen
        
    def remove_shape(self,shape,dx=0,dy=0,**kwargs):
        gen = self.get_shape(shape)
        C,I,A,hole = gen(**kwargs)
        C,I = self.translate(C,I,A,hole,dx,dy,append=False)
        self.downdate(C,I,A)  # update properties
        
    def add_shape(self,shape,dx=0,dy=0,**kwargs):
        gen = self.get_shape(shape)
        C,I,A,patch = gen(**kwargs)
        C,I = self.translate(C,I,A,patch,dx,dy)
        self.update(C,I,A)  # update properties
        
    def plot(self):
        for p in self.patch:
            geom.polyfill(p['x'],p['y'],color=next(color))
        for h in self.hole:
            geom.polyfill(h['x'],h['y'],color=np.ones(3))    
        pl.axis('equal')
        pl.plot(0,0,'s')
        pl.plot(self.C['x'],self.C['y'],'o')

        
        
        
if __name__ == '__main__':
    
    section = {}
    section['case'] = {'side':0.1,'nose':0.51,'inboard':0.04,
                                'outboard':0.19,'external':0.225}
    section['winding_pack'] = {'width':0.625,'depth':1.243}
    

        
    
    w,d = 0.625,1.243
    i,o,s = 0.04,0.19,0.1
    
            
    ey = build()

    ey.add_shape('rect',b=w+i+o,h=d+2*s,dx=(i-o)/2)
    ey.remove_shape('rect',b=w,h=d,dx=0)
    l,f=1.3,0.45
    ey.add_shape('tri',b=l,h=d/2+s,dx=-w/2-o)
    
    ey.remove_shape('tri',b=(1-f)*l,h=(1-f)*(d/2+s),dx=-w/2-o-f*l)
    
    '''
    ey.add_shape('rect',b=3,h=0.5,dx=-1.5)
    ey.add_shape('rect',b=4,h=0.5,dx=-5)

    ey.remove_shape('rect',b=14,h=0.5,dx=0)
    '''
    print(ey.I)
    
    '''
    #ey.add_shape('circ',r=5,ro=3,dx=3)
    ey.add_shape('rect',
                 b=1.3*section['winding_pack']['width'],
                 h=1.3*section['winding_pack']['depth'])
    ey.remove_shape('rect',
                 b=section['winding_pack']['width'],
                 h=section['winding_pack']['depth'])
    ey.add_shape('rect',
                 b=section['case']['inboard'],
                 h=section['winding_pack']['depth']+\
                          2*section['case']['side'],
                 dx=section['winding_pack']['width']/2+\
                 section['case']['inboard']/2)
    #ey.add_shape('rect',b=7,h=0.5,dx=3.5)
    #ey.add_shape('tri',b=10,h=4,dx=-3.0,dy=0.25)
    '''
    ey.plot()
    