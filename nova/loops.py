import scipy as sp
import numpy as np
from scipy.special import binom as bn
from scipy.special import iv as besl  
from amigo import geom
import pylab as pl
from collections import OrderedDict
import seaborn as sns
import pandas as pd
from amigo.IO import trim_dir
import pickle
import json

def add_value(Xo,i,name,value,lb,ub,clip=True):
    if clip:
        if value<lb:
            value = lb
        if value>ub:
            value = ub
    Xo['name'][i] = name
    Xo['value'][i] = value
    Xo['lb'][i] = lb
    Xo['ub'][i] = ub

def normalize_variables(Xo):
    X = (Xo['value']-Xo['lb'])/(Xo['ub']-Xo['lb'])
    return X
    
def denormalize_variables(x,Xo):
    Xo['value'] = x*(Xo['ub']-Xo['lb'])+Xo['lb']
    return Xo['value']
    
def plot_variables(Xo,eps=1e-2,fmt='1.2f',scale=1,postfix=''):
    xo = normalize_variables(Xo)
    Xo['norm'] = xo
    data = pd.DataFrame(Xo)
    data.reset_index(level=0,inplace=True)
    pl.figure(figsize=(6,3))
    sns.set_color_codes("muted")
    sns.barplot(x='norm',y='name',data=data,color="b")
    sns.despine(bottom=True)
    pl.ylabel('')
    ax = pl.gca()
    ax.get_xaxis().set_visible(False)
    patch = ax.patches
    #values = [xo[var]['value'] for var in xo]
    #xnorms = [xo[var]['xnorm'] for var in xo]
    for p,value,norm,var in zip(patch,Xo['value'],Xo['norm'],Xo['name']):
        x = p.get_width()
        if norm < 0:
            x = 0
        y = p.get_y()+p.get_height()/2
        size = 'small'
        if norm < eps or norm > 1-eps:
            size = 'large'
        text =' {:{fmt}}'.format(scale*value,fmt=fmt)
        text += postfix+' '
        #if var not in oppvar:
        #     text += '*'
        if value < 0.5:
            ha = 'left'
            color = 0.25*np.ones(3)
        else:
            ha = 'right'
            color = 0.75*np.ones(3)
        ax.text(x,y,text,ha=ha,va='center',
                size=size,color=color)
    pl.plot(0.5*np.ones(2),np.sort(ax.get_ylim()),'--',color=0.5*np.ones(3),
            zorder=0,lw=1)
    pl.plot(np.ones(2),np.sort(ax.get_ylim()),'-',color=0.25*np.ones(3),
            zorder=0,lw=1.5)
    xlim = ax.get_xlim()
    xmin,xmax = np.min([0,xlim[0]]),np.max([1,xlim[1]])
    pl.xlim([xmin,xmax])

def check_var(var,xo):
    if var == 'l':
        var = 'l0s'
    if var not in xo:
        var = next((v for v in xo if var in v))  # match sub-string
    return var

def set_oppvar(xo,oppvar):  # set optimization bounds and normalize
    nopp = len(oppvar)
    xnorm,bnorm = np.zeros(nopp),np.zeros((nopp,2))
    for i,var in enumerate(oppvar):
        var = check_var(var,xo)
        xnorm[i] = (xo[var]['value']-xo[var]['lb'])/(xo[var]['ub']-
                                                     xo[var]['lb'])
        bnorm[i,:] = [0,1]
    return xnorm,bnorm
    
def get_oppvar(xo,oppvar,xnorm):
    x = np.copy(xnorm)
    for i,var in enumerate(oppvar):
        var = check_var(var,xo)
        x[i] = x[i]*(xo[var]['ub']-xo[var]['lb'])+xo[var]['lb']
        #xo[var]['value'] = x[i]
    return x
    
def remove_oppvar(oppvar,var):
    if var in oppvar:
        oppvar.remove(var)

def plot_oppvar(xo,oppvar,eps=1e-2,fmt='1.2f',scale=1,postfix=''):
    xnorm,bnorm = set_oppvar(xo,oppvar)
    for var in xo:
        xo[var]['xnorm'] = (xo[var]['value']-xo[var]['lb'])/(xo[var]['ub']-
                                                             xo[var]['lb'])
    data = pd.DataFrame(xo).T
    data.reset_index(level=0,inplace=True)
    #pl.figure(figsize=(8,8))
    sns.set_color_codes("muted")
    sns.barplot(x='xnorm',y='index',data=data,color="b")
    sns.despine(bottom=True)
    pl.ylabel('')
    ax = pl.gca()
    ax.get_xaxis().set_visible(False)
    patch = ax.patches
    values = [xo[var]['value'] for var in xo]
    xnorms = [xo[var]['xnorm'] for var in xo]
    for p,value,xnorm,var in zip(patch,values,xnorms,xo):
        x = p.get_width()
        if xnorm < 0:
            x = 0
        y = p.get_y()+p.get_height()/2
        size = 'small'
        if xnorm < eps or xnorm > 1-eps:
            size = 'large'
        text =' {:{fmt}}'.format(scale*value,fmt=fmt)
        text += postfix+' '
        if var not in oppvar:
             text += '*'
        if xnorm < 0.1:
            ha = 'left'
            color = 0.25*np.ones(3)
        else:
            ha = 'right'
            color = 0.75*np.ones(3)
        ax.text(x,y,text,ha=ha,va='center',
                size=size,color=color)
    pl.plot(0.5*np.ones(2),np.sort(ax.get_ylim()),'--',color=0.5*np.ones(3),
            zorder=0,lw=1)
    pl.plot(np.ones(2),np.sort(ax.get_ylim()),'-',color=0.25*np.ones(3),
            zorder=0,lw=1.5)
    xlim = ax.get_xlim()
    xmin,xmax = np.min([0,xlim[0]]),np.max([1,xlim[1]])
    pl.xlim([xmin,xmax])

def get_input(oppvar=[],**kwargs):
    if 'x' in kwargs:
        inputs = {}
        X = kwargs.get('x')
        try:
            for var,x in zip(oppvar,X):
                inputs[var] = x
        except:
            errtxt = '\n'
            errtxt += 'Require self.variables'
            raise ValueError(errtxt)
    elif 'inputs' in kwargs:
        inputs = kwargs.get('inputs')
    else:
        inputs = {}
    return inputs
    
def close_loop(x,npoints):
    for var in ['r','z']:
        x[var] = np.append(x[var],x[var][0])
    x['r'],x['z'] = geom.rzSLine(x['r'],x['z'],npoints=npoints)
    return x
    
def set_limit(xo,limits=True):
    if limits:
        if xo['value'] < xo['lb']:
            xo['value'] = xo['lb']
        if xo['value'] > xo['ub']:
            xo['value'] = xo['ub']
    return xo

class Aloop(object):  # tripple arc loop
    def  __init__(self,npoints=200,limits=True):
        self.npoints = npoints
        self.initalise_parameters()
        self.name = 'Aloop'
        self.limits = limits
        
    def initalise_parameters(self):
        self.xo = OrderedDict()
        self.xo['ro'] = {'value':4.486,'lb':3,'ub':5}  # r origin
        self.xo['zo'] = {'value':0,'lb':-1,'ub':1}  # z origin
        self.xo['sl'] = {'value':6.428,'lb':0.5,'ub':10}  # straight length
        self.xo['f1'] = {'value':2,'lb':0.1,'ub':10}  # rs == f1*z small
        self.xo['f2'] = {'value':5,'lb':0.1,'ub':10}  # rm == f2*rs mid
        self.xo['a1'] = {'value':45,'lb':5,'ub':65}  # small arc angle, deg
        self.xo['a2'] = {'value':75,'lb':5,'ub':110}  # middle arc angle, deg
        self.oppvar = list(self.xo.keys())
        #for rmvar in ['a1','a2']:  # remove arc angles from oppvar
        #    self.oppvar.remove(rmvar)
        #self.oppvar.remove('a2')
           
    def set_input(self,**kwargs): 
        inputs = get_input(self.oppvar,**kwargs)     
        for key in inputs:
            if key in self.xo:
                try:  # dict
                    for k in inputs[key]:
                        self.xo[key][k] = inputs[key][k]
                except:  # single value
                    self.xo[key]['value'] = inputs[key]
                self.xo[key] = set_limit(self.xo[key],limits=self.limits)
      
    def get_xo(self):
        values = []
        for n in ['ro','zo','sl','f1','f2','a1','a2']:
            values.append(self.xo[n]['value'])
        return values
                   
    def draw(self,**kwargs): 
        self.npoints = kwargs.get('npoints',self.npoints)
        self.set_input(**kwargs)
        self.segments = {'r':[],'z':[]}
        ro,zo,sl,f1,f2,a1,a2 = self.get_xo()
        a1 *= np.pi/180  # convert to radians
        a2 *= np.pi/180
        asum = a1+a2
          # straight section
        r = ro*np.ones(2)
        z = np.array([zo,zo+sl])
        self.segments['r'].append(r)
        self.segments['z'].append(z)
          # small arc
        theta = np.linspace(0,a1,round(0.5*self.npoints*a1/np.pi))
        rx,zx = r[-1],z[-1]
        r = np.append(r,r[-1]+f1*(1-np.cos(theta)))
        z = np.append(z,z[-1]+f1*np.sin(theta))
        self.segments['r'].append(rx+f1*(1-np.cos(theta)))
        self.segments['z'].append(zx+f1*np.sin(theta))
          # mid arc
        theta = np.linspace(theta[-1],asum,round(0.5*self.npoints*a2/np.pi))
        rx,zx = r[-1],z[-1]
        r = np.append(r,r[-1]+f2*(np.cos(a1)-np.cos(theta)))
        z = np.append(z,z[-1]+f2*(np.sin(theta)-np.sin(a1)))
        self.segments['r'].append(rx+f2*(np.cos(a1)-np.cos(theta)))
        self.segments['z'].append(zx+f2*(np.sin(theta)-np.sin(a1)))
          # large arc 
        rl = (z[-1]-zo)/np.sin(np.pi-asum)            
        theta = np.linspace(theta[-1],np.pi,60)
        rx,zx = r[-1],z[-1]
        r = np.append(r, r[-1]+rl*(np.cos(np.pi-theta)-np.cos(np.pi-asum)))
        z = np.append(z,z[-1]-rl*(np.sin(asum)-np.sin(np.pi-theta)))
        self.segments['r'].append(rx+rl*(np.cos(np.pi-theta)-np.cos(np.pi-asum)))
        self.segments['z'].append(zx-rl*(np.sin(asum)-np.sin(np.pi-theta)))
        r = np.append(r,r[::-1])[::-1]
        z = np.append(z,-z[::-1]+2*zo)[::-1] 
        r,z = geom.rzSLine(r,z,self.npoints)  # distribute points
        x = {'r':r,'z':z}
        x = close_loop(x,self.npoints)
        x['r'],x['z'] = geom.clock(x['r'],x['z'])
        return x
        
    def plot(self,inputs={}):
        x = self.draw(inputs=inputs)
        pl.plot(x['r'],x['z'],color=0.4*np.ones(3))
        for r,z in zip(self.segments['r'],self.segments['z']):
            pl.plot(r,z,lw=3)
        pl.axis('equal')
        pl.axis('off')

class Dloop(object):  # Princton D
    def  __init__(self,npoints=100,limits=True):
        self.npoints = npoints
        self.initalise_radii()
        self.name = 'Dloop'
        self.limits = limits
        
    def initalise_radii(self):
        self.xo = OrderedDict()
        self.xo['r1'] = {'value':4.486,'lb':3,'ub':5}  # inner radius
        self.xo['r2'] = {'value':15.708,'lb':10,'ub':20}  # outer radius
        self.xo['dz'] = {'value':0,'lb':-10,'ub':10}  # vertical offset
        self.oppvar = list(self.xo.keys())
        #self.oppvar.remove('r1')
           
    def set_input(self,**kwargs): 
        inputs = get_input(self.oppvar,**kwargs)   
        for key in inputs:
            if key in self.xo:
                try:  # dict
                    for k in inputs[key]:
                        self.xo[key][k] = inputs[key][k]
                except:  # single value
                    self.xo[key]['value'] = inputs[key]
                self.xo[key] = set_limit(self.xo[key],limits=self.limits)
                
    def get_xo(self):
        values = []
        for n in ['r1','r2','dz']:
            values.append(self.xo[n]['value'])
        return values
   
    def draw(self,**kwargs):
        self.npoints = kwargs.get('npoints',self.npoints)
        self.set_input(**kwargs)
        self.segments = {'r':[],'z':[]}
        r1,r2,dz = self.get_xo()
        ro=np.sqrt(r1*r2)
        k=0.5*np.log(r2/r1)
        theta=np.linspace(-0.5*np.pi,1.5*np.pi,2*self.npoints)
        r,z = np.zeros(2*self.npoints),np.zeros(2*self.npoints)
        s = np.zeros(2*self.npoints,dtype='complex128')
        for n in np.arange(1,20):  # sum convergent series
            ds = 1j/n*(np.exp(-1j*n*theta)-1)*(1+np.exp(1j*n*(theta+np.pi)))*\
            np.exp(1j*n*np.pi/2)*(besl(n-1,k)+besl(n+1,k))/2
            s += ds
            if np.max(abs(ds)) < 1e-14:
                break
        z=abs(ro*k*(besl(1,k)*theta+s))
        r=ro*np.exp(k*np.sin(theta))
        z -= np.mean(z)
        r,z = geom.space(r,z,self.npoints)
        z += dz  # vertical shift
        self.segments['r'].append([r[-1],r[0]])
        self.segments['z'].append([z[-1],z[0]])
        self.segments['r'].append(r)
        self.segments['z'].append(z)
        x = {'r':r,'z':z}
        x = close_loop(x,self.npoints)
        x['r'],x['z'] = geom.clock(x['r'],x['z'])
        return x
        
    def plot(self,inputs={}):
        x = self.draw(inputs=inputs)
        pl.plot(x['r'],x['z'],color=0.4*np.ones(3))
        for r,z in zip(self.segments['r'],self.segments['z']):
            pl.plot(r,z,lw=3)
            pl.axis('equal')
            pl.axis('off')
            
class Sloop(object):  # polybezier
    def  __init__(self,npoints=200,symetric=False,tension='full',limits=True):
        self.name = 'Sloop'
        self.symetric = symetric
        self.tension = tension
        self.limits = limits
        self.npoints = npoints
        self.initalise_nodes()
        self.set_symetric()
        self.set_tension()

    def initalise_nodes(self):
        self.xo = OrderedDict()
        self.xo['r1'] = {'value':4.486,'lb':3,'ub':8}  # inner radius
        self.xo['r2'] = {'value':15.708,'lb':5,'ub':25}  # outer radius 
        self.xo['z2'] = {'value':0,'lb':-0.9,'ub':0.9} # outer node vertical shift
        self.xo['height'] = {'value':17.367,'lb':0.1,'ub':50} # full loop height
        self.xo['top'] = {'value':0.5,'lb':0.05,'ub':1}  # horizontal shift
        self.xo['upper'] = {'value':0.7,'lb':0,'ub':1}  # vertical shift
        self.set_lower()  # lower loop parameters (bottom,lower)
        self.xo['dz'] = {'value':0,'lb':-5,'ub':5}  # vertical offset
        self.xo['flat'] = {'value':0,'lb':0,'ub':0.8}  # fraction outboard straight
        self.xo['tilt'] = {'value':0,'lb':-45,'ub':45}  # outboard angle [deg]
        self.oppvar = list(self.xo.keys())
        self.lkeyo = ['l0s','l0e','l1s','l1e','l2s','l2e','l3s','l3e']
        self.set_l({'value':0.8,'lb':0.45,'ub':1.8})  # 1/tesion
        #self.oppvar.remove('flat')
        #self.oppvar.remove('tilt')
        
    def adjust_xo(self,name,**kwargs):  # value,lb,ub
        for var in kwargs:
            if name == 'l':
                for lkey in self.lkeyo:
                    self.xo[lkey][var] = kwargs[var]
            else:
                self.xo[name][var] = kwargs[var]
        self.set_symetric()
            

    def check_tension_length(self,tension):
        tension = tension.lower()
        options = OrderedDict()
        options['full'] = 8
        options['half'] = 4
        options['dual'] = 2
        options['single'] = 1
        if tension in options:
            self.nl = options[tension]
        else:
            errtxt = '\n'
            errtxt += 'Select Sloop tension length multiple from:\n'
            for option in options:
                errtxt += '\'{}\''.format(option)
                errtxt += ' (nl={:1.0f})\n'.format(options[option])
            raise ValueError(errtxt)
            
    def set_lower(self):
        for u,l in zip(['top','upper'],['bottom','lower']):
            self.xo[l] = {}
            for key in self.xo[u]:
                self.xo[l][key] = self.xo[u][key]
                    
    def set_symetric(self):
        if self.symetric:  # set lower equal to upper
            for u,l in zip(['top','upper'],['bottom','lower']):
                self.xo[l] = self.xo[u]
                if l in self.oppvar:  # remove lower from oppvar
                    self.oppvar.remove(l)
            if 'tilt' in self.oppvar:
                self.oppvar.remove('tilt')
        
    def set_tension(self):
        tension = self.tension.lower()
        if self.symetric:
            if tension == 'full':
                self.tension = 'half'
            elif tension == 'half':
                self.tension = 'dual'
            else:
                self.tension = tension 
        else:
            self.tension = tension
            
        self.check_tension_length(self.tension)
        if self.tension == 'single':
            self.lkey = ['l','l','l','l','l','l','l','l']
        elif self.tension == 'dual':
            self.lkey = ['l0','l0','l1','l1','l1','l1','l0','l0']
        elif self.tension == 'half':
            if self.symetric:
                self.lkey = ['l0s','l0e','l1s','l1e','l1e','l1s','l0e','l0s']
            else:
                self.lkey = ['l0','l0','l1','l1','l2','l2','l3','l3']
        elif self.tension == 'full':
            self.lkey = ['l0s','l0e','l1s','l1e','l2s','l2e','l3s','l3e']
        oppvar = np.copy(self.oppvar)  # remove all length keys
        for var in oppvar:
            if var[0] == 'l' and var != 'lower':
                self.oppvar.remove(var)
        for oppkey in np.unique(self.lkey):  # re-populate
            self.oppvar.append(oppkey)
        for i,(lkey,lkeyo) in enumerate(zip(self.lkey,self.lkeyo)):
            if len(lkey)==1:
                lkey += '0s'
            elif len(lkey)==2:
                lkey += 's'
            self.xo[lkeyo] = self.xo[lkey].copy()

    def get_xo(self):
        values = []
        for var in ['r1','r2','z2','height','top',
                    'bottom','upper','lower','dz','flat','tilt']:
            if var not in self.xo:
                if var == 'bottom':
                    var = 'top'
                if var == 'lower':
                    var = 'upper'
            values.append(self.xo[var]['value'])
        return values
        
    def set_l(self,l):
        for lkey in self.lkeyo:
            self.xo[lkey] = l.copy()  

    def get_l(self):
        ls,le = [],[]  # start,end
        for i in range(4):
            ls.append(self.xo['l{:1.0f}s'.format(i)]['value'])
            le.append(self.xo['l{:1.0f}e'.format(i)]['value'])
        return ls,le

    def basis(self,t,v):
        n = 3  # spline order
        return bn(n,v)*t**v*(1-t)**(n-v)

    def midpoints(p):  # convert polar to cartesian
        r = p['r']+p['l']*np.cos(p['t'])
        z = p['z']+p['l']*np.sin(p['t'])
        return r,z

    def control(p0,p3):  # add control points (length and theta or midpoint)
        p1,p2 = {},{}
        xm,ym = np.mean([p0['r'],p3['r']]),np.mean([p0['z'],p3['z']])
        dl = sp.linalg.norm([p3['r']-p0['r'],p3['z']-p0['z']])
        for p,pm in zip([p0,p3],[p1,p2]):
            if 'l' not in p:  # add midpoint length
                p['l'] = dl/2
            else:
                p['l'] *= dl/2
            if 't' not in p:  # add midpoint angle
                p['t'] = np.arctan2(ym-p['y'],xm-p['x'])
            pm['r'],pm['z'] = Sloop.midpoints(p)
        return p1,p2,dl
        
    def append_keys(self,key):
        if key[0] == 'l' and key != 'lower':
            length = len(key)
            if length == 1:
                key = self.lkeyo  #  control all nodes
            elif length == 2:
                key = [key+'s',key+'e']
            else:
                key = [key]
        else:
            key = [key]
        return key
            
    def set_input(self,**kwargs): 
        inputs = get_input(self.oppvar,**kwargs)   
        for key in inputs:
            keyo = self.append_keys(key)
            for keyo_ in keyo:
                if keyo_ in self.xo:
                    try:  # dict
                        for k in inputs[key]:
                            self.xo[keyo_][k] = inputs[key][k]
                    except:  # single value
                        self.xo[keyo_]['value'] = inputs[key]
                    self.xo[keyo_] = set_limit(self.xo[keyo_],
                                               limits=self.limits)
        self.set_symetric()
        self.set_tension()
            
    def verticies(self):
        r1,r2,z2,height,top,bottom,upper,lower,dz,ds,alpha_s = self.get_xo()
        r,z,theta = np.zeros(6),np.zeros(6),np.zeros(6)
        alpha_s *= np.pi/180
        ds_z = ds*height/2*np.cos(alpha_s)
        ds_r = ds*height/2*np.sin(alpha_s)
        r[0],z[0],theta[0] = r1,upper*height/2,np.pi/2  # upper sholder
        r[1],z[1],theta[1] = r1+top*(r2-r1),height/2,0  # top
        r[2],z[2],theta[2] = r2+ds_r,z2*height/2+ds_z,-np.pi/2-alpha_s  # outer, upper
        r[3],z[3],theta[3] = r2-ds_r,z2*height/2-ds_z,-np.pi/2-alpha_s # outer, lower
        r[4],z[4],theta[4] = r1+bottom*(r2-r1),-height/2,-np.pi  # bottom
        r[5],z[5],theta[5] = r1,-lower*height/2,np.pi/2  # lower sholder
        z += dz  # vertical loop offset
        return r,z,theta
            
    def linear_loop_length(self,r,z):
        self.L = 0
        for i in range(len(r)-1):
            self.L += sp.linalg.norm([r[i+1]-r[i],z[i+1]-z[i]])
            
    def segment(self,p,dl):
        n = int(np.ceil(self.npoints*dl/self.L))  # segment point number 
        t = np.linspace(0,1,n)
        curve = {'r':np.zeros(n),'z':np.zeros(n)}
        for i,pi in enumerate(p):
            for var in ['r','z']:
                curve[var] += self.basis(t,i)*pi[var]
        return curve

    def polybezier(self,r,z,theta):
        x = {'r':np.array([]),'z':np.array([])}
        self.p = []
        self.linear_loop_length(r,z)
        ls,le = self.get_l()
        for i,j,k in zip(range(len(r)-1),[0,1,3,4],[1,2,4,5]):
            p0 = {'r':r[j],'z':z[j],'t':theta[j],'l':ls[i]}
            p3 = {'r':r[k],'z':z[k],'t':theta[k]-np.pi,'l':le[i]}
            p1,p2,dl = Sloop.control(p0,p3)
            curve = self.segment([p0,p1,p2,p3],dl)
            self.p.append({'p0':p0,'p1':p1,'p2':p2,'p3':p3})
            for var in ['r','z']:
                x[var] = np.append(x[var],curve[var][:-1])
        for var in ['r','z']:
            x[var] = np.append(x[var],curve[var][-1])
            x[var] = x[var][::-1]
        return x
        
    def draw(self,**kwargs):
        self.npoints = kwargs.get('npoints',self.npoints)
        self.set_input(**kwargs)  
        r,z,theta = self.verticies()
        x = self.polybezier(r,z,theta)
        x = close_loop(x,self.npoints)
        x['r'],x['z'] = geom.clock(x['r'],x['z'])
        return x
        
    def plot(self,inputs={},ms=3):
        #color = cycle(sns.color_palette('Set2',5))
        x = self.draw(inputs=inputs)
        r,z,theta = self.verticies()
        c1,c2 = 0.75*np.ones(3),0.4*np.ones(3)
        pl.plot(r,z,'s',color=c1,ms=2*ms,zorder=10)
        pl.plot(r,z,'s',color=c2,ms=ms,zorder=10)
        pl.plot(x['r'],x['z'],color=c2,ms=ms)
        for p in self.p:
            pl.plot([p['p0']['r'],p['p1']['r']],
                    [p['p0']['z'],p['p1']['z']],color=c1,ms=ms,zorder=5)
            pl.plot(p['p1']['r'],p['p1']['z'],'o',color=c1,ms=2*ms,zorder=6)
            pl.plot(p['p1']['r'],p['p1']['z'],'o',color=c2,ms=ms,zorder=7)
            pl.plot([p['p3']['r'],p['p2']['r']],
                    [p['p3']['z'],p['p2']['z']],color=c1,ms=ms,zorder=5)
            pl.plot(p['p2']['r'],p['p2']['z'],'o',color=c1,ms=2*ms,zorder=6)
            pl.plot(p['p2']['r'],p['p2']['z'],'o',color=c2,ms=ms,zorder=7)
        pl.axis('equal')
        pl.axis('off')
        
class Profile(object):
    
    def __init__(self,name,family='S',part='TF',npoints=200,
                 load=False,symetric=False,**kwargs):
        self.npoints = npoints
        self.name = name
        self.part = part
        self.initalise_loop(family,npoints,symetric=symetric)  # initalize loop object
        data_dir = trim_dir('../../Data/')
        self.dataname = data_dir+self.name+'_{}.pkl'.format(part)
        self.read_loop_dict()
        self.nTF=kwargs.get('nTF','unset')
        self.obj=kwargs.get('obj','unset')
        
        if load:
            #try:  # try to load loop using kwargs or unset data
            self.load(nTF=self.nTF,obj=self.obj)
            #except:
            #    pass

    def initalise_loop(self,family,npoints=100,symetric=False):
        self.family = family  # A==arc, D==Princton-D, S==spline
        if self.family == 'A':  # tripple arc (5-7 parameter)
            self.loop = Aloop(npoints=npoints)
        elif self.family == 'D':  # princton D (3 parameter)
            self.loop = Dloop(npoints=npoints)
        elif self.family == 'S':  # polybezier (8-16 parameter)
            self.loop = Sloop(npoints=npoints,symetric=symetric)
        else:
            errtxt = '\n'
            errtxt += 'loop type \''+self.family+'\'\n'
            errtxt += 'select from [A,D,S]\n'
            raise ValueError(errtxt)
        
    def read_loop_dict(self):
        try:
            with open(self.dataname,'rb') as input:
                self.loop_dict = pickle.load(input)
        except:
            print('file '+self.dataname+' not found')
            print('initializing new loop_dict')
            self.loop_dict = {}
        self.frame_data()
            
    def frame_data(self):
        self.data_frame = {}   
        for family in self.loop_dict:
            data = {}
            for nTF in self.loop_dict[family]:
                data[nTF] = {}
                for obj in self.loop_dict[family][nTF]:
                    data[nTF][obj] = True
            self.data_frame[family] = pd.DataFrame(data)
 
    def load(self,nTF='unset',obj='unset'):
        if obj in self.loop_dict.get(self.family,{}).get(nTF,{}):
            loop_dict = self.loop_dict[self.family][nTF][obj]
            for key in loop_dict:
                if hasattr(self.loop,key):
                    setattr(self.loop,key,loop_dict[key])
        else:
            errtxt = '\n'
            errtxt += 'data not found:\n'
            errtxt += 'loop type {}, nTF {}, obj {}\n'.\
            format(self.family,nTF,obj)
            errtxt += self.avalible_data(verbose=False)
            raise ValueError(errtxt)
     
    def avalible_data(self,verbose=True):
        if len(self.loop_dict) == 0:
            datatxt = 'no data avalible'
        else:
            datatxt ='\n{}: data avalible [obj,nTF]'.format(self.name)
            for family in self.data_frame:
                datatxt += '\n\nloop type {}:\n{}'.\
                format(family,self.data_frame[family].fillna(''))
        if verbose:
            print(datatxt)
        else:
            return datatxt
            
    def clear_data(self):
        with open(self.dataname, 'wb') as output:
            self.loop_dict = {}
            pickle.dump(self.loop_dict,output,-1)
        
    def write(self,nTF='unset',obj='unset'):  # write xo and oppvar to file
        if self.family in self.loop_dict:
            if nTF not in self.loop_dict[self.family]:
                self.loop_dict[self.family][nTF] = {obj:[]}
        else:
            self.loop_dict[self.family] = {nTF:{obj:[]}}
        cdict = {}
        for key in ['xo','oppvar','family','symetric','tension','limits']:
            if hasattr(self.loop,key):
                cdict[key] = getattr(self.loop,key)
        self.loop_dict[self.family][nTF][obj] = cdict
                     
        print(self.dataname)
        with open(self.dataname, 'wb') as output:
            pickle.dump(self.loop_dict,output,-1)
        self.frame_data()

if __name__ is '__main__':  # plot loop classes
    #loop = Aloop()
    #x = loop.plot()
    loop = Sloop(limits=False,symetric=False,tension='single')
    loop.set_tension('full')
    #x = loop.plot({'l2':1.5})
    #loop.draw()
    profile = Profile('DEMO_SN',family='S',part='TF',nTF=18,obj='L')
    

    


