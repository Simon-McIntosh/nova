import pylab as pl
import numpy as np
from matplotlib.animation import FuncAnimation
from shelf import PKL
import pickle
import seaborn as sns
import matplotlib.gridspec as gridspec
rc = {'figure.figsize':[25,1/3*25],'savefig.dpi':120, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0,
      'lines.linewidth':1.75}
sns.set(context='poster',style='white',font='sans-serif',palette='Set2',
        font_scale=3.5,rc=rc)
color = sns.color_palette('Set2',8)
from addtext import linelabel
from elliptic import EQ
from itertools import cycle
Color = cycle(sns.color_palette('Set2',8))
from radial_build import RB
from eqConfig import Config
conf = Config('SXex')

pkl = PKL('moveSX')
sf,eq,inv = pkl.fetch(['sf','eq','inv'])

sf.conf = conf
conf.TF(sf)
rb = RB(conf,sf,Np=100)
rb.divertor_outline(False,plot=False,debug=False)
eq.grid(boundary={'R':rb.Rb,'Z':rb.Zb},n=5e3)
eq.set_sf_psi()  # set psi
eq.gen()

fig,ax = pl.subplots()
nr,nz = 1,3
N = nr*nz
pl.subplots_adjust(wspace=0,hspace=0)
   
Swing =  np.linspace(np.max(inv.Swing),np.min(inv.Swing),N)#-10
Swing_label = ['SOF','MOF','EOF']
for i,swing in enumerate(Swing): 
    Color = cycle(sns.color_palette('Set2'))
    pl.tight_layout()
    ax = pl.subplot(nr,nz,i+1)
    pl.sca(ax)
    ax.set_axis_off()
    pl.text(8.47,5.4,'{:1.1f}'.format(np.abs(2*np.pi*(swing-Swing[0]))),
            ha='center',va='bottom')
    inv.swing_fix(swing)
    inv.solve_slsqp()
    B = eq.Bfeild([inv.fix['r'][-1],inv.fix['z'][-1]])
    arg = 180*np.arctan2(B[1],B[0])/np.pi
    print('swing {:1.2f} arg {:1.2f} rms {:1.3f}'.format(swing,arg,inv.rms))

    eq.set_sf_psi()  # set psi
    #eq.run(update=False)
    eq.get_Vcoil()
    eq.gen(Vtarget=inv.Vtarget)
    sf.eqwrite(config='SX_'+Swing_label[i])
    
    sf.get_sol_psi(dr=0.005,Nsol=5)
    rb.trim_sol(plot=True,update=True)
    rb.get_legs()
    sf.contour(levels=np.linspace(-150,150,100),Xnorm=False,lw=0.5)
    
    r,z = sf.get_boundary(alpha=1-1e-3)
    r,z = np.append(r,r[0]),np.append(z,z[0])
    pl.plot(r,z,color=color[2],linewidth=1)
    pl.plot(rb.Rb,rb.Zb,'-',color='k',alpha=0.75,markersize=4)
    #rb.FWfill(dt=conf.tfw,loop=True,alpha=0.7,color=[0.75,0.75,0.75],s=2e-3)
    
    #pl.axes([10,13,-10,-7]
    pl.axis('equal')

    pl.text(8.75,-10.25,'swing {:1.0f}'.format(np.abs(2*np.pi*(swing-Swing[0]))),
            ha='center',va='bottom')
      
    #rs,zs = sf.legs['outer']['R'][0][-1],sf.legs['outer']['Z'][0][-1]
    rt,zt = 10.6279638945,-9.16327265735
    delta = 0.075
    ax.set_xlim(rt+2.5*delta*np.array([-1.25,0.75]))
    ax.set_ylim(zt+delta*np.array([-1.5,0.5]))


    rs,zs = sf.legs['outer']['R'][0][-1],sf.legs['outer']['Z'][0][-1]
    index = np.argmin((rb.Rb-rs)**2+(rb.Zb-zs)**2)-1
    Rp,Zp = np.array([rb.Rb[index],rs]),np.array([rb.Zb[index],zs])
    graze = sf.get_graze(Rp,Zp)
    
    print(rs,zs,graze*180/np.pi)

    
    sep = np.sign(rs-rt)*np.array([rs-rt,zs-zt])
    dx = np.sqrt(np.dot(sep,sep))
    if dx < 5e-3:
        norm = np.array([Zp[0]-Zp[1],Rp[1]-Rp[0]])
        norm /= np.sqrt(np.dot(norm,norm))
    else:
        norm = np.array([-sep[1],sep[0]])/dx
    
    scale = 0.015
    pl.plot(rs+scale*np.array([0.2,0.8*norm[0]]),
            zs+scale*np.array([0.2,0.8*norm[1]]),
            color=0.95*np.ones(3))
            
    pl.plot(rt+scale*np.array([0.2,0.8*norm[0]]),
            zt+scale*np.array([0.2,0.8*norm[1]]),
            color='k',linewidth=2)
    if dx > 5e-3:
        ra = scale*norm[0]/2+np.array([rs,rt])
        za = scale*norm[1]/2+np.array([zs,zt])
        pl.arrow(ra[0],za[0],np.diff(ra)[0]-scale/2*norm[1],
                 np.diff(za)[0]+scale/2*norm[0],
                 fc=0.95*np.ones(3),ec=0.95*np.ones(3),
                 head_width=scale/2,head_length=scale/2)  
    
    text = Swing_label[i]+'\n\n'
    text += '$\Delta$x {:1.2f}mm\n'.format(np.sign(rs-rt)*dx*1e3)
    text += '$\Theta$ {:1.2f}$^\circ$\n'.format(arg)
    text += '$\gamma$ {:1.2f}$^\circ$\n'.format(graze*180/np.pi)
    pl.text(rt-0.25,zt+0.01,text,ha='left',va='top')

pl.tight_layout()
pl.savefig('../Figs/Plasma_movment.png',dpi=300)
    
'''
with open('./plot_data/'+rb.conf.config+'_FW.pkl', 'rb') as input:
            rb.targets = pickle.load(input)
            rb.Rb = pickle.load(input)
            rb.Zb = pickle.load(input)
rb.Rb,rb.Zb = rb.midplane_loop(rb.Rb,rb.Zb)  # clockwise LFS
rb.sf.nlim = len(rb.Rb)  # update sf
rb.sf.xlim,rb.sf.ylim = rb.Rb,rb.Zb
'''