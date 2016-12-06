from amigo.png_tools import data_mine,data_load
import pylab as pl
from nova.shape import Shape
from nova.loops import Profile
import numpy as np
from amigo import geom

path = '../../Figs/'
file = 'negitive_tri.png'

#data_mine(path,file,label='vv',xlim=[0,16],ylim=[0,10])


vv = data_load(path,file,label='vv')
sep = data_load(path,file,label='sep')
pl.plot(vv['x'],vv['y'])
pl.plot(sep['x'],sep['y'])


config = {'TF':'NTP'}
#setup = Setup(config)
#sf = SF(setup.filename)
#sf.get_boundary(plot=True)

profile = Profile(config['TF'],family='S',part='TF')
shp = Shape(profile,obj='L',nTF=18,sep={'r':sep['x'],'z':sep['y']})  # ,eqconf=config['eq']
      
#shp.cage.pattern(plot=True)


rvv,zvv = geom.rzSLine(vv['x'],vv['y'],60)
rvv,zvv = geom.offset(rvv,zvv,0.2)
rmin = np.min(rvv)
rvv[rvv<=rmin+0.12] = rmin+0.12
shp.add_bound({'r':rvv,'z':zvv},'internal')  # vessel
#shp.plot_bounds()
shp.minimise(ripple=True,ripple_limit=0.6)

shp.update()
shp.tf.fill()

shp.cage.plot_contours(variable='ripple',n=2e3,loop={'r':vv['x'],'z':vv['y']})

#plot_oppvar(shp.loop.xo,shp.loop.oppvar)
