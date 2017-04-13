from nova.DEMOxlsx import DEMO
from nova.loops import Profile
from nova.shape import Shape
import pylab as pl
import matplotlib.animation as manimation
import numpy as np

nTF = 18
family = 'S'
config = {'TF': 'dtt', 'eq': 'SN'}
config['TF'] = '{}{}{:d}'.format(config['eq'], config['TF'], nTF)

demo = DEMO()
demo.fill_part('Blanket')
demo.fill_part('Vessel')
# demo.fill_part('TF_Coil')

profile = Profile(config['TF'], family=family, part='tmp', load=False)
shp = Shape(profile, nTF=nTF, obj='L', eqconf=config['eq'], ny=3)
shp.add_vessel(demo.parts['Vessel']['out'])
shp.minimise(ripple=False, verbose=True)

# shp.update()

shp.loop.set_input(x=shp.xo[0])
shp.tf.fill()

shp.plot_bounds()


'''
filename = 'movie'
moviename = '../Movies/{}'.format(filename)
moviename += '.mp4'
print(moviename)

with writer.saving(fig,moviename,100): 
    for xo in shp.xo:
        pl.plot(xo[0],xo[1])
        ax.cla()
        #self.loop.set_input(x=xo)
        #demo.fill_part('Blanket')
        #demo.fill_part('Vessel')
        #self.plot_bounds()
        #self.update()
        #self.tf.fill()
        writer.grab_frame()
        
        fig,ax = pl.subplots()
        

def animate(i):
    pl.plot(1,1)





FFMpegWriter = manimation.writers['ffmpeg']
writer = FFMpegWriter(fps=10, bitrate=5000,codec='libx264',
                      extra_args=['-pix_fmt','yuv420p'])
    
with writer.saving(pl.gcf(),'tmp.mp4',100):  
    for i in np.arange(0,5):  
        animate(i)
        #pl.tight_layout()
        writer.grab_frame()
'''
