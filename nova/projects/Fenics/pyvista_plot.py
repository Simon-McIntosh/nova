
#import pyvista
#import dolfinx.plot

# Start virtual framebuffer
#pyvista.start_xvfb(wait=0.0)

#import os
#os.system('/usr/bin/Xvfb :98 -screen 0 1024x768x24 &')
#os.environ['DISPLAY'] = ':98'

#import panel as pn
#pn.extension('vtk')

#pn.extension('vtk')  # this needs to be at the top of each cell for some reason
import pyvista as pv

#from pyvista.utilities import xvfb
#xvfb.start_xvfb()

sphere = pv.Sphere()
sphere.plot(color='w')

#plotter = pyvista.Plotter()
#plotter.show()
