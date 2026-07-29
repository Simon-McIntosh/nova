"""

import numpy as np
from mayavi import mlab

x,y=np.ogrid[-2:2:20j,-2:2:20j]
z=x*np.exp(-x**2-y**2)
"""

# pl=mlab.surf(x,y,z,warp_scale="auto")
# mlab.axes(xlabel='x',ylabel='y',zlabel='z')
# mlab.outline(pl)

# mlab.show()


# import pylab as plt

# plt.plot(1, 1, 'o')
import pyvista as pv

# from pyvista.utilities import xvfb
# xvfb.start_xvfb()

# import dolfinx.plot

# Start virtual framebuffer
# pv.start_xvfb(wait=0.5)

# import os
# os.system('/usr/bin/Xvfb :98 -screen 0 1024x768x24 &')
# os.environ['DISPLAY'] = ':98'

# import panel as pn
# pn.extension('vtk')

# pn.extension('vtk')  # this needs to be at the top of each cell for some reason

# from pyvista.utilities import xvfb
# xvfb.start_xvfb(wait=0.5)

sphere = pv.Sphere()
sphere.plot(color="w")

# plotter = pv.Plotter()
# plotter.add_mesh(sphere, color='w')
# plotter.show()
# plotter.close()
