import matplotlib._cntr as cntr
import numpy as np
import numpy.ma as ma
import pylab as pl


# Test data.
x = np.arange(0, 10, 1)
y = np.arange(0, 10, 1)
x, y = np.meshgrid(x, y)
z = np.sin(x) + np.cos(y)

z = ma.asarray(z, dtype=np.float64)  # Import if want filled contours.

pl.contour(x,y,z,levels=[0.5])
level = 0.5
c = cntr.Cntr(x, y, z)

psi_line = c.trace(level,level,0)
psi_line = psi_line[:len(psi_line)//2]
#print(feildlines)    # x,y coords of contour points.

for psi in psi_line:
    r,z = zip(psi)