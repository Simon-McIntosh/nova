import numpy as np
import pylab as pl

from config import layout
from extract import extract_geom
from streamfunction import SF


config = 'DD3'  # 'DD1'
geom = extract_geom(config)

zo = 0  # vertical position
ro = 9.5  # major radius
A = 3  # aspect ratio
k = 1.5  # elongation
delta = 0.3  # triangularity

theta = np.linspace(0, 2 * np.pi, 50)

r = ro + ro / A * np.cos(theta + delta * np.sin(theta))
z = zo + ro / A * k * np.sin(theta)

pl.figure(figsize=(14, 10))
pl.axis('equal')
pl.plot(r, z, 'b-')
