from tokamak.formats import geqdsk as eqdsk
import numpy as np
import pylab as pl

data = eqdsk.read('../Data/DemoSN.eqdsk')


pl.figure(figsize=(12,10))
pl.pcolor(data['r'],data['z'],data['qpsi'])
pl.axis('equal')


