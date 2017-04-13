import numpy as np
import pylab as pl
import scipy.optimize as op
from amigo.surface import bernstein
from amigo.addtext import linelabel
from amigo import geom
from nova.beam import Dcoil
from nova.beam.finite_element import FE
text = linelabel(value='', postfix='', Ndiv=10)
import seaborn as sns
rc = {'figure.figsize': [8, 8 * 10 / 16], 'savefig.dpi': 160,
      'savefig.jpeg_quality': 100, 'savefig.pad_inches': 0.1,
      'lines.linewidth': 2}
sns.set(context='talk', style='white', font='sans-serif', palette='Set2',
        font_scale=7 / 8, rc=rc)


def Dshape(x, r1=4.486, r2=15.708, npoints=120):
    #zscale = x[0]
    b = x[1:]
    b = np.append(np.append(0, b), 0)  # zero offset
    # b = np.append(np.append(0,b),0)  # zero gradient
    rd, zd = Dcoil.pD(r1, r2, npoints=npoints)  # referance D-coil
    r, z = np.copy(rd), np.copy(zd)
    # z = zscale*z  # scale height
    L = geom.length(r, z)
    bern = bernstein(L, n=len(b) - 1)
    dn = bern.gen(b)  # calculate bernstein polynomials
    nr, nz = geom.normal(r, z)  # offset shape
    r += nr * dn
    z += nz * dn
    r, z = geom.space(r, z, npoints=npoints)

    zshift = np.mean([z.min(), z.max()])
    z -= zshift
    znorm = zd.max() / z.max()
    z *= znorm
    return r, z


def Dsolve(r, z):
    X = np.zeros((len(r), 3))
    X[:, 0] = r
    X[:, 1] = z
    fe = FE(frame='3D')
    fe.add_mat(0, E=1, I=1, A=1, G=1, J=1, rho=1)
    fe.add_nodes(X)
    fe.add_elements(part_name='outerD')  # outer d coil

    # fe.add_nodes([13,-12,0])
    # fe.add_elements(n=[fe.part['outerD']['el'][30],fe.nndo],part_name='support')

    fe.freeze()
    fe.addBC(['fix'], [0, -1], part='outerD')
    # fe.addBC(['fix'],[-1],part='support')
    # fe.add_weight()  # add weight to all elements

    bm = -51.00305085  # magnetic moment
    for el in fe.part['outerD']['el']:
        n = fe.el['n'][el]
        r = np.mean(fe.X[n, 0])  # element radius
        b = np.array([0, 0, bm / r])
        w = -2e3 * np.cross(fe.el['dx'][el], b)
        fe.add_load(el=el, W=w)  # self weight
    fe.solve()
    return fe.part


def b_shape(x, npoints=100):
    r, z = Dshape(x)
    part = Dsolve(r, z)
    d2v = abs(part['outerD']['d2u'][:, 1]).max()
    print(x, d2v)
    return d2v


#xo = np.append(0.5,[1,-3,4,-1,2,5])
xo = np.append(1, np.zeros(6))

x = op.minimize(b_shape, xo, options={'disp': True, 'maxiter': 500},
                method='SLSQP', tol=1e-6).x  # ,constraints=constraint,bounds=bounds
print(x)

#x = [1.008,0.032,-0.092,0.079,0.073,-0.078,0.049]
#x = [ 1.039,  2.561, -1.364,  2.761,  0.66,   0.447,  3.163]
ro, zo = Dshape(xo)  # sead
parto = Dsolve(ro, zo)

r, z = Dshape(x)  # optimized
part = Dsolve(r, z)

rd, zd = Dshape(np.append(1, np.zeros(len(xo) - 1)))  # pD
partd = Dsolve(rd, zd)

pl.figure()
text = linelabel(value='', postfix='', Ndiv=10, loc='max')
pl.plot(ro, zo)
text.add('sead')
pl.plot(r, z)
text.add('optimised')
pl.plot(rd, zd)
text.add('PrincetonD')
pl.axis('equal')
pl.axis('off')
text.plot(Ralign=True, Roffset=7)
'''
part = Dsolve(r,z)
for i,p in enumerate(part):
    pl.plot(part[p]['U'][:,0],part[p]['U'][:,1])
pl.axis('equal')
'''
pl.figure()
text = linelabel(value='', postfix='', Ndiv=10)
pl.plot(parto['outerD']['l'], parto['outerD']['d2u'][:, 1])
text.add('sead')
pl.plot(part['outerD']['l'], part['outerD']['d2u'][:, 1])
text.add('optimised')
pl.plot(partd['outerD']['l'], partd['outerD']['d2u'][:, 1])
text.add('PrincetonD')
text.plot()
sns.despine()
pl.xlabel('part length')
pl.ylabel('part curvature')
