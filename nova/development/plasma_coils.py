import numpy as np
import shapely.geometry
import shapely.ops
import pygeos

from nova.electromagnetic.coilgeom import ITERcoilset
from nova.utilities.pyplot import plt

ITER = ITERcoilset(coils='pf vv trs dir', dCoil=0.25, dPlasma=0.25, n=2e3,
                   read_txt=True, limit=[3, 10, -6, 6])

ITER.filename = -1
ITER.scenario = 'SOF'

'''
ITER.add_coil(ITER.d2.vector['Rcur']+2, ITER.d2.vector['Zcur'], 3, 5,
              cross_section='ellipse', name='Plasma', plasma=True)
ITER.field.solve()
ITER.grid.solve()
ITER.scenario = 'SOF'
'''

#ITER.plasma.generate_grid()
#ITER.grid.generate_grid()

poly = pygeos.creation.polygons(ITER.data['separatrix'].values)
poly = pygeos.constructive.make_valid(poly)
area = [pygeos.area(pygeos.get_geometry(poly, i))
        for i in range(pygeos.get_num_geometries(poly))]
poly = pygeos.get_geometry(poly, np.argmax(area))



tree = pygeos.STRtree(pygeos.points(
            ITER.subcoil.loc[ITER.plasma_index, ['x', 'z']].values))


ITER.subcoil.ionize = tree.query(poly, predicate='contains')
ITER.coil.loc['Plasma', 'polygon'] = \
    shapely.geometry.Polygon(pygeos.get_coordinates(poly))


plt.set_aspect(0.9)
ITER.plot(True, plasma=True, label='active')
ITER.grid.plot_flux()

#ITER.plot_data(['firstwall', 'divertor'])
#plt.plot(*ITER.data['divertor'].iloc[1:].values.T)
#sep = pygeos.get_coordinates(poly)

plt.plot(*pygeos.get_coordinates(poly).T)


#Xpoint = re.findall(r'\d+\.\d+', shapely.validation.explain_validity(ring))

#poly = shapely.geometry.Polygon(ITER.data['separatrix'].values)
#poly = shapely.ops.polygonize_full(ITER.data['separatrix'].values)
#ITER.plasma.plot()
