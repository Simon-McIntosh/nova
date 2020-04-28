from shapely.geometry import Polygon, Point, LinearRing

def
poly = Polygon([(0, 0), (2,8), (14, 10), (6,1)])
point = Point(7,6)

pol_ext = LinearRing(poly.exterior.coords)
d = pol_ext.project(point)
p = pol_ext.interpolate(d)
closest_point_coords = list(p.coords)[0]


x,y = poly.exterior.xy
plt.plot(x, y)
plt.plot(*point.xy, 'C3o')
plt.plot(*closest_point_coords, 'C3d')