import numpy as np

from nova.biot.biotframe import Source
from nova.imas.coils_non_axisymmetric import CoilsNonAxisymmetyric

ids = CoilsNonAxisymmetyric(115001, 2, field_attrs=["Ax", "Ay", "Az"])

space = Source(ids.subframe.loc[ids.subframe.frame == "EU1B", :]).space

points = space.centerline(2)
tangent = points[1:] - points[:-1]
segment_length = np.linalg.norm(tangent, axis=-1)
mid_point = points[:-1] + tangent / 2


space.set_axes("3d")

space.axes.plot(*points.T, ".-")
space.axes.plot(*mid_point.T, ".")

ids.point.solve(mid_point)


resolve = False
if resolve:
    ids.grid.solve(5e3, [2.5, 8, -4, 4])  # limit=dataset.grid.limit
    ids._clear()
    ids.store()
