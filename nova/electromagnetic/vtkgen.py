"""Manage vtk instances within DataFrame.poly."""
import json
import numpy as np
import vedo

from nova.electromagnetic.geoframe import GeoFrame


class VtkFrame(vedo.Mesh, GeoFrame):
    """Manage vtk serialization via json strings."""

    def __init__(self, points: list[float], cells: list[int], **kwargs):
        super().__init__([points, cells], **kwargs)

    def __str__(self):
        """Return polygon name."""
        return 'vtk'

    def dumps(self) -> str:
        """Return string representation of vtk object."""
        return json.dumps(
            {'type': 'VTK',
             'points': self.points().tolist(),
             'cells': np.array(self.cells(), dtype=int).tolist(),
             'color': self.color().tolist(), 'opacity': self.opacity()})

    @classmethod
    def loads(cls, vtk: str):
        """Load json prepresentation."""
        vtk = json.loads(vtk)
        return cls(vtk['points'], vtk['cells'],
                   c=vtk['color'], alpha=vtk['opacity'])

    @property
    def mesh(self) -> vedo.Mesh:
        """Return mesh instance."""
        return vedo.Mesh(self).c(self.color()).opacity(self.opacity())

##@dataclass
#class VtkGen:
#    """VTK generator class."""
#    #return [json.dumps(panel.points().tolist())for panel in self[col]]
