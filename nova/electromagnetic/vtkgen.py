"""Manage vtk instances within DataFrame.poly."""
import vedo

from nova.electromagnetic.geoframe import GeoFrame


class VtkFrame(vedo.Mesh, GeoFrame):
    
    def __str__(self):
        """Return polygon name."""
        return 'vtk'
    
    def dumps(self) -> str:
        """Return string representation of vtk object."""
        return {'type': 'VTK', 'points': self.points().tolist()}
    
    @classmethod
    def loads(cls, poly: str):
        """Load json prepresentation."""
        #return cls(shapely.geometry.shape(geojson.loads(poly)))
        
    

##@dataclass
#class VtkGen:
#    """VTK generator class."""
#    #return [json.dumps(panel.points().tolist())for panel in self[col]]