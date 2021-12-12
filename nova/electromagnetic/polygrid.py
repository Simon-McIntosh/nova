"""Grid methods."""
from dataclasses import dataclass, field
from typing import Union

import shapely.geometry
import shapely.strtree
import numpy as np
import pandas

from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.polyplot import PolyPlot
from nova.geometry import inpoly
from nova.geometry.polyframe import PolyFrame
from nova.geometry.polygen import PolyGen
from nova.geometry.polygeom import PolyGeom
from nova.geometry.polygon import Polygon
from nova.utilities.pyplot import plt


# pylint:disable=unsubscriptable-object


@dataclass
class PolyDelta(PolyGeom):
    """Manage grid spacing and cell dimension."""

    delta: Union[int, float] = 0
    turn: str = 'hexagon'
    nturn: float = 1.
    tile: bool = False
    fill: bool = False
    cell_delta: list[float] = field(init=False)
    grid_delta: list[float] = field(init=False)

    def __post_init__(self):
        """Generate deltas."""
        super().__post_init__()  # generate bounding polygon
        self.generate_deltas()

    def generate_deltas(self):
        """Generate grid and cell deltas."""
        self.turn = PolyGen.polyshape[self.turn]  # inflate turn name
        self.cell_delta = self.dimension_cell()
        self.grid_delta = self.dimension_grid()

    def dimension_cell(self):
        """Return cell dimensions."""
        ndiv_x, ndiv_z = self.divide()
        ndiv_x, ndiv_z = np.max([ndiv_x, 1]), np.max([ndiv_z, 1])
        if self.tile or self.fill:
            return self.width/ndiv_x, self.height/ndiv_z
        return self.width / np.round(ndiv_x), self.height / np.round(ndiv_z)

    def divide(self):
        """Return number of cell divisions along x and z axis."""
        if self.delta <= 0:
            if self.delta == 0:
                filament_number = self.nturn
            else:
                filament_number = -self.delta
            fill_fraction = self.area / self.box_area
            box_number = filament_number / fill_fraction
            width, height = self.width, self.height
            if self.tile:
                if self.turn in ['disc', 'skin']:
                    box_number *= np.sqrt(3)/2
                elif self.turn == 'hexagon':
                    box_number *= 3/8 * np.sqrt(3)
            elif self.turn == 'hexagon':
                box_number *= np.sqrt(3)/2
            if np.isclose(aspect := width/height, 1) and \
                    self.turn in ['disc', 'square', 'skin']:
                delta = np.sqrt(self.box_area / box_number)
                return (ndiv := width/delta, ndiv)
            if self.turn == 'hexagon' and not self.tile:
                aspect /= np.sqrt(3)/2
            if aspect > 1:
                return self.divide_box(box_number, aspect)
            return self.divide_box(box_number, 1/aspect)[::-1]
        ndiv_x = self.width / self.delta
        ndiv_z = self.height / self.delta
        return ndiv_x, ndiv_z

    def divide_box(self, box_number, aspect):
        """Return box divisions, stretch if not tile or fill."""
        ndiv_x = np.sqrt(box_number * aspect)
        if not (self.tile or self.fill):
            ndiv_x = np.round(ndiv_x)
        ndiv_z = box_number / ndiv_x
        return ndiv_x, ndiv_z

    def dimension_grid(self):
        """Return grid delta."""
        grid_delta = list(self.cell_delta)
        if self.tile:
            grid_delta = [delta := PolyGen.boxbound(*grid_delta), delta]
            if self.turn == 'hexagon':
                grid_delta[0] *= 3/2
                grid_delta[1] *= np.sqrt(3)/4
                return grid_delta
            if self.turn in ['disc', 'skin']:
                grid_delta[1] *= np.sqrt(3)/2
                return grid_delta
        if self.turn == 'hexagon' and self.delta != 0:
            grid_delta[1] *= np.sqrt(3)/2
        return grid_delta


@dataclass
class PolyCell(PolyDelta):
    """Define PolyGrid cell."""

    scale: Union[int, float] = 1
    skin: float = 0.65

    def __post_init__(self):
        """Size polycell."""
        super().__post_init__()  # size grid and polycell
        self._scale()
        self._skin()

    def polycell(self, x_center, z_center):
        """Return cell polygon."""
        poly = {f'{self.turn}': (x_center, z_center, *self.cell_delta)}
        return PolyGeom(poly)

    @property
    def unitcell(self):
        """Return referance polygon."""
        return self.polycell(0, 0)  # refernace cell polygon

    def _scale(self):
        """Apply scaling to cell."""
        self.cell_delta = [self.scale*delta for delta in self.cell_delta]

    def _skin(self):
        """Update thickness parameter for skin sections."""
        if self.turn == 'skin':
            self.cell_delta[1] = self.skin


@dataclass
class PolySpace:
    """Manage polygon bounds, generate 1D mesh."""

    start: float
    stop: float
    cell_delta: float
    grid_delta: float

    def __post_init__(self):
        """Init limit parameters, center 1D mesh on base."""
        self.divide()
        self.center()

    def divide(self):
        """Divide base length by delta, update stop parameter."""
        self.base_length = self.stop-self.start
        self.ndiv = int(np.ceil(self.base_length / self.grid_delta))
        self.stop = self.start + self.ndiv*self.grid_delta

    def center(self):
        """Center grid on base."""
        balance = self.ndiv*self.grid_delta - self.base_length
        self.start -= balance/2
        self.stop -= balance/2

    def corner(self):
        """Return edge-edge spacing."""
        return np.linspace(self.start, self.stop, self.ndiv+1)

    def face(self):
        """Return face centered spacing."""
        return np.linspace(self.start + self.grid_delta/2,
                           self.stop - self.grid_delta/2, self.ndiv)

    def buffer_face(self):
        """Return extended face centered spacing."""
        return np.linspace(self.start - self.grid_delta/2,
                           self.stop + self.grid_delta/2, self.ndiv+2)


@dataclass
class PolyVector:
    """Manage 1D mesh for x and z coordinates."""

    limit: tuple[float]
    cell_delta: tuple[float]
    grid_delta: tuple[float]
    tile: bool
    turn: str
    polyspace: dict[str, PolySpace] = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Init polyspace methods."""
        self.polyspace['x'] = \
            PolySpace(*self.limit[:2], self.cell_delta[0], self.grid_delta[0])
        self.polyspace['z'] = \
            PolySpace(*self.limit[2:], self.cell_delta[1], self.grid_delta[1])

    def __call__(self, direction):
        """Return 1D spacing vector in the requested direction."""
        if self.tile:
            if self.turn == 'hexagon':
                return self.polyspace[direction].buffer_face()
            return self.polyspace[direction].corner()
        return self.polyspace[direction].face()


@dataclass
class PolyGrid(PolyCell):
    """Construct 2d grid from polycell basis."""

    trim: bool = True
    vector: PolyVector = field(init=False)
    frame: pandas.DataFrame = field(init=False, repr=False)
    columns: list[str] = field(init=False, default_factory=lambda: [
        'x', 'z', 'dl', 'dt', 'dx', 'dz', 'rms', 'area', 'section', 'poly'])

    def __post_init__(self):
        """Generate grid."""
        super().__post_init__()
        self.vector = PolyVector(
            self.limit, self.cell_delta, self.grid_delta, self.tile, self.turn)
        self.frame = self.dataframe()

    def __len__(self):
        """Return dataframe length."""
        return len(self.frame)

    def grid_coordinates(self):
        """Return grid coordinates."""
        grid = np.meshgrid(self.vector('x'), self.vector('z'), indexing='ij')
        if self.tile:
            grid[0][:, 1::2] += self.grid_delta[0]/2
        return np.hstack((grid[0].reshape(-1, 1), grid[1].reshape(-1, 1)))

    def polycells(self, coords):
        """Return polycells."""
        polys = [self.polycell(*coord) for coord in coords]
        if self.trim:
            return self.polytrim(coords, polys)
        return coords, polys

    def polytrim(self, coords, polys):
        """Return polycells trimed to bounding polygon."""
        polytree = shapely.STRtree([poly.poly for poly in polys])
        buffer = self.poly.buffer(1e-12*self.cell_delta[0])
        index = polytree.query_items(self.poly)
        polys = np.array(polys)[index]
        polys = [PolyFrame(polytrim, poly.metadata if poly.poly.within(buffer)
                           else dict(name='polygon'))
                 for poly in polys
                 if (polytrim := poly.poly.intersection(buffer))
                 and isinstance(polytrim, shapely.geometry.Polygon)]
        return polys

    def dataframe(self):
        """Bulid polygeom dataframe."""
        coords = self.grid_coordinates()  # build coordinate grid
        polys = self.polycells(coords)  # build trimmed cell polygons
        data = [[] for __ in range(len(polys))]
        for i, poly in enumerate(polys):
            geom = PolyGeom(poly, 'ring').geometry
            data[i] = {name: geom[name] for name in self.columns}
        frame = pandas.DataFrame(data, columns=self.columns)
        frame['nturn'] = self.nturn * frame['area'] / frame['area'].sum()
        return frame

    @property
    def polyarea(self):
        """Return sum of polycell areas."""
        return self.frame.area.sum()

    @property
    def polyturns(self):
        """Return sum of polycell turns."""
        return self.frame.nturn.sum()

    @property
    def polyfilaments(self):
        """Return effective filament number."""
        return self.polyarea / self.unitcell.area

    @property
    def polyplot(self):
        """Return polyplot instance."""
        frame = self.frame.copy()
        frame['part'] = 'cs'
        return PolyPlot(DataFrame(frame))

    def plot(self):
        """Plot polygon exterior and polycells."""
        self.plot_boundary()
        self.polyplot()
        plt.axis('off')
        plt.axis('equal')


if __name__ == '__main__':

    polygrid = PolyGrid({'hx': [6, 3, 2.5, 2.5]}, delta=-60,
                        turn='hex', tile=True)
    polygrid.plot()
