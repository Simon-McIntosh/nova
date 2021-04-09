"""Grid methods."""
from dataclasses import dataclass, field
from typing import Union

import shapely.geometry
import shapely.strtree
import numpy as np

from nova.electromagnetic.frame import Frame
from nova.electromagnetic.polygen import (
    polygen, polyframe, polyshape, boxbound
    )
from nova.utilities.pyplot import plt


# pylint:disable=unsubscriptable-object


@dataclass
class Polygon:
    """Generate bounding polygon."""

    poly: Union[dict[str, list[float]], list[float], shapely.geometry.Polygon]

    def __post_init__(self):
        """Init bounding polygon."""
        self.polygon = self.generate_polygon(self.poly)

    def generate_polygon(self, poly):
        """
        Generate polygon.

        Parameters
        ----------
        poly :
            - dict[str, list[float]], polyname: *args
            - list[float], shape(4,) bounding box [xmin, xmax, zmin, zmax]
            - array-like, shape(n,2) bounding loop [x, z]

        Raises
        ------
        IndexError
            Malformed bounding box, shape is not (4,).
            Malformed bounding loop, shape is not (n, 2).

        Returns
        -------
        polygon : Polygon
            Limit boundary.

        """
        if isinstance(poly, Polygon):
            return poly
        if isinstance(poly, shapely.geometry.Polygon):
            return self.orient(poly)
        if isinstance(poly, dict):
            polys = [polygen(section)(*poly[section]) for section in poly]
            polygon = shapely.ops.unary_union(polys)
            try:
                return self.orient(polygon)
            except AttributeError as nonintersecting:
                raise AttributeError('non-overlapping polygons specified in '
                                     f'{poly}') from nonintersecting
        poly = np.array(poly)  # to numpy array
        if poly.ndim == 1:   # poly bounding box
            if len(poly) == 4:  # [xmin, xmax, zmin, zmax]
                xlim, zlim = poly[:2], poly[2:]
                x_center, z_center = np.mean(xlim), np.mean(zlim)
                width, height = np.diff(xlim)[0], np.diff(zlim)[0]
                polygon = polygen('rectangle')(x_center, z_center,
                                               width, height)
                return self.orient(polygon)
            raise IndexError('malformed bounding box\n'
                             f'poly: {poly}\n'
                             'require [xmin, xmax, zmin, zmax]')
        if poly.shape[1] != 2:
            poly = poly.T
        if poly.ndim == 2 and poly.shape[1] == 2:  # loop
            polygon = shapely.geometry.Polygon(poly)
            return self.orient(polygon)
        raise IndexError('malformed bounding loop\n'
                         f'shape(poly): {poly.shape}\n'
                         'require (n,2)')

    @staticmethod
    def orient(polygon):
        """Return coerced polygon boundary as a positively oriented curve."""
        return shapely.geometry.polygon.orient(polygon)

    def plot_exterior(self):
        """Plot boundary polygon."""
        plt.plot(*self.polygon.exterior.xy)

    @property
    def xlim(self) -> tuple[float]:
        """Return polygon bounding box x limit (xmin, xmax)."""
        return self.polygon.bounds[::2]

    @property
    def width(self) -> float:
        """Return polygon bounding box width."""
        return np.diff(self.xlim)[0]

    @property
    def zlim(self) -> tuple[float]:
        """Return polygon bounding box x limit (xmin, xmax)."""
        return self.polygon.bounds[1::2]

    @property
    def height(self) -> float:
        """Return polygon bounding box height, [xmin, xmax]."""
        return np.diff(self.zlim)[0]

    @property
    def box_area(self):
        """Return bounding box area."""
        return self.width*self.height

    @property
    def area(self) -> float:
        """Return polygon area."""
        return self.polygon.area

    @property
    def limit(self):
        """Return polygon bounding box (xmin, xmax, zmin, zmax)."""
        return self.xlim + self.zlim


@dataclass
class PolyDelta(Polygon):
    """Manage grid spacing and cell dimension."""

    delta: Union[int, float] = 0
    turn: str = 'hex'
    nturn: float = 1.
    tile: bool = False
    cell_delta: list[float] = field(init=False)
    grid_delta: list[float] = field(init=False)

    def __post_init__(self):
        """Generate deltas."""
        super().__post_init__()  # generate bounding polygon
        self.generate_deltas()

    def generate_deltas(self):
        """Generate grid and cell deltas."""
        self.turn = polyshape[self.turn]  # inflate turn name
        self.cell_delta = self.dimension_cell()
        self.grid_delta = self.dimension_grid()

    def dimension_cell(self):
        """Return cell dimensions."""
        ndiv_x, ndiv_z = self.divide()
        ndiv_x, ndiv_z = np.max([ndiv_x, 1]), np.max([ndiv_z, 1])
        if self.tile:
            return self.width/ndiv_x, self.height/ndiv_z
        return self.width / np.round(ndiv_x), self.height / np.round(ndiv_z)

    def divide(self):
        """Return number of cell divisions along x and z axis."""
        if self.delta is None or self.delta == 0:
            return 1, 1
        if self.delta < 0:
            if self.delta == -1:
                filament_number = self.nturn
            else:
                filament_number = -self.delta
            fill_fraction = self.area / self.box_area
            box_number = filament_number / fill_fraction
            if self.tile:
                if self.turn in ['circle', 'skin']:
                    box_number *= np.sqrt(3)/2
                elif self.turn == 'hexagon':
                    box_number *= 3/8 * np.sqrt(3)
                    print(box_number, self.turn)

            if np.isclose(aspect := self.width/self.height, 1) and \
                    self.turn in ['circle', 'square', 'skin']:
                delta = np.sqrt(self.box_area / box_number)
                return (ndiv := self.width/delta, ndiv)
            if aspect > 1:
                ndiv_x = np.sqrt(box_number * aspect)
                if not self.tile:
                    ndiv_x = np.round(ndiv_x)
                ndiv_z = box_number / ndiv_x
                return ndiv_x, ndiv_z
            ndiv_z = np.sqrt(box_number / aspect)
            if not self.tile:
                ndiv_z = np.round(ndiv_z)
            ndiv_x = box_number / ndiv_z
            return ndiv_x, ndiv_z
        ndiv_x = self.width / self.delta
        ndiv_z = self.height / self.delta
        return ndiv_x, ndiv_z

    def dimension_grid(self):
        """Return grid delta."""
        grid_delta = list(self.cell_delta)
        if self.tile:
            grid_delta = [delta := boxbound(*grid_delta), delta]
            if self.turn == 'hexagon':
                grid_delta[0] *= 3/2
                grid_delta[1] *= np.sqrt(3)/4
                return grid_delta
            if self.turn in ['circle', 'skin']:
                grid_delta[1] *= np.sqrt(3)/2
                return grid_delta
        return grid_delta


@dataclass
class PolyCell(PolyDelta):
    """Define PolyGrid cell."""

    scale: Union[int, float] = 1
    fill: float = 0.65

    def __post_init__(self):
        """Size polycell."""
        super().__post_init__()  # size grid and polycell
        self._scale_cell()
        self._fill_cell()

    def __call__(self, x_center, z_center):
        """Return cell polygon."""
        polygon = polygen(self.turn)(x_center, z_center, *self.cell_delta)
        return polyframe(polygon, self.turn)

    @property
    def cell(self):
        """Return referance polygon."""
        return self(0, 0)  # refernace cell polygon

    def _scale_cell(self):
        """Apply scaling to cell."""
        self.cell_delta = [self.scale*delta for delta in self.cell_delta]

    def _fill_cell(self):
        """Update fill parameter for skin sections."""
        if self.turn == 'skin':
            self.cell_delta[1] = self.fill


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


@dataclass
class PolyVector:
    """Manage 1D mesh for x and z coordinates."""

    limit: tuple[float]
    cell_delta: tuple[float]
    grid_delta: tuple[float]
    tile: bool = False
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
            return self.polyspace[direction].corner()
        return self.polyspace[direction].face()


@dataclass
class PolyGrid(PolyCell):
    """Construct 2d grid."""

    label: str = 'Poly'
    trim: bool = True
    vector: PolyVector = field(init=False)
    frame: Frame = field(init=False)

    def __post_init__(self):
        """Generate grid."""
        super().__post_init__()
        self.vector = PolyVector(self.limit, self.cell_delta, self.grid_delta,
                                 self.tile)
        self.frame = Frame(required=['x', 'z', 'dl', 'dt'],
                           available=['poly'])
        self.generate_grid()

    def __len__(self):
        """Return grid length."""
        return len(self.frame)

    def clear(self):
        """Clear grid."""
        self.frame = self.frame.iloc[0:0]

    @property
    def coordinates(self):
        """Return grid coordinates."""
        grid = np.meshgrid(self.vector('x'), self.vector('z'), indexing='ij')
        if self.tile:
            grid[0][:, 1::2] += self.grid_delta[0]/2
        return np.hstack((grid[0].reshape(-1, 1), grid[1].reshape(-1, 1)))

    def generate_grid(self):
        """Generate grid."""
        cells = self.generate_cells()
        if self.trim:
            cells = self.trim_cells(cells)
        section = [cell.name for cell in cells]
        cell_area = np.sum([cell.area for cell in cells])
        nturn = [self.nturn*cell.area / cell_area for cell in cells]
        self.frame.insert(0, 0, *self.cell_delta, poly=cells,
                          section=section, nturn=nturn, label=self.label,
                          link=True, delim='_')

    def generate_cells(self):
        """Return polycells."""
        return [self(*coord) for coord in self.coordinates]

    def trim_cells(self, cells):
        """Return polycells trimed to bounding polygon."""
        strtree = shapely.strtree.STRtree(cells)
        buffer = self.polygon.buffer(1e-12*self.cell_delta[0])
        cells = [polyframe(cell, self.turn
                           if poly.within(buffer) else 'polygon')
                 for poly in strtree.query(self.polygon)
                 if (cell := poly.intersection(buffer)) and
                 isinstance(cell, shapely.geometry.Polygon)]
        return cells

    @property
    def cell_area(self):
        """Return sum of polycell areas."""
        return self.frame.area.sum()

    @property
    def cell_nturn(self):
        """Return sum of polycell turns."""
        return self.frame.nturn.sum()

    @property
    def nfilament(self):
        """Return effective filament number."""
        return self.cell_area / self.cell.area

    def plot(self):
        """Plot polygon exterior and polycells."""
        self.plot_exterior()
        self.frame.polyplot()
        plt.axis('off')
        plt.axis('equal')


if __name__ == '__main__':

    polygrid = PolyGrid({'hx': [6, 3, 2.5, 2.5]}, delta=-60,
                        turn='hex', tile=True, trim=True)

    #polygrid = PolyGrid({'r': [6, 3, 2.5, 2.5]}, delta=-4,
    #                    turn='sk', tile=False, trim=True)
    print(polygrid.delta, polygrid.nfilament)
    polygrid.plot()
