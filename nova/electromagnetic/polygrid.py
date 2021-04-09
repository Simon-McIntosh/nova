"""Grid methods."""
from dataclasses import dataclass, field
from typing import Union

import shapely.geometry
import numpy as np

from nova.electromagnetic.frame import Frame
from nova.electromagnetic.polygen import (
    polygen, polyframe, polyshape, boxbound
    )
from nova.utilities.pyplot import plt


# pylint:disable=unsubscriptable-object


@dataclass
class PolyPatch:
    """Generate polygons."""

    patch: Union[dict[str, list[float]], list[float], shapely.geometry.Polygon]

    def __post_init__(self):
        """Init bounding polygon."""
        self.polygon = self.generate_polygon(self.patch)

    def generate_polygon(self, patch):
        """
        Generate polygon.

        Parameters
        ----------
        patch :
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
        if isinstance(patch, PolyPatch):
            return patch
        if isinstance(patch, shapely.geometry.Polygon):
            return self.orient(patch)
        if isinstance(patch, dict):
            polys = [polygen(section)(*patch[section]) for section in patch]
            polygon = shapely.ops.unary_union(polys)
            try:
                return self.orient(polygon)
            except AttributeError as nonintersecting:
                raise AttributeError('non-overlapping polygons specified in '
                                     f'{patch}') from nonintersecting
        patch = np.array(patch)  # to numpy array
        if patch.ndim == 1:   # patch bounding box
            if len(patch) == 4:  # [xmin, xmax, zmin, zmax]
                xlim, zlim = patch[:2], patch[2:]
                x_center, z_center = np.mean(xlim), np.mean(zlim)
                width, height = np.diff(xlim)[0], np.diff(zlim)[0]
                polygon = polygen('rectangle')(x_center, z_center,
                                               width, height)
                return self.orient(polygon)
            raise IndexError('malformed bounding box\n'
                             f'patch: {patch}\n'
                             'require [xmin, xmax, zmin, zmax]')
        if patch.shape[1] != 2:
            patch = patch.T
        if patch.ndim == 2 and patch.shape[1] == 2:  # loop
            polygon = shapely.geometry.Polygon(patch)
            return self.orient(polygon)
        raise IndexError('malformed bounding loop\n'
                         f'shape(patch): {patch.shape}\n'
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
    def area(self) -> float:
        """Return polygon area."""
        return self.polygon.area

    @property
    def limit(self):
        """Return polygon bounding box (xmin, xmax, zmin, zmax)."""
        return self.xlim + self.zlim


@dataclass
class PolyCell(PolyPatch):
    """Define PolyGrid unit cell."""

    delta: Union[int, float] = 0
    scale: Union[int, float] = 1
    section: str = 'hex'
    fill: float = 0.65
    tile: bool = False
    cell_delta: list[float] = field(init=False)
    grid_delta: list[float] = field(init=False)

    def __post_init__(self):
        """Set step."""
        super().__post_init__()  # generate bounding polygon
        self.generate_cell()

    def __call__(self, x_center, z_center):
        """Return cell polygon."""
        polygon = polygen(self.section)(x_center, z_center, *self.cell_delta)
        return polyframe(polygon, self.section)

    def generate_cell(self):
        """Generate unit cell."""
        self.section = polyshape[self.section]  # inflate section name
        self.cell_delta = self.dimension_cell()
        self.grid_delta = self.dimension_grid()
        self.scale_cell()
        if self.section == 'skin':
            self.cell_delta[1] = self.fill
        self.cell = self(0, 0)  # refernace cell polygon

    def dimension_cell(self):
        """Return cell dimensions."""
        ndiv_x, ndiv_z = self.divisions()
        return self.width/ndiv_x, self.height/ndiv_z

    def divisions(self):
        """Return number of cell divisions along x and z axis."""
        if self.delta is None or self.delta == 0:
            return 1, 1
        if self.delta < 0:
            filament_number = -self.delta
            if np.isclose(aspect := self.width/self.height, 1):
                delta = np.sqrt(self.area / filament_number)
                ndiv_x = ndiv_z = np.max([np.round(self.width/delta), 1])
                return ndiv_x, ndiv_z
            fill_fraction = self.area / (self.width*self.height)
            box_number = filament_number / fill_fraction
            if aspect > 1:
                ndiv_x = np.max([np.round(np.sqrt(box_number * aspect)), 1])
                ndiv_z = np.max([np.round(box_number / ndiv_x), 1])
                return ndiv_x, ndiv_z
            ndiv_z = np.max([np.round(np.sqrt(box_number / aspect)), 1])
            ndiv_x = np.max([np.round(box_number / ndiv_z), 1])
            return ndiv_x, ndiv_z
        ndiv_x = self.width / self.delta
        ndiv_z = self.height / self.delta
        return ndiv_x, ndiv_z
        #ndiv_x = np.max([np.round(self.width / self.delta), 1])
        #ndiv_z = np.max([np.round(self.height / self.delta), 1])
        return ndiv_x, ndiv_z

    def scale_cell(self):
        """Apply scaling to cell."""
        self.cell_delta = [self.scale*delta for delta in self.cell_delta]

    def dimension_grid(self):
        """Return grid delta."""
        grid_delta = list(self.cell_delta)
        print(self.tile, grid_delta, self.cell_delta)
        if self.tile:
            grid_delta = [delta := boxbound(*grid_delta), delta]
        if np.isclose(*grid_delta, 1e-3*np.mean(grid_delta)) and \
                self.section == 'hexagon':
            length = boxbound(grid_delta[0]/2, grid_delta[0]/np.sqrt(3))
            grid_delta = [2*length, np.sqrt(3)*length]
        if self.tile and self.section == 'hexagon':
            grid_delta[0] *= 3/2
            grid_delta[1] *= 0.5
            return grid_delta
        if self.tile and self.section in ['circle', 'skin']:
            grid_delta[1] *= np.sqrt(3)/2
            return grid_delta
        return grid_delta


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

        #if self.tile:
        #    self.start -= self.grid_delta
        #    #self.stop += self.grid_delta
        #    self.ndiv += 1

    def divide(self):
        """Divide base length by delta, update stop parameter."""
        self.base_length = self.stop-self.start
        self.ndiv = int(np.round(self.base_length / self.grid_delta))
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
        sections = [cell.name for cell in cells]
        turns = [cell.area / self.cell.area for cell in cells]
        self.frame.insert(0, 0, *self.cell_delta, poly=cells, section=sections,
                          Nt=turns)

    def generate_cells(self):
        """Return polycells."""
        return [self(*coord) for coord in self.coordinates]

    def trim_cells(self, cells):
        """Return polycells trimed to bounding polygon."""
        strtree = shapely.strtree.STRtree(cells)
        buffer = self.polygon.buffer(1e-12*self.cell_delta[0])
        cells = [polyframe(cell, self.section
                           if poly.within(buffer) else 'polygon')
                 for poly in strtree.query(self.polygon)
                 if (cell := poly.intersection(self.polygon)) and
                 isinstance(cell, shapely.geometry.Polygon)]
        return cells

    @property
    def cell_area(self):
        """Return sum of polycell areas."""
        return self.frame.dA.sum()

    @property
    def cell_turns(self):
        """Return sum of polycell turns."""
        return self.frame.Nt.sum()


if __name__ == '__main__':

    polygrid = PolyGrid({'r': [6, 3, 0.4, 0.65]}, delta=-5,
                        section='s', scale=0.99, fill=0.65,
                        tile=False, trim=True)
    print(len(polygrid), polygrid.cell_turns)

    plt.axis('off')
    plt.axis('equal')
    polygrid.frame.polyplot()
    polygrid.plot_exterior()
