
import numpy as np
from pandas import DataFrame
from pandas.api.types import is_list_like
from scipy.interpolate import interp1d
import shapely.geometry

from nova.utilities.pyplot import plt
from nova.utilities.geom import length
from nova.electromagnetic.meshgrid import MeshGrid
from nova.electromagnetic.biotsavart import BiotSet
from nova.electromagnetic.coilmatrix import CoilMatrix
from nova.electromagnetic.topology import Topology


class Mutual(BiotSet):
    _biot_attributes = []
    _default_biot_attributes = {'target_turns': True,
                                'reduce_target': True}

    def __init__(self, subcoil, **biot_attributes):
        BiotSet.__init__(self, source=subcoil, target=subcoil,
                         **biot_attributes)


class ForceField(Mutual):
    _biot_attributes = []
    _default_biot_attributes = {'target_turns': False,
                                'reduce_target': False}

    def __init__(self, subcoil, **biot_attributes):
        Mutual.__init__(self, subcoil, **biot_attributes)


class Probe(BiotSet):
    """Probe interaction methods and data. Extends BiotSet class."""

    _biot_attributes = ['target']
    _default_biot_attributes = {'target_turns': False, 'reduce_target': False}

    def __init__(self, subcoil, **biot_attributes):
        BiotSet.__init__(self, source=subcoil, **biot_attributes)

    def add_target(self, *args, **kwargs):
        """
        Add target to probe.

        Target addition managed by CoilFrame.add_coil()

        Parameters
        ----------
        *args : DataFrame or a pair of 1D arrays
            Data to be inserted into CoilFrame.
        **kwargs : dict
            See CoilFrame for description of keyword arguments.

        Returns
        -------
        None.

        """
        self.target.add_coil(*args, name='Target', delim='', **kwargs)
        self.assemble_biotset()

    def plot_targets(self, ax=None, **kwargs):
        """
        Plot targets.

        Parameters
        ----------
        ax : Axes, optional
            target axes. The default is None.
        **kwargs : dict
            Keyword arguments passed to ax.plot.

        Returns
        -------
        None.

        """
        if ax is None:
            ax = plt.gca()
        ls = kwargs.pop('ls', '.')
        color = kwargs.pop('color', 'C3')
        ax.plot(self.target.x, self.target.z, ls, color=color, **kwargs)


class Field(Probe):
    """Field values imposed on coil boundaries - extends Probe class."""

    _biot_attributes = Probe._biot_attributes + ['_coil_index']

    def __init__(self, subcoil, **biot_attributes):
        Probe.__init__(self, subcoil, **biot_attributes)

    def add_coil(self, coil, parts, dField=0.5):
        """
        Add field probes spaced around each coil perimiter.

        Parameters
        ----------
        coil : CoilFrame
            Coil coilframe.
        parts : str or list
            Part names to include field calculation.
        dField : float, optional
            Coil boundary probe resoultion. The default is 0.5.

        Returns
        -------
        None.

        """
        if not is_list_like(parts):
            parts = [parts]
        self._coil_index = []
        target = {'x': [], 'z': [], 'coil': [], 'nC': []}
        for part in parts:
            for index in coil.index[coil.part == part]:
                self._coil_index.append(index)
                x, z = coil.polygon[index].boundary.coords.xy
                if dField == 0:  # no interpolation
                    polygon = {'x': x, 'z': z}
                else:
                    if dField == -1:  # extract dField from coil
                        _dL = coil.loc[index, 'dCoil']
                    else:
                        _dL = dField
                    nPoly = len(x)
                    polygon = {'_x': x, '_z': z,
                               '_L': length(x, z, norm=False)}
                    for attr in ['x', 'z']:
                        polygon[f'interp{attr}'] = \
                            interp1d(polygon['_L'], polygon[f'_{attr}'])
                        dL = [polygon[f'interp{attr}'](
                            np.linspace(
                                polygon['_L'][i], polygon['_L'][i+1],
                                1+int(np.diff(polygon['_L'][i:i+2])[0]/_dL),
                                endpoint=False))
                              for i in range(nPoly-1)]
                        polygon[attr] = np.concatenate(dL).ravel()
                nP = len(polygon['x'])
                target['x'].extend(polygon['x'])
                target['z'].extend(polygon['z'])
                target['coil'].extend([index for __ in range(nP)])
                target['nC'].append(nP)
        self.target.add_coil(target['x'], target['z'],
                             label='Field', delim='',
                             coil=target['coil'])
        _nC = 0
        for nC in target['nC']:
            index = [f'Field{i}' for i in np.arange(_nC, _nC+nC)]
            self.target.add_mpc(index)
            _nC += nC
        self.assemble_biotset()

    @property
    def frame(self):
        """
        Return DataFrame of coil properties (reduceat), read-only.

        Returns
        -------
        frame : DataFrame
            Coil properties:

                - B : maximum L2norm field on perimeter of each coil.

        """
        return DataFrame(
            np.maximum.reduceat(self.B, self.target._reduction_index),
            index=self._coil_index, columns=['B'])


class PlasmaFilament(Probe):
    """Plasma filament interaction methods and data. Class extends Probe."""

    def __init__(self, subcoil, **biot_attributes):
        Probe.__init__(self, subcoil, **biot_attributes)

    def add_plasma(self):
        """Add plasma filaments from source coilframe as targets."""
        self.source.update_coilframe()
        self.add_target(
            self.source.coilframe.x[self.source.coilframe.plasma],
            self.source.coilframe.z[self.source.coilframe.plasma])


class Colocate(Probe):
    """Colocation probes - used by inverse (nova.design)."""

    _biot_attributes = ['label', 'x', 'z', 'value',
                        'nx', 'nz', 'd_dx', 'd_dz',
                        'factor', 'weight']


    def add_targets(self, *args, **kwargs):
        '''
        add colocation points

        len(args) == 1 (DataFrame or dict): colocation points as frame
        len(args) == fix.ncol (float or array): colocation points as args
        len(args) == 0

            (DataFrame or fix.columns): fix data points
        kwargs:
            (float): alternate input method
        '''
        default = {'label': 'Psi', 'x': 0., 'z': 0.,
                   'value': 0., 'd_dx': 0., 'd_dz': 0.,
                   'nx': 0., 'nz': 0., 'factor': 1., 'weight': 1.}
        if len(args) == 1:  # as frame
            target = args[0]
        else:  # as args and kwargs
            target = {key: value
                      for key, value in zip(self._target_attributes, args)}
            for key in kwargs:
                target[key] = kwargs[key]
        # populate missing entries with defaults
        if len(target) != self.targets.shape[1]:  # fill defaults
            for key in default:
                if key not in target:
                    target[key] = default[key]
        nrow = np.max([len(arg) if is_list_like(arg) else 1 for arg in args])
        target = DataFrame(target, index=range(nrow))
        norm = np.linalg.norm([target['nx'], target['nz']], axis=0)
        for nhat in ['nx', 'nz']:
            target.loc[target.index[norm != 0], nhat] /= norm[norm != 0]
        Probe.add_targets(self, target)  # append Biot colocation targets

    def update_targets(self):
        'update targets.value from Psi and/or field'
        psi_index = ['Psi' in l for l in self.targets.label]
        self.targets.loc[psi_index, 'value'] = self.Psi[psi_index]

    def set_weight(self, index, gradient):
        index &= (gradient > 0)  # ensure gradient > 0
        self.targets.loc[index, 'weight'] = 1 / gradient[index]

    def update_weight(self):
        'update colocation weight based on inverse of absolute gradient'
        gradient = self.targets.loc[:, ['d_dx', 'd_dz']].to_numpy()
        normal = self.targets.loc[:, ['nx', 'nz']].to_numpy()
        d_dx, d_dz = gradient.T
        # compute gradient magnitudes
        gradient_L2 = np.linalg.norm(gradient, axis=1)  # L2norm
        field_index = np.array(['B' in l for l in self.targets.label])
        self.set_weight(field_index, gradient_L2)
        gradient_dot = abs(np.array([g @ n for g, n in zip(gradient, normal)]))
        bndry_index = ['bndry' in l for l in self.targets.label]
        self.set_weight(bndry_index, gradient_dot)
        # calculate mean weight
        if sum(bndry_index) > 0:
            mean_index = bndry_index
        elif sum(field_index) > 0:
            mean_index = field_index
        else:
            mean_index = slice(None)
        mean_weight = self.targets.weight[mean_index].mean()
        # not field or Psi_bndry (separatrix)
        mean_index = [not field and not bndry for field, bndry in zip(
                    field_index, bndry_index)]

        self.targets.loc[mean_index, 'weight'] = mean_weight
        self.wsqrt = np.sqrt(self.targets.factor *
                             self.targets.weight)
        self.wsqrt /= np.mean(self.wsqrt)  # normalize weight


    def plot_colocate(self, tails=True):
        self.update_weight()

        style = DataFrame(index=['color', 'marker', 'markersize',
                                 'markeredgewidth'])

        plt.plot(self.colocate)
        '''
            psi, Bdir, Bxz = [], [], []
            tail_length = 0.75
            for bc, w in zip(self.fix['BC'], weight):
                if 'psi' in bc:
                    psi.append(w)
                elif bc == 'Bdir':
                    Bdir.append(w)
                elif bc == 'Bx' or bc == 'Bz':
                    Bxz.append(w)
            if len(psi) > 0:
                psi_norm = tail_length / np.mean(psi)
            if len(Bdir) > 0:
                Bdir_norm = tail_length / np.mean(Bdir)
            if len(Bxz) > 0:
                Bxz_norm = tail_length / np.mean(Bxz)
            for x, z, bc, bdir, w in zip(self.fix['x'], self.fix['z'],
                                         self.fix['BC'], self.fix['Bdir'],
                                         weight):
                if bdir[0]**2 + bdir[1]**2 == 0:  # tails
                    direction = [0, -1]
                else:
                    direction = bdir
                # else:
                #    d_dx,d_dz = self.get_gradients(bc,x,z)
                #    direction = np.array([d_dx,d_dz])/np.sqrt(d_dx**2+d_dz**2)
                if 'psi' in bc:
                    norm = psi_norm
                    marker, size, color = 'o', 7.5, 'C0'
                    plt.plot(x, z, marker, color=color, markersize=size)
                    plt.plot(x, z, marker, color=[1, 1, 1],
                             markersize=0.3 * size)
                else:
                    if bc == 'Bdir':
                        norm = Bdir_norm
                        marker, size, color, mew = 'o', 4, 'C1', 0.0
                    elif bc == 'null':
                        norm = Bxz_norm
                        marker, size, color, mew = 'o', 2, 'C2', 0.0
                    elif bc == 'Bx':
                        norm = Bxz_norm
                        marker, size, color, mew = '_', 10, 'C2', 1.75
                    elif bc == 'Bz':
                        norm = Bxz_norm
                        marker, size, color, mew = '|', 10, 'C2', 1.75
                    plt.plot(x, z, marker, color=color, markersize=size,
                             markeredgewidth=mew)
                if tails:
                    plt.plot([x, x + direction[0] * norm * w],
                             [z, z + direction[1] * norm * w],
                             color=color, linewidth=2)
            plt.axis('equal')
        '''

    def plot(self):
        plt.plot(self.targets.x, self.targets.z, 'o')


class Grid(BiotSet, Topology):
    """
    Grid interaction methods and data.

    Key Attributes
    --------------
        n : int
            Grid dimension.
        limit : array-like, shape(4,)
            Grid limits.
        expand : float
            Grid expansion beyond coils.
        nlevels : int
            Number of contour levels.
        boundary : str
            Limit boundary ['limit', 'expand_limit'].

    Derived Attributes
    ------------------
        n2d : array-like, shape(2,)
            As meshed dimension.
        x2d : array-like shape(*n2D)
            2D x-coordinates (radial).
        z2d : array-like shape(*n2D)
            2D z-coordinates.
        levels : array-like
            Contour levels.
        Psi : array-like
            Poloidal flux.
        Bx : array-like
            Radial field.
        Bz : array-like
            Vertical field.

    """

    _biot_attributes = ['n', 'n2d', 'limit', 'expand_limit', 'boundary',
                        'grid_polygon', 'expand', 'nlevels', 'levels',
                        'x', 'z', 'x_index', 'z_index', 'x2d', 'z2d',
                        'dx', 'dz', 'target']

    _default_biot_attributes = {'n': 1e4, 'expand': 0.05, 'nlevels': 51,
                                'boundary': 'coilset'}

    def __init__(self, subcoil, **biot_attributes):
        """
        Links Grid class to subcoil and (re)initalizes biot_attributes.

        Extends methods provided by BiotSet and CoilMatrix

        Parameters
        ----------
        subcoil : CoilFrame
            Source coilframe (subcoils).
        **biot_attributes : dict
            Composite biot attributes.

        Returns
        -------
        None.

        """
        Topology.__init__(self)
        BiotSet.__init__(self, source=subcoil, **biot_attributes)

    def _flag_update(self, status):
        """
        Extend CoilMatrix flag_update.

        Parameters
        ----------
        status : bool
            flag status, set update flags if True.

        Returns
        -------
        None.

        """
        BiotSet._flag_update(self, status)
        Topology._flag_update(self, status)

    def solve_biot(self):
        """
        Evaluate all biot attributes.

        Returns
        -------
        None.

        """
        if self.target.nT > 0:
            for attribute in self._interpolate_attributes:
                self._evaluate_spline(attribute)
        CoilMatrix.solve_biot(self)

    @property
    def grid_boundary(self):
        """
        Manage grid bounding box based on self.boundary.

        Parameters
        ----------
        boundary : str
            Boundary flag ether 'limit' or 'expand_limit'.
            Set self.regen=True if boundary flag is changed.

        Raises
        ------
        IndexError
            Flag not in [limit, expand_limit].

        Returns
        -------
        limit : array_like, shape(4,)
            Grid bounding box [xmin, xmax, zmin, zmax]

        """
        if self.boundary == 'limit':
            return self.limit
        elif self.boundary == 'expand_limit':
            return self.expand_limit
        else:
            raise IndexError(f'boundary {self.boundary} not in '
                             '[limit, expand_limit]')

    @grid_boundary.setter
    def grid_boundary(self, boundary):
        if boundary != self.boundary:
            self.regen = True  # force update after boundary switch
        if boundary in ['limit', 'expand_limit']:
            self.boundary = boundary
        else:
            raise IndexError(f'boundary label {boundary} not in '
                             '[limit, expand_limit]')

    def _set_boundary(self, **kwargs):
        if 'boundary' in kwargs:  # set flag directly [limit, expand_limit]
            boundary = kwargs['boundary']
        elif 'expand' in kwargs:  # then expand (coilset limit)
            boundary = 'expand_limit'
        elif 'limit' in kwargs:  # then limit
            boundary = 'limit'
        else:  # defaults to coilset limit
            boundary = 'expand_limit'
        self.grid_boundary = boundary

    def generate_grid(self, regen=False, **kwargs):
        """
        Generate grid for use as targets in Biot Savart calculations.

        Compare kwargs to current grid settings, update grid on-demand and
        return update flag

        Parameters
        ----------
        regen : bool
            Force grid regeneration.
        boundary : str
            Grid boundary flag ['expand_limit' or 'limit'].

        Keyword Arguments
        -----------------
        n : int, optional
            Grid node number.
        limit : array_like, size(4,)
            Grid bounding box [xmin, xmax, zmin, zmax].
        expand : float, optional
            Expansion beyond coil/boundary (when limit not set)
        nlevels : int, optional
            Number of contour levels.
        levels : array_like, optional

        Returns
        -------
        regenerate_grid : bool
            Generation flag.

        """
        self.regen = regen
        self._set_boundary(**kwargs)
        self.assemble_biotset()
        grid_attributes = {}  # grid attributes
        for key in ['n', 'limit', 'expand', 'levels', 'nlevels']:
            attribute = kwargs.get(key, getattr(self, key))
            if key in ['n', 'nlevels']:
                attribute = int(attribute)
            grid_attributes[key] = attribute
        # calculate coilset limit
        grid_attributes['expand_limit'] = \
            self._get_expand_limit(grid_attributes['expand'])
        # switch to coilset limit if limit is None
        if grid_attributes['limit'] is None:
            self.grid_boundary = 'expand_limit'
        regenerate_grid = \
            not np.array_equal(grid_attributes[self.boundary],
                               self.grid_boundary) or \
            grid_attributes['expand'] != self.expand or \
            grid_attributes['n'] != self.n or self.regen
        if regenerate_grid:
            self._generate_grid(**grid_attributes)
            self.grid_polygon = shapely.geometry.box(
                *self.grid_boundary[::2], *self.grid_boundary[1::2])
        return regenerate_grid

    def _generate_grid(self, **grid_attributes):
        self.biot_attributes = grid_attributes  # update attributes
        if self.n > 0:
            mg = MeshGrid(self.n, self.grid_boundary)  # set mesh
            self.n2d = [mg.nx, mg.nz]  # shape
            self.x, self.z = mg.x, mg.z  # axes
            # trace index interpolators
            self.x_index = interp1d(range(self.n2d[0]), self.x)
            self.z_index = interp1d(range(self.n2d[1]), self.z)
            # grid deltas
            self.dx = np.diff(self.grid_boundary[:2])[0] / (mg.nx - 1)
            self.dz = np.diff(self.grid_boundary[2:])[0] / (mg.nz - 1)
            # 2d coordinates
            self.x2d = mg.x2d
            self.z2d = mg.z2d
            self.target.drop_coil()  # clear target
            self.target.add_coil(self.x2d.flatten(), self.z2d.flatten(),
                                 name='Grid', delim='')
            self.assemble_biotset()
            self.update_biot = True

    @property
    def grid_polygon(self):
        """
        Manage boundary polygon.

        Returns
        -------
        grid_polygon : shapely.polygon or None
            Grid bounding box.

        """
        return self._grid_polygon

    @grid_polygon.setter
    def grid_polygon(self, polygon):
        self._grid_polygon = polygon
        self._update_coil_center = True

    def _get_expand_limit(self, expand=None, xmin=1e-3):
        if expand is None:
            expand = self.expand  # use default
        if self.source.empty:
            raise IndexError('source coilframe empty')
        x, z, = self.source.x, self.source.z
        dx, dz = self.source.dx, self.source.dz
        limit = np.array([(x - dx/2).min(), (x + dx/2).max(),
                          (z - dz/2).min(), (z + dz/2).max()])
        return self._expand_limit(limit, expand, xmin)

    @staticmethod
    def _expand_limit(limit, expand, xmin, fix_aspect=False):
        dx, dz = np.diff(limit[:2])[0], np.diff(limit[2:])[0]
        if not fix_aspect:
            dx = dz = np.mean([dx, dz])
        limit += expand/2 * np.array([-dx, dx, -dz, dz])
        if limit[0] < xmin:
            limit[0] = xmin
        return limit

    def plot_grid(self, ax=None, **kwargs):
        """
        Extend MeshGrid.plot - plot target grid.

        Parameters
        ----------
        ax : Axes, optional
            Plot Axes.
        **kwargs : dict
            Keyword arguments passed to MeshGrid.plot.

        Returns
        -------
        None.

        """
        self.generate_grid(**kwargs)
        if ax is None:
            ax = plt.gca()
        MeshGrid._plot(self.x2d, self.z2d, self._limit[:2], self._limit[2:],
                       ax=ax, zorder=-500, **kwargs)  # plot grid

    def plot_flux(self, ax=None, lw=1, color='lightgrey', **kwargs):
        """
        Plot constant flux contours.

        Parameters
        ----------
        ax : Axes, optional
            Plot axes.
        lw : float, optional
            line weight.
        color : str, optional
            line color. The default is 'lightgrey'.
        **kwargs : dict, optional
            Keyword arguments passed to ax.contour.

        Returns
        -------
        levels : QuadContourSet.levels
            Contour levels.

        """
        if self.n > 0:
            if ax is None:
                ax = plt.gca()
            levels = kwargs.get('levels', self.levels)
            if levels is None:
                levels = self.nlevels
            QuadContourSet = ax.contour(
                    self.x2d, self.z2d, self.Psi,
                    levels, colors=color, linestyles='-',
                    linewidths=lw,
                    alpha=0.9, zorder=4)
            if self.levels is None:
                self.levels = QuadContourSet.levels
            plt.axis('equal')
            return QuadContourSet.levels

    def plot_field(self, ax=None, **kwargs):
        """
        Generate field quiver plot.

        Returns
        -------
        None.

        """
        if self.n > 0:
            if ax is None:
                ax = plt.gca()
            levels = kwargs.get('levels', self.levels)
            if levels is None:
                levels = self.nlevels
            ax.contour(
                    self.x2d, self.z2d, self.B,
                    31, linestyles='-', alpha=0.9, zorder=4)


class PlasmaGrid(Grid):
    """Plasma grid interaction methods and data. Class extends Grid."""

    _biot_attributes = Grid._biot_attributes + ['plasma_boundary']

    _default_biot_attributes = {**Grid._default_biot_attributes,
                                **{'expand': 0.1, 'nlevels': 21,
                                   'boundary': 'limit'}}

    def __init__(self, subcoil, **biot_attributes):
        Grid.__init__(self, subcoil, **biot_attributes)

    def _get_expand_limit(self, expand=None, xmin=1e-3):
        """Overide Grid._get_expand_limit."""
        if expand is None:
            expand = self.expand  # use default
        else:
            self.expand = expand  # update
        if self.plasma_boundary is None:
            raise IndexError('Plasma boundary not set')
        bounds = self.plasma_boundary.bounds
        limit = [*bounds[::2], *bounds[1::2]]
        return self._expand_limit(limit, expand, xmin)

    @property
    def plasma_boundary(self):
        """
        Manage plasma boundary (limit surface).

        Parameters
        ----------
        plasma_boundary : array-like, shape(n,2) or Polygon
            Plasma boundary.

        Returns
        -------
        plasma_boundary : shapely.Polygon
            Plasma boundary.

        """
        return self._plasma_boundary

    @plasma_boundary.setter
    def plasma_boundary(self, plasma_boundary):
        if plasma_boundary is None:
            pass
        elif not isinstance(plasma_boundary, shapely.geometry.Polygon):
            plasma_boundary = shapely.geometory.Polygon(plasma_boundary)
        self._plasma_boundary = plasma_boundary

    @property
    def plasma_n(self):
        """
        Return default plasma grid point number.

        Point number calculated to produce a grid with a nodal density
        equal the plasma filament density.

        Returns
        -------
        n_plasma : int
            plasmagrid node number.

        """
        grid_limit = self._get_expand_limit()
        grid_area = (grid_limit[1]-grid_limit[0]) * \
            (grid_limit[3]-grid_limit[2])
        plasma_filament_density = \
            np.sum(self.source.coilframe.plasma) /\
            self.source.coilframe.dA[self.source.coilframe.plasma].sum()
        return int(plasma_filament_density*grid_area)

    def generate_grid(self, regen=False, **kwargs):
        """
        Generate plasma grid.

        Accepts keyword aguments with plasma_* prefix. Plasma prefix
        to enables shared attribute setting with base Grid instance
        in CoilSet.

        Parameters
        ----------
        regen : bool
            Force grid regeneration.

        Keyword Arguments
        -----------------
        n : int, optional
            Plasma grid node number.
        boundary : array_like or Polygon, optional
            External plasma boundary. A positively oriented curve or
            bounding box.
        expand : float, optional
            Expansion beyond boundary (when limit not set)
        nlevels : int, optional
            Number of contour levels
        levels : float array_like, optional
            Explicit values for contour levels
        regen : bool
            Force grid regeneration

        Returns
        -------
        regen flag : bool.

        """
        if 'n' not in kwargs and 'plasma_n' not in kwargs:  # auto size
            kwargs['plasma_n'] = self.plasma_n
        return Grid.generate_grid(self, **self._merge_plasma_kwargs(**kwargs))

    def _merge_plasma_kwargs(self, **kwargs):
        """
        Merge plasma kwargs with Grid defaults.

        Extract keys with plasma_* prefix. Strip prefix and merge with
        base kwargs. Priority given to the plasma_* kwarg.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to Grid.generate_grid.

        Returns
        -------
        kwargs: dict
            Merged kwargs.

        """
        base_kwargs = {}
        plasma_kwargs = {}
        for key in kwargs:
            if key[:7] == 'plasma_':
                plasma_kwargs[key[7:]] = kwargs[key]
            else:
                base_kwargs[key] = kwargs[key]
        return {**base_kwargs, **plasma_kwargs}

    @property
    def bounds(self):
        """
        Return grid bounds.

        Returns
        -------
        bounds: array-like, shape(2, 2)
            Grid bounds [(xmin, xmax), (zmin, zmax)].

        """
        return np.array([self.grid_boundary[:2], self.grid_boundary[2:]])


class BiotMethods:
    """
    Manage biot methods for CoilSet.

    Examples
    --------
    Initialize biot methods

    >>> cs = CoilSet()
    >>> cs.biot_instances = 'field'
    >>> cs.biot_instances = ['probe', 'colocate']
    >>> cs.biot_instances = {'local_grid': 'grid'}
    >>> print(cs.biot_instances)
    {'plasmagrid': 'plasmagrid',
     'plasmafilament': 'plasmafilament',
     'local_grid': 'grid'}

    """

    _biot_methods = {'mutual': Mutual,
                     'forcefield': ForceField,
                     'probe': Probe,
                     'field': Field,
                     'colocate': Colocate,
                     'grid': Grid,
                     'plasmagrid': PlasmaGrid,
                     'plasmafilament': PlasmaFilament}

    def __init__(self):
        """Initialize biot instances."""
        self._biot_instances = {}

    @property
    def biot_instances(self):
        """
        Initialize biot methods.

        Maintain dictionary of initialized biot methods

        Parameters
        ----------
        biot_instances : str or list or dict
            Collection of biot instances to initialize.

            - str or list : biot name and method assumed equal
            - dict : biot name and method diffrent {biot_name: biot_method}
            - dict : biot name, method and attributes
              {biot_name: [biot_method, biot_attributes], ...}

        Raises
        ------
        IndexError
            biot_method not given in self._biot_methods.

        Returns
        -------
        biot_instances : dict
            Dict of initialized biot methods.

        """
        return self._biot_instances

    @biot_instances.setter
    def biot_instances(self, biot_instances):
        if not is_list_like(biot_instances):
            biot_instances = [biot_instances]
        for biot_name in biot_instances:
            if isinstance(biot_instances, dict):
                if is_list_like(biot_instances[biot_name]):
                    biot_method, biot_attributes = biot_instances[biot_name]
                else:
                    biot_method = biot_instances[biot_name]
                    biot_attributes = {}
            else:
                biot_method = biot_name
                biot_attributes = {}
            if biot_method in self._biot_methods:
                if biot_name not in self._biot_instances:
                    self._biot_instances.update({biot_name: biot_method})
                if not hasattr(self, biot_name):  # initialize method
                    self._initialize_biot_method(biot_name, biot_method,
                                                 **biot_attributes)
            else:
                raise IndexError(f'method {biot_method} not found in '
                                 f'{self._biot_methods}\n'
                                 'unable to initialize method')

    def clear_biot(self):
        """Remove all biot instances."""
        for biot_name in self.biot_instances:
            delattr(self, biot_name)
        self._biot_instances = {}

    def _initialize_biot_method(self, name, method, **attributes):
        """Create biot instance and link to method."""
        setattr(self, name,
                self._biot_methods[method](self.subcoil, **attributes))

    @property
    def biot_attributes(self):
        """
        Manage attributes for all biot_instances.

        Parameters
        ----------
        biot_attributes : dict
            Set biot_attributes, default {}.

        Returns
        -------
        _biot_attributes : dict
            biot_attributes for all biot_instances.

        """
        _biot_attributes = {}
        for instance in self._biot_instances:
            biot_attribute = '_'.join([instance, 'biot_attributes'])
            _biot_attributes[biot_attribute] = \
                getattr(getattr(self, instance), 'biot_attributes')
        return _biot_attributes

    @biot_attributes.setter
    def biot_attributes(self, biot_attributes):
        for instance in self._biot_instances:
            biot_attribute = '_'.join([instance, 'biot_attributes'])
            setattr(getattr(self, instance), 'biot_attributes',
                    biot_attributes.get(biot_attribute, {}))
            getattr(self, instance).assemble_biotset()

    def _get_instance_attributes(self, attribute):
        return {instance: getattr(getattr(self, instance), attribute)
                for instance in self._biot_instances}

    def _set_instance_attributes(self, attribute, status):
        if not isinstance(status, bool):
            raise ValueError(f'flag type {type(status)} must be bool')
        else:
            for instance in self._biot_instances:
                setattr(getattr(self, instance), attribute, status)

    @property
    def update_plasma_turns(self):
        r"""
        Manage biot instance plasma_turn flags.

        Parameters
        ----------
        status : bool
            Set update flag for all biot_instances.
            Set flag to True following a change to plasma interaction
            matrix (plasma turns).
            Setting flag to True ensures that interaction matrix
            :math:`\_m\_` is re-evaluated

        Returns
        -------
        status : nested dict
            plasma_turn flag status for all biot instances.

        """
        return self._get_instance_attributes('update_plasma_turns')

    @update_plasma_turns.setter
    def update_plasma_turns(self, status):
        self._set_instance_attributes('update_plasma_turns', status)

    @property
    def update_coil_current(self):
        r"""
        Manage biot instance coil_current flags.

        Parameters
        ----------
        status : bool
            Set update flag for all biot_instances.
            Set flag to True following a change to coil currents.
            Setting flag to True ensures that interaction matrix
            dot product is re-evaluated

            .. math::
                \_M = \_m \cdot I_c

        Returns
        -------
        status : nested dict
            coil_current flag status for all biot instances.

        """
        return self._get_instance_attributes('update_coil_current')

    @update_coil_current.setter
    def update_coil_current(self, status):
        self._set_instance_attributes('update_coil_current', status)

    @property
    def update_plasma_current(self):
        r"""
        Manage biot instance plasma_current flags.

        Parameters
        ----------
        status : bool
            Set update flag for all biot_instances.
            Set flag to True following a change to plasma current or plasma
            interaction matrix (plasma turns).
            Setting flag to True ensures that interaction matrix
            dot product is re-evaluated

            .. math::
                \_M\_ = \_m\_ \cdot I_p

        Returns
        -------
        status : nested dict
            plasma_current flag status for all biot instances.

        """
        return self._get_instance_attributes('update_plasma_current')

    @update_plasma_current.setter
    def update_plasma_current(self, status):
        self._set_instance_attributes('update_plasma_current', status)

    def solve_biot(self):
        """
        Calculate interactions for all biot instances.

        Returns
        -------
        None.

        """
        for instance in self._biot_instances:
            getattr(self, instance).solve_biot()

    @property
    def dField(self):
        """
        Field probe resolution.

        Parameters
        ----------
        dField : float
            Resoultion of field probes spaced around the perimiters of
            specified coils.

            - 0: No interpolation - probes plaed at polygon boundary points
            - -1: dField set equal to each coils' dCoil parameter

        Returns
        -------
        dField: float
            Field probe resolution.

        """
        self._check_default('dField')
        return self._dField

    @dField.setter
    def dField(self, dField):
        self._dField = dField

    def update_field(self):
        """
        Update field biot instance.

        Calculate maximum L2 norm of magnetic field around the perimiters of
        specified coils. Probe resolution specified via dField property

        Returns
        -------
        None.

        """
        self.coil.refresh_dataframe()  # flush updates
        if self.field.nT > 0:  # maximum of coil boundary values
            frame = self.field.frame
            self.coil.loc[frame.index, frame.columns] = self.field.frame

    def update_forcefield(self, subcoil=False):
        if subcoil and self.forcefield.reduce_target:
            self.forcefield.solve_interaction(reduce_target=False)

        for variable in ['Psi', 'Bx', 'Bz']:
            setattr(self.subcoil, variable,
                    getattr(self.forcefield, variable))
        self.subcoil.B = \
            np.linalg.norm([self.subcoil.Bx, self.subcoil.Bz], axis=0)
        # set coil variables to maximum of subcoil bundles

        for variable in ['Psi', 'Bx', 'Bz', 'B']:
            setattr(self.coil, variable,
                    np.maximum.reduceat(getattr(self.subcoil, variable),
                                        self.subcoil._reduction_index))


if __name__ == '__main__':

    from nova.electromagnetic.coilset import CoilSet

    cs = CoilSet()
    #polygon = shapely.geometry.Point(5, 1).buffer(0.5)
    cs.add_plasma([4.5, 5.5, 0, 1], dPlasma=0.1)
    cs.add_coil(5, -0.5, 0.75, 0.75, dCoil=0.2)

    cs.plot(True)
    '''
    cs.plasmagrid.generate_grid(expand=1, n=2e4)  # generate plasma grid
    cs.Ic = [15e6, 15e6]


    grid = Grid(coilset={'x': [1, 2, 3], 'z': [3, 2, 5]}, limit=[-1, 1, -2, 2])
    grid.plot_grid(color='C3')
    grid.plot_grid(limit=[-2, 2, 0,5])
    '''

