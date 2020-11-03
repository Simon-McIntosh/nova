
import numpy as np
from pandas import DataFrame, Series
from pandas.api.types import is_list_like
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.optimize import minimize
import shapely.geometry
from skimage import measure
import nlopt

from nova.utilities.pyplot import plt
from nova.utilities.geom import length
from nova.electromagnetic.meshgrid import MeshGrid
from nova.electromagnetic.biotsavart import BiotSet
from nova.electromagnetic.coilmatrix import CoilMatrix


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


class Grid(BiotSet):
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

    _rbs_attributes = ['Psi', 'B']

    _biot_attributes = ['n', 'n2d', 'limit', 'expand_limit', 'boundary',
                        'expand', 'nlevels', 'levels',
                        'x', 'z', '_x', '_z', 'x2d', 'z2d',
                        'target']
    # extend rbs attributes
    _biot_attributes += [f'_{rbs}_rbs' for rbs in _rbs_attributes]
    # extend rbs update flags
    _biot_attributes += [f'_update_{rbs}_rbs' for rbs in _rbs_attributes]

    _default_biot_attributes = {'n': 1e4, 'expand': 0.05, 'nlevels': 51,
                                'boundary': 'coilset'}
    _default_biot_attributes.update({f'_update_{rbs}_rbs': True
                                     for rbs in _rbs_attributes})

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
        BiotSet.__init__(self, source=subcoil, **biot_attributes)

    def _update(self, status):
        if status:
            for attribute in self._rbs_attributes:
                setattr(self, f'_update_{attribute}_rbs', True)

    @property
    def B_rbs(self):
        return self.rbs('B')

    @property
    def Psi_rbs(self):
        return self.rbs('Psi')

    def rbs(self, attribute):
        """
        Return RectBivariateSpline for attribute.

        Lazy evaluation.

        Parameters
        ----------
        attribute : str
            Attriburte label.

        Returns
        -------
        _{attribute}_rbs: RectBivariateSpline
            Grid interpolant.

        """
        self._evaluate_rbs(attribute)
        return getattr(self, f'_{attribute}_rbs')  # interpolant

    def _evaluate_rbs(self, attribute):
        update_flag = f'_update_{attribute}_rbs'
        if getattr(self, update_flag):
            # compute interpolant
            setattr(self, f'_{attribute}_rbs',
                    RectBivariateSpline(self.x, self.z,
                                        getattr(self, attribute)))
            setattr(self, update_flag, False)

    @property
    def update_rbs(self):
        """
        Return update status for rbs interpolants.

        Returns
        -------
        rbs_update_status: Series
            Update status for rbs interpolants.

        """
        return Series({attribute:
                       getattr(self, f'_update_{attribute}_rbs')
                       for attribute in self._rbs_attributes})

    def generate_biot(self):
        """
        Evaluate all biot attributes.

        Returns
        -------
        None.

        """
        if self.target.nT > 0:
            for attribute in self._rbs_attributes:
                self._evaluate_rbs(attribute)
        CoilMatrix.generate_biot(self)

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
            grid_attributes[key] = kwargs.get(key, getattr(self, key))
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
        return regenerate_grid

    def _generate_grid(self, **grid_attributes):
        self.biot_attributes = grid_attributes  # update attributes
        if self.n > 0:
            mg = MeshGrid(self.n, self.grid_boundary)  # set mesh
            self.n2d = [mg.nx, mg.nz]  # shape
            self.x, self.z = mg.x, mg.z  # axes
            # trace index interpolators
            self._x = interp1d(range(self.n2d[0]), self.x)
            self._z = interp1d(range(self.n2d[1]), self.z)
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

    def contour(self, flux, plot=False, ax=None, **kwargs):
        """
        Return flux contours.

        Parameters
        ----------
        flux : float or list[float]
            Contour levels.
        plot : bool, optional
            Plot contours. The default is False.
        ax : axes, optional
            Plot axes. The default is None.

            - None: plots to current axes

        **kwargs : dict
            Keyword arguments passed to plot.

        Returns
        -------
        contours : list[array-like, shape(n, 2)]
            Contour coordinates.

        """
        index = measure.find_contours(self.Psi, flux)
        contours = [[] for __ in range(len(index))]
        for i, idx in enumerate(index):
            contours[i] = np.array([self._x(idx[:, 0]), self._z(idx[:, 1])]).T
        if plot:
            if ax is None:
                ax = plt.gca()
            for contour in contours:
                plt.plot(contour[:, 0], contour[:, 1], **kwargs)
        return contours

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

class TopologyError(Exception):
    """Raise topology error."""


class PlasmaGrid(Grid):
    """Plasma grid interaction methods and data. Class extends Grid."""

    _plasma_attributes = ['polarity', 'Opoint', 'Opsi', 'Xpoint', 'Xpsi']

    _biot_attributes = Grid._biot_attributes + ['plasma_boundary']

    # extend biot attributes
    _biot_attributes += [f'_{attribute}' for attribute in _plasma_attributes]
    # extend rbs update flags
    _biot_attributes += [f'_update_{attribute}'
                         for attribute in _plasma_attributes]

    _default_biot_attributes = {
        **Grid._default_biot_attributes,
        **{'expand': 0.1, 'nlevels': 21, 'boundary': 'limit'}}

    def __init__(self, subcoil, **biot_attributes):
        Grid.__init__(self, subcoil, **biot_attributes)

    def _update(self, status):
        if status:
            for attribute in self._plasma_attributes:
                setattr(self, f'_update_{attribute}', True)
        Grid._update(self, status)

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
        plasma_boundary : Polygon
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
        2.5 times the plasma filament density.

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
        return 2.5*int(plasma_filament_density*grid_area)

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
        None.

        """
        kwargs['plasma_n'] = kwargs.get('plasma_n', self.plasma_n)  # auto size
        return Grid.generate_grid(self, **self._strip_plasma(**kwargs))

    def _strip_plasma(self, **kwargs):
        """
        Coerce kwargs to Grid format (extract keys with plasma_* prefix).

        Strip plasma_ prefix from kwargs and merge with base kwargs.
        Priority given to plasma_* kwargs.

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
        return {**plasma_kwargs, **base_kwargs}

    def _update_topology(self):
        a = 1
        # _topology = pd.DataFrame(columns=['x', 'z', 'Psi', 'B'])

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

    @property
    def polarity(self):
        """
        Return plasma current polarity.

        Returns
        -------
        polarity: int
            Plasma current polarity.

        """
        if self._update_polarity:
            self._polarity = self.source.coilframe.Ip_sign
            self._update_polarity = False
        return self._polarity

    def _flux_curvature(self, x):
        """
        Return principal curvatures in poloidal flux at x.

        Parameters
        ----------
        x : array-like, shape(2,)
            Polidal coordinates for curvature calculation.

        Returns
        -------
        Pmax : float
            Maximum principal curvature.
        Pmin : float
            Minimum principal curvature.

        """
        # flux derivatives
        Px = self.Psi_rbs.ev(*x, dx=1)
        Pz = self.Psi_rbs.ev(*x, dy=1)
        Pxx = self.Psi_rbs.ev(*x, dx=2)
        Pzz = self.Psi_rbs.ev(*x, dy=2)
        Pxz = self.Psi_rbs.ev(*x, dx=1, dy=1)
        # mean curvature
        H = (Px**2 + 1)*Pzz - 2*Px*Pz*Pxz + (Pz**2 + 1)*Pxx
        H = -H/(2*(Px**2 + Pz**2 + 1)**(1.5))
        # gaussian curvature
        K = (Pxx*Pzz - Pxz**2) / (1 + Px**2 + Pz**2)**2
        # principal curvatures
        Pmax = H + np.sqrt(H**2 - K)
        Pmin = H - np.sqrt(H**2 - K)
        return Pmax, Pmin

    def null_type(self, x):
        """
        Return feild null type.

        Parameters
        ----------
        x : array-like, shape(2,)
            Coordinates of null-point (x, z).

        Raises
        ------
        TopologyError
            One or both princaple flux curvatures equal to zero
            (plane or cylindrical surface).

        Returns
        -------
        null_type : str

            - X : X-point
            - O : O-point.

        """
        Pmax, Pmin = self._flux_curvature(x)
        if np.isclose(Pmax, 0) or np.isclose(Pmin, 0):
            raise TopologyError('Field null froms cylinder or plane surface')
        elif np.sign(Pmax) == np.sign(Pmin):
            return 'O'
        else:
            return 'X'

    def _signed_flux(self, x):
        return -1 * self.polarity * self.Psi_rbs.ev(*x)

    def _signed_flux_gradient(self, x):
        return -1 * self.polarity * np.array([self.Psi_rbs.ev(*x, dx=1),
                                              self.Psi_rbs.ev(*x, dy=1)])

    def get_Opoint(self, xo=None):
        """
        Return coordinates of plasma O-point.

        O-point defined as center of nested flux surfaces.

        Parameters
        ----------
        xo : array-like(float), shape(2,), optional
            Sead coordinates (x, z). The default is None.

            - None: xo set to grid center

        Raises
        ------
        TopologyError
            Failed to find signed flux minimum.

        Returns
        -------
        Opoint, array-like(float), shape(2,)
            Coordinates of O-point.

        """
        if xo is None:
            xo = self.bounds.mean(axis=1)
        res = minimize(self._signed_flux, xo,
                       jac=self._signed_flux_gradient, bounds=self.bounds)
        if not res.success:
            raise TopologyError('Opoint signed flux minimization failure\n\n'
                                f'{res}.')
        return res.x

    @property
    def Opoint(self):
        """
        Return coordinates of center of nested flux surfaces.

        Returns
        -------
        tuple
            Plasma O-point coordinates (x, z).

        """
        if self._update_Opoint or self._Opoint is None:
            self._Opoint = self.get_Opoint(xo=self._Opoint)
            self._update_Opoint = False
        return self._Opoint

    @property
    def Opsi(self):
        """
        Return poloidal flux calculated at O-point.

        Returns
        -------
        Opsi: float
            O-point poloidal flux.

        """
        if self._update_Opsi:
            self._Opsi = float(self.Psi_rbs.ev(*self.Opoint))
            self._update_Opsi = False
        return self._Opsi

    def _field_null(self, x, grad):
        print(x)
        if grad.size > 0:
            grad[:] = self._field_gradient(x)
        return self.B_rbs.ev(*x).item()

    def _field_gradient(self, x):
        return np.array([self.B_rbs.ev(*x, dx=1), self.B_rbs.ev(*x, dy=1)])

    def get_Xpoint(self, xo):
        """
        Return X-point coordinates.

        Resolve X-point location based on solution of field minimum in
        proximity to sead location, *xo*.

        Parameters
        ----------
        xo : array-like(float), shape(2,)
            Sead coordinates (x, z).

        Raises
        ------
        TopologyError
            Field minimization failure.

        Returns
        -------
        Xpoint: array-like(float), shape(2,)
            X-point coordinates (x, z).

        """

        opt = nlopt.opt(nlopt.G_MLSL_LDS, 2)
        local = nlopt.opt(nlopt.LD_MMA, 2)
        '''
        local.set_ftol_rel(1e-4)
        local.set_min_objective(self._field_null)
        local.set_lower_bounds([4, -4])
        local.set_upper_bounds([8, 4])
        '''

        opt.set_local_optimizer(local)
        opt.set_min_objective(self._field_null)
        opt.set_ftol_rel(1e-4)
        opt.set_maxeval(50)
        # grid limits
        opt.set_lower_bounds([4, -4])
        opt.set_upper_bounds([8, 4])

        opt.set_population(2)

        x = opt.optimize(xo)

        print(opt)

        #print(self.grid_boundary[1::2])
        #print(x)

        '''
        res = minimize(self._field_null, xo,
                       jac=self._field_gradient,
                       #bounds=self.bounds,
                       )
        if not res.success:
            raise TopologyError('Xpoint signed |B| minimization failure\n\n'
                                f'{res}.')
        '''
        return opt

    @property
    def Xpoint(self):
        """
        Manage Xpoint locations.

        Parameters
        ----------
        xo : array-like, shape(n, 2)
            Sead Xpoints.

        Returns
        -------
        Xpoint: ndarray, shape(2)
            Coordinates of primary Xpoint (x, z).

        """
        if self._update_Xpoint or self._Xpoint is None:
            if self._Xpoint is None:  # sead with boundary midsides
                bounds = self.bounds
                self.Xpoint = [[np.mean(bounds[0]), bounds[1][i]]
                               for i in range(2)]
            nX = len(self._Xpoint)
            _Xpoint = np.zeros((nX, 2))
            _Xpsi = np.zeros(nX)
            for i in range(nX):
                _Xpoint[i] = self.get_Xpoint(self._Xpoint[i])
                _Xpsi[i] = self.Psi_rbs.ev(*_Xpoint[i])
            self._Xpoint = _Xpoint[np.argsort(_Xpsi)]
            if self.source.coilframe.Ip_sign > 0:
                self._Xpoint = self._Xpoint[::-1]
            self._update_Xpoint = False
        return self._Xpoint[0]

    @Xpoint.setter
    def Xpoint(self, xo):
        if not isinstance(xo, np.ndarray):
            xo = np.array(xo)
        if xo.ndim == 1:
            xo = xo.reshape(1, -1)
        if xo.shape[1] != 2:
            raise IndexError(f'shape(xo) {xo.shape} not (n, 2)')
        self._Xpoint = xo


class BiotMethods:
    """Manage biot methods for CoilSet."""

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
        self.biot_instances = ['field', 'forcefield', 'grid']  # base set

    @property
    def biot_instances(self):
        """
        Initialize biot methods.

        Maintain dictionary of initialized biot methods

        Parameters
        ----------
        biot_instances : str or list or dict
            Collection of biot instances to initialize.

            - str or list: biot name and method assumed equal
            - dict : option to set diffrent values for biot name and method

            Possible to pass biot_attributes as
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

    def generate_biot(self):
        """
        Generate for all biot instances.

        Returns
        -------
        None.

        """
        for instance in self._biot_instances:
            print(instance)
            getattr(self, instance).generate_biot()

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

    #data = SimulationData()
    #data.generate_grid()

    grid = Grid(coilset={'x': [1, 2, 3], 'z': [3, 2, 5]}, limit=[-1, 1, -2, 2])
    grid.plot_grid(color='C3')
    grid.plot_grid(limit=[-2, 2, 0,5])

