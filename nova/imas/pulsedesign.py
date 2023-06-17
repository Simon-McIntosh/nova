"""Generate feed-forward coil current waveforms from pulse schedule IDS."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np
from scipy.optimize import minimize, newton_krylov, LinearConstraint
from tqdm import tqdm
import xarray

from nova.biot.biot import Nbiot
from nova.biot.plot import Plot1D
from nova.graphics.plot import Animate, Plot

from nova.geometry.plasmapoints import PlasmaPoints

from nova.imas.database import Database, IDS, Ids, IdsEntry
from nova.imas.equilibrium import EquilibriumData
from nova.imas.machine import Machine
from nova.imas.metadata import Metadata
from nova.imas.profiles import Profile
from nova.imas.pf_active import PF_Active
from nova.linalg.regression import MoorePenrose


@dataclass
class ConstraintData:
    """Manage masked constraint data."""

    point_number: int
    array: np.ndarray = field(init=False)
    mask: np.ndarray = field(init=False)

    def __post_init__(self):
        """Initialize data and mask arrays."""
        self.array = np.zeros(self.point_number, float)
        self.mask = np.ones(self.point_number, bool)

    def __len__(self):
        """Return constraint number."""
        return np.sum(~self.mask)

    def update(self, data, index=None):
        """Update constraint."""
        if index is None:
            index = self.point_index
        self.array[index] = data
        self.mask[index] = False

    @cached_property
    def point_index(self):
        """Return full point index."""
        return np.arange(self.point_number)

    @property
    def index(self):
        """Return select point index."""
        return self.point_index[~self.mask]

    @property
    def data(self):
        """Return select data."""
        return self.array[~self.mask]


@dataclass
class Constraint(Plot):
    """Manage flux and field constraints."""

    points: np.ndarray = field(default_factory=lambda: np.array([]))
    constraint: dict[str, ConstraintData] = field(init=False, default_factory=dict)

    attrs: ClassVar[list[str]] = ["psi", "br", "bz"]

    def __post_init__(self):
        """Initialize constraint data."""
        for attr in self.attrs:
            self.constraint[attr] = ConstraintData(self.point_number)

    def __len__(self):
        """Return contstraint number."""
        return np.sum([len(self[attr]) for attr in self.attrs])

    @cached_property
    def point_number(self):
        """Return point number."""
        return len(self.points)

    @cached_property
    def point_index(self):
        """Return full point index."""
        return np.arange(self.point_number)

    def __getitem__(self, attr: str):
        """Return constraint data."""
        return self.constraint[attr]

    def index(self, attr: str):
        """Return constraint point index."""
        if attr == "null":
            return np.intersect1d(
                self["br"].index[self["br"].data == 0],
                self["bz"].index[self["bz"].data == 0],
                assume_unique=True,
            )
        if attr == "radial":
            return np.intersect1d(
                self["br"].index[self["br"].data == 0],
                self.point_index[self["bz"].mask],
                assume_unique=True,
            )
        if attr == "vertical":
            return np.intersect1d(
                self["bz"].index[self["bz"].data == 0],
                self.point_index[self["br"].mask],
                assume_unique=True,
            )
        return self[attr].index

    def _points(self, attr: str):
        """Return constraint points."""
        return self.points[self.index(attr)]

    def update(self, attr: str, constraint):
        """Update constraint."""
        match constraint:
            case (value, index):
                self[attr].update(value, index)
            case value:
                self[attr].update(value)

    @property
    def poloidal_flux(self):
        """Return poloidal flux constraints."""
        return self["psi"].data

    @poloidal_flux.setter
    def poloidal_flux(self, constraint):
        """Set poloidal flux constraint."""
        self.update("psi", constraint)

    @property
    def radial_field(self):
        """Return radial_field constraints."""
        return self["br"].data

    @radial_field.setter
    def radial_field(self, constraint):
        """Set radial field constraint."""
        self.update("br", constraint)

    @property
    def vertical_field(self):
        """Return vertical_field constraints."""
        return self["bz"].data

    @vertical_field.setter
    def vertical_field(self, constraint):
        """Set vertical field constraint."""
        self.update("bz", constraint)

    def plot(self, axes=None, ms=10, color="C2"):
        """Plot constraint."""
        if self.point_number == 0:
            return
        self.axes = axes
        self.axes.plot(*self._points("psi").T, "s", ms=ms, mec=color, mew=2, mfc="none")
        self.axes.plot(*self._points("radial").T, "|", ms=2 * ms, mec=color)
        self.axes.plot(*self._points("vertical").T, "_", ms=2 * ms, mec=color)
        self.axes.plot(*self._points("null").T, "x", ms=2 * ms, mec=color)


@dataclass
class Control(PlasmaPoints, Profile):
    """Extract control points and flux profiles from equilibrium data."""

    constraint: Constraint = field(init=False, default_factory=Constraint, repr=False)

    def update_constraints(self, psi=0):
        """Update flux and field constraints."""
        self.constraint = Constraint(self.control_points)
        self.constraint.poloidal_flux = psi, range(4)
        if self.square:
            self.constraint.poloidal_flux = psi, range(4, 8)
        if self.strike and not self.limiter:
            self.constraint.poloidal_flux = (
                psi,
                self.constraint.point_number + np.array([-2, -1]),
            )
        self.constraint.radial_field = 0, [0, 2]
        self.constraint.vertical_field = 0, [1, 3]
        if not self.limiter:
            self.constraint.radial_field = 0, [3]

    def update(self):
        """Update source equilibrium."""
        super().update()
        self.update_constraints()

    def plot(self, index=None, axes=None, **kwargs):
        """Extend PlasmaPoints.plot to include constraints."""
        super().plot(index, axes, **kwargs)
        self.constraint.plot()


@dataclass
class ITER(Machine):
    """ITER machine description.

    Extend `Machine` class with ITER specific defaults.

    pf_active: Ids | bool | str, optional
        Machine description ids for the `pf_active coil geometry,
        turns and limits. The default is 'iter_md'.

    pf_passive: Ids | bool | str, optional
        Machine description ids for axisymetric passive structure. The default
        is False.

    wall: Ids | bool | str, optional
        Machine description ids for first wall's poloidal contour. The default
        is 'iter_md'.

    tplasma: {'hex', 'rect'}
        Plasma filament geometry. The default is 'hex'.

    dplasma: int | float, optional
        Plasma filament resolution. The default is -3000
            - dplasma < 0: aproximate filament number ~ -int(dplasma)
            - dplasma > 0: aproximate filament linear dimension

    See :func:`nova.imas.machine.Geometry.get_ids_attrs` for usage details for
    the pf_active, pf_passive, and wall attributes.

    """

    pf_active: Ids | bool | str = field(default="iter_md", repr=False)
    pf_passive: Ids | bool | str = field(default=False, repr=False)
    wall: Ids | bool | str = field(default="iter_md", repr=False)
    tplasma: str = "hex"
    dplasma: int | float = -3000

    def __post_init__(self):
        """Disable vs3 current updates."""
        super().__post_init__()
        self.saloc["free"][-2] = False  # TODO implement nturn_min filter


@dataclass
class PulseDesign(Animate, Plot1D, Control, ITER):
    """Generate Pulse Design Simulator current waveforms.

    Transform a prototype pulse design from a four-point bounding-box plasma
    with boundary psi and profile information to self-consistent set of
    external coil current waveforms.

    Parameters
    ----------
    ids: ImasIds
        Source equilibrium IDS. This source `ids` may also be referanced via a
        series of attributes to read data from an IMAS database stored to file.
        This `ids` must have a homogeneous timebase.
        The following parameters must be pressent in this `ids`:

        - time_slice(:).boundary_separatrix
            - type
            - psi
            - geometric_axis
            - minor_radius
            - elongation
            - elongation_upper (a proxy for triangulation_outer)  # IMAS-4682
            - elongation_lower (a proxy for triangulation_inner)  # IMAS-4682
            - triangularity_upper
            - triangularity_lower
        - time_slice(:).global_quantities
            - ip
        - time_slice(:).profiles_1d
            - dpressure_dpsi
            - f_df_dpsi

        The eight geometry parameters defined the boundary_separatrix node
        prescribe the positions of four points in the poloidal plane
        where the plasma touches its r,z alligned bounding box.

        Future versions of this code will replace the psi attribute with
        definitions of Cejima and Li.

        Future versions of this code will replace the profiles_1d attributes
        with definitinos of Li and Beta.

    pf_active: Ids | bool | str, optional
        Machine description ids for the `pf_active coil geometry,
        turns and limits. The default is 'iter_md'.

    pf_passive: Ids | bool | str, optional
        Machine description ids for axisymetric passive structure. The default
        is False.

    wall: Ids | bool | str, optional
        Machine description ids for first wall's poloidal contour. The default
        is 'iter_md'.

    tplasma: {'hex', 'rect'}
        Plasma filament geometry. The default is 'hex'.

    dplasma: int | float, optional
        Plasma filament resolution
            - dplasma < 0: aproximate filament number ~ -int(dplasma)
            - dplasma > 0: aproximate filament linear dimension

    nwall: Nbiot, optional
        Plasma wall subpanel resolution. The default is 3.

    nlevelset: Nbiot, optional
        Levelset resoultion for contouring and control point location.
        The default is 3000.

    ninductance: Nbiot, optional
        Coil target subgrid resolution for self and mutual inductance
        calculations. The default is None.
        Inductance calculations are forseen in a future version. This attribute
        will be used to calculate and minimize the machines `flux-state`'.

    nforce: Nbiot, optional
        Coil target subgrid resoultion for force calculations. The default is
        None.
        Force calculations are forseen in a future version. The coil force
        vector and associated Jacobian will be used to constrain scenarios
        such that the coil current waveforms keep the machines operating within
        its force limits.

    nfield: Nbiot, optional
        Coil boundary subgrid resoultion for L2 norm maximum field
        calculations. The default is None.
        Maximum on-coil field calculations are forseen in a future version.
        The coil field vector and associated Jacobian will be used to constrain
        scenarios such that the coil current waveforms keep the maximum on-coil
        fields below their respective limits.

    gamma: float, optional
        Tikhonov regularization factor used by Moore Penrose inversion. This
        factor is multiplied by the absolute value of plasma current before
        being applied to the rectangular diagnal matrix of the Singular Value
        Decomposition. The default value is 1e-12

    field_weight: float | int, optional
        Weighting factor for all field constraints. The default value is 50.

    Raises
    ------
    ValueError
        Source `ids` has non-homogeneus time.
        # TODO

    AttributeError
        Attributes missing from source equilibrium `ids`.
        # TODO

    ValueError
        Bounding-box control points lie outside of first-wall contour.
        # TODO

    See Also
    --------
    `nova.imas.database.IDS` :
        For a list of attributes and their definitons to be used in place of
        the `ids` keyword when referancing the source equilibrium `ids` that
        should be read from file.
    `nova.imas.machine.Geometry.get_ids_attrs` :
        For complete usage details for the pf_active, pf_passive,
        and wall attributes.

    Notes
    -----
    The class may be run in one of three modes:

        - As an python IMAS **actor**, accepts and returns IDS(s)
        - As an python IMAS **code**, reads and writes IDS(s)
        - As a command line **script** see `pulsedesign --help` for details

    # TODO: provide details for numerical method.

    There is currently no way to store the inner and outer triangularities in
    the IMAS data dictionary. Thes attributes are required by this class to
    locate the heights of the inner and outer separatrix raidal turning points.
    A workaround is implemented here whereby the redundant upper and lower
    elongations are used to pass the outer and inner triangularities.
    This fix will remain in-place until the JIRA ticket IMAS-4682 is fix on
    the master IMAS-AL branch.

    Examples
    --------
    A pulse design workflow will typicaly include a dedicated tool for the
    creation and modification of the source equilibrium `ids`
    used by this class. A dummy `ids` is created here using the
    :func:`imas.database.IdsEntry` class to provide a concrete usage example
    for the `PulseDesign` class.

    import IdsEntry and instantiate as an equilibrium `ids`.

    >>> import pytest
    >>> from nova.imas.database import IdsEntry, IMAS_MODULE_NOT_FOUND
    >>> if IMAS_MODULE_NOT_FOUND:
    ...     pytest.skip('imas module not found')
    >>> ids_entry = IdsEntry(name='equilibrium')

    Define time vector, size time_slice, and define homogeneous_time.

    >>> time = [1.5, 19, 110, 600, 670]
    >>> ids_entry.ids_data.time = time
    >>> ids_entry.ids_data.time_slice.resize(len(time))
    >>> ids_entry.ids_data.ids_properties.homogeneous_time = 1

    Populate boundary_separatrix node. Low precision used here for readability.

    >>> with ids_entry.node('time_slice:boundary_separatrix.*'):
    ...     ids_entry['type', :] = [0, 1, 1, 1, 1]
    ...     ids_entry['psi', :] = [107.8,  73.5,  17.4, -13.7,  -7.5]
    ...     ids_entry['minor_radius', :] = [1.7, 2. , 2. , 2. , 1.9]
    ...     ids_entry['elongation', :] = [1.1, 1.8, 1.8, 1.9, 1.1]
    ...     ids_entry['elongation_upper', :] = [0. , 0.2, 0.1, 0.1, 0.1]
    ...     ids_entry['elongation_lower', :] = [0. ,  0.3,  0.2,  0.3,  0.3]
    ...     ids_entry['triangularity_upper', :] = [0. ,  0.3,  0.4,  0.5,  0.3]
    ...     ids_entry['triangularity_lower', :] = [0.1, 0.6, 0.5, 0.6, 0.6]
    >>> with ids_entry.node('time_slice:boundary_separatrix.geometric_axis.*'):
    ...     ids_entry['r', :] = [5.8, 6.2, 6.2, 6.2, 6.1]
    ...     ids_entry['z', :] = [ 0. ,  0.1,  0.3,  0.3, -1. ]

    Update plasma current.

    >>> import numpy as np
    >>> with ids_entry.node('time_slice:global_quantities.*'):
    ...     ids_entry['ip', :] = 1e6 * np.array([-0.4, -5.1, -15, -15, -1.5])

    Populate profiles_1d node.

    >>> with ids_entry.node('time_slice:profiles_1d.*'):
    ...     ids_entry['dpressure_dpsi', :] = 1e3 * np.array(
    ...         [[ 0.2,  0.2,  0.2,  0.1,  0.1],
    ...          [ 0. ,  0.7,  0.5,  0.4,  0.3],
    ...          [ 0.4,  6.4,  5.7,  5.6,  5.7],
    ...          [ 0.4,  7.2,  6.9,  6.5,  6.2],
    ...          [-0. ,  0.3,  0.3,  0.2,  0.1]])
    ...     ids_entry['f_df_dpsi', :] = np.array(
    ...         [[ 0. ,  0.1,  0.1,  0.1,  0. ],
    ...          [ 1.4,  0.4,  0.3,  0.2,  0.2],
    ...          [ 2. ,  1.5,  0.7,  0.3,  0. ],
    ...          [ 2. ,  1. ,  0.6,  0.3,  0.2],
    ...          [ 1.7,  0.6,  0.1, -0. , -0.1]])

    Instantiate PulseDesign using source equilibrium ids.

    >>> design = PulseDesign(ids=ids_entry.ids_data)

    Set time instance to update constraints and solve external currents.

    >>> design.itime = 0

    Plot solution at first time slice.

    >>> design.plot('plasma')  # doctest: +SKIP

    Plot coil current waveform.

    >>> design.plot_waveform()  # doctest: +SKIP

    Extract pf_active ids for all times present in source equilibrium `ids`.

    >>> pf_active = design.pf_active_ids

    Extract equilibrium ids for all times present in source equilibrium `ids`.
    Notice that the PDS computation trigered by the pf_active_ids property is
    not repeated and the equilibrium ids is constructed from a cached datased.

    >>> equilibrium = design.equilibrium_ids

    """

    nwall: Nbiot = 3
    nlevelset: Nbiot = 6000
    ninductance: Nbiot = None
    nforce: Nbiot = 15
    nfield: Nbiot = 100
    gamma: float = 1e-12
    field_weight: float | int = 50
    name: str = "equilibrium"

    def update_constraints(self):
        """Extend ControlPoint.update_constraints to include boundary psi."""
        super().update_constraints(-self["psi_boundary"])  # COCOS11

    def update(self):
        """Extend itime update."""
        super().update()
        self.sloc["plasma", "Ic"] = self["ip"]
        self.solve()

    def _constrain(self, constraint):
        """Return coupling matrix and vectors."""
        if len(constraint) == 0:
            return
        point_index = np.array(
            [self.levelset.kd_query(point) for point in constraint.points]
        )
        _matrix, _vector = [], []
        for attr in constraint.attrs:
            if len(constraint[attr]) == 0:
                continue
            index = point_index[constraint[attr].index]
            matrix = getattr(self.levelset, attr.capitalize())[index]
            vector = (
                constraint[attr].data
                - matrix[:, self.plasma_index] * self.saloc["plasma", "Ic"]
            )
            if attr != "psi":
                matrix *= np.sqrt(self.field_weight)
                vector *= np.sqrt(self.field_weight)
            _matrix.append(matrix)
            _vector.append(vector)
        matrix = np.vstack(_matrix)
        vector = np.hstack(_vector)
        return matrix[:, self.saloc["free"]], vector

    def _stack(self, *args):
        """Stack coupling matrix and vectors."""
        matrix = np.vstack([arg[0] for arg in args if arg is not None])
        data = np.hstack([arg[1] for arg in args if arg is not None])
        return matrix, data

    def solve_current(self):
        """Solve coil currents given flux and field targets."""
        coupling = [self._constrain(self.constraint)]
        matrix, vector = self._stack(*coupling)
        gamma = self.gamma * abs(self["ip"])
        self.saloc["free", "Ic"] = MoorePenrose(matrix, gamma=gamma) / vector
        """
        bounds = [(self.frame.loc[index, 'Imin'],
                   self.frame.loc[index, 'Imax'])
                  for index in self.sloc().index[self.saloc['free']]]
        res = minimize(self.fun, self.saloc['free', 'Ic'],
                       args=(matrix, vector), bounds=bounds)
        self.saloc['free', 'Ic'] = res.x
        """

    def fun(self, xin, matrix, vector):
        """Return optimization goal."""
        return np.linalg.norm(matrix @ xin - vector)

    def hess(self, x):
        """Return Hessian for a linear operator."""
        return np.zeros((len(x), len(x)))

    def optimize_current(self):
        """Optimize external coil currents."""
        coupling = [self._constrain(self.constraint)]
        matrix, vector = self._stack(*coupling)
        fmatrix, fvector = self._constrain(self.field)
        self.solve_current()
        constraints = [
            LinearConstraint(matrix, vector, vector),
            LinearConstraint(fmatrix, fvector, fvector),
        ]
        sol = minimize(
            self.fun,
            self.saloc["free", "Ic"],
            hess=self.hess,
            method="trust-constr",
            constraints=constraints,
        )
        self.saloc["free", "Ic"] = sol.x

    @property
    def psi_boundary(self):
        """Return boundary psi."""
        if self.limiter:
            return self.plasma.psi_w
        return self.plasma.psi_x

    def residual(self, xin):
        """Return psi grid residual."""
        self.plasma.nturn = xin[:-1]
        self.solve_current()
        self.plasma.separatrix = xin[-1]
        xout = np.r_[self.plasma.nturn, np.sum(self.plasma.nturn)]
        residual = xout - np.r_[xin[:-1], 1]
        residual[-1] /= self.plasmagrid.number
        return residual

    def psi_residual(self, psi):
        """Return psi residual."""
        self.plasma.psi = psi
        with self.plasma.profile(self.p_prime, self.ff_prime):
            self.plasma.separatrix = self.plasma.psi_boundary
        self.solve_current()
        return np.r_[self.plasmagrid.psi, self.plasmawall.psi] - psi

    def _solve(self, verbose=True):
        """Solve waveform with Newton Krylov scheame."""
        self.solve_current()
        psi = np.r_[self.plasmagrid.psi, self.plasmawall.psi]
        psi = newton_krylov(self.psi_residual, self.plasma.psi, verbose=verbose, iter=5)
        self.psi_residual(psi)

    def solve(self, verbose=False):
        """Solve waveform using basic Picard itteration."""
        self.plasma.separatrix = {
            "ellipse": np.r_[
                self["geometric_axis"],
                2 * self["minor_radius"] * np.array([1, self["elongation"]]),
            ]
        }
        for _ in range(3):
            self.solve_current()
            with self.plasma.profile(self.p_prime, self.ff_prime):
                self.plasma.separatrix = self.plasma.psi_boundary
        self.solve_current()

    def plot(self, index=None, axes=None, **kwargs):
        """Extend plot to include plasma contours."""
        super().plot(index, axes, **kwargs)
        self.plasma.plot()

    def solve_waveform(self, verbose=False):
        """Solve current waveform."""
        current = np.zeros((self.data.dims["time"], np.sum(self.saloc["free"])))

        for itime in tqdm(
            self.data.itime.data[:-1], "solving current waveform", disable=~verbose
        ):
            self.itime = itime
            self.solve(verbose=False)
            current[itime] = self.saloc["free", "Ic"]
        return current

    def update_metadata(self, ids_entry: IdsEntry):
        """Update ids with instance metadata."""
        metadata = Metadata(ids_entry.ids_data)
        comment = (
            "Coil current waveforms to match 4-point bounding-box "
            "separatrix targets."
        )
        provenance = [self.uri]
        provenance.extend(
            [
                IDS(*value.split(",")).uri
                if not isinstance(value, (int, np.integer))
                else f"imas:ids?name={attr[:-3]};hash={value}"
                for attr, value in self.data.attrs.items()
                if attr[-3:] == "_md"
            ]
        )
        metadata.put_properties(comment, homogeneous_time=1, provenance=provenance)
        code_parameters = self.data.attrs
        code_parameters |= {
            attr: getattr(self, attr) for attr in ["gamma", "field_weight"]
        }
        metadata.put_code(code_parameters)

    @cached_property
    def _data(self) -> xarray.Dataset:
        """Return waveform dataset."""
        attrs_0d = [
            "li_3",
            "psi_axis",
            "psi_boundary",
            "minor_radius",
            "elongation",
            "triangularity",
            "triangularity_upper",
            "triangularity_lower",
            "triangularity_inner",
            "triangularity_outer",
            "squareness_upper_inner",
            "squareness_upper_outer",
            "squareness_lower_inner",
            "squareness_lower_outer",
        ]

        data = xarray.Dataset()
        data["time"] = self.data.time
        data["point"] = ["r", "z"]
        data["r"] = self.levelset.data.x.data
        data["z"] = self.levelset.data.z.data
        data["r2d"] = ("r", "z"), self.levelset.data.x2d.data
        data["z2d"] = ("r", "z"), self.levelset.data.z2d.data
        data["boundary_index"] = np.arange(500)
        data["strike_point_index"] = np.arange(2)
        data["coil_name"] = self.coil_name
        data["field_coil_name"] = self.field.coil_name

        data["current"] = xarray.DataArray(
            0.0, coords=[data.time, data.coil_name], dims=["time", "coil_name"]
        )
        data["vertical_force"] = xarray.DataArray(
            0.0, coords=[data.time, data.coil_name], dims=["time", "coil_name"]
        )
        data["field"] = xarray.DataArray(
            0.0,
            coords=[data.time, data.field_coil_name],
            dims=["time", "field_coil_name"],
        )
        data["boundary"] = xarray.DataArray(
            0.0,
            coords=[data.time, data.boundary_index, data.point],
            dims=["time", "boundary_index", "point"],
        )
        for attr in attrs_0d:
            data[attr] = xarray.DataArray(0.0, coords=[data.time], dims=["time"])
        for axis in ["magnetic_axis", "geometric_axis"]:
            data[axis] = xarray.DataArray(
                0.0, coords=[data.time, data.point], dims=["time", "point"]
            )
        for attr in ["boundary_type", "x_point_number", "strike_point_number"]:
            data[attr] = xarray.DataArray(0, coords=[data.time], dims=["time"])
        data["x_point"] = xarray.DataArray(
            0.0, coords=[data.time, data.point], dims=["time", "point"]
        )
        data["strike_point"] = xarray.DataArray(
            0.0,
            coords=[data.time, data.strike_point_index, data.point],
            dims=["time", "strike_point_index", "point"],
        )
        length = np.linspace(0, 1, data.dims["boundary_index"])
        for itime in tqdm(self.data.itime.data, "Solving PDS waveform"):
            self.itime = itime
            data["current"][itime] = self.current
            data["vertical_force"][itime] = self.force.fz
            data["field"][itime] = self.field.bp
            data["boundary"][itime] = self.plasma.boundary(length)
            for attr in attrs_0d:
                data[attr][itime] = getattr(self.plasma, attr)
            data["magnetic_axis"][itime] = self.plasma.magnetic_axis
            data["boundary_type"][itime] = int(not self.plasma.limiter)
            data["geometric_axis"][itime] = self.plasma.geometric_axis
            data["x_point_number"][itime] = int(not self.plasma.limiter)
            if not self.plasma.limiter:
                data["x_point"][itime] = self.plasma.x_point

            strike_points = self.plasma.strike_points
            data["strike_point_number"][itime] = len(strike_points)
            if (n_points := len(strike_points)) > 0:
                data["strike_point"][itime, :n_points] = strike_points
        data.attrs["attrs_0d"] = attrs_0d
        return data

    @cached_property
    def pf_active_ids(self) -> Ids:
        """Return waveform pf_active ids."""
        pf_active_md = Database(**self.pf_active)  # type: ignore
        ids_entry = IdsEntry(ids_data=pf_active_md.ids_data, name="pf_active")
        self.update_metadata(ids_entry)
        ids_entry.ids_data.time = self._data.time.data
        with ids_entry.node("coil:*.data"):
            ids_entry["current", :] = self._data["current"].data.T
        return ids_entry.ids_data

    @cached_property
    def equilibrium_ids(self) -> Ids:
        """Return waveform equilibrium ids."""
        ids_entry = IdsEntry(ids_data=self.ids_data, name="equilibrium")
        self.update_metadata(ids_entry)
        ids_entry.ids_data.time = self._data.time.data
        with ids_entry.node("time_slice:global_quantities.*"):
            for attr in ["li_3", "psi_axis", "psi_boundary"]:
                data = self._data[attr].data
                if "psi" in attr:
                    data *= -1  # COCOS
                ids_entry[attr, :] = data
        with ids_entry.node("time_slice:global_quantities.magnetic_axis*"):
            for i, attr in enumerate("rz"):
                ids_entry[attr, :] = self._data.magnetic_axis.data[:, i]
        with ids_entry.node("time_slice:boundary_separatrix.*"):
            for attr in self._data.attrs_0d:
                ids_entry[attr, :] = self._data[attr].data
            ids_entry["type", :] = self._data["boundary_type"].data
            ids_entry["psi", :] = -self._data["psi_boundary"].data  # COCOS
        with ids_entry.node("time_slice:boundary_separatrix.outline.*"):
            for i, attr in enumerate("rz"):
                ids_entry[attr, :] = self._data["boundary"].data[..., i]
        with ids_entry.node("time_slice:boundary_separatrix.geometric_axis.*"):
            for i, attr in enumerate("rz"):
                ids_entry[attr, :] = self._data["geometric_axis"].data[:, i]
        for itime in range(self.data.dims["time"]):
            boundary = ids_entry.ids_data.time_slice[itime].boundary_separatrix
            # boundary x_point
            if self._data.x_point_number[itime].data > 0:
                x_point = self._data.x_point[itime].data
                boundary.x_point.resize(1)
                boundary.x_point[0].r = x_point[0]
                boundary.x_point[0].z = x_point[1]
            # divertor strike points
            if (number := self._data.strike_point_number[itime].data) > 0:
                strike_point = self._data.strike_point[itime].data
                boundary.strike_point.resize(number)
                for point in range(number):
                    boundary.strike_point[point].r = strike_point[point, 0]
                    boundary.strike_point[point].z = strike_point[point, 1]
            # profiles 2D
            profiles_2d = ids_entry.ids_data.time_slice[itime].profiles_2d
            profiles_2d.resize(1)
            profiles_2d[0].type.name = "total"
            profiles_2d[0].type.index = 0
            profiles_2d[0].type.name = "total field and flux"
            profiles_2d[0].grid_type.name = "rectangular"
            profiles_2d[0].grid_type.index = 1
            profiles_2d[0].grid_type.description = "cylindrical grid"
            profiles_2d[0].grid.dim1 = self._data.r.data
            profiles_2d[0].grid.dim2 = self._data.z.data
            profiles_2d[0].r = self._data.r2d.data
            profiles_2d[0].z = self._data.z2d.data
            profiles_2d[0].psi = self.levelset.psi_
            # only write field for high order plasma elements
            if self.tplasma == "rectangle":
                profiles_2d[0].b_field_r = self.levelset.br_
                profiles_2d[0].b_field_z = self.levelset.bz_

        return ids_entry.ids_data

    def plot_waveform(self):
        """Extend plot_waveform to compare with benchmark."""
        self.set_axes("1d")
        for i, name in enumerate(self._data.coil_name.data[:-2]):
            self.axes.plot(self.data.time, 1e-3 * self._data.current[:, i], label=name)
        self.axes.set_ylabel("coil current, kA")
        self.axes.set_xlabel("time, s")
        self.axes.legend()

    def make_frame(self, time):
        """Make animation frame."""
        self.reset_data()
        scene = self.scene(time)
        self.time = scene.pop("time", self.time)
        if len(scene) > 0:
            for attr, value in scene.items():
                self[attr] = value
                self.fit()
            self.update()
        if isinstance(self.axes, list) and len(self.axes) == 4:
            axes = self.axes
            self.bar(
                "current",
                "free",
                scale=1e-3,
                limit=[-45, 45],
                label=r"$I$ kA",
                color="C0",
                axes=self.axes[0],
            )
            self.axes = axes
            self.force.bar(
                "fz",
                "free",
                scale=1e-6,
                limit=[-300, 300],
                label=r"$F_z$ MN",
                color="C1",
                axes=self.axes[1],
            )
            self.force.axes.get_xaxis().set_visible(False)

            self.field.bar(
                "bp",
                "free",
                scale=1,
                limit=[0, 15],
                label=r"$|\mathbf{B_p}|$ T",
                color="C2",
                axes=self.axes[2],
            )
            axes = self.axes
            self.plot("plasma", axes=self.axes[3])
            self.axes = axes

        else:
            self.plot("plasma")
            # self.force.plot(norm=6e8)


@dataclass
class AnimateDesign(PulseDesign):
    """Extend pulse design to include control-point annimation."""

    sequence: tuple[str | tuple[int]] = (
        (2, 0),
        "triangularity",
        (6, 50),
        "box",
        (6, -50),
    )

    def __post_init__(self):
        """Build animation sequence."""
        super().__post_init__()
        for sequence in self.sequence:
            match sequence:
                case str():
                    getattr(self, f"animate_{sequence}")()
                case duration, ramp:
                    self.add_animation("time", duration, ramp=ramp)
                case _:
                    raise IndexError(f"invalid sequence {sequence}")

    def animate_square(self):
        """Add square animation sequence."""
        self.add_animation("squareness_upper_outer", 2, amplitude=0.3)
        self.add_animation("squareness_upper_inner", 2, amplitude=0.2)
        self.add_animation("squareness_lower_inner", 2, amplitude=0.2)
        self.add_animation("squareness_lower_outer", 2, amplitude=0.1)

    def animate_triangularity(self):
        """Add triangularity animation sequence."""
        self.add_animation("elongation_lower", 2, amplitude=0.1)  # TODO IDS fix
        self.add_animation("triangularity_upper", 2, amplitude=0.1)
        self.add_animation("elongation_upper", 2, amplitude=0.1)  # TODO IDS fix
        self.add_animation("triangularity_lower", 2, amplitude=0.1)

    def animate_box(self):
        """Add bounding box animation."""
        self.add_animation("elongation", 4, amplitude=-0.1)
        self.add_animation("elongation_lower", 2, amplitude=0.2)
        self.add_animation("triangularity_upper", 2, amplitude=0.2)
        self.add_animation("elongation_upper", 2, amplitude=0.2)
        self.add_animation("minimum_gap", 6, amplitude=0.2)


@dataclass
class BenchmarkDesign(PulseDesign):
    """Benchmark pulse design with source IDSs."""

    source_data: dict[str, Ids] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        """Load source equilibrium instance."""
        self.source_data["equilibrium"] = EquilibriumData(self.pulse, self.run)
        self.source_data["pf_active"] = PF_Active(self.pulse, self.run)
        super().__post_init__()

    def __getitem__(self, attr):
        """Extend getitem to include source data lookup."""
        if attr in self.source_data:
            return self.source_data[attr]
        return super().__getitem__(attr)

    def update(self):
        """Extend update to include source IDSs."""
        super().update()
        for attr in self.source_data:
            self[attr].time = self.time

    def plot(self, index=None, axes=None, **kwargs):
        """Extend plot to include source flux map and separatrix."""
        super().plot(index, axes, **kwargs)
        self["equilibrium"].plot_boundary(self.axes, "C2")

    def plot_current(self):
        """Compare benchmark coil curents."""
        self.set_axes("1d")
        coil_name = self["pf_active"].data.coil_name
        current = self["pf_active"]["current"]
        self.axes.bar(coil_name[:-1], 1e-3 * current[:-1], label="DINA")

        self.axes.bar(
            coil_name[:-1], 1e-3 * self.saloc["Ic"][:-2], width=0.5, label="NOVA"
        )
        self.axes.legend()
        self.axes.set_xlabel("coil name")
        self.axes.set_ylabel("coil current")

    def plot_waveform(self):
        """Extend plot_waveform to compare with benchmark."""
        # TODO extend from pulsedesign
        benchmark = self["pf_active"].data
        coil_name = benchmark.coil_name.data

        for group in ["CS", "PF"]:
            self.set_axes("1d")
            for i, name in enumerate(coil_name[:-1]):
                if group not in name:
                    continue
                self.axes.plot(
                    benchmark.time, 1e-3 * benchmark.current[:, i], color="gray"
                )
                self.axes.plot(
                    self.data.time, 1e-3 * self._data.current[:, i], label=name
                )
            self.axes.set_ylabel(f"{group} coil current, kA")
            self.axes.set_xlabel("time, s")
            self.axes.legend()

    def rms(self):
        """Calculate benchmark coil current rms error."""
        benchmark = self["pf_active"].data

        CS1U = benchmark.sel(coil_name="CS1").assign_coords(coil_name="CS1U")
        CS1L = benchmark.sel(coil_name="CS1").assign_coords(coil_name="CS1L")
        VS3U = benchmark.sel(coil_name="VS3").assign_coords(coil_name="VS3U")
        VS3L = benchmark.sel(coil_name="VS3").assign_coords(coil_name="VS3L")
        VS3U["current"] *= -1
        benchmark = xarray.concat([benchmark, CS1U, CS1L, VS3U, VS3L], "coil_name")
        benchmark = benchmark.interp(time=self.data.time)
        benchmark = benchmark.sel(coil_name=self.sloc["free", :].index)
        waveform = self._data.sel(coil_name=self.sloc["free", :].index)

        error = np.abs(waveform.current - benchmark.current)
        mean = np.mean((waveform.current + benchmark.current) / 2, axis=0)
        relative_error = np.mean(error, axis=0) / mean

        print(relative_error)

        # self.set_axes('1d')
        # self.axes.plot(error[:, 1])

        self.set_axes("1d")
        self.axes.plot(benchmark.current[:, 1], color="gray")
        self.axes.plot(waveform.current[:, 1], color="C0")


if __name__ == "__main__":
    design = AnimateDesign(
        135013,
        2,
        "iter",
        1,
        square=False,
        strike=True,
        fps=5,
    )
    # design = Benchmark(135013, 2, "iter", 1)

    # design.levelset.solve(limit=0.1, index="coil")
    design.itime = 0

    # design.plot_animation(False)
    # design.set_axes("triple")

    # design.set_axes("2d", aspect=1.5)

    # design.add_animation("time", 10, ramp=100)

    design.make_frame(100)

    # design.time = design.scene(20)["time"]
    # design.plasma.lcfs.plot()
    # design.fig.tight_layout(pad=0)
    # design.savefig("frame")

    # design.make_frame(84 / 5)

    # design.animate()

    """
    design.itime = 5
    # design["minor_radius"] = 0.5
    # design.update()
    design.square = False
    design["triangularity_upper"] = 0
    design.fit()
    design.update()
    design.plot("plasma")
    """

    """


    # Cocos
    design.levelset.plot_levelset(-design['psi_boundary'], False, color='k')
    design.levelset.plot_levelset(
        design.plasma.psi_boundary, False, color='C3')

    design.plot_waveform()
    """
