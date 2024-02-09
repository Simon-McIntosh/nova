"""Calculate plasma shape from pulse schedule."""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import KDTree

from nova.geometry.plasmaprofile import PlasmaProfile
from nova.imas.pulseschedule import PulseSchedule


@dataclass
class PulseProfile(PlasmaProfile, PulseSchedule):
    """Fit Last Closed Flux Surface to Pulse Schedule gap parameters."""

    gap_head: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """Extract geometric axis from pulse schedule dataset."""
        super().__post_init__()
        self.extract_geometric_axis()

    def extract_geometric_axis(self):
        """Extract geometric radius and height from geometric axis."""
        self.data["geometric_radius"] = self.data.geometric_axis[:, 0]
        self.data["geometric_height"] = self.data.geometric_axis[:, 1]

    @cached_property
    def gap_tail(self) -> np.ndarray:
        """Return gap tail from pulse schedule dataset."""
        return self.data.gap_tail.data

    @cached_property
    def gap_vector(self):
        """Return gap vactor from pulse schedule dataset."""
        return self.data.gap_vector.data

    def update(self):
        """Extend GetSlice.update to include gap_head calculation."""
        super().update()
        try:  # clear cached property
            delattr(self, "topology")
        except AttributeError:
            pass
        self.gap_head = self.get("gap")[:, np.newaxis] * self.gap_vector
        self.gap_head += self.gap_tail

    @cached_property
    def hfs_radius(self):
        """Return first wall radius on the high field side."""
        return self.wall_segment[:, 0].min()

    @cached_property
    def lfs_gap_index(self):
        """Return low field side gap index."""
        return self.gap_tail[:, 0] > self.hfs_radius + 1e-3

    @cached_property
    def topology(self):
        """Return plasma topology discriptor."""
        if self.get("x_point")[0, 0] < self.hfs_radius + 1e-3:
            return "limiter"
        return "single_null"

    def update_separatrix(self, coef: np.ndarray):
        """Update plasma boundary points."""
        if self.topology == "limiter":
            return self.limiter(*coef)
        return self.single_null(*coef, x_point=self.get("x_point")[0])

    def kd_tree(self, points: np.ndarray) -> np.ndarray:
        """Return boundary point selection index using a 2d partition tree."""
        return KDTree(self.points).query(self.gap_head)[1]

    def kd_index(self, coef: np.ndarray) -> np.ndarray:
        """Update separatrix with coef and return gap-point selection index."""
        self.update_separatrix(coef)
        return self.kd_tree(self.points)

    def objective(self, coef: np.ndarray) -> float:
        """Return lcfs fitting objective."""
        index = self.kd_index(coef)
        error = np.linalg.norm(self.points[index, :] - self.gap_head, axis=1)
        return np.mean(error**2)

    def gap_constraint(self, coef: np.ndarray) -> np.ndarray:
        """Return lcfs fillting constraints."""
        index = self.kd_index(coef)
        gap_delta = np.einsum(
            "ij,ij->i", self.gap_vector, self.points[index, :] - self.gap_head
        )
        if self.topology == "limiter":
            return gap_delta[self.lfs_gap_index]
        return gap_delta

    def limiter_constraint(self, coef: np.ndarray) -> np.ndarray:
        """Return lcfs radial hfs limiter constraint."""
        self.update_separatrix(coef)
        return np.array([self.points[:, 0].min() - self.hfs_radius])

    @property
    def constraints(self):
        """Return gap and limiter constraints."""
        gap_constraint = dict(type="ineq", fun=self.gap_constraint)
        if self.topology == "limiter":
            limiter_constraint = dict(type="eq", fun=self.limiter_constraint)
            return [gap_constraint, limiter_constraint]
        return gap_constraint

    @property
    def bounds(self):
        """Return parameter bounds."""
        return [
            (-1, 1) if attr == "triangularity" else (None, None)
            for attr in self.profile_attrs
        ]

    @property
    def coef_o(self):
        """Return IDS profile coeffients."""
        return [self.get(attr) for attr in self.profile_attrs]

    def initialize(self):
        """Initialize analytic separatrix with pulse schedule data."""
        self.objective(self.coef_o)

    def fit(self):
        """Fit analytic separatrix to pulse schedule gaps."""
        self.initialize()
        if np.allclose(self.get("gap"), 0):
            return
        sol = minimize(
            self.objective,
            self.coef_o,
            method="SLSQP",
            bounds=self.bounds,
            constraints=self.constraints,
            tol=0.001,
        )
        self.objective(sol.x)

    def plot(self, axes=None, **kwargs):
        """Plot first wall, gaps, and plasma profile."""
        if not np.isclose(self["geometric_radius"], 0):
            super().plot(axes=axes, **kwargs)
        # if self.topology == 'single_null':
        #    self.axes.plot(*self.get('x_point')[0], 'x',
        #                   ms=6, mec='C3', mew=1, mfc='none')
        self.plot_gaps()

    '''
    def _make_frame(self, time):
        """Make frame for annimation."""
        self.axes.clear()
        max_time = np.min([self.data.time[-1], self.max_time])
        try:
            self.itime = bisect.bisect_left(
                self.data.time, max_time * time / self.duration
            )
        except ValueError:
            pass
        self.initialize()
        self.fit()
        self.plot()
        return self.mpy.mplfig_to_npimage(self.fig)

    def annimate(self, duration: float, filename="gaps"):
        """Generate annimiation."""
        self.duration = duration
        self.max_time = 15
        self.set_axes("2d")
        animation = self.mpy.editor.VideoClip(self._make_frame, duration=duration)
        animation.write_gif(f"{filename}.gif", fps=10)
    '''


if __name__ == "__main__":
    pulse, run = 135003, 5
    pulseprofile = PulseProfile(pulse, run)

    pulseprofile.time = 11.656 - 0.5

    pulseprofile.fit()
    pulseprofile.plot()

    pulseprofile.annimate(10, "gaps_fit_limiter")
