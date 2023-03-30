"""Manage access to dynamic coil data data."""
import bisect
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy
from shapely.geometry import LinearRing


from nova.frame.baseplot import Plot
from nova.imas.scenario import Scenario
from nova.imas.machine import Wall


@dataclass
class PulseSchedule(Plot, Scenario):
    """Manage access to pf_active ids."""

    name: str = 'pulse_schedule'
    ids_node: str = ''

    def __call__(self, attrs: list[str]) -> dict:
        """Return attribute listing."""
        return {attr: self[attr] for attr in attrs}

    def time_coordinate(self, path: str, attr: str, index=None):
        """Return time coordinate."""
        if self.data.homogeneous_time == 1:
            return ('time',)
        coord = f'{attr}_time'
        if index is None:
            self.data.coords[coord] = self.ids_index.get(path + '.time')
        else:
            self.data.coords[coord] = \
                self.ids_index.get_slice(index, path + '.time')
        return (coord,)

    def build_0d(self, ids_node: str, attrs: list[str]):
        """Extract position control attributes from ids."""
        with self.ids_index.node(ids_node):
            for attr in attrs:
                path = self.ids_index.get_path('*.reference', attr)
                if len(data := self.ids_index.get(path + '.data')) == 0:
                    continue
                self.data[attr] = self.time_coordinate(path, attr), data

    def build_point(self, ids_node: str):
        """Extract rz points."""
        name = ids_node.split('.')[-1]
        with self.ids_index.node(ids_node):
            if self.ids_index.empty('r.reference.data') and \
                    self.ids_index.empty('z.reference.data'):
                return
            time = self.time_coordinate('r.reference', name)
            shape = self.ids_index.shape('r.reference.data')[::-1] + (2,)
            data = np.zeros(shape, dtype=float)
            for i, attr in enumerate(['r', 'z']):
                if self.ids_index.empty(f'{attr}.reference.data'):
                    continue  # skip empty fields
                data[..., i] = self.ids_index.array(f'{attr}.reference.data')
            self.data.coords['point'] = ['r', 'z']
            if self.ids_index.length == 0:
                coordinate = time + ('point',)
            else:
                coordinate = time + (f'{name}_index', 'point')
            self.data[name] = coordinate, data

    def build_points(self, ids_node: str, attrs: list[str]):
        """Build point set."""
        for attr in attrs:
            self.build_point('.'.join([ids_node, attr]))

    @cached_property
    def wall_segment(self):
        """Return iter_md first wall instance."""
        return self.data.wall.data

    @cached_property
    def _angle(self, corner_eps=0.01):
        """Return gap angle interpolater (firstwall inward facing normals)."""
        boundary = self.wall_segment
        panel = boundary[1:] - boundary[:-1]
        number = 3*len(panel) - 1
        points = np.zeros((number, 2))
        points[::3] = boundary[:-1] + corner_eps*panel
        points[1::3] = boundary[:-1] + (1 - corner_eps)*panel
        points[2::3] = boundary[1:-1]
        tangent = panel / np.linalg.norm(panel, axis=1)[:, np.newaxis]
        tangent = np.append(tangent, np.zeros((len(tangent), 1)), axis=1)
        if LinearRing(boundary).is_ccw:
            normal = np.cross([0, 0, 1], tangent)[:, :2]
        else:
            normal = np.cross(tangent, [0, 0, 1])[:, :2]
        panel_angle = np.arctan2(normal[:, 1], normal[:, 0])
        angle = np.zeros(number)
        angle[::3] = panel_angle
        angle[1::3] = panel_angle
        angle[2::3] = np.mean(
            np.c_[panel_angle[:-1], panel_angle[1:]], axis=1)
        return scipy.interpolate.NearestNDInterpolator(points, angle)

    def build_gaps(self):
        """Build firstwall gaps."""
        with self.ids_index.node('position_control.gap'):
            if self.ids_index.empty('value.reference.data'):
                return
            gap_id = self.ids_index.array('identifier')
            self.data.coords['gap_index'] = range(len(gap_id))
            self.data.coords['gap_id'] = 'gap_index', gap_id
            if not self.ids_index.empty('name'):
                self.data.coords['gap_name'] = 'gap_index', \
                    self.ids_index.array('name')
            if 'point' not in self.data:
                self.data.coords['point'] = ['r', 'z']
            self.data.coords['gap_tail'] = ('gap_index', 'point'), \
                np.c_[self.ids_index.array('r'), self.ids_index.array('z')]
            if not self.ids_index.empty('angle'):
                self.data.coords['gap_angle'] = 'gap_index', \
                    self.ids_index.array('angle')
            else:
                self.data.coords['gap_angle'] = 'gap_index', \
                    self._angle(self.ids_index.array('r'),
                                self.ids_index.array('z'))
            self.data.coords['gap_vector'] = ('gap_index', 'point'), \
                np.c_[np.cos(self.data.gap_angle), np.sin(self.data.gap_angle)]
            if self.data.homogeneous_time == 1:
                self.data['gap'] = ('time', 'gap_index'), \
                    self.ids_index.array('value.reference.data')
            else:
                for index in self.data.gap_index.data:
                    path = 'value.reference'
                    attr = f'gap{index}'
                    coordinate = self.time_coordinate(path, attr, index)
                    self.data[attr] = coordinate, self.ids_index.get_slice(
                        index, path + '.data')

    def build_derived(self):
        """Build derived attributes."""
        if 'loop_voltage' in self.data:
            self.data['loop_psi'] = 'time', \
                -scipy.integrate.cumulative_trapezoid(
                    self.data.loop_voltage, self.data.time, initial=0)
            self.data['loop_psi'] -= self.data.loop_psi[-1]/2

    def build_wall(self):
        """Extract wall profile from IDS."""
        wall = Wall()
        self.data['wall'] = ('wall_index', 'point'), wall.segment()
        self.data.attrs['wall_md'] = ','.join(
            [str(value) for _, value in wall.ids_attrs.items()])

    def build(self):
        """Build netCDF database using data extracted from imasdb."""
        with self.build_scenario():
            self.build_0d(
                'flux_control',
                ['i_plasma', 'loop_voltage', 'li_3', 'beta_normal'])
            self.build_0d(
                'position_control',
                ['minor_radius', 'elongation', 'elongation_upper',
                 'elongation_lower', 'triangularity', 'triangularity_upper',
                 'triangularity_lower', 'active_limiter_point.r',
                 'active_limiter_point.z'])
            self.build_points(
                'position_control',
                ['magnetic_axis', 'geometric_axis',
                 'x_point', 'strike_point', 'boundary_outline'])
            self.build_wall()
            self.build_gaps()
            self.build_derived()
        return self

    def plot_profile(self):
        """Plot pulse schedule profile."""
        self.set_axes('1d', nrows=4, sharex=True)
        self.axes[0].plot(self.data.time, 1e-6*self.data.i_plasma, 'C0')
        self.axes[0].set_ylabel(r'$I_p$ MA')
        self.axes[1].plot(self.data.time, self.data.loop_voltage, 'C1')
        self.axes[1].set_ylabel(r'$V_{loop}$ Vm$^{-1}$')
        self.axes[2].plot(self.data.time, self.data.li_3, 'C2')
        self.axes[2].set_ylabel(r'$Li_3$')
        self.axes[3].plot(self.data.time, self.data.minor_radius, 'C3',
                          label=r'$r$')
        self.axes[3].plot(self.data.time, self.data.elongation, 'C4',
                          label=r'$\kappa$')
        self.axes[3].plot(self.data.time, self.data.triangularity, 'C5',
                          label=r'$\delta$')
        self.axes[3].set_ylabel('section')
        self.axes[3].legend()
        self.axes[-1].set_xlabel('time s')

    def plot_0d(self, attr, axes=None):
        """Plot 0D parameter timeseries."""
        self.set_axes('1d', axes=axes)
        self.axes.plot(self.data.time, self.data[attr], label=attr)
        self.axes.set_xlabel(r'$t$ s')
        self.axes.set_ylabel(attr)

    def plot_gaps(self):
        """Plot gaps."""
        self.get_axes('2d')
        self.axes.plot(self.wall_segment[:, 0], self.wall_segment[:, 1],
                       color='gray', lw=1.5)
        tail = self.data.gap_tail
        vector = self['gap'][:, np.newaxis] * \
            np.c_[np.cos(self.data.gap_angle), np.sin(self.data.gap_angle)]
        patch = self.mpl['patches'].FancyArrowPatch
        arrows = [patch((x, z), (x+dx, z+dz),
                        arrowstyle='|-|,'
                        'widthA=0.075, angleA=0, widthB=0.075, angleB=0',
                        shrinkA=0, shrinkB=0)
                  for x, z, dx, dz in
                  zip(tail[:, 0], tail[:, 1], vector[:, 0], vector[:, 1])]
        collections = self.mpl.collections.PatchCollection(
            arrows, facecolor='gray', edgecolor='gray')
        self.axes.add_collection(collections)

    def _make_frame(self, time):
        """Make frame for annimation."""
        self.axes.clear()
        max_time = np.min([self.data.time[-1], self.max_time])
        try:
            self.itime = bisect.bisect_left(
                self.data.time, max_time * time / self.duration)
        except ValueError:
            pass
        self.plot_gaps()
        return self.mpy.mplfig_to_npimage(self.fig)

    def annimate(self, duration: float, filename='gaps'):
        """Generate annimiation."""
        self.duration = duration
        self.max_time = 100
        self.set_axes('2d')
        animation = self.mpy.editor.VideoClip(
            self._make_frame, duration=duration)
        animation.write_gif(f'{filename}.gif', fps=10)


if __name__ == '__main__':

    pulse, run = 135007, 4
    pulse, run = 135011, 7
    pulse, run = 135003, 5
    # pulse, run = 105028, 1  # Maksim
    pulse, run = 105027, 1  # Maksim

    # PulseSchedule(pulse, run)._clear()
    schedule = PulseSchedule(pulse, run)

    #schedule.time = 500
    #schedule.plot_gaps()



    #schedule.plot_profile()

    # schedule.annimate(2.5)

    #schedule.plot_gaps()
    #schedule.plot_0d('loop_voltage')
    #schedule.plot_0d('loop_psi')
