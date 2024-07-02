"""Generate and benchmark force and field coupling matricies."""

from dataclasses import dataclass
from functools import cached_property

from nova.imas.database import Database
from nova.imas.dataset import Ids
from nova.imas.operate import Operate
from nova.imas.profiles import Profile


@dataclass
class Matrix(Operate):
    """Calculate force and field copuling matricies + write to file."""

    pulse: int = 135014
    run: int = 1
    pf_active: Ids | bool | str = "iter_md"
    time_index: int = 315
    nforce: int | float = 500
    nfield: int | float = 50

    def write(self):
        """Write coupling matricies to file."""
        self.fsys.makedirs(str(self.path / self.filename), exist_ok=True)
        for attr in ["field", "force"]:
            data_attrs = getattr(self, attr).data.attributes + ["index"]
            data = getattr(self, attr).data[data_attrs].copy()
            data.attrs = dict(itime=self.itime) | self.ids_attrs
            data.attrs |= self.metadata
            filepath = self.path / self.filename / f"{attr}.nc"
            data.to_netcdf(filepath)

    def plot(self):
        """Plot coilset, fluxmap and coil force vectors."""
        super().plot()
        self.grid.plot()
        self.plasma.wall.plot(limitflux=False)
        self.force.plot(scale=2)


@dataclass
class Benchmark(Matrix):
    """Benchmark EM coupling matricies with other IDS."""

    profile_ids: Ids = (135007, 4)

    def __post_init__(self):
        """Generate profile instance."""
        self.profile_ids = Database.update_ids_attrs(self.profile_ids)
        self.profile = Profile(**self.profile_ids)
        super().__post_init__()
        self.update_plasma_shape()

    @cached_property
    def profile_index(self):
        """Return common coil index for profile data."""
        return [
            i
            for i, name in enumerate(self.profile.data.coil_name.data)
            if name in self.sloc.frame.index
        ]

    @cached_property
    def sloc_index(self):
        """Return common coil index for profile data."""
        return [
            self.sloc.frame.index.get_loc(name)
            for name in self.profile.data.coil_name[self.profile_index].data
            if name in self.sloc.frame.index
        ]

    def _check_coil_names(self):
        """Check for consistent coil labels between frame and profile data."""
        assert all(
            n == i
            for index, name in zip(
                self.sloc["coil", :].index, self.profile.data.coil_name.data
            )
            for n, i in zip(name, index)
        )

    def update(self):
        """Supress plasma shape update."""
        self.profile.itime = self.time_index
        self.update_current()

    def update_current(self):
        """Update coil currents from profile data."""
        self.sloc["coil", "Ic"] = 0
        self.sloc[self.sloc_index, "Ic"] = self.profile["current"][self.profile_index]
        self.sloc["plasma", "Ic"] = self.profile["ip"]

    def plot_force(self):
        """Plot timeslice force benchmark."""
        self.set_axes("1d", nrows=3, sharex=True)
        self.axes[0].bar(self.force.coil_name, self.force.fr * 1e-6, label="matrix")
        self.axes[0].bar(
            self.profile.data.coil_name,
            self.profile["radial_force"] * 1e-6,
            width=0.6,
            label="ids",
        )
        self.axes[0].legend()

        self.axes[1].bar(self.force.coil_name, self.force.fz * 1e-6)
        self.axes[1].bar(
            self.profile.data.coil_name,
            self.profile["vertical_force"] * 1e-6,
            width=0.6,
        )

        self.axes[2].bar(self.force.coil_name, self.force.fc * 1e-6)
        self.axes[2].set_xticks(range(len(self.force.coil_name)))
        self.axes[2].set_xticklabels(self.force.coil_name, rotation=90)

        self.axes[0].set_ylabel(r"$f_r$ MN")
        self.axes[1].set_ylabel(r"$f_z$ MN")
        self.axes[2].set_ylabel(r"$f_c$ MN")

    def plot_field(self):
        """Plot timeslice maximum L2 norm field benchmark."""
        self.set_axes("1d")
        self.axes.bar(self.field.coil_name, self.field.bp, label="matrix")
        self.axes.bar(
            self.profile.data.coil_name,
            self.profile["b_field_max_timed"],
            width=0.6,
            label="ids",
        )
        self.axes.set_xticks(range(len(self.field.coil_name)))
        self.axes.set_xticklabels(self.field.coil_name, rotation=90)
        self.axes.set_ylabel(r"$b_n$ T")
        self.axes.legend()


if __name__ == "__main__":
    # matrix = Matrix()
    # matrix.write()

    benchmark = Benchmark(ngrid=None, tplasma="hex")
    benchmark.itime = -1
    benchmark.plot_force()
    benchmark.plot_field()

    benchmark.set_axes("2d")
    benchmark.plot()

    # benchmark.plasmagrid.plot()

    # matrix = Matrix()

    # matrix.itime = 300
    # matrix.plot()
