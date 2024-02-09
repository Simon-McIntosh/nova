"""Manage access to dynamic coil data data."""

from dataclasses import dataclass, field

from nova.graphics.plot import Plot
from nova.imas.coil import coil_names, coil_labels
from nova.imas.scenario import Scenario


@dataclass
class PF_Active(Plot, Scenario):
    """Manage access to pf_active ids."""

    name: str = "pf_active"
    ids_node: str = "coil"
    coil_attrs: list[str] = field(
        default_factory=lambda: ["current", "b_field_max_timed"]
    )

    def build(self):
        """Build netCDF database using data extracted from imasdb."""
        name = coil_names(self.ids_data.coil)
        label = coil_labels(self.ids_data.coil)
        with self.build_scenario():
            self.data.coords["coil_name"] = name
            self.data.coords["coil_labels"] = label
            self.data.coords["coil_index"] = "coil_name", range(len(name))
            self.append(("time", "coil_name"), self.coil_attrs, "*.data")
            with self.ids_index.node("coil"):
                if not self.ids_index.empty("current_limit_max"):
                    self.data["maximum_current"] = (
                        "coil_name",
                        self.ids_index.array("current_limit_max")[0, 0],
                    )
            coil_number = len(self.data.coil_name)
            for force in ["radial", "vertical"]:
                with self.ids_index.node(f"{force}_force"):
                    if self.ids_index.empty("force.data"):
                        continue
                    self.data[f"{force}_force"] = (
                        "time",
                        "coil_name",
                    ), self.ids_index.array("force.data")[:, :coil_number]
        return self

    def plot(self, axes=None, **kwargs):
        """Plot current timeseries."""
        self.set_axes("1d", axes=axes)
        self.axes.plot(self.data.time, 1e-3 * self.data.current, **kwargs)
        self.axes.set_xlabel("$t$ s")
        self.axes.set_ylabel("$I$ kA")


if __name__ == "__main__":
    # pf_active = PF_Active(130506, 403, machine='iter')
    pulse, run = 105028, 1
    pulse, run = 105011, 9
    pulse, run = 105007, 9
    pulse, run = 105011, 10
    pulse, run = 135003, 5
    # pulse, run = 115002, 4
    pulse, run = 135007, 4
    # pulse, run = 135011, 7
    pulse, run = 135013, 2

    # pulse, run = 135014, 1

    # pulse, run = 45272, 1, "mast_u"

    # pulse, run = 111001, 202
    # PF_Active(pulse, run, "iter")._clear()
    pf_active = PF_Active(pulse, run)
    # pf_active = PF_Active(105007, 9)  # b field max timed 135002, 5
    pf_active.plot()
