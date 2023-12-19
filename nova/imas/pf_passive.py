"""Manage access to dynamic pf passive data."""
from dataclasses import dataclass, field


from nova.graphics.plot import Plot
from nova.imas.scenario import Scenario


@dataclass
class PF_Passive(Plot, Scenario):
    """Manage access to pf_passive ids."""

    name: str = "pf_passive"
    ids_node: str = "loop"
    loop_attrs: list[str] = field(default_factory=lambda: ["current"])

    def build(self):
        """Build netCDF database using data extracted from imasdb."""
        with self.build_scenario():
            if len(self.ids.loop) == 0:
                return self
            self.data.coords["loop_name"] = self.ids_index.array("name")
            self.data.coords["loop_index"] = "loop_name", range(
                self.data.sizes["loop_name"]
            )
            self.append(("time", "loop_name"), self.loop_attrs)
        return self

    def plot(self, axes=None, **kwargs):
        """Plot current timeseries."""
        self.set_axes("1d", axes=axes)
        self.axes.plot(self.data.time, 1e-3 * self.data.current, **kwargs)
        self.axes.set_xlabel("$t$ s")
        self.axes.set_ylabel("$I$ kA")


if __name__ == "__main__":
    pulse, run = 105007, 9
    pulse, run = 134173, 106  # DINA / JINTRAC
    pulse, run = 135013, 2

    # PF_Passive(pulse, run)._clear()
    import time

    start_time = time.perf_counter()
    pf_passive = PF_Passive(pulse, run)
    print(f"run time {time.perf_counter() - start_time:1.3f}s")

    pf_passive = PF_Passive(pulse, run)
    pf_passive.plot()
    pf_passive.axes.set_xlim([0, 60])
