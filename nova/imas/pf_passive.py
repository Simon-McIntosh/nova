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
                self.data.dims["loop_name"]
            )
            self.append(("time", "loop_name"), self.loop_attrs)
        return self


if __name__ == "__main__":
    pulse, run = 105007, 9
    pulse, run = 134173, 106  # DINA / JINTRAC

    # PF_Passive(pulse, run)._clear()
    pf_passive = PF_Passive(pulse, run)
