"""Manage access to non-axisymmetric coil data."""
from dataclasses import dataclass, field

from nova.imas.coil import coil_names
from nova.graphics.plot import Plot
from nova.imas.scenario import Scenario


@dataclass
class Coils_Non_Axisymmetyric(Plot, Scenario):
    """Manage access to coils_non_axisymmetric ids."""

    name: str = "coils_non_axisymmetric"
    ids_node: str = "coil"
    coil_attrs: list[str] = field(default_factory=lambda: ["current", "voltage"])

    def build(self):
        """Build netCDF database using data extracted from imasdb."""
        # name = [self.coil_name(coil).strip() for coil in self.ids_data.coil]
        coil_name = coil_names(self.ids_data.coil)
        with self.build_scenario():
            self.data.coords["coil_name"] = coil_name


if __name__ == "__main__":
    pulse, run = 115001, 1
    Coils_Non_Axisymmetyric(pulse, run, "iter_md")._clear()
    coil = Coils_Non_Axisymmetyric(pulse, run, "iter_md")

    print(coil.data)
