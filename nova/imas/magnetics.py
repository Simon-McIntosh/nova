"""Load magnetics from machine description IDS."""

from dataclasses import dataclass

from tqdm import tqdm

from nova.frame.coilset import CoilSet
from nova.graphics.plot import Plot
from nova.imas.database import CoilData
from nova.imas.scenario import Scenario


@dataclass
class Magnetics(Plot, CoilSet, CoilData, Scenario):
    """Manage imas magnetics IDS."""

    pulse: int = 150100
    run: int = 4
    machine: str = "iter_md"
    occurence: int = 0
    user: str = "public"
    name: str = "magnetics"

    def build(self):
        """Build magnetics data."""
        with self.build_scenario():
            print(len(self.ids_data.flux_loop))
            self.data.coords["flux_loop_name"] = [
                loop.identifier for loop in self.ids_data.flux_loop
            ]
            for loop in tqdm(self.ids_data.flux_loop, "building flux loops"):
                print(loop.name, loop.identifier)
        print(self.data)


if __name__ == "__main__":
    magnetics = Magnetics()
    magnetics._clear()
