"""Load magnetics from machine description IDS."""
from dataclasses import dataclass
from operator import attrgetter

from functools import cached_property


from imaspy.ids_struct_array import IDSStructArray
from imaspy.ids_toplevel import IDSToplevel

import xarray

from nova.frame.coilset import CoilSet
from nova.graphics.plot import Plot
from nova.imas.database import CoilData
from nova.imas.scenario import Scenario


@dataclass
class AoS:
    """Manage IMAS Array of Structures."""

    aos: IDSStructArray

    def __post_init__(self):
        """Extract data from array of structure container."""
        print(self.aos._element_structure)

    def __getitem__(self, index: int):
        """Return item from array of structures."""
        match index:
            case int(i):
                return self.aos[i]
            case str(path):
                metadata = attrgetter(path)
                return metadata(self.aos[0])
            case _:
                raise KeyError(f"Unable to index AoS with {index}.")
        return None

    @cached_property
    def data(self) -> xarray.Dataset:
        """Return AoS as xarray Dataset."""


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
        with self.ids_data() as ids_data:
            self.flux_loop(ids_data)

        """
        with self.build_scenario():
            print(len(self.ids_data.flux_loop))
            self.data.coords["flux_loop_name"] = [
                loop.identifier for loop in self.ids_data.flux_loop
            ]
            for loop in tqdm(self.ids_data.flux_loop, "building flux loops"):
                print(loop.name, loop.identifier)
        print(self.data)
        """

    def from_AoS(self, *names: str, dims: str | tuple[str], array: IDSStructArray):
        """Return data unpacked from from an IMAS Array of Structures."""

        return {
            name: (
                dims,
                [getattr(index, name).value for index in array],
                # {"units": array[0].metadata.units},
            )
            for name in names
        }

    def flux_loop(self, ids_data: IDSToplevel):
        """Return flux loop Dataset."""
        flux_loop = ids_data.flux_loop
        data = self.from_AoS("name", "identifier", dims="name", array=flux_loop)

        self.aos = AoS(ids_data.flux_loop)
        print(self.aos["area.metadata.units"])
        data = xarray.Dataset(data)
        # print(flux_loop[0].name.metadata.units)
        # print(flux_loop[0].)
        # reduce(operator.ior, list_of_dicts, {})

        # self.data.coords["name"] = [loop.name.value for loop in flux_loop]
        # self.data.coords["identifier"] = [loop.identifier.value for loop in flux_loop]

        # coordinates = {"name": [loop.name.value for loop in flux_loop]}

        # data = xarray.Dataset(coords=coordinates)
        # _ = [loop.identifier.value for loop in flux_loop]
        # name = [[] for _ in range(len(flux_loop))]
        # number = len(flux_loop)
        # data =

        # data =

        """
        identifier, name, _type = [], [], []
        for i, loop in enumerate(flux_loop):
            identifier.append(loop.identifier.value)
            name.append(loop.name.value)
            _type.append(loop.type)
        """


if __name__ == "__main__":
    magnetics = Magnetics()

    import imaspy

    db_entry = imaspy.DBEntry(magnetics.uri, "a")
    db_entry.open()
    ids_data = db_entry.get("magnetics")
    db_entry.close()

    magnetics._clear()
