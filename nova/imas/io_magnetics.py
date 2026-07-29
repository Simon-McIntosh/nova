"""Load magnetics from machine description."""

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from nova.graphics.plot import Plot
from nova.imas.database import Database


@dataclass
class Magnetics(Plot, Database):
    """
    Manage active poloidal loop ids, pf_passive.

    Sensors
    -------
    A1	TF Rogowski
    A2	Diamagnetic Loop Rogowski
    A3	Tangential Coils (Outer)
    A4	Normal Coils (Outer)
    A5	Tangential Steady State Sensors
    A6	Normal Steady Steady Sensors
    A7	Continuous Flux Loops (Outer)
    A8	Fibre Optic Current Sensor
    A9	Diamagnetic Compensation (Outer)
    AA	Tangential Coils (Inner)
    AB	Normal Coils (Inner)
    AC	Toroidal Coils
    AD	Partial Flux Loops
    AE	Continuous Flux Loops (inner)
    AF	Diamagnetic loop (Main)
    AG	Diamagnetic Compensation (Inner)
    AH	Diamagnetic saddles (inner)
    AI	MHD Saddles
    AJ	HF Sensors
    AK	RWM Sensors
    AL	Divertor EquilibriumData Sensors
    AM	Divertor Shunts
    AN	Rogowskis (Divertor )
    AO	Toroidal Coils (Divertor)
    AP	Rogowskis (Blanket)

    """

    pulse: int = 150100
    run: int = 4
    machine: str = "iter_md"
    occurence: int = 0
    user: str = "public"
    name: str = "magnetics"

    data: dict[str, dict[str, np.ndarray]] = field(
        init=False, repr=False, default_factory=dict
    )

    signal: ClassVar[dict[str, str]] = dict(
        A1="i",
        A2="i",
        A3="i",
        A4="i",
        A5="p",
        A6="p",
        A7="i",
        A8="p",
        A9="i",
        AA="i",
        AB="i",
        AC="i",
        AD="i",
        AE="i",
        AF="i",
        AG="i",
        Ah="i",
        AI="i",
        AJ="i",
        AK="i",
        Al="i",
        AM="i",
        AN="i",
        AO="i",
        AP="i",
    )

    diagnostic: ClassVar[dict[str, list[str]]] = dict(
        flux_loop=[
            "toroidal",
            "saddle",
            "diamagnetic_internal",
            "diamagnetic_external",
            "diamagnetic_compensation",
            "diamagnetic_differential",
        ],
        b_field_pol_probe=[
            "position",
            "mirnov",
            "hall",
            "flux_gate",
            "faraday_fiber",
            "differential",
        ],
        b_field_tor_probe=[
            "position",
            "mirnov",
            "hall",
            "flux_gate",
            "faraday_fiber",
            "differential",
        ],
        rogowski_coil=[],
        shunt=[],
    )

    def __post_init__(self):
        """Load data from magnetics IDS and build overview."""
        super().__post_init__()
        self.build_frame()
        self.build_summary()
        self.build_flux_loops()

    def __getitem__(self, key):
        """Return item from data dict."""
        return self.data[key]

    def __setitem__(self, key, item):
        """Return item from data dict."""
        self.data[key] = item

    def build_frame(self):
        """Extract magnetics data into a columnar table keyed by identifier."""
        identifier, name, diagnostic_name, diagnostic_type = [], [], [], []
        for diagnostic in self.diagnostic:
            for ids in self.get_ids(diagnostic):
                name.append(ids.name)
                identifier.append(ids.identifier)
                diagnostic_name.append(diagnostic)
                try:
                    diagnostic_type.append(
                        self.diagnostic[diagnostic][ids.type.index - 1]
                    )
                except AttributeError:
                    diagnostic_type.append(diagnostic)
        self.data["frame"] = {
            "identifier": np.array(identifier, dtype=object),
            "name": np.array(name, dtype=object),
            "diagnostic_name": np.array(diagnostic_name, dtype=object),
            "diagnostic_type": np.array(diagnostic_type, dtype=object),
        }

    def build_summary(self):
        """Extract a per-sensor overview from the diagnostic table."""
        frame = self["frame"]
        index, identifier, name, diagnostic_type, number = [], [], [], [], []
        for data_name in self._unique(frame["name"]):
            select = frame["name"] == data_name
            row_identifier = frame["identifier"][select]
            index.append(data_name.split(" ")[0].split(".")[1])
            identifier.append("-".join(row_identifier[0].split("-")[:-1]))
            name.append(" ".join(data_name.split(" ")[1:]))
            type_array = self._unique(frame["diagnostic_type"][select])
            if len(type_array) != 1:
                raise ValueError(
                    f"diagnostic type not unique for {data_name} {type_array}"
                )
            diagnostic_type.append(type_array[0])
            number.append(int(select.sum()))
        self.data["summary"] = {
            "index": np.array(index, dtype=object),
            "name": np.array(name, dtype=object),
            "identifier": np.array(identifier, dtype=object),
            "diagnostic": np.array(diagnostic_type, dtype=object),
            "number": np.array(number, dtype=int),
        }

    def build_flux_loops(self):
        """Build Partial Flux Loop diagnostic table."""
        columns = ["name", "identifier", "group", "type", "r", "z", "phi", "indices"]
        columns += ["area", "gm9"]
        rows = {column: [] for column in columns}
        for ids in self.get_ids("flux_loop"):
            group = ids.identifier.split(".", 3)[1]
            rows["name"].append(ids.name)
            rows["identifier"].append(ids.identifier)
            rows["group"].append(group)
            rows["type"].append(ids.type.index)
            for attr in ["r", "z", "phi"]:
                rows[attr].append(
                    np.array([getattr(position, attr) for position in ids.position])
                )
            rows["indices"].append(ids.indices_differential)
            rows["area"].append(ids.area)
            rows["gm9"].append(ids.gm9)
        flux_loop = {column: np.array(rows[column], dtype=object) for column in columns}
        flux_loop["type"] = np.array(rows["type"], dtype=int)
        flux_loop["gm9"] = np.where(flux_loop["type"] < 3, 0, flux_loop["gm9"])
        self.data["flux_loop"] = flux_loop

    @staticmethod
    def _unique(values: np.ndarray) -> list:
        """Return unique values preserving first-seen order."""
        return list(dict.fromkeys(values.tolist()))

    def plot(self, axes=None):
        """Plot diagnostics."""
        self.set_axes("2d", axes=axes)
        data = self["flux_loop"]
        for i in np.flatnonzero(data["group"] == "AD"):
            self.axes.plot(data["phi"][i], data["z"][i], "o-")

    def signal_types(self):
        """Add signal type information."""


if __name__ == "__main__":
    args = 45272, 1, "mast_u"

    args = []
    magnetics = Magnetics(*args)
    magnetics.plot()
    # print(magnetics['flux_loop']['r'][0])
    # print(magnetics['summary'])
