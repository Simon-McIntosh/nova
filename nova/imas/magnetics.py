"""Load magnetics from machine description."""
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas

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
    name: str = "magnetics"
    user: str = "public"
    machine: str = "iter_md"
    data: dict[str, pandas.DataFrame] = field(
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
        """Extract magnetics data."""
        self.data["frame"] = pandas.DataFrame()
        for diagnostic in self.diagnostic:
            name, identifier, diagnostic_name, diagnostic_type = [], [], [], []
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

            frame = pandas.DataFrame(
                dict(
                    name=name,
                    diagnostic_name=diagnostic_name,
                    diagnostic_type=diagnostic_type,
                ),
                index=identifier,
            )
            self["frame"] = pandas.concat([self["frame"], frame])

    def build_summary(self):
        """Extract overview from dataframe."""
        index, identifier, name, diagnostic_type, number = [], [], [], [], []
        for data_name in self["frame"].name.unique():
            frame = self["frame"].loc[self["frame"].name == data_name, :]
            index.append(data_name.split(" ")[0].split(".")[1])
            identifier.append("-".join(frame.index[0].split("-")[:-1]))
            name.append(" ".join(data_name.split(" ")[1:]))
            type_array = frame.diagnostic_type.unique()
            if len(type_array) != 1:
                raise ValueError(
                    f"diagnostic type not unique for {data_name} "
                    f"{frame.diagnostic_type.unique()}"
                )
            diagnostic_type.append(type_array[0])
            number.append(len(frame))
        self.data["summary"] = pandas.DataFrame(
            dict(
                name=name,
                identifier=identifier,
                diagnostic=diagnostic_type,
                number=number,
            ),
            index=index,
        )

    def build_flux_loops(self):
        """Build Partial FLux Loop diagnostic."""
        self.data["flux_loop"] = pandas.DataFrame(
            columns=["name", "identifier", "type"]
        )
        data = []
        for ids in self.get_ids("flux_loop"):
            group = ids.identifier.split(".", 3)[1]
            data.append(
                [
                    ids.name,
                    ids.identifier,
                    group,
                    ids.type.index,
                    *[
                        np.array([getattr(position, attr) for position in ids.position])
                        for attr in ["r", "z", "phi"]
                    ],
                    ids.indices_differential,
                    ids.area,
                    ids.gm9,
                ]
            )
        self.data["flux_loop"] = pandas.DataFrame(
            data,
            columns=[
                "name",
                "identifier",
                "group",
                "type",
                "r",
                "z",
                "phi",
                "indices",
                "area",
                "gm9",
            ],
        )
        self["flux_loop"].loc[self["flux_loop"].type < 3, "gm9"] = 0

    def plot(self, axes=None):
        """Plot diagnostics."""
        self.set_axes("2d", axes=axes)
        data = self["flux_loop"]

        for index in data.loc[data.group == "AD"].index:
            self.axes.plot(data.loc[index, "phi"], data.loc[index, "z"], "o-")
            print(data.loc[index, "identifier"])
            print(data.loc[index, "group"])

    def signal_types(self):
        """Add signal type information."""


if __name__ == "__main__":
    magnetics = Magnetics()
    magnetics.plot()
    # print(magnetics['flux_loop'].loc[0, 'r'])
    # print(magnetics['summary'])
