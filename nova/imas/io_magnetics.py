"""Load magnetics from machine description."""

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from packaging.version import Version

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
        if self.ids is None:
            raise ValueError("magnetics IDS is not available")
        self.build_frame()
        self.build_summary()
        self.build_flux_loops()

    def __getitem__(self, key):
        """Return item from data dict."""
        return self.data[key]

    def __setitem__(self, key, item):
        """Return item from data dict."""
        self.data[key] = item

    @property
    def schema_major(self) -> int:
        """Return the opened schema, independent of producer version metadata."""
        return Version(self.ids._dd_version).major

    def sensor_identity(self, sensor) -> tuple[str, str]:
        """Return display name and stable identity under the opened schema."""
        name = str(sensor.name)
        identifier = str(sensor.identifier) if self.schema_major < 4 else name
        return name, identifier

    def build_frame(self):
        """Extract diagnostic collections directly from the supplied IDS."""
        identifier, name, diagnostic_name, diagnostic_type = [], [], [], []
        for diagnostic, labels in self.diagnostic.items():
            collection = diagnostic
            if self.schema_major >= 4 and diagnostic == "b_field_tor_probe":
                collection = "b_field_phi_probe"
            for sensor in getattr(self.ids, collection):
                sensor_name, sensor_identifier = self.sensor_identity(sensor)
                name.append(sensor_name)
                identifier.append(sensor_identifier)
                diagnostic_name.append(collection)
                kind = int(sensor.type.index) if labels else 0
                diagnostic_type.append(
                    labels[kind - 1] if 1 <= kind <= len(labels) else collection
                )
        self.data["frame"] = {
            "identifier": np.array(identifier, dtype=object),
            "name": np.array(name, dtype=object),
            "diagnostic_name": np.array(diagnostic_name, dtype=object),
            "diagnostic_type": np.array(diagnostic_type, dtype=object),
        }

    def build_summary(self):
        """Summarize repeated sensor names, preserving named ITER groups."""
        frame = self["frame"]
        index, identifier, name, diagnostic_type, number = [], [], [], [], []
        for data_name in self._unique(frame["name"]):
            select = frame["name"] == data_name
            row_identifier = frame["identifier"][select]
            token, separator, description = data_name.partition(" ")
            grouped = separator and "." in token
            index.append(token.split(".")[1] if grouped else data_name)
            identifier.append(
                row_identifier[0].rsplit("-", 1)[0] if grouped else row_identifier[0]
            )
            name.append(description if grouped else data_name)
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
        """Read loop identities and full cylindrical contours without flattening."""
        columns = ["name", "identifier", "group", "type", "r", "z", "phi", "indices"]
        columns += ["area", "gm9"]
        rows = {column: [] for column in columns}
        for sensor in self.ids.flux_loop:
            name, identifier = self.sensor_identity(sensor)
            kind = int(sensor.type.index)
            parts = identifier.split(".", 3)
            labels = self.diagnostic["flux_loop"]
            group = (
                parts[1]
                if len(parts) > 1
                else (labels[kind - 1] if 1 <= kind <= len(labels) else "flux_loop")
            )
            rows["name"].append(name)
            rows["identifier"].append(identifier)
            rows["group"].append(group)
            rows["type"].append(kind)
            for attr in ["r", "z", "phi"]:
                rows[attr].append(
                    np.array([float(getattr(p, attr)) for p in sensor.position])
                )
            rows["indices"].append(np.asarray(sensor.indices_differential).copy())
            rows["area"].append(float(sensor.area))
            rows["gm9"].append(float(sensor.gm9))
        flux_loop = {}
        for column, values in rows.items():
            array = np.empty(len(values), dtype=object)
            array[:] = values
            flux_loop[column] = array
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
