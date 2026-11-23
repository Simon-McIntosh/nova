"""Load magnetics from machine description."""

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import xarray
from packaging.version import Version

from nova.graphics.plot import Plot
from nova.imas.database import Database

DIAMAGNETIC_FLUX_NAME = "diamagnetic_flux"
DIAMAGNETIC_UNITS = "Wb"
DIAMAGNETIC_LOOP_TYPES = (
    "diamagnetic_internal",
    "diamagnetic_external",
    "diamagnetic_compensation",
    "diamagnetic_differential",
)
"""The flux-loop type labels under which a diamagnetic channel is published."""

DIAMAGNETIC_TARGET = "equilibrium/time_slice/constraints/diamagnetic_flux/measured"
"""Where a reconstruction carries the flux, declared b0_like by the dictionary
declaration of the field."""

DIAMAGNETIC_CALIBRATION_NOTE = (
    "the magnetics interface publishes a diamagnetic loop's flux already "
    "calibrated to weber, so the reader applies no channel scaling"
)


class DiamagneticConventionError(ValueError):
    """Raised when a b0_like diamagnetic flux contradicts its toroidal field."""


@dataclass(frozen=True)
class DiamagneticFlux:
    """One diamagnetic-flux channel and the provenance of its numbers.

    ``data`` is the loop flux in weber in the convention the magnetics entry was
    read at, so it carries the same sign sense as every other field read from
    that entry.  ``calibration`` is the factor the reader applied to the
    published signal to reach weber; the archive publishes a calibrated flux, so
    it is unity, and it is carried explicitly so a source that needs a factor
    can state one instead of hiding it in the data.
    """

    source: str
    channel: str
    time: np.ndarray
    data: np.ndarray
    cocos: int
    calibration: float = 1.0
    calibration_note: str = DIAMAGNETIC_CALIBRATION_NOTE
    error_upper: np.ndarray | None = None
    method: str = ""
    b0_like: bool = True

    def attributes(self) -> dict[str, object]:
        """Return the store attributes that carry this row's provenance."""

        attributes: dict[str, object] = {
            "standard_name": DIAMAGNETIC_FLUX_NAME,
            "source": self.source,
            "units": DIAMAGNETIC_UNITS,
            "cocos": int(self.cocos),
            "calibration": float(self.calibration),
            "calibration_note": self.calibration_note,
            "b0_like": bool(self.b0_like),
            "target_path": DIAMAGNETIC_TARGET,
        }
        if self.method:
            attributes["method"] = self.method
        return attributes


def b0_like_sense(
    data: np.ndarray,
    toroidal_field: float | np.ndarray,
    *,
    calibration: float = 1.0,
) -> bool | None:
    """Return whether a flux is b0_like with respect to the toroidal field.

    ``True`` when the row's largest excursion carries the field's own sense,
    ``False`` when it is reversed, and ``None`` when the row states no sense at
    all: no finite sample, no field sample, a field that changes sign, or an
    all-zero excursion.  A drive that reverses the field without reversing the
    flux therefore reads ``False``.
    """

    values = np.asarray(data, dtype=float) * float(calibration)
    measured = values[np.isfinite(values)]
    field = np.asarray(toroidal_field, dtype=float).reshape(-1)
    field = field[np.isfinite(field)]
    if measured.size == 0 or field.size == 0:
        return None
    if np.any(np.sign(field) != np.sign(field[0])) or field[0] == 0.0:
        return None
    peak = measured[np.argmax(np.abs(measured))]
    if peak == 0.0:
        return None
    return bool(np.sign(peak) == np.sign(field[0]))


def require_b0_like_sign(
    data: np.ndarray,
    toroidal_field: float | np.ndarray,
    *,
    calibration: float = 1.0,
    name: str = "diamagnetic flux",
) -> None:
    """Refuse a b0_like flux whose excursion contradicts the toroidal field.

    A b0_like quantity reverses with the toroidal field, so a diamagnetic loop's
    largest excursion carries the field's own sense.  A flux left unflipped
    under a reversed field therefore reads as a measurement while stating the
    opposite sign, and filling a b0_like target with it would invert the
    beta_p + l_i/2 the constraint exists to measure.
    """

    values = np.asarray(data, dtype=float) * float(calibration)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise DiamagneticConventionError(f"{name} carries no finite sample")
    field = np.asarray(toroidal_field, dtype=float).reshape(-1)
    field = field[np.isfinite(field)]
    if field.size == 0:
        raise DiamagneticConventionError(
            f"{name} is checked against no toroidal-field sample"
        )
    if np.any(np.sign(field) != np.sign(field[0])):
        raise DiamagneticConventionError(
            f"{name}: the toroidal field changes sign within the row, so the row "
            "names no single b0_like sense"
        )
    if field[0] == 0.0:
        raise DiamagneticConventionError(f"{name}: the toroidal field is zero")
    measured = values[finite]
    peak = measured[np.argmax(np.abs(measured))]
    if peak == 0.0:
        raise DiamagneticConventionError(
            f"{name}: every sample is zero, so the row carries no excursion to "
            "declare a b0_like sense"
        )
    if np.sign(peak) != np.sign(field[0]):
        raise DiamagneticConventionError(
            f"{name} peaks at {peak:+.6g} Wb while the toroidal field is "
            f"{field[0]:+.6g} T, so its b0_like sense is reversed: the flux was "
            "not flipped with the field"
        )


def diamagnetic_flux_dataset(
    fluxes: tuple[DiamagneticFlux, ...],
    *,
    toroidal_field: float | np.ndarray,
    channels: tuple[str, ...] | None = None,
) -> xarray.Dataset:
    """Return the diamagnetic rows as one store-keyed dataset.

    Every stored channel is measured against the toroidal field before it is
    written, so a row whose b0_like sense is reversed is refused here rather
    than discovered by the constraint that consumes it.  ``channels`` selects by
    channel name or index; an archive may publish auxiliary loops alongside the
    main diamagnetic loop and only the selected ones are asserted b0_like.  One
    clock is required because a store entry is keyed to one time base, and
    resampling a measurement onto another clock would be a modelling choice this
    reader does not make.
    """

    rows = tuple(fluxes)
    if channels is not None:
        by_name = {row.channel: row for row in rows}
        by_index = {str(index): row for index, row in enumerate(rows)}
        selected, missing = [], []
        for name in (str(channel) for channel in channels):
            row = by_name.get(name) or by_index.get(name)
            if row is None:
                missing.append(name)
            else:
                selected.append(row)
        if missing:
            raise ValueError(f"diamagnetic channels not found: {sorted(missing)}")
        rows = tuple(selected)
    if not rows:
        raise ValueError("no diamagnetic flux channel to store")
    clock = np.asarray(rows[0].time, dtype=float)
    conventions = {int(row.cocos) for row in rows}
    if len(conventions) != 1:
        raise ValueError("diamagnetic channels carry different COCOS conventions")
    for row in rows:
        require_b0_like_sign(
            row.data, toroidal_field, calibration=row.calibration, name=row.channel
        )
        if np.shape(row.time) != np.shape(clock) or not np.array_equal(row.time, clock):
            raise ValueError(
                "diamagnetic channels do not share one clock, so they cannot "
                "share one store time base"
            )
    channel_dim = f"{DIAMAGNETIC_FLUX_NAME}_channel"
    stacked = np.stack(
        [np.asarray(row.data, dtype=float) * row.calibration for row in rows], axis=1
    )
    attrs = {
        "standard_name": DIAMAGNETIC_FLUX_NAME,
        "source": [row.source for row in rows],
        "units": DIAMAGNETIC_UNITS,
        "cocos": conventions.pop(),
        "calibration": [float(row.calibration) for row in rows],
        "calibration_note": DIAMAGNETIC_CALIBRATION_NOTE,
        "b0_like": True,
        "target_path": DIAMAGNETIC_TARGET,
        "toroidal_field": float(np.asarray(toroidal_field).reshape(-1)[0]),
    }
    data_vars = {
        DIAMAGNETIC_FLUX_NAME: xarray.DataArray(
            stacked,
            dims=("time", channel_dim),
            coords={channel_dim: [row.channel for row in rows]},
            attrs=attrs,
        )
    }
    if all(row.error_upper is not None for row in rows):
        data_vars[f"{DIAMAGNETIC_FLUX_NAME}_error_upper"] = xarray.DataArray(
            np.stack(
                [np.asarray(row.error_upper, dtype=float) for row in rows], axis=1
            ),
            dims=("time", channel_dim),
            coords={channel_dim: [row.channel for row in rows]},
            attrs={"units": DIAMAGNETIC_UNITS, "role": "stated error"},
        )
    return xarray.Dataset(data_vars, coords={"time": clock}, attrs=attrs)


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

    def diamagnetic_fluxes(
        self,
        source: str | None = None,
        toroidal_field: float | np.ndarray | None = None,
    ) -> tuple[DiamagneticFlux, ...]:
        """Return every diamagnetic-flux channel this magnetics entry carries.

        The dedicated ``diamagnetic_flux`` structure is read first; a diamagnetic
        flux loop published only through ``flux_loop`` is read second, using the
        loop type labels the diagnostic table names.  An entry that carries
        neither returns no row rather than a zero-filled one, so a consumer can
        tell an absent measurement from a measured zero.

        ``source`` names the entry the channels were read from.  It defaults to
        the reader's own URI; pass it when the IDS was supplied directly and the
        URI therefore does not identify the archive it came from.
        """

        from nova.io.sourcemap import dd_equilibrium_convention

        prefix = self.uri if source is None else source
        cocos = dd_equilibrium_convention(str(self.ids._dd_version))
        channels: list[DiamagneticFlux] = []
        for index, structure in enumerate(getattr(self.ids, "diamagnetic_flux", ())):
            data = np.asarray(structure.data, dtype=float)
            if data.size == 0:
                continue
            channels.append(
                DiamagneticFlux(
                    source=f"{prefix}#magnetics.diamagnetic_flux({index}).data",
                    channel=f"diamagnetic_flux[{index}]",
                    time=self._channel_time(structure.time),
                    data=data,
                    cocos=cocos,
                    error_upper=self._stated_error(structure, data.size),
                    method=str(structure.method_name).strip(),
                    b0_like=self._b0_like(data, toroidal_field, 1.0),
                )
            )
        if channels:
            return tuple(channels)
        return self._flux_loop_diamagnetics(cocos, prefix, toroidal_field)

    @staticmethod
    def _b0_like(
        data: np.ndarray,
        toroidal_field: float | np.ndarray | None,
        calibration: float,
    ) -> bool:
        """Return whether a row's excursion follows the toroidal field."""

        if toroidal_field is None:
            return True
        return b0_like_sense(data, toroidal_field, calibration=calibration) is True

    def _flux_loop_diamagnetics(
        self,
        cocos: int,
        prefix: str,
        toroidal_field: float | np.ndarray | None = None,
    ) -> tuple[DiamagneticFlux, ...]:
        """Return the flux loops whose type label names a diamagnetic channel."""

        labels = self.diagnostic["flux_loop"]
        channels: list[DiamagneticFlux] = []
        for index, loop in enumerate(self.ids.flux_loop):
            kind = int(loop.type.index) if hasattr(loop, "type") else 0
            label = labels[kind - 1] if 1 <= kind <= len(labels) else ""
            if label not in DIAMAGNETIC_LOOP_TYPES:
                continue
            data = np.asarray(loop.flux.data, dtype=float)
            if data.size == 0:
                continue
            name, identifier = self.sensor_identity(loop)
            channels.append(
                DiamagneticFlux(
                    source=f"{prefix}#magnetics.flux_loop({index}).flux.data",
                    channel=identifier or name,
                    time=self._channel_time(loop.flux.time),
                    data=data,
                    cocos=cocos,
                    error_upper=self._stated_error(loop.flux, data.size),
                    method=label,
                    b0_like=self._b0_like(data, toroidal_field, 1.0),
                )
            )
        return tuple(channels)

    def _channel_time(self, structure_time) -> np.ndarray:
        """Return the clock a channel's samples sit on under the entry's layout."""

        if int(self.ids.ids_properties.homogeneous_time) == 1:
            return np.asarray(self.ids.time, dtype=float)
        return np.asarray(structure_time, dtype=float)

    @staticmethod
    def _stated_error(structure, size: int) -> np.ndarray | None:
        """Return a channel's stated upper error when it is published."""

        upper = np.asarray(getattr(structure, "data_error_upper", ()), dtype=float)
        if upper.shape != (size,):
            return None
        return upper

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
