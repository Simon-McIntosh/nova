"""Read MAST Thomson scattering profiles from the level-1 shot store.

MAST published its Thomson data through two DISJOINT eras, and a reader that
knows only the later one cannot open most of the archive. Measured across
1712 shots sampled every tenth through the level-1 store: ``atm`` alone on 615
of them, up to shot 22822; ``ayc`` with ``aye`` on 374 and ``ayc`` alone on
101, from shot 23138; and ``atm`` co-occurring with ``ayc`` or ``aye`` on
exactly NONE. So the era is a property of the shot, not a set of optional
groups, and an early shot has a core string with no edge companion at all.

``atm`` is the earlier core system: 36 scattering volumes whose ``radius`` is
stored once per channel, so its geometry is already fixed. ``ayc`` is the
later core string at 131 volumes with a per-time radius, and ``aye`` the edge
string at 16 volumes clustered outboard -- optional even within its own era.

Every system measures along R at Z of about zero, so MAST offers no
vertical-versus-horizontal sightline split; the distinction a figure can draw
is core against edge, and on an ``atm`` shot there is only core. The stored
group code is preserved in each string's provenance because the systems are
not interchangeable -- ``ayc`` covers core and edge together where ``atm``
covers the core alone -- and a consumer needing that distinction should read
the code rather than the role name.

Three properties of the stored arrays are handled at the read, because each
one silently corrupts a figure otherwise:

*Padding masquerades as data.* A store group's time axis can carry real shot
times followed by integer index-like rows. Measured on shot 27079, ``aye``
holds 282 rows of which the first 142 span 0 to 0.587 s and the last 140 are
the integers 1 to 140; the real rows are 95 per cent finite positive
temperature and the padding is zero per cent. Rows are therefore admitted by
falling inside the shot's own time window, not by position.

*Channel radius drifts between laser pulses.* ``radius`` is stored per time
row and wanders by a few millimetres, so a channel's plotted position is the
median of its finite radii rather than one arbitrary row's value.

*There is no stored vertical coordinate.* Both systems view the midplane, so
Z is taken as zero and that assumption is recorded in the provenance rather
than being buried as a literal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nova.imas.mast_vacuum_cohort import SHOT_STORE

#: Core systems in era order, with the radius leaf each stores. The first
#: one present decides the shot's era; they never co-occur.
_CORE_SYSTEMS = (("atm", "radius"), ("ayc", "radius"))

#: Edge systems, which exist only in the later era and are optional there.
_EDGE_SYSTEMS = (("aye", "r"),)

#: MAST Thomson views the midplane; neither group stores a vertical position.
_MIDPLANE_HEIGHT = 0.0


@dataclass(frozen=True)
class ThomsonString:
    """One Thomson system's scattering volumes and their measured profiles."""

    name: str
    positions: np.ndarray
    time: np.ndarray
    temperature: np.ndarray
    density: np.ndarray
    provenance: dict = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the admitted time-row count."""
        return int(self.time.size)

    def at(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the nearest row's temperature in eV and density per cubic m.

        Nearest rather than interpolated: the laser fires at its own cadence
        and each row is one measurement, so averaging two of them would report
        a profile the instrument never took.
        """
        if self.time.size == 0:
            empty = np.full(self.positions.shape[0], np.nan)
            return empty, empty
        row = int(np.argmin(np.abs(self.time - float(time))))
        return self.temperature[row], self.density[row]

    def finite(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the radius, temperature and density of the usable channels.

        A channel is usable only when its POSITION is finite as well as its
        measurement. Both eras carry a channel whose stored radius is
        non-finite while its temperature is valid -- one of 36 on ``atm``, one
        of 131 on ``ayc`` -- and filtering on the measurement alone returns a
        NaN abscissa that matplotlib drops silently while any scale or fit
        computed from it inherits the gap.
        """
        temperature, density = self.at(time)
        usable = (
            np.isfinite(temperature)
            & (temperature > 0.0)
            & np.isfinite(self.positions[:, 0])
        )
        return self.positions[usable, 0], temperature[usable], density[usable]


def _channel_positions(radius: Any) -> np.ndarray:
    """Return one ``(channel, 2)`` position array from the stored radii.

    A channel whose radius is non-finite in every row keeps a non-finite
    position rather than being dropped, so channel indices stay aligned with
    the temperature and density columns; the painters and profile readers
    already discard non-finite entries.
    """
    stored = np.asarray(radius, dtype=float)
    if stored.ndim == 2:
        present = np.any(np.isfinite(stored), axis=0)
        centres = np.full(stored.shape[1], np.nan)
        if np.any(present):
            centres[present] = np.nanmedian(stored[:, present], axis=0)
    else:
        centres = stored
    return np.column_stack((centres, np.full(centres.size, _MIDPLANE_HEIGHT)))


def _shot_window(group: Any) -> tuple[float, float]:
    """Return the shot's own time window from the core system's real rows.

    The core group's own axis is used as the reference because it is the one
    whose rows are contiguous shot times; a window taken from a padded group
    would admit its own padding.
    """
    time = np.asarray(group["time"], dtype=float)
    finite = time[np.isfinite(time)]
    if finite.size == 0:
        raise ValueError("the core Thomson group carries no finite time")
    # Real MAST shot times are sub-second; anything past a second is padding.
    real = finite[(finite >= 0.0) & (finite <= 1.0)]
    if real.size == 0:
        raise ValueError("the core Thomson group carries no sub-second time row")
    return float(real.min()), float(real.max())


def read_string(
    group: Any, name: str, radius_leaf: str, window: tuple[float, float]
) -> ThomsonString:
    """Return one system's admitted rows and its channel positions."""
    time = np.asarray(group["time"], dtype=float)
    admitted = np.flatnonzero(
        np.isfinite(time) & (time >= window[0]) & (time <= window[1])
    )
    temperature = np.asarray(group["te"], dtype=float)[admitted]
    density = np.asarray(group["ne"], dtype=float)[admitted]
    usable = np.isfinite(temperature) & (temperature > 0.0)
    return ThomsonString(
        name=name,
        positions=_channel_positions(group[radius_leaf]),
        time=time[admitted],
        temperature=temperature,
        density=density,
        provenance={
            "stored_row_count": int(time.size),
            "admitted_row_count": int(admitted.size),
            "admission": "time inside the core system's shot window",
            "usable_sample_fraction": float(usable.mean()) if usable.size else 0.0,
            "height_m": _MIDPLANE_HEIGHT,
            "height_source": "both systems view the midplane; no stored Z leaf",
        },
    )


def read_thomson(
    shot: int, store: Path | str = SHOT_STORE
) -> tuple[ThomsonString, ...]:
    """Return the Thomson strings of one MAST shot, whichever era it is from.

    The first core system present decides the era; an absent edge system is
    omitted rather than returned empty, so a caller sees the systems that
    exist instead of a string with no channels. An early shot therefore
    returns one string and a later one may return two.
    """
    import zarr

    source = Path(store) / f"{shot}.zarr"
    if not source.is_dir():
        raise FileNotFoundError(f"no MAST level-1 shot store at {source}")
    root = zarr.open_group(str(source), mode="r")
    core = next(
        ((key, leaf) for key, leaf in _CORE_SYSTEMS if key in root),
        None,
    )
    if core is None:
        present = sorted(name for name, _ in root.groups())
        raise ValueError(
            f"MAST {shot} carries no core Thomson group: expected one of "
            f"{[key for key, _ in _CORE_SYSTEMS]}, found {present[:12]}"
        )
    window = _shot_window(root[core[0]])
    systems = [("core", *core)] + [
        ("edge", key, leaf) for key, leaf in _EDGE_SYSTEMS if key in root
    ]
    strings = []
    for name, key, radius_leaf in systems:
        string = read_string(root[key], name, radius_leaf, window)
        string.provenance["group"] = key
        string.provenance["era"] = "atm" if core[0] == "atm" else "ayc"
        string.provenance["source"] = str(source)
        string.provenance["shot_window_s"] = list(window)
        strings.append(string)
    return tuple(strings)
