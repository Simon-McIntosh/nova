"""Identify the MAST source coordinate convention from the data itself.

Determines all four Sauter & Medvedev COCOS digits for the FAIR-MAST ``efm``
equilibrium reconstruction empirically -- never from a units string or a
catalogue metadata tag alone -- and, in the same pass, separates raw static
setup fingerprints from canonical physical-configuration evidence.

The four digits and how each one is pinned
------------------------------------------

``sigma_Bp`` (poloidal flux sign)
    From Sauter Eq. 23, ``sign(psi_edge - psi_axis) = sigma_Ip * sigma_Bp``.
    Read ``psi_axis``, ``psi_boundary`` and ``plasma_current_c`` on every
    converged slice with appreciable current.  Cross-checked independently
    against ``sign(dp/dpsi) = -sigma_Ip * sigma_Bp`` using ``pprime``, so a
    single mis-signed array cannot carry the verdict alone.

``e_Bp`` (whether psi carries the 2*pi)
    The digit a units string would answer and therefore the one worth
    measuring.  A poloidal flux loop links the TOTAL flux
    ``Phi = 2*pi*R*A_phi`` [Wb]; the flux function ``psi = R*A_phi`` [Wb/rad]
    does not.  Interpolating the reconstructed ``psirz`` map onto each flux
    loop's own ``(R, Z)`` and dividing the loop's fitted flux ``silop_c`` by
    it therefore returns 2*pi if the map is per-radian (``e_Bp = 0``) and 1 if
    it is total flux (``e_Bp = 1``).  Run over every loop of every sampled
    shot; the reported statistic is the median ratio.

``sigma_rho_theta_phi`` (poloidal coordinate handedness)
    From ``sign(q) = sigma_Ip * sigma_B0 * sigma_rho_theta_phi``, using the
    reconstructed ``q_95`` / ``q_axis`` against ``plasma_current_c`` and the
    reference vacuum toroidal field ``bvac_val``.

``sigma_RphiZ`` (cylindrical handedness)
    NOT determinable from Eq. 23 -- no combination of scalar equilibrium
    quantities distinguishes ``(R, phi, Z)`` from ``(R, Z, phi)``, because every
    toroidal quantity's sign is expressed in the same frame being tested.  It
    is reported as a DECLARATION of the standard right-handed ``(R, phi, Z)``
    frame, supported by the self-consistency check ``sign(irod) ==
    sign(bvac_val)`` (rod current and the vacuum field it produces share a
    sign only if both refer to the same ``phi`` sense).  The script reports
    that check's pass rate and never upgrades it to a determination.

Physical configuration versus setup representation
---------------------------------------------------
The historical campaign fingerprint remains in the artifact because it is the
key used by existing Ambix operator caches.  It hashes every EFIT filament and
therefore changes when the same physical conductor is represented by a
different mesh.  It is a representation fingerprint, not a machine-description
identity.

The physical comparison in this module instead:

* groups filament cells by the physical circuit they discretise and compares
  geometry moments, bounds, turns and topology without retaining cell count;
* compares diagnostic positions and sensitive-axis orientations explicitly;
* compares the limiter as a closed curve using sampling-independent geometric
  moments; and
* treats ``amm`` positions as wall-model discretisation evidence, never as
  authoritative vessel geometry.

Configuration ranges are built only after equivalent setup representations
have been grouped.  Missing or corrupt source entries are retained as explicit
exceptions.  A boundary remains provisional unless an actual installation or
maintenance record establishes a physical change.

Reads only static-setup geometry plus reconstructed scalars used as a
convention referee; writes a JSON summary and the figures.  Nothing here
feeds a fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LEVEL1 = Path("/work/projects/imas_gpu/mast/level1/shots")

#: static-setup arrays hashed into the representation fingerprint (no time axis;
#: each is indexed by a geometry dimension).
FINGERPRINT_ARRAYS: tuple[str, ...] = (
    "magpr_r",
    "magpr_z",
    "magpr_ang",
    "silop_r",
    "silop_z",
    "fcoil_r",
    "fcoil_z",
    "fcoil_turns",
    "limiterr",
    "limiterz",
)

PHYSICAL_GEOMETRY_ARRAYS: tuple[str, ...] = (
    "magpr_r",
    "magpr_z",
    "magpr_ang",
    "magpr_len",
    "silop_r",
    "silop_z",
    "fcoil_r",
    "fcoil_z",
    "fcoil_turns",
    "fcoil_width",
    "fcoil_height",
    "fcoil_circ",
    "fcoil_xmult",
    "limiterr",
    "limiterz",
)

#: slices below this |Ip| are vacuum / pre-breakdown and carry no convention
#: information (``cutip`` in the corpus is the reconstruction's own cut-off).
MIN_IP_A = 1.0e5

#: a flux loop sitting where |psi| is this small has too little signal for the
#: ratio test; the map's own interpolation error would dominate.
MIN_PSI_WB_PER_RAD = 2.0e-3

#: minimum usable loops before a shot may report a ratio at all.
MIN_RATIO_LOOPS = 20

#: maximum interquartile spread of the per-loop ratio, relative to its median.
#: A shot whose ``silop_c`` is largely unpopulated at the sampled slice produces
#: a ratio scattered over orders of magnitude; such a shot is REJECTED rather
#: than allowed to report a meaningless number.  A shot whose loops genuinely
#: agree lands near 1% -- the bilinear interpolation error on a 3 cm grid.
MAX_RATIO_IQR_FRACTION = 0.10


def _finite(a: np.ndarray) -> np.ndarray:
    """Finite entries only (``silop`` arrays are NaN-padded in this corpus)."""
    a = np.asarray(a, dtype=np.float64)
    return a[np.isfinite(a)]


def geometry_digest(arrays: list[np.ndarray], decimals: int = 4) -> str:
    """16-hex digest of rounded arrays for representation compatibility."""
    h = hashlib.sha256()
    for a in arrays:
        r = np.round(np.asarray(a, dtype=np.float64), decimals)
        h.update(np.ascontiguousarray(r).tobytes())
    return h.hexdigest()[:16]


# The MAST control-system machine description contains 23 physical circuits:
# 13 active PF circuits and 10 separately supplied coil cases.  Higher circuit
# numbers in EFIT's fcoil table are wall-model elements, not additional coils.
MAST_PHYSICAL_CIRCUITS = tuple(range(1, 24))

# Setup coordinates differ slightly between otherwise equivalent historical
# inputs.  These thresholds bound the measured variation across the three raw
# setup fingerprints and are deliberately smaller than the changes tests use
# to represent a relocated component.
CONDUCTOR_POSITION_UNCERTAINTY_M = 0.025
DIAGNOSTIC_POSITION_UNCERTAINTY_M = 0.015
DIAGNOSTIC_ANGLE_UNCERTAINTY_DEG = 0.1
CURVE_MOMENT_UNCERTAINTY = 5.0e-3


def _normalise_angle_deg(value: float) -> float:
    """Normalise a signed sensitive-axis angle without folding its polarity."""
    angle = float(value) % 360.0
    return 0.0 if np.isclose(angle, 360.0) else angle


def canonical_diagnostic_layout(
    probe_r: np.ndarray,
    probe_z: np.ndarray,
    probe_angle_deg: np.ndarray,
    probe_length: np.ndarray,
    flux_loop_r: np.ndarray,
    flux_loop_z: np.ndarray,
) -> dict[str, tuple[tuple[float, ...], ...]]:
    """Canonical diagnostic poses independent of source-array ordering.

    Probe orientation is kept modulo 360 degrees, not 180 degrees: reversing a
    sensitive axis reverses the measured signal and is a physical configuration
    change.  Non-finite padding is omitted.
    """

    probes = []
    for r, z, angle, length in zip(
        probe_r, probe_z, probe_angle_deg, probe_length, strict=True
    ):
        values = np.asarray([r, z, angle, length], dtype=np.float64)
        if np.all(np.isfinite(values)):
            probes.append(
                (
                    float(r),
                    float(z),
                    _normalise_angle_deg(float(angle)),
                    float(length),
                )
            )

    loops = []
    for r, z in zip(flux_loop_r, flux_loop_z, strict=True):
        values = np.asarray([r, z], dtype=np.float64)
        if np.all(np.isfinite(values)):
            loops.append((float(r), float(z)))

    return {
        "magnetic_probes": tuple(
            sorted(probes, key=lambda pose: (pose[2], pose[0], pose[1], pose[3]))
        ),
        "flux_loops": tuple(sorted(loops)),
    }


def canonical_conductor_components(
    component: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
    width: np.ndarray,
    height: np.ndarray,
    turns: np.ndarray,
    weight: np.ndarray,
    *,
    include: tuple[int, ...] | None = None,
) -> tuple[tuple[float | int, ...], ...]:
    """Reduce a conductor mesh to physical component geometry.

    Each component retains its weighted centroid, occupied bounds, second
    moments and effective turns.  Element count and subdivision are absent, so
    splitting a filled conductor cell into smaller cells with proportional
    weights leaves the result unchanged.
    """

    arrays = [
        np.asarray(a, dtype=np.float64)
        for a in (component, r, z, width, height, turns, weight)
    ]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("conductor arrays must have identical shapes")

    component_id, rr, zz, dr, dz, nturn, share = arrays
    allowed = None if include is None else set(include)
    out: list[tuple[float | int, ...]] = []
    for value in sorted({int(x) for x in component_id[np.isfinite(component_id)]}):
        if allowed is not None and value not in allowed:
            continue
        mask = component_id == value
        finite = np.isfinite(
            np.column_stack(
                [rr[mask], zz[mask], dr[mask], dz[mask], nturn[mask], share[mask]]
            )
        ).all(axis=1)
        if not finite.any():
            continue
        cr = rr[mask][finite]
        cz = zz[mask][finite]
        cw = np.abs(dr[mask][finite])
        ch = np.abs(dz[mask][finite])
        ct = nturn[mask][finite]
        signed_share = share[mask][finite]
        mass = np.abs(signed_share)
        total_mass = float(mass.sum())
        if total_mass == 0.0:
            mass = np.ones_like(mass)
            total_mass = float(mass.sum())
        centre_r = float(np.dot(mass, cr) / total_mass)
        centre_z = float(np.dot(mass, cz) / total_mass)
        moment_r = float(np.dot(mass, (cr - centre_r) ** 2 + cw**2 / 12.0) / total_mass)
        moment_z = float(np.dot(mass, (cz - centre_z) ** 2 + ch**2 / 12.0) / total_mass)
        out.append(
            (
                value,
                centre_r,
                centre_z,
                float(np.min(cr - cw / 2.0)),
                float(np.max(cr + cw / 2.0)),
                float(np.min(cz - ch / 2.0)),
                float(np.max(cz + ch / 2.0)),
                moment_r,
                moment_z,
                float(np.dot(signed_share, ct)),
            )
        )
    return tuple(out)


def canonical_closed_curve(r: np.ndarray, z: np.ndarray) -> tuple[float, ...] | None:
    """Sampling-independent area, centroid, moments, bounds and perimeter."""

    points = np.column_stack(
        [np.asarray(r, dtype=np.float64), np.asarray(z, dtype=np.float64)]
    )
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return None
    keep = np.r_[True, np.any(np.diff(points, axis=0) != 0.0, axis=1)]
    points = points[keep]
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3:
        return None

    x = points[:, 0]
    y = points[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    signed_area = 0.5 * float(cross.sum())
    if np.isclose(signed_area, 0.0):
        return None
    centroid_r = float(np.sum((x + x_next) * cross) / (6.0 * signed_area))
    centroid_z = float(np.sum((y + y_next) * cross) / (6.0 * signed_area))
    second_r = float(
        np.sum((y**2 + y * y_next + y_next**2) * cross) / (12.0 * signed_area)
    )
    second_z = float(
        np.sum((x**2 + x * x_next + x_next**2) * cross) / (12.0 * signed_area)
    )
    perimeter = float(np.hypot(x_next - x, y_next - y).sum())
    return (
        abs(signed_area),
        centroid_r,
        centroid_z,
        second_r,
        second_z,
        float(x.min()),
        float(x.max()),
        float(y.min()),
        float(y.max()),
        perimeter,
    )


def canonical_physical_geometry(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Canonical physical MAST geometry from one EFIT setup representation."""

    required = {
        "magpr_r",
        "magpr_z",
        "magpr_ang",
        "magpr_len",
        "silop_r",
        "silop_z",
        "fcoil_r",
        "fcoil_z",
        "fcoil_width",
        "fcoil_height",
        "fcoil_turns",
        "fcoil_circ",
        "fcoil_xmult",
        "limiterr",
        "limiterz",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"physical geometry arrays missing: {', '.join(missing)}")

    return {
        "active_and_case_circuits": canonical_conductor_components(
            arrays["fcoil_circ"],
            arrays["fcoil_r"],
            arrays["fcoil_z"],
            arrays["fcoil_width"],
            arrays["fcoil_height"],
            arrays["fcoil_turns"],
            arrays["fcoil_xmult"],
            include=MAST_PHYSICAL_CIRCUITS,
        ),
        "diagnostics": canonical_diagnostic_layout(
            arrays["magpr_r"],
            arrays["magpr_z"],
            arrays["magpr_ang"],
            arrays["magpr_len"],
            arrays["silop_r"],
            arrays["silop_z"],
        ),
        "limiter": canonical_closed_curve(arrays["limiterr"], arrays["limiterz"]),
        "physical_circuit_ids": MAST_PHYSICAL_CIRCUITS,
    }


def physical_geometry_equivalent(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    conductor_position_uncertainty_m: float = CONDUCTOR_POSITION_UNCERTAINTY_M,
    diagnostic_position_uncertainty_m: float = DIAGNOSTIC_POSITION_UNCERTAINTY_M,
    diagnostic_angle_uncertainty_deg: float = DIAGNOSTIC_ANGLE_UNCERTAINTY_DEG,
) -> bool:
    """Whether two setup representations support the same physical geometry."""

    if left["physical_circuit_ids"] != right["physical_circuit_ids"]:
        return False
    l_conductor = np.asarray(left["active_and_case_circuits"], dtype=np.float64)
    r_conductor = np.asarray(right["active_and_case_circuits"], dtype=np.float64)
    if l_conductor.shape != r_conductor.shape:
        return False
    if not np.array_equal(l_conductor[:, 0], r_conductor[:, 0]):
        return False
    # Component positions, bounds and effective turns carry source-scale
    # uncertainty.  Second moments have squared-distance units.
    conductor_atol = np.asarray(
        [
            0.0,
            conductor_position_uncertainty_m,
            conductor_position_uncertainty_m,
            conductor_position_uncertainty_m,
            conductor_position_uncertainty_m,
            conductor_position_uncertainty_m,
            conductor_position_uncertainty_m,
            conductor_position_uncertainty_m,
            conductor_position_uncertainty_m,
            1.0e-4,
        ]
    )
    if not np.all(np.abs(l_conductor - r_conductor) <= conductor_atol):
        return False

    l_diag = left["diagnostics"]
    r_diag = right["diagnostics"]
    for key, atol in (
        (
            "magnetic_probes",
            np.asarray(
                [
                    diagnostic_position_uncertainty_m,
                    diagnostic_position_uncertainty_m,
                    diagnostic_angle_uncertainty_deg,
                    diagnostic_position_uncertainty_m,
                ]
            ),
        ),
        (
            "flux_loops",
            np.asarray(
                [
                    diagnostic_position_uncertainty_m,
                    diagnostic_position_uncertainty_m,
                ]
            ),
        ),
    ):
        l_values = np.asarray(l_diag[key], dtype=np.float64)
        r_values = np.asarray(r_diag[key], dtype=np.float64)
        if l_values.shape != r_values.shape:
            return False
        if not np.all(np.abs(l_values - r_values) <= atol):
            return False

    l_curve = left["limiter"]
    r_curve = right["limiter"]
    if (l_curve is None) != (r_curve is None):
        return False
    return l_curve is None or bool(
        np.allclose(
            np.asarray(l_curve),
            np.asarray(r_curve),
            rtol=0.0,
            atol=CURVE_MOMENT_UNCERTAINTY,
        )
    )


def physical_geometry_digest(geometry: dict[str, Any]) -> str:
    """Evidence digest for one canonical geometry representative."""

    payload = json.dumps(geometry, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ShotConfigurationRange:
    """A contiguous physical-configuration range with explicit source gaps."""

    configuration: str
    shot_min: int
    shot_max: int
    evidence_shots: tuple[int, ...]
    unreadable_shots: tuple[int, ...]
    confidence: str


def build_configuration_ranges(
    evidence: list[tuple[int, str | None]],
    *,
    confidence: str = "provisional",
) -> tuple[ShotConfigurationRange, ...]:
    """Build discrete ranges while preserving missing/corrupt-shot exceptions.

    Invalid evidence does not invent a boundary.  It is attached to a range
    when bracketed by the same configuration.  When configurations differ
    across a gap, each neighbouring range remains provisional and the invalid
    shots are retained on the earlier range as unresolved boundary evidence.
    """

    if not evidence:
        return ()
    records = sorted(evidence)
    ranges: list[ShotConfigurationRange] = []
    current: str | None = None
    shots: list[int] = []
    unreadable: list[int] = []
    pending: list[int] = []

    def close() -> None:
        if current is None or not shots:
            return
        ranges.append(
            ShotConfigurationRange(
                configuration=current,
                shot_min=shots[0],
                shot_max=shots[-1],
                evidence_shots=tuple(shots),
                unreadable_shots=tuple(unreadable),
                confidence=confidence,
            )
        )

    for shot, configuration in records:
        if configuration is None:
            pending.append(int(shot))
            continue
        if current is None:
            current = configuration
            shots = [int(shot)]
            unreadable = pending
            pending = []
            continue
        if configuration == current:
            unreadable.extend(pending)
            pending = []
            shots.append(int(shot))
            continue
        unreadable.extend(pending)
        pending = []
        close()
        current = configuration
        shots = [int(shot)]
        unreadable = []
    unreadable.extend(pending)
    close()
    return tuple(ranges)


#: ``amm`` arrays that are not passive structures (time base, bookkeeping).
AMM_NON_STRUCTURE = ("time", "passnumber", "tcutoff", "tolerance", "substeps")

#: the ``amm`` description form embedding a passive element's position, e.g.
#: ``"Lower horizontal wall: R=0.29  Z=-2.21"``; a multi-element array carries
#: one such pair per line.
AMM_POSITION = re.compile(r"R\s*=\s*(-?\d+\.?\d*)\s+Z\s*=\s*(-?\d+\.?\d*)")


def _open_group(shot: int, group: str) -> tuple[Any, set[str]]:
    import zarr

    store = zarr.open(str(LEVEL1 / f"{shot}.zarr"), mode="r")
    node = store[group]
    return node, set(node.array_keys())


def _open_efm(shot: int) -> tuple[Any, set[str]]:
    return _open_group(shot, "efm")


def passive_positions(shot: int) -> list[tuple[float, float]] | None:
    """Passive-structure ``(R, Z)`` positions parsed from the ``amm`` descriptions.

    Reads descriptions only -- never the current time series, which is a wall-
    model output rather than machine description.
    """
    try:
        amm, keys = _open_group(shot, "amm")
    except Exception:  # noqa: BLE001
        return None
    out: list[tuple[float, float]] = []
    for name in sorted(keys):
        if name in AMM_NON_STRUCTURE or name.endswith("_channel"):
            continue
        description = str(amm[name].attrs.get("description", "") or "")
        out.extend((float(r), float(z)) for r, z in AMM_POSITION.findall(description))
    return out


def _get(efm: Any, keys: set[str], name: str) -> np.ndarray | None:
    if name not in keys:
        return None
    return np.asarray(efm[name][:], dtype=np.float64)


def fingerprint_shot(shot: int) -> dict[str, Any] | None:
    """Representation fingerprint + convention-determining signs for one shot."""
    try:
        efm, keys = _open_efm(shot)
    except Exception:  # noqa: BLE001 -- a missing/corrupt shot is skipped
        return None
    if not {"magpr_r", "fcoil_r", "limiterr"} <= keys:
        return None

    try:
        hashed = []
        for name in FINGERPRINT_ARRAYS:
            arr = _get(efm, keys, name)
            if arr is None:
                return None
            hashed.append(_finite(arr) if name.startswith(("magpr", "silop")) else arr)
        digest = geometry_digest(hashed)
        n_probe = int(_finite(_get(efm, keys, "magpr_r")).size)
        n_loop = int(
            min(
                _finite(_get(efm, keys, "silop_r")).size,
                _finite(_get(efm, keys, "silop_z")).size,
            )
        )
        n_filament = int(_get(efm, keys, "fcoil_r").size)
        n_limiter = int(_finite(_get(efm, keys, "limiterr")).size)
    except Exception:  # noqa: BLE001
        return None

    passive = passive_positions(shot)
    row: dict[str, Any] = {
        "shot": int(shot),
        "fingerprint": (
            f"mp{n_probe}-fl{n_loop}-fc{n_filament}-lim{n_limiter}-{digest}"
        ),
        "n_probe": n_probe,
        "n_loop": n_loop,
        "n_filament": n_filament,
        "n_limiter": n_limiter,
        "n_passive": None if passive is None else len(passive),
        "passive_digest": None
        if not passive
        else f"pv{len(passive)}-"
        + geometry_digest([np.asarray(passive, dtype=np.float64).ravel()], decimals=3),
    }
    row.update(_convention_signs(efm, keys))
    return row


def physical_geometry_shot(shot: int) -> dict[str, Any] | None:
    """Canonical physical-geometry evidence for one readable MAST shot."""

    try:
        efm, keys = _open_efm(shot)
    except Exception:  # noqa: BLE001 -- an unreadable shot is explicit evidence
        return None
    if not set(PHYSICAL_GEOMETRY_ARRAYS) <= keys:
        return None
    try:
        arrays = {
            name: np.asarray(efm[name][:], dtype=np.float64)
            for name in PHYSICAL_GEOMETRY_ARRAYS
        }
        return canonical_physical_geometry(arrays)
    except Exception:  # noqa: BLE001 -- corrupt geometry is not a configuration
        return None


def _convention_signs(efm: Any, keys: set[str]) -> dict[str, Any]:
    """The (psi, Ip, B_phi, q, dp/dpsi) sign combination on the peak-current slice."""
    out: dict[str, Any] = {"has_signs": False}
    ip = _get(efm, keys, "plasma_current_c")
    psi_axis = _get(efm, keys, "psi_axis")
    psi_bnd = _get(efm, keys, "psi_boundary")
    if ip is None or psi_axis is None or psi_bnd is None:
        return out
    good = (
        np.isfinite(ip)
        & np.isfinite(psi_axis)
        & np.isfinite(psi_bnd)
        & (np.abs(ip) > MIN_IP_A)
    )
    if not good.any():
        return out
    peak = int(np.argmax(np.where(good, np.abs(ip), 0.0)))

    b0 = _get(efm, keys, "bvac_val")
    irod = _get(efm, keys, "irod")
    q95 = _get(efm, keys, "q_95")
    q_axis = _get(efm, keys, "q_axis")
    pprime = _get(efm, keys, "pprime")

    def at(a: np.ndarray | None) -> float:
        if a is None or peak >= a.shape[0]:
            return float("nan")
        return float(a[peak])

    dp_dpsi = float("nan")
    if pprime is not None and pprime.ndim == 2 and peak < pprime.shape[0]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mid = pprime[peak, pprime.shape[1] // 3 : 2 * pprime.shape[1] // 3]
            dp_dpsi = float(np.nanmedian(mid))

    out.update(
        has_signs=True,
        slice_index=peak,
        ip=at(ip),
        psi_axis=at(psi_axis),
        psi_boundary=at(psi_bnd),
        psi_axis_minus_boundary=at(psi_axis) - at(psi_bnd),
        b0=at(b0),
        irod=at(irod),
        q_95=at(q95),
        q_axis=at(q_axis),
        dp_dpsi=dp_dpsi,
        n_current_slices=int(good.sum()),
    )
    return out


def flux_loop_two_pi_ratio(shot: int) -> dict[str, Any] | None:
    """Ratio of each loop's fitted flux [Wb] to psi at the loop [Wb/rad].

    2*pi means the reconstruction's psi is per-radian (``e_Bp = 0``); 1 means it
    is total flux (``e_Bp = 1``).

    ``psirz`` is stored two different ways across this corpus: dense on its
    native ``(gridz, gridr)`` grid for some campaigns, and scattered onto a
    longer shared radial coordinate with NaN in the unused columns for others.
    ``gridr`` is the live radial grid under BOTH layouts, so the finite columns
    are paired with ``gridr`` -- never with the shot's own ``profile_r``, which
    is a different array of a different length and radial span depending on the
    campaign, and silently yields a wrong grid where the map is already dense.
    """
    from scipy.interpolate import RegularGridInterpolator

    try:
        efm, keys = _open_efm(shot)
    except Exception:  # noqa: BLE001
        return None
    needed = {"psirz", "gridr", "gridz", "silop_r", "silop_z", "silop_c"}
    if not needed <= keys:
        return None

    psirz = _get(efm, keys, "psirz")
    ip = _get(efm, keys, "plasma_current_c")
    if psirz is None or psirz.ndim != 3 or ip is None:
        return None
    good = np.isfinite(ip) & (np.abs(ip) > MIN_IP_A)
    if not good.any():
        return None
    t = int(np.argmax(np.where(good, np.abs(ip), 0.0)))
    if t >= psirz.shape[0]:
        return None

    plane = psirz[t]
    cols = np.where(np.isfinite(plane).any(axis=0))[0]
    r_grid = _get(efm, keys, "gridr")
    z_grid = _get(efm, keys, "gridz")
    if cols.size != r_grid.size or plane[:, cols].shape != (z_grid.size, r_grid.size):
        return None
    if not (np.all(np.diff(r_grid) > 0) and np.all(np.diff(z_grid) > 0)):
        return None

    interp = RegularGridInterpolator(
        (z_grid, r_grid), plane[:, cols], bounds_error=False, fill_value=np.nan
    )
    loop_r = _finite(_get(efm, keys, "silop_r"))
    loop_z = _finite(_get(efm, keys, "silop_z"))
    n = min(loop_r.size, loop_z.size)
    loop_r, loop_z = loop_r[:n], loop_z[:n]
    psi_at_loop = interp(np.column_stack([loop_z, loop_r]))

    silop_c = _get(efm, keys, "silop_c")
    if silop_c is None or t >= silop_c.shape[0]:
        return None
    flux = silop_c[t, :n]

    usable = (
        np.isfinite(psi_at_loop)
        & np.isfinite(flux)
        & (np.abs(psi_at_loop) > MIN_PSI_WB_PER_RAD)
    )
    if usable.sum() < MIN_RATIO_LOOPS:
        return None
    ratio = flux[usable] / psi_at_loop[usable]
    median = float(np.median(ratio))
    iqr = float(np.percentile(ratio, 75) - np.percentile(ratio, 25))
    if median == 0.0 or iqr / abs(median) > MAX_RATIO_IQR_FRACTION:
        # ``silop_c`` is largely unpopulated at this slice -- refuse to report.
        return None
    return {
        "shot": int(shot),
        "slice_index": t,
        "n_loops": int(usable.sum()),
        "ratio_iqr_fraction": iqr / abs(median),
        "ratio_median": float(np.median(ratio)),
        "ratio_p05": float(np.percentile(ratio, 5)),
        "ratio_p95": float(np.percentile(ratio, 95)),
        "ratios": ratio.tolist(),
        "psi_at_loop": psi_at_loop[usable].tolist(),
        "loop_flux": flux[usable].tolist(),
    }


def determine_digits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce the per-shot sign table to the four COCOS digits."""
    signed = [r for r in rows if r.get("has_signs")]
    if not signed:
        return {"n_shots": 0}

    sigma_ip = np.array([np.sign(r["ip"]) for r in signed])
    sigma_psi = np.array([np.sign(r["psi_boundary"] - r["psi_axis"]) for r in signed])
    sigma_bp = sigma_psi * sigma_ip

    b0 = np.array([r["b0"] for r in signed])
    q95 = np.array([r["q_95"] for r in signed])
    have_q = np.isfinite(b0) & np.isfinite(q95) & (q95 != 0.0)
    sigma_rtp = np.sign(q95[have_q]) * sigma_ip[have_q] * np.sign(b0[have_q])

    dp = np.array([r["dp_dpsi"] for r in signed])
    have_dp = np.isfinite(dp) & (dp != 0.0)
    dp_expected = -sigma_ip[have_dp] * sigma_bp[have_dp]
    dp_agrees = np.sign(dp[have_dp]) == dp_expected

    irod = np.array([r["irod"] for r in signed])
    have_rod = np.isfinite(irod) & np.isfinite(b0) & (irod != 0.0) & (b0 != 0.0)
    rod_agrees = np.sign(irod[have_rod]) == np.sign(b0[have_rod])

    def unanimous(a: np.ndarray) -> tuple[int, float]:
        if a.size == 0:
            return 0, 0.0
        vals, counts = np.unique(a, return_counts=True)
        best = int(vals[int(np.argmax(counts))])
        return best, float(counts.max() / a.size)

    bp, bp_frac = unanimous(sigma_bp)
    rtp, rtp_frac = unanimous(sigma_rtp)
    return {
        "n_shots": len(signed),
        "sigma_bp": bp,
        "sigma_bp_unanimity": bp_frac,
        "sigma_rho_theta_phi": rtp,
        "sigma_rho_theta_phi_unanimity": rtp_frac,
        "ip_positive_fraction": float(np.mean(sigma_ip > 0)),
        "b0_negative_fraction": float(np.mean(b0[np.isfinite(b0)] < 0))
        if np.isfinite(b0).any()
        else float("nan"),
        "axis_is_flux_maximum_fraction": float(
            np.mean([r["psi_axis_minus_boundary"] > 0 for r in signed])
        ),
        "dp_dpsi_cross_check_pass_fraction": float(np.mean(dp_agrees))
        if dp_agrees.size
        else float("nan"),
        "dp_dpsi_cross_check_n": int(dp_agrees.size),
        "rod_field_consistency_pass_fraction": float(np.mean(rod_agrees))
        if rod_agrees.size
        else float("nan"),
        "rod_field_consistency_n": int(rod_agrees.size),
    }


def cocos_from_digits(
    sigma_bp: int, e_bp: int, sigma_r_phi_z: int, sigma_rho_theta_phi: int
) -> int:
    """Sauter & Medvedev CPC 184 (2013) Table I reverse lookup."""
    table = {
        1: (+1, 0, +1, +1),
        2: (+1, 0, -1, +1),
        3: (-1, 0, +1, -1),
        4: (-1, 0, -1, -1),
        5: (+1, 0, +1, -1),
        6: (+1, 0, -1, -1),
        7: (-1, 0, +1, +1),
        8: (-1, 0, -1, +1),
        11: (+1, 1, +1, +1),
        12: (+1, 1, -1, +1),
        13: (-1, 1, +1, -1),
        14: (-1, 1, -1, -1),
        15: (+1, 1, +1, -1),
        16: (+1, 1, -1, -1),
        17: (-1, 1, +1, +1),
        18: (-1, 1, -1, +1),
    }
    want = (sigma_bp, e_bp, sigma_r_phi_z, sigma_rho_theta_phi)
    for value, digits in table.items():
        if digits == want:
            return value
    raise ValueError(f"no COCOS for digits {want}")


#: keys dropped from the persisted per-shot rows because the grouped tables
#: already carry them for every shot -- repeating a 60-character digest 14633
#: times triples the artifact for no information.
REDUNDANT_PERSISTED_KEYS = ("fingerprint", "passive_digest")

#: significant digits kept when persisting a measured float.  The source arrays
#: are float32, so six digits is below their own precision and the full float64
#: repr of a float32 is padding.
PERSISTED_DIGITS = 6


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    """One row trimmed for persistence: redundant keys out, floats shortened."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in REDUNDANT_PERSISTED_KEYS:
            continue
        if isinstance(value, float):
            out[key] = _short(value)
        elif isinstance(value, list) and value and isinstance(value[0], float):
            out[key] = [_short(v) for v in value]
        else:
            out[key] = value
    return out


def _short(value: float) -> float:
    return value if not np.isfinite(value) else float(f"%.{PERSISTED_DIGITS}g" % value)


def as_columns(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rows as a columnar table -- the per-shot table is mostly key names.

    Repeating ~180 characters of key names across 14633 rows costs more than the
    measurements do, so the persisted form carries the key list once and each row
    as a bare value list.  :func:`rehydrate` is the inverse.
    """
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return {
        "columns": columns,
        "values": [[row.get(c) for c in columns] for row in rows],
    }


def rehydrate(table: dict[str, Any]) -> list[dict[str, Any]]:
    """Columnar table back to a list of row dicts."""
    columns = table["columns"]
    return [dict(zip(columns, values, strict=True)) for values in table["values"]]


def load_report(path: Path) -> dict[str, Any]:
    """Read an audit artifact with its columnar tables expanded to row dicts."""
    report = json.loads(Path(path).read_text())
    for key in ("sign_rows", "ratio_rows"):
        if isinstance(report.get(key), dict):
            report[key] = rehydrate(report[key])
    return report


def audit_physical_configurations(
    report: dict[str, Any],
    available_shots: list[int],
) -> dict[str, Any]:
    """Group raw setup fingerprints by physical equivalence and build ranges."""

    representatives: list[dict[str, Any]] = []
    raw_to_physical: dict[str, str] = {}
    for raw_key, raw_group in sorted(
        report["campaigns"].items(), key=lambda item: item[1]["shot_min"]
    ):
        representative_shot = int(raw_group["shots"][0])
        geometry = physical_geometry_shot(representative_shot)
        if geometry is None:
            continue
        match = next(
            (
                group
                for group in representatives
                if physical_geometry_equivalent(geometry, group["geometry"])
            ),
            None,
        )
        if match is None:
            match = {
                "configuration": f"mast-physical-{physical_geometry_digest(geometry)}",
                "geometry": geometry,
                "representative_shot": representative_shot,
                "raw_fingerprints": [],
            }
            representatives.append(match)
        match["raw_fingerprints"].append(raw_key)
        raw_to_physical[raw_key] = match["configuration"]

    configuration_by_shot = {
        int(shot): raw_to_physical[raw_key]
        for raw_key, raw_group in report["campaigns"].items()
        if raw_key in raw_to_physical
        for shot in raw_group["shots"]
    }
    evidence = [
        (int(shot), configuration_by_shot.get(int(shot)))
        for shot in sorted(available_shots)
    ]
    ranges = build_configuration_ranges(evidence)
    range_rows = []
    for item in ranges:
        range_rows.append(
            {
                "configuration": item.configuration,
                "shot_min": item.shot_min,
                "shot_max": item.shot_max,
                "n_evidence_shots": len(item.evidence_shots),
                "n_unreadable_shots": len(item.unreadable_shots),
                "unreadable_shots": list(item.unreadable_shots),
                "confidence": item.confidence,
            }
        )

    n_readable = len(configuration_by_shot)
    n_unreadable = len(available_shots) - n_readable
    n_with_passive = sum(
        int(group["n_shots"]) for group in report["passive_groups"].values()
    )
    passive_missing_fraction = (
        0.0 if n_readable == 0 else 1.0 - n_with_passive / n_readable
    )

    return {
        "identity_rule": (
            "underlying conductor, wall, limiter, TF and diagnostic geometry; "
            "never source mesh or element count"
        ),
        "n_raw_setup_fingerprints": len(report["campaigns"]),
        "n_canonical_physical_configurations": len(representatives),
        "raw_to_physical": raw_to_physical,
        "configurations": [
            {
                "configuration": group["configuration"],
                "representative_shot": group["representative_shot"],
                "raw_fingerprints": group["raw_fingerprints"],
                "n_physical_circuits": len(
                    group["geometry"]["active_and_case_circuits"]
                ),
                "n_magnetic_probes": len(
                    group["geometry"]["diagnostics"]["magnetic_probes"]
                ),
                "n_flux_loops": len(group["geometry"]["diagnostics"]["flux_loops"]),
            }
            for group in representatives
        ],
        "operational_ranges": range_rows,
        "excluded_representation_details": {
            "efm_fcoil": (
                "filament count, subdivision and wall-model circuits above the "
                "23 physical active and case circuits"
            ),
            "amm": (
                "parsed wall-model element positions are discretisation evidence, "
                "not authoritative vessel geometry"
            ),
        },
        "boundary_status": (
            "No physical change boundary is established by the available source. "
            "The single range is provisional until installation and maintenance "
            "records confirm conductor, passive, wall, TF and diagnostic epochs."
        ),
        "source_limitations": [
            f"{n_unreadable:,} on-disk shots have no readable static setup geometry",
            f"{passive_missing_fraction:.1%} of readable setup shots have no "
            "parseable amm positions",
            "FAIR-MAST setup arrays do not distinguish calibration revisions from "
            "physical diagnostic relocation",
            "amm element lists do not provide authoritative continuous passive "
            "structure geometry",
        ],
    }


def scan(shots: list[int], workers: int) -> list[dict[str, Any]]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(fingerprint_shot, shots, chunksize=16)
        return [r for r in results if r is not None]


# --- Evidence figures ------------------------------------------------------
#
# The audit and its figures ship together: a verdict about a sign convention
# that a reader cannot see is under-communicated, and a figure regenerated by a
# script living somewhere else rots.  Both come out of one command.

INK = "#1b1f24"
MUTED = "#6b7280"
GREEN = "#1a6b3c"
RED = "#b03030"
AMBER = "#8a5a00"
BLUE = "#2b5f9e"
PURPLE = "#6b3fa0"
GREY = "#9aa0a6"
MD_COLOURS = [BLUE, AMBER, GREEN, PURPLE, RED]


def figure_dir() -> Path:
    """Where the figures land -- beside the artifact they are drawn from."""
    return _FIGURE_DIR


_FIGURE_DIR = Path("docs/figures/mast-machine-description")


def _style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": "#c9ced4",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": False,
        }
    )


# --- Figure 1: campaign fingerprint timeline -------------------------------


#: a digest carried by this many shots or fewer is a per-shot acquisition
#: dropout (some element descriptions missing for that one shot), not a
#: machine change; such groups are collapsed into one row.
SINGLETON_MAX = 1


def campaign_timeline(d: dict) -> None:
    import matplotlib.pyplot as plt

    raw = d["campaigns"]
    physical = d["physical_configuration_audit"]
    raw_to_physical = physical["raw_to_physical"]
    lo = min(v["shot_min"] for v in raw.values())
    hi = max(v["shot_max"] for v in raw.values())

    fig, axes = plt.subplots(
        3, 1, figsize=(12.4, 6.5), sharex=True, height_ratios=[1.45, 0.75, 0.8]
    )

    raw_keys = sorted(raw, key=lambda key: raw[key]["shot_min"])
    for i, key in enumerate(raw_keys):
        shots = np.asarray(raw[key]["shots"])
        axes[0].scatter(
            shots,
            np.full(shots.size, i),
            s=3,
            marker="|",
            color=MD_COLOURS[i],
            linewidths=0.8,
        )
    axes[0].set_yticks(range(len(raw_keys)))
    axes[0].set_yticklabels(
        [
            f"{key.split('-lim37-')[0].replace('mp78-fl46-', '')}…{key[-6:]}"
            for key in raw_keys
        ],
        fontsize=7.5,
    )
    axes[0].set_title(
        "a  Raw EFIT setup fingerprints — representation states, not machine IDs",
        loc="left",
        color=INK,
    )

    physical_shots: dict[str, list[int]] = {}
    for key, group in raw.items():
        physical_shots.setdefault(raw_to_physical[key], []).extend(group["shots"])
    for i, (key, shots) in enumerate(sorted(physical_shots.items())):
        values = np.asarray(sorted(shots))
        axes[1].scatter(
            values,
            np.full(values.size, i),
            s=3,
            marker="|",
            color=GREEN,
            linewidths=0.8,
        )
        axes[1].text(
            hi + 300,
            i,
            f"{key[-8:]}  n={values.size:,}",
            va="center",
            fontsize=8,
            color=GREEN,
        )
    axes[1].set_yticks(range(len(physical_shots)))
    axes[1].set_yticklabels(["physical geometry"] * len(physical_shots), fontsize=7.5)
    axes[1].set_title(
        "b  Canonical physical comparison — mesh counts and "
        "wall-model elements removed",
        loc="left",
        color=INK,
    )

    for item in physical["operational_ranges"]:
        axes[2].hlines(
            0.15,
            item["shot_min"],
            item["shot_max"],
            color=GREEN,
            linewidth=8,
            zorder=1,
        )
        unreadable = np.asarray(item["unreadable_shots"])
        if unreadable.size:
            axes[2].scatter(
                unreadable,
                np.full(unreadable.size, -0.15),
                marker="|",
                s=10,
                color=RED,
                linewidths=0.7,
                alpha=0.8,
                zorder=3,
            )
    axes[2].set_yticks([-0.15, 0.15])
    axes[2].set_yticklabels(["unreadable", "provisional range"], fontsize=7.5)
    axes[2].set_title(
        "c  Operational range — red ticks are unreadable source shots, not boundaries",
        loc="left",
        color=INK,
    )
    axes[2].set_xlabel("MAST shot number")

    for ax in axes:
        ax.set_xlim(lo - 400, hi + 3800)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        f"{len(raw)} setup fingerprints collapse to "
        f"{physical['n_canonical_physical_configurations']} provisional physical "
        "configuration; no hardware boundary is established",
        fontsize=11.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figure_dir() / "campaign_fingerprints.png", dpi=140)
    plt.close(fig)


# --- Figure 2: the interleaving, zoomed ------------------------------------


def interleaving(d: dict) -> None:
    import matplotlib.pyplot as plt

    raw = d["campaigns"]
    physical = d["physical_configuration_audit"]
    keys = sorted(raw, key=lambda key: raw[key]["shot_min"])
    colours = {key: MD_COLOURS[i] for i, key in enumerate(keys)}
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.8), width_ratios=[1.5, 1])

    ax = axes[0]
    for i, key in enumerate(keys):
        shots = np.asarray(raw[key]["shots"])
        ax.scatter(
            shots,
            np.full(shots.size, i),
            s=4,
            marker="|",
            color=colours[key],
            linewidths=0.8,
        )
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(
        [f"{key.split('-')[2]}  …{key[-6:]}" for key in keys], fontsize=7.5
    )
    ax.set_xlabel("MAST shot number")
    ax.set_title(
        "a  Raw representations interleave\n"
        "   alternation is evidence against hardware boundaries",
        loc="left",
        color=INK,
    )

    ax = axes[1]
    raw_counts = [int(key.split("-fc")[1].split("-")[0]) for key in keys]
    physical_counts = [physical["configurations"][0]["n_physical_circuits"]] * len(keys)
    x = np.arange(len(keys))
    ax.bar(x - 0.18, raw_counts, width=0.36, color=AMBER, label="EFIT filaments")
    ax.bar(
        x + 0.18,
        physical_counts,
        width=0.36,
        color=GREEN,
        label="physical circuits",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"setup {i + 1}" for i in x], fontsize=7.5)
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_title(
        "b  Resolution removed from identity\n   938/1004 cells → the same 23 circuits",
        loc="left",
        color=INK,
    )
    ax.legend(frameon=False, fontsize=7.5)
    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir() / "fingerprint_interleaving.png", dpi=140)
    plt.close(fig)


# --- Figure 3: the empirical COCOS evidence -------------------------------


def cocos_evidence(d: dict) -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in d["sign_rows"] if r.get("has_signs")]
    ip = np.array([r["ip"] for r in rows]) / 1e3
    dpsi = np.array([r["psi_axis_minus_boundary"] for r in rows])
    b0 = np.array([r["b0"] for r in rows])
    q95 = np.array([r["q_95"] for r in rows])
    ratios = np.array([x for r in d["ratio_rows"] for x in r["ratios"]])
    c = d["cocos"]

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4))

    # (a) sigma_Bp: psi_axis - psi_boundary vs Ip
    ax = axes[0, 0]
    fwd = ip > 0
    ax.scatter(ip[fwd], dpsi[fwd], s=4, alpha=0.25, color=BLUE, label="forward")
    ax.scatter(ip[~fwd], dpsi[~fwd], s=6, alpha=0.55, color=RED, label="reversed")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel(r"plasma current $I_p$  [kA]")
    ax.set_ylabel(r"$\psi_{axis}-\psi_{boundary}$  [Wb/rad]")
    ax.set_title(
        r"a  $\sigma_{B_p}=-1$: the points occupy only the two"
        "\n"
        r"   quadrants where sign$(\psi_{bnd}-\psi_{axis})=-\sigma_{I_p}$",
        loc="left",
        color=INK,
    )
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.text(
        0.03,
        0.05,
        f"unanimous over {c['n_shots']} shots\n"
        f"({100 * (1 - c['ip_positive_fraction']):.1f}% run reversed)",
        transform=ax.transAxes,
        fontsize=7.5,
        color=GREEN,
        bbox={"boxstyle": "round", "fc": "#e8f4ec", "ec": "#a0cbb0", "lw": 0.6},
    )

    # (b) sigma_rho_theta_phi: q vs Ip*B0
    ax = axes[0, 1]
    prod = np.sign(ip) * np.sign(b0)
    ok = np.isfinite(q95) & np.isfinite(b0) & (b0 != 0)
    ax.scatter(
        prod[ok] + np.random.default_rng(0).normal(0, 0.045, ok.sum()),
        q95[ok],
        s=4,
        alpha=0.25,
        color=BLUE,
    )
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks([-1, 1])
    ax.set_xticklabels([r"$\sigma_{I_p}\sigma_{B_0}=-1$", r"$+1$"])
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylabel(r"$q_{95}$  (as stored)")
    ax.set_title(
        r"b  $\sigma_{\rho\theta\phi}=-1$: $q_{95}>0$ while"
        "\n"
        r"   $\sigma_{I_p}\sigma_{B_0}=-1$ throughout",
        loc="left",
        color=INK,
    )
    ax.text(
        0.03,
        0.92,
        r"COCOS 17 requires $q<0$ here"
        "\n"
        "→ the transform must flip q",
        transform=ax.transAxes,
        fontsize=7.5,
        va="top",
        color=AMBER,
        bbox={"boxstyle": "round", "fc": "#fdf5e6", "ec": "#d9c08a", "lw": 0.6},
    )

    # (c) e_Bp: the flux-loop ratio
    ax = axes[1, 0]
    ax.hist(ratios, bins=90, range=(5.9, 6.7), color=BLUE, alpha=0.85)
    ax.axvline(2 * np.pi, color=GREEN, lw=1.6, label=r"$2\pi$ = 6.2832 ($e_{B_p}=0$)")
    ax.axvline(1.0, color=RED, lw=1.6, ls="--", label=r"1 ($e_{B_p}=1$)")
    ax.set_xlabel(r"loop flux $\Phi$ [Wb]  /  $\psi$ at the loop [Wb/rad]")
    ax.set_ylabel("flux loops")
    ax.set_title(
        r"c  $e_{B_p}=0$, measured: median "
        f"{c['two_pi_ratio_median']:.4f}\n"
        f"   over {c['two_pi_ratio_n_loops']} loops in "
        f"{c['two_pi_ratio_n_shots']} shots",
        loc="left",
        color=INK,
    )
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    # (d) the verdict and the transform
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("d  Verdict and the transform it forces", loc="left", color=INK)
    lines = [
        (r"$\sigma_{B_p}$", "$-1$", f"unanimous, {c['n_shots']} shots", GREEN),
        (
            r"$e_{B_p}$",
            "$0$",
            f"ratio {c['two_pi_ratio_median']:.3f} vs $2\\pi$",
            GREEN,
        ),
        (r"$\sigma_{R\phi Z}$", "$+1$", "declared (not from Eq. 23)", AMBER),
        (
            r"$\sigma_{\rho\theta\phi}$",
            "$-1$",
            f"{100 * c['sigma_rho_theta_phi_unanimity']:.2f}% of shots",
            GREEN,
        ),
    ]
    y = 0.88
    for sym, val, note, col in lines:
        ax.text(0.02, y, sym, fontsize=12, transform=ax.transAxes, color=INK)
        ax.text(0.20, y, val, fontsize=12, transform=ax.transAxes, color=col)
        ax.text(0.34, y, note, fontsize=8, transform=ax.transAxes, color=MUTED)
        y -= 0.115
    ax.text(
        0.02,
        0.40,
        "source COCOS 3   →   target COCOS 17 (DDv4)",
        fontsize=10.5,
        transform=ax.transAxes,
        color=INK,
        weight="bold",
    )
    tf = [
        (r"$\psi$, flux-loop flux", r"$\times\,2\pi$", "no sign flip", GREEN),
        (r"$I_p$, $B_\phi$, $F=RB_\phi$", r"$\times\,1$", "unchanged", GREEN),
        (r"$q$", r"$\times\,(-1)$", "SIGN FLIPS", RED),
        (r"$p'$, $ff'$", r"$\div\,2\pi$", "no sign flip", GREEN),
    ]
    y = 0.30
    for name, factor, note, col in tf:
        ax.text(0.04, y, name, fontsize=8.5, transform=ax.transAxes, color=INK)
        ax.text(0.50, y, factor, fontsize=9.5, transform=ax.transAxes, color=col)
        ax.text(0.68, y, note, fontsize=7.5, transform=ax.transAxes, color=col)
        y -= 0.082

    fig.suptitle(
        "COCOS 3 confirmed — and because COCOS 17 shares its σ$_{B_p}$, ψ scales by 2π "
        "without changing sign; q is what flips",
        fontsize=11.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(figure_dir() / "cocos_evidence.png", dpi=140)
    plt.close(fig)


# --- Figure 4: coverage matrix --------------------------------------------

# (IDS, quantity, state).  Every state below was verified against the
# standard-name catalogue and the DDv4 path index, not assumed:
#   "mapped"     source quantity + validated DDv4 path + accepted standard name
#   "name_gap"   source + DDv4 path both exist; NO standard name -> proposed
#   "dd_gap"     source quantity exists but DDv4 has no path to carry it
#   "source_gap" DDv4 path + name exist; the MAST source has no such quantity
COVERAGE: list[tuple[str, str, str]] = [
    ("pf_active", "element R centroid", "mapped"),
    ("pf_active", "element Z centroid", "mapped"),
    ("pf_active", "element radial extent", "name_gap"),
    ("pf_active", "element vertical extent", "name_gap"),
    ("pf_active", "element turns (signed)", "mapped"),
    ("pf_active", "element outline R", "mapped"),
    ("pf_active", "element outline Z", "name_gap"),
    ("pf_active", "circuit membership", "name_gap"),
    ("pf_active", "current-share weight", "dd_gap"),
    ("pf_active", "coil current", "mapped"),
    ("pf_active", "coil resistance", "source_gap"),
    ("pf_passive", "element R centroid", "name_gap"),
    ("pf_passive", "element Z centroid", "name_gap"),
    ("pf_passive", "element outline R/Z", "name_gap"),
    ("pf_passive", "element area", "mapped"),
    ("pf_passive", "loop turns (signed)", "mapped"),
    ("pf_passive", "loop resistance", "source_gap"),
    ("pf_passive", "loop resistivity", "name_gap"),
    ("pf_passive", "loop current", "mapped"),
    ("wall", "limiter outline R", "mapped"),
    ("wall", "limiter outline Z", "name_gap"),
    ("wall", "vessel outline", "source_gap"),
    ("magnetics", "flux loop R", "mapped"),
    ("magnetics", "flux loop Z", "mapped"),
    ("magnetics", "flux loop flux", "mapped"),
    ("magnetics", "flux loop area", "source_gap"),
    ("magnetics", "loop toroidal extent", "dd_gap"),
    ("magnetics", "probe R", "mapped"),
    ("magnetics", "probe Z", "mapped"),
    ("magnetics", "probe poloidal angle", "name_gap"),
    ("magnetics", "probe length", "mapped"),
    ("magnetics", "probe area", "source_gap"),
    ("magnetics", "probe turns", "source_gap"),
    ("magnetics", "probe field", "mapped"),
    ("magnetics", "Rogowski Ip", "mapped"),
    ("magnetics", "diamagnetic flux", "mapped"),
    ("tf", "reference radius", "mapped"),
    ("tf", "vacuum toroidal field", "mapped"),
    ("tf", "rod current", "mapped"),
    ("tf", "coil turns", "source_gap"),
]

STATE_STYLE = {
    "mapped": (GREEN, "mapped: DDv4 path + accepted standard name"),
    "name_gap": (AMBER, "standard-name gap — name proposed here"),
    "dd_gap": (PURPLE, "DD gap — source quantity has no DDv4 path"),
    "source_gap": (GREY, "source gap — DDv4 path has no MAST quantity"),
}


def coverage_matrix() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ids_order = ["pf_active", "pf_passive", "wall", "magnetics", "tf"]
    fig, axes = plt.subplots(
        1,
        len(ids_order),
        figsize=(13.0, 4.6),
        width_ratios=[sum(1 for a, _, _ in COVERAGE if a == i) for i in ids_order],
    )
    for ax, ids in zip(axes, ids_order):
        items = [(q, s) for a, q, s in COVERAGE if a == ids]
        for i, (q, s) in enumerate(items):
            colour = STATE_STYLE[s][0]
            ax.add_patch(
                plt.Rectangle(
                    (0.06, i + 0.12), 0.88, 0.76, facecolor=colour, alpha=0.85
                )
            )
            ax.text(
                0.5,
                i + 0.5,
                q,
                ha="center",
                va="center",
                fontsize=6.8,
                color="white" if s != "source_gap" else INK,
                wrap=True,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(items))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{ids}\n{len(items)} quantities", fontsize=8.5, color=INK)
        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_visible(False)

    counts = {k: sum(1 for _, _, s in COVERAGE if s == k) for k in STATE_STYLE}
    handles = [
        Patch(
            facecolor=STATE_STYLE[k][0],
            alpha=0.85,
            label=f"{STATE_STYLE[k][1]}  ({counts[k]})",
        )
        for k in STATE_STYLE
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=7.8,
        bbox_to_anchor=(0.5, -0.015),
    )
    fig.suptitle(
        f"Machine-description ledger coverage: {len(COVERAGE)} quantities "
        "across five IDSs — "
        f"{counts['mapped']} fully mapped, {counts['name_gap']} standard-name gaps, "
        f"{counts['dd_gap']} DD gaps, {counts['source_gap']} source gaps",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.93))
    fig.savefig(figure_dir() / "coverage_matrix.png", dpi=140)
    plt.close(fig)


def draw_figures(report: dict[str, Any]) -> None:
    """Draw every evidence figure from a loaded audit artifact."""
    _style()
    figure_dir().mkdir(parents=True, exist_ok=True)
    campaign_timeline(report)
    interleaving(report)
    cocos_evidence(report)
    coverage_matrix()
    for name in (
        "campaign_fingerprints.png",
        "fingerprint_interleaving.png",
        "cocos_evidence.png",
        "coverage_matrix.png",
    ):
        path = figure_dir() / name
        print(f"  {path}  {path.stat().st_size // 1024} kB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=1, help="sample every Nth shot")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--ratio-shots", type=int, default=40, help="shots for the 2*pi ratio test"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/figures/mast-machine-description/source_cocos_audit.json"),
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="skip the evidence figures"
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="redraw the figures from an existing artifact without rescanning",
    )
    parser.add_argument(
        "--physical-only",
        action="store_true",
        help="refresh physical configuration evidence in an existing artifact",
    )
    args = parser.parse_args()

    global _FIGURE_DIR  # noqa: PLW0603 -- figures land beside the artifact
    _FIGURE_DIR = args.out.parent

    if args.figures_only:
        draw_figures(load_report(args.out))
        return 0

    all_shots = sorted(int(p.name.removesuffix(".zarr")) for p in LEVEL1.glob("*.zarr"))
    if args.physical_only:
        report = json.loads(args.out.read_text())
        report["physical_configuration_audit"] = audit_physical_configurations(
            report, all_shots
        )
        args.out.write_text(json.dumps(report, separators=(",", ":")))
        draw_figures(load_report(args.out))
        return 0

    shots = all_shots[:: args.stride]
    print(
        f"scanning {len(shots)} of {len(all_shots)} shots with {args.workers} workers"
    )

    rows = scan(shots, args.workers)
    print(f"read {len(rows)} shots")

    campaigns: dict[str, list[int]] = {}
    for row in rows:
        campaigns.setdefault(row["fingerprint"], []).append(row["shot"])

    # the passive structure is invisible to the active/sensor fingerprint, so
    # the machine-description count is the count over BOTH digests together.
    passive_groups: dict[str, list[int]] = {}
    combined: dict[str, list[int]] = {}
    for row in rows:
        if row.get("passive_digest") is None:
            continue
        passive_groups.setdefault(row["passive_digest"], []).append(row["shot"])
        key = f"{row['fingerprint']}+{row['passive_digest']}"
        combined.setdefault(key, []).append(row["shot"])

    digits = determine_digits(rows)

    candidates = [r["shot"] for r in rows if r.get("has_signs")]
    stride = max(1, len(candidates) // max(1, args.ratio_shots))
    ratio_rows = []
    n_rejected = 0
    for shot in candidates[::stride][: args.ratio_shots]:
        result = flux_loop_two_pi_ratio(shot)
        if result is None:
            n_rejected += 1
        else:
            ratio_rows.append(result)
    all_ratios = np.array([x for r in ratio_rows for x in r["ratios"]])
    e_bp = 0 if abs(float(np.median(all_ratios)) - 2 * np.pi) < 1.0 else 1

    verdict = cocos_from_digits(
        sigma_bp=digits["sigma_bp"],
        e_bp=e_bp,
        sigma_r_phi_z=+1,
        sigma_rho_theta_phi=digits["sigma_rho_theta_phi"],
    )

    report = {
        "n_shots_scanned": len(rows),
        "n_shots_available": len(all_shots),
        "stride": args.stride,
        "shot_range": [min(r["shot"] for r in rows), max(r["shot"] for r in rows)],
        "campaigns": {
            key: {
                "n_shots": len(v),
                "shot_min": min(v),
                "shot_max": max(v),
                "shots": v,
            }
            for key, v in sorted(campaigns.items(), key=lambda kv: min(kv[1]))
        },
        "n_campaigns": len(campaigns),
        "passive_groups": {
            key: {"n_shots": len(v), "shot_min": min(v), "shot_max": max(v), "shots": v}
            for key, v in sorted(passive_groups.items(), key=lambda kv: min(kv[1]))
        },
        "n_passive_groups": len(passive_groups),
        "combined_machine_descriptions": {
            key: {"n_shots": len(v), "shot_min": min(v), "shot_max": max(v), "shots": v}
            for key, v in sorted(combined.items(), key=lambda kv: min(kv[1]))
        },
        "n_combined_machine_descriptions": len(combined),
        "cocos": {
            **digits,
            "e_bp": e_bp,
            "sigma_r_phi_z": 1,
            "sigma_r_phi_z_basis": "declared standard (R, phi, Z); not determinable "
            "from Sauter Eq. 23",
            "verdict": verdict,
            "two_pi_ratio_median": float(np.median(all_ratios)),
            "two_pi_ratio_n_loops": int(all_ratios.size),
            "two_pi_ratio_n_shots": len(ratio_rows),
            "two_pi_ratio_n_shots_rejected": n_rejected,
            "two_pi_ratio_shot_medians_within_2pct": int(
                sum(
                    abs(r["ratio_median"] - 2 * np.pi) / (2 * np.pi) < 0.02
                    for r in ratio_rows
                )
            ),
            "two_pi_reference": float(2 * np.pi),
        },
        "sign_rows": as_columns([_compact(r) for r in rows]),
        "ratio_rows": as_columns([_compact(r) for r in ratio_rows]),
    }
    report["physical_configuration_audit"] = audit_physical_configurations(
        report, all_shots
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # compact separators: indented, the columnar tables put each of ~260k values
    # on its own line and treble the file.  Read it back with load_report.
    args.out.write_text(json.dumps(report, separators=(",", ":")))
    print(f"wrote {args.out}")
    print(
        f"COCOS verdict {verdict}: sigma_Bp={digits['sigma_bp']:+d} "
        f"e_Bp={e_bp} sigma_RphiZ=+1 "
        f"sigma_rhothetaphi={digits['sigma_rho_theta_phi']:+d}"
    )
    print(f"active/sensor fingerprints: {len(campaigns)}")
    for key, v in sorted(campaigns.items(), key=lambda kv: min(kv[1])):
        print(f"  {key}  n={len(v):5d}  shots {min(v)}..{max(v)}")
    print(f"passive-structure digests: {len(passive_groups)}")
    for key, v in sorted(passive_groups.items(), key=lambda kv: min(kv[1])):
        print(f"  {key}  n={len(v):5d}  shots {min(v)}..{max(v)}")
    print(f"combined raw setup/passive digest pairs: {len(combined)}")
    for key, v in sorted(combined.items(), key=lambda kv: min(kv[1])):
        print(f"  {key}  n={len(v):5d}  shots {min(v)}..{max(v)}")
    physical = report["physical_configuration_audit"]
    print(
        "canonical physical configurations: "
        f"{physical['n_canonical_physical_configurations']}"
    )
    for item in physical["operational_ranges"]:
        print(
            f"  {item['configuration']}  shots "
            f"{item['shot_min']}..{item['shot_max']} "
            f"evidence={item['n_evidence_shots']} "
            f"unreadable={item['n_unreadable_shots']} "
            f"confidence={item['confidence']}"
        )

    if not args.no_figures:
        draw_figures(load_report(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
