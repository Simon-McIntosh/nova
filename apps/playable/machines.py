"""Carrier selection and session construction for the playable app.

The Solov'ev machine is the default document.  A session argument selects the
MAST frozen-six response carrier instead — the shared, digest-verified
response the millisecond route is measured on — so the same app serves one
machine today and the carrier-backed layout when the mast profile is
assembled.  Each machine also exposes the conductor outlines the poloidal
view draws beside the wall: the Solov'ev fixture carries a ring per fitted
conductor on the carrier, and MAST reads its pf_active element rectangles
from its content-addressed machine description.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from apps.playable.production import ForwardMachine, ProductionSolver
from apps.playable.session import PlayableSession
from apps.playable.solovev import (
    build_machine as build_solovev_machine,
    coil_outlines as solovev_coil_outlines,
)
from apps.playable.shape import PlasmaShape

#: The shared MAST frozen-six response carrier (digest-verified at load).
MAST_FROZEN_SIX_CARRIER = Path(
    "/work/projects/imas_gpu/sophelio/mast_frozen_six_response_carriers"
)

#: The content-addressed MAST machine description the carrier geometry is
#: authored from (see ``nova/imas/AGENTS.md`` for the locating recipe).
MAST_MACHINE_ARTIFACT_CACHE = Path("/run/user/39486/imas-ambix-machine-artifact")
MAST_MACHINE_SEMANTIC_IDENTITY = (
    "sha256:8df7a0a6c3f6162dbe0f226660bc069f37de8eb69f0f7c80bbfedc2bd4be220c"
)
MAST_MACHINE_PHYSICAL_DIGEST = "b55c5bb005a2cb67"
MAST_MACHINE_REGISTRY_DIGEST = (
    "2a26cc0a3a22e7fb8f42a53ee4c45e639290f0c5587e5f56405772b007f31bfd"
)

#: Machine names the session argument may select.
AVAILABLE_MACHINES = ("solovev", "mast")


@dataclass(frozen=True)
class ConductorMachine(ForwardMachine):
    """A forward machine that also carries the conductor outlines to draw.

    The base carrier holds the solve profile, seed, wall and identity; this
    adds the outline polygons the poloidal view's coil channel draws, carried
    beside the wall exactly as the wall itself is carried.  The Solov'ev
    fixture contributes a ring per fitted conductor; MAST contributes the
    pf_active element rectangles from its machine description.
    """

    coils: tuple[np.ndarray, ...] = ()


class MachineUnavailable(RuntimeError):
    """The named carrier exists but its forward profile is not assembled here."""


def machine_argument(arguments, default: str = "solovev") -> str:
    """Resolve the ``machine`` session argument from Bokeh request arguments.

    Bokeh passes application arguments as lists of decoded bytestrings;
    ``machine=mast`` on the session URL selects the MAST carrier and the
    default (no argument) stays on the Solov'ev machine.
    """
    values = arguments.get("machine")
    if not values:
        return default
    value = values[0].decode() if isinstance(values[0], bytes) else str(values[0])
    if value not in AVAILABLE_MACHINES:
        raise ValueError(f"unknown machine {value!r}; choose from {AVAILABLE_MACHINES}")
    return value


def mast_carrier_path() -> Path:
    """Return the resolved frozen-six carrier directory, verified present."""
    directory = MAST_FROZEN_SIX_CARRIER
    if not directory.is_dir():
        raise FileNotFoundError(
            f"the MAST frozen-six response carrier is not reachable at {directory}"
        )
    return directory


def build_machine(machine: str) -> ConductorMachine:
    """Return the forward machine the named carrier supports."""
    if machine == "solovev":
        carrier = build_solovev_machine()
        return ConductorMachine(
            profile=carrier.profile,
            seed=carrier.seed,
            wall=carrier.wall,
            identity=carrier.identity,
            coils=solovev_coil_outlines(),
        )
    if machine == "mast":
        carrier = mast_carrier_path()
        raise MachineUnavailable(
            f"the MAST frozen-six carrier is reachable at {carrier}, but "
            "assembling its forward profile from the carrier is the "
            "carrier-backed receipt node's work, not the playable session's"
        )
    raise ValueError(f"unknown machine {machine!r}; choose from {AVAILABLE_MACHINES}")


def _pf_active_outlines(directory: Path, dd_version: str) -> tuple[np.ndarray, ...]:
    """Read every pf_active element outline from one opened description."""
    import imas

    entry = imas.DBEntry(f"imas:hdf5?path={directory}", "r", dd_version=dd_version)
    try:
        active = entry.get("pf_active", 0, lazy=False, autoconvert=False)
        outlines = []
        for coil in active.coil:
            for element in coil.element:
                outline = element.geometry.outline
                outlines.append(
                    np.column_stack(
                        (
                            np.asarray(outline.r, dtype=float),
                            np.asarray(outline.z, dtype=float),
                        )
                    )
                )
        return tuple(outlines)
    finally:
        entry.close()


def mast_pf_active_outlines(
    *,
    cache: Path | str = MAST_MACHINE_ARTIFACT_CACHE,
    semantic_identity: str = MAST_MACHINE_SEMANTIC_IDENTITY,
) -> tuple[np.ndarray, ...]:
    """Return the MAST pf_active element outlines from its machine description.

    The carrier geometry is authored from the content-addressed machine
    description, located the way ``nova/imas/AGENTS.md`` describes: resolve the
    artifact whose authored semantics match the canonical MAST identity, then
    open its static pf_active IDS through imas-python at the manifest DD pin
    and read each element's geometry outline as a rectangle.  An unreachable
    or unmatched description raises :class:`MachineUnavailable`.
    """
    from nova.imas.machine_artifact import (
        MachineArtifactManifest,
        resolve_machine_artifact,
    )

    cache = Path(cache)
    object_root = cache / "sha256"
    if not object_root.is_dir():
        raise MachineUnavailable(
            f"the MAST machine description cache is unreachable at {cache}"
        )
    for directory in sorted(object_root.iterdir()):
        if not directory.is_dir() or directory.is_symlink():
            continue
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            continue
        try:
            manifest = MachineArtifactManifest.from_bytes(manifest_path.read_bytes())
        except OSError, ValueError:
            continue
        if manifest.semantic_identity() != semantic_identity:
            continue
        artifact = resolve_machine_artifact(
            cache,
            manifest.digest,
            expected_physical_digest=MAST_MACHINE_PHYSICAL_DIGEST,
            expected_registry_digest=MAST_MACHINE_REGISTRY_DIGEST,
            allow_incomplete=True,
        )
        return _pf_active_outlines(artifact.directory, artifact.manifest.dd_version)
    raise MachineUnavailable(
        f"no MAST machine description with identity {semantic_identity} under {cache}"
    )


def machine_coil_outlines(
    machine: str, carrier: ForwardMachine | None
) -> tuple[np.ndarray, ...]:
    """Return the conductor outlines one machine draws beside its wall.

    The Solov'ev machine carries its fitted conductor rings on the carrier;
    MAST reads its pf_active element rectangles from the machine description.
    """
    if machine == "mast":
        return mast_pf_active_outlines()
    if carrier is not None:
        return tuple(getattr(carrier, "coils", ()))
    return ()


def build_session(
    machine: str = "solovev", *, shape: PlasmaShape | None = None
) -> PlayableSession:
    """Build a playable session over the machine the argument selects."""
    carrier = build_machine(machine)
    radius, height, _shape = carrier.profile.operator.raster_geometry()
    return PlayableSession(
        solver=ProductionSolver(carrier),
        shape=shape if shape is not None else PlasmaShape(),
        machine=machine,
        wall=carrier.wall,
        raster_bounds=(
            (float(radius[0]), float(radius[-1])),
            (float(height[0]), float(height[-1])),
        ),
    )
