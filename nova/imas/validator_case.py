"""Open a pinned validator case's fetched layers as IMAS DBEntry objects.

The imas-efit validator store distributes independent equilibria as compact
NetCDF layers in an OCI registry.  A forward-solve gate needs them as the
per-IDS HDF5 layout imas-python reads, at the data-dictionary version each
layer was written in.  This module is the seam between those two:

* :func:`resolve_validator_case` fetches each layer the pinned case record
  names by its sha256 digest (never a mutable tag) through
  :func:`nova.database.layer_fetch.fetch_layer`, converting every NetCDF blob
  into a per-case HDF5 directory with a ``master.h5`` external-linking the
  per-IDS files, exactly the layout ``efit nc2hdf5`` produces.
* :class:`ValidatorCase` hands back an opened :class:`imas.DBEntry` per role
  (``input``, ``machine_description``, ``reference``) at the layer's own
  ``dd_version`` from ``tests/data-manifest.json``.

The NetCDF to HDF5 conversion reads and writes exclusively through imas-python
:class:`imas.DBEntry`; no field is touched with ``h5py``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    import imas

from nova.database.layer_fetch import (
    LayerFetchError,
    LayerRegistryUnreachable,
    fetch_layer,
)

#: The pinned case record lives beside the test corpus and is the single source
#: of truth for layer digests and per-dataset data-dictionary versions.
MANIFEST_PATH = Path(__file__).resolve().parents[2] / "tests" / "data-manifest.json"
RECORD_KEY = "efitpp_validator_layers"

#: Environment override for the local verified layer store, so a test run and a
#: CI run can share a cache without either hard-coding a host path.
STORE_ROOT_ENVIRONMENT = "NOVA_VALIDATOR_STORE"

__all__ = [
    "MANIFEST_PATH",
    "LayerFetchError",
    "LayerRegistryUnreachable",
    "ValidatorCase",
    "ValidatorCaseError",
    "ValidatorLayer",
    "default_store_root",
    "load_case_record",
    "netcdf_to_hdf5",
    "resolve_validator_case",
]


class ValidatorCaseError(Exception):
    """A pinned case could not be opened as DBEntry objects."""


def default_store_root() -> Path:
    """Return the verified layer store root, honouring the environment override."""

    override = os.environ.get(STORE_ROOT_ENVIRONMENT)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "nova" / "validator-layers"


def load_case_record(case_key: str) -> dict:
    """Return one case record from ``tests/data-manifest.json``.

    The record carries the case's ``dd_version`` and its ``layers`` list; each
    layer names a ``role``, a ``name``, a sha256 ``digest`` and its own
    ``dd_version``.  A missing case is a hard error rather than a skip, because
    a typo in a case key must not read as an unavailable registry.
    """

    payload = json.loads(MANIFEST_PATH.read_text())
    try:
        section = payload[RECORD_KEY]
    except KeyError as error:
        raise ValidatorCaseError(
            f"{MANIFEST_PATH} carries no {RECORD_KEY!r} section"
        ) from error
    cases = section.get("cases", {})
    if case_key not in cases:
        raise ValidatorCaseError(
            f"case {case_key!r} is absent from the {RECORD_KEY!r} records; "
            f"known cases: {sorted(cases)}"
        )
    record = dict(cases[case_key])
    record.setdefault("registry", section["registry"])
    return record


def _sanitize_object_arrays(ids: object) -> None:
    """Decode byte-string object arrays in place so HDF5 can serialise them.

    NetCDF string arrays read back as ``dtype=object`` and are not directly
    writable by the HDF5 backend.  This mirrors ``efit nc2hdf5`` and touches
    only object arrays whose first element is a byte string.
    """

    import numpy as np

    def walk(node: object) -> None:
        if isinstance(node, np.ndarray) and node.dtype == object:
            if node.size == 0:
                return
            sample = node.flat[0]
            if isinstance(sample, bytes):
                for index in range(node.size):
                    value = node.flat[index]
                    if isinstance(value, bytes):
                        node.flat[index] = value.decode("utf-8", errors="replace")
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
        elif hasattr(node, "__dict__") and not isinstance(node, type):
            for name in list(vars(node)):
                if name.startswith("_"):
                    continue
                try:
                    walk(getattr(node, name))
                except AttributeError, RuntimeError:
                    pass

    walk(ids)


def netcdf_to_hdf5(
    nc_path: Path | str,
    hdf5_directory: Path | str,
    dd_version: str,
    ids_names: Sequence[str],
) -> Path:
    """Convert one NetCDF layer into a per-IDS HDF5 directory.

    Returns the directory holding ``master.h5``.  The conversion is idempotent:
    an existing ``master.h5`` is reused so repeated test runs do not rewrite the
    store.  Each IDS is read from the NetCDF entry and written to the HDF5 entry
    through imas-python; an IDS the layer does not carry is skipped.
    """

    import imas

    source = Path(nc_path)
    target = Path(hdf5_directory)
    master = target / "master.h5"
    if master.is_file():
        return target
    target.mkdir(parents=True, exist_ok=True)

    entry_nc = imas.DBEntry(str(source), "r", dd_version=dd_version)
    entry_hdf5 = imas.DBEntry(f"imas:hdf5?path={target}", "w", dd_version=dd_version)
    try:
        for name in ids_names:
            try:
                ids = entry_nc.get(name)
            except Exception:  # noqa: BLE001 - absent IDS is a legitimate skip
                continue
            if ids is None:
                continue
            _sanitize_object_arrays(ids)
            entry_hdf5.put(ids)
    finally:
        entry_nc.close()
        entry_hdf5.close()

    if not master.is_file():
        raise ValidatorCaseError(
            f"conversion of {source} produced no master.h5 under {target}"
        )
    return target


@dataclass
class ValidatorLayer:
    """One fetched and converted layer of a pinned validator case."""

    role: str
    name: str
    digest: str
    dd_version: str
    nc_path: Path
    hdf5_directory: Path

    def entry(self) -> "imas.DBEntry":
        """Return an opened read-only DBEntry at this layer's DD version."""

        import imas

        return imas.DBEntry(
            f"imas:hdf5?path={self.hdf5_directory}",
            "r",
            dd_version=self.dd_version,
        )


@dataclass
class ValidatorCase:
    """A pinned validator case resolved to fetched, converted layers."""

    key: str
    machine: str
    code: str
    dd_version: str
    registry: str
    layers: dict[str, ValidatorLayer] = field(default_factory=dict)

    def layer(self, role: str) -> ValidatorLayer:
        try:
            return self.layers[role]
        except KeyError as error:
            raise ValidatorCaseError(
                f"case {self.key!r} carries no {role!r} layer; "
                f"present roles: {sorted(self.layers)}"
            ) from error

    def entry(self, role: str) -> "imas.DBEntry":
        """Return an opened DBEntry for one role's layer."""

        return self.layer(role).entry()


def _layer_specs(
    record: Mapping[str, object],
    extra_layers: Iterable[Mapping[str, object]],
) -> list[dict]:
    specs = [dict(layer) for layer in record["layers"]]
    for extra in extra_layers:
        specs.append(dict(extra))
    return specs


def resolve_validator_case(
    case_key: str,
    *,
    store_root: Path | str | None = None,
    extra_layers: Iterable[Mapping[str, object]] = (),
    layer_fetch=fetch_layer,
) -> ValidatorCase:
    """Fetch, verify and convert every layer a pinned case names.

    *extra_layers* supplies layers the pinned record omits — a case whose
    reference equilibrium is published under a second file name, for instance —
    each as a mapping with ``role``, ``name``, ``digest`` and optional
    ``dd_version``.  They are fetched and verified the same way.

    *layer_fetch* is the seam a test substitutes to exercise the converter
    without a network round trip.
    """

    record = load_case_record(case_key)
    root = Path(store_root) if store_root is not None else default_store_root()
    root.mkdir(parents=True, exist_ok=True)

    case = ValidatorCase(
        key=case_key,
        machine=str(record.get("machine", "")),
        code=str(record.get("code", "")),
        dd_version=str(record["dd_version"]),
        registry=str(record["registry"]),
    )
    for spec in _layer_specs(record, extra_layers):
        role = str(spec["role"])
        name = str(spec["name"])
        digest = str(spec["digest"])
        dd_version = str(spec.get("dd_version", case.dd_version))
        directory = layer_fetch(
            digest, cache_directory=root, name=name, registry=case.registry
        )
        nc_path = Path(directory) / name
        if not nc_path.is_file():
            raise ValidatorCaseError(
                f"store published {name} for case {case_key!r} but it is absent"
            )
        hdf5_directory = root / "hdf5" / case_key / role
        netcdf_to_hdf5(
            nc_path,
            hdf5_directory,
            dd_version,
            ("equilibrium", "pf_active", "pf_passive", "wall", "tf", "magnetics"),
        )
        case.layers[role] = ValidatorLayer(
            role=role,
            name=name,
            digest=digest,
            dd_version=dd_version,
            nc_path=nc_path,
            hdf5_directory=hdf5_directory,
        )
    return case
