"""Author the static DIII-D machine description as IMAS IDSs.

Only geometry is copied from the source entry.  Dynamic actuator and diagnostic
arrays, reconstructed equilibrium content, and unsupported geometry are kept out
of the published description by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import imas
import numpy as np


SOURCE_PATH = Path("/home/ITER/tribolp/Public/imasdb/DIII-D/200000.nc")
DD_VERSION = "3.41.0"
IDS_NAMES = ("wall", "pf_active", "magnetics")


@dataclass(frozen=True)
class AbsentQuantity:
    """A machine-description quantity that its authoritative sources lack."""

    quantity: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-ready absence declaration."""

        return {
            "quantity": self.quantity,
            "status": "declared_absent",
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DiiidMachineIds:
    """The publishable IDS set and its explicit source gaps."""

    ids: Mapping[str, Any]
    source_path: Path
    dd_version: str
    absent: tuple[AbsentQuantity, ...]

    def validate(self) -> None:
        """Validate the static IDS set and its content firewall."""

        if tuple(self.ids) != IDS_NAMES:
            raise ValueError(f"machine IDS set must contain exactly {IDS_NAMES!r}")
        for name, ids in self.ids.items():
            ids.validate()
            if int(ids.ids_properties.homogeneous_time) != 0:
                raise ValueError(f"{name} must describe time-independent geometry")
        magnetics = self.ids["magnetics"]
        if any(probe.field.data.has_value for probe in magnetics.b_field_pol_probe):
            raise ValueError("magnetics probe signals are forbidden in the description")
        if any(loop.flux.data.has_value for loop in magnetics.flux_loop):
            raise ValueError("magnetics flux signals are forbidden in the description")
        if len(magnetics.ip):
            raise ValueError("plasma-current signals are forbidden in the description")


def _new_ids(factory: imas.IDSFactory, name: str, source_path: Path) -> Any:
    ids = factory.new(name)
    properties = ids.ids_properties
    properties.homogeneous_time = 0
    properties.source = str(source_path)
    properties.comment = (
        f"Static DIII-D machine geometry copied from {source_path} at IMAS "
        f"Data Dictionary {DD_VERSION}; dynamic signals and reconstruction "
        "content are excluded."
    )
    properties.provenance.node.resize(1)
    provenance = properties.provenance.node[0]
    provenance.path = str(source_path)
    provenance.sources = [f"IMAS netCDF entry; Data Dictionary {DD_VERSION}"]
    return ids


def _copy_present(source: Any, target: Any, name: str) -> bool:
    """Copy one primitive leaf only when the source actually supplies it."""

    leaf = getattr(source, name)
    if not leaf.has_value:
        return False
    value = leaf.value
    if isinstance(value, np.ndarray):
        value = value.copy()
    setattr(target, name, value)
    return True


def _copy_enum(source: Any, target: Any) -> None:
    """Copy the populated leaves of one IMAS identifier structure."""

    _copy_present(source, target, "index")
    _copy_present(source, target, "name")
    _copy_present(source, target, "description")


def _outline(element: Any) -> np.ndarray:
    """Return an element's exact stored outline or exact rectangle expansion."""

    geometry = element.geometry
    geometry_type = int(geometry.geometry_type)
    if geometry_type == 1:
        return np.column_stack(
            [
                np.asarray(geometry.outline.r, dtype=float),
                np.asarray(geometry.outline.z, dtype=float),
            ]
        )
    if geometry_type == 2:
        rectangle = geometry.rectangle
        radius = float(rectangle.r)
        height = float(rectangle.z)
        half_width = float(rectangle.width) / 2.0
        half_height = float(rectangle.height) / 2.0
        return np.asarray(
            [
                [radius - half_width, height - half_height],
                [radius + half_width, height - half_height],
                [radius + half_width, height + half_height],
                [radius - half_width, height + half_height],
            ],
            dtype=float,
        )
    raise ValueError(f"unsupported pf_active geometry type {geometry_type}")


def _author_wall(factory: imas.IDSFactory, source: Any, source_path: Path) -> Any:
    target = _new_ids(factory, "wall", source_path)
    if len(source.description_2d) != 1:
        raise ValueError("expected exactly one wall description")
    source_description = source.description_2d[0]
    if len(source_description.limiter.unit) != 1:
        raise ValueError("expected exactly one limiter unit")
    target.description_2d.resize(1)
    description = target.description_2d[0]
    _copy_enum(source_description.type, description.type)
    description.limiter.unit.resize(1)
    source_limiter = source_description.limiter.unit[0]
    limiter = description.limiter.unit[0]
    _copy_present(source_limiter, limiter, "name")
    _copy_present(source_limiter, limiter, "identifier")
    limiter.outline.r = np.asarray(source_limiter.outline.r, dtype=float)
    limiter.outline.z = np.asarray(source_limiter.outline.z, dtype=float)
    return target


def _author_pf_active(factory: imas.IDSFactory, source: Any, source_path: Path) -> Any:
    target = _new_ids(factory, "pf_active", source_path)
    target.coil.resize(len(source.coil))
    for source_coil, coil in zip(source.coil, target.coil, strict=True):
        _copy_present(source_coil, coil, "name")
        _copy_present(source_coil, coil, "identifier")
        coil.function.resize(len(source_coil.function))
        for source_function, function in zip(
            source_coil.function, coil.function, strict=True
        ):
            _copy_enum(source_function, function)
        coil.element.resize(len(source_coil.element))
        for source_element, element in zip(
            source_coil.element, coil.element, strict=True
        ):
            _copy_present(source_element, element, "name")
            _copy_present(source_element, element, "identifier")
            _copy_present(source_element, element, "turns_with_sign")
            vertices = _outline(source_element)
            element.geometry.geometry_type = 1
            element.geometry.outline.r = vertices[:, 0]
            element.geometry.outline.z = vertices[:, 1]
    return target


def _copy_position(source: Any, target: Any) -> None:
    for coordinate in ("r", "z", "phi"):
        _copy_present(source, target, coordinate)


def _author_magnetics(factory: imas.IDSFactory, source: Any, source_path: Path) -> Any:
    target = _new_ids(factory, "magnetics", source_path)
    target.b_field_pol_probe.resize(len(source.b_field_pol_probe))
    for source_probe, probe in zip(
        source.b_field_pol_probe, target.b_field_pol_probe, strict=True
    ):
        for name in (
            "name",
            "identifier",
            "length",
            "turns",
            "poloidal_angle",
            "toroidal_angle",
        ):
            _copy_present(source_probe, probe, name)
        _copy_enum(source_probe.type, probe.type)
        _copy_position(source_probe.position, probe.position)

    target.flux_loop.resize(len(source.flux_loop))
    for source_loop, loop in zip(source.flux_loop, target.flux_loop, strict=True):
        _copy_present(source_loop, loop, "name")
        _copy_present(source_loop, loop, "identifier")
        _copy_enum(source_loop.type, loop.type)
        loop.position.resize(len(source_loop.position))
        for source_position, position in zip(
            source_loop.position, loop.position, strict=True
        ):
            _copy_position(source_position, position)
    return target


def build_diiid_machine_ids(source_path: Path | str = SOURCE_PATH) -> DiiidMachineIds:
    """Read the source entry and build a static, firewalled IDS set."""

    source_path = Path(source_path)
    with imas.DBEntry(source_path, "r") as source_entry:
        source_ids = {
            name: source_entry.get(name, 0, autoconvert=False) for name in IDS_NAMES
        }
        for name, ids in source_ids.items():
            written = str(ids.ids_properties.version_put.data_dictionary)
            if written != DD_VERSION:
                raise ValueError(
                    f"source {name} carries Data Dictionary {written}, "
                    f"expected {DD_VERSION}"
                )
        if source_entry.list_all_occurrences("pf_passive"):
            raise ValueError("source unexpectedly contains pf_passive geometry")

    factory = imas.IDSFactory(version=DD_VERSION)
    bundle = DiiidMachineIds(
        ids={
            "wall": _author_wall(factory, source_ids["wall"], source_path),
            "pf_active": _author_pf_active(
                factory, source_ids["pf_active"], source_path
            ),
            "magnetics": _author_magnetics(
                factory, source_ids["magnetics"], source_path
            ),
        },
        source_path=source_path,
        dd_version=DD_VERSION,
        absent=(
            AbsentQuantity(
                "pf_passive",
                "the source entry has no pf_passive occurrence; no passive "
                "loop or vessel conductor is fabricated",
            ),
            AbsentQuantity(
                "tf static conductor geometry",
                "the source tf IDS supplies a time-dependent vacuum-field "
                "signal but no static conductor geometry; the signal is excluded",
            ),
            AbsentQuantity(
                "Thomson scattering line-of-sight endpoints",
                "neither the source entry nor the competition dataset supplies "
                "endpoint pairs; representative positions are not fabricated "
                "into sightlines",
            ),
        ),
    )
    bundle.validate()
    return bundle


def _primitive_leaves(parent: Any, names: tuple[str, ...]) -> Iterator[tuple[str, Any]]:
    for name in names:
        leaf = getattr(parent, name)
        if leaf.has_value:
            value = leaf.value
            if isinstance(value, np.ndarray):
                value = value.copy()
            yield name, value


def machine_ids_snapshot(ids: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return every authored static leaf under stable, indexed paths."""

    result: dict[str, dict[str, Any]] = {name: {} for name in IDS_NAMES}
    for name in IDS_NAMES:
        properties = ids[name].ids_properties
        for leaf, value in _primitive_leaves(
            properties, ("homogeneous_time", "source", "comment")
        ):
            result[name][f"ids_properties/{leaf}"] = value
        for index, node in enumerate(properties.provenance.node):
            for leaf, value in _primitive_leaves(node, ("path", "sources")):
                result[name][f"ids_properties/provenance/node[{index}]/{leaf}"] = value

    wall = ids["wall"]
    for description_index, description in enumerate(wall.description_2d):
        for leaf, value in _primitive_leaves(
            description.type, ("index", "name", "description")
        ):
            result["wall"][f"description_2d[{description_index}]/type/{leaf}"] = value
        for unit_index, unit in enumerate(description.limiter.unit):
            prefix = f"description_2d[{description_index}]/limiter/unit[{unit_index}]"
            for leaf, value in _primitive_leaves(unit, ("name", "identifier")):
                result["wall"][f"{prefix}/{leaf}"] = value
            result["wall"][f"{prefix}/outline/r"] = np.asarray(unit.outline.r).copy()
            result["wall"][f"{prefix}/outline/z"] = np.asarray(unit.outline.z).copy()

    active = ids["pf_active"]
    for coil_index, coil in enumerate(active.coil):
        coil_prefix = f"coil[{coil_index}]"
        for leaf, value in _primitive_leaves(coil, ("name", "identifier")):
            result["pf_active"][f"{coil_prefix}/{leaf}"] = value
        for function_index, function in enumerate(coil.function):
            for leaf, value in _primitive_leaves(
                function, ("index", "name", "description")
            ):
                result["pf_active"][
                    f"{coil_prefix}/function[{function_index}]/{leaf}"
                ] = value
        for element_index, element in enumerate(coil.element):
            prefix = f"{coil_prefix}/element[{element_index}]"
            for leaf, value in _primitive_leaves(
                element, ("name", "identifier", "turns_with_sign")
            ):
                result["pf_active"][f"{prefix}/{leaf}"] = value
            result["pf_active"][f"{prefix}/geometry/geometry_type"] = int(
                element.geometry.geometry_type
            )
            result["pf_active"][f"{prefix}/geometry/outline/r"] = np.asarray(
                element.geometry.outline.r
            ).copy()
            result["pf_active"][f"{prefix}/geometry/outline/z"] = np.asarray(
                element.geometry.outline.z
            ).copy()

    magnetics = ids["magnetics"]
    for probe_index, probe in enumerate(magnetics.b_field_pol_probe):
        prefix = f"b_field_pol_probe[{probe_index}]"
        for leaf, value in _primitive_leaves(
            probe,
            (
                "name",
                "identifier",
                "length",
                "turns",
                "poloidal_angle",
                "toroidal_angle",
            ),
        ):
            result["magnetics"][f"{prefix}/{leaf}"] = value
        for leaf, value in _primitive_leaves(
            probe.type, ("index", "name", "description")
        ):
            result["magnetics"][f"{prefix}/type/{leaf}"] = value
        for leaf, value in _primitive_leaves(probe.position, ("r", "z", "phi")):
            result["magnetics"][f"{prefix}/position/{leaf}"] = value

    for loop_index, loop in enumerate(magnetics.flux_loop):
        prefix = f"flux_loop[{loop_index}]"
        for leaf, value in _primitive_leaves(loop, ("name", "identifier")):
            result["magnetics"][f"{prefix}/{leaf}"] = value
        for leaf, value in _primitive_leaves(
            loop.type, ("index", "name", "description")
        ):
            result["magnetics"][f"{prefix}/type/{leaf}"] = value
        for position_index, position in enumerate(loop.position):
            for leaf, value in _primitive_leaves(position, ("r", "z", "phi")):
                result["magnetics"][f"{prefix}/position[{position_index}]/{leaf}"] = (
                    value
                )
    return result


def round_trip_leaf_receipt(
    expected: Mapping[str, Mapping[str, Any]],
    actual: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compare every static leaf, requiring identical structure and values."""

    receipt: dict[str, dict[str, dict[str, Any]]] = {}
    for ids_name in IDS_NAMES:
        expected_leaves = expected[ids_name]
        actual_leaves = actual[ids_name]
        if set(expected_leaves) != set(actual_leaves):
            missing = sorted(set(expected_leaves) - set(actual_leaves))
            extra = sorted(set(actual_leaves) - set(expected_leaves))
            raise ValueError(
                f"{ids_name} leaf structure changed; missing={missing}, extra={extra}"
            )
        ids_receipt = {}
        for path, left in expected_leaves.items():
            right = actual_leaves[path]
            left_array = np.asarray(left)
            right_array = np.asarray(right)
            if left_array.dtype.kind in "SUO" or right_array.dtype.kind in "SUO":
                equal = bool(np.array_equal(left_array, right_array))
                item = {"exact_equal": equal}
            else:
                equal = bool(np.array_equal(left_array, right_array))
                difference = (
                    0.0
                    if left_array.size == 0
                    else float(np.max(np.abs(left_array - right_array)))
                )
                item = {
                    "exact_equal": equal,
                    "maximum_absolute_difference": difference,
                }
            if not equal:
                raise ValueError(f"{ids_name}/{path} changed during round trip")
            ids_receipt[path] = item
        receipt[ids_name] = ids_receipt
    return receipt
