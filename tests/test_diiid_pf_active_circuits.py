"""Tests for the persisted DIII-D pf_active circuits and pf_passive loops."""

from __future__ import annotations

from pathlib import Path

import imas
import numpy as np
import pytest

import math

from nova.imas.diiid_description import (
    ALL_PF_ACTIVE_CIRCUITS,
    ALL_PF_ACTIVE_SUPPLIES,
    CIRCUIT_DRIVEN_CONDUCTORS,
    F_COIL_BULK_ELEMENT_TURNS_WITH_SIGN,
    F_COILS,
    author_pf_active_circuits,
    correct_f_coil_bulk_element_turns,
)
from nova.imas.diiid_machine_ids import SOURCE_PATH, build_diiid_machine_ids
from nova.imas.diiid_passive import LOOP_COUNT

REGENERATE_MODULE_PATH = Path(
    "docs/figures/diiid-forward-onboarding/ids-set/regenerate_pf_active_and_passive.py"
)


def _load_regenerate_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "diiid_ids_set_regeneration", REGENERATE_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory) -> dict:
    module = _load_regenerate_module()
    output = tmp_path_factory.mktemp("diiid-ids-set") / "diiid_machine_description.nc"
    receipt = module.regenerate(output)
    with imas.DBEntry(
        output.resolve(), "r", dd_version=receipt["manifest"]["dd_version"]
    ) as entry:
        pf_active = entry.get("pf_active", 0, autoconvert=False)
        pf_passive = entry.get("pf_passive", 0, autoconvert=False)
        coils = {str(coil.name).strip(): coil for coil in pf_active.coil}
        yield {
            "receipt": receipt,
            "output": output,
            "pf_active": pf_active,
            "pf_passive": pf_passive,
            "coils": coils,
        }


@pytest.fixture(scope="module")
def netcdf_description_coils() -> dict:
    """The existing (unmodified) writer's coil geometry, read from the source.

    ``build_diiid_machine_ids`` -- the writer this node does not touch --
    canonicalises every source element (whether stored as an outline or a
    rectangle) into an outline polygon. This is the netCDF machine
    description the new ohmic coils' geometry and turns must match: it is
    the same route the persisted artifact's own coil array is authored
    through, so this fixture never re-derives geometry independently.
    """

    bundle = build_diiid_machine_ids(SOURCE_PATH)
    return {str(coil.name).strip(): coil for coil in bundle.ids["pf_active"].coil}


def test_regenerated_ids_set_has_expected_counts(regenerated):
    receipt = regenerated["receipt"]
    assert receipt["counts"]["poloidal_coils"] == 24
    assert receipt["counts"]["pf_active_circuits"] == len(ALL_PF_ACTIVE_CIRCUITS) == 19
    assert receipt["counts"]["pf_active_supplies"] == len(ALL_PF_ACTIVE_SUPPLIES) == 19
    assert receipt["counts"]["pf_passive_loops"] == LOOP_COUNT == 47
    assert receipt["round_trip"]["exact_equal"] is True
    assert receipt["round_trip"]["maximum_absolute_difference"] == 0.0


@pytest.mark.parametrize("name", CIRCUIT_DRIVEN_CONDUCTORS)
def test_new_ohmic_coil_geometry_and_turns_match_netcdf_source(
    regenerated, netcdf_description_coils, name
):
    written = regenerated["coils"][name]
    source = netcdf_description_coils[name]
    assert len(written.element) == len(source.element)
    for written_element, source_element in zip(
        written.element, source.element, strict=True
    ):
        assert int(written_element.geometry.geometry_type) == int(
            source_element.geometry.geometry_type
        )
        assert np.array_equal(
            np.asarray(written_element.geometry.outline.r, dtype=float),
            np.asarray(source_element.geometry.outline.r, dtype=float),
        )
        assert np.array_equal(
            np.asarray(written_element.geometry.outline.z, dtype=float),
            np.asarray(source_element.geometry.outline.z, dtype=float),
        )
        assert float(written_element.turns_with_sign) == float(
            source_element.turns_with_sign
        )


@pytest.mark.parametrize("name", F_COILS)
def test_f_coil_bulk_element_carries_total_ampere_turn_convention(
    regenerated, netcdf_description_coils, name
):
    """The F-coil channel already carries total ampere-turns (TurnConvention:
    applied_multiplier=1.0), so the bulk winding-pack element must drive with
    turns_with_sign = 1 (sign preserved), not the physical turn count -- else
    active_coil_response_from_imas double-counts the field by that count."""

    written = regenerated["coils"][name]
    source = netcdf_description_coils[name]
    assert len(written.element) == 1
    assert len(source.element) == 1
    written_element = written.element[0]
    source_element = source.element[0]

    # geometry is untouched by the turns correction
    assert int(written_element.geometry.geometry_type) == int(
        source_element.geometry.geometry_type
    )
    assert np.array_equal(
        np.asarray(written_element.geometry.outline.r, dtype=float),
        np.asarray(source_element.geometry.outline.r, dtype=float),
    )
    assert np.array_equal(
        np.asarray(written_element.geometry.outline.z, dtype=float),
        np.asarray(source_element.geometry.outline.z, dtype=float),
    )

    # the source netCDF still carries the physical turn count (58 or 55);
    # the persisted, corrected description carries exactly one signed turn
    source_turns = float(source_element.turns_with_sign)
    written_turns = float(written_element.turns_with_sign)
    assert abs(source_turns) in (55.0, 58.0)
    assert written_turns == math.copysign(
        F_COIL_BULK_ELEMENT_TURNS_WITH_SIGN, source_turns
    )

    correction = regenerated["receipt"]["f_coil_turns_convention_correction"]
    assert correction["original_turns_with_sign_by_coil"][name] == source_turns
    assert correction["verified"][name] == written_turns


def test_correct_f_coil_bulk_element_turns_rejects_multi_element_coil():
    bundle = build_diiid_machine_ids()
    pf_active = bundle.ids["pf_active"]
    author_pf_active_circuits(pf_active)
    coil = next(c for c in pf_active.coil if str(c.name).strip() == "F1A")
    coil.element.resize(2)
    with pytest.raises(Exception):
        correct_f_coil_bulk_element_turns(pf_active)


def test_new_coils_use_the_same_geometry_representation_as_existing_coils(regenerated):
    coils = regenerated["coils"]
    geometry_types = {
        int(element.geometry.geometry_type)
        for coil in coils.values()
        for element in coil.element
    }
    assert geometry_types == {1}
    for name in (*CIRCUIT_DRIVEN_CONDUCTORS, *F_COILS, "ECOILA"):
        assert name in coils
        assert coils[name].name == name


def test_ohmic_circuit_connections_wire_supply_to_recorded_scale_coils(regenerated):
    pf_active = regenerated["pf_active"]
    ohmic = ALL_PF_ACTIVE_CIRCUITS[0]
    coil_names = [str(coil.name).strip() for coil in pf_active.coil]
    supply_names = [str(supply.name).strip() for supply in pf_active.supply]
    matrix = np.asarray(pf_active.circuit[0].connections)

    assert str(pf_active.circuit[0].name) == ohmic.name
    assert matrix.shape == (7, len(supply_names) + len(coil_names))
    assert supply_names[0] == ALL_PF_ACTIVE_SUPPLIES[0].name

    touched_supply_columns = np.nonzero(matrix[:, : len(supply_names)].any(axis=0))[0]
    assert touched_supply_columns.tolist() == [0]

    touched_coil_columns = np.nonzero(matrix[:, len(supply_names) :].any(axis=0))[0]
    touched_names = {coil_names[index] for index in touched_coil_columns}
    assert touched_names == {"ECOILA", *CIRCUIT_DRIVEN_CONDUCTORS}

    recorded_gains = {drive.conductor: drive.gain for drive in ohmic.drives}
    assert recorded_gains == {
        "ECOILB": 2.0,
        "E567UP": 1.0,
        "E567DN": 1.0,
        "E89UP": 1.0456947569496173,
        "E89DN": 1.0456240764323717,
    }
    # every row is a signed two-terminal node: one +1 and one -1, never a
    # fitted coefficient baked into the topology matrix itself
    assert np.all(matrix.sum(axis=1) == 0)
    assert np.all(np.abs(matrix).sum(axis=1) == 2)


@pytest.mark.parametrize("name", F_COILS)
def test_f_coil_circuit_wires_its_own_supply_directly_to_its_own_coil(
    regenerated, name
):
    pf_active = regenerated["pf_active"]
    coil_names = [str(coil.name).strip() for coil in pf_active.coil]
    supply_names = [str(supply.name).strip() for supply in pf_active.supply]
    circuits_by_name = {str(circuit.name): circuit for circuit in pf_active.circuit}

    circuit = circuits_by_name[f"DIII-D {name} circuit"]
    matrix = np.asarray(circuit.connections)
    assert matrix.shape == (2, len(supply_names) + len(coil_names))

    touched_supply_columns = np.nonzero(matrix[:, : len(supply_names)].any(axis=0))[0]
    touched_coil_columns = np.nonzero(matrix[:, len(supply_names) :].any(axis=0))[0]
    assert supply_names[touched_supply_columns[0]] == f"DIII-D {name} supply"
    assert coil_names[touched_coil_columns[0]] == name
    # a direct one-supply-to-one-coil circuit carries no derived gain
    record = next(c for c in ALL_PF_ACTIVE_CIRCUITS if c.source_conductor == name)
    assert record.drives == ()


def test_pf_passive_loops_present_with_geometry(regenerated):
    pf_passive = regenerated["pf_passive"]
    assert len(pf_passive.loop) == LOOP_COUNT
    for loop in pf_passive.loop:
        assert len(loop.element) == 1
        element = loop.element[0]
        assert int(element.geometry.geometry_type) == 1
        assert len(np.asarray(element.geometry.outline.r)) >= 3
        assert float(loop.resistance) > 0.0


def test_author_pf_active_circuits_rejects_absent_coils():
    bundle = build_diiid_machine_ids()
    pf_active = bundle.ids["pf_active"]
    pf_active.coil.resize(0)
    with pytest.raises(Exception):
        author_pf_active_circuits(pf_active)
