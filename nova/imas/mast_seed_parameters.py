"""Documented MAST circuit relations and nominal material parameters.

The catalog geometry fixes where every conductor is, not how it is wired or what
it is made of.  This module supplies the electrical and material half from public
authoritative sources, and computes the parameters those sources license as
nominal seeds: a passive section's resistance follows from its resistivity, its
toroidal path length and its poloidal section, all of which are either published
or measured in the registry.

Two boundaries are held deliberately.  A solver's element or filament count is
never read as a physical turn count or as an electrical connection, so the turn
magnitudes stay unresolved until a vacuum-shot response identifies them.  Sources
that describe only the upgraded machine are inadmissible; every citation here
either names original MAST or names MAST alongside its upgrade.

Evidence records span the registry's whole assigned shot extent.  Whether a shot
was directly observed or inherited is carried once, by the artifact's shot
ranges, and is not repeated per field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import shapely

from nova.imas.machine_evidence import (
    EvidenceLedger,
    EvidenceRecord,
    FieldEvidence,
    SourceReference,
    Uncertainty,
)

MAST_MACHINE_SCOPES = frozenset({"mast", "mast-and-mast-u"})


def _startup_model(locator: str) -> SourceReference:
    """Cite the reduced-model startup study that documents MAST coil wiring."""

    return SourceReference(
        title=(
            "Battaglia et al., Reduced-model framework supporting direct induction "
            "startup scenario development for MAST-U and NSTX-U, UKAEA-CCFE-PR1934"
        ),
        url=(
            "https://scientific-publications.ukaea.uk/wp-content/uploads/"
            "UKAEA-CCFE-PR1934.PDF"
        ),
        locator=locator,
        machine="mast-and-mast-u",
        text_verified=True,
    )


def _magnetic_diagnostics(locator: str) -> SourceReference:
    """Cite the original MAST magnetic-diagnostic description."""

    return SourceReference(
        title=(
            "Edlington, Martin and Pinfold, MAST magnetic diagnostics, "
            "Review of Scientific Instruments 72, 421 (2001)"
        ),
        url=(
            "https://scientific-publications.ukaea.uk/wp-content/uploads/Published/"
            "RSIVOL72p421.pdf"
        ),
        locator=locator,
        machine="mast",
        text_verified=False,
    )


def _toroidal_field_topology(locator: str) -> SourceReference:
    """Cite the published MAST toroidal-field coil topology."""

    return SourceReference(
        title="PROCESS toroidal-field coil engineering model documentation",
        url="https://ukaea.github.io/PROCESS/eng-models/tf-coil/",
        locator=locator,
        machine="mast",
        text_verified=True,
    )


def _reduced_vessel_model(locator: str) -> SourceReference:
    """Cite the reconstruction study behind the reduced vessel representation."""

    return SourceReference(
        title=(
            "Berkery et al., Kinetic equilibrium reconstructions of plasmas in the "
            "MAST database and preparation for reconstruction of the first plasmas "
            "in MAST Upgrade, UKAEA-CCFE-PR(21)79"
        ),
        url=(
            "https://scientific-publications.ukaea.uk/papers/"
            "kinetic-equilibrium-reconstructions-of-plasmas-in-the-mast-database-"
            "and-preparation-for-reconstruction-of-the-first-plasmas-in-mast-upgrade/"
        ),
        locator=locator,
        machine="mast-and-mast-u",
        text_verified=True,
    )


def _catalog(locator: str) -> SourceReference:
    """Cite the recovered catalog geometry that measures a placement."""

    return SourceReference(
        title=(
            "FAIR-MAST level-1 and level-2 shot catalogs, recovered as the Nova "
            "physical machine-geometry registry"
        ),
        url="https://www.ukaea.org/service/fair-mast/",
        locator=locator,
        machine="mast",
        text_verified=True,
    )


@dataclass(frozen=True)
class ConductorMaterial:
    """Bulk resistivity range for one conductor material."""

    name: str
    resistivity: float
    resistivity_lower: float
    resistivity_upper: float
    assumption: str
    source: SourceReference

    def resistivity_interval(self) -> Uncertainty:
        """Return the admissible bulk-resistivity interval."""

        return Uncertainty(
            lower=self.resistivity_lower,
            upper=self.resistivity_upper,
            unit="ohm.m",
        )

    def loop_resistance(self, area: float, major_radius: float) -> float:
        """Return the nominal resistance of one axisymmetric conducting loop."""

        if area <= 0.0 or major_radius <= 0.0:
            raise ValueError(
                f"loop section must be positive, got area {area} and "
                f"major radius {major_radius}"
            )
        return self.resistivity * 2.0 * math.pi * major_radius / area

    def resistance_interval(self, area: float, major_radius: float) -> Uncertainty:
        """Return the resistance interval implied by the resistivity range."""

        scale = 2.0 * math.pi * major_radius / area
        return Uncertainty(
            lower=self.resistivity_lower * scale,
            upper=self.resistivity_upper * scale,
            unit="ohm",
        )


STAINLESS_STEEL = ConductorMaterial(
    name="austenitic stainless steel",
    resistivity=7.4e-7,
    resistivity_lower=6.9e-7,
    resistivity_upper=9.0e-7,
    assumption=(
        "bulk resistivity of austenitic stainless grades between room temperature "
        "and vessel bake temperature; the grade of an individual component is not "
        "documented, so the interval spans the common grades and the thermal range"
    ),
    source=_magnetic_diagnostics("stainless-steel vacuum vessel and coil cases"),
)

INCONEL = ConductorMaterial(
    name="Inconel",
    resistivity=1.15e-6,
    resistivity_lower=1.0e-6,
    resistivity_upper=1.35e-6,
    assumption=(
        "bulk resistivity of nickel-chromium superalloys sold as Inconel; the alloy "
        "number is not documented, so the interval spans the grades used for "
        "structural centre columns and the thermal range"
    ),
    source=_magnetic_diagnostics("Inconel centre tube"),
)

_VESSEL_STRUCTURE_FAMILIES = (
    "botcol",
    "endcrown_l",
    "endcrown_u",
    "lhorw",
    "mid",
    "p2larm",
    "p2ldivpl",
    "p2uarm",
    "p2udivpl",
    "ring",
    "topcol",
    "uhorw",
    "vertw",
)

_NAMED_MATERIAL_FAMILIES = {
    "coil_cases": STAINLESS_STEEL,
    "incon": INCONEL,
}

_UNRESOLVED_MATERIAL_FAMILIES = {
    "rodgr": (
        "a rod and ground-return family may be the copper centre conductor rather "
        "than vessel steel, and the two differ in resistivity by more than an "
        "order of magnitude, so no material is assigned"
    ),
}


def passive_material(family: str) -> ConductorMaterial | None:
    """Return the seeded material for a passive family, or None if unresolved."""

    if family in _UNRESOLVED_MATERIAL_FAMILIES:
        return None
    if family in _NAMED_MATERIAL_FAMILIES:
        return _NAMED_MATERIAL_FAMILIES[family]
    if family in _VESSEL_STRUCTURE_FAMILIES:
        return STAINLESS_STEEL
    raise KeyError(f"unknown passive family {family!r}")


def material_is_named_in_source(family: str) -> bool:
    """Return whether a source names this family's material directly."""

    return family in _NAMED_MATERIAL_FAMILIES


@dataclass(frozen=True)
class LoopSection:
    """Measured poloidal section of one passive family."""

    family: str
    area: float
    major_radius: float
    parts: int

    @property
    def is_single_loop(self) -> bool:
        """Return whether the family is one connected axisymmetric loop."""

        return self.parts == 1


def loop_sections(geometry: Mapping[str, Any]) -> dict[str, LoopSection]:
    """Measure section area, area-centroid radius and part count per family."""

    sections: dict[str, LoopSection] = {}
    for family, wkb_hex in sorted(geometry["passive_components"].items()):
        outline = shapely.from_wkb(bytes.fromhex(wkb_hex))
        parts = getattr(outline, "geoms", None)
        sections[family] = LoopSection(
            family=family,
            area=float(outline.area),
            major_radius=float(outline.centroid.x),
            parts=1 if parts is None else len(parts),
        )
    return sections


@dataclass(frozen=True)
class CoilCircuitRelation:
    """One documented poloidal-field circuit and how its coils are connected."""

    name: str
    families: tuple[str, ...]
    connection: str
    statement: str
    source: SourceReference
    turn_to_feed_current_ratio: float | None = None


CIRCUIT_RELATIONS = (
    CoilCircuitRelation(
        name="P1",
        families=("sol",),
        connection="two parallel circuits of two alternating layers",
        statement=(
            "the solenoid is driven as two parallel circuits of two alternating "
            "layers, four layers in total, so the feed current is twice the current "
            "carried by one turn"
        ),
        source=_startup_model("p. 9"),
        turn_to_feed_current_ratio=0.5,
    ),
    CoilCircuitRelation(
        name="P2",
        families=(
            "p2_inner_lower",
            "p2_inner_upper",
            "p2_outer_lower",
            "p2_outer_upper",
        ),
        connection="series",
        statement=(
            "the upper and lower coil of the set are wired in series to produce an "
            "up-down symmetric field"
        ),
        source=_startup_model("p. 6"),
    ),
    CoilCircuitRelation(
        name="P3",
        families=("p3_lower", "p3_upper"),
        connection="series",
        statement=(
            "the upper and lower coil of the set are wired in series to produce an "
            "up-down symmetric field"
        ),
        source=_startup_model("p. 6"),
    ),
    CoilCircuitRelation(
        name="P4",
        families=("p4_lower", "p4_upper"),
        connection="series",
        statement=(
            "the up-down pair is wired in series and driven by a single unipolar "
            "power supply that can only drive a confining field"
        ),
        source=_startup_model("pp. 6-7"),
    ),
    CoilCircuitRelation(
        name="P5",
        families=("p5_lower", "p5_upper"),
        connection="series",
        statement=(
            "the up-down pair is wired in series and driven by a single unipolar "
            "power supply"
        ),
        source=_startup_model("p. 7"),
    ),
    CoilCircuitRelation(
        name="P6",
        families=("p6_lower", "p6_upper"),
        connection="anti-series",
        statement=(
            "the upper and lower coils are wired in anti-series and driven by a "
            "bipolar supply in order to move the plasma vertically"
        ),
        source=_startup_model("p. 6"),
    ),
)

_PASSIVE_TURNS_PER_SECTION = 1.0


@dataclass(frozen=True)
class ProposedStandardName:
    """A machine-description quantity with no entry in the standard-name catalog."""

    name: str
    dd_path: str
    note: str


PROPOSED_STANDARD_NAMES = (
    ProposedStandardName(
        name="poloidal_field_coil_current",
        dd_path="pf_active/coil/current",
        note="absent from the catalog; nearest existing entry is unrelated",
    ),
    ProposedStandardName(
        name="poloidal_field_coil_turns",
        dd_path="pf_active/coil/element/turns_with_sign",
        note="absent; the signed effective turn count has no semantic identity",
    ),
    ProposedStandardName(
        name="passive_loop_resistance",
        dd_path="pf_passive/loop/resistance",
        note="absent; needed to key the seeded and later fitted passive model",
    ),
    ProposedStandardName(
        name="passive_loop_resistivity",
        dd_path="pf_passive/loop/resistivity",
        note="absent; needed to key the material seed",
    ),
    ProposedStandardName(
        name="vacuum_toroidal_magnetic_field_times_major_radius",
        dd_path="tf/b_field_phi_vacuum_r",
        note="absent; the toroidal-field reference product has no entry",
    ),
    ProposedStandardName(
        name="magnetic_probe_position_toroidal_angle",
        dd_path="magnetics/b_field_pol_probe/position/phi",
        note="absent; required to key the unresolved probe-bank assignment",
    ),
    ProposedStandardName(
        name="toroidal_magnetic_field_probe_orientation",
        dd_path="magnetics/b_field_phi_probe/toroidal_angle",
        note="absent; required to key the unresolved toroidal-probe orientation",
    ),
    ProposedStandardName(
        name="saddle_loop_traversal_sign",
        dd_path="magnetics/flux_loop/position",
        note=(
            "absent; the Faraday-law sign carried by vertex order has no semantic "
            "identity"
        ),
    ),
)

_SUBDIVISION_IS_NOT_TOPOLOGY = (
    "a solver's filament or element count is a discretization of the same "
    "conductor and is never read as a physical turn count or connection"
)


def _active_records(first_shot: int, last_shot: int) -> list[EvidenceRecord]:
    """Record what the sources fix about the active coils and their circuits."""

    records = [
        EvidenceRecord(
            path="pf_active/coil/element/geometry/outline",
            evidence=FieldEvidence.MEASURED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "each winding-pack outline is the subdivision-independent hull of "
                "the named catalog conductor cells"
            ),
            source=_catalog("level-2 pf_active named component arrays"),
        ),
        EvidenceRecord(
            path="pf_active/coil/element/turns_with_sign",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "no source gives a physical turn count or an absolute current "
                "direction for any poloidal field coil"
            ),
            assumptions=(
                _SUBDIVISION_IS_NOT_TOPOLOGY,
                "the signed coil-to-diagnostic vacuum response identifies effective "
                "turns and absolute polarity together, so both are promoted by a fit "
                "rather than asserted here",
            ),
            blocks_axisymmetric_forward_model=True,
        ),
        EvidenceRecord(
            path="pf_active/circuit/connections",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "the node matrix cannot be filled because it indexes every supply "
                "and coil terminal, and only part of the supply inventory is "
                "published"
            ),
            assumptions=(
                "the dictionary encodes connectivity as one column per supply and "
                "coil with a positive or negative side per node, so a partial supply "
                "list would silently assert absent terminals",
                "the coil grouping and connection kind each circuit needs are "
                "recorded per circuit and become authorable once the supply "
                "inventory is sourced",
            ),
        ),
        EvidenceRecord(
            path="pf_active/circuit(P2)/connections",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "the published relation covers the upper-to-lower pairing of the "
                "coil set, not whether the inner and outer winding packs are "
                "connected to each other"
            ),
            assumptions=(
                "the catalog resolves four separate P2 outlines, so the pack "
                "interconnection is a distinct electrical question from the "
                "documented up-down pairing",
            ),
        ),
    ]
    for relation in CIRCUIT_RELATIONS:
        records.append(
            EvidenceRecord(
                path=f"pf_active/circuit({relation.name})/type",
                evidence=FieldEvidence.PUBLISHED,
                first_shot=first_shot,
                last_shot=last_shot,
                statement=relation.statement,
                source=relation.source,
            )
        )
    ratio = CIRCUIT_RELATIONS[0].turn_to_feed_current_ratio
    records.append(
        EvidenceRecord(
            path="pf_active/coil(sol)/current",
            evidence=FieldEvidence.PUBLISHED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                f"the current in one solenoid turn is {ratio} of the measured feed "
                "current because two parallel circuits are driven"
            ),
            source=_startup_model("p. 9"),
        )
    )
    return records


def _passive_records(
    geometry: Mapping[str, Any],
    first_shot: int,
    last_shot: int,
) -> list[EvidenceRecord]:
    """Seed passive material, resistivity and single-loop resistance per family."""

    sections = loop_sections(geometry)
    records = [
        EvidenceRecord(
            path="pf_passive/loop/element/geometry/outline",
            evidence=FieldEvidence.MEASURED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "each passive outline is the union of the named catalog sections and "
                "is invariant under their subdivision"
            ),
            source=_catalog("level-2 pf_passive named component arrays"),
        ),
        EvidenceRecord(
            path="pf_passive/loop/element/turns_with_sign",
            evidence=FieldEvidence.GENERATED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "every axisymmetric passive section carries one toroidal turn in the "
                "positive toroidal direction"
            ),
            assumptions=(
                "an axisymmetric conducting section closes on itself once around the "
                "torus, so its effective turn count is one whatever subdivision the "
                "catalog used",
                "the induced-current direction follows the coordinate system rather "
                "than a wiring choice, so the sign is positive by construction",
            ),
            uncertainty=Uncertainty(lower=1.0, upper=1.0, unit="turn"),
            source=_reduced_vessel_model(
                "abstract; three-dimensional eddy currents are mapped to effective "
                "resistances in two-dimensional vessel groupings"
            ),
        ),
    ]
    for family, section in sections.items():
        material = passive_material(family)
        if material is None:
            reason = _UNRESOLVED_MATERIAL_FAMILIES[family]
            records.append(
                EvidenceRecord(
                    path=f"pf_passive/loop({family})/resistivity",
                    evidence=FieldEvidence.UNRESOLVED,
                    first_shot=first_shot,
                    last_shot=last_shot,
                    statement=f"no source names the material of the {family} family",
                    assumptions=(reason,),
                )
            )
        else:
            assumptions = [material.assumption]
            if not material_is_named_in_source(family):
                assumptions.append(
                    "the family is assigned to the documented vessel material by "
                    "association with the vessel structure, because no source names "
                    "this component individually"
                )
            records.append(
                EvidenceRecord(
                    path=f"pf_passive/loop({family})/resistivity",
                    evidence=FieldEvidence.GENERATED,
                    first_shot=first_shot,
                    last_shot=last_shot,
                    statement=(
                        f"the {family} family is seeded as {material.name} with a "
                        f"nominal bulk resistivity of {material.resistivity} ohm.m"
                    ),
                    assumptions=tuple(assumptions),
                    uncertainty=material.resistivity_interval(),
                    source=material.source,
                )
            )
        records.append(_resistance_record(section, material, first_shot, last_shot))
    return records


def _resistance_record(
    section: LoopSection,
    material: ConductorMaterial | None,
    first_shot: int,
    last_shot: int,
) -> EvidenceRecord:
    """Seed one family's loop resistance, or state why it stays unresolved."""

    path = f"pf_passive/loop({section.family})/resistance"
    if material is None or not section.is_single_loop:
        if material is None:
            reason = (
                "the resistance follows from a resistivity that no source fixes for "
                "this family"
            )
        else:
            reason = (
                f"the family resolves into {section.parts} disjoint sections, and no "
                "source states whether they form one electrical loop, so a single "
                "loop resistance would assert an unsourced galvanic grouping"
            )
        return EvidenceRecord(
            path=path,
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=f"no admissible loop resistance for the {section.family} family",
            assumptions=(reason,),
        )
    return EvidenceRecord(
        path=path,
        evidence=FieldEvidence.GENERATED,
        first_shot=first_shot,
        last_shot=last_shot,
        statement=(
            f"resistivity times a toroidal path length of "
            f"{2.0 * math.pi * section.major_radius:.4f} m over a measured section "
            f"of {section.area:.5f} m^2 gives "
            f"{material.loop_resistance(section.area, section.major_radius):.3e} ohm"
        ),
        assumptions=(
            "the section is one connected axisymmetric loop whose path length is the "
            "circumference at its area centroid and whose conducting cross-section "
            "is its poloidal area",
            material.assumption,
            "the material interval dominates the geometric contribution, because the "
            "outline is stored to a tenth of a millimetre",
        ),
        uncertainty=material.resistance_interval(section.area, section.major_radius),
        source=_reduced_vessel_model(
            "abstract; effective vessel resistances are refined against vacuum "
            "coil-test shots"
        ),
    )


def _wall_records(first_shot: int, last_shot: int) -> list[EvidenceRecord]:
    """Record the sourced limiter contour."""

    return [
        EvidenceRecord(
            path="wall/description_2d/limiter/unit/outline",
            evidence=FieldEvidence.MEASURED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "the limiter contour is the catalog wall cycle, canonicalized "
                "against start index and traversal direction"
            ),
            source=_catalog("level-2 wall limiter arrays"),
        )
    ]


def _magnetics_records(
    geometry: Mapping[str, Any],
    first_shot: int,
    last_shot: int,
) -> list[EvidenceRecord]:
    """Record measured diagnostic pose and the discrete choices still open."""

    magnetics = geometry["magnetics"]
    probes = len(magnetics["poloidal_probes"])
    loops = len(magnetics["flux_loops"])
    saddles = sum(len(paths) for paths in magnetics["saddle_paths"].values())
    vertices = sum(
        len(path) for paths in magnetics["saddle_paths"].values() for path in paths
    )
    additional = magnetics["additional_points"]
    extra_poloidal = sum(
        len(points)
        for family, points in additional.items()
        if family.startswith("poloidal_")
    )
    toroidal = sum(
        len(points)
        for family, points in additional.items()
        if family.startswith("toroidal_")
    )
    return [
        EvidenceRecord(
            path="magnetics/b_field_pol_probe/position",
            evidence=FieldEvidence.MEASURED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                f"{probes} primary probes and {extra_poloidal} additional poloidal "
                "probes carry a catalog major radius and height"
            ),
            source=_catalog("level-2 magnetics poloidal probe arrays"),
        ),
        EvidenceRecord(
            path="magnetics/b_field_pol_probe/poloidal_angle",
            evidence=FieldEvidence.MEASURED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                f"the sensitive-axis angle of the {probes} primary probes is joined "
                "from the level-1 setup arrays and converted once to radians"
            ),
            source=_catalog("level-1 magnetic probe orientation arrays"),
        ),
        EvidenceRecord(
            path="magnetics/b_field_pol_probe/position/phi",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                f"each of the {probes} primary probes sits at one of two catalog "
                "toroidal bank positions and the assignment is not sourced"
            ),
            assumptions=(
                "the two candidates are the toroidal angles the catalog itself "
                "carries, so neither is invented and no midpoint is admissible",
                "an axisymmetric forward model uses the major radius and height "
                "only, so the assignment changes no axisymmetric prediction",
            ),
            candidates=(
                "bank at toroidal angle 150 degree",
                "bank at toroidal angle 330 degree",
            ),
        ),
        EvidenceRecord(
            path="magnetics/b_field_pol_probe/toroidal_angle",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "no source gives an independent toroidal tilt for the poloidal probes"
            ),
            assumptions=(
                "reusing the poloidal sensitive-axis angle or a coordinate angle "
                "would change which field component the probe is modelled to "
                "measure",
            ),
        ),
        EvidenceRecord(
            path="magnetics/b_field_pol_probe/area",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement="no source gives the effective area or turns of a pickup coil",
            assumptions=(
                "the reconstruction consumes calibrated field values rather than raw "
                "coil voltages, so the effective area enters no axisymmetric "
                "prediction made from this artifact",
            ),
        ),
        EvidenceRecord(
            path="magnetics/b_field_phi_probe/position",
            evidence=FieldEvidence.MEASURED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                f"{toroidal} toroidal field probes carry a catalog position only"
            ),
            source=_catalog("level-2 magnetics toroidal probe arrays"),
        ),
        EvidenceRecord(
            path="magnetics/b_field_phi_probe/toroidal_angle",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                f"the installed orientation of the {toroidal} toroidal field probes "
                "is not sourced"
            ),
            assumptions=(
                "an orientation copied from a poloidal probe or from the coordinate "
                "frame would silently change the measured field component",
                "these sensors enter three-dimensional validation rather than a "
                "poloidal-magnetics equilibrium",
            ),
        ),
        EvidenceRecord(
            path="magnetics/flux_loop/position",
            evidence=FieldEvidence.MEASURED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                f"{loops} full flux loops carry a catalog position and a full "
                "toroidal span"
            ),
            source=_catalog("level-2 magnetics flux loop arrays"),
        ),
        EvidenceRecord(
            path="magnetics/flux_loop(saddle)/position",
            evidence=FieldEvidence.MEASURED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                f"{saddles} saddle paths carry {vertices} ordered catalog vertices, "
                "canonicalized so the geometry is direction neutral"
            ),
            source=_catalog("level-2 magnetics saddle path arrays"),
        ),
        EvidenceRecord(
            path="magnetics/flux_loop(saddle)/traversal_sign",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "the sign of the flux derivative a saddle sensor reports is not sourced"
            ),
            assumptions=(
                "a closed path and its reversal enclose the same area, so geometry "
                "alone cannot fix the sign; only the acquisition polarity chain or a "
                "known coil pulse can",
                "signed saddle measurements enter three-dimensional validation, not "
                "a poloidal-magnetics equilibrium",
            ),
            candidates=("negative traversal", "positive traversal"),
        ),
    ]


def _toroidal_field_records(first_shot: int, last_shot: int) -> list[EvidenceRecord]:
    """Record the published toroidal-field topology and the absent winding detail."""

    return [
        EvidenceRecord(
            path="tf",
            evidence=FieldEvidence.PUBLISHED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "the copper toroidal field coil inboard legs pass through the middle "
                "of the central solenoid and the coils can be dismantled so the "
                "solenoid can be inserted or removed"
            ),
            source=_toroidal_field_topology(
                "section: Topology of TF coils and Central Solenoid"
            ),
        ),
        EvidenceRecord(
            path="tf/r0",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "the official device reference radius is not sourced with a "
                "verifiable locator"
            ),
            assumptions=(
                "this node holds a declared machine constant, so deriving it from "
                "the limiter midplane would publish a number that disagrees with the "
                "machine's own description",
                "the vacuum toroidal field varies as one over major radius, so the "
                "field is fixed by the measured radius-field product carried per "
                "shot and no axisymmetric prediction needs the labelling radius",
            ),
        ),
        EvidenceRecord(
            path="tf/coils_n",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement="the number of toroidal field limbs is not sourced",
            assumptions=(
                "the published topology describes how the coils are arranged and "
                "dismantled without stating how many there are",
            ),
        ),
        EvidenceRecord(
            path="tf/coil/conductor/elements",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement="no installed three-dimensional winding path is sourced",
            assumptions=(
                "ripple, intrinsic error field and structural loads need this "
                "geometry, and an axisymmetric equilibrium does not",
            ),
        ),
    ]


def seed_evidence(
    geometry: Mapping[str, Any],
    *,
    first_shot: int,
    last_shot: int,
) -> EvidenceLedger:
    """Build the complete evidence ledger for one physical configuration."""

    records = [
        *_active_records(first_shot, last_shot),
        *_passive_records(geometry, first_shot, last_shot),
        *_wall_records(first_shot, last_shot),
        *_magnetics_records(geometry, first_shot, last_shot),
        *_toroidal_field_records(first_shot, last_shot),
    ]
    ledger = EvidenceLedger.create(records)
    for record in ledger.records:
        if record.source is not None and record.source.machine not in (
            MAST_MACHINE_SCOPES
        ):
            raise ValueError(
                f"field {record.path!r} cites a source for {record.source.machine!r}, "
                "which does not describe the original machine"
            )
    return ledger
