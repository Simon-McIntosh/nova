"""The passive circuits' geometry, quadrature and flux linkage.

The linkage is the one quantity in the passive calibration that is not fitted, so
it is checked against something outside the code rather than against itself: the
mutual inductance of two coaxial rings has a closed form, and a small enough
section is a filament.  The rest of the module is checked for the invariants a
linkage matrix has to satisfy to be an inductance at all -- symmetry, positive
definiteness, and self terms that dominate their own row.

The registry-backed tests stay away from the full matrix on purpose.  Building it
costs minutes of kernel evaluation, which belongs in the fit driver and not in a
test; what the tests check on the registry is the circuit basis it produces.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import shapely
from scipy.special import ellipe, ellipk

from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.imas.mast_passive_inductance import (
    CELLS_ACROSS_THICKNESS,
    PassiveTurn,
    QUADRATURE_POINT_TARGET,
    linkage_matrix,
    nominal_resistance,
    passive_turns,
    probe_coupling,
    quadrature_grid,
    section_quadrature,
    unsourced_material,
)
from nova.imas.mast_passive_response import PassiveError
from nova.imas.mast_seed_parameters import STAINLESS_STEEL
from nova.imas.mast_vacuum_response import ProbeTarget

MU_0 = 4.0e-7 * math.pi


def rectangle(r: float, z: float, width: float, height: float) -> np.ndarray:
    """Return an axis-aligned rectangular section centred on ``(r, z)``."""

    return np.array(
        [
            [r - 0.5 * width, z - 0.5 * height],
            [r + 0.5 * width, z - 0.5 * height],
            [r + 0.5 * width, z + 0.5 * height],
            [r - 0.5 * width, z + 0.5 * height],
        ]
    )


def turn(name: str, vertices: np.ndarray, family: str = "vertw") -> PassiveTurn:
    """Return a passive circuit carrying one named section."""

    return PassiveTurn(name=name, family=family, enclosed_coil="", vertices=vertices)


def coaxial_mutual(r_one: float, r_two: float, separation: float) -> float:
    """Return the mutual inductance of two coaxial filamentary rings [H].

    Maxwell's closed form in terms of the complete elliptic integrals.  It is the
    reference the finite-section kernel has to reproduce once the sections are
    small enough to be filaments.
    """

    modulus_squared = 4.0 * r_one * r_two / ((r_one + r_two) ** 2 + separation**2)
    modulus = math.sqrt(modulus_squared)
    return (
        MU_0
        * math.sqrt(r_one * r_two)
        * (
            (2.0 / modulus - modulus) * float(ellipk(modulus_squared))
            - (2.0 / modulus) * float(ellipe(modulus_squared))
        )
    )


class TestSectionGeometry:
    """The section measurements the quadrature and the resistance both read."""

    def test_thickness_of_a_strip_is_its_width(self):
        """A long strip's thickness is twice its area over its perimeter."""

        strip = turn("strip", rectangle(2.0, 0.0, 0.02, 4.0))
        assert strip.thickness == pytest.approx(0.02, rel=0.01)

    def test_thickness_of_a_square_is_half_its_side(self):
        """A compact block returns half its side, which only asks for more points."""

        block = turn("block", rectangle(1.5, 0.0, 0.19, 0.19))
        assert block.thickness == pytest.approx(0.095, rel=1e-9)

    def test_a_degenerate_outline_is_refused(self):
        """A section with no outline cannot be given a quadrature."""

        flat = turn("flat", np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]))
        with pytest.raises(PassiveError):
            flat.thickness

    def test_centroid_and_area_come_from_the_outline(self):
        """The section measurements are the polygon's own."""

        block = turn("block", rectangle(1.25, -0.4, 0.1, 0.2))
        assert block.area == pytest.approx(0.02, rel=1e-9)
        assert block.centroid == pytest.approx((1.25, -0.4), abs=1e-12)


class TestSectionQuadrature:
    """The observer point set the linkage averages flux over."""

    def test_weights_are_a_section_mean(self):
        """Equal-area weights summing to one make the reduction a mean."""

        _, _, weight = section_quadrature(
            turn("block", rectangle(1.5, 0.0, 0.19, 0.19))
        )
        assert weight.sum() == pytest.approx(1.0, rel=1e-12)

    def test_every_point_lies_inside_the_outline(self):
        """A point outside the wall would average flux the conductor never links."""

        vertices = np.array(
            [
                [1.0, 0.0],
                [1.4, 0.0],
                [1.4, 0.01],
                [1.05, 0.01],
                [1.05, 0.4],
                [1.0, 0.4],
            ]
        )
        point_r, point_z, _ = section_quadrature(turn("bent", vertices))
        polygon = shapely.Polygon(vertices)
        assert bool(np.all(shapely.contains_xy(polygon, point_r, point_z)))
        assert point_r.size > 20

    def test_a_thin_wall_is_resolved_across_its_thickness(self):
        """The pitch follows the thickness, so a thin plate gets rows inside it."""

        plate = turn("plate", rectangle(1.1, 0.5, 0.003, 0.083))
        _, point_z, _ = section_quadrature(plate)
        point_r, _, _ = section_quadrature(plate)
        assert np.unique(point_r).size >= CELLS_ACROSS_THICKNESS
        assert np.unique(point_z).size > np.unique(point_r).size

    def test_the_point_target_coarsens_a_long_shell(self):
        """A four-metre shell resolved across its wall would never be affordable."""

        shell = turn("shell", rectangle(2.0, 0.0, 0.02, 4.16))
        point_r, _, _ = section_quadrature(shell)
        assert point_r.size < 3 * QUADRATURE_POINT_TARGET
        assert point_r.size > QUADRATURE_POINT_TARGET // 4

    def test_a_sliver_the_grid_misses_still_gets_a_point_inside_it(self):
        """A section no cell centre lands in falls back to a point inside, not none.

        A thin diagonal sliver is the shape that defeats an axis-aligned grid: its
        bounding box is mostly void, so a coarse pitch can miss the wall entirely.
        Returning nothing there would leave the circuit's row of the linkage empty
        and its self inductance zero, which reads as a superconductor.
        """

        sliver = turn(
            "sliver",
            np.array([[1.0, 0.0], [1.4, 0.4], [1.4, 0.401], [1.0, 0.001]]),
        )
        point_r, point_z, weight = section_quadrature(sliver, point_target=1)
        assert point_r.size >= 1
        assert weight.sum() == pytest.approx(1.0)
        assert bool(
            np.all(
                shapely.contains_xy(shapely.Polygon(sliver.vertices), point_r, point_z)
            )
        )

    def test_the_grid_owns_every_point_back_to_its_circuit(self):
        """The owner index is what reduces one kernel call to a linkage column."""

        turns = [
            turn("one", rectangle(1.0, 0.0, 0.05, 0.05)),
            turn("two", rectangle(1.5, 0.2, 0.05, 0.05)),
        ]
        point_r, _, weight, owner = quadrature_grid(turns)
        assert owner.min() == 0 and owner.max() == 1
        assert point_r.size == owner.size == weight.size
        for index in (0, 1):
            assert weight[owner == index].sum() == pytest.approx(1.0, rel=1e-12)


class TestLinkage:
    """The flux linkage matrix, against a closed form and against its own rules."""

    def test_a_small_section_reproduces_the_coaxial_mutual(self):
        """Two sections small against their separation are two filaments."""

        turns = [
            turn("inner", rectangle(1.0, 0.0, 0.004, 0.004)),
            turn("outer", rectangle(1.5, 0.0, 0.004, 0.004)),
        ]
        linkage = linkage_matrix(turns)
        expected = coaxial_mutual(1.0, 1.5, 0.0)
        assert linkage.matrix[0, 1] == pytest.approx(expected, rel=2e-3)

    def test_an_axially_separated_mutual_matches_too(self):
        """The closed form is reproduced off the midplane as well as on it."""

        turns = [
            turn("lower", rectangle(1.2, -0.6, 0.004, 0.004)),
            turn("upper", rectangle(1.2, 0.6, 0.004, 0.004)),
        ]
        linkage = linkage_matrix(turns)
        expected = coaxial_mutual(1.2, 1.2, 1.2)
        assert linkage.matrix[0, 1] == pytest.approx(expected, rel=2e-3)

    def test_the_matrix_is_symmetric_and_positive_definite(self):
        """An inductance matrix has no other option, whatever the quadrature did."""

        turns = [
            turn("one", rectangle(1.0, 0.0, 0.05, 0.05)),
            turn("two", rectangle(1.1, 0.05, 0.03, 0.20)),
            turn("three", rectangle(1.8, -0.4, 0.02, 0.60)),
        ]
        linkage = linkage_matrix(turns)
        assert np.allclose(linkage.matrix, linkage.matrix.T, atol=0.0)
        assert np.linalg.eigvalsh(linkage.matrix).min() > 0.0

    def test_a_self_term_dominates_its_own_row(self):
        """A circuit links more of its own flux than of any neighbour's."""

        turns = [
            turn("one", rectangle(1.0, 0.0, 0.05, 0.05)),
            turn("two", rectangle(1.1, 0.05, 0.03, 0.20)),
            turn("three", rectangle(1.8, -0.4, 0.02, 0.60)),
        ]
        matrix = linkage_matrix(turns).matrix
        for row in range(len(turns)):
            off = np.abs(np.delete(matrix[row], row))
            assert matrix[row, row] > off.max()

    def test_reciprocity_residual_is_reported_and_small(self):
        """The observer quadrature error is measurable, and it is the error bar."""

        turns = [
            turn("one", rectangle(1.0, 0.0, 0.05, 0.05)),
            turn("two", rectangle(1.6, 0.30, 0.02, 0.40)),
        ]
        linkage = linkage_matrix(turns)
        assert 0.0 < linkage.reciprocity_residual < 0.05

    def test_a_finite_section_self_term_is_below_the_filament_limit(self):
        """Averaging over the section is what keeps a self inductance finite."""

        thin = linkage_matrix([turn("thin", rectangle(1.0, 0.0, 0.004, 0.004))])
        thick = linkage_matrix([turn("thick", rectangle(1.0, 0.0, 0.08, 0.08))])
        assert thick.matrix[0, 0] < thin.matrix[0, 0]

    def test_the_names_and_point_counts_travel_with_the_matrix(self):
        """A committed matrix is only reusable if its rows are labelled."""

        turns = [
            turn("one", rectangle(1.0, 0.0, 0.05, 0.05)),
            turn("two", rectangle(1.5, 0.0, 0.05, 0.05)),
        ]
        linkage = linkage_matrix(turns)
        assert linkage.names == ("one", "two")
        assert linkage.quadrature_points.sum() > 0


class TestNominalResistance:
    """The seeded ring resistance the fit scales rather than replaces."""

    def test_resistance_is_resistivity_over_the_measured_section(self):
        """The nominal value is the section's own geometry times its resistivity."""

        block = turn("block", rectangle(1.5, 0.0, 0.1, 0.2))
        expected = STAINLESS_STEEL.resistivity * 2.0 * math.pi * 1.5 / 0.02
        assert nominal_resistance([block])[0] == pytest.approx(expected, rel=1e-9)

    def test_a_thinner_section_carries_more_resistance(self):
        """Halving the conducting area doubles the ring resistance."""

        wide = turn("wide", rectangle(1.5, 0.0, 0.1, 0.2))
        narrow = turn("narrow", rectangle(1.5, 0.0, 0.05, 0.2))
        values = nominal_resistance([wide, narrow])
        assert values[1] == pytest.approx(2.0 * values[0], rel=1e-9)


class TestProbeCoupling:
    """The field each circuit produces on each probe's sensitive axis."""

    def test_each_probe_reads_the_axis_it_is_posed_along(self):
        """A radial probe and an axial probe at one place read different columns."""

        source = turn("source", rectangle(1.0, 0.0, 0.05, 0.05))
        radial = ProbeTarget("obr01", "obr", 0, 1.6, 0.30, 1.0, 0.0)
        axial = ProbeTarget("obv01", "obv", 1, 1.6, 0.30, 0.0, 1.0)
        coupling = probe_coupling([source], [radial, axial])
        assert coupling.shape == (2, 1)
        assert not math.isclose(coupling[0, 0], coupling[1, 0], rel_tol=1e-6)

    def test_the_field_falls_off_with_distance(self):
        """A probe further from a circuit reads less of it."""

        source = turn("source", rectangle(1.0, 0.0, 0.05, 0.05))
        near = ProbeTarget("obv01", "obv", 0, 1.2, 0.0, 0.0, 1.0)
        far = ProbeTarget("obv02", "obv", 1, 2.0, 0.0, 0.0, 1.0)
        coupling = probe_coupling([source], [near, far])
        assert abs(coupling[0, 0]) > abs(coupling[1, 0])


class TestRegistryCircuits:
    """The circuit basis the corrected registry geometry produces."""

    @pytest.fixture(scope="class")
    def turns(self):
        """Return the passive circuits of the authored configuration."""

        registry = MachineGeometryRegistry.default()
        geometry = registry.select(11766).configuration.geometry
        return passive_turns(geometry)

    def test_one_circuit_per_disjoint_section(self, turns):
        """Separateness is the default; a parallel connection would be asserted."""

        assert len(turns) > 40
        assert len({each.name for each in turns}) == len(turns)

    def test_every_case_plate_names_the_coil_it_encloses(self, turns):
        """A case circuit is only reachable if the excited coil inside it is known."""

        cases = [each for each in turns if each.family == "coil_cases"]
        assert cases
        assert all(each.enclosed_coil for each in cases)
        assert {each.enclosed_coil for each in cases} == {
            f"p{index}_{end}" for index in (2, 3, 4, 5, 6) for end in ("lower", "upper")
        }

    def test_the_corrected_cases_carry_both_p6_groups(self, turns):
        """The unchannelled P6 cases are present and stay induced-only."""

        p6 = [each for each in turns if each.enclosed_coil.startswith("p6")]
        assert len(p6) == 2

    def test_the_unsourced_material_family_is_named(self, turns):
        """A fit that moves this family fits a resistivity nobody published."""

        assert all(name.startswith("rodgr") for name in unsourced_material(turns))
        assert unsourced_material(turns)

    def test_every_circuit_admits_a_quadrature(self, turns):
        """A section the grid cannot sample would silently leave a row empty."""

        _, _, weight, owner = quadrature_grid(turns)
        counts = np.bincount(owner, minlength=len(turns))
        assert counts.min() >= 1
        for index in range(len(turns)):
            assert weight[owner == index].sum() == pytest.approx(1.0, rel=1e-9)

    def test_every_circuit_has_a_positive_nominal_resistance(self, turns):
        """A resistance of zero or less would produce a non-decaying mode."""

        assert nominal_resistance(turns).min() > 0.0
