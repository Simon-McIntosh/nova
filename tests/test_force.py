import os
from pathlib import Path
import pytest
import tempfile

import numpy as np
from nova.biot.force import Force
from nova.biot.target import ForceTargetPolicy, section_force_target
from nova.frame.coilset import CoilSet

# a conductor pair whose vertical force is set by the other coil and whose radial
# force is dominated by its own section, so both limits of the target rule show
PAIR = [
    ("PFa", 8.0, 6.5, 0.65, 0.45, 248.0, 45e3),
    ("PFb", 3.9, 7.6, 0.80, 0.60, 553.0, -30e3),
]

# the rule a test names when it measures the tiling itself rather than the force
SUBDIVISION = ForceTargetPolicy(rule="subdivision")


def conductor_pair(nforce=16, dcoil=-2, target_policy=None):
    """Return the two-conductor coilset the target-rule measurements share."""
    route = {} if target_policy is None else {"force_target_policy": target_policy}
    coilset = CoilSet(nforce=nforce, dcoil=dcoil, **route)
    for name, x, z, dx, dz, nturn, current in PAIR:
        coilset.coil.insert(x, z, dx, dz, nturn=nturn, Ic=current, name=name)
    return coilset


def tiled_pair(nforce=16, dcoil=-2):
    """Return the same pair with the tiling named, whatever the shipped rule."""
    return conductor_pair(nforce=nforce, dcoil=dcoil, target_policy=SUBDIVISION)


def section_quadrature_force(order, dcoil=-2):
    """Return a force operator solved on the positive material rule."""
    coilset = conductor_pair(dcoil=dcoil)
    force = Force(
        *coilset.frames,
        name="force",
        target_policy=ForceTargetPolicy(rule="positive_material", order=order),
    )
    force.solve(1)
    return force


@pytest.fixture
def linked():
    coilset = CoilSet(nforce=10, dcoil=-2, dplasma=-3, tplasma="hex")
    coilset.coil.insert(5, 1, 0.1, 0.1, nturn=1)
    coilset.shell.insert({"e": [5, 1, 1.75, 1.0]}, 13, 0.05, delta=-9)
    coilset.shell.insert({"e": [5, 1, 1.95, 1.2]}, 13, 0.05, delta=-9)
    coilset.coil.insert(5, 2, 0.1, 0.2, nturn=1.3)
    coilset.coil.insert(5.2, 2, 0.1, 0.2, nturn=1.25)
    coilset.firstwall.insert(5.4, 1, 0.3, 0.6, section="e", Ic=-15e6)
    coilset.linkframe(["Coil2", "Coil0"])
    coilset.sloc["coil", "Ic"] = -15e6
    coilset.force.solve()
    return coilset


def test_turn_number():
    coilset = CoilSet(nforce=5, dcoil=-2)
    coilset.coil.insert(5, range(3), 0.1, 0.3, nturn=[1, 2, 3])
    coilset.force.solve()
    assert np.isclose(coilset.force.target.nturn.sum(), 6)


def tiled_coil(nforce, dcoil, width, height):
    """Return one tiled conductor, the segment number being what is measured."""
    coilset = CoilSet(nforce=nforce, dcoil=dcoil, force_target_policy=SUBDIVISION)
    coilset.coil.insert(5, 6, width, height)
    coilset.force.solve()
    return coilset


def test_negative_delta_frame():
    assert len(tiled_coil(9, -1, 0.9, 0.1).force) == 9


def test_negative_delta_subframe():
    assert len(tiled_coil(12, -16, 0.3, 0.3).force) == 12


def test_positive_delta():
    assert len(tiled_coil(-0.1, -2, 0.9, 0.1).force) == 9


def test_unit_delta():
    assert len(tiled_coil(1, -2, 0.9, 0.1).force) == 1


def test_matrix_attrs(linked):
    for attr in ["Fr", "Fz", "Fc"]:
        assert attr in linked.force.data


def test_matrix_length(linked):
    assert len(linked.Loc["coil", :]) == len(linked.force.Fr)


def test_store_load(linked):
    fr = linked.force.fr
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        linked.filepath = tmp.name
        linked.store()
        del linked
        path = Path(tmp.name)
        coilset = CoilSet(filename=path.name, dirname=path.parent).load()
        coilset._clear()
    os.unlink(tmp.name)
    assert np.allclose(fr, coilset.force.fr)


def test_resolution():
    coilset = CoilSet(dcoil=-1, force_target_policy=SUBDIVISION)
    coilset.coil.insert(5, [5, 6], 0.9, 0.1, Ic=45e3, nturn=500)
    coilset.force.solve(100)
    fr_lowres = coilset.force.fr
    coilset.force.solve(200)
    fr_highres = coilset.force.fr
    assert np.allclose(fr_lowres, fr_highres, rtol=1e-3)


def test_totals_do_not_depend_on_the_source_mesh():
    """Integrating each source section leaves the force free of its subdivision."""
    reference = tiled_pair(dcoil=-2)
    reference.force.solve()
    for dcoil in (-5, -20):
        coilset = tiled_pair(dcoil=dcoil)
        coilset.force.solve()
        np.testing.assert_allclose(coilset.force.fr, reference.force.fr, rtol=1e-12)
        np.testing.assert_allclose(coilset.force.fz, reference.force.fz, rtol=1e-12)
        np.testing.assert_allclose(coilset.force.fc, reference.force.fc, rtol=1e-12)


def test_moment_arm_turns_about_the_conductor():
    """An arm divided by the cell would grow without bound as the tiling refines."""
    for nforce in (1, 4, 16, 64, 256):
        coilset = tiled_pair(nforce=nforce)
        coilset.force.solve()
        assert np.max(np.abs(coilset.force.target.delta_z)) <= 0.5 + 1e-12
        assert np.max(np.abs(coilset.force.target.delta_r)) <= 0.5 + 1e-12


def test_crushing_moment_converges_as_the_tiling_refines():
    """The first moment has a limit, so successive refinements approach it."""
    moment = []
    for nforce in (8, 32, 128, 512):
        coilset = tiled_pair(nforce=nforce)
        coilset.force.solve()
        moment.append(coilset.force.fc)
    step = np.abs(np.diff(np.asarray(moment), axis=0))
    assert np.all(step[1] < step[0])
    assert np.all(step[2] < step[1])


def test_section_quadrature_reproduces_the_refined_tiling():
    """Both rules integrate one quantity, so they meet where both have converged.

    The residual gap is the tiling's, which falls only as the reciprocal of its
    cell count, so the bound here is set by the coarser of the two rules.
    """
    coilset = tiled_pair(nforce=1024)
    coilset.force.solve()
    force = section_quadrature_force(order=6)
    np.testing.assert_allclose(force.fr, coilset.force.fr, rtol=2e-3)
    np.testing.assert_allclose(force.fz, coilset.force.fz, rtol=2e-3)
    np.testing.assert_allclose(force.fc, coilset.force.fc, rtol=2e-3)


def test_section_quadrature_balances_action_against_reaction():
    """Two conductors exchange equal and opposite vertical force at any rule.

    The residual sum measures a target rule without needing a reference, and the
    fan closes it further than the tiling from fewer nodes.
    """
    force = section_quadrature_force(order=6)
    imbalance = abs(force.fz.sum()) / np.max(np.abs(force.fz))
    assert imbalance < 5e-5

    coilset = tiled_pair(nforce=256)
    coilset.force.solve()
    assert len(coilset.force) > len(force)
    assert abs(coilset.force.fz.sum()) / np.max(np.abs(coilset.force.fz)) > imbalance


def test_section_quadrature_converges_with_its_order():
    """Raising the order drives the rule towards the integral it approximates."""
    reference = section_quadrature_force(order=12).fr
    error = [
        np.max(np.abs(section_quadrature_force(order=order).fr / reference - 1))
        for order in (2, 4, 6)
    ]
    assert error[0] > error[1] > error[2]
    assert error[2] < 5e-5


def test_section_quadrature_holds_the_conductor_turns():
    """Contracting the nodes must leave each conductor's own turn count intact."""
    force = section_quadrature_force(order=3)
    assert np.isclose(force.target.nturn.sum(), sum(row[5] for row in PAIR))
    assert force.target.index.tolist() == [row[0] for row in PAIR]


def test_section_quadrature_is_free_of_the_source_mesh():
    """The fan's targets are the conductor outlines, not the conducting cells."""
    coarse = section_quadrature_force(order=4, dcoil=-2)
    fine = section_quadrature_force(order=4, dcoil=-20)
    np.testing.assert_allclose(coarse.fr, fine.fr, rtol=1e-12)
    np.testing.assert_allclose(coarse.fz, fine.fz, rtol=1e-12)


def test_default_force_rule_is_the_section_fan():
    """The shipped route is the fifth-order fan over positive material."""
    default = conductor_pair()
    default.force.solve()
    explicit = section_quadrature_force(order=5)
    np.testing.assert_array_equal(default.force.fr, explicit.fr)
    np.testing.assert_array_equal(default.force.fz, explicit.fz)
    np.testing.assert_array_equal(default.force.fc, explicit.fc)
    policy = ForceTargetPolicy.resolve(default.force.target_policy)
    assert (policy.rule, policy.order) == ("positive_material", 5)


def test_the_tiling_stays_reachable_and_exact():
    """Naming the tiling reproduces it bit for bit, and it is a distinct answer."""
    coilset = tiled_pair()
    coilset.force.solve()
    pair = conductor_pair()
    explicit = Force(*pair.frames, name="force", target_policy=SUBDIVISION)
    explicit.solve(16)
    np.testing.assert_array_equal(coilset.force.fr, explicit.fr)
    np.testing.assert_array_equal(coilset.force.fz, explicit.fz)
    np.testing.assert_array_equal(coilset.force.fc, explicit.fc)
    default = conductor_pair()
    default.force.solve()
    assert not np.array_equal(default.force.fr, coilset.force.fr)


def test_force_records_its_target_rule():
    """The rule that produced a stored operator must travel with it."""
    force = section_quadrature_force(order=3)
    policy = ForceTargetPolicy.resolve(force.data.attrs["force_target_policy"])
    assert (policy.rule, policy.order) == ("positive_material", 3)


def test_force_target_policy_is_fixed_after_construction():
    """A route swap between construction and solve would break cache identity."""
    coilset = conductor_pair()
    force = Force(*coilset.frames, name="force")
    force.target_policy = SUBDIVISION
    with pytest.raises(ValueError):
        force.solve(1)


def test_section_quadrature_refuses_a_force_map():
    """The fan integrates whole sections and cannot resolve force within one."""
    coilset = conductor_pair()
    force = Force(
        *coilset.frames,
        name="force",
        target_policy=ForceTargetPolicy(rule="positive_material"),
    )
    force.reduce = False
    with pytest.raises(ValueError):
        force.solve(1)


def plasma_coilset(nforce=4):
    """Return a plasma whose own force the operator is asked to integrate."""
    coilset = CoilSet(nforce=nforce, dplasma=-9, tplasma="hex", force_index="plasma")
    coilset.firstwall.insert({"o": [5, 0, 1.2]}, Ic=15e6)
    return coilset


def test_the_plasma_force_path_takes_the_tiling():
    """A plasma spreads its turns over its cells, which one uniform fan cannot."""
    force = plasma_coilset().force
    assert force.target_carries_plasma
    configured = ForceTargetPolicy.resolve(force.target_policy)
    assert configured.rule == "positive_material"
    admitted = force.material_rule(configured)
    assert admitted.rule == "subdivision"
    assert admitted.order == configured.order


@pytest.mark.xfail(
    raises=IndexError,
    strict=True,
    reason="PolyTarget hands a plasma's subframe rows to Target.insert, which "
    "reads its first argument as a polygon, so neither rule can build the target",
)
def test_a_plasma_force_target_builds():
    """The rule a plasma admits is the tiling, which must then build its target."""
    plasma_coilset().force.solve()


def test_the_plasma_rule_follows_the_target_and_not_the_coilset():
    """A coil force keeps the fan on a machine that also carries a plasma."""
    coilset = CoilSet(nforce=4, dcoil=-2, dplasma=-9, tplasma="hex")
    coilset.coil.insert(8.0, 6.5, 0.65, 0.45, nturn=248, Ic=45e3, name="PFa")
    coilset.firstwall.insert({"o": [5, 0, 1.2]}, Ic=15e6)
    coilset.force.solve()
    assert not coilset.force.target_carries_plasma
    solved = ForceTargetPolicy.resolve(coilset.force.data.attrs["force_target_policy"])
    assert solved.rule == "positive_material"


def test_section_force_nodes_refuse_plasma_material():
    """The rule refuses a turn density it cannot hold rather than averaging it."""
    coilset = plasma_coilset()
    with pytest.raises(NotImplementedError):
        section_force_target(coilset.frame, coilset.Loc["plasma", :].index)


@pytest.mark.parametrize(
    "policy",
    [
        {"rule": "midpoint"},
        {"order": 0},
        {"order": 2.5},
        {"precision": "float32"},
        {"device_eligibility": "device"},
    ],
)
def test_force_target_policy_rejects_an_unsupported_setting(policy):
    """Every setting that moves the integral belongs to the cache identity."""
    with pytest.raises(ValueError):
        ForceTargetPolicy(**policy)


if __name__ == "__main__":
    pytest.main([__file__])
