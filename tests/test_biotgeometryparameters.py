"""Parameter blocks for a winding pack, and the sensor-space geometry Jacobian.

The kernel family differentiates with respect to raw section vertices
(``tests/test_biotgeometryautograd.py``).  What an inverse solve consumes is one
step up: a small parameter vector whose components are named deformations of a
whole pack -- a rigid shift, a dilation, a tilt -- and the Jacobian of the
sensor readings with respect to it.  This file pins that layer in the same three
parts the kernel contract uses (value parity, jacfwd against central differences
of the shipped numpy kernel, jacrev against jacfwd) and adds the two properties
the parameter layer alone has to carry:

* the generators are LINEAR in their parameters, so the parameter space is a
  vector space with an unambiguous metric -- the pack's root-mean-square vertex
  displacement in metres.  The singular values of a whitened Jacobian are then
  signal-to-noise per metre and their reciprocals are resolvable displacements,
  independent of how the generators are scaled.
* a section edge of constant z is a coordinate singularity of the pack's edge
  parametrisation (each edge is integrated in z, so an edge of zero height
  carries no gradient at all).  A pack of axis-aligned turns is entirely made of
  such edges, so the deformations that would tilt one -- tilt, shear, free
  vertices -- need the Jacobian taken away from the singular configuration.
"""

import numpy as np
import pytest

from nova.biot.geometryparameters import (
    AFFINE_BLOCKS,
    SensorArray,
    branch_discriminability,
    coupling_jacobian,
    mirror_rows,
    mode_spectrum,
    pack_coupling,
    pack_deformation,
    standoff_widths,
    without_span,
)
from nova.biot.polygon import polygon_greens

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
from nova.jax.config import enable_x64  # noqa: E402

enable_x64()

# Three axis-aligned turns of a winding pack: every section carries two edges of
# constant z, which is the configuration the traced pack cannot differentiate in
# place.
BOX_PACK = np.array(
    [
        [[1.423, z - 0.015], [1.577, z - 0.015], [1.577, z + 0.015], [1.423, z + 0.015]]
        for z in (-1.14, -1.10, -1.06)
    ]
)

# The same pack's outline, chamfered on one corner exactly as the described
# P4/P5 packs are -- six vertices, two of them on edges of constant z.
HULL = np.array(
    [
        [
            [1.4230, -1.1745],
            [1.4230, -1.0235],
            [1.5770, -1.0235],
            [1.5770, -1.1365],
            [1.5470, -1.1775],
            [1.4480, -1.1775],
        ]
    ]
)

# A section with no edge of constant z: the traced pack differentiates it in
# place, so it is the control for every flat-edge measurement below.
PENTAGON = np.array(
    [[[1.42, -1.16], [1.55, -1.19], [1.60, -1.07], [1.50, -1.015], [1.405, -1.06]]]
)

RULE = (4, 16)

SENSORS = SensorArray.field_probes(
    r=np.array([1.8449, 0.1803, 1.5913, 2.0]),
    z=np.array([0.3070, 0.5000, -0.7904, -1.1]),
    cos=np.array([1.0, 0.0, 0.7071067811865476, -0.5]),
    sin=np.array([0.0, 1.0, 0.7071067811865476, 0.8660254037844387]),
).join(SensorArray.flux_loops(r=np.array([1.7, 0.9]), z=np.array([-1.05, 0.2])))


def peak_scaled_gap(taken, reference):
    """Return the disagreement against each column's own peak.

    A Jacobian column passes through zero channel by channel, where a pointwise
    ratio measures the finite difference's cancellation rather than the
    derivative's accuracy.
    """
    taken = np.asarray(taken)
    reference = np.asarray(reference)
    peak = np.max(np.abs(reference), axis=0, keepdims=True)
    return np.max(np.abs(taken - reference) / peak)


def central_difference(deformation, sensors, axis, step=1e-5, rule=RULE):
    """Return the shipped kernel's own central difference along one parameter."""
    parameters = np.zeros(deformation.size)
    parameters[axis] = step
    plus = pack_coupling(deformation.deform(np, parameters), sensors, rule=rule)
    parameters[axis] = -step
    minus = pack_coupling(deformation.deform(np, parameters), sensors, rule=rule)
    return (plus - minus) / (2.0 * step)


# --- the generators ---------------------------------------------------------


def test_zero_parameters_return_the_base_pack():
    """The deformation is a displacement field, so it starts at the identity."""
    deformation = pack_deformation(BOX_PACK)
    assert np.array_equal(deformation.deform(np, np.zeros(deformation.size)), BOX_PACK)


def test_the_rigid_block_translates_every_vertex_alike():
    """Both rigid components move the whole pack and nothing else."""
    deformation = pack_deformation(BOX_PACK, blocks=("rigid",))
    assert deformation.labels == ("rigid_r", "rigid_z")
    for axis, direction in enumerate(([1.0, 0.0], [0.0, 1.0])):
        parameters = np.zeros(2)
        parameters[axis] = 0.004
        offset = deformation.deform(np, parameters) - BOX_PACK
        assert np.allclose(offset, 0.004 * np.asarray(direction))


def test_every_generator_carries_unit_root_mean_square_displacement():
    """One metre of parameter is one metre of pack displacement, per block."""
    for blocks in (AFFINE_BLOCKS, ("section",), ("vertex",)):
        deformation = pack_deformation(BOX_PACK, blocks=blocks)
        rms = np.sqrt(np.mean(np.sum(deformation.generators**2, axis=-1), axis=(1, 2)))
        assert np.allclose(rms, 1.0)


def test_the_tilt_generator_is_the_derivative_of_a_finite_rotation():
    """The tilt block is the rotation's generator, not a small finite rotation."""
    deformation = pack_deformation(BOX_PACK, blocks=("tilt",))
    centre = BOX_PACK.reshape(-1, 2).mean(axis=0)
    local = BOX_PACK - centre
    step = 1e-6
    rotation = np.array([[np.cos(step), -np.sin(step)], [np.sin(step), np.cos(step)]])
    turned = centre + local @ rotation.T
    scale = np.sqrt(np.mean(np.sum(local**2, axis=-1)))
    taken = deformation.deform(np, np.array([step * scale]))
    assert np.max(np.abs(taken - turned)) < 1e-5 * step * scale


def test_the_free_vertex_block_contains_the_affine_span():
    """Every named affine deformation is a special case of free vertices."""
    affine = pack_deformation(BOX_PACK, blocks=AFFINE_BLOCKS)
    vertex = pack_deformation(BOX_PACK, blocks=("vertex",))
    flat = vertex.generators.reshape(vertex.size, -1)
    residual = (
        affine.generators.reshape(affine.size, -1)
        - np.linalg.lstsq(
            flat.T, affine.generators.reshape(affine.size, -1).T, rcond=None
        )[0].T
        @ flat
    )
    assert np.max(np.abs(residual)) < 1e-12


def test_the_metric_is_the_gram_of_the_generators():
    """Dilation and stretch are not orthogonal unless the pack is square."""
    deformation = pack_deformation(BOX_PACK, blocks=AFFINE_BLOCKS)
    gram = deformation.gram()
    assert np.allclose(np.diag(gram), 1.0)
    assert np.allclose(gram, gram.T)
    dilation = deformation.labels.index("dilation")
    stretch = deformation.labels.index("stretch")
    assert abs(gram[dilation, stretch]) > 1e-3


# --- the coupling ------------------------------------------------------------


def test_the_pack_coupling_averages_the_shipped_kernel_over_the_turns():
    """A pack of turns reads the mean of its turns, per ampere-turn."""
    taken = pack_coupling(BOX_PACK, SENSORS, rule=RULE)
    rows = []
    for section in BOX_PACK:
        psi, b_r, b_z = polygon_greens(
            SENSORS.r, SENSORS.z, section, n_panels=RULE[0], n_nodes=RULE[1]
        )
        rows.append(np.stack([psi, b_r, b_z], axis=-1))
    expected = np.einsum("ij,sij->i", SENSORS.projection, np.asarray(rows)) / len(
        BOX_PACK
    )
    assert np.max(np.abs(taken - expected)) < 1e-15 * np.max(np.abs(expected))


def test_a_flux_channel_reads_flux_and_a_probe_reads_its_own_axis():
    """The projection weights select which kernel row each channel carries."""
    psi, b_r, b_z = polygon_greens(
        SENSORS.r, SENSORS.z, PENTAGON[0], n_panels=RULE[0], n_nodes=RULE[1]
    )
    taken = pack_coupling(PENTAGON, SENSORS, rule=RULE)
    probes = SensorArray.field_probes(
        SENSORS.r[:4],
        SENSORS.z[:4],
        np.array([1.0, 0.0, 0.7071067811865476, -0.5]),
        np.array([0.0, 1.0, 0.7071067811865476, 0.8660254037844387]),
    )
    assert np.allclose(
        taken[:4], b_r[:4] * probes.projection[:, 1] + b_z[:4] * probes.projection[:, 2]
    )
    assert np.allclose(taken[4:], psi[4:])


def test_the_traced_coupling_reproduces_the_shipped_kernel():
    """One arithmetic, two namespaces -- the trace changes nothing."""
    from nova.biot.geometryparameters import traced_pack_coupling
    from nova.biot.polygon import horizontal_edges

    masks = np.asarray([horizontal_edges(section) for section in PENTAGON])
    traced = np.asarray(
        traced_pack_coupling(jnp, jnp.asarray(PENTAGON), SENSORS, masks, rule=RULE)
    )
    host = pack_coupling(PENTAGON, SENSORS, rule=RULE)
    assert np.max(np.abs(traced - host)) < 1e-13 * np.max(np.abs(host))


# --- the Jacobian ------------------------------------------------------------


@pytest.mark.parametrize("block", AFFINE_BLOCKS)
def test_affine_jacobian_matches_central_differences_without_flat_edges(block):
    """The control: a section with no edge of constant z differentiates in place."""
    deformation = pack_deformation(PENTAGON, blocks=(block,))
    taken = coupling_jacobian(deformation, SENSORS, rule=RULE)
    for axis in range(deformation.size):
        difference = central_difference(deformation, SENSORS, axis)
        assert peak_scaled_gap(taken[:, axis], difference) < 1e-7


@pytest.mark.parametrize("block", AFFINE_BLOCKS)
def test_affine_jacobian_matches_central_differences_on_a_pack_of_turns(block):
    """Every named block on axis-aligned turns, against the kernel's own step."""
    deformation = pack_deformation(BOX_PACK, blocks=(block,))
    taken = coupling_jacobian(deformation, SENSORS, rule=RULE)
    for axis in range(deformation.size):
        difference = central_difference(deformation, SENSORS, axis)
        assert peak_scaled_gap(taken[:, axis], difference) < 1e-6


def test_free_vertex_jacobian_matches_central_differences_on_the_chamfered_hull():
    """Free vertices on a six-corner pack, two of whose edges have constant z."""
    deformation = pack_deformation(HULL, blocks=("vertex",))
    taken = coupling_jacobian(deformation, SENSORS, rule=RULE)
    for axis in range(deformation.size):
        difference = central_difference(deformation, SENSORS, axis)
        assert peak_scaled_gap(taken[:, axis], difference) < 1e-6


def test_per_turn_jacobian_matches_central_differences():
    """Each turn free to move on its own -- the lattice's per-element freedom."""
    deformation = pack_deformation(BOX_PACK, blocks=("section",))
    assert deformation.size == 2 * len(BOX_PACK)
    taken = coupling_jacobian(deformation, SENSORS, rule=RULE)
    for axis in range(deformation.size):
        difference = central_difference(deformation, SENSORS, axis)
        assert peak_scaled_gap(taken[:, axis], difference) < 1e-7


def test_a_constant_z_edge_carries_no_gradient_of_its_own():
    """Why the Jacobian is not taken at the pack's own configuration.

    Each edge is integrated in z, so an edge of constant z spans an interval of
    zero length: its contribution is exactly zero and stays zero under any
    perturbation the pack's own parametrisation can express.  A rigid shift and
    the two axis-aligned scalings keep such an edge flat, so their derivatives
    are untouched; shear and tilt do not, and taken in place they are not the
    derivative of anything -- they miss the whole contribution of the edges they
    lift out of the horizontal, and what survives does not even carry the right
    sign.
    """
    for block, flat_preserving in (
        ("rigid", True),
        ("dilation", True),
        ("stretch", True),
        ("shear", False),
        ("tilt", False),
    ):
        deformation = pack_deformation(BOX_PACK, blocks=(block,))
        taken = coupling_jacobian(deformation, SENSORS, rule=RULE, tilt_offset=0.0)
        difference = central_difference(deformation, SENSORS, 0, 1e-5)
        gap = peak_scaled_gap(taken[:, 0], difference)
        if flat_preserving:
            assert gap < 1e-6
        else:
            assert gap > 1.0


def test_reverse_mode_agrees_with_forward_mode():
    """What a large least-squares wants agrees with what the Jacobian is built by."""
    deformation = pack_deformation(BOX_PACK, blocks=AFFINE_BLOCKS)
    forward = coupling_jacobian(deformation, SENSORS, rule=RULE)
    reverse = coupling_jacobian(deformation, SENSORS, rule=RULE, mode="reverse")
    assert peak_scaled_gap(reverse, forward) < 1e-7


# --- the discrete branch -----------------------------------------------------


def test_mirroring_the_rows_moves_a_partial_row_to_the_other_side():
    """The chamfer side is a discrete state, reached by reflecting the turns."""
    pack = BOX_PACK[:2]
    mirrored = mirror_rows(pack)
    assert np.allclose(np.sort(mirrored[..., 0].ravel()), np.sort(pack[..., 0].ravel()))
    height = pack[..., 1]
    assert np.allclose(
        np.sort(mirrored[..., 1].ravel()),
        np.sort((height.min() + height.max()) - height.ravel()),
    )
    assert np.allclose(mirror_rows(mirrored), pack)


def test_a_symmetric_pack_cannot_discriminate_its_own_mirror():
    """A branch comparison is a distance, so an unchanged pack scores zero."""
    noise = np.full(len(SENSORS), 1e-9)
    described = pack_coupling(BOX_PACK, SENSORS, rule=RULE)
    mirrored = pack_coupling(mirror_rows(BOX_PACK), SENSORS, rule=RULE)
    assert branch_discriminability(described, mirrored, noise) < 1e-9
    partial = np.concatenate([BOX_PACK, BOX_PACK[:1] + [0.05, 0.12]])
    assert (
        branch_discriminability(
            pack_coupling(partial, SENSORS, rule=RULE),
            pack_coupling(mirror_rows(partial), SENSORS, rule=RULE),
            noise,
        )
        > 1.0
    )


# --- the spectrum ------------------------------------------------------------


def test_the_spectrum_reads_signal_to_noise_per_metre():
    """A diagonal response resolves each parameter at its own noise ratio."""
    jacobian = np.diag([4.0, 1.0, 0.25])
    noise = np.full(3, 0.5)
    spectrum = mode_spectrum(jacobian, np.eye(3), noise, labels=("a", "b", "c"))
    assert np.allclose(spectrum.values, [8.0, 2.0, 0.5])
    assert np.allclose(spectrum.resolution, [0.125, 0.5, 2.0])


def test_the_spectrum_does_not_depend_on_how_the_generators_are_scaled():
    """The metric is carried by the Gram, so rescaling a generator is a no-op."""
    generator = np.random.default_rng(3).normal(size=(6, 3))
    noise = np.full(6, 0.7)
    plain = mode_spectrum(generator, np.eye(3), noise)
    scale = np.diag([1.0, 40.0, 0.05])
    scaled = mode_spectrum(generator @ scale, scale @ scale, noise)
    assert np.allclose(plain.values, scaled.values)


def test_the_spectrum_drops_directions_the_generators_do_not_span():
    """A dependent generator set contributes no extra resolved direction."""
    basis = np.random.default_rng(5).normal(size=(8, 2))
    generators = np.column_stack([basis, basis @ np.array([1.0, -2.0])])
    gram = generators.T @ generators / 8
    spectrum = mode_spectrum(generators, gram, np.ones(8))
    assert spectrum.values.size == 2


def test_removing_a_column_span_leaves_nothing_of_it_behind():
    """Current freedoms are marginalised by projecting the whitened rows."""
    generator = np.random.default_rng(11)
    span = generator.normal(size=(9, 3))
    matrix = generator.normal(size=(9, 4))
    residual = without_span(matrix, span)
    assert np.max(np.abs(span.T @ residual)) < 1e-12
    assert np.max(np.abs(without_span(span, span))) < 1e-12


# --- sensor standoff ---------------------------------------------------------


def test_standoff_is_measured_in_the_pack_own_width():
    """A near-field screen has to scale with the pack, not with the machine."""
    widths = standoff_widths(SENSORS, BOX_PACK)
    assert widths.shape == (len(SENSORS),)
    assert np.all(widths > 0.0)
    apart = standoff_widths(SENSORS, BOX_PACK + np.array([0.0, 10.0]))
    assert np.all(apart > widths)
