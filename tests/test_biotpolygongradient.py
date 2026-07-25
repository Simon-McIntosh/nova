"""Equivalence contract for the closed-form gradient of the polygon kernel.

The polygon section's field is the curl of its vector potential. Taking that
curl by complex-step differentiation is exact but pays for two complex passes
over the boundary quadrature, and every transcendental in the antiderivative is
evaluated twice in complex arithmetic. Differentiating the antiderivative in
closed form evaluates the same transcendentals once, in real arithmetic, and
reuses them for the value and both derivatives.

That is only worth doing if it is exact. These tests pin the closed-form
gradient against a complex-step reference built directly from the verified
antiderivative. The two agree to ~1e-11 relative, and the floor is summation
round-off rather than the derivation: psi is the SAME expression in both paths
and disagrees by as much as the derivatives do, because the edge-limit
difference cancels several digits. The gate is set at 1e-9 -- two orders inside
that floor, and ten orders inside anything a mis-derived term would produce --
while the absolute accuracy of the field is pinned against the closed-form
rectangle oracle in ``test_biotpolygon.py``.
"""

import numpy as np
import pytest

from nova.biot import polygon
from nova.biot.polygon import polygon_greens

CSTEP = 1e-30


def hexagon(r0=6.2, z0=0.0, radius=0.06):
    """Return flat-top regular hexagon vertices, counter-clockwise."""
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)])


def trapezium(r0=2.0, z0=0.1):
    """Return a slanted, non-symmetric quadrilateral section."""
    return np.array(
        [[r0, z0], [r0 + 0.25, z0 - 0.05], [r0 + 0.3, z0 + 0.2], [r0 - 0.05, z0 + 0.12]]
    )


def thin_plate(r0=3.0):
    """Return a high aspect-ratio parallelogram."""
    return np.array([[r0, 0.0], [r0 + 0.4, 0.06], [r0 + 0.4, 0.075], [r0, 0.015]])


def complex_step_reference(target_r, target_z, vertices, n_panels=16, n_nodes=48):
    """Return ``(psi, dpsi_dr, dpsi_dz)`` by complex-step on the antiderivative.

    This is the formulation the closed-form gradient replaces, kept here as the
    reference it has to reproduce rather than as a second shipped code path.
    """
    v = np.asarray(vertices, dtype=np.float64)
    r = np.asarray(target_r, dtype=np.float64).ravel()[:, None]
    z = np.asarray(target_z, dtype=np.float64).ravel()[:, None]
    sign, area = polygon._orientation(v)
    phi, wts = polygon._phi_rule(n_panels, n_nodes)
    cosp, sinp = np.cos(phi), np.sin(phi)
    rule = (v, cosp, sinp, np.sin(2.0 * phi), wts * cosp, sign, area)
    psi_r = polygon._psi_hat(r + 1j * CSTEP, z, *rule)
    dpsi_dz = polygon._psi_hat(r, z + 1j * CSTEP, *rule).imag / CSTEP
    return psi_r.real, psi_r.imag / CSTEP, dpsi_dz


def closed_form(target_r, target_z, vertices, n_panels=16, n_nodes=48):
    """Return ``(psi, dpsi_dr, dpsi_dz)`` from the closed-form gradient."""
    v = np.asarray(vertices, dtype=np.float64)
    r = np.asarray(target_r, dtype=np.float64).ravel()[:, None]
    z = np.asarray(target_z, dtype=np.float64).ravel()[:, None]
    edge, weight, norm = polygon.pack_section(v)
    phi, wts = polygon._phi_rule(n_panels, n_nodes)
    cosp, sinp = np.cos(phi), np.sin(phi)
    return polygon._psi_gradient(
        r, z, edge, weight, cosp, sinp, np.sin(2.0 * phi), wts * cosp, norm
    )


def ring(vertices, count=64, radii=(0.4, 1.3, 4.0, 30.0)):
    """Return targets on rings of increasing section-radii offset."""
    v = np.asarray(vertices, float)
    centre = v.mean(axis=0)
    radius = float(np.max(np.hypot(*(v - centre).T)))
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    out_r, out_z = [], []
    for scale in radii:
        out_r.append(centre[0] + scale * radius * np.cos(angle))
        out_z.append(centre[1] + scale * radius * np.sin(angle))
    return np.concatenate(out_r), np.concatenate(out_z)


SECTIONS = [hexagon(), trapezium(), thin_plate()]


@pytest.mark.parametrize("vertices", SECTIONS)
def test_closed_form_gradient_matches_complex_step(vertices):
    """Value and both derivatives agree to near machine precision."""
    target_r, target_z = ring(vertices)
    expected = complex_step_reference(target_r, target_z, vertices)
    computed = closed_form(target_r, target_z, vertices)
    for got, want in zip(computed, expected):
        scale = np.max(np.abs(want))
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12 * scale)


@pytest.mark.parametrize("rule", [(16, 48), (8, 24), (4, 16), (2, 12)])
def test_agreement_is_independent_of_the_quadrature_rule(rule):
    """Both formulations integrate the same integrand, so they track rule for rule."""
    vertices = hexagon()
    target_r, target_z = ring(vertices, count=32)
    expected = complex_step_reference(target_r, target_z, vertices, *rule)
    computed = closed_form(target_r, target_z, vertices, *rule)
    for got, want in zip(computed, expected):
        scale = np.max(np.abs(want))
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12 * scale)


def test_targets_inside_the_conductor_agree():
    """The interior is where the finite-area kernel earns its cost."""
    vertices = hexagon()
    centre = vertices.mean(axis=0)
    offset = np.linspace(-0.02, 0.02, 17)
    target_r = centre[0] + offset
    target_z = np.full(offset.size, centre[1] + 0.005)
    expected = complex_step_reference(target_r, target_z, vertices)
    computed = closed_form(target_r, target_z, vertices)
    for got, want in zip(computed, expected):
        scale = np.max(np.abs(want))
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-11 * scale)


def test_target_blocking_does_not_change_the_result():
    """Splitting the target axis is a memory decision, not a numerical one.

    Not bit-for-bit: the per-edge node sum is a matrix-vector product, and BLAS
    picks its accumulation order from the operand shape, so a different block
    size reassociates the sum. The spread is round-off on a cancelling
    difference (~1e-11 relative), well under the quadrature error of any rule,
    but it does mean a stored operator is reproducible only for a fixed block.
    """
    vertices = hexagon()
    target_r, target_z = ring(vertices, count=48)
    whole = polygon_greens(target_r, target_z, vertices, block=None)
    split = polygon_greens(target_r, target_z, vertices, block=7)
    for got, want in zip(split, whole):
        np.testing.assert_allclose(
            got, want, rtol=1e-9, atol=1e-12 * np.max(np.abs(want))
        )


def test_public_kernel_reproduces_the_complex_step_field():
    """``polygon_greens`` output is the complex-step curl, to machine precision."""
    vertices = trapezium()
    target_r, target_z = ring(vertices, count=48)
    psi_ref, dpsi_dr, dpsi_dz = complex_step_reference(target_r, target_z, vertices)
    two_pi_r = 2.0 * np.pi * target_r
    psi, br, bz = polygon_greens(target_r, target_z, vertices)
    expected_bz = dpsi_dr / two_pi_r
    expected_br = -dpsi_dz / two_pi_r
    np.testing.assert_allclose(
        psi, psi_ref, rtol=1e-9, atol=1e-12 * np.max(np.abs(psi_ref))
    )
    np.testing.assert_allclose(
        bz, expected_bz, rtol=1e-9, atol=1e-12 * np.max(np.abs(expected_bz))
    )
    np.testing.assert_allclose(
        br, expected_br, rtol=1e-9, atol=1e-12 * np.max(np.abs(expected_br))
    )
