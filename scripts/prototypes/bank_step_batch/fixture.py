"""Certificate case whose jet operands are cached, so a CPU arm can reuse them.

The jet build integrates every nodal source on the device and does not fit a
CPU allocation at the certificate's 64-cell resolution. Published once through
the shared operand cache, the same case loads from that cache in any later
process, on any device.
"""

import jax
import numpy as np

import tests.equilibrium.test_topology_read as fixture
from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, fixed_points, topology
from nova.equilibrium.clip_quadrature import ClippedCurrentMoments
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures.measure import analytic_profile
from scripts.prototypes.read_route.jets import build_jet_matrix
from scripts.prototypes.read_route.measure import sample_field


def build_case(kind="diverted", cells=64, policy=0.054806712567833996, persist=True):
    """Geometry, analytic field, states and bank operands for one certificate."""
    oracle, analytic, wall, axis, saddle = fixture._analytic_inputs(kind)
    geometry = fixture._realised_hex_geometry(wall, cells)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    read_policy = TopologyPolicy(
        normal_form_radius=policy if saddle is not None else 0.0
    )
    analytic_read = jax.jit(topology.read)(analytic, geometry, convention, read_policy)
    sampled_read = jax.jit(topology.read)(
        sample_field(analytic, geometry), geometry, convention, read_policy
    )
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    _, source, _ = certificate._case(name, clip_mode="exact")
    profile = analytic_profile(source)
    psi = jax.vmap(jax.vmap(analytic.value))(geometry.sample_points)
    reference = fixture._kernel_backed_field(kind, oracle, analytic, geometry)
    quadrature = current.Quadrature(sigma=convention.sigma, segments=64)

    def book(reading):
        return ClippedCurrentMoments(
            *current.book(psi, reading, geometry, profile, quadrature)[:3]
        )

    states = {"converged": book(analytic_read), "trial": book(sampled_read)}
    roots = np.asarray([axis] if saddle is None else [axis, saddle])
    matrix, exterior, nodes, sample_index, _build = build_jet_matrix(
        reference.coupling,
        geometry,
        analytic,
        reference.moments,
        persist=persist,
    )
    jax.block_until_ready(matrix)
    mesh = fixed_points.build_mesh(nodes, sample_index, geometry)
    return dict(
        kind=kind,
        geometry=geometry,
        analytic=analytic,
        convention=convention,
        policy=read_policy,
        states=states,
        roots=roots,
        matrix=matrix,
        exterior=exterior,
        nodes=nodes,
        sample_index=sample_index,
        mesh=mesh,
        profile=profile,
        quadrature=quadrature,
    )
