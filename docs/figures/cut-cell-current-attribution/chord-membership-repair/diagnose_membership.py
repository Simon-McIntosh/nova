"""Compare the carried field and shared spline at the exterior control cells."""

from __future__ import annotations

import jax
import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64 is True

import jax.numpy as jnp  # noqa: E402

from benchmarks import solovev_certificate as certificate  # noqa: E402
from nova.equilibrium.clip_quadrature import (  # noqa: E402
    _density_sample_field,
    _global_separatrix_membership,
)
from nova.equilibrium.forward_operator import (  # noqa: E402
    flux_field_polynomial,
    set_support_clip_mode,
)
from nova.linalg.split_spline import fit_split_spline  # noqa: E402
from scripts.analytic_oracle_fixtures import measure as fixture  # noqa: E402


def main() -> None:
    case = "diverted-single-null"
    carrier, source, exact = certificate._case(case)
    machine = certificate._case_machine(case, carrier, exact, -500)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = certificate._exact_state(case, exact, coordinates)
    operator = fixture.forward_operator(source, machine)
    set_support_clip_mode("chord")
    masks, topology, sample_psi_norm, support = operator._support_partition(
        jnp.asarray(analytic)
    )
    field = flux_field_polynomial(
        operator._support_moment_stencils, masks.psi_norm, sample_psi_norm
    )
    centre_values = np.asarray(field.coefficient)[:, 0]
    print(
        "CARRIED "
        f"active={int(np.count_nonzero(np.asarray(field.active)))} "
        f"value_sup={np.max(np.abs(centre_values - np.asarray(masks.psi_norm))):.12g} "
        f"centre_sup={np.max(np.abs(np.asarray(field.centre) - machine.node)):.12g}"
    )

    def spline(coordinate, values):
        value = jnp.asarray(values)[None, :]
        return fit_split_spline(
            jnp.asarray(coordinate)[None, :, 0],
            jnp.asarray(coordinate)[None, :, 1],
            value,
            value - 1.0,
            order=6,
            regularization=1.0e-14,
        )

    carried = spline(field.centre, centre_values)
    direct = spline(operator.grid.coordinate, masks.psi_norm)
    membership = _global_separatrix_membership(
        field, support.support_vertices, support.vertex_count
    )
    boundary_flux = float(exact.flux(np.asarray(exact.x_point)[None, :])[0])
    for cell in (7, 64):
        points = _density_sample_field(field, jnp.asarray([cell], dtype=jnp.int32))[0]
        carried_level = -carried._patch_evaluation(
            carried.level_set_coefficients, points[..., 0], points[..., 1]
        ).value
        direct_level = -direct._patch_evaluation(
            direct.level_set_coefficients, points[..., 0], points[..., 1]
        ).value
        exact_level = operator.polarity * (
            exact.flux(np.asarray(points).reshape(-1, 2)).reshape(points.shape[:-1])
            - boundary_flux
        )
        member = np.asarray(membership(points, jnp.asarray([cell], dtype=jnp.int32)))
        direct_positive = int(np.count_nonzero(np.asarray(direct_level) >= 0.0))
        print(
            f"CELL cell={cell} carried_positive="
            f"{int(np.count_nonzero(np.asarray(carried_level) >= 0.0))}/25 "
            f"direct_positive={direct_positive}/25 "
            f"exact_positive={int(np.count_nonzero(exact_level >= 0.0))}/25 "
            f"connected_positive={int(np.count_nonzero(member))}/25 "
            f"carried_range={float(jnp.min(carried_level)):.12g},"
            f"{float(jnp.max(carried_level)):.12g} "
            f"direct_range={float(jnp.min(direct_level)):.12g},"
            f"{float(jnp.max(direct_level)):.12g}"
        )


if __name__ == "__main__":
    main()
