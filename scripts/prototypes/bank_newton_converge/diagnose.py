"""Compare the bank and production moment equations at one physical state."""

# ruff: noqa: E402

from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import current, fixed_points, support_read
from scripts.prototypes.bank_newton_step.measure import machine
from scripts.prototypes.fast_map_production import program
from scripts.prototypes.fast_map_production.kernel import read as production_read
from scripts.prototypes.lean_map.jets import JetMap
from scripts.prototypes.lean_map.quadrature import quadratic_fit, triangle_rule
from scripts.prototypes.lean_map.values import build_values


def emit(name, **entries):
    print(name + " " + json.dumps(entries, default=float, sort_keys=True), flush=True)


def norm(array):
    return float(jnp.linalg.norm(jnp.asarray(array)))


def compare(kind, state_path=None):
    (
        operands,
        cache,
        roots,
        convention,
        policy,
        profile,
        quadrature,
        moments,
        analytic,
    ) = machine(kind)
    if state_path is not None:
        with np.load(state_path, allow_pickle=False) as stored:
            moments = tuple(jnp.asarray(part) for part in stored["moments"])
    geometry = operands.geometry
    state = jnp.stack(moments, axis=-1)
    polygons = tuple(
        np.asarray(geometry.vertices[i, :n])
        for i, n in enumerate(np.asarray(geometry.vertex_count))
    )
    rule = triangle_rule(geometry)
    fit, design = quadratic_fit(rule, geometry)
    values, receipt = build_values(
        SimpleNamespace(cell_polygons=polygons), rule, analytic, state, persist=False
    )
    jets = JetMap(
        operands.rows,
        operands.exterior,
        operands.nodes,
        operands.sample_index,
        operands.mesh,
    )
    axis_sign = float(jnp.sign(analytic.evaluate(roots[0]).hessian[0, 0]))
    target = jnp.sum(moments[0])
    route_quadrature = replace(quadrature, target_current=target)
    seed = roots + 0.03 * jnp.median(geometry.pitch)
    support = support_read.read(
        operands, cache, moments, seed, roots, convention, policy
    )
    bank = support_read.step_residual(
        operands, moments, moments, support, profile, route_quadrature
    )
    routed = program.map_moments(
        state,
        values,
        jets,
        cache,
        rule,
        fit,
        design,
        geometry,
        profile,
        roots,
        convention,
        policy,
        target,
        axis_sign,
    )

    cached = fixed_points._CachedField(
        cache.mesh, cache.lookup, cache.rows, state, cache.exterior
    )
    solved = fixed_points.solve(
        cache, state, roots, roots, jnp.median(geometry.pitch), axis_sign
    )
    gauss_flux = values.apply(state).reshape(rule.points.shape[:-1])
    gauss_coeff = jnp.einsum("nkq,nq->nk", fit, gauss_flux)
    slots = cache.lookup[jets.sample_index]
    complete = jnp.all(slots >= 0, axis=1)
    safe_slots = jnp.maximum(slots, 0)
    band_values = cache.exterior[safe_slots, 0] + jnp.einsum(
        "nscm,cm->ns", cache.rows[safe_slots, ..., 0], state
    )
    band_coeff = jnp.einsum("nks,ns->nk", geometry.fit_inverse, band_values)
    coefficient = jnp.where(complete[:, None], band_coeff, gauss_coeff)
    production = production_read(
        program.FusedField(cached, coefficient), solved, geometry, convention, policy
    )
    bank_on_route = current.book(
        support_read.nodal_jets(operands, moments)[operands.sample_index, 0],
        production,
        geometry,
        profile,
        route_quadrature,
    )

    def largest(part):
        array = np.asarray(part)
        index = int(np.nanargmax(np.abs(array)))
        return {
            "max": float(np.nanmax(np.abs(array))),
            "index": index,
            "value": float(array.reshape(-1)[index]),
        }

    emit(
        "STATE",
        kind=kind,
        bank_valid=bool(bank.valid),
        route_valid=bool(routed.booking_valid),
        base_residual=norm(jnp.stack(bank.residual, axis=-1)),
        route_residual=norm(routed.moment_residual),
        residual_diff=norm(jnp.stack(bank.residual, axis=-1) - routed.moment_residual),
        value_receipt=receipt,
    )
    emit(
        "READ",
        axis_shift=norm(support.reading.axis - production.axis),
        boundary_shift=norm(support.reading.boundary - production.boundary),
        axis_flux_gap=float(support.reading.axis_flux - production.axis_flux),
        boundary_flux_gap=float(
            support.reading.boundary_flux - production.boundary_flux
        ),
        membership_gap=largest(support.reading.membership - production.membership),
        coefficient_gap=largest(
            support.reading.field_coefficients - production.field_coefficients
        ),
        selected_gap=int(
            np.count_nonzero(
                np.asarray(support.reading.membership > 0)
                != np.asarray(production.membership > 0)
            )
        ),
        booking_cell_gap=int(
            np.count_nonzero(
                np.asarray(support.reading.normal_form_cells)
                != np.asarray(production.normal_form_cells)
            )
        ),
    )
    bank_selected = np.asarray(support.reading.membership > 0)
    route_selected = np.asarray(production.membership > 0)
    changed = np.flatnonzero(bank_selected != route_selected)
    emit(
        "READ_SELECTION",
        changed=changed.tolist(),
        bank_membership=np.asarray(support.reading.membership)[changed].tolist(),
        route_membership=np.asarray(production.membership)[changed].tolist(),
        route_path_weight=np.asarray(production.path_weight)[changed].tolist(),
        bank_normal=np.asarray(support.reading.normal_form_cells)[changed].tolist(),
        route_booking=np.asarray(production.booking_cells)[changed].tolist(),
        bank_moments=np.asarray(jnp.stack(bank.booked[:3], axis=-1))[changed].tolist(),
        route_moments=np.asarray(routed.moments)[changed].tolist(),
    )
    emit(
        "BOOKING",
        bank_vs_route_booking=norm(
            jnp.stack(bank.booked[:3], axis=-1) - routed.moments
        ),
        bank_with_route_read_vs_route_booking=norm(
            jnp.stack(bank_on_route[:3], axis=-1) - routed.moments
        ),
        bank_booking_vs_bank_with_route_read=norm(
            jnp.stack(bank.booked[:3], axis=-1) - jnp.stack(bank_on_route[:3], axis=-1)
        ),
        bank_moment_gap=largest(bank.booked.cell_current - routed.moments[:, 0]),
        bank_with_route_gap=largest(bank_on_route.cell_current - routed.moments[:, 0]),
    )


if __name__ == "__main__":
    import argparse
    import subprocess
    import nova

    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("diverted", "limited"), default="diverted")
    parser.add_argument("--state-path", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    emit(
        "HEADER",
        revision=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        tree=str(root),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
    )
    compare(args.kind, args.state_path)
