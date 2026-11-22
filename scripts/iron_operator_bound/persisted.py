"""Measure the demagnetisation bound on the persisted WEST interactions.

The affine Galerkin interaction ``G`` must satisfy
``0 <= x.T Q G x <= mu0 x.T Q x`` with ``Q`` the repeated cell Gram block, so
the quantity the bound constrains is the largest metric-weighted Rayleigh
quotient of ``G / mu0``.  This module reads operators that were assembled and
persisted earlier, so no assembly is repeated, and answers three questions with
one instrument:

* the largest quotient of each operator, with its eigen-residual;
* on the mesh whose factor refused, the most negative direction of the
  metric-symmetric Newton matrix at the seed state, its quotient against
  ``G / mu0`` and against that matrix, and the cells and cell pairs that carry
  it;
* a negative control: the same instrument on a copy of an operator with a
  planted rank-one term, which must read above the bound.

The bound is measured in a standard symmetric form.  ``sym(Q G)`` carries the
same quadratic form as ``Q G`` because the antisymmetric part contributes
nothing, and ``S = Q^-1/2 sym(Q G) Q^-1/2`` is symmetric with the same
eigenvalues, so its largest element is the largest quotient.  The per-cell
square roots also keep the solve honest: the interaction is unsymmetric, so
handing ``Q G`` and ``Q`` to an iterative eigensolver assumes a symmetry the
matrix does not have.

The Newton matrix is the one the in-place Cholesky route assembles.  At a state
the local material differential is ``A`` (six by six per cell, from the law's
tangent at the quadrature nodes), ``D = Q A``, ``R`` its symmetric square root
and ``W = Q^-1 R``; the rows written are ``I - R G W``.  The route reads only a
triangle of that matrix, so the object it actually factors is its symmetric
part, and that is what this module measures.  The solve seeds at a zero state,
so the tangent depends only on the field prescribed in the cells.

    python scripts/iron_operator_bound/persisted.py --mode bound \
        --label graded-16-80 --output <run-dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402

assert jax.config.jax_enable_x64

EVIDENCE = Path("/work/projects/imas_gpu/nova/evidence/iron-core-magnetisation")
OPERATORS = EVIDENCE / "linear-magnetisation" / "west-h200"

LABELS = {
    "uniform-0.08": (0.08, None),
    "uniform-0.04": (0.04, None),
    "graded-16-80": (0.016, 0.08),
}
CELLS_ACROSS = 20.0

# Share of the witness energy that names the cells carrying it, the number of
# cells whose pairs are reported, and the rows per cell.
CELL_SHARE = 0.90
PAIR_COUNT = 12
PAIR_ROWS = 3
PLANT = 0.05
PLANT_SEED = 4242


def cell_size(label, inputs):
    """Return the cell-size field named by a persisted operator's label."""
    fine, coarse = LABELS[label]
    if coarse is None:
        return fine
    return iron.graded_cell_size(
        inputs["surfaces"]["radial"],
        inputs["surfaces"]["vertical"],
        fine=fine,
        coarse=coarse,
        cells_across=CELLS_ACROSS,
    )


def layout(count):
    """Return the component-major position of each cell-major entry.

    The state and the interaction are ordered by component, then cell, then
    basis; the material differential and its roots are ordered by cell.  The
    mapping between the two is what every projection here reads through.
    """
    position = np.empty(6 * count, dtype=np.int64)
    for component in range(2):
        for basis in range(3):
            position[6 * np.arange(count) + 3 * component + basis] = (
                3 * component * count + 3 * np.arange(count) + basis
            )
    return position


def to_cells(vector, position):
    """Return a component-major vector as (cell, 6)."""
    cell = np.empty(len(vector))
    cell[position] = vector
    return cell.reshape(-1, 6)


def to_components(cells, position):
    """Return a (cell, 6) array as a component-major vector."""
    vector = np.empty(6 * len(cells))
    vector[position] = np.asarray(cells).reshape(-1)
    return vector


def gram_metric(gram):
    """Return the per-cell six-by-six metric from the three-by-three blocks."""
    metric = np.zeros((gram.shape[0], 6, 6))
    metric[:, :3, :3] = gram
    metric[:, 3:, 3:] = gram
    return metric


def symmetric_square_root(blocks):
    """Return the symmetric square root of each symmetric positive block.

    The SVD of a symmetric matrix is ``U S V^T`` with ``V = U`` up to column
    signs, so ``U sqrt(S) V^T`` is its symmetric square root and needs no
    second eigensolver beside the one this module reports.
    """
    left, singular, right = np.linalg.svd(blocks)
    return np.einsum("cij,cj,ckj->cik", left, np.sqrt(singular), right)


def cellwise(blocks, vector):
    """Apply per-cell six-by-six blocks to a component-major vector."""
    return jnp.einsum("cij,kci->kcj", blocks, vector.reshape(2, -1, 3)).reshape(-1)


def load(label, inputs):
    """Return the persisted interaction, its mesh and its linear geometry."""
    began = time.monotonic()
    path = OPERATORS / f"{label}-operator.npy"
    size = cell_size(label, inputs)
    interaction = iron.load_magnetisation_interaction(
        path, region=inputs["region"], cell_size=size, basis="linear"
    )
    cells = iron._mesh(inputs["region"], size)
    geometry = iron.linear_magnetisation_geometry(cells.polygons)
    gram = np.asarray(geometry["gram"])
    metric = gram_metric(gram)
    source = dict(
        label=label,
        path=path,
        interaction=interaction,
        cells=cells,
        geometry=geometry,
        count=len(cells.polygons),
        dimension=6 * len(cells.polygons),
        gram=gram,
        metric=metric,
        position=layout(len(cells.polygons)),
        root=symmetric_square_root(metric),
        load_seconds=time.monotonic() - began,
    )
    source["inverse"] = np.linalg.inv(source["root"])
    return source


def stage(source, *, modify=None):
    """Stage the interaction and return the actions every mode reads.

    ``modify(matrix, source)`` may rewrite the host matrix first, which is how
    the control plants its perturbation without a second definition of the
    quotient.
    """
    began = time.monotonic()
    matrix = np.asarray(
        source["matrix"] if "matrix" in source else source["interaction"].matrix
    )
    if modify is not None:
        matrix = modify(matrix, source)
    device = jax.device_put(np.asarray(matrix, dtype=np.float64))
    del matrix
    metric = jax.device_put(source["metric"])
    root = jax.device_put(source["root"])
    inverse = jax.device_put(source["inverse"])

    @jax.jit
    def action(vector):
        return device @ vector

    @jax.jit
    def metric_action(vector):
        return cellwise(metric, vector)

    @jax.jit
    def deflate(vector):
        return cellwise(inverse, vector)

    @jax.jit
    def symmetric_action(vector):
        left = cellwise(root, device @ cellwise(inverse, vector))
        right = cellwise(inverse, device.T @ cellwise(root, vector))
        return (left + right) / 2

    @jax.jit
    def state_action(vector):
        left = cellwise(metric, device @ vector)
        right = device.T @ cellwise(metric, vector)
        return (left + right) / 2

    source["state_action"] = lambda vector: np.asarray(
        state_action(jnp.asarray(vector))
    )
    source["device"] = device
    source["root_device"] = root
    source["inverse_device"] = inverse
    source["action"] = lambda vector: np.asarray(action(jnp.asarray(vector)))
    source["metric_action"] = lambda vector: np.asarray(
        metric_action(jnp.asarray(vector))
    )
    source["deflate"] = lambda vector: np.asarray(deflate(jnp.asarray(vector)))
    source["symmetric_action"] = lambda vector: np.asarray(
        symmetric_action(jnp.asarray(vector))
    )
    source["row"] = dict(
        label=source["label"],
        cells=source["count"],
        unknowns=source["dimension"],
        operator_path=str(source["path"]),
        geometry_digest=source.get("geometry_digest", "synthetic"),
        module=iron.__file__,
        cwd=str(Path.cwd().resolve()),
        load_seconds=source["load_seconds"],
        stage_seconds=time.monotonic() - began,
    )
    return source


def quotient(source, vector):
    """Return the metric-weighted Rayleigh quotient of ``G / mu0`` at ``vector``."""
    weighted = source["metric_action"](vector)
    return float(weighted @ source["action"](vector)) / (
        iron._MU0 * float(weighted @ vector)
    )


def bound(source, *, tolerance=1e-10, ncvec=48, seed=20261122):
    """Return the largest metric-weighted quotient and the state carrying it."""
    dimension = source["dimension"]
    operator = LinearOperator(
        (dimension, dimension), matvec=source["symmetric_action"], dtype=np.float64
    )
    began = time.monotonic()
    values, vectors = eigsh(
        operator,
        k=1,
        which="LA",
        tol=tolerance,
        ncv=ncvec,
        maxiter=4000,
        v0=np.random.default_rng(seed).normal(size=dimension),
    )
    value = float(values[-1])
    witness = source["deflate"](np.asarray(vectors[:, -1]))
    scaled = source["metric_action"](witness)
    action = source["state_action"](witness)
    return dict(
        quotient=value / iron._MU0,
        witness=witness,
        rayleigh=quotient(source, witness),
        metric_quadratic=float(scaled @ witness),
        residual=float(
            np.linalg.norm(action - value * scaled)
            / (abs(value) * np.linalg.norm(scaled))
        ),
        seconds=time.monotonic() - began,
    )


def external_coefficients(source, inputs):
    """Return the prescribed field in the affine basis, component-major.

    This is the projection the solve applies before its first residual: the
    field sampled at the quadrature nodes and tested with the same volume weight
    the Gram blocks carry.
    """
    geometry = source["geometry"]
    owners = np.asarray(geometry["owners"])
    project = np.einsum(
        "qij,qj,q->qi",
        np.linalg.inv(source["gram"])[owners],
        np.asarray(geometry["basis"]),
        np.asarray(geometry["weights"]),
    )
    values = np.asarray(inputs["external"](np.asarray(geometry["points"])))
    coefficients = np.zeros((2, source["count"], 3))
    for component in range(2):
        np.add.at(coefficients[component], owners, project * values[:, component, None])
    return coefficients.reshape(-1)


def material_differential(source, inputs):
    """Return the per-cell material differential at the seed state.

    The law is differentiated at the field each quadrature node sees, which is
    exactly what the residual's Jacobian carries at a zero state.
    """
    geometry = source["geometry"]
    owners = np.asarray(geometry["owners"])
    basis_values = np.asarray(geometry["basis"])
    project = jnp.asarray(
        np.einsum(
            "qij,qj,q->qi",
            np.linalg.inv(source["gram"])[owners],
            basis_values,
            np.asarray(geometry["weights"]),
        )
    )
    basis = jnp.asarray(basis_values)
    knots, coefficients, ends = iron._fit_law(inputs["table_b"], inputs["table_mu"])
    knots = jnp.asarray(knots)
    polynomial = jnp.asarray(coefficients)
    end_values = jnp.asarray(ends)
    total = jnp.asarray(external_coefficients(source, inputs)).reshape(2, -1, 3)
    values = jnp.einsum("cqi,qi->qc", total[:, owners], basis)

    def material(value):
        squared = jnp.sum(value**2)
        magnitude = jnp.where(
            squared > 0, jnp.sqrt(jnp.where(squared > 0, squared, 1.0)), 0.0
        )
        return value * iron._coefficient(magnitude, knots, polynomial, end_values)

    tangent = jax.vmap(jax.jacfwd(material))(values)
    integrand = jnp.einsum("qi,qab,qj->qaibj", project, tangent, basis)
    local = jnp.zeros((source["count"], 2, 3, 2, 3), dtype=values.dtype)
    local = local.at[owners].add(integrand).reshape(-1, 6, 6)
    return np.asarray(local), float(np.asarray(end_values)[0])


def newton_blocks(source, inputs):
    """Return the per-cell differential, its root and the route's transform."""
    local, origin = material_differential(source, inputs)
    differential = source["metric"] @ local
    differential = (differential + differential.transpose(0, 2, 1)) / 2
    root = symmetric_square_root(differential)
    transform = np.linalg.solve(source["metric"], root)
    try:
        np.linalg.cholesky(differential)
        definite = True
    except np.linalg.LinAlgError:
        definite = False
    return dict(
        local=local,
        differential=differential,
        root=root,
        transform=transform,
        definite=definite,
        origin=origin,
    )


def newton_actions(source, blocks):
    """Return the seeded Newton matrix's action, its symmetric part and skew.

    ``I - R G W`` is not symmetric, and the route reads only one triangle of
    it, so the matrix it factors is the symmetric part.  The skew part is
    returned too, because its size says how far the route's object sits from
    the matrix the formula states.
    """
    device = source["device"]
    root = jax.device_put(blocks["root"])
    transform = jax.device_put(blocks["transform"])

    @jax.jit
    def action(vector):
        return vector - cellwise(root, device @ cellwise(transform, vector))

    @jax.jit
    def symmetric(vector):
        left = cellwise(root, device @ cellwise(transform, vector))
        right = cellwise(transform, device @ cellwise(root, vector))
        return vector - (left + right) / 2

    @jax.jit
    def skew(vector):
        left = cellwise(root, device @ cellwise(transform, vector))
        right = cellwise(transform, device @ cellwise(root, vector))
        return (left - right) / 2

    return (
        lambda vector: np.asarray(action(jnp.asarray(vector))),
        lambda vector: np.asarray(symmetric(jnp.asarray(vector))),
        lambda vector: np.asarray(skew(jnp.asarray(vector))),
    )


def localise(source, vector, *, share=CELL_SHARE, pairs=PAIR_COUNT, rows=PAIR_ROWS):
    """Return the cells and cell pairs that carry a witness's energy.

    A witness ``w`` carries ``w.T Q G w``; cell ``c`` contributes
    ``(Q w)_c . (G w)_c`` and the pair ``(c, d)`` contributes
    ``(Q w)_c . (G[c, d] w_d)``, so the split sums to the numerator without a
    second quadratic form.  Pairs are taken from the cells that carry most of
    the witness, which is where a direction held by a few near pairs sits.
    """
    position = source["position"]
    weighted_vector = source["metric_action"](vector)
    product_vector = source["action"](vector)
    cell_vector = to_cells(vector, position)
    weighted = to_cells(weighted_vector, position)
    product = to_cells(product_vector, position)
    energy = np.einsum("ci,ci->c", weighted, product)
    total = float(weighted_vector @ product_vector)
    contribution = energy / total
    order = np.argsort(-np.abs(contribution))
    cumulative = np.cumsum(np.abs(contribution[order]))
    kept = order[: int(np.searchsorted(cumulative, share)) + 1]
    mesh = source["cells"]
    centroids = np.asarray(mesh.centroids)
    areas = np.asarray(mesh.areas)
    carried = []
    for index in kept:
        body = np.asarray(mesh.polygons[index])
        carried.append(
            dict(
                cell=int(index),
                share=float(contribution[index]),
                centroid_m=[float(value) for value in centroids[index]],
                area_m2=float(areas[index]),
                radial_extent_m=[float(body[:, 0].min()), float(body[:, 0].max())],
                vertical_extent_m=[float(body[:, 1].min()), float(body[:, 1].max())],
            )
        )
    selected = []
    for first in order[:pairs]:
        choice = np.zeros(source["dimension"])
        choice[position[6 * first : 6 * first + 6]] = cell_vector[first]
        local = to_cells(source["action"](choice), position)
        share_of = np.einsum("ci,ci->c", weighted, local) / total
        ranked = [int(i) for i in np.argsort(-np.abs(share_of)) if i != first]
        for second in ranked[:rows]:
            selected.append(
                dict(
                    first=int(first),
                    second=second,
                    share=float(share_of[second]),
                    centroid_separation_m=float(
                        np.linalg.norm(centroids[first] - centroids[second])
                    ),
                )
            )
    selected.sort(key=lambda item: -abs(item["share"]))
    return dict(
        total=total,
        weight_scale=float(np.sqrt(weighted_vector @ vector)),
        metric_quadratic=float(weighted_vector @ vector),
        cells=carried,
        pairs=selected[: pairs * rows],
    )


def check(size=0.04, *, seed=4):
    """Compare the Lanczos quotient against the owner's dense spectrum.

    The gate shell is small enough for the owner's dense solve, so the
    instrument's largest quotient is read against the owner's answer on the
    same assembled interaction rather than against a second Lanczos run.
    """
    region = iron.gate_shell()
    cells = iron._mesh(region, size)
    geometry = iron.linear_magnetisation_geometry(cells.polygons)
    operator, _ = iron.linear_galerkin_cell_interaction(cells.polygons, plan=iron._PLAN)
    owner = iron.galerkin_spectrum(cells, operator, basis="linear", witness=True)
    gram = np.asarray(geometry["gram"])
    metric = gram_metric(gram)
    count = len(cells.polygons)
    source = dict(
        label="gate-shell",
        cells=cells,
        geometry=geometry,
        count=count,
        dimension=6 * count,
        gram=gram,
        metric=metric,
        position=layout(count),
        root=symmetric_square_root(metric),
        matrix=np.asarray(operator),
        geometry_digest="gate-shell",
        load_seconds=0.0,
    )
    source["inverse"] = np.linalg.inv(source["root"])
    stage(source)
    peak = bound(source, ncvec=min(24, 2 * count), seed=seed)
    row = dict(
        cell_library=iron.__file__,
        device=str(jax.devices()[0]),
        x64=bool(jax.config.jax_enable_x64),
        cells=count,
        unknowns=source["dimension"],
        owner_maximum=owner["maximum"],
        owner_minimum=owner["minimum"],
        owner_volume_symmetry=owner["volume_symmetry"],
        lanczos_maximum=peak["quotient"],
        lanczos_rayleigh=peak["rayleigh"],
        lanczos_residual=peak["residual"],
        relative_difference=abs(peak["quotient"] - owner["maximum"])
        / abs(owner["maximum"]),
    )
    return row


def apply_metric(metric, vector):
    """Apply per-cell metric blocks to a component-major vector, in numpy."""
    return np.einsum("cij,kci->kcj", metric, vector.reshape(2, -1, 3)).reshape(-1)


def witness(source, inputs, peak, *, tolerance=1e-8, ncvec=48, seed=20261122, probes=4):
    """Measure the most negative direction of the seeded Newton matrix.

    ``peak`` is the bound result for the same operator, so the witness's
    quotient and alignment are read against one measurement of the operator's
    top direction rather than a second solve.
    """
    blocks = newton_blocks(source, inputs)
    action, symmetric, skew = newton_actions(source, blocks)
    dimension = source["dimension"]
    operator = LinearOperator(
        (dimension, dimension), matvec=symmetric, dtype=np.float64
    )
    random = np.random.default_rng(seed)
    began = time.monotonic()
    values, vectors = eigsh(
        operator,
        k=1,
        which="SA",
        tol=tolerance,
        ncv=ncvec,
        maxiter=4000,
        v0=random.normal(size=dimension),
    )
    direction = np.asarray(vectors[:, 0])
    acted = symmetric(direction)
    asymmetry = 0.0
    for _ in range(probes):
        trial = random.normal(size=dimension)
        trial /= np.linalg.norm(trial)
        asymmetry = max(asymmetry, float(np.linalg.norm(skew(trial))))
    weighted = source["metric_action"](direction)
    top = source["metric_action"](peak["witness"])
    row = dict(source["row"])
    row.update(
        newton_seconds=time.monotonic() - began,
        newton_minimum=float(values[0]),
        newton_residual=float(
            np.linalg.norm(acted - values[0] * direction)
            / (abs(values[0]) * np.linalg.norm(direction))
        ),
        newton_rayleigh=float(direction @ acted / (direction @ direction)),
        newton_asymmetry=asymmetry,
        differential_positive_definite=blocks["definite"],
        origin_coefficient_times_mu0=blocks["origin"] * iron._MU0,
        seed_definiteness_limit=1.0 / (1.0 - blocks["origin"] * iron._MU0),
        witness_operator_quotient=quotient(source, direction),
        operator_maximum=peak["quotient"],
        operator_rayleigh=peak["rayleigh"],
        alignment_with_operator=abs(
            float(weighted @ top) / (np.linalg.norm(weighted) * np.linalg.norm(top))
        ),
    )
    row["localisation"] = localise(source, direction)
    position = source["position"]
    weighted_vector = source["metric_action"](direction)
    energy = np.einsum(
        "ci,ci->c",
        to_cells(weighted_vector, position),
        to_cells(source["action"](direction), position),
    )
    return row, direction, energy / float(weighted_vector @ direction)


def plant(matrix, source):
    """Add a rank-one term that lifts the quotient by ``PLANT`` at one direction.

    With ``p`` normalised so ``p.T Q p = 1`` and ``G' = G + mu0 PLANT p (Q p).T``
    the added quadratic form is ``mu0 PLANT (x.T Q p)^2``, which is positive
    semidefinite, so the largest quotient rises by at least ``PLANT`` and a
    correct instrument reads above the bound.
    """
    direction = np.random.default_rng(PLANT_SEED).normal(size=source["dimension"])
    norm = float(apply_metric(source["metric"], direction) @ direction)
    direction /= np.sqrt(norm)
    weighted = apply_metric(source["metric"], direction)
    return matrix + iron._MU0 * PLANT * np.outer(direction, weighted)


def write(path, row):
    """Write a receipt and print its scalar fields."""
    path.write_text(json.dumps(row, indent=1) + "\n")
    scalars = {key: value for key, value in row.items() if np.isscalar(value)}
    print(json.dumps(scalars), flush=True)


def record_bound(label, inputs, output):
    """Measure one persisted operator's bound and write its receipt."""
    source = stage(load(label, inputs))
    peak = bound(source)
    np.save(output / f"bound-{label}-witness.npy", peak["witness"])
    row = dict(source["row"])
    row.update(
        method="lanczos-standard-symmetric",
        maximum=peak["quotient"],
        rayleigh=peak["rayleigh"],
        eigen_residual=peak["residual"],
        eigen_seconds=peak["seconds"],
        metric_quadratic=peak["metric_quadratic"],
    )
    write(output / f"bound-{label}.json", row)
    return source, peak


def record_witness(label, inputs, output):
    """Measure one operator's bound and the refusal's most negative direction."""
    source = stage(load(label, inputs))
    peak = bound(source)
    row, direction, shares = witness(source, inputs, peak)
    np.save(output / f"bound-{label}-witness.npy", peak["witness"])
    np.save(output / f"witness-{label}.npy", direction)
    np.save(output / f"shares-{label}.npy", shares)
    write(output / f"witness-{label}.json", row)
    return row


def record_control(label, inputs, output):
    """Measure the bound on a deliberately perturbed copy of an operator."""
    source = stage(load(label, inputs), modify=plant)
    peak = bound(source)
    row = dict(source["row"])
    row.update(
        method="lanczos-standard-symmetric",
        control="planted rank-one term",
        plant=PLANT,
        plant_seed=PLANT_SEED,
        maximum=peak["quotient"],
        rayleigh=peak["rayleigh"],
        eigen_residual=peak["residual"],
        eigen_seconds=peak["seconds"],
    )
    write(output / f"control-{label}.json", row)
    if not peak["quotient"] > 1.0 + PLANT / 2:
        raise SystemExit("the planted term lifted the quotient by less than half")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("bound", "witness", "control", "check"), required=True
    )
    parser.add_argument("--label", choices=tuple(LABELS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "check":
        write(args.output / "check.json", check())
        return
    from scripts.iron_graded_refinement.measure import west_inputs

    inputs = west_inputs()
    if args.mode == "bound":
        record_bound(args.label, inputs, args.output)
    elif args.mode == "witness":
        record_witness(args.label, inputs, args.output)
    else:
        record_control(args.label, inputs, args.output)


if __name__ == "__main__":
    main()
