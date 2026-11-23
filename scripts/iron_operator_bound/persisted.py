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
  rank-one term planted along its own top direction, which must raise the
  largest quotient by exactly the planted amount.

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
and ``W = Q^-1 R``; the rows written are ``I - R G W``.  That matrix is not
symmetric, and the route factors it with ``symmetrize_input=False``, which
reads one triangle and reflects it: the object factored is the symmetric part
plus the lower triangle of the skew part, reflected, not the symmetric part
alone.  Both are measured here, so a refusal can be attributed to whichever of
them is indefinite.  The solve seeds at a zero state, so the tangent depends
only on the field prescribed in the cells.

For the seed state the relation is closed.  A uniform tangent is ``A = c I``,
so ``R = sqrt(c) Q^1/2`` and ``W = sqrt(c) Q^-1/2``, and the symmetric part is
``I - c S`` with ``S = Q^-1/2 sym(Q G) Q^-1/2``.  The largest element of ``S``
is the largest quotient times ``mu0``, so the seed matrix is definite exactly
while the largest quotient stays below ``1 / (c mu0)``: a linear law of slope
``c`` amplifies every metric quotient by ``c mu0`` and the material is what
sets the distance to the singularity, not the bound.

``c`` is the differential susceptibility ``dM/dB``, the tangent of the law.
The fitted law's origin value is the reluctivity ``dH/dB``: the law reads
``M = (1/mu0 - reluctivity) B``, so the two slopes differ by ``1/mu0``.  The
witness row carries the measured per-cell band of the tangent, so the closed
form is evaluated against what the differential actually is rather than
against the origin value.

    python scripts/iron_operator_bound/persisted.py --mode bound \
        --label graded-16-80 --output <run-dir>
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import shapely
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

    ``U sqrt(S) V.T`` squares to the block but is not symmetric when a singular
    value repeats, and this metric repeats every eigenvalue between its two
    components.  Each left singular vector of a positive block is an
    eigenvector, so ``sum sqrt(s_i) u_i u_i.T`` is the symmetric root and needs
    no second eigensolver beside the one this module reports.
    """
    left, singular, _ = np.linalg.svd(blocks)
    return np.einsum("cij,cj,ckj->cik", left, np.sqrt(singular), left)


def cellwise(blocks, vector):
    """Apply per-cell six-by-six blocks to a component-major vector.

    A block's row and column both index (component, basis) and couple the two
    components, so the product is taken on the cell-major view and returned to
    the state's component-major order.
    """
    rows = np.asarray(blocks)
    cells = np.asarray(vector).reshape(2, -1, 3).transpose(1, 0, 2).reshape(-1, 6)
    acted = np.einsum("cij,cj->ci", rows, cells)
    return acted.reshape(-1, 2, 3).transpose(1, 0, 2).reshape(-1)


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
        geometry_digest=interaction.geometry_digest,
        load_seconds=time.monotonic() - began,
    )
    source["inverse"] = np.linalg.inv(source["root"])
    return source


def stage(source):
    """Stage the interaction and return the actions every mode reads.

    The actions are plain array products.  A jitted closure over the matrix
    makes JAX capture it as a constant of the compiled executable, and lowering
    reported 9.9 GB captured for the graded operator and 12.7 GB for the
    uniform forty-millimetre one, which the allocator cannot hold beside the
    matrix itself.  A single matrix-vector product gains nothing from a
    compilation, so this is BLAS and the matrix stays an ordinary array.
    """
    began = time.monotonic()
    matrix = np.array(
        source["matrix"] if "matrix" in source else source["interaction"].matrix,
        dtype=np.float64,
    )
    metric = np.asarray(source["metric"], dtype=np.float64)
    root = np.asarray(source["root"], dtype=np.float64)
    inverse = np.asarray(source["inverse"], dtype=np.float64)

    def symmetric_action(vector):
        left = cellwise(root, matrix @ cellwise(inverse, vector))
        right = cellwise(inverse, matrix.T @ cellwise(root, vector))
        return (left + right) / 2

    def state_action(vector):
        left = cellwise(metric, matrix @ vector)
        right = matrix.T @ cellwise(metric, vector)
        return (left + right) / 2

    source["state_action"] = state_action
    source["operator"] = matrix
    source["metric_blocks"] = metric
    source["root_blocks"] = root
    source["inverse_blocks"] = inverse
    source["action"] = lambda vector: matrix @ vector
    source["metric_action"] = lambda vector: cellwise(metric, vector)
    source["deflate"] = lambda vector: cellwise(inverse, vector)
    source["symmetric_action"] = symmetric_action
    source["row"] = dict(
        label=source["label"],
        cells=source["count"],
        unknowns=source["dimension"],
        operator_path=str(source.get("path", "synthetic")),
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


def generalised_spectrum(blocks, metric):
    """Return the extreme eigenvalues of each block against the metric block.

    ``Q`` is symmetric positive definite, so ``C`` with ``Q = C C.T`` turns the
    generalised problem ``D z = l Q z`` into the ordinary one for
    ``C**-1 D C**-T``, which is symmetric and whose signed singular values are
    its eigenvalues.

    ``nova.biot.magnetisation.galerkin_block_spectrum`` is the owner of the
    global form of this problem: it assembles one ``block_diag`` metric for the
    whole mesh and returns the extremes of a single assembled energy against
    it.  What is read here is the batched problem instead, 6632 independent
    6-by-6 generalised eigenproblems against the per-cell gram block, one per
    cell, and the owner's signature expresses the global solve rather than the
    batch.  The conventions are still the owner's: the blocks are the same
    gram blocks, and the caller multiplies by ``mu0`` to read the band the
    closed form uses.
    """
    factor = np.linalg.cholesky(metric)
    inverse = np.linalg.inv(factor)
    reduced = np.einsum("cij,cjk,clk->cil", inverse, blocks, inverse)
    left, singular, right = np.linalg.svd(reduced)
    sign = np.einsum("cji,cij->ci", left, right)
    values = sign * singular
    return float(np.min(values)), float(np.max(values))


def newton_blocks(source, inputs):
    """Return the per-cell differential, its root and the route's transform.

    The route forms ``D = Q A``, takes ``R = sqrt(sym D)`` with negative
    eigenvalues clamped at zero, sets ``W = Q**-1 R`` and factors
    ``B = I - R G W``.  A symmetric block's left singular vectors are its
    eigenvectors, so the signed polar root built here is that decomposition
    with its negative eigenvalues dropped, and a differential the route would
    have refused lands on the root the route would have used.

    The differential's eigenvalues against the metric are returned beside the
    ones against the identity.  The differential is ``dM/dB``, the
    susceptibility, and the fitted law's origin value is the reluctivity
    ``dH/dB``; the two differ by ``1/mu0`` and only the susceptibility enters
    the closed form.  For a uniform linear law the material blocks are ``A = c
    I`` with ``c`` the susceptibility, so the generalised spectrum is that one
    repeated value; how far the cells depart from it is what makes the
    seed-state Newton matrix's definiteness a per-cell statement.
    """
    local, origin_reluctivity = material_differential(source, inputs)
    differential = source["metric"] @ local
    differential = (differential + differential.transpose(0, 2, 1)) / 2
    root, minimum = signed_root(differential)
    transform = np.linalg.solve(source["metric"], root)
    low, high = generalised_spectrum(differential, source["metric"])
    try:
        np.linalg.cholesky(differential)
        definite = True
    except np.linalg.LinAlgError:
        definite = False
    return dict(
        local=local,
        differential=differential,
        minimum=minimum,
        generalised_minimum=low,
        generalised_maximum=high,
        block_asymmetry=float(
            np.linalg.norm(local - local.transpose(0, 2, 1)) / np.linalg.norm(local)
        ),
        root=root,
        transform=transform,
        definite=definite,
        origin_reluctivity=origin_reluctivity,
    )


def newton_actions(source, blocks):
    """Return the seeded Newton matrix's action, its symmetric part and skew.

    ``B = I - R G W`` is symmetric only up to the antisymmetric part of ``G``
    in the ``Q`` inner product: its transpose carries ``W.T G.T R``, not
    ``W G R``.  The route reads one triangle of ``B`` as if it were symmetric,
    so the matrix it factors is ``(B + B.T)/2`` and that is what the eigen-solve
    here takes.  The skew part is returned too, because its size says how far
    the route's object sits from the matrix its own docstring states.
    """
    operator = source["operator"]
    root = np.asarray(blocks["root"])
    transform = np.asarray(blocks["transform"])
    transpose = np.moveaxis(transform, 1, 2)

    def left(vector):
        return cellwise(root, operator @ cellwise(transform, vector))

    def right(vector):
        return cellwise(transpose, operator.T @ cellwise(root, vector))

    def action(vector):
        return vector - left(vector)

    def symmetric(vector):
        return vector - (left(vector) + right(vector)) / 2

    def skew(vector):
        return (right(vector) - left(vector)) / 2

    return action, symmetric, skew


def block_matrix(blocks, count):
    """Expand per-cell blocks into the full matrix in the state's own order."""
    position = layout(count)
    full = np.zeros((6 * count, 6 * count), dtype=np.float64)
    for cell in range(count):
        index = position[6 * cell : 6 * cell + 6]
        full[np.ix_(index, index)] = blocks[cell]
    return full


def newton_identity(count=3, seed=7):
    """Compare the Newton actions against the dense matrices they state.

    The action path is where a transposed root or a mistransposed transform
    would hide, so it is checked against a literal ``I - R G W`` on a synthetic
    per-cell symmetric differential and an unsymmetric interaction.
    """
    rng = np.random.default_rng(seed)
    dimension = 6 * count
    metric = np.zeros((count, 6, 6))
    local = np.zeros((count, 6, 6))
    for cell in range(count):
        for component in range(2):
            factor = rng.normal(size=(3, 3))
            block = factor @ factor.T + 0.5 * np.eye(3)
            span = slice(3 * component, 3 * component + 3)
            metric[cell, span, span] = block
            factor = rng.normal(size=(3, 3))
            local[cell, span, span] = factor @ factor.T + 0.1 * np.eye(3)
    differential = metric @ local
    differential = (differential + differential.transpose(0, 2, 1)) / 2
    root, _ = signed_root(differential)
    transform = np.linalg.solve(metric, root)
    interaction = rng.normal(size=(dimension, dimension))
    source = dict(operator=interaction, dimension=dimension)
    blocks = dict(root=root, transform=transform)
    action, symmetric, skew = newton_actions(source, blocks)
    dense = np.eye(dimension) - (
        block_matrix(root, count) @ interaction @ block_matrix(transform, count)
    )
    dense_symmetric = (dense + dense.T) / 2
    dense_skew = (dense - dense.T) / 2
    error = 0.0
    for _ in range(3):
        vector = rng.normal(size=dimension)
        vector /= np.linalg.norm(vector)
        scale = max(np.linalg.norm(dense @ vector), np.finfo(np.float64).tiny)
        error = max(
            error,
            float(np.linalg.norm(action(vector) - dense @ vector) / scale),
            float(np.linalg.norm(symmetric(vector) - dense_symmetric @ vector) / scale),
            float(np.linalg.norm(skew(vector) - dense_skew @ vector) / scale),
        )
    return error


def seed_relation(count=8, seed=11):
    """Compare the closed-form seed minimum against the dense symmetric part.

    A uniform differential ``A = alpha I`` makes the route write ``I -
    alpha Q^1/2 G Q^-1/2``, whose symmetric part is ``I - alpha S`` with ``S =
    Q^-1/2 sym(Q G) Q^-1/2``.  Its smallest element is ``1 - (alpha mu0) q`` at
    the operator's largest quotient ``q``, so the closed form a witness row is
    attributed against is checked here through the same block and action paths
    that carry the measurement.
    """
    rng = np.random.default_rng(seed)
    gram = np.zeros((count, 3, 3))
    for cell in range(count):
        factor = rng.normal(size=(3, 3))
        gram[cell] = factor @ factor.T + np.eye(3)
    metric = gram_metric(gram)
    root_blocks = symmetric_square_root(metric)
    dimension = 6 * count
    interaction = rng.normal(size=(dimension, dimension))
    interaction = (interaction + interaction.T) / 2
    source = dict(
        label="seed-relation",
        metric=metric,
        root=root_blocks,
        inverse=np.linalg.inv(root_blocks),
        count=count,
        dimension=dimension,
        position=layout(count),
        matrix=interaction,
        load_seconds=0.0,
    )
    stage(source)
    peak = bound(source, ncvec=min(24, 2 * count), seed=seed)
    alpha = 0.95 / (iron._MU0 * peak["quotient"])
    differential = metric @ (alpha * np.eye(6))[None, :, :]
    differential = (differential + differential.transpose(0, 2, 1)) / 2
    root, _ = signed_root(differential)
    transform = np.linalg.solve(metric, root)
    dense = np.eye(dimension) - block_matrix(root, count) @ interaction @ block_matrix(
        transform, count
    )
    minimum = float(np.linalg.eigvalsh((dense + dense.T) / 2)[0])
    predicted = 1.0 - alpha * iron._MU0 * peak["quotient"]
    return abs(minimum - predicted) / abs(predicted)


def signed_root(blocks):
    """Return the symmetric square root with negatives clamped, and its minimum.

    For a symmetric block each left singular vector is its right one up to
    sign, so ``u_i`` is an eigenvector with eigenvalue ``+/- s_i`` and
    ``u_i . v_i`` carries the sign.  The root is ``sum sqrt(s_i) u_i u_i.T``
    over the positive eigenvalues, which is symmetric and equals the clamped
    root ``eigh`` returns, and the minimum eigenvalue is the smallest signed
    singular value.  ``numpy.linalg.svd`` returns ``V`` transposed, so the sign
    is taken against ``right`` with the left vectors' first axis last.
    """
    left, singular, right = np.linalg.svd(blocks)
    sign = np.einsum("cji,cij->ci", left, right)
    eigenvalues = sign * singular
    weight = np.where(eigenvalues > 0.0, np.sqrt(singular), 0.0)
    root = np.einsum("cij,cj,ckj->cik", left, weight, left)
    return root, float(np.min(eigenvalues))


def numerator_matrix(source):
    """Return ``Q G`` materialised, the energy a witness is weighted by."""
    return cellwise(source["metric_blocks"], source["operator"])


def localise(source, vector, *, share=CELL_SHARE, pairs=PAIR_COUNT, rows=PAIR_ROWS):
    """Return the cells and cell pairs that carry a witness's energy.

    The split is the owner's: ``localise.localise`` reads ``x_i.T E_ij x_j``
    for the self cells and for every cell pair from a materialised energy
    matrix, and the energy a witness is weighted by is ``Q G``.  This
    materialises that one matrix, hands it to the owner and normalises the
    owner's contributions by the numerator ``w.T Q G w``, so the shares sum to
    one and the enumeration is the owner's rather than a second copy of it.
    """
    from scripts.iron_operator_bound.localise import localise as split

    weighted_vector = source["metric_action"](vector)
    product_vector = source["action"](vector)
    quadratic = float(weighted_vector @ product_vector)
    total = quadratic / iron._MU0
    owner = split(
        [np.asarray(body) for body in source["cells"].polygons],
        {0: numerator_matrix(source)},
        np.asarray(vector, dtype=np.float64),
    )
    mesh = source["cells"]
    centroids = np.asarray(mesh.centroids)
    areas = np.asarray(mesh.areas)
    carried = []
    for entry in owner["cells"]:
        index = entry["cell"]
        bounds = entry["bounds_m"]
        carried.append(
            dict(
                cell=int(index),
                share=float(entry["self_contribution"] / total),
                centroid_m=[float(value) for value in centroids[index]],
                area_m2=float(entry["area_m2"]),
                radial_extent_m=[float(bounds[0]), float(bounds[2])],
                vertical_extent_m=[float(bounds[1]), float(bounds[3])],
            )
        )
    selected = [
        dict(
            first=int(entry["first"]),
            second=int(entry["second"]),
            share=float(entry["contribution"] / total),
            centroid_separation_m=float(entry["spacing_m"]),
        )
        for entry in owner["pairs"][: pairs * rows]
    ]
    return dict(
        total=quadratic,
        weight_scale=float(np.sqrt(quadratic)),
        metric_quadratic=float(weighted_vector @ vector),
        cells=carried,
        pairs=selected,
        area_total_m2=float(np.sum(np.asarray(areas))),
    )


CHECK_TOLERANCE = dict(
    relative_difference=1e-12,
    newton_identity_error=1e-12,
    seed_relation_error=1e-10,
)


def check(size=0.04, *, seed=4, mutate=None):
    """Compare the Lanczos quotient against the owner's dense spectrum.

    The gate shell is small enough for the owner's dense solve, so the
    instrument's largest quotient is read against the owner's answer on the
    same assembled interaction rather than against a second Lanczos run.
    Each calibration identity carries a stated tolerance in
    ``CHECK_TOLERANCE`` and :func:`check_verdict` reports the ones that miss
    it, so ``--mode check`` exits nonzero on a regression instead of printing
    a fresh reading.  ``mutate`` displaces one identity by a deliberate
    amount, which is how the guard is shown to fire.
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
        backend="numpy-blas",
        x64=True,
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
        newton_identity_error=newton_identity(),
        seed_relation_error=seed_relation(),
        regularisation=1e-6,
    )
    if mutate is not None:
        if mutate not in CHECK_TOLERANCE:
            raise SystemExit(f"no calibration identity named {mutate!r}")
        row[mutate] = row[mutate] + 1000.0 * CHECK_TOLERANCE[mutate]
    return row


def check_verdict(row):
    """Return the calibration identities that exceed their stated tolerance."""
    return [
        (name, row[name], tolerance)
        for name, tolerance in CHECK_TOLERANCE.items()
        if not abs(row[name]) <= tolerance
    ]


def apply_metric(metric, vector):
    """Apply per-cell metric blocks to a component-major vector."""
    return cellwise(metric, vector)


def skew_norm(skew, dimension, *, tolerance=1e-8, ncvec=48, seed=424242):
    """Return the spectral norm of an antisymmetric operator.

    The operator's square is negative semidefinite, so the largest singular
    value is the square root of the largest eigenvalue of ``-A A``.  The
    relative residual of that eigenpair is not reported: the eigenvalue of
    ``-A A`` is the square of a norm of a matrix that is zero to round-off, so
    the denominator a relative residual divides by is itself round-off and the
    ratio says nothing about convergence.  The norm is what is compared against
    the symmetric matrix's smallest eigenvalue, and it is reported alone.
    """
    operator = LinearOperator(
        (dimension, dimension),
        matvec=lambda vector: -skew(skew(vector)),
        dtype=np.float64,
    )
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
    return float(np.sqrt(max(value, 0.0)))


def witness(source, inputs, peak, *, tolerance=1e-8, ncvec=48, seed=20261122, probes=4):
    """Measure the most negative direction of the seeded Newton matrix.

    ``peak`` is the bound result for the same operator, so the witness's
    quotient and alignment are read against one measurement of the operator's
    top direction rather than a second solve.

    The matrix the route actually factors is ``tril(B)`` reflected, which is
    ``sym(B)`` plus the reflected lower triangle of ``B``'s skew part.  That
    addition has norm at most the spectral norm of the skew part, so a skew
    norm below the symmetric matrix's smallest eigenvalue proves the factored
    object is positive definite however the route reads it.
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
        asymmetry = max(
            asymmetry,
            float(
                np.linalg.norm(skew(trial))
                / max(np.linalg.norm(symmetric(trial)), np.finfo(np.float64).tiny)
            ),
        )
    weighted = source["metric_action"](direction)
    top = source["metric_action"](peak["witness"])
    skew_spectral = skew_norm(skew, dimension)
    row = dict(source["row"])
    row.update(
        newton_seconds=time.monotonic() - began,
        newton_minimum=float(values[0]),
        newton_residual=float(
            np.linalg.norm(acted - values[0] * direction)
            / (abs(values[0]) * np.linalg.norm(direction))
        ),
        newton_rayleigh=float(direction @ acted / (direction @ direction)),
        newton_skew_fraction=asymmetry,
        newton_skew_norm=skew_spectral,
        differential_minimum=blocks["minimum"],
        material_generalised_minimum=blocks["generalised_minimum"],
        material_generalised_maximum=blocks["generalised_maximum"],
        material_anisotropy=blocks["generalised_maximum"]
        / blocks["generalised_minimum"],
        material_block_asymmetry=blocks["block_asymmetry"],
        differential_positive_definite=blocks["definite"],
        origin_reluctivity_times_mu0=blocks["origin_reluctivity"] * iron._MU0,
        susceptibility_lower_times_mu0=blocks["generalised_minimum"] * iron._MU0,
        susceptibility_upper_times_mu0=blocks["generalised_maximum"] * iron._MU0,
        seed_limit_quotient=1.0 / (blocks["generalised_maximum"] * iron._MU0),
        seed_definite_quotient_interval=[
            1.0 / (blocks["generalised_maximum"] * iron._MU0),
            1.0 / (blocks["generalised_minimum"] * iron._MU0),
        ],
        predicted_newton_lower=1.0
        - blocks["generalised_maximum"] * iron._MU0 * peak["quotient"],
        predicted_newton_upper=1.0
        - blocks["generalised_minimum"] * iron._MU0 * peak["quotient"],
        witness_operator_quotient=quotient(source, direction),
        operator_maximum=peak["quotient"],
        operator_rayleigh=peak["rayleigh"],
        alignment_euclid=abs(
            float(direction @ np.asarray(peak["witness"], dtype=np.float64))
            / (
                np.linalg.norm(direction)
                * np.linalg.norm(np.asarray(peak["witness"], dtype=np.float64))
            )
        ),
        alignment_q=abs(
            float(weighted @ peak["witness"])
            / np.sqrt(
                max(float(weighted @ direction) * float(top @ peak["witness"]), 0.0)
            )
        ),
    )
    row["localisation"] = localise(source, direction)
    position = source["position"]
    weighted_vector = source["metric_action"](direction)
    product = source["action"](direction)
    energy = np.einsum(
        "ci,ci->c",
        to_cells(weighted_vector, position),
        to_cells(product, position),
    )
    return row, direction, energy / float(weighted_vector @ product)


def plant(matrix, source, direction):
    """Add a rank-one term that lifts the largest quotient by ``PLANT``.

    With ``p`` normalised so ``p.T Q p = 1`` and
    ``G' = G + mu0 PLANT p (Q p).T`` the added quadratic form is
    ``mu0 PLANT (x.T Q p)^2``, so every quotient rises by
    ``PLANT (x.T Q p)^2 / (x.T Q x)``.  At ``x = p`` that shift is exactly
    ``PLANT``, and at the top eigenvector it is ``PLANT`` times the squared
    Q-overlap of ``p`` with it.  A random direction has an overlap of order
    ``1 / n``, so its lift is ``PLANT / n`` and unmeasurable; the control plants
    along the direction its own bound solve found, which is the only choice that
    tests the instrument rather than the arithmetic of an overlap.
    """
    direction = np.asarray(direction, dtype=np.float64)
    norm = float(apply_metric(source["metric"], direction) @ direction)
    direction = direction / np.sqrt(norm)
    weighted = apply_metric(source["metric"], direction)
    return matrix + iron._MU0 * PLANT * np.outer(direction, weighted)


def write(path, row):
    """Write a receipt and print its scalar fields."""
    path.write_text(json.dumps(row, indent=1) + "\n")
    scalars = {key: value for key, value in row.items() if np.isscalar(value)}
    print(json.dumps(scalars), flush=True)


def record_bound(label, inputs, output, *, tolerance=1e-10):
    """Measure one persisted operator's bound and write its receipt."""
    source = stage(load(label, inputs))
    peak = bound(source, tolerance=tolerance)
    np.save(output / f"bound-{label}-witness.npy", peak["witness"])
    row = dict(source["row"])
    row.update(
        method="lanczos-standard-symmetric",
        maximum=peak["quotient"],
        rayleigh=peak["rayleigh"],
        eigen_residual=peak["residual"],
        eigen_seconds=peak["seconds"],
        metric_quadratic=peak["metric_quadratic"],
        eigen_tolerance=tolerance,
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


def record_control(label, inputs, output, *, tolerance=1e-4):
    """Measure the bound on a deliberately perturbed copy of an operator.

    The perturbation is planted along the top direction the unperturbed solve
    found, so the largest quotient must rise by ``PLANT`` exactly.  The lift is
    compared against that value rather than against a bare threshold: a threshold
    would pass for any rise of the right sign, and the lift is what says the
    planted term was read at its true size.
    """
    source = load(label, inputs)
    baseline = bound(stage(source))
    direction = np.asarray(baseline["witness"], dtype=np.float64)
    matrix = np.asarray(source["interaction"].matrix)
    planted_matrix = plant(matrix, source, direction)
    del matrix
    perturbed = stage(dict(source, matrix=planted_matrix))
    peak = bound(perturbed)
    lift = peak["quotient"] - baseline["quotient"]
    row = dict(perturbed["row"])
    row.update(
        method="lanczos-standard-symmetric",
        control="rank-one term planted along the operator's top direction",
        plant=PLANT,
        baseline=baseline["quotient"],
        maximum=peak["quotient"],
        lift=lift,
        lift_expected=PLANT,
        rayleigh=peak["rayleigh"],
        eigen_residual=peak["residual"],
        eigen_seconds=peak["seconds"],
        baseline_residual=baseline["residual"],
    )
    write(output / f"control-{label}.json", row)
    if not abs(lift - PLANT) <= tolerance:
        raise SystemExit(
            f"the planted term lifted the quotient by {lift:.9f}, not {PLANT}"
        )
    return row


def region_record(label, inputs, output):
    """Describe the mesh cells a witness receipt names as carrying the energy.

    Three causes are separated by measurement rather than by eye.  A cell on a
    grading transition has neighbours whose equivalent size differs from its
    own.  A near-singular cell pair has a non-touching cell closer than its own
    size, which is where the boundary integral's kernel is steepest.  A
    distorted cell departs from the mesh's typical compactness, read as
    ``4 pi area / perimeter^2``, which is 1 for a disc and falls for an
    elongated or slivered polygon.
    """
    receipt = json.loads((output / f"witness-{label}.json").read_text())
    carried = [entry["cell"] for entry in receipt["localisation"]["cells"]]
    size = cell_size(label, inputs)
    cells = iron._mesh(inputs["region"], size)
    bodies = [shapely.Polygon(p) for p in cells.polygons]
    areas = np.asarray([body.area for body in bodies])
    equivalent = np.sqrt(areas)
    compactness = np.asarray(
        [4 * np.pi * body.area / body.length**2 for body in bodies]
    )
    tree = shapely.STRtree(bodies)
    rows = []
    for index in carried:
        body = bodies[index]
        own = float(equivalent[index])
        touching = [
            int(other)
            for other in tree.query(body, predicate="intersects")
            if other != index
        ]
        neighbours = (
            np.asarray([equivalent[other] for other in touching])
            if touching
            else np.asarray([])
        )
        near = [
            int(other)
            for other in tree.query(body, predicate="dwithin", distance=4.0 * own)
            if other != index
        ]
        separated = [
            (int(other), float(body.distance(bodies[other]))) for other in near
        ]
        apart = [(gap, other) for other, gap in separated if gap > 0.0]
        nearest_gap, nearest_cell = min(apart) if apart else (float("nan"), -1)
        same_lobe = [
            (float(body.distance(bodies[other])), int(other))
            for other in carried
            if other != index
        ]
        lobe_gap, lobe_cell = min(same_lobe) if same_lobe else (float("nan"), -1)
        bounds = body.bounds
        rows.append(
            dict(
                cell=int(index),
                centroid_m=[float(value) for value in body.centroid.coords[0]],
                area_m2=float(areas[index]),
                equivalent_size_m=own,
                vertex_count=len(np.asarray(cells.polygons[index])),
                compactness=float(compactness[index]),
                mesh_compactness_median=float(np.median(compactness)),
                neighbours=len(touching),
                neighbour_size_min_m=float(neighbours.min())
                if len(neighbours)
                else float("nan"),
                neighbour_size_max_m=float(neighbours.max())
                if len(neighbours)
                else float("nan"),
                neighbour_size_ratio=float(neighbours.max() / neighbours.min())
                if len(neighbours)
                else float("nan"),
                closest_separated_cell=nearest_cell,
                closest_separated_m=nearest_gap,
                closest_separated_over_size=nearest_gap / own,
                nearest_carrying_pair_cell=lobe_cell,
                nearest_carrying_pair_m=lobe_gap,
                nearest_carrying_pair_over_size=lobe_gap / own,
                bounds_m=[float(value) for value in bounds],
            )
        )
    return dict(
        label=label,
        cells=int(len(bodies)),
        fine_m=LABELS[label][0],
        coarse_m=LABELS[label][1],
        mesh_size_min_m=float(equivalent.min()),
        mesh_size_median_m=float(np.median(equivalent)),
        mesh_size_max_m=float(equivalent.max()),
        distinct_sizes=int(len(np.unique(np.round(equivalent, 9)))),
        mesh_compactness_median=float(np.median(compactness)),
        carried=rows,
    )


def provenance_record(output):
    """Record each persisted operator's digest from its own sidecar.

    The bound and control receipts were written before the loader reported the
    sidecar's digest and carry the placeholder ``synthetic`` instead, so the
    digests are recorded here from the ``*-operator.json`` files the persisting
    job wrote beside each ``*.npy``, beside a table of what every receipt on
    disk actually reads.
    """
    operators = []
    for label in LABELS:
        operator = OPERATORS / f"{label}-operator.npy"
        sidecar = OPERATORS / f"{label}-operator.json"
        meta = json.loads(sidecar.read_text())
        stat = operator.stat()
        receipts = {}
        for path in sorted(output.glob("*.json")):
            row = json.loads(path.read_text())
            if row.get("operator_path") == str(operator) or row.get("label") == label:
                receipts[path.name] = row.get("geometry_digest", "absent")
        operators.append(
            dict(
                label=label,
                operator_path=str(operator),
                operator_bytes=int(stat.st_size),
                operator_mtime_utc=datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                sidecar_path=str(sidecar),
                geometry_digest=meta["geometry_digest"],
                basis=meta["basis"],
                receipts_and_their_digest_field=receipts,
            )
        )
    return dict(operators=operators)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("bound", "witness", "control", "check", "region", "provenance"),
        required=True,
    )
    parser.add_argument("--label", choices=tuple(LABELS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--mutate", choices=tuple(CHECK_TOLERANCE))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "check":
        row = check(mutate=args.mutate)
        write(args.output / "check.json", row)
        print(json.dumps(row, indent=1), flush=True)
        missed = check_verdict(row)
        if missed:
            raise SystemExit(
                "calibration identity outside its tolerance: "
                + "; ".join(
                    f"{name} {value:.3e} exceeds {tolerance:.1e}"
                    for name, value, tolerance in missed
                )
            )
        return
    if args.mode == "provenance":
        write(args.output / "operator-digests.json", provenance_record(args.output))
        return
    from scripts.iron_graded_refinement.measure import west_inputs

    inputs = west_inputs()
    if args.mode == "bound":
        record_bound(args.label, inputs, args.output, tolerance=args.tolerance)
    elif args.mode == "witness":
        record_witness(args.label, inputs, args.output)
    elif args.mode == "region":
        write(
            args.output / f"region-{args.label}.json",
            region_record(args.label, inputs, args.output),
        )
    else:
        record_control(args.label, inputs, args.output)


if __name__ == "__main__":
    main()
