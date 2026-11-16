"""Apply the retained diagnosis references and trace checks to a repaired kernel."""

from nova.jax.config import configure_dtypes

configure_dtypes()


import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from nova.equilibrium.topology import BiotMomentCoupling  # noqa: E402

assert jax.config.jax_enable_x64 is True
jax.config.update("jax_enable_compilation_cache", False)


class PointCheckpoint(Exception):
    """A bounded process has persisted its point budget for the next process."""


def evaluate(
    coupling,
    points,
    moments,
    output,
    label,
    *,
    resume=False,
    point_budget=0,
    remaining=None,
):
    def gradient(point, coefficients):
        return BiotMomentCoupling.value_gradient.__wrapped__(
            coupling, point, tuple(coefficients.T)
        )[1]

    def both(point, coefficients):
        def with_aux(target):
            value = gradient(target, coefficients)
            return value, value

        hessian, value = jax.jacfwd(with_aux, has_aux=True)(point)
        return value, hessian

    batched = jax.vmap(both, in_axes=(None, 0))
    gradients, hessians = [], []
    partial = output / f"{label}-partial.npz"
    if resume and partial.exists():
        with np.load(partial) as prior:
            g, h = prior["gradient"], prior["hessian"]
            assert g.shape[0] == h.shape[0] == len(moments)
            assert g.shape[1] == h.shape[1] <= len(points)
            assert g.shape[2:] == (2,) and h.shape[2:] == (2, 2)
            assert np.isfinite(g).all() and np.isfinite(h).all()
            if "points" in prior:
                np.testing.assert_array_equal(prior["points"], points[: g.shape[1]])
            gradients = list(g.swapaxes(0, 1))
            hessians = list(h.swapaxes(0, 1))
        print("RESUME_DIRECT", label, len(gradients), flush=True)
    start = len(gradients)
    allowance = point_budget if remaining is None else remaining[0]
    stop = min(len(points), start + allowance) if point_budget else len(points)
    for index in range(start, stop):
        point = points[index]
        g, h = batched(jnp.asarray(point), jnp.asarray(moments))
        gradients.append(np.asarray(g))
        hessians.append(np.asarray(h))
        temporary = partial.with_suffix(".pending.npz")
        np.savez(
            temporary,
            gradient=np.asarray(gradients).swapaxes(0, 1),
            hessian=np.asarray(hessians).swapaxes(0, 1),
            points=points[: index + 1],
        )
        temporary.replace(partial)
        print("DIRECT_NODE", label, index, flush=True)
    if remaining is not None:
        remaining[0] -= stop - start
    if stop < len(points):
        raise PointCheckpoint(f"{label}: {stop}/{len(points)} points persisted")
    return np.asarray(gradients).swapaxes(0, 1), np.asarray(hessians).swapaxes(0, 1)
