"""Vary the existing Hessian stencil without changing the topology algorithm."""

from types import FunctionType

from nova.equilibrium import topology


def make_read(spacing_fraction=0.01):
    """Return the kernel-backed read with a stated fraction-of-pitch stencil.

    TotalField already uses central differences of its point Hessian. Reuse
    that implementation: scaling the pitch changes only the stencil spacing.
    Other field types are refused so their exact higher-derivative branch
    cannot silently become the measured implementation.
    """
    if spacing_fraction <= 0:
        raise ValueError("the stencil spacing must be positive")

    def curvature(field, point, pitch):
        if not isinstance(field, topology.TotalField):
            raise TypeError("the stencil prototype requires a TotalField")
        scaled_pitch = (
            pitch if spacing_fraction == 0.01 else pitch * (spacing_fraction / 0.01)
        )
        return topology._curvature_derivatives(field, point, scaled_pitch)

    namespace = dict(topology.read.__globals__, _curvature_derivatives=curvature)
    return FunctionType(
        topology.read.__code__,
        namespace,
        "hessian_stencil_read",
        topology.read.__defaults__,
        topology.read.__closure__,
    )


def batched_field(field):
    """Share one point-kernel body across the plasma and reference image.

    The analytic exterior has total minus reference-image structure. Mapping
    its reference and plasma operands together shares their point program
    while retaining their arithmetic order when composing the total field.
    The read's existing map evaluates the centre and eight stencil offsets.
    """
    import jax
    import jax.numpy as jnp
    from dataclasses import dataclass

    if not hasattr(field.exterior, "reference_moments"):
        raise TypeError("the batch requires an analytic reference-image exterior")

    @jax.tree_util.register_dataclass
    @dataclass(frozen=True)
    class BatchedField(topology.TotalField):
        def sources(self):
            coupling = jax.tree.map(
                lambda a, b: jnp.stack((a, b)), self.coupling, self.exterior.coupling
            )
            moments = jax.tree.map(
                lambda a, b: jnp.stack((a, b)),
                self.moments,
                self.exterior.reference_moments,
            )
            return coupling, moments

        def value(self, point):
            values = jax.lax.map(
                lambda pair: pair[0].value_gradient(point, pair[1])[0], self.sources()
            )
            return values[0] + (self.exterior.total.value(point) - values[1])

        def evaluate(self, point):
            jets = jax.lax.map(
                lambda pair: pair[0].evaluate(point, pair[1]), self.sources()
            )
            return jax.tree.map(
                lambda pair, total: pair[0] + (total - pair[1]),
                jets,
                self.exterior.total.evaluate(point),
            )

    return BatchedField(field.moments, field.coupling, field.exterior)
