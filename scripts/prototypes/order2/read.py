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
