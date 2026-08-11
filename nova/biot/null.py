"""Fixed-shape field-null categorization kernels."""

from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import numpy as np

from nova.geometry import select
from nova.jax.config import Precision, resolve_precision
from nova.jax.tree_util import Pytree


@dataclass
@jax.tree_util.register_pytree_node_class
class NullBase(Pytree):
    """Null pytree base class."""

    coordinate: jnp.ndarray = field(repr=False)

    def __post_init__(self):
        """Calculate node number."""
        self.node_number = self.coordinate.shape[0]

    def tree_flatten(self):
        """Return flattened pytree."""
        children = (self.coordinate,)
        return (children, {})


@dataclass
@jax.tree_util.register_pytree_node_class
class Null1D(NullBase):
    """Locate and label field nulls on 1D loop."""

    @jax.jit
    def __call__(self, psi, polarity):
        """Return subgrid interpolated field null."""
        return select.traced_wall_flux(
            self.coordinate[:, 0], self.coordinate[:, 1], psi, polarity
        )


@dataclass
@jax.tree_util.register_pytree_node_class
class Null2D(NullBase):
    """Locate nulls from normalized stencils and precise physical metadata.

    ``precision`` selects the dtype of the local sub-cell fit, and that dtype
    is the ladder every flux read taken through this locator lands on: the
    fitted null flux cannot express a difference finer than one step of it,
    and any quantity normalized against that flux inherits the same step.
    Float64 is the default so the ladder sits at the arithmetic floor of the
    surrounding fp64 map.  Float32 halves the fit's device footprint and its
    memory traffic, and is selected explicitly where that throughput is worth
    a ladder some seven decades coarser.
    """

    stencil: jnp.ndarray = field(repr=False)
    local_coordinate_stencil: jnp.ndarray = field(repr=False)
    physical_origin: jnp.ndarray = field(repr=False)
    physical_scale: jnp.ndarray = field(repr=False)
    maxsize: int = 5
    precision: Precision = Precision.DOUBLE

    @classmethod
    def from_coordinates(
        cls,
        coordinate,
        stencil,
        maxsize=5,
        precision: Precision | str = Precision.AUTOMATIC,
    ):
        """Construct normalized fit data from host float64 geometry.

        Normalization occurs before any selected-precision device cast.  Passing
        an absolute fp32 coordinate grid is rejected because its lost low bits
        cannot be recovered by centering later.  ``precision="auto"`` resolves
        to float64: the fit dtype sets the ladder of every flux read the
        locator returns, so the automatic policy is the one that keeps that
        ladder at the arithmetic floor.
        """
        physical = np.asarray(coordinate)
        if physical.dtype != np.float64:
            raise TypeError("Null2D physical coordinates must be host float64")
        if physical.ndim != 2 or physical.shape[1] != 2:
            raise ValueError("Null2D coordinates must have shape (nodes, 2)")
        stencil_index = np.asarray(stencil)
        if stencil_index.ndim != 2:
            raise ValueError("Null2D stencil must have shape (centres, vertices)")
        clusters = physical[stencil_index]
        origin = clusters[:, 0]
        offsets = clusters - origin[:, None, :]
        scale = np.max(np.abs(offsets), axis=1)
        if np.any(scale <= 0):
            raise ValueError("Null2D stencils must span both physical axes")
        local = offsets / scale[:, None, :]
        precision = resolve_precision(precision, Precision.DOUBLE)
        fit_dtype = jnp.float32 if precision is Precision.SINGLE else jnp.float64
        return cls(
            coordinate=jnp.asarray(physical, dtype=jnp.float64),
            stencil=jnp.asarray(stencil_index),
            local_coordinate_stencil=jnp.asarray(local, dtype=fit_dtype),
            physical_origin=jnp.asarray(origin, dtype=jnp.float64),
            physical_scale=jnp.asarray(scale, dtype=jnp.float64),
            maxsize=maxsize,
            precision=precision,
        )

    @property
    def fit_dtype(self):
        """Return the local-fit dtype, which is the ladder of every flux read."""
        return jnp.float32 if self.precision is Precision.SINGLE else jnp.float64

    @jax.jit
    def __call__(self, psi):
        """Return subgrid interpolated field nulls."""
        psi_stencil = jnp.asarray(psi, dtype=self.fit_dtype)[self.stencil]
        number, cluster, origin, scale = self.categorize(psi_stencil)
        return jax.vmap(self.interpolate, (0, 0, 0, 0))(number, cluster, origin, scale)

    @staticmethod
    @jax.jit
    def crossing_count(psi_stencil):
        """Count the sign changes around the vertex ring of every stencil.

            - 0: minima / maxima point
            - 2: regular point
            - 4: saddle point

        The count belongs to each ring alone: its vertices are compared against
        its own centre, and the traversal is closed — the first vertex follows
        the last — so the count is the number of sign changes around the cycle.
        Written that way the whole grid is one elementwise comparison against
        the same comparison rolled by one vertex.

        From On detecting all saddle points in 2D images, A. Kuijper
        """
        sign = psi_stencil[:, 1:] > psi_stencil[:, :1]
        return jnp.sum(sign != jnp.roll(sign, 1, axis=1), axis=1)

    @jax.jit
    def categorize(self, psi_stencil):
        """Categorize points in 1d hexagonal grid.

        The crossing count separates the two null types the fit is taken on,
        and both selections are drawn from that one count: the extrema first,
        the saddles second, each padded to ``maxsize`` so the returned clusters
        carry a fixed shape whatever the flux map holds. Every cluster comes
        back as its ring's normalized coordinates beside the flux sampled on
        them, with the physical origin and scale that map a local fit back.
        """
        psi_stencil = jnp.asarray(psi_stencil, dtype=self.fit_dtype)
        count = self.crossing_count(psi_stencil)
        number = jnp.array([jnp.sum(count == 0), jnp.sum(count == 4)])
        index = jnp.stack(
            [
                jnp.where(count == null_type, size=self.maxsize)[0]
                for null_type in (0, 4)
            ]
        )
        cluster = jnp.concatenate(
            (
                self.local_coordinate_stencil[index],
                psi_stencil[index][..., jnp.newaxis],
            ),
            axis=-1,
        )
        return number, cluster, self.physical_origin[index], self.physical_scale[index]

    @jax.jit
    def interpolate(self, number, cluster, origin, scale):
        """Interpolate subnull from cluster data.

        The selection is padded to a fixed capacity, and a cluster's position
        in it is the whole of what distinguishes a located null from the
        padding. That position is known without traversing the clusters, so
        each quadratic is fitted independently of the others and the positions
        past the counted number are replaced by not-a-number.
        """

        def subnull(one_cluster, physical_origin, physical_scale):
            """Return one sub-cell null in physical coordinates."""
            local = select.traced_subnull(
                one_cluster[:, 0], one_cluster[:, 1], one_cluster[:, 2]
            )
            physical = physical_origin + local[:2].astype(jnp.float64) * physical_scale
            return jnp.concatenate((physical, local[2:].astype(jnp.float64)))

        result = jax.vmap(subnull)(cluster, origin, scale)
        position = jnp.arange(1, cluster.shape[0] + 1)
        return jnp.where(
            (position <= number)[:, jnp.newaxis],
            result,
            jnp.full(4, jnp.nan, dtype=jnp.float64),
        )

    def tree_flatten(self):
        """Return flattened pytree."""
        children, aux_data = super().tree_flatten()
        children += (
            self.stencil,
            self.local_coordinate_stencil,
            self.physical_origin,
            self.physical_scale,
        )
        aux_data |= {"maxsize": self.maxsize, "precision": self.precision}
        return (children, aux_data)
