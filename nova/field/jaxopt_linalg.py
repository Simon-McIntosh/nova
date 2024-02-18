from dataclasses import dataclass, field
from functools import cached_property
import jax
import jax.numpy as jnp

# import jaxopt
import numpy as np

from nova.frame.coilset import CoilSet


@jax.tree_util.register_pytree_node_class
@dataclass
class Clebsch:
    """Calculate Clebsch potential fields A=alpha*nabla(beta)."""

    delta: tuple[float]
    vector_potential: np.ndarray | jnp.ndarray = field(repr=False)
    magnetic_field: np.ndarray | jnp.ndarray = field(repr=False)

    def tree_flatten(self):
        """Return flattened pytree structure."""
        children = (self.vector_potential, self.magnetic_field)
        aux_data = (self.delta, self.axes, self.ndim, self._delta, self.shape)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild instance from pytree variables."""
        return cls(aux_data[0], *children)

    @cached_property
    def axes(self):
        """Return active coordinate axes."""
        return tuple(axis for axis, delta in enumerate(self.delta) if delta is not None)

    @cached_property
    def ndim(self):
        """Return axis number."""
        return len(self.axes)

    @cached_property
    def _delta(self):
        """Return axes deltas."""
        return tuple(self.delta[axis] for axis in self.axes)

    @cached_property
    def shape(self):
        """Return data shape."""
        return self.vector_potential.shape

    @jax.jit
    def dot(self, a, b):
        """Return tenesor dot product taken across first dimension."""
        return jnp.einsum("i...,i...", a, b)

    @jax.jit
    def grad(self, x):
        """Return grad x."""
        gradient = jnp.gradient(x, *self._delta)
        if None in self.delta:
            gradient.insert(self.delta.index(None), jnp.zeros_like(x))
        return jnp.stack(gradient, axis=0)

    @jax.jit
    def vector_grad(self, x):
        """Return stacked gradient of vector field."""
        return jnp.stack([self.grad(x[i]) for i in range(x.shape[0])], axis=0)

    @jax.jit
    def div(self, x):
        """Return div x."""
        return jnp.sum(
            jnp.stack(
                [
                    jnp.gradient(x[axis], self.delta[axis], axis=i)
                    for i, axis in enumerate(self.axes)
                ],
                axis=0,
            ),
            axis=0,
        )

    @jax.jit
    def laplacian(self, x):
        """Return the Laplacian of x."""
        return jnp.sum(
            jnp.stack(
                [
                    jnp.gradient(
                        jnp.gradient(x, self.delta[axis], axis=i),
                        self.delta[axis],
                        axis=i,
                    )
                    for i, axis in enumerate(self.axes)
                ],
                axis=0,
            ),
            axis=0,
        )

    @jax.jit
    def matvec(self, x):
        """Return stacked matrix multiplication Ax."""
        alpha, beta, gamma = x.reshape(self.shape)
        return jnp.ravel(
            jnp.stack(
                [
                    self._get_slice(gamma * self.grad(alpha)),
                    self._get_slice(alpha * self.grad(beta)),
                    self._get_slice(
                        jnp.cross(self.grad(alpha), self.grad(beta), 0, 0, 0)
                    ),
                ]
            )
        )

    @jax.jit
    def _get_slice(self, vector):
        """Return core/edge mapping for the given vector field."""
        result = self.div(vector)
        for i, axis in enumerate(self.axes):
            for face in [0, -1]:
                offset = tuple(face if _axis == axis else 0 for _axis in self.axes)
                update = jax.lax.index_in_dim(vector[axis], face, axis=i, keepdims=True)
                result = jax.lax.dynamic_update_slice(result, update, offset)
        return result

    @jax.jit
    def rhs(self):
        """Return right hand side of linear equation system Ax=y."""
        return jnp.ravel(
            jnp.stack(
                [
                    self._get_slice(
                        self.dot(
                            self.vector_grad(self.magnetic_field),
                            self.magnetic_field / jnp.linalg.norm(self.magnetic_field),
                        )
                    ),
                    self._get_slice(self.vector_potential),
                    self._get_slice(self.magnetic_field),
                ],
                axis=0,
            )
        )


def test_laplacian():
    ids = CoilSet(field_attrs=["Ax", "Ay", "Az", "Bx", "By", "Bz"])
    ids.coil.insert(8, 0, 0.5, 0.5, Ic=1e4, segment="circle")
    ids.grid.solve(1e2, [2, 8, -2, 2])
    clebsch = Clebsch(
        ids.grid.delta,
        ids.grid.vector_potential,
        ids.grid.magnetic_field,
    )
    ay = clebsch.vector_potential[1]
    assert np.allclose(clebsch.laplacian(ay), clebsch.div(clebsch.grad(ay)))


if __name__ == "__main__":

    ids = CoilSet(field_attrs=["Bx", "By", "Bz", "Ax", "Ay", "Az"])
    radius = 9
    theta = np.linspace(0, 2 * np.pi)
    ids.winding.insert(
        np.c_[radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)],
        {"c": [0, 0, 0.1]},
        Ic=1e4,
    )

    # ids.coil.insert(9, 0, 0.5, 0.5, Ic=1e4)  #  , segment="circle")
    # ids.coil.insert(7, 0, 0.5, 0.5, Ic=-1e4, segment="circle")

    ids.grid.solve(5e3, [4, 12, -2, 2])

    """

    i = ids.grid.shape[1] // 2

    clebsch = Clebsch(
        ids.grid.delta,
        ids.grid.vector_potential,
        ids.grid.magnetic_field,
    )

    def scalar(h, magnetic_field, delta):
        grad_h = jnp.stack(jnp.gradient(h, *delta))
        return jnp.einsum("i...,i...", grad_h, magnetic_field)
    """

    from scipy.interpolate import BSpline

    k = 2
    t = [0, 1, 2, 3, 4, 5, 6]

    n = len(t) - (k + 1)
    c = jnp.array([-1, 2, 0, 2])
    spl = BSpline(t, c, k)
    print(spl(2.5))

    def ramp(vector, pad_width, *args, **kwargs):
        """Pad vector with linear extrapolation."""
        delta = np.mean(np.diff(vector[pad_width[0] : -pad_width[1]]))
        vector[: pad_width[0]] = vector[pad_width[0]] + delta * np.arange(
            -pad_width[0], 0
        )
        vector[-pad_width[1] :] = vector[-(1 + pad_width[1])] + delta * np.arange(
            1, pad_width[1] + 1
        )

    np.pad(ids.grid.data.x, k, mode=ramp)

    BSpline.design_matrix(x := np.linspace(2, 3), t, k)

    """
    result = np.einsum("i,j", matrix @ c1, matrix @ c2)

    ids.grid.axes.streamplot(
        ids.grid.data.x.data,
        ids.grid.data.z.data,
        ids.grid.magnetic_field[0, :, i].T,
        ids.grid.magnetic_field[-1, :, i].T,
    )

    ids.grid.axes.pcolor(x, x, result)

    @jax.jit
    def mult(c, matrix):
        return matrix @ c

    # ids.grid.axes.plot(x, mult(c, matrix))
    """

    """

    import jaxopt

    gn = jaxopt.GaussNewton(residual_fun=scalar, verbose=True)

    ho = clebsch.vector_potential[1]

    sol = gn.run(ho, clebsch.magnetic_field, clebsch._delta)

    # import jaxopt
    # sol = jaxopt.linear_solve.solve_gmres(clebsch.matvec, clebsch.rhs()).reshape(
    #    clebsch.shape
    # )

    # ids.grid.axes.contour(
    #    ids.grid.data.X[:, 0].data, ids.grid.data.Z[:, 0].data, sol[0, :, 0]
    # )
    ids.grid.axes.streamplot(
        ids.grid.data.x.data,
        ids.grid.data.z.data,
        ids.grid.magnetic_field[0, :, i].T,
        ids.grid.magnetic_field[-1, :, i].T,
    )

    ids.grid.axes.contour(
        ids.grid.data.X[:, i], ids.grid.data.Z[:, i], sol[0][:, i], levels=51
    )
    """

    """

    ids.grid.set_axes("2d")
    # ids.grid.axes.contour(ids.grid.data.x2d, ids.grid.data.z2d, sol, levels=51)
    # ids.grid.plot(nulls=False)

    normal = clebsch.dot(
        clebsch.vector_grad(clebsch.magnetic_field),
        clebsch.magnetic_field / jnp.linalg.norm(clebsch.magnetic_field, axis=0),
    )

    clebsch.magnetic_field = jax.lax.dynamic_update_slice(
        clebsch.magnetic_field,
        1 / ids.grid.data.X.data[jnp.newaxis],
        (
            1,
            0,
            0,
        ),
    )
    """

    '''
    B = clebsch.magnetic_field  # / np.linalg.norm(clebsch.magnetic_field, axis=0)

    normal = jnp.sum(
        jnp.stack(
            [
                clebsch.magnetic_field[0] * jnp.gradient(B, clebsch.delta[0], axis=1),
                clebsch.magnetic_field[1] * jnp.gradient(B, clebsch.delta[1], axis=2),
                clebsch.magnetic_field[2] * jnp.gradient(B, clebsch.delta[2], axis=3),
            ]
        ),
        axis=0,
    )
    gradB = clebsch.vector_grad(B)  # / np.linalg.norm(clebsch.vector_grad(B), axis=1)
    normal = np.einsum("ij...,j...->i...", gradB, clebsch.magnetic_field)
    normal = normal / np.linalg.norm(normal[::2], axis=0)

    ids.grid.axes.quiver(
        ids.grid.data.X[:, i], ids.grid.data.Z[:, i], normal[0, :, i], normal[-1, :, i]
    )

    ids.grid.set_axes("2d")
    ids.grid.axes.streamplot(
        ids.grid.data.x.data,
        ids.grid.data.z.data,
        normal[0, :, i].T,
        normal[-1, :, i].T,
    )



    
    B = clebsch.magnetic_field / jnp.linalg.norm(clebsch.magnetic_field, axis=0)

    normal = jnp.stack([-B[2], B[1], B[0]])

    # normal = jax.lax.dynamic_update_slice(
    #    normal, normal[0] * ids.grid.data.x2d.data[jnp.newaxis], (0, 0, 0)
    # )

    # B = clebsch.magnetic_field
    # Bdot = clebsch.dot(clebsch.vector_grad(B), B)
    # normal = clebsch.dot(clebsch.vector_grad(Bdot), B)
    # normal = np.cross(Bdot, Bdotdot, 0, 0, 0)
    """
    normal = jnp.sum(
        jnp.stack(
            [
                clebsch.magnetic_field[0]
                * jnp.gradient(clebsch.magnetic_field, clebsch.delta[0], axis=1),
                clebsch.magnetic_field[2]
                * jnp.gradient(clebsch.magnetic_field, clebsch.delta[2], axis=2),
            ]
        ),
        axis=0,
    )
    """
    # normal = np.einsum("ji...,i...->j...", gradB, B)

    # normal = jnp.sum(jnp.stack([gradB[:, i] * B[i] for i in range(3)]), axis=0)
    # normal = clebsch.magnetic_field / jnp.linalg.norm(clebsch.magnetic_field, axis=0)

    ids.grid.axes.quiver(ids.grid.data.x2d, ids.grid.data.z2d, B[0], B[-1])
    # ids.grid.axes.quiver(ids.grid.data.x2d, ids.grid.data.z2d, normal[0], normal[-1])
    # print(clebsch.solve(jnp.stack([ids.grid.psi_, ids.grid.psi_, ids.grid.psi_])))
    # clebsch.laplacian(clebsch.vector_potential[1])


    '''
