import functools
import jax
import jax.numpy as jnp

# import jaxopt
import numpy as np

from nova.frame.coilset import CoilSet


ids = CoilSet(field_attrs=["Bx", "By", "Bz", "Ax", "Ay", "Az", "Psi"])
ids.coil.insert(8, 0, 0.5, 0.5, Ic=1e4, segment="circle")
ids.grid.solve(5e3, [2, 5, -2, 2])


def poisson(x):
    """Return result of matrix multiplication Ax."""
    divgrad = (
        x[2:, 1:-1] + x[:-2, 1:-1] + x[1:-1, 2:] + x[1:-1, :-2] - 4 * x[1:-1, 1:-1]
    )
    grad_z = jnp.stack([x[:1, 1:] - x[:1, :-1], x[-1:, 1:] - x[-1:, :-1]])
    grad_x = jnp.stack([x[1:, :1] - x[:-1, :1], x[1:, -1:] - x[:-1, -1:]])

    rhs = jnp.zeros_like(x)
    rhs = rhs.at[0, 0].set(x[0, 0])
    rhs = jax.lax.dynamic_update_slice(rhs, divgrad, (1, 1))
    rhs = jax.lax.dynamic_update_slice(rhs, grad_z[0], (0, 1))
    rhs = jax.lax.dynamic_update_slice(rhs, grad_z[1], (-1, 1))
    rhs = jax.lax.dynamic_update_slice(rhs, grad_x[0][:-1], (1, 0))
    rhs = jax.lax.dynamic_update_slice(rhs, grad_x[1][:-1], (1, -1))
    return rhs


dx = 3


@functools.partial(jax.jit, static_argnames=["delta"])
def grad(x, delta=(1,)):
    """Return grad x."""
    return jnp.stack(jnp.gradient(x, *delta))


@jax.jit
def dot(a, b):
    """Return tenesor dot product taken across first dimension."""
    return jnp.einsum("i...,i...->...", a, b)


@functools.partial(jax.jit, static_argnames=["delta"])
def laplacian(x, delta=(1,)):
    """Return Laplacian of x."""
    return jnp.sum(
        jnp.stack(
            [
                jnp.gradient(jnp.gradient(x, *delta, axis=i), *delta, axis=i)
                for i in range(x.ndim)
            ],
            axis=0,
        ),
        axis=0,
    )


# @jax.jit
def solve(x):
    """Return stacked matrix multiplication Ax."""
    alpha, beta, gamma = x

    rhs = jnp.zeros_like(x)

    rhs = jax.lax.dynamic_update_slice(rhs, gamma * grad(alpha), (0, 0, 0))

    # grad_alpha = jnp.gradient(alpha, dx)

    # return jnp.stack([poisson(x) for x in x_stack])


def update(rhs, div, grad, index, dx):
    rhs = jax.lax.dynamic_update_slice(rhs, div_axb * dx**2, (index, 1, 1))
    rhs = jax.lax.dynamic_update_slice(rhs, axb[:1, 1:, 2] * dx, (index, 0, 1))
    rhs = jax.lax.dynamic_update_slice(rhs, axb[-1:, 1:, 2] * dx, (index, -1, 1))
    rhs = jax.lax.dynamic_update_slice(rhs, axb[1:, :1, 0] * dx, (index, 1, 0))
    rhs = jax.lax.dynamic_update_slice(rhs, axb[1:, -1:, 0] * dx, (index, 1, -1))
    return rhs


dx = np.diff(ids.grid.data.x[:2].data)[0]

vector_potential = jnp.stack([ids.grid.ax_, ids.grid.ay_, ids.grid.az_], axis=-1)
vector_potential_hat = (
    vector_potential / jnp.linalg.norm(vector_potential, axis=-1)[..., jnp.newaxis]
)
magnetic_field = jnp.stack([ids.grid.bx_, ids.grid.by_, ids.grid.bz_], axis=-1)

axb = jnp.cross(vector_potential_hat, magnetic_field)
div_axb = np.gradient(axb[..., 0], dx, axis=0, edge_order=2) + np.gradient(
    axb[..., 2], dx, axis=1, edge_order=2
)


rhs = jnp.zeros(ids.grid.shape)
rhs = update(
    rhs,
)

"""
rhs = jnp.concatenate(
    [
        axb[:, -1:, 2] * dx,
        jnp.concatenate(
            [
                axb[:1, 1:-1, 0] * dx,
                div_axb[1:-1, 1:-1] * dx**2,
                axb[-1:, 1:-1, 0] * dx,
            ],
            axis=0,
        ),
        axb[:, :1, 2] * dx,
    ],
    axis=1,
)
"""

"""

sol = jaxopt.linear_solve.solve_normal_cg(A, rhs)

print(A(rhs).shape)

ids.grid.set_axes("2d")
ids.grid.axes.contour(
    ids.grid.data.x2d[1:-1, 1:-1],
    ids.grid.data.z2d[1:-1, 1:-1],
    sol[1:-1, 1:-1],
    31,
)
ids.grid.plot()


ids.grid.set_axes("1d")
ids.grid.axes.plot(A(ids.grid.psi_)[0, 1:] / dx)
ids.grid.axes.plot(np.gradient(ids.grid.psi_[0, 1:], dx), "C0--")


ids.grid.axes.plot(A(ids.grid.psi_)[-1, 1:] / dx)
ids.grid.axes.plot(np.gradient(ids.grid.psi_[-1, 1:], dx), "C1--")

ids.grid.axes.plot(A(ids.grid.psi_)[1:-1, 0] / dx)
ids.grid.axes.plot(np.gradient(ids.grid.psi_[1:-1, 0], dx), "C2--")

ids.grid.axes.plot(A(ids.grid.psi_)[1:-1, -1] / dx)
ids.grid.axes.plot(np.gradient(ids.grid.psi_[1:-1, -1], dx), "C3--")

ids.grid.set_axes("2d")
psi_xx = np.gradient(
    np.gradient(ids.grid.psi_, dx, axis=0, edge_order=2), dx, axis=0, edge_order=2
)
psi_zz = np.gradient(
    np.gradient(ids.grid.psi_, dx, axis=1, edge_order=2), dx, axis=1, edge_order=2
)
"""

"""
levels = ids.grid.axes.contour(
    ids.grid.data.x2d[1:-1, 1:-1],
    ids.grid.data.z2d[1:-1, 1:-1],
    psi_xx[1:-1, 1:-1] + psi_zz[1:-1, 1:-1],
    colors="C0",
).levels
ids.grid.axes.contour(
    ids.grid.data.x2d[1:-1, 1:-1],
    ids.grid.data.z2d[1:-1, 1:-1],
    A(ids.grid.psi_)[1:-1, 1:-1] / dx**2,
    levels=levels,
    linestyles="--",
    colors="C1",
)


grad_alpha_2d = np.gradient(sol, 2)
grad_alpha = np.stack(
    [grad_alpha_2d[0], np.zeros(ids.grid.shape), grad_alpha_2d[1]], -1
)
B_sol = np.cross(grad_alpha, vector_potential_hat)


ids.grid.axes.streamplot(
    ids.grid.data.x.data, ids.grid.data.z.data, B_sol[..., 0].T, B_sol[..., 2].T
)

ids.grid.axes.streamplot(
    ids.grid.data.x.data, ids.grid.data.z.data, ids.grid.bx_.T, ids.grid.bz_.T
)
"""
ids.grid.axes.contour(ids.grid.data.x2d, ids.grid.data.z2d, ids.grid.ay_)
ids.grid.plot(colors="C2")
# ids.grid.axes.plot(A(ids.grid.psi_)[-1, :])
# ids.grid.axes.plot(A(ids.grid.psi_)[:, 0])
# ids.grid.axes.plot(A(ids.grid.psi_)[:, -1])


# ids.grid.set_axes("2d")
# ids.grid.plot()

"""
print()
print(j[:, :1])
print(A(j)[1:, -1:])
print(ids.grid.bx_[:, :1])
# print(np.diff(ids.grid.bx_[:, :1], axis=0) * dx)
# ids.grid.axes.contour(ids.grid.data.x2d, ids.grid.data.z2d, sol, 61)
ids.grid.axes.contour(ids.grid.data.x2d, ids.grid.data.z2d, ids.grid.psi_, 61)

ids.grid.set_axes("1d")
ids.grid.axes.plot(ids.grid.psi_[0, 1:])

ids.grid.set_axes("1d")
ids.grid.axes.plot(A(ids.grid.psi_)[0, 1:] / (2 * dx))
ids.grid.axes.plot(np.gradient(ids.grid.psi_[0, 1:], dx))
ids.grid.axes.plot(-ids.grid.bx_[0, 1:], "--")
# ids.grid.axes.plot(j[0, 1:])
"""


# alpha_o = ids.grid.ay_

"""
@jax.jit
def residual_fun(alpha, x, z, bx, bz):
    dalpha = jnp.gradient(alpha, 1, 1)
    return jnp.linalg.norm(
        jnp.stack([dalpha[1] - bx, -dalpha[0] - bz], axis=-1), axis=-1
    )
"""

"""
alpha_x = scipy.integrate.cumulative_trapezoid(
    ids.grid.bz_, ids.grid.data.x.data, initial=0, axis=0
)
alpha_z = scipy.integrate.cumulative_trapezoid(
    -ids.grid.bx_, ids.grid.data.z.data, initial=0, axis=1
)

n2 = ids.grid.shape[1] // 2
alpha = alpha_x[0, n2] + alpha_z[:, n2 : n2 + 1] + alpha_x - alpha_x[:, n2 : n2 + 1]
"""

# root = jaxopt.ScipyRootFinding("hybr", optimality_fun=res)
# root.run(
#    ids.grid.ay_, ids.grid.data.x.data, ids.grid.data.z.data, ids.grid.bx_,
# ids.grid.bz_
# )

# gauss_newton = jaxopt.GaussNewton(residual_fun)
# sol = gauss_newton.run(
#    ids.grid.ay_, ids.grid.data.x.data, ids.grid.data.z.data, ids.grid.bx_,
# ids.grid.bz_
# )

# ids.plot()
# ids.grid.plot()
