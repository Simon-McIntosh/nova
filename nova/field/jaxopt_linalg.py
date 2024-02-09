import jax
import jax.numpy as jnp
import numpy as np

from nova.frame.coilset import CoilSet


ids = CoilSet(field_attrs=["Bx", "By", "Bz", "Ax", "Ay", "Az", "Psi"])
ids.coil.insert(8, 0, 0.5, 0.5, Ic=1e4, segment="circle")
ids.grid.solve(5e3, [2, 4, -2, 2])


# @jax.jit
def A(alpha):
    j = jnp.zeros_like(alpha)

    """
    j = jax.lax.dynamic_update_slice(
        j,
        alpha[2:, 1:-1]
        + alpha[:-2, 1:-1]
        + alpha[1:-1, 2:]
        + alpha[1:-1, :-2]
        - 4 * alpha[1:-1, 1:-1],
        (1, 1),
    )
    """
    # j = j.at[0, 0].set(alpha[0, 0])
    j = j.at[0, -1].set(alpha[0, -1] - alpha[0, -3])
    # j = j.at[-1, -1].set(alpha[-1, -1] - alpha[-1, -3])
    # j = j.at[-1, 0].set(alpha[-1, 1] - alpha[-1, 0])

    # j = jax.lax.dynamic_update_slice(j, alpha[:-2, :1] - alpha[2:, :1], (1, 0))
    # j = jax.lax.dynamic_update_slice(j, alpha[:-2, -1:] - alpha[2:, -1:], (1, -1))
    j = jax.lax.dynamic_update_slice(j, alpha[:1, 2:] - alpha[:1, :-2], (0, 1))
    # j = jax.lax.dynamic_update_slice(j, alpha[-1:, 2:] - alpha[-1:, :-2], (-1, 1))

    """
    j[1:-1, 1:-1] = (
        alpha[2:, 1:-1]
        + alpha[:-2, 1:-1]
        + alpha[1:-1, 2:]
        + alpha[1:-1, :-2]
        - 4 * alpha[1:-1, 1:-1]
    )

    j[:, 0] = alpha[:, 1] - alpha[:, 0]
    j[:, -1] = alpha[:, -1] - alpha[:, -2]
    j[0, :] = alpha[0, :] - alpha[1, :]
    j[-1, :] = alpha[-2, :] - alpha[-1, :]
    """
    return j


dx = np.diff(ids.grid.data.x[:2].data)[0]


bx_z = jnp.gradient(ids.grid.bx_, dx, axis=1)
bz_x = jnp.gradient(ids.grid.bz_, dx, axis=0)
j = jnp.zeros(ids.grid.shape)
j = jax.lax.dynamic_update_slice(j, (bz_x[1:-1] - bx_z[1:-1]) * dx**2, (1, 1))

j = j.at[0, 0].set(0)
j = j.at[0, -1].set(-ids.grid.bz_[0, -1] * dx)
j = j.at[-1, -1].set(-ids.grid.bz_[-1, -1] * dx)
j = j.at[-1, 0].set(-ids.grid.bz_[-1, 0] * dx)


j = jax.lax.dynamic_update_slice(j, ids.grid.bx_[1:-1, :1] * 2 * dx, (1, 0))
j = jax.lax.dynamic_update_slice(j, ids.grid.bx_[1:-1, -1:] * 2 * dx, (1, -1))
j = jax.lax.dynamic_update_slice(j, -ids.grid.bz_[:1, 1:-1] * 2 * dx, (0, 1))
j = jax.lax.dynamic_update_slice(j, -ids.grid.bz_[-1:, 1:-1] * 2 * dx, (-1, 1))

# sol = jaxopt.linear_solve.solve_gmres(A, j)

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
