"""Methods for calculating the position and value of x-points and o-points."""

from dataclasses import dataclass, field

from functools import cached_property
import jax
import jax.numpy as jnp
import numpy as np
import seaborn as sns
import xarray

from nova.biot.array import Array
from nova.jax.null import Null


@dataclass
class FieldNull(Array):
    """Calculate positions of all field nulls."""

    data: xarray.Dataset = field(repr=False, default_factory=xarray.Dataset)
    array_attrs: list[str] = field(
        default_factory=lambda: ["x", "z", "stencil", "stencil_index"]
    )
    maxsize: int = 6
    data_o: dict[str, np.ndarray] = field(init=False, default_factory=dict, repr=False)
    data_x: dict[str, np.ndarray] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self):
        """Instigate jax Null class."""
        self.null = Null(
            jnp.array(self.stencil), jnp.array(self.coordinate_stencil), self.maxsize
        )

    @cached_property
    def stencil(self):
        """Return grid stencil."""
        if "stencil" in self.data:
            return self.data["stencil"].data
        patch = np.array([(0, 0), (-1, 0), (0, -1), (1, -1), (1, 0), (0, 1), (-1, 1)])
        return np.ravel_multi_index(
            np.indices((self.data.sizes["x"] - 2, self.data.sizes["z"] - 2)).reshape(
                2, -1, 1
            )
            + 1
            + patch.T[:, np.newaxis],
            (self.data.sizes["x"], self.data.sizes["z"]),
        )

    @cached_property
    def coordinate_stencil(self):
        """Return stencil geometry."""
        if "stencil" in self.data:  # unstructured grid
            return np.c_[self.data.x, self.data.z][self.stencil]
        return np.c_[
            self.data.x2d.data.ravel(),
            self.data.z2d.data.ravel(),
        ][self.stencil]

    def update_null(self, psi):
        """Update null points."""
        return self.null.update(psi)
        # nulls = nulls[number]
        # print(nulls)

        # for null, mask in zip("ox", [mask_o, mask_x]):
        #    setattr(self, f"data_{null}", self.update_mask(mask, psi))

    # return {"index": index, "points": points, "psi": psi[index]}

    @staticmethod
    def _unique(nulls, decimals=3):
        """Return unique field nulls."""
        points = np.array([null[0] for null in nulls])
        psi = np.array([null[1] for null in nulls])
        null_type = np.array([null[2] for null in nulls])
        _, index = np.unique(points.round(decimals), axis=0, return_index=True)
        return {
            "points": points[index],
            "psi": psi[index],
            "null_type": null_type[index],
        }

    def delete(self, null: str, index):
        """Delete elements in data specified by index.

        Parameters
        ----------
            index: slice, int or array of ints
                index to remove.

        """
        data = getattr(self, f"data_{null}")
        for attr in data:
            data[attr] = np.delete(data[attr], index, axis=0)

    def plot(self, axes=None):
        """Plot null points."""
        self.get_axes(axes)
        if self.o_point_number > 0:
            self.axes.plot(
                *self.data_o["points"].T, "o", ms=4, mec="C3", mew=1, mfc="none"
            )
        if self.x_point_number > 0:
            self.axes.plot(
                *self.data_x["points"].T, "x", ms=6, mec="C3", mew=1, mfc="none"
            )

    @property
    def o_points(self):
        """Return o-point locations."""
        return self.data_o["points"]

    @property
    def o_psi(self):
        """Return flux values at o-point locations."""
        return self.data_o["psi"]

    @property
    def o_point_number(self):
        """Return o-point number."""
        return len(self.o_psi)

    @property
    def x_points(self):
        """Return x-point locations."""
        return self.data_x["points"]

    @property
    def x_psi(self):
        """Return flux values at x-point locations."""
        return self.data_x["psi"]

    @property
    def x_point_number(self):
        """Return x-point number."""
        return len(self.x_psi)

    # def update_mask(self):
    #    return {"index": index, "points": points, "psi": psi[index]}


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    plasmagrid = xarray.open_dataset("plasmagrid.nc")
    levelset = xarray.open_dataset("levelset.nc")
    data = xarray.open_dataset("data.nc")

    @jax.jit
    def psi_plasma(currents):
        """Return plasmagrid flux map."""
        return jnp.matmul(plasmagrid.Psi.data, currents)

    @jax.jit
    def psi_levelset(currents):
        """Return levelset flux map."""
        return jnp.matmul(levelset.Psi.data, currents)

    itime = 20

    current = data.current[itime]
    passive_current = data.passive_current[itime]
    plasma_current = data.ip[itime]
    currents = np.r_[current, passive_current, plasma_current]

    psi_1d = psi_plasma(currents)
    psi_2d = psi_levelset(currents)

    fieldnull = FieldNull(data=plasmagrid, maxsize=2)

    fieldnull.update_null(psi_1d)

    sns.set_theme("notebook", "ticks")
    plt.figure(figsize=(9, 7))

    plt.triplot(
        plasmagrid.x,
        plasmagrid.z,
        plasmagrid.triangles,
        lw=1.5,
        color="C0",
        alpha=0.2,
    )

    plt.tricontour(
        plasmagrid.x,
        plasmagrid.z,
        plasmagrid.triangles,
        psi_plasma(currents),
        levels=71,
        colors="C0",
        linestyles="solid",
        linewidths=1.5,
    )
    plt.axis("equal")
    plt.axis("off")

    plt.plot(
        fieldnull.null.o_point(psi_1d, 0), fieldnull.null.o_point(psi_1d, 1), "C3o"
    )

    @dataclass
    @jax.tree_util.register_pytree_node_class
    class Operate:
        """Locate and label field nulls on structured and unstructured grids."""

        Psi: jnp.ndarray

        def tree_flatten(self):
            """Return flattened pytree."""
            children = (self.Psi,)
            aux_data = {}
            return (children, aux_data)

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            """Return unflattened pytree."""
            return cls(*children, **aux_data)

        @jax.jit
        def psi(self, currents):
            return jnp.matmul(self.Psi, currents)

    opp = Operate(plasmagrid.Psi.data)

    Psi = plasmagrid.Psi.data
    null = fieldnull.null

    @jax.jit
    def o_point(currents):
        psi = jnp.matmul(Psi, currents)
        return null.update(psi)[0][0, 0]

    d_o_point = jax.jacfwd(o_point)
    dd_o_point = jax.jacfwd(d_o_point)

    factor = np.linspace(0.8, 1.2, 500)

    radius = np.zeros_like(factor)
    gradient = np.zeros_like(factor)
    curve = np.zeros_like(factor)

    coil_index = 2
    Io = currents[coil_index]

    I_dynamic = np.copy(currents)
    for i, fact in enumerate(factor):
        I_dynamic[coil_index] = fact * Io
        radius[i] = np.asarray(o_point(I_dynamic))
        gradient[i] = np.asarray(d_o_point(I_dynamic))[coil_index]
        # curve[i] = np.asarray(dd_o_point(currents, 0))[coil_index]

    axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)[1]
    axes[0].plot(1e-3 * factor * Io, radius)
    axes[1].plot(
        1e-3 * factor * Io, np.gradient(radius, factor * Io), "C1", label="numeric"
    )
    axes[1].plot(1e-3 * factor * Io, gradient, "C0", label="jax")
    axes[1].legend()
    axes[0].set_ylabel(r"radius $m$")
    axes[1].set_xlabel(r"Ics1 $kA$")
    axes[1].set_ylabel(r"$\frac{dr}{dI_{CS1}}$ $mA^{-1}$")
    sns.despine()

    """


    (o_point_number, x_point_number), count = categorize_1d(psi_1d, stencil)

    plt.plot(plasmagrid.x[o_mask], plasmagrid.z[o_mask], "r.")

    o_mask, x_mask = categorize_2d(psi_2d)
    plt.plot(levelset.x2d.data[o_mask], levelset.z2d.data[o_mask], "r.")
    plt.plot(levelset.x2d.data[x_mask], levelset.z2d.data[x_mask], "rx")

    plt.tricontour(
        plasmagrid.x,
        plasmagrid.z,
        plasmagrid.triangles,
        psi_plasma(currents),
        levels=71,
        colors="C0",
        linestyles="solid",
        linewidths=1.5,
    )

    plt.triplot(
        plasmagrid.x,
        plasmagrid.z,
        plasmagrid.triangles,
        lw=1.5,
        color="C0",
        alpha=0.2,
    )


    """
