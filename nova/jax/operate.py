"""Manage jax backed operator classes."""

from dataclasses import dataclass, field, InitVar
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import xarray

from nova.frame.framesetloc import ArrayLocIndexer
from nova.jax.tree_util import Pytree


MISSING_FORCE_INDEX = (
    "A Force operator scales its interaction by the source current at index, "
    "so the index is required: none was passed and the dataset carries no "
    "index variable. Indexing with a missing index inserts an axis instead of "
    "selecting a gain, which returns a silently wrong result."
)


class MatrixData(NamedTuple):
    """EM coupling data for jax backed computations."""

    plasma_target: jnp.ndarray | None = None
    source_plasma: jnp.ndarray | None = None
    plasma_plasma: jnp.ndarray | None = None
    force_index: jnp.ndarray | None = None


@dataclass
@jax.tree_util.register_pytree_node_class
class Operator(Pytree):
    """Manage EM influence matrices."""

    source_target: jnp.ndarray
    matrix_data: MatrixData
    source_plasma_index: int = -1
    target_plasma_index: int = -1
    classname: str = ""

    @property
    def target(self):
        """Return target attributes."""
        return (
            self.source_target,
            self.matrix_data.plasma_target,
            self.source_plasma_index,
        )

    @jax.jit
    def evaluate(self, source_target, source_current):
        """Return source-target interaction."""
        result = source_target @ source_current
        if self.classname == "Force":
            if (force_index := self.matrix_data.force_index) is None:
                raise ValueError(MISSING_FORCE_INDEX)
            return source_current[force_index] * result
        return result

    @jax.jit
    def evaluate_external(self, source_current):
        """Return source-target interaction excluding plasma."""
        source_current = source_current.at[self.source_plasma_index].set(0.0)
        return self.evaluate(self.source_target, source_current)

    @jax.jit
    def update_plasma_turns(self, plasma_nturn):
        """Update plasma turns inplace."""
        source_target = self.source_target
        if update_source := self.source_plasma_index != -1:
            source_target = source_target.at[:, self.source_plasma_index].set(
                self.matrix_data.plasma_target @ plasma_nturn
            )
        if update_target := self.target_plasma_index != -1:
            source_target = source_target.at[self.target_plasma_index, :].set(
                plasma_nturn @ self.matrix_data.source_plasma
            )
        if update_source and update_target:
            source_target = source_target.at[
                self.target_plasma_index, self.source_plasma_index
            ].set(plasma_nturn @ self.matrix_data.plasma_plasma @ plasma_nturn)
        return source_target

    def tree_flatten(self):
        """Return flattened pytree."""
        children = (
            self.source_target,
            self.matrix_data,
        )
        aux_data = {
            "source_plasma_index": self.source_plasma_index,
            "target_plasma_index": self.target_plasma_index,
            "classname": self.classname,
        }
        return (children, aux_data)


@dataclass
class Operators:
    """Generate EM coupling matricies."""

    data: xarray.Dataset = field(repr=False)
    index: np.ndarray | None = None

    def force_index(self, classname: str) -> jnp.ndarray | None:
        """Return the source-current index a Force operator applies as a gain.

        Only the Force interaction reads an index, so every other classname
        returns None. The caller's index takes precedence; when none is passed
        the index is read from the dataset. An absent index raises rather than
        leaving a Force operator to index with None.
        """
        if classname != "Force":
            return None
        index = self.index
        if index is None:
            index = self.data.get("index", xarray.DataArray([])).data
        if np.asarray(index).size == 0:
            raise ValueError(MISSING_FORCE_INDEX)
        return jnp.asarray(index)

    def __getitem__(self, attr: str) -> Operator:
        """Retrun jax Operator instance."""
        # attrs, not attribute access: a data_var or coord of the same name
        # would shadow the dataset attribute it resolves to.
        source_plasma_index = self.data.attrs["source_plasma_index"]
        target_plasma_index = self.data.attrs["target_plasma_index"]
        classname = self.data.attrs["classname"]

        plasma_dataset = {}
        if source_plasma := source_plasma_index != -1:
            plasma_dataset["plasma_target"] = jnp.array(self.data[f"{attr}_"])
        if target_plasma := target_plasma_index != -1:
            plasma_dataset["source_plasma"] = jnp.array(self.data[f"_{attr}"])
        if source_plasma and target_plasma:
            plasma_dataset["plasma_plasma"] = jnp.array(self.data[f"_{attr}_"])
        if (force_index := self.force_index(classname)) is not None:
            plasma_dataset["force_index"] = force_index

        return Operator(
            jnp.array(self.data[attr]),
            MatrixData(**plasma_dataset),
            source_plasma_index,
            target_plasma_index,
            classname,
        )


@dataclass
class BiotOperator:
    """Jitted operator presenting the numpy operator's mutating interface.

    Wraps the jitted pytree :class:`Operator` so it drops into the biot
    version-counter cache in place of the eager numpy operator: the
    source-target matmul and plasma-turn update run through jitted code while
    the ``source_target`` matrix and the ``evaluate``/``update_turns`` surface
    stay in the mutating shape Operate's invalidation contract drives. The
    plasma row/column are always re-derived from the pristine coupling
    matrices, so repeated turn updates carry no accumulated state.
    """

    aloc: ArrayLocIndexer
    saloc: ArrayLocIndexer
    classname: str
    index: np.ndarray
    dataset: InitVar[xarray.Dataset]

    def __post_init__(self, dataset):
        """Build the jitted operator and link the mutable source-target."""
        attr = list(dataset.data_vars)[0]
        # the caller's index is the Force gain, so the jitted and the numpy
        # operator scale by the same array rather than by separate lookups.
        self._operator = Operators(dataset, self.index)[attr]
        self.source_plasma_index = self._operator.source_plasma_index
        self.target_plasma_index = self._operator.target_plasma_index
        # source_target stays a live view into the dataset array so a plasma-turn
        # update propagates back to data[attr], as the numpy operator did.
        self.source_target = dataset[attr].data

    def evaluate(self):
        """Return the source-target interaction for the current currents."""
        source_current = jnp.asarray(self.saloc["Ic"])
        result = self._operator.evaluate(
            jnp.asarray(self.source_target), source_current
        )
        return np.asarray(result)

    @property
    def plasma_nturn(self):
        """Return plasma turns."""
        return self.aloc["nturn"][self.aloc["plasma"]]

    def update_turns(self, svd=True):
        """Re-derive the plasma row/column of the source-target matrix in place."""
        plasma_nturn = jnp.asarray(self.plasma_nturn)
        updated = np.asarray(self._operator.update_plasma_turns(plasma_nturn))
        self.source_target[...] = updated
