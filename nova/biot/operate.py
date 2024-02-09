"""Manage matmul operations and svd reductions on Biot Data."""

from contextlib import contextmanager
from dataclasses import dataclass, field, InitVar
from functools import cached_property

import numpy as np
import xarray

from nova.biot.data import Data
from nova.frame.framesetloc import ArrayLocIndexer


'''
@numba.njit(fastmath=True, parallel=True)
def matmul(A, B):
    """Perform fast matmul operation."""
    row_number = len(A)
    vector = np.empty(row_number, dtype=numba.float64)
    for i in numba.prange(row_number):  # pylint: disable=not-an-iterable
        vector[i] = np.dot(A[i], B)
    return vector
'''


@dataclass
class BiotOp:
    """Fast array opperations for Biot Data arrays."""

    aloc: ArrayLocIndexer
    saloc: ArrayLocIndexer
    classname: str
    index: np.ndarray
    dataset: InitVar[xarray.Dataset]

    def __post_init__(self, dataset):
        """Extract matrix, plasma_matrices, and plasma indicies from dataset."""
        attr = list(dataset.data_vars)[0]
        self.matrix = dataset[attr].data
        self.source_plasma_index = dataset.attrs["source_plasma_index"]
        self.target_plasma_index = dataset.attrs["target_plasma_index"]
        if source_plasma := self.source_plasma_index != -1:
            self.matrix_ = dataset[f"{attr}_"].data
        if target_plasma := self.target_plasma_index != -1:
            self._matrix = dataset[f"_{attr}"].data
        if source_plasma and target_plasma:
            self._matrix_ = dataset[f"_{attr}_"].data

        """
        #  perform svd order reduction
        self.svd_rank = min([len(plasma_s), svd_rank])

        # TODO fix svd_rank == -1 bug - crop plasma_U
        self.plasma_U = plasma_U.copy()#[:, :self.svd_rank].copy()
        self.plasma_s = plasma_s.copy()#[:self.svd_rank].copy()
        self.plasma_V = plasma_V.copy()#[:self.svd_rank, :].copy()
        """

    def evaluate(self):
        """Return interaction."""
        result = self.matrix @ self.saloc["Ic"]
        if self.classname == "Force":
            return self.saloc["Ic"][self.index] * result
        return result

    @property
    def plasma_nturn(self):
        """Return plasma turns."""
        return self.aloc["nturn"][self.aloc["plasma"]]

    def update_turns(self, svd=True):
        """Update plasma turns."""
        """
        if svd:
            self.matrix[:, self.plasma_index] = self.plasma_U @ \
                (self.plasma_s * (self.plasma_V @ self.plasma_nturn))
            return
        print('svd == -1')
        """
        plasma_nturn = self.plasma_nturn
        if update_source := self.source_plasma_index != -1:
            self.matrix[:, self.source_plasma_index] = self.matrix_ @ plasma_nturn
        if update_target := self.target_plasma_index != -1:
            self.matrix[self.target_plasma_index, :] = plasma_nturn @ self._matrix
        if update_source and update_target:
            self.matrix[self.target_plasma_index, self.source_plasma_index] = (
                plasma_nturn @ self._matrix_ @ plasma_nturn
            )


@dataclass
class Operate(Data):
    """Multi-attribute interface to Biot Evaluate methods."""

    version: dict[str, int | None] = field(init=False, repr=False, default_factory=dict)
    svd_rank: int = field(init=False, default=0)
    index: np.ndarray = field(init=False, repr=False)
    operator: dict[str, BiotOp] = field(init=False, default_factory=dict, repr=False)
    array: dict = field(init=False, repr=False, default_factory=dict)

    @property
    def rank(self):
        """Manage svd rank. Set to 0 to disable svd plasma turn update."""
        return self.svd_rank

    @rank.setter
    def rank(self, rank: int):
        if rank != self.svd_rank:
            self.svd_rank = rank
            self.load_operators()

    @property
    def shape(self):
        """Return target shape."""
        return (self.data.sizes["target"],)

    @contextmanager
    def solve_biot(self, number: int | float | None):
        """Manage biot solution - update number and execute post_solve."""
        if number is not None:
            self.number = number
        yield self.number
        if self.number is not None:
            self.post_solve()

    def post_solve(self):
        """Solve biot interaction - extened by subclass."""
        super().post_solve()
        self.load_operators()
        for attr in self.attrs:
            self.update_turns(attr)

    def load(self):
        """Extend netCDF load."""
        super().load()
        self.load_operators()

    def load_operators(self):
        """Link fast biot operators."""
        self.operator = {}
        if "attributes" not in self.data.attrs:
            return
        self.attrs = self.data.attrs["attributes"]
        if isinstance(self.attrs, str):
            self.attrs = [self.attrs]
        self.index = self.data.get("index", xarray.DataArray([])).data
        self.classname = self.data.classname
        self.number = self.data.sizes["target"]
        for attr in np.array(self.attrs):
            attrs = [
                _attr
                for _attr in [attr, f"_{attr}", f"{attr}_", f"_{attr}_"]
                if _attr in self.data
            ]
            dataset = self.data[attrs]
            self.operator[attr] = BiotOp(
                self.aloc, self.saloc, self.classname, self.index, dataset
            )
        self.load_version()
        self.load_arrays()

    def load_version(self):
        """Initialize biot version identifiers."""
        self.version |= {attr: self.data.attrs.get(attr, None) for attr in self.attrs}
        self.version |= {attr.lower(): None for attr in self.attrs}
        if "Br" in self.attrs and "Bz" in self.attrs:
            self.version["bp"] = None

    def load_arrays(self):
        """Link data arrays."""
        for attr in self.version:
            if attr.capitalize() in self.attrs or attr == "bp":
                if attr.islower():
                    if attr == "bp" and self.classname == "Field":
                        self.array[attr] = np.zeros(self.data.sizes["index"])
                    else:
                        self.array[attr] = np.zeros(self.data.sizes["target"])
                    if len(self.shape) == 1:
                        continue
                    ndarray = self.array[attr].reshape(self.shape)
                    self.array[f"{attr}_"] = ndarray
                    continue
                self.array[attr] = self.operator[attr].matrix

    @cached_property
    def source_plasma_index(self):
        """Return source plasma index."""
        return self.data.attrs["source_plasma_index"]

    @cached_property
    def target_plasma_index(self):
        """Return target plasma index."""
        return self.data.attrs["target_plasma_index"]

    def update_turns(self, Attr: str, svd=True):
        """Update plasma turns."""
        if self.source_plasma_index == -1 and self.target_plasma_index == -1:
            return
        self.operator[Attr].update_turns(svd)
        self.version[Attr] = self.data.attrs[Attr] = self.subframe.version["nturn"]

    def calculate_norm(self):
        """Return calculated L2 norm."""
        result = np.linalg.norm([self.br, self.bz], axis=0)
        if self.classname == "Field":
            return np.maximum.reduceat(result, self.index)
        return result

    def get_norm(self):
        """Return cached field L2 norm."""
        if (version := self.aloc_hash["Ic"]) != self.version["bp"]:
            self.version["bp"] = version
            self.array["bp"][:] = self.calculate_norm()
        return self.array["bp"]

    def __getattr__(self, attr):
        """Return variable data - lazy evaluation - cached."""
        attr = attr.replace("_field_", "")
        if attr.islower() and attr[-1] == "_":  # return shaped array
            if len(self.shape) == 1:
                return getattr(self, attr[:-1])
            self.array[attr][:] = getattr(self, attr[:-1]).reshape(self.shape)
            return self.array[attr]
        if attr not in (
            avalible := [
                attr
                for attr in self.version
                if attr.capitalize() in self.attrs or attr == "bp"
            ]
        ):
            raise AttributeError(f"Attribute {attr} " f"not defined in {avalible}.")
        if len(self.data) == 0:
            return self.array[attr]
        if attr == "bp":
            return self.get_norm()
        Attr = attr.capitalize()
        self.check_plasma(Attr)
        if attr == Attr:
            return self.array[Attr]
        self.check_source(attr)
        return self.array[attr]

    @property
    def Avector(self):
        """Return magnetic vector potential."""
        return np.stack([self.ax, self.ay, self.az], axis=-1)

    @property
    def Bvector(self):
        """Return magnetic field vector."""
        return np.stack([self.bx, self.by, self.bz], axis=-1)

    @cached_property
    def _source_version(self) -> list[str]:
        return [
            attr
            for attr in self.version
            if attr.islower() and attr not in ["frameloc", "subframeloc"]
        ]

    def check_plasma(self, Attr: str):
        """Check plasma turn status, update coupling matrix if required."""
        if self.version[Attr] != self.subframe.version["nturn"]:
            self.update_turns(Attr)
            for attr in self._source_version:
                self.version[attr] = None

    def check_source(self, attr: str):
        """Check source current, re-evaluate if requried."""
        if self.version[attr] != (version := self.aloc_hash["Ic"]):
            self.version[attr] = version
            self.array[attr][:] = self.operator[attr.capitalize()].evaluate()

    def check(self, attr: str):
        """Check plasma and source attributes."""
        self.check_plasma(attr.capitalize())
        self.check_source(attr)

    def __getitem__(self, attr: str):
        """Return array attribute via dict-like access."""
        return getattr(self, attr)

    def __setitem__(self, attr: str, value: np.ndarray):
        """Update array attribute in-place."""
        self.array[attr][:] = value.copy()
        self.version[attr] = None
