"""Manage multi-point constraints across the frame index.

Resolves the ``link`` column (bool / numeric / label) into canonical link
labels plus the ``ref`` / ``subref`` reduction indices that map every row to
its independent degree of freedom. Operates on the columnar frame store; no
DataFrame library dependency.
"""

from dataclasses import dataclass, field

import numpy as np

import nova.frame.metamethod as metamethod
from nova.frame.columnar import Index, is_list_like


def _isna(value) -> bool:
    """Return True for a missing link value (None or float nan)."""
    if value is None:
        return True
    if isinstance(value, float):
        return np.isnan(value)
    return False


@dataclass
class MultiPoint(metamethod.MultiPoint):
    """Manage multi-point constraints applied across frame.index."""

    name = "multipoint"

    frame: object = field(repr=False)
    additional: list[str] = field(default_factory=lambda: ["factor", "ref", "subref"])
    indexer: list[int] = field(init=False, repr=False)
    index: Index = field(default_factory=lambda: Index([]))

    def initialize(self):
        """Normalise the link column and build the reduction indices."""
        link = np.asarray(self.frame.link, dtype=object)
        self.frame.link = link
        isna = np.array([_isna(value) for value in link], dtype=bool)
        if isna.any():
            self.frame.loc[isna, "link"] = self.frame.metaframe.default["link"]
            self.frame.loc[isna, "factor"] = self.frame.metaframe.default["factor"]
        link = np.asarray(self.frame.link, dtype=object)
        isnumeric = np.array(
            [
                isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_))
                for value in link
            ],
            dtype=bool,
        )
        istrue = np.array(
            [isinstance(value, (bool, np.bool_)) and bool(value) for value in link],
            dtype=bool,
        )
        isstr = np.array([isinstance(value, str) for value in link], dtype=bool)
        reset = ~istrue & ~isnumeric & ~isstr
        if reset.any():
            self.frame.loc[reset, "link"] = ""
        index = self.frame.index[istrue | isnumeric]
        if len(index) > 0:
            with self.frame.setlock(True, "multipoint"):
                factor = np.asarray(self.frame.factor)[istrue | isnumeric][1:]
                self.link(index, factor)
        self.frame.loc[:, "link"] = np.array(
            [str(value) for value in self.frame.link], dtype=object
        )
        self.sort_link()
        self.build()

    def sort_link(self):
        """Ensure links point to a monotonically earlier row."""
        for position, link in enumerate(self.frame.link):
            if link and link in self.frame.index:
                name = self.frame.index[position]
                link_index = self.frame.index.get_loc(link)
                if link_index > position:  # reverse
                    self.frame.at[link, "link"] = name
                    self.frame.loc[np.asarray(self.frame.link) == link, "link"] = name
                    self.frame.at[name, "link"] = ""

    def build(self):
        """Compute the ref / subref reduction indices from the links."""
        range_index = np.arange(len(self.frame), dtype=int)
        link = np.asarray(self.frame.link)
        self.indexer = list(range_index[link == ""])
        self.index = self.frame.index[self.indexer]
        ref = self.frame.index.get_indexer(link)
        if np.any(ref == -1):
            split = [name.split("_")[0] for name in self.frame.index]
            ref[ref == -1] = [
                split.index(value) if value in split else 0
                for position, value in enumerate(link)
                if ref[position] == -1
            ]
        ref[self.indexer] = range_index[self.indexer]
        self.frame.ref = ref
        subref = np.zeros(len(self.frame), dtype=int)
        subref[self.indexer] = np.arange(len(self.indexer), dtype=int)
        self.frame.subref = subref[ref]

    def expand_index(self, index, factor):
        """Expand link targets across a subframe's frame membership column."""
        if "frame" not in self.frame:
            raise IndexError("frame column required for index expansion")
        factor = [1] + list(factor)
        subindex, subfactor = [], []
        frame_col = np.asarray(self.frame["frame"])
        for name, fact in zip(index, factor):
            names = self.frame.index[frame_col == name]
            if len(names) == 0:
                raise IndexError(
                    f"name {name} not listed in frame {np.unique(frame_col)}"
                )
            subindex.extend(list(names))
            subfactor.extend(fact * np.ones(len(names)))
        return subindex, subfactor[1:]

    def link(self, index, factor=1, expand=False):
        """Define a multi-point constraint linking a set of rows."""
        if not is_list_like(index):
            raise IndexError(f"index: {index} is not list like")
        if not is_list_like(factor):
            factor = factor * np.ones(len(index) - 1)
        if expand:
            index, factor = self.expand_index(index, factor)
        name = index[0]
        link = self.frame.at[name, "link"]
        if isinstance(link, str) and link != "":
            name = link
        else:
            self.frame.at[name, "link"] = ""
            self.frame.at[name, "factor"] = 1
        index_number = len(index)
        if index_number == 1:
            return
        if len(factor) != index_number - 1:
            raise IndexError(
                f"len(factor={factor}) must == 1 for == len(index={index})-1"
            )
        for i in np.arange(1, index_number):
            self.frame.at[index[i], "link"] = str(name)
            self.frame.at[index[i], "factor"] = factor[i - 1]
        if self.frame.lock("multipoint") is False:
            # snapshot the store before reinitialising: __init__ clears _store
            # before reading its data, so a bare self-reinit would empty the frame
            columns = {
                name: np.asarray(self.frame._store.get(name))
                for name in self.frame._store.column_names()
            }
            self.frame.__init__(
                columns,
                index=list(self.frame.index),
                attrs={"metaframe": self.frame.metaframe},
            )

    def drop(self, index):
        """Reset links referencing dropped rows, then rebuild."""
        if not self.generate:
            return
        if not is_list_like(index):
            index = [index]
        link = np.asarray(self.frame.link)
        reset = np.array([value in index for value in link], dtype=bool)
        if reset.any():
            self.frame.loc[reset, "link"] = ""
        self.initialize()
