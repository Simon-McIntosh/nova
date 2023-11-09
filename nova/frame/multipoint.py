"""Manage mulit-point constraints."""
from dataclasses import dataclass, field

import numpy as np
import pandas

import nova.frame.metamethod as metamethod
from nova.frame.dataframe import DataFrame


@dataclass
class MultiPoint(metamethod.MultiPoint):
    """Manage multi-point constraints applied across frame.index."""

    name = "multipoint"

    frame: DataFrame = field(repr=False)
    additional: list[str] = field(default_factory=lambda: ["factor", "ref", "subref"])
    indexer: list[int] = field(init=False, repr=False)
    index: pandas.Index = field(default_factory=lambda: pandas.Index([]))

    def initialize(self):
        """
        Init multipoint.frame constraints if key_attributes in columns.

            - link is none or NaN:

                - link = ''
                - factor = 0

            - link is bool:

                - link = '' if False else 'index[0]'
                - factor = 0 if False else 1

            - link is int or float:

                - link = 'index[0]'
                - factor = value

        """
        self.frame.link = self.frame.link.astype(object)
        isna = pandas.isna(self.frame.link)
        self.frame.loc[isna, "link"] = self.frame.metaframe.default["link"]
        self.frame.loc[isna, "factor"] = self.frame.metaframe.default["factor"]
        isnumeric = np.array(
            [
                isinstance(link, (int, float)) & (not isinstance(link, bool))
                for link in self.frame.link
            ],
            dtype=bool,
        )
        istrue = np.array([link is True for link in self.frame.link], dtype=bool)
        isstr = np.array(
            [isinstance(link, str) for link in self.frame.link], dtype=bool
        )
        self.frame.loc[~istrue & ~isnumeric & ~isstr, "link"] = ""
        index = self.frame.index[istrue | isnumeric]
        if not index.empty:
            with self.frame.setlock(True, "multipoint"):
                factor = self.frame.factor
                factor = factor[istrue | isnumeric][1:]
                self.link(index, factor.values)
        self.frame.link = self.frame.link.astype(str)
        self.sort_link()
        self.build()

    def sort_link(self):
        """Update frame.links to ensure monotonic increasing."""
        for index, link in enumerate(self.frame.link):
            if link and link in self.frame.index:
                name = self.frame.index[index]
                link_index = self.frame.index.get_loc(link)
                if link_index > index:  # reverse
                    self.frame.loc[link, "link"] = name
                    self.frame.loc[self.frame.link == link, "link"] = name
                    self.frame.loc[name, "link"] = ""

    def build(self):
        """Update multi-point parameters."""
        range_index = np.arange(len(self.frame), dtype=int)
        self.indexer = list(range_index[self.frame.link == ""])
        self.index = self.frame.index[self.indexer]
        ref = self.frame.index.get_indexer(self.frame.link)
        ref[ref == -1] = 0
        ref[self.indexer] = range_index[self.indexer]
        self.frame.ref = ref
        subref = np.zeros(len(self.frame), dtype=int)
        subref[self.indexer] = np.arange(len(self.indexer), dtype=int)
        self.frame.subref = subref[ref]

    def expand_index(self, index, factor):
        """Return subindex extracted from frame column."""
        if "frame" not in self.frame:
            raise IndexError("frame column required for index expansion")
        factor = [1] + list(factor)
        subindex, subfactor = [], []
        for name, fact in zip(index, factor):
            names = self.frame.index[self.frame.frame == name]
            if len(names) == 0:
                raise IndexError(
                    f"name {name} not listed in frame " f"{np.unique(self.frame.frame)}"
                )
            subindex.extend(names)
            subfactor.extend(fact * np.ones(len(names)))
        return subindex, subfactor[1:]

    def link(self, index, factor=1, expand=False):
        """
        Define multi-point constraint linking a set of coils.

        Parameters
        ----------
        index : list[str]
            List of coil names (present in self.frame.index).
        factor : float, optional
            Inter-coil coupling factor. The default is 1.

        Raises
        ------
        IndexError

            - index must be list-like
            - len(index) must be greater than l
            - len(factor) must equal 1 or len(name)-1.

        Returns
        -------
        None.

        """
        if not pandas.api.types.is_list_like(index):
            raise IndexError(f"index: {index} is not list like")
        if not pandas.api.types.is_list_like(factor):
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
            self.frame.at[index[i], "link"] = name
            self.frame.at[index[i], "factor"] = factor[i - 1]
        if self.frame.lock("multipoint") is False:
            self.frame.__init__(self.frame, attrs=self.frame.attrs)

    def drop(self, index):
        """
        Remove multi-point constraints referancing dropped coils.

        Parameters
        ----------
        index : Union[str, list[str], pandas.Index]
            Dropped coil index.

        Returns
        -------
        None.

        """
        if self.generate:
            if not pandas.api.types.is_list_like(index):
                index = [index]
            reset = [link in index for link in self.frame.link]
            self.frame.loc[reset, "link"] = ""
            self.initialize()
