"""Manage file data access for frame and biot instances."""

from contextlib import contextmanager
from dataclasses import dataclass, field

import pandas

from nova.frame.framespace import FrameSpace


@dataclass
class FrameData:
    """Package FrameSpace instances. Manage boolean methods."""

    frame: FrameSpace = field(repr=False, default_factory=FrameSpace)
    subframe: FrameSpace = field(repr=False, default_factory=FrameSpace)

    @property
    def frames(self):
        """Return frame and subframe."""
        return self.frame, self.subframe

    @frames.setter
    def frames(self, frames):
        """Set frame and subframe."""
        self.frame, self.subframe = frames

    @contextmanager
    def insert_required(self, required=None):
        """Manage local required arguments."""
        _required = self.frame.metaframe.required.copy()
        if required is None:
            required = self.required
        if isinstance(required, str):
            required = [required]
        self.update_required(required)
        if len(required) != len(self.frame.metaframe.required):
            raise AttributeError(
                f"Required attrs not set {required} ensure that attrs are "
                "present in frame.columns via updates to frame.metaframe:\n"
                f"required: {self.frame.metaframe.required}\n"
                f"additional: {self.frame.metaframe.additional}\n"
                f"available: {self.frame.metaframe.available}\n\n"
            )
        yield
        self.update_required(_required)

    def update_required(self, required):
        """Update frame and subframe required arguments."""
        for frame in self.frames:
            frame.update_metaframe(dict(Required=required))

    def linkframe(self, index, factor=1):
        """Apply multipoint link to subframe.

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
        self.frame.multipoint.link(index, factor)
        self.subframe.multipoint.link(index, factor, expand=True)

    def linksubframe(self, index):
        """Propogate subframe links from frame."""
        for i, (link, factor) in enumerate(
            zip(self.frame.loc[index, "link"], self.frame.loc[index, "factor"])
        ):
            if link == "":
                continue
            self.subframe.multipoint.link([link, index[i]], factor, expand=True)

    def drop(self, index=None):
        """
        Remove frame and subframe.

        Parameters
        ----------
        index : int or list or pandas.Index, optional
            Index of coils to be removed. The default is None (all coils).

        Returns
        -------
        loc : [int, int]
            Location index of first removed [frame, subframe].

        """
        if index is None:  # drop all coils
            index = self.coil.index
        if not pandas.api.types.is_list_like(index):
            index = [index]
        loc = self.get_loc(index)
        for name in index:
            if name in self.frame.index:
                self.subframe.drop(self.coil.loc[name, "subindex"])
                self.frame.drop(name)
        return loc

    def translate(self, index=None, dx=0, dz=0):
        """
        Translate coil in polidal plane.

        Parameters
        ----------
        index : int or array-like or Index, optional
            Coil index. The default is None (all coils).
        dx : float, optional
            x-coordinate translation. The default is 0.
        dz : float, optional
            z-coordinate translation. The default is 0.

        Returns
        -------
        None.

        """
        if index is None:
            index = self.coil.index
        elif not pandas.api.types.is_list_like(index):
            index = [index]
        self.coil.translate(index, dx, dz)
        for name in index:
            self.subcoil.translate(self.coil.loc[name, "subindex"], dx, dz)

    def _get_iloc(self, index):
        iloc = [None, None]
        for name in index:
            if name in self.coil.index:
                iloc[0] = self.coil.index.get_loc(index[0])
                subindex = self.coil.subindex[index[0]][0]
                iloc[1] = self.subcoil.index.get_loc(subindex)
                break
        return iloc
