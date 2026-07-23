"""Frame with row insertion, union, energization and link metamethods.

``FrameLink`` adds the mutating frame surface on top of the columnar
:class:`~nova.frame.dataarray.DataArray`: schema-checked row ``insert`` (array
concatenation), ``+=`` union, ``drop``, the It = Ic * nturn energization
coupling, and the multi-point link metamethod. Pandas is reached only as
optional interchange input.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy as copy_module

import numpy as np

from nova.frame.columnar import ColumnStore, Vector, is_list_like
from nova.frame.dataarray import DataArray
from nova.frame.metamethod import Energize, MultiPoint, Select

# pylint: disable=too-many-ancestors


class FrameLink(DataArray):
    """Columnar frame with insertion, union and link / energize metamethods."""

    def __init__(self, data=None, index=None, columns=None, attrs=None, **metadata):
        """Build the store then load the select / multipoint / energize methods."""
        super().__init__(data, index, columns, attrs, **metadata)
        self.frame_attrs(Select, MultiPoint, Energize)

    # -- energization: It = Ic * nturn --------------------------------------

    def _energized(self, col) -> bool:
        """Return True when col is the coupled turn current and derivable."""
        if col != "It":
            return False
        if self.__dict__.get("_store") is None or not self.hasattrs("metaframe"):
            return False
        return (
            self.hascol("energize", "It")
            and not self.hascol("subspace", "It")
            and self.lock("energize") is False
            and "Ic" in self._store
            and "nturn" in self._store
        )

    def _turn_current(self) -> Vector:
        """Return the derived turn current Ic * nturn."""
        return (np.asarray(self["Ic"]) * np.asarray(self["nturn"])).view(Vector)

    def __getitem__(self, col):
        """Return a column, deriving It from Ic * nturn when energized."""
        if self._energized(col):
            return self._turn_current()
        return super().__getitem__(col)

    def __setitem__(self, col, value):
        """Set a column, mapping an It write onto Ic = It / nturn."""
        if self._energized(col):
            with self.setlock(True, "energize"):
                self["Ic"] = np.asarray(value, dtype=float) / np.asarray(
                    self["nturn"], dtype=float
                )
            return
        super().__setitem__(col, value)

    def __getattr__(self, name):
        """Expose the derived turn current as an attribute."""
        if name == "It" and self.__dict__.get("_store") is not None:
            if self._energized("It"):
                return self._turn_current()
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        """Route an It attribute write through the energization coupling."""
        if name == "It" and self.__dict__.get("_store") is not None:
            if self._energized("It"):
                self["It"] = value
                return
        super().__setattr__(name, value)

    # -- union operators -----------------------------------------------------

    @staticmethod
    def _default_index(frame) -> bool:
        """Return True when a frame carries only the default positional index.

        A default index (``0, 1, 2, …``, from either a pandas RangeIndex or the
        columnar store's auto-labels) carries no naming intent, so an insert
        should derive row labels from the label / delim tags instead.
        """
        labels = [str(label) for label in np.asarray(frame.index)]
        return labels == [str(position) for position in range(len(labels))]

    @staticmethod
    def isframe(obj, dataframe=True) -> bool:
        """Return True when obj is a frame (or a pandas.DataFrame interchange)."""
        if isinstance(obj, FrameLink):
            return True
        if dataframe and hasattr(obj, "columns") and hasattr(obj, "to_dict"):
            return True
        return False

    @contextmanager
    def insert_required(self, required=None):
        """Temporarily set the required column list for an insert."""
        stored = self.metaframe.required.copy()
        if required is None:
            required = stored
        self.update_metaframe(dict(Required=required))
        try:
            yield
        finally:
            self.update_metaframe(dict(Required=stored))

    @staticmethod
    def _unpack_add(other):
        """Return (args, kwargs, required) for the union operand."""
        if FrameLink.isframe(other):
            return [other], {}, list(other.columns)
        if isinstance(other, dict):
            return [], other, None
        return other, {}, None

    def __copy__(self):
        """Return a deep copy of the frame."""
        frame = self.__class__()
        frame.__init__(self, attrs={"metaframe": copy_module.deepcopy(self.metaframe)})
        return frame

    def __add__(self, other):
        """Return the union of self and other."""
        frame = copy_module.copy(self)
        frame += other
        return frame

    def __iadd__(self, other):
        """Augment self in place by other."""
        args, kwargs, required = self._unpack_add(other)
        with self.insert_required(required):
            self.insert(*args, **kwargs)
        return self

    # -- insertion -----------------------------------------------------------

    def insert(self, *args, iloc=None, **kwargs):
        """Insert row(s) assembled from required args and optional kwargs."""
        self.metaframe.metadata = kwargs.pop("metadata", {})
        insert = self.assemble(*args, **kwargs)
        self.concatenate(insert, iloc=iloc)
        return insert.index

    def assemble(self, *args, **kwargs):
        """Return a FrameLink assembled from required and optional input."""
        args, kwargs = self._extract_frame(*args, **kwargs)
        args, kwargs = self._extract_polygon(*args, **kwargs)
        data = self._build_data(*args, **kwargs)
        length = self._index_length(data)
        index = self.build_index(length, **kwargs) if length else []
        return FrameLink(data, index=index, attrs={"metaframe": self.metaframe})

    def concatenate(self, *inserts, iloc=None):
        """Concatenate assembled insert(s) with self, then reinitialise."""
        store = ColumnStore(
            {
                name: np.asarray(self._store.get(name))
                for name in self._store.column_names()
            },
            index=list(self.index),
            defaults=self._store_defaults(),
        )
        for insert in inserts:
            insert_store = ColumnStore(
                {
                    name: np.asarray(insert._store.get(name))
                    for name in insert._store.column_names()
                },
                index=list(insert.index),
                defaults=self._store_defaults(),
            )
            if len(store) == 0:
                store = insert_store
            else:
                store.concatenate(insert_store, iloc=iloc)
        columns = {name: store.get(name) for name in store.column_names()}
        self.__init__(
            columns, index=list(store.index), attrs={"metaframe": self.metaframe}
        )
        self.update_version()
        return self

    def _extract_frame(self, *args, **kwargs):
        """Replace a frame arg[0] with its required columns and kwargs."""
        required = self.metaframe.required
        if len(args) != 1:
            args += tuple(kwargs.pop(arg) for arg in required if arg in kwargs)
            return args, kwargs
        if not self.isframe(args[0]):
            return args, kwargs
        frame = args[0]
        missing = [arg not in frame.columns for arg in required]
        if np.array(missing, dtype=bool).any():
            missing_cols = np.array(required)[np.array(missing, dtype=bool)]
            raise KeyError(
                f"required arguments {missing_cols} not present in frame "
                f"{list(frame.columns)}"
            )
        columns = list(frame.columns)
        args = [np.asarray(frame[col]) for col in required]
        for attr in required:
            kwargs.pop(attr, None)
        if not self._default_index(frame):
            # a frame carrying meaningful labels supplies the row names; a
            # default positional index defers naming to the label / delim tags
            kwargs["name"] = np.asarray(frame.index)
        kwargs |= {
            col: np.asarray(frame[col])
            for col in self.metaframe.columns
            if col in columns
        }
        if len(args) != len(required):
            raise IndexError(
                f"incorrect required argument number {len(args)} != {len(required)}"
            )
        return args, kwargs

    def _extract_polygon(self, *args, **kwargs):
        """Replace a polygon arg[0] with its geometry kwargs."""
        import shapely

        from nova.geometry.polygeom import PolyGeom
        from nova.geometry.polygon import Polygon

        if len(args) != 1:
            return args, kwargs
        if len(self.metaframe.required) == 1 and (
            not isinstance(
                args[0], (shapely.geometry.Polygon, shapely.geometry.MultiPolygon, dict)
            )
            or hasattr(args[0], "faces")
        ):
            return args, kwargs
        if isinstance(args[0], list) and all(
            isinstance(poly, (Polygon, shapely.geometry.Polygon)) for poly in args[0]
        ):
            multipoly = {attr: [] for attr in PolyGeom(args[0][0]).geometry}
            for poly in args[0]:
                geometry = PolyGeom(poly).geometry
                for attr in multipoly:
                    multipoly[attr].append(geometry[attr])
            kwargs = kwargs | multipoly
        else:
            kwargs = kwargs | PolyGeom(args[0]).geometry
        args = [kwargs.pop(attr) for attr in self.metaframe.required]
        return args, kwargs

    def _build_data(self, *args, **kwargs):
        """Return a column dict built from required args and optional kwargs."""
        data: dict = {}
        kwargs = self._exclude(kwargs)
        attrs = self.metaframe.required + list(kwargs)
        self._build_required(data, *args)
        self._build_additional(data, **kwargs)
        self._patch_current(data, attrs)
        return data

    def _exclude(self, kwargs):
        """Drop excluded attributes from kwargs."""
        for attr in self.metaframe.exclude:
            kwargs.pop(attr, None)
        return kwargs

    def _build_required(self, data, *args):
        """Populate required columns from positional args."""
        required = self.metaframe.required
        if len(args) != len(required):
            raise IndexError(f"len(args) {len(args)} != len(required) {len(required)}")
        for attr, arg in zip(required, args):
            try:
                data[attr] = np.array(arg, dtype=float)
            except TypeError, ValueError:
                data[attr] = arg

    def _build_additional(self, data, **kwargs):
        """Populate additional columns from defaults then kwargs."""
        for attr in self.metaframe.additional:
            data[attr] = self.metaframe.default[attr]
        additional = []
        for attr in list(kwargs):
            if attr in self.metaframe.tag:
                kwargs.pop(attr)
            elif attr in self.metaframe.default:
                value = kwargs.pop(attr)
                try:
                    data[attr] = np.array(value)
                except ValueError:
                    data[attr] = value
                if attr not in self.metaframe.additional:
                    additional.append(attr)
        if additional:
            self.metaframe.metadata = {"additional": additional}
        if kwargs:
            unset = np.array(list(kwargs))
            raise IndexError(
                f"unset kwargs: {unset}\nenter a default value in metaframe.default"
            )

    def _patch_current(self, data, attrs=None):
        """Patch the Ic / It pair from whichever was supplied."""
        if attrs is None:
            attrs = data
        nturn = data.get("nturn", self.metaframe.default["nturn"])
        if "It" in attrs and "Ic" not in attrs:
            data["Ic"] = np.asarray(data["It"]) / nturn
        elif "It" in attrs and "Ic" in attrs:
            data["It"] = np.asarray(data["Ic"]) * nturn
        elif "Ic" in attrs and "It" not in attrs and "It" in self.metaframe.columns:
            data["It"] = np.asarray(data["Ic"]) * nturn

    # -- mutation ------------------------------------------------------------

    def drop(self, index=None):
        """Drop frame row(s) by label, then rebuild links and version."""
        if index is None:
            index = list(self.index)
        elif not is_list_like(index):
            index = [index]
        if self.hasattrs("multipoint"):
            self.multipoint.drop(index)
        positions = [self.index.get_loc(name) for name in index if name in self.index]
        self._store.drop(positions)
        self.__init__(
            {
                name: np.asarray(self._store.get(name))
                for name in self._store.column_names()
            },
            index=list(self.index),
            attrs={"metaframe": self.metaframe},
        )
        self.update_version()

    def translate(self, index=None, xoffset=0, zoffset=0):
        """Translate row polygons and coordinates in the poloidal plane."""
        import shapely

        if index is None:
            index = list(self.index)
        elif not is_list_like(index):
            index = [index]
        if xoffset != 0:
            self.loc[index, "x"] += xoffset
        if zoffset != 0:
            self.loc[index, "z"] += zoffset
        for name in index:
            self.loc[name, "poly"] = shapely.affinity.translate(
                self.loc[name, "poly"], xoff=xoffset, yoff=zoffset
            )


if __name__ == "__main__":
    framelink = FrameLink(required=["x", "z"], Available=["It"], Array=["Ic"])
    framelink.insert([-4, -5], 1, Ic=6.5, name="PF1", active=False)
