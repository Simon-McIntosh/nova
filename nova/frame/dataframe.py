"""Columnar frame base: typed struct-of-arrays table with a schema.

``DataFrame`` replaces the former pandas.DataFrame subclass. Data lives in a
:class:`~nova.frame.columnar.ColumnStore`; schema policy (required / additional
columns, ~90 defaults, column groups, locks, version hashes) lives in
:class:`~nova.frame.metaframe.MetaFrame`. Row insertion is array
concatenation and label lookup is index ``get_loc`` + slice. Pandas and xarray
are reached only through the lazy interchange helpers.
"""

from __future__ import annotations

from contextlib import contextmanager
import re
import string

import numpy as np
import xxhash

from nova.frame.columnar import ColumnStore, Index, is_list_like
from nova.frame.error import ColumnError
from nova.frame.metaframe import MetaFrame

# pylint: disable=too-many-public-methods


class Columns(list):
    """Column-name view with the small pandas.Index surface callers use."""

    def to_list(self) -> list:
        """Return the column names as a plain list."""
        return list(self)

    @property
    def empty(self) -> bool:
        """Return True when no columns are defined."""
        return len(self) == 0


class DataFrame:
    """Typed columnar table governed by a MetaFrame schema."""

    def __init__(self, data=None, index=None, columns=None, attrs=None, **metadata):
        """Assemble the store from data / index / columns and schema metadata."""
        self.attrs: dict = {}
        self._store = ColumnStore()
        self.extract_attrs(data, attrs)
        self.trim_columns(columns)
        data, source_index = self._as_dict(data, columns)
        self.extract_available(data, columns)
        self.update_metaframe(metadata)
        self._build_store(data, index, columns, source_index)
        self.match_columns()
        self.update_columns()
        self.update_version()

    # -- construction helpers ------------------------------------------------

    @staticmethod
    def _poison_store(store):
        """Fill a detached store's float arrays with NaN.

        A row insert rebuilds the store, detaching the arrays a caller may
        still hold from ``loc`` / ``sloc``; poisoning those float columns turns
        a silently-stale read into an obvious one. Only used where the surviving
        store is a fresh copy, so the live data is never touched.
        """
        for array in store.columns.values():
            if array.dtype.kind == "f":
                array[:] = np.nan

    def _as_dict(self, data, columns):
        """Return (column dict, source index) for the input data.

        A DataFrame or pandas.DataFrame (interchange input) is read
        column-wise and its labels returned so that a requested index can
        reindex by label; a mapping is copied; None becomes empty.
        """
        if data is None:
            return {}, None
        if isinstance(data, DataFrame):
            names = columns if columns is not None else data.columns.to_list()
            # read the raw store, not data[name] (which would inflate subspace)
            frame = {
                name: np.asarray(data._store.get(name))
                for name in names
                if name in data
            }
            return frame, data.index.values
        if hasattr(data, "columns") and hasattr(data, "to_dict"):
            # pandas.DataFrame interchange input
            frame = {col: np.asarray(data[col]) for col in list(data.columns)}
            if columns is not None:
                frame = {col: frame[col] for col in columns if col in frame}
            return frame, np.asarray(data.index)
        if isinstance(data, dict):
            return dict(data), None
        raise TypeError(f"unsupported data input {type(data)}")

    def _build_store(self, data, index, columns, source_index):
        """Build the column store, honouring an explicit or reindexing index."""
        length = self._index_length(data)
        if (
            index is not None
            and source_index is not None
            and len(list(index)) != length
        ):
            # reindex a frame input by label selection
            self._store = ColumnStore(
                data, index=source_index, defaults=self._store_defaults()
            )
            positions = [
                int(np.flatnonzero(source_index == label)[0]) for label in index
            ]
            self._store.drop(
                [i for i in range(len(source_index)) if i not in positions]
            )
            self._store.index = Index(list(index))
        else:
            if index is None and source_index is not None:
                # a frame input carries its own labels; adopt them unless they
                # are only a default positional index (then defer to build_index)
                labels = [str(label) for label in np.asarray(source_index)]
                if labels != [str(position) for position in range(len(labels))]:
                    index = labels
            if length == 0 and index is not None:
                length = len(list(index))
            built_index = self._build_index(length, index)
            self._store = ColumnStore(
                data,
                index=built_index if built_index else None,
                defaults=self._store_defaults(),
            )
            if not data and length == 0:
                self._store.index = Index([])
        if columns is not None and not data:
            for col in columns:
                if col not in self._store:
                    self._store.set(col, self.metaframe.default.get(col))
        self.metaframe.index = self._store.index

    def match_columns(self):
        """Trim required columns to those actually present in the store."""
        if not self.columns.empty and len(self) > 0:
            self.metaframe.metadata = {
                "Required": [a for a in self.metaframe.required if a in self._store]
            }

    # -- schema / metadata ---------------------------------------------------

    def _store_defaults(self):
        """Return the schema defaults with polymorphic columns typed as object.

        ``link`` may hold a bool flag, a numeric factor or a label, so it must
        never take a scalar's dtype (which would forbid the other forms).
        """
        defaults = dict(self.metaframe.default)
        defaults["link"] = None
        return defaults

    @property
    def metaframe(self) -> MetaFrame:
        """Return the schema / metadata container."""
        if "metaframe" not in self.attrs:
            self.attrs["metaframe"] = MetaFrame(Index([]))
        return self.attrs["metaframe"]

    @property
    def version(self):
        """Return the metaframe version hash container."""
        return self.metaframe.version

    def extract_attrs(self, data, attrs):
        """Adopt a metaframe / metamethods carried on data or attrs."""
        for source in (getattr(data, "attrs", None), attrs):
            if not source:
                continue
            for key, value in source.items():
                if isinstance(value, (MetaFrame,)) or key != "metaframe":
                    self.attrs[key] = value
        if "metaframe" not in self.attrs:
            self.attrs["metaframe"] = MetaFrame(Index([]))

    def trim_columns(self, columns):
        """Trim required / additional / available to the requested columns."""
        if columns:
            self.metaframe.metadata = {
                "Required": [a for a in self.metaframe.required if a in columns],
                "Additional": [a for a in self.metaframe.additional if a in columns],
                "Available": [a for a in self.metaframe.available if a in columns],
            }

    def extract_available(self, data, columns):
        """Record data + requested columns as available."""
        data_columns = list(data) if data else []
        frame_columns = list(dict.fromkeys(list(columns))) if columns else []
        self.metaframe.metadata = {"available": data_columns + frame_columns}

    def update_metaframe(self, metadata):
        """Update the metaframe, promoting a version list to a dict."""
        if isinstance(metadata.get("version", None), list):
            metadata["version"] = dict.fromkeys(metadata["version"])
        if "metadata" in metadata:
            metadata = {**metadata, **metadata.pop("metadata")}
        self.metaframe.update(metadata)
        if self.metaframe.columns:
            self.metaframe.metadata = {"available": self.metaframe.columns}

    def update_columns(self):
        """Ensure required columns exist and additional columns get defaults."""
        columns = self._store.column_names()
        if not columns:
            for attr in self.metaframe.columns:
                self._store.set(attr, self.metaframe.default[attr])
                self._store.columns[attr] = self._store.columns[attr][:0]
            return
        required_unset = [a for a in self.metaframe.required if a not in columns]
        if required_unset:
            raise IndexError(f"required attributes missing {required_unset}")
        additional = [a for a in columns if a not in self.metaframe.columns]
        if additional:
            self.metaframe.metadata = {"additional": additional}
        for attr in self.metaframe.columns:
            if attr not in self._store.columns:
                self._store.set(attr, self.metaframe.default[attr])
        turn_set = all(a in self._store.columns for a in ["It", "nturn"])
        if "Ic" in [a for a in self.metaframe.columns if a not in columns] and turn_set:
            self._store.set("Ic", self._store.get("It") / self._store.get("nturn"))

    # -- index construction --------------------------------------------------

    @staticmethod
    def _index_length(data) -> int:
        """Return the row count implied by the longest list-like column."""
        if not data:
            return 0
        lengths = [len(list(v)) for v in data.values() if is_list_like(v)]
        return int(max(lengths)) if lengths else 1

    def _build_index(self, length, index):
        """Return a label list of the given length.

        Explicit index labels pass through; otherwise labels are built from
        the schema tag defaults (name / label / delim / offset / append).
        """
        if length == 0:
            return []
        if index is not None:
            return self._check_index(list(index), length)
        return self.build_index(length)

    def build_index(self, count, **kwargs):
        """Return a label index of ``count`` rows from the tag defaults.

        ``count`` is positional (never a keyword) so a ``length`` tag carried
        in ``kwargs`` cannot collide with it.
        """
        metatag = {
            key: kwargs.get(key, self.metaframe.default[key])
            for key in self.metaframe.tag
        }
        name = metatag["name"]
        if is_list_like(name) or (isinstance(name, str) and len(name) > 0):
            if is_list_like(name) or count == 1:
                return self._check_index(name, count)
            self._set_label(metatag, name)
        self._set_offset(metatag)
        label_delim = metatag["label"] + metatag["delim"]
        index = [f"{label_delim}{i + metatag['offset']:d}" for i in range(count)]
        if metatag["delim"] and metatag["label"] not in self.index:
            index[0] = metatag["label"]
        return self._check_index(index, count)

    def _set_label(self, metatag, name):
        """Split a name into label + offset per the append / delim policy."""
        if not metatag["append"]:
            metatag["label"] = name
            metatag["offset"] = 0
            return
        if metatag["delim"] and metatag["delim"] in name:
            split_name = name.split(metatag["delim"])
            metatag["label"] = metatag["delim"].join(split_name[:-1])
            metatag["offset"] = int(split_name[-1])
        else:
            metatag["label"] = name.rstrip(string.digits)
            try:
                metatag["offset"] = int(name.lstrip(string.ascii_letters))
            except ValueError:
                pass

    def _set_offset(self, metatag):
        """Advance the offset past any existing label matches in the index."""
        try:
            match = next(name for name in self.index[::-1] if metatag["label"] in name)
            if metatag["delim"] and metatag["delim"] in match:
                offset = int(match.split(metatag["delim"])[-1])
            else:
                offset = re.sub(r"[a-zA-Z]", "", match)
            if isinstance(offset, str):
                offset = offset.replace(metatag["delim"], "").replace("_", "")
                offset = int(offset)
            offset += 1
        except TypeError, ValueError, StopIteration:
            offset = 0
        metatag["offset"] = int(np.max([offset, metatag["offset"]]))

    def _check_index(self, index, length):
        """Validate a candidate label index: length, uniqueness, no clashes."""
        if not is_list_like(index):
            index = [index]
        index = [str(label) for label in index]
        if len(index) != length:
            raise IndexError(
                f"missmatch between len(index) {len(index)} and "
                f"maximum length data column {length}"
            )
        if len(index) != len(np.unique(index)):
            raise IndexError(f"index not unique {index}")
        taken = [name in self.index for name in index]
        if np.array(taken).any():
            raise IndexError(
                f"{np.array(index)[taken]} already defined in self.index: {self.index}"
            )
        return index

    # -- accessors -----------------------------------------------------------

    @property
    def index(self) -> Index:
        """Return the string label index."""
        return self._store.index

    @property
    def columns(self) -> Columns:
        """Return the column-name view."""
        return Columns(self._store.column_names())

    @property
    def empty(self) -> bool:
        """Return True when the frame holds no rows."""
        return len(self._store) == 0

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self._store)

    def __contains__(self, name) -> bool:
        """Return True when name is a stored column."""
        return name in self._store

    def __getitem__(self, col):
        """Return a stored column as a Vector."""
        return self._store.get(col)

    def __setitem__(self, col, value):
        """Set a stored column, guarding creation of unknown named columns."""
        if self.lock("column") is False:
            self.check_column(col)
        self._store.set(col, value)

    def __getattr__(self, name):
        """Expose attrs entries and stored columns as attributes."""
        if name.startswith("_") or name in ("attrs",):
            raise AttributeError(name)
        attrs = self.__dict__.get("attrs", {})
        if name in attrs:
            return attrs[name]
        store = self.__dict__.get("_store")
        if store is not None and name in store:
            return store.get(name)
        self.check_column(name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        """Set stored columns by attribute; other names set normally."""
        if name in ("attrs", "_store") or name.startswith("_"):
            super().__setattr__(name, value)
            return
        store = self.__dict__.get("_store")
        if store is not None and name in store:
            self._store.set(name, value)
            return
        if store is not None and name in self.metaframe.default:
            self.check_column(name)
        super().__setattr__(name, value)

    def get(self, col, default=None):
        """Return a column if present, else default."""
        if col in self._store:
            return self._store.get(col)
        return default

    def check_column(self, name):
        """Raise ColumnError for a schema column absent from the store."""
        if name in self.metaframe.default and name not in self._store:
            raise ColumnError(name)

    # -- locks / dtype / hashing --------------------------------------------

    def frame_attrs(self, *args):
        """Load and initialise the given metamethods when their columns exist."""
        for arg in args:
            metamethod = arg(self)
            if not metamethod.generate:
                continue
            method = metamethod()
            if not self.hasattrs(method.name):
                self.update_columns()
            self.attrs[method.name] = method
            self.attrs[method.name].initialize()

    def frame_attr(self, method, *method_args):
        """Load and initialise a single metamethod subclass."""
        name = method.name
        if method(self, *method_args).generate:
            self.update_columns()
            self.attrs[name] = method(self)
            self.attrs[name].initialize()

    def hasattrs(self, attr) -> bool:
        """Return True when attr is present in attrs."""
        return attr in self.attrs

    def hascol(self, attr, col) -> bool:
        """Return True when col is a member of column group attr."""
        return self.metaframe.hascol(attr, col)

    def col_dtype(self, col, value):
        """Return the python dtype implied by the schema default for col."""
        if col == "link":
            return None
        try:
            default = self.metaframe.default[col]
        except KeyError, TypeError:
            return None
        if default is None:
            return None
        return type(default)

    def format_value(self, col, value):
        """Cast value to the schema dtype for col."""
        dtype = self.col_dtype(col, value)
        if dtype is None:
            return value
        if value is None:
            return dtype(0)
        if is_list_like(value):
            return np.array(value, dtype)
        return dtype(value)

    def lock(self, key=None):
        """Return the metaframe lock status (all locks when key is None)."""
        if key is None:
            return self.metaframe.lock
        return self.metaframe.lock[key]

    @contextmanager
    def setlock(self, status, keys=None):
        """Temporarily set lock(s), restoring prior state on exit."""
        if keys is None:
            keys = list(self.metaframe.lock.keys())
        if isinstance(keys, str):
            keys = [keys]
        lock = {key: self.metaframe.lock[key] for key in keys}
        self.metaframe.lock |= {key: status for key in keys}
        try:
            yield
        finally:
            self.metaframe.lock |= lock

    def hash_array(self, attr):
        """Return the array hashed for versioning (subspace-aware)."""
        if self.hasattrs("subspace") and attr in getattr(self, "subspace", []):
            return getattr(self.subspace, attr)
        if attr == "index":
            return self.index.values
        return np.asarray(self[attr])

    def loc_hash(self, attr):
        """Return the xxh64 hash of a loc attribute, or None when absent."""
        try:
            value = self.hash_array(attr)
        except ColumnError, KeyError, AttributeError:
            return None
        try:
            return xxhash.xxh64(np.ascontiguousarray(value)).intdigest()
        except TypeError, ValueError:
            return xxhash.xxh64(np.ascontiguousarray(value.astype(str))).intdigest()

    def update_version(self):
        """Refresh the version hash dict from the current columns."""
        self.metaframe.update(
            dict(version={attr: self.loc_hash(attr) for attr in self.version})
        )

    def update_metaframe_required(self, required):
        """Set the required column list on the metaframe."""
        self.metaframe.metadata = {"Required": required}

    # -- interchange / persistence ------------------------------------------

    def to_pandas(self):
        """Return a pandas.DataFrame view (lazy interchange shim)."""
        import pandas

        return pandas.DataFrame(
            {name: np.asarray(self[name]) for name in self.columns},
            index=np.asarray(self.index),
        )

    def to_xarray(self):
        """Return an xarray.Dataset view (lazy interchange shim)."""
        import xarray

        return xarray.Dataset(
            {name: ("index", self._netcdf_array(self[name])) for name in self.columns},
            coords={"index": self._netcdf_array(self.index.values)},
        )

    @staticmethod
    def _netcdf_array(array):
        """Return an array cast to a netCDF-writable dtype.

        Object arrays of strings become fixed-width unicode; other object
        arrays (geometry) are handled separately by the caller.
        """
        array = np.asarray(array)
        if array.dtype == object and (
            array.size == 0 or isinstance(array.flat[0], str)
        ):
            return array.astype(str)
        return array

    def store(self, filepath, group=None, mode="w", vtk=False):
        """Store the frame as a netCDF group via xarray."""
        dataset = self.to_xarray()
        dataset.attrs = self._extract_metadata()
        for col in ["poly", "vtk"]:
            if col == "vtk" and not vtk:
                dataset = dataset.drop_vars("vtk", errors="ignore")
                continue
            if col in self.columns:
                dataset[col] = ("index", self._dumps(col))
        dataset.to_netcdf(filepath, group=group, mode=mode)

    def load(self, filepath, group=None):
        """Load the frame from a netCDF group via xarray."""
        import xarray

        with xarray.open_dataset(filepath, group=group, cache=False) as dataset:
            dataset.load()
            metadata = self._insert_metadata(dict(dataset.attrs))
            columns = {name: np.asarray(dataset[name]) for name in dataset.data_vars}
            index = np.asarray(dataset["index"]).astype(str)
        self.__init__(columns, index=list(index), **metadata)
        for col in ("poly", "vtk"):
            if col in self.columns:
                self._loads(col)
        self.update_version()
        return self

    def _loads(self, col):
        """Rebuild geometry objects in a column from stored json strings."""
        import json

        def parse(geom):
            if not geom:
                return geom
            try:
                geo = json.loads(geom)["type"]
            except (TypeError, ValueError, KeyError):
                return geom  # not json-encoded geometry (e.g. a vtk blob)
            return self.geoframe(geo).loads(geom)

        self.loc[:, col] = [parse(geom) for geom in list(self[col])]

    def _extract_metadata(self):
        """Return schema metadata with the version promoted to a list."""
        import copy

        metadata = copy.deepcopy(self.metaframe.metadata)
        if "version" in metadata:
            metadata["version"] = list(metadata["version"])
        return {k: v for k, v in metadata.items() if k != "index"}

    @staticmethod
    def _insert_metadata(attrs):
        """Return metadata with a scalar version promoted to a list."""
        import copy

        metadata = copy.deepcopy(attrs)
        if "version" in metadata and isinstance(metadata["version"], str):
            metadata["version"] = [metadata["version"]]
        return metadata

    _geoframe = {
        "LineString": ".geometry.polyframe.PolyFrame",
        "MultiLineString": ".geometry.polyframe.PolyFrame",
        "Polygon": ".geometry.polyframe.PolyFrame",
        "MultiPolygon": ".geometry.polyframe.PolyFrame",
        "VTK": ".geometry.vtkgen.VtkFrame",
        "Geo": ".geometry.geoframe.GeoFrame",
    }

    def geoframe(self, geo: str):
        """Return the geometry class registered for a geometry type name."""
        from importlib import import_module

        if geo == "Json":
            return str
        module = ".".join(self._geoframe[geo].split(".")[:-1])
        method = self._geoframe[geo].split(".")[-1]
        return getattr(import_module(module, "nova"), method)

    def geotype(self, geo: str, col: str) -> np.ndarray:
        """Return a boolean mask of column entries matching a geometry type."""
        return np.array(
            [isinstance(geom, self.geoframe(geo)) for geom in self[col]], dtype=bool
        )

    def _dumps(self, col):
        """Serialise geometry objects in a column to json strings."""
        return [geom.dumps() if geom else "" for geom in self[col]]

    def __repr__(self):
        """Return a concise representation of the frame."""
        return f"{type(self).__name__}(index={self.index.to_list()!r}, columns={self.columns.to_list()!r})"


if __name__ == "__main__":
    dataframe = DataFrame(
        base=["x", "y", "z"], required=["x"], additional=["Ic", "z"], label="PF"
    )
