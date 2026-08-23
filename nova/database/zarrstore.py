"""Facilitate structured access to a zarr-backed identity cache.

The identity cache stores compiled frames and solved operators keyed by a
content hash of the machine identity. This store backs that cache with zarr:
each cache entry is a named group inside a single ``.zarr`` store, and stale
entries are evicted with a native group delete rather than the whole-file
rewrite the netCDF backend is forced into. netCDF remains available through
:class:`~nova.database.netcdf.netCDF` for interchange and export.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import gc
from threading import RLock
import warnings

import numpy as np
import pandas
import xarray
import zarr
from xarray.backends.zarr import ZarrStore as XarrayZarrStore

from nova.database.filepath import FilePath


_zarr_string_encoding_lock = RLock()


def _numpy_string_variable(variable: xarray.Variable) -> xarray.Variable:
    """Convert a pandas string extension variable before xarray encodes it."""
    dtype = variable.data.dtype
    try:
        np.dtype(dtype)
    except TypeError:
        is_extension_string = pandas.api.types.is_string_dtype(dtype)
    else:
        is_extension_string = False
    if not is_extension_string:
        return variable
    values = np.asarray(variable.data)
    if not pandas.isna(values).any():
        values = values.astype(str)
    return xarray.Variable(
        variable.dims,
        values,
        attrs=variable.attrs,
        encoding=variable.encoding,
    )


@contextmanager
def suppress_unstable_string_spec():
    """Prepare strings for zarr and silence its unsettled string-spec warning.

    Frame labels, links and serialised geometry persist as fixed-length
    unicode. Pandas string extension arrays must first become ordinary NumPy
    arrays because xarray's CF encoder asks NumPy to classify every dtype and
    NumPy cannot classify ``StringDtype``. The conversion is confined to the
    zarr write and retains the variable's dimensions, attributes and encoding.

    Zarr v3 has no settled on-disk specification for fixed-length unicode. The
    store is rebuilt from source whenever its identity key changes, so a future
    change to zarr's string encoding costs a rebuild, not data loss.
    """
    with _zarr_string_encoding_lock:
        encode_variable = XarrayZarrStore.encode_variable

        def encode_numpy_string(store, variable):
            return encode_variable(store, _numpy_string_variable(variable))

        XarrayZarrStore.encode_variable = encode_numpy_string
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*does not have a Zarr V3 specification.*"
                )
                yield
        finally:
            XarrayZarrStore.encode_variable = encode_variable


@dataclass
class ZarrStore(FilePath):
    """Provide regulated access to a grouped zarr store."""

    group: str | None = field(default=None, repr=False)
    data: xarray.Dataset = field(default_factory=xarray.Dataset, repr=False)

    @FilePath.filepath.getter  # type: ignore
    def filepath(self):
        """Extend FilePath.filepath to include the zarr store suffix."""
        return super().filepath.with_suffix(".zarr")

    def subgroup(self, *subgroups: str) -> str | None:
        """Return the compound group path for the given subgroups."""
        subgroup = tuple(
            group for group in (self.group,) + subgroups if group is not None
        )
        if len(subgroup) == 0:
            return None
        return "/".join(subgroup)

    def is_store(self) -> bool:
        """Return True when the zarr store directory exists on the host."""
        return self.fsys.isdir(str(self.filepath))

    def get_mode(self, mode=None) -> str:
        """Return the zarr write mode, appending into an existing store."""
        if mode is not None:
            return mode
        if self.is_store():
            return "a"
        return "w"

    def _mapper(self):
        """Return the store target, an fsspec mapper for remote hosts."""
        if self.host is not None:
            return self.fsys.get_mapper(str(self.filepath))
        return str(self.filepath)

    def store(self, mode=None):
        """Store data as a group within the zarr store."""
        with suppress_unstable_string_spec():
            self.data.to_zarr(
                self._mapper(),
                group=self.group,
                mode=self.get_mode(mode),
                consolidated=False,
            )
        self.data.close()
        gc.collect()
        return self

    def group_names(self, *subgroups: str) -> list[str]:
        """Return the child group names beneath the compound subgroup path.

        Enumerates the named groups written into the store so a loader can
        discover which method groups were persisted without reading them.
        Returns an empty list when the store or the path is absent.
        """
        if not self.is_store():
            return []
        root = zarr.open_group(store=self._mapper(), mode="r")
        path = self.subgroup(*subgroups)
        node = root if path is None else root[path]
        return list(node.group_keys())

    def delete_group(self):
        """Evict a cache group via a native zarr delete, keeping its siblings."""
        if self.group is None or not self.is_store():
            return self
        root = zarr.open_group(store=self._mapper(), mode="a")
        if self.group in root:
            del root[self.group]
        return self

    def store_overwrite(self):
        """Store data, replacing the group in place if it already exists."""
        self.delete_group()
        return self.store(mode=self.get_mode())

    def load(self):
        """Load a group from the zarr store and merge it into data.

        A missing store or absent group signals a cache miss as
        :class:`FileNotFoundError` so callers can treat every backend
        uniformly; zarr otherwise reports an absent group as a
        :class:`zarr.errors.GroupNotFoundError` (a ``ValueError``).
        """
        try:
            data = xarray.open_zarr(
                self._mapper(), group=self.group, consolidated=False
            )
        except zarr.errors.NodeNotFoundError as error:
            raise FileNotFoundError(str(self.filepath)) from error
        with data:
            data.load()
            self.data = self.data.merge(
                data, combine_attrs="drop_conflicts", compat="override"
            )
        return self
