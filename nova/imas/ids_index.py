"""Manage IMAS high level IDS attributes."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from operator import attrgetter

import numpy as np


from nova.imas.dataset import IDSToplevel


@dataclass
class IdsIndex:
    """
    Methods for indexing data as arrays from an ids.

    Parameters
    ----------
    ids : ImasIds
        IMAS IDS (in-memory).
    ids_node : str
        Array extraction node.

    Examples
    --------
    Check access to required IDS(s).

    >>> import pytest
    >>> from nova.imas.database import Database
    >>> try:
    ...     _ = Database(105028, 1, 'pf_active')
    ...     _ = Database(105028, 1, 'equilibrium')
    ...     _ = Database(135007, 4, 'pf_active')
    ... except:
    ...     pytest.skip('IMAS not installed or 105028/1, 135007/4 unavailable')

    First load an ids, accomplished here using the Database class from
    nova.imas.database.

    >>> from nova.imas.ids_index import IdsIndex
    >>> pulse, run = 105028, 1  # DINA scenario data
    >>> pf_active = Database(pulse, run, 'pf_active')

    Initiate an instance of IdsIndex using ids from pf_active and
    specifying 'coil' as the array extraction node.

    >>> ids_index = IdsIndex(pf_active.ids, 'coil')

    Get first 5 coil names.

    >>> ids_index.array('name')[:5]
    array(['CS3U', 'CS2U', 'CS1', 'CS2L', 'CS3L'], dtype=object)

    Get full array of current data (551 time slices for all 12 coils).

    >>> current = ids_index.array('current.data')
    >>> current.shape
    (551, 12)

    Get vector of coil currents at single time slice (itime=320)

    >>> current = ids_index.vector(320, 'current.data')
    >>> current.shape
    (12,)

    Load equilibrium ids and initiate new instance of ids_index.

    >>> equilibrium = Database(pulse, run, name='equilibrium')
    >>> ids_index = IdsIndex(equilibrium.ids, 'time_slice')

    Get psi at itime=30 from profiles_1d and profiles_2d.

    >>> ids_index.vector(30, 'profiles_1d.psi').shape
    (50,)
    >>> ids_index.vector(30, 'profiles_2d.psi').shape
    (65, 129)

    Load pf_active ids containing force data.

    >>> pulse, run = 135007, 4  # DINA scenario including force data
    >>> pf_active = Database(pulse, run, name='pf_active')
    >>> ids_index = IdsIndex(pf_active.ids, 'coil')

    Use context manager to temporarily switch the ids_node to radial_force
    and vertical_force and extract force data at itime=100 from each node.

    >>> with ids_index.node('radial_force'):
    ...     print(ids_index.vector(100, 'force.data').shape)
    (12,)

    >>> with ids_index.node('vertical_force'):
    ...     print(ids_index.vector(100, 'force.data').shape)
    (17,)

    """

    ids: IDSToplevel
    ids_node: str | None = "time_slice"
    transpose: bool = field(init=False, default=False)
    shapes: dict[str, tuple[int, ...] | tuple[()]] = field(
        init=False, default_factory=dict
    )
    _ids: IDSToplevel | None = field(init=False, repr=False, default=None)
    _ids_node: str | None = field(init=False, repr=False, default="time_slice")

    # def __post_init__(self):
    #    """Initialize ids node."""
    #    if self._ids is None:
    #        raise AttributeError("required attribute 'ids' not set")

    @property  # type: ignore[no-redef]
    def ids(self):  # noqa
        """Manage ids."""
        if self._ids_node:
            return attrgetter(self.ids_node)(self._ids)
        return self._ids

    @ids.setter
    def ids(self, ids: IDSToplevel):
        if type(ids) is property:
            ids = self._ids
        self._ids = ids

    @property  # type: ignore[no-redef]
    def ids_node(self) -> str | None:  # noqa
        """Manage ids_node."""
        return self._ids_node

    @ids_node.setter
    def ids_node(self, ids_node: str | None):
        if type(ids_node) is property:
            ids_node = self._ids_node
        if ids_node is not None:
            self.transpose = ids_node != "time_slice"
            self._ids_node = ids_node

    @contextmanager
    def node(self, ids_node: str):
        """
        Permit tempary change to an instance's ids_node.

        Example
        -------
        Check access to required IDS(s).

        >>> import pytest
        >>> from nova.imas.database import Database
        >>> try:
        ...     _ = Database(135007, 4, 'pf_active')
        ... except:
        ...     pytest.skip('IMAS not installed or 135007/4 unavailable')

        Demonstrate use of context manager for switching active ids_node.

        >>> from nova.imas.ids_index import IdsIndex
        >>> ids = Database(135007, 4, name='pf_active').ids
        >>> ids_index = IdsIndex(ids, 'coil')
        >>> with ids_index.node('vertical_force'):
        ...     print(ids_index.array('force.data').shape)
        (2338, 17)
        """
        _ids_node = self.ids_node
        self.ids_node = ids_node
        yield
        self.ids_node = _ids_node

    @property
    def length(self):
        """Return ids_node length."""
        try:
            return len(self.ids)
        except (AttributeError, TypeError):
            return 0
        raise ValueError(f"unable determine ids_node length {self.ids_node}")

    def _ids_path(self, path: str) -> str:
        """Return full ids path."""
        if self.ids_node is None:
            return path
        return f"{self.ids_node}.{path}"

    def shape(self, path) -> tuple[int, ...]:
        """Return attribute array shape."""
        if self.length == 0:
            return self.get_shape(path)
        return (self.length,) + self.get_shape(path)

    def get_shape(self, path: str) -> tuple[int, ...] | tuple[()]:
        """Return cached dimension length."""
        _path = self._ids_path(path)
        try:
            return self.shapes[_path]
        except KeyError:
            self.shapes[_path] = self._get_shape(path)
            return self.get_shape(path)

    def _get_shape(self, path: str) -> tuple[int, ...] | tuple[()]:
        """Return data shape at index=0 on path."""
        return self.get_slice(0, path).shape

    def get(self, path: str):
        """Return attribute from ids path."""
        return attrgetter(path)(self.ids)

    def resize(self, path: str, number: int):
        """Resize structured array."""
        attrgetter(path)(self.ids).resize(number)

    @staticmethod
    def get_index(key):
        """Return formated item key."""
        match key:
            case str(attr):
                index, subindex = 0, 0
            case (str(attr), index):
                subindex = 0
            case (str(attr), index, int(subindex)):
                pass
            case _:
                raise KeyError(f"invalid key {key}")
        return attr, index, subindex

    def __setitem__(self, key, value):
        """Set attribute on ids path."""
        attr, index, subindex = self.get_index(key)

        if isinstance(index, slice):
            # recursive update for all indicies specified in slice.
            for _index, _value in zip(range(len(value))[index], value[index]):
                self.__setitem__((attr, _index, subindex), _value)
            return

        path = self.get_path(self.ids_node, attr)
        split_path = path.split(".")
        node = ".".join(split_path[:-1])
        leaf = split_path[-1]
        match node.split(":"):
            case ("",):
                branch = self.ids
            case (str(node),):
                try:
                    branch = attrgetter(node)(self._ids)[index]
                except TypeError:
                    branch = attrgetter(node)(self._ids)
            case (str(array), ""):
                branch = attrgetter(array)(self._ids)[index]
            case (str(array), str(node)):
                trunk = attrgetter(array)(self._ids)[index]
                branch = attrgetter(node)(trunk)
            case _:
                raise IndexError(f"invalid node {node}")
        match leaf.split(":"):
            case ("",):
                setattr(branch, value)
            case (str(leaf),):
                setattr(branch, leaf, value)
            case str(stem), str(leaf):
                try:
                    shoot = attrgetter(stem)(branch)[subindex]
                except IndexError:
                    attrgetter(stem)(branch).resize(subindex + 1)
                    shoot = attrgetter(stem)(branch)[subindex]
                setattr(shoot, leaf, value)
            case _:
                raise NotImplementedError(f"invalid leaf {leaf}")

    def get_slice(self, index: int, path: str):
        """Return attribute slice at node index."""
        try:
            return attrgetter(path)(self.ids[index])
        except AttributeError:  # __structArray__
            node, path = path.split(".", 1)
            return attrgetter(path)(attrgetter(node)(self.ids[index])[0])
        except ValueError:  # Invalid node name
            return self.get(path)

    def vector(self, itime: int, path: str):
        """Return attribute data vector at itime."""
        if len(self.get_shape(path)) == 0:
            raise IndexError(
                f"attribute {path} is 0-dimensional " "access with self.array(path)"
            )
        if self.transpose:
            data = np.zeros(self.shape(path)[:-1], dtype=self.dtype(path))
            for index in range(self.length):
                try:
                    data[index] = self.get_slice(index, path).value[itime]
                except (ValueError, IndexError):  # empty slice
                    pass
            return data
        return self.get_slice(itime, path).value

    def array(self, path: str):
        """Return attribute data array."""
        if self.length == 0:
            return self.get(path).value
        dtype = self.dtype(path)
        data = np.zeros(self.shape(path), dtype)
        for index in range(self.length):
            try:
                data[index] = self.get_slice(index, path).value
            except ValueError:  # empty slice
                pass
        if self.transpose:
            return data.T

        return data

    def valid(self, path: str):
        """Return validity flag for ids path."""
        try:
            self.empty(path)
            return True
        except TypeError:
            return False

    def empty(self, path: str):
        """Return status based on first data point extracted from ids."""
        try:
            data = self.get_slice(0, path)
        except IndexError:
            return True
        if hasattr(data, "flat"):
            try:
                data = data.flat[0]
            except IndexError:
                return True
        try:  # string
            return len(data) == 0
        except TypeError:
            return (
                data is None or np.isclose(data, -9e40) or np.isclose(data, -999999999)
            )

    def dtype(self, path: str):
        """Return data point type."""
        if self.empty(path):
            raise ValueError(f"data entry at {path} is empty")
        data = self.get_slice(0, path).value
        if isinstance(data, str):
            return object
        if hasattr(data, "flat"):
            return type(data.flat[0])
        return type(data)

    @staticmethod
    def get_path(branch: str, attr: str) -> str:
        """Return ids attribute path."""
        if not branch:
            return attr
        if "*" in branch:
            return branch.replace("*", attr)
        return ".".join((branch, attr))


if __name__ == "__main__":

    import doctest

    doctest.testmod(verbose=False)
