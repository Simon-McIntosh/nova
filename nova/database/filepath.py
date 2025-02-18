"""Manage file data access for frame and biot instances."""

from __future__ import annotations
from dataclasses import dataclass, field
from functools import wraps
import os
from pathlib import Path
import sys

import appdirs
import fsspec

import nova
from nova.definitions import root_dir


def stardot(func):
    """Return resolved path with '.' replaced with '*'."""

    @wraps(func)
    def wrapper(path: str) -> str:
        return func(path).replace(".", "*")

    return wrapper


@dataclass
class FilePath:
    """Manage to access to data via store and load methods."""

    filename: str = ""
    dirname: Path | str = field(default="", repr=False)
    basename: Path | str = field(default="user_data", repr=False)
    hostname: str | None = field(default=None, repr=False)
    parents: int = field(default=6, repr=False)
    fsys: fsspec.filesystem = field(init=False, repr=False)

    def __post_init__(self):
        """Set host and path. Forward post init for cooperative inheritance."""
        self.host = self.hostname
        self.path = self.dirname
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    @property
    def host(self):
        """Manage filesysetm on host."""
        return self.hostname

    @host.setter
    def host(self, hostname: str | None):
        match hostname:
            case str():
                self.fsys = fsspec.filesystem("ssh", host=hostname)
            case None:
                self.fsys = fsspec.filesystem("file")
            case _:
                raise NotImplementedError(
                    "filesystem for hostname " f"{hostname} not implemented"
                )
        self.hostname = hostname

    @property
    def path(self):
        """Manage file path."""
        return self.dirname

    @path.setter
    def path(self, dirname: str):
        if isinstance(dirname, Path):
            self.dirname = dirname
            self.checkpath()
            return
        absolute_path = os.path.isabs(dirname)
        match dirname.split("."):
            case [str(path)] if absolute_path:
                self.path = Path(dirname.replace("*", "."))
            case [str(path), *subpath] if not absolute_path:
                if path == "":
                    path = str(self.basename)
                path = self._resolve_absolute(path)
                self.path = ".".join((path, *subpath))
            case [str(path), str(subpath), *rest]:
                subpath = self._resolve_relative(subpath)
                path = os.path.join(path, str(subpath))
                self.path = ".".join((path, *rest))
            case _:
                raise IndexError(f"unable to match dirname {dirname}")

    @staticmethod
    @stardot
    def _resolve_absolute(path: str) -> str:
        """Return resolved absolute path."""

        def get_appdir(path: str) -> str:
            """Return appdir path."""
            try:
                return getattr(appdirs, f"{path}_dir")()
            except AttributeError as error:
                raise AttributeError(f"{path} is not a valid appdirs path") from error

        match path.split("_"):
            case ["root"]:
                return root_dir
            case ["user", "cache" | "config" | "data"]:
                return get_appdir(path)
            case ["site", "config" | "data"]:
                return get_appdir(path)
            case _:
                raise ValueError(f"unable to resolve absolute path {path}")

    @staticmethod
    @stardot
    def _resolve_relative(path: str) -> str:
        """Return resolved relative path."""
        match path:
            case "nova":
                version = nova.__version__.replace(".post", "+").split("+")[0]
                return os.path.join(nova.__name__, version)
            case "imas":
                return os.path.join("imas", os.environ.get("IMAS_VERSION", ""))
            case str(path):
                return path
            case _:
                raise ValueError(f"unable to resolve relative path {path}")

    def checkpath(self) -> str:
        """Return existing parent. Raise if not found beyond self.parents."""
        for i, parent in zip(range(self.parents), self.path.parents):
            if self.fsys.isdir(str(parent)):
                return parent
        raise FileNotFoundError(
            f"directory not found for {parent} at "
            f"depth {self.parents} of "
            f"{len(self.path.parents)}"
        )

    def is_file(self) -> bool:
        """Return status of filesystem isfile evaluated on host."""
        return self.fsys.isfile(str(self.filepath))

    def is_path(self) -> bool:
        """Return status of filesystem isdir evaluated on host."""
        return self.fsys.isdir(str(self.path))

    def makepath(self):
        """Make path if not found."""
        if not self.is_path():
            self.fsys.makedirs(str(self.path), exist_ok=True)

    @property
    def filepath(self):
        """Return full filepath."""
        if self.filename == "":
            raise FileNotFoundError("filename not set")
        self.makepath()
        return self.path / self.filename

    @filepath.setter
    def filepath(self, filepath):
        path = Path(filepath)
        self.path = path.parent
        self.filename = path.name

    def file(self, filename, extension: str | None = None):
        """Return resolved filepath combining instance path, filename, and extension."""
        filepath = self.path / self.filename
        if extension is not None:
            return filepath.with_suffix(extension)
        return filepath

    def _clear(self):
        """Clear datafile at self.filepath."""
        if os.path.isfile(self.filepath):
            os.remove(self.filepath)

    @property
    def clear(self):
        """Clear cached datafile at self.filepath."""
        if os.path.isfile(self.filepath):
            remove = input(
                "Confirm removal of the following cached datafile:"
                f"\n{self.filepath}\nProceed (Y/n)?"
            )
            if remove == "" or remove.lower() == "y":
                os.remove(self.filepath)
            return
        sys.stdout.write(f"Cached datafile clear:\n{self.filepath}")


if __name__ == "__main__":
    filepath = FilePath(parents=2, filename="test")
    filepath.path = ".nova"

    # filepath.filepath = "/home/mcintos/Code/nova/nova/2022.3.0/tests"

    # filepath.filepath =

    print(filepath.filepath)
