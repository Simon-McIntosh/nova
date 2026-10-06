"""Manage file data access for frame and biot instances."""

from __future__ import annotations
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import wraps
import marshal
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Literal

import appdirs
import fsspec
import numpy as np
import xxhash

import nova
from nova.definitions import root_dir


WORKTREE_ROOT = "reckon-worktrees"
"""Directory name at the head of every reckon worktree layout.

Both worktree roots carry it: the preferred ``.reckon-worktrees`` beside the
repository and the legacy ``.cache/reckon-worktrees`` under the temporary
directory. Every other module imports this name instead of spelling the
pattern.
"""

WORKTREE_PATH_EXEMPTIONS = (
    "docs/plans",
    "docs/state",
    "docs/research",
    "nova/database/filepath.py",
)
"""Repository paths whose worktree-path mentions are recorded provenance.

``docs/state`` is reckon's run ledger, which reports where a run executed;
``docs/plans`` and ``docs/research`` are edited only through reckon's tools and
discuss the pattern as a subject; this module holds the rule and its constants.
The commit check and the repository test both skip exactly these paths.
"""

_WORKTREE_PROJECT = re.compile(r"nova-[0-9a-f]+")
_WORKTREE_PATH = re.compile(
    r"[A-Za-z0-9._~/\-]*" + WORKTREE_ROOT + r"(?:/[A-Za-z0-9._~/\-]*)?"
)


def _map_nova_worktree(segments: list[str]) -> str | None:
    """Return the repository-relative name for worktree layout ``segments``.

    ``segments`` begins at the project directory beneath the worktree root:
    ``<project>/<session>/<node>/<rest>``.  A complete node worktree maps to
    ``<rest>``, a bare worktree root maps to ``.``, and a layout the matcher
    cannot complete -- an incomplete path or a path under another repository's
    worktree -- returns ``None`` so the caller reports it unmapped.
    """
    if len(segments) < 3:
        return None
    if _WORKTREE_PROJECT.fullmatch(segments[0]) is None:
        return None
    if not segments[1] or not segments[2]:
        return None
    rest = segments[3:]
    if any(segment == "" for segment in rest):
        return None
    return "/".join(rest) if rest else "."


def relativize_worktree_paths(text: str) -> tuple[str, list[str]]:
    """Rewrite nova worktree paths in ``text`` to repository-relative names.

    Returns the rewritten text together with the list of worktree-path spans
    left unmapped.  A path under a ``nova-<hash>`` worktree,
    ``.reckon-worktrees/nova-<hash>/<session>/<node>/<rest>``, is replaced by
    the bare string ``<rest>``, and a bare worktree root by ``.``.  A path
    under another repository's worktree, or one whose layout cannot be
    completed (a truncated path, a path split across two lines), is reported
    in the unmapped list and left unchanged.  Every other byte, and every line
    ending, is preserved.
    """
    unmapped: list[str] = []
    pieces: list[str] = []
    cursor = 0
    for match in _WORKTREE_PATH.finditer(text):
        raw = match.group(0)
        head = raw.index(WORKTREE_ROOT)
        if head and raw[head - 1] not in "/.":
            unmapped.append(raw)
            continue
        tail = raw[head + len(WORKTREE_ROOT) :]
        segments = tail.lstrip("/").split("/") if tail else []
        mapped = _map_nova_worktree(segments)
        if mapped is None:
            unmapped.append(raw)
            continue
        pieces.append(text[cursor : match.start()])
        pieces.append(mapped)
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces), unmapped


def _is_worktree_path_exempt(path: str) -> bool:
    """Return whether ``path`` is outside the worktree-path guard's reach."""
    normalized = PurePosixPath(path).as_posix()
    return any(
        normalized == exempt or normalized.startswith(exempt + "/")
        for exempt in WORKTREE_PATH_EXEMPTIONS
    )


def worktree_path_offenders(
    paths: Iterable[str | os.PathLike],
    read: Callable[[str | os.PathLike], str],
) -> list[str | os.PathLike]:
    """Return each non-exempt path whose contents name a worktree.

    ``read`` maps a path to its text.  Paths under
    :data:`WORKTREE_PATH_EXEMPTIONS` are skipped, so the one scanning decision
    serves both the commit-time check and the repository-wide test.
    """
    offenders: list[str | os.PathLike] = []
    for path in paths:
        if _is_worktree_path_exempt(str(path)):
            continue
        if WORKTREE_ROOT in read(path):
            offenders.append(path)
    return offenders


def worktree_path_candidates(
    root: str | os.PathLike,
    *,
    cached: bool = False,
) -> list[str]:
    """Return tracked paths whose contents name a worktree root.

    ``git grep`` enumerates the candidates, so a scan is a search rather than a
    full read of every tracked file and binary files are covered (``-a``). The
    working tree is searched by default and the index with ``cached``, so the
    commit-time check and the repository-wide test share one enumeration. This
    function makes no scanning decision of its own: the caller passes the result
    to :func:`worktree_path_offenders`.
    """
    command = ["git", "-C", str(root), "grep", "--no-color", "-l", "-a", "-F"]
    if cached:
        command.append("--cached")
    command += ["-e", WORKTREE_ROOT]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            "worktree-path candidate search failed: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    return [
        name
        for name in completed.stdout.decode("utf-8", errors="replace").splitlines()
        if name
    ]


def worktree_path_text(root: str | os.PathLike, path: str | os.PathLike) -> str:
    """Return a tracked path's working-tree bytes decoded, replacing bad bytes.

    The reader the repository test passes to :func:`worktree_path_offenders`
    when its candidates come from the working tree. Binary artifacts are read
    rather than skipped, so a binary file that carries a worktree root is
    reported instead of reading as text-free.
    """
    return (Path(root) / path).read_bytes().decode("utf-8", errors="replace")


def worktree_path_index_text(root: str | os.PathLike, path: str | os.PathLike) -> str:
    """Return a tracked path's staged bytes decoded, replacing bad bytes.

    The reader the commit check passes to :func:`worktree_path_offenders` when
    its candidates come from the index: the staged blob, not the working-tree
    file, is what a commit would record. ``git show :path`` reads the blob the
    index holds, so a file staged with a worktree path and then edited in the
    working tree without restaging is still reported. Undecodable bytes are
    replaced rather than skipping a binary blob.
    """
    completed = subprocess.run(
        ["git", "-C", str(root), "show", f":{path}"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"worktree-path index read failed for {path}: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    return completed.stdout.decode("utf-8", errors="replace")


def relativize_cprofile_dump(path: str | os.PathLike) -> list[str]:
    """Rewrite every recorded file path in a cProfile dump, in place.

    A cProfile dump stores each frame's file path verbatim, so a profile written
    from a worktree embeds that worktree's path. Each recorded path is rewritten
    through :func:`relativize_worktree_paths` and the dump is written back in the
    format :meth:`cProfile.Profile.dump_stats` produced -- a marshal of the
    stats mapping, wrapped or bare. Returns the spans the rewrite left unmapped.
    """
    with open(path, "rb") as stream:
        dump = marshal.load(stream)
    wrapped = isinstance(dump, dict) and isinstance(dump.get("stats"), dict)
    stats = dump["stats"] if wrapped else dump
    unmapped: list[str] = []

    def _path_key(key):
        filename, line, name = key
        mapped, spans = relativize_worktree_paths(filename)
        unmapped.extend(spans)
        return (mapped, line, name)

    rewritten: dict = {}
    for key, value in stats.items():
        primitive, calls, self_time, cumulative, callers = value
        rewritten[_path_key(key)] = (
            primitive,
            calls,
            self_time,
            cumulative,
            {_path_key(caller): counts for caller, counts in callers.items()},
        )
    result = {**dump, "stats": rewritten} if wrapped else rewritten
    with open(path, "wb") as stream:
        marshal.dump(result, stream)
    return unmapped


def compute_provenance(
    backend: Literal["jax", "numpy"],
    *,
    platform: str | None = None,
    device_kind: str | None = None,
) -> str:
    """Return the code-generator identity for one cached numerical artifact.

    NumPy is an explicitly host-built reference. JAX artifacts additionally
    name the runtime platform, device family and jaxlib version because each
    combination may select a different floating-point code generator. Optional
    platform and device values make persisted identities independently
    reconstructible without requiring the original device to be present.
    """
    if backend == "numpy":
        if platform not in (None, "cpu") or device_kind not in (None, "host"):
            raise ValueError("NumPy cache provenance is the host CPU lane")
        return "numpy|platform=cpu|device=host"
    if backend != "jax":
        raise ValueError(f"unsupported compute backend {backend!r}")

    try:
        import jax
        import jaxlib
    except ModuleNotFoundError as error:
        if error.name != "jax":
            raise
        return compute_provenance("numpy")

    resolved_platform = jax.default_backend() if platform is None else platform
    if device_kind is None:
        kinds = sorted(
            {device.device_kind for device in jax.devices(resolved_platform)}
        )
        device_kind = ",".join(kinds)
    return (
        f"jax|platform={resolved_platform}|device={device_kind}"
        f"|jaxlib={jaxlib.__version__}"
    )


def repository_relative(path: str | os.PathLike) -> str:
    """Return ``path`` relative to its containing repository root, else absolute.

    The root is found from the path itself, so a path recorded from another
    checkout of this project still resolves to the same repository-relative
    name, and a receipt keeps locating its artifacts once the worktree that
    produced it is reclaimed.  A path under a reckon worktree is rewritten
    lexically through :func:`relativize_worktree_paths`, since the worktree it
    names has been removed and no ``.git`` marker remains to find.  A path with
    no repository ancestor is returned absolute, since it cannot be named
    relative to a root.
    """
    text = str(path)
    rewritten, _ = relativize_worktree_paths(text)
    if rewritten != text:
        return rewritten
    resolved = Path(path).resolve()
    root = next(
        (
            parent
            for parent in (resolved, *resolved.parents)
            if (parent / ".git").exists()
        ),
        Path(root_dir),
    )
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def stardot(func):
    """Return resolved path with '.' replaced with '*'."""

    @wraps(func)
    def wrapper(path: str) -> str:
        return func(path).replace(".", "*")

    return wrapper


def canonical_key(value) -> str:
    """Return a deterministic, type-tagged serialisation for cache-key hashing.

    Unlike ``str(dict)`` the result is independent of mapping insertion order
    and never conflates values that merely share a printed form: ``1`` (int),
    ``1.0`` (float), ``"1"`` (str) and ``True`` (bool) each serialise to a
    distinct key. Nested mappings and sequences are handled recursively so a
    composite identity descriptor -- identity attributes, per-source content
    hashes and discretisation parameters -- hashes unambiguously.
    """
    if value is None:
        return "none"
    if isinstance(value, bool):  # bool before int: bool subclasses int
        return f"bool:{int(value)}"
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}"
    if isinstance(value, (float, np.floating)):
        return f"float:{float(value)!r}"
    if isinstance(value, str):
        return f"str:{value}"
    if isinstance(value, bytes):
        return f"bytes:{value.hex()}"
    if isinstance(value, Mapping):
        items = sorted(
            (canonical_key(key), canonical_key(item)) for key, item in value.items()
        )
        return "{" + ",".join(f"{key}={item}" for key, item in items) + "}"
    if isinstance(value, np.ndarray):
        body = ",".join(canonical_key(item) for item in value.ravel().tolist())
        return f"array:{value.dtype.str}:{value.shape}:[{body}]"
    if isinstance(value, (list, tuple)):
        prefix = "list" if isinstance(value, list) else "tuple"
        return f"{prefix}:[" + ",".join(canonical_key(item) for item in value) + "]"
    # Version and any other opaque object: type-tagged repr keeps it stable.
    return f"{type(value).__name__}:{value!r}"


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

    def hash_attrs(self, attrs: Mapping) -> str:
        """Return a stable hex digest labelling data by its identity mapping.

        The mapping is serialised through :func:`canonical_key` so the digest
        is insertion-order independent and type-aware, guarding against stale
        cache reuse when only a value's type (not its printed form) differs.
        """
        xxh = xxhash.xxh64()
        xxh.update(canonical_key(attrs).encode("utf-8"))
        return xxh.hexdigest()

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
                    f"filesystem for hostname {hostname} not implemented"
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

    def _remove(self):
        """Remove the cached datafile, whether a single file or a store dir."""
        if os.path.isdir(self.filepath):  # grouped zarr store is a directory
            import shutil

            shutil.rmtree(self.filepath)
        elif os.path.isfile(self.filepath):
            os.remove(self.filepath)

    def _clear(self):
        """Clear datafile at self.filepath."""
        self._remove()

    @property
    def clear(self):
        """Clear cached datafile at self.filepath."""
        if os.path.isfile(self.filepath) or os.path.isdir(self.filepath):
            remove = input(
                "Confirm removal of the following cached datafile:"
                f"\n{self.filepath}\nProceed (Y/n)?"
            )
            if remove == "" or remove.lower() == "y":
                self._remove()
            return
        sys.stdout.write(f"Cached datafile clear:\n{self.filepath}")


if __name__ == "__main__":
    filepath = FilePath(parents=2, filename="test")
    filepath.path = ".nova"

    # filepath.filepath = "/home/mcintos/Code/nova/nova/2022.3.0/tests"

    # filepath.filepath =

    print(filepath.filepath)
