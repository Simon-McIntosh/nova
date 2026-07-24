"""Content-addressed digests of files and directory trees.

The digest primitives here stream file content in bounded chunks so that
multi-gigabyte measurement archives (for example NetCDF exports) are hashed
without loading them into memory. Digest strings are formatted
``"sha256:<hexdigest>"`` so the algorithm travels with the value.
"""

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path

# 1 MiB read window: large enough to amortise syscalls, small enough that a
# multi-GB file never resides in memory in full.
DEFAULT_CHUNK_SIZE = 1 << 20

ALGORITHM = "sha256"


def _format(hexdigest: str) -> str:
    """Return a hexdigest tagged with its algorithm.

    Parameters
    ----------
    hexdigest : str
        Bare hexadecimal digest.

    Returns
    -------
    str
        Digest string of the form ``"sha256:<hexdigest>"``.
    """
    return f"{ALGORITHM}:{hexdigest}"


def digest_bytes(data: bytes) -> str:
    """Digest an in-memory byte string.

    Parameters
    ----------
    data : bytes
        Content to hash.

    Returns
    -------
    str
        Digest string of the form ``"sha256:<hexdigest>"``.
    """
    return _format(hashlib.sha256(data).hexdigest())


def digest_file(path: os.PathLike | str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Digest a file by streaming it in bounded chunks.

    Parameters
    ----------
    path : os.PathLike or str
        File to hash.
    chunk_size : int, optional
        Read window in bytes. The default is one mebibyte.

    Returns
    -------
    str
        Digest string of the form ``"sha256:<hexdigest>"``.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(block)
    return _format(hasher.hexdigest())


def mtime_iso(timestamp: float) -> str:
    """Format a POSIX modification time as a UTC ISO-8601 string.

    Parameters
    ----------
    timestamp : float
        Seconds since the epoch, as returned by :func:`os.stat`.

    Returns
    -------
    str
        ISO-8601 timestamp in UTC, for example ``"2026-07-24T09:15:00+00:00"``.
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def digest_tree(
    root: os.PathLike | str, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> dict[str, dict]:
    """Digest every file beneath a directory tree.

    The walk is order-independent: entries are keyed by POSIX-style relative
    path and returned in sorted order so the mapping is deterministic
    regardless of the order :func:`os.walk` yields directory contents.

    Parameters
    ----------
    root : os.PathLike or str
        Directory to walk.
    chunk_size : int, optional
        Read window forwarded to :func:`digest_file`.

    Returns
    -------
    dict[str, dict]
        Mapping of relative path to ``{"digest", "size", "mtime"}`` where
        ``mtime`` is a UTC ISO-8601 string.
    """
    root = Path(root)
    entries: dict[str, dict] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            absolute = Path(dirpath) / name
            relative = absolute.relative_to(root).as_posix()
            stat = absolute.stat()
            entries[relative] = {
                "digest": digest_file(absolute, chunk_size=chunk_size),
                "size": stat.st_size,
                "mtime": mtime_iso(stat.st_mtime),
            }
    return {key: entries[key] for key in sorted(entries)}
