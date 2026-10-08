"""Content-addressed object store with verified, atomic publication."""

import hashlib
import os
import re
import secrets
import shutil
from ctypes import CDLL, c_char_p, c_int, get_errno
from errno import EEXIST, EINVAL, ENOSYS, ENOTEMPTY, EOPNOTSUPP, EPERM
from pathlib import Path, PurePosixPath
from stat import S_ISDIR, S_ISLNK, S_ISREG
from typing import Any, Iterable, Protocol

MANIFEST_FILENAME = "manifest.json"
_HEX_PATTERN = re.compile(r"[0-9a-f]+")
_PORTABLE_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_HDF5_CONSISTENCY_FIELDS = {
    0: (20, 4),
    1: (20, 4),
    2: (11, 1),
    3: (11, 1),
}
_WINDOWS_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_RENAME_NO_REPLACE = 1
_RENAME_UNSUPPORTED_ERRORS = frozenset({EINVAL, ENOSYS, EOPNOTSUPP, EPERM})


class ContentStoreError(Exception):
    """Base error for an invalid, altered, or unpublishable store object."""


class ContentFile(Protocol):
    """Per-file identity a descriptor declares and a reader verifies."""

    name: str
    sha256: str
    size: int


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hdf5_consistency_field(
    descriptor: int,
    size: int,
    path: Path,
) -> tuple[int, int] | None:
    """Locate the transient file-consistency field in an HDF5 superblock.

    HDF5 permits a user block before the superblock, but only at byte zero or
    at powers of two starting at 512.  Superblock versions zero and one carry
    a four-byte consistency field; versions two and three carry a one-byte
    field.  No user-block byte or other superblock byte is excluded.
    """

    offset = 0
    while offset + len(_HDF5_SIGNATURE) <= size:
        signature = os.pread(descriptor, len(_HDF5_SIGNATURE), offset)
        if signature == _HDF5_SIGNATURE:
            version_bytes = os.pread(descriptor, 1, offset + len(_HDF5_SIGNATURE))
            if len(version_bytes) != 1:
                raise ContentStoreError(f"truncated HDF5 superblock in {path}")
            version = version_bytes[0]
            try:
                relative_offset, width = _HDF5_CONSISTENCY_FIELDS[version]
            except KeyError as error:
                raise ContentStoreError(
                    f"unsupported HDF5 superblock version {version} in {path}"
                ) from error
            field_offset = offset + relative_offset
            if field_offset + width > size:
                raise ContentStoreError(f"truncated HDF5 superblock in {path}")
            return field_offset, width
        offset = 512 if offset == 0 else offset * 2
    return None


def file_content_identity(
    path: Path,
    *,
    consistency_field_locator: Any = None,
) -> tuple[str, int]:
    """Return content identity with only HDF5 open-state flags canonicalized."""

    if consistency_field_locator is None:
        consistency_field_locator = hdf5_consistency_field
    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ContentStoreError(f"cannot open artifact file {path}") from error
    metadata = os.fstat(descriptor)
    if not S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise ContentStoreError(f"artifact path is not a regular file: {path}")
    consistency_field = consistency_field_locator(descriptor, metadata.st_size, path)
    with os.fdopen(descriptor, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            if consistency_field is not None:
                field_offset, field_width = consistency_field
                block_stop = size + len(block)
                overlap_start = max(size, field_offset)
                overlap_stop = min(block_stop, field_offset + field_width)
                if overlap_start < overlap_stop:
                    canonical = bytearray(block)
                    canonical[overlap_start - size : overlap_stop - size] = b"\x00" * (
                        overlap_stop - overlap_start
                    )
                    block = canonical
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def read_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ContentStoreError(f"cannot open artifact file {path}") from error
    metadata = os.fstat(descriptor)
    if not S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise ContentStoreError(f"artifact path is not a regular file: {path}")
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read()


def validate_hex(value: str, lengths: tuple[int, ...], context: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or _HEX_PATTERN.fullmatch(value) is None
    ):
        allowed = " or ".join(str(length) for length in lengths)
        raise ContentStoreError(
            f"{context} must be lowercase hexadecimal with length {allowed}"
        )


def safe_relative_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value == "."
        or "\\" in value
        or ":" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise ContentStoreError(f"unsafe artifact file name {value!r}")
    components = value.split("/")
    if any(
        not component
        or component in {".", ".."}
        or _PORTABLE_COMPONENT_PATTERN.fullmatch(component) is None
        or component.endswith((".", " "))
        or component.split(".", 1)[0].upper() in _WINDOWS_DEVICE_NAMES
        for component in components
    ):
        raise ContentStoreError(f"unsafe artifact file name {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ContentStoreError(f"unsafe artifact file name {value!r}")
    normalized = path.as_posix()
    if normalized != value:
        raise ContentStoreError(f"non-canonical artifact file name {value!r}")
    if normalized.casefold() == MANIFEST_FILENAME.casefold():
        raise ContentStoreError(
            f"unsafe artifact file name {value!r}: {MANIFEST_FILENAME!r} is reserved"
        )
    return normalized


def validate_portable_name_set(names: Iterable[str]) -> None:
    seen: dict[str, str] = {}
    for name in names:
        safe = safe_relative_name(name)
        parts = PurePosixPath(safe).parts
        for length in range(1, len(parts) + 1):
            prefix = "/".join(parts[:length])
            folded = prefix.casefold()
            previous = seen.get(folded)
            if previous is not None and previous != prefix:
                raise ContentStoreError(
                    f"case-insensitive artifact path collision: "
                    f"{previous!r} and {prefix!r}"
                )
            seen[folded] = prefix


def entry_metadata(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ContentStoreError(f"cannot inspect artifact path {path}") from error


def require_contained(path: Path, root: Path, context: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ContentStoreError(f"cannot resolve {context}: {path}") from error
    if not resolved.is_relative_to(root):
        raise ContentStoreError(
            f"{context} escapes canonical cache root {root}: {resolved}"
        )
    return resolved


def canonical_cache_root(cache_directory: Path | str, *, create: bool) -> Path:
    requested = Path(cache_directory)
    if create:
        try:
            requested.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ContentStoreError(f"cannot create cache root {requested}") from error
    try:
        root = requested.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ContentStoreError(f"cannot resolve cache root {requested}") from error
    if not root.is_dir():
        raise ContentStoreError(f"cache root is not a directory: {root}")
    return root


def verified_object_root(cache_directory: Path | str, *, create: bool) -> Path:
    cache_root = canonical_cache_root(cache_directory, create=create)
    object_root = cache_root / "sha256"
    metadata = entry_metadata(object_root)
    if metadata is None and create:
        try:
            object_root.mkdir()
        except FileExistsError:
            pass
        except OSError as error:
            raise ContentStoreError(
                f"cannot create cache object root {object_root}"
            ) from error
        metadata = entry_metadata(object_root)
    if metadata is None:
        raise ContentStoreError(f"cache object root is missing: {object_root}")
    if object_root.is_symlink():
        raise ContentStoreError(
            f"cache object root must not be a symlink: {object_root}"
        )
    if not object_root.is_dir():
        raise ContentStoreError(f"cache object root is not a directory: {object_root}")
    resolved = require_contained(object_root, cache_root, "cache object root")
    if resolved != object_root:
        raise ContentStoreError(f"cache object root is not canonical: {object_root}")
    return object_root


def verified_destination(object_root: Path, digest_hex: str) -> Path | None:
    destination = object_root / digest_hex
    metadata = entry_metadata(destination)
    if metadata is None:
        return None
    if destination.is_symlink():
        raise ContentStoreError(
            f"cache digest destination must not be a symlink: {destination}"
        )
    if not destination.is_dir():
        raise ContentStoreError(
            f"cache digest destination is not a directory: {destination}"
        )
    resolved = require_contained(destination, object_root.parent, "cache object")
    if resolved != destination:
        raise ContentStoreError(
            f"cache digest destination is not canonical: {destination}"
        )
    return destination


def inventory_files(
    directory: Path,
    *,
    allow_manifest: bool,
    containment_root: Path | None = None,
) -> dict[str, Path]:
    if not directory.is_dir():
        raise ContentStoreError(f"artifact directory does not exist: {directory}")
    if directory.is_symlink():
        raise ContentStoreError(
            f"artifact directory must not be a symlink: {directory}"
        )
    if containment_root is not None:
        resolved = require_contained(directory, containment_root, "artifact directory")
        if resolved != directory:
            raise ContentStoreError(f"artifact directory is not canonical: {directory}")
    inventory: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise ContentStoreError(f"artifact contains symlink {relative!r}")
        if containment_root is not None:
            require_contained(path, containment_root, f"artifact path {relative!r}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ContentStoreError(f"artifact contains non-file {relative!r}")
        if relative == MANIFEST_FILENAME and allow_manifest:
            continue
        safe = safe_relative_name(relative)
        if safe in inventory:
            raise ContentStoreError(f"duplicate artifact file {safe!r}")
        inventory[safe] = path
    validate_portable_name_set(inventory)
    return inventory


def verify_directory_files(
    directory: Path,
    files: Iterable[ContentFile],
    *,
    allow_manifest: bool,
    containment_root: Path | None = None,
) -> None:
    files = tuple(files)
    inventory = inventory_files(
        directory,
        allow_manifest=allow_manifest,
        containment_root=containment_root,
    )
    expected_names = {artifact_file.name for artifact_file in files}
    actual_names = set(inventory)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ContentStoreError(
            f"artifact files differ: missing={missing}, unexpected={unexpected}"
        )
    for artifact_file in files:
        digest, size = file_content_identity(inventory[artifact_file.name])
        if size != artifact_file.size:
            raise ContentStoreError(
                f"size mismatch for {artifact_file.name!r}: "
                f"expected {artifact_file.size}, got {size}"
            )
        if digest != artifact_file.sha256:
            raise ContentStoreError(
                f"checksum mismatch for {artifact_file.name!r}: "
                f"expected {artifact_file.sha256}, got {digest}"
            )


def digest_hex(digest: str) -> str:
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ContentStoreError("artifact digest must use the sha256 algorithm")
    value = digest.removeprefix("sha256:")
    validate_hex(value, (64,), "artifact digest")
    return value


def linux_rename_no_replace() -> Any:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required_flags):
        raise ContentStoreError(
            "descriptor-relative artifact publication requires Linux open flags"
        )
    required_dir_fd = (os.mkdir, os.open, os.stat)
    if any(function not in os.supports_dir_fd for function in required_dir_fd):
        raise ContentStoreError(
            "descriptor-relative artifact publication is unavailable"
        )
    if not Path("/proc/self/fd").is_dir():
        raise ContentStoreError(
            "descriptor-relative artifact paths require the Linux proc filesystem"
        )
    library = CDLL(None, use_errno=True)
    try:
        rename_no_replace = library.renameat2
    except AttributeError as error:
        raise ContentStoreError(
            "atomic no-clobber directory publication is unavailable"
        ) from error
    rename_no_replace.argtypes = (c_int, c_char_p, c_int, c_char_p, c_int)
    rename_no_replace.restype = c_int
    return rename_no_replace


def open_pinned_object_root(object_root: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(object_root, flags)
    except OSError as error:
        raise ContentStoreError(
            f"cannot pin cache object root {object_root}"
        ) from error
    opened = os.fstat(descriptor)
    visible = entry_metadata(object_root)
    if (
        visible is None
        or S_ISLNK(visible.st_mode)
        or not S_ISDIR(visible.st_mode)
        or (visible.st_dev, visible.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        os.close(descriptor)
        raise ContentStoreError(
            f"cache object root changed while being pinned: {object_root}"
        )
    return descriptor


def pinned_root_path(descriptor: int, cache_root: Path) -> Path:
    proc_path = Path("/proc/self/fd") / str(descriptor)
    resolved = require_contained(proc_path, cache_root, "pinned cache object root")
    opened = os.fstat(descriptor)
    current = resolved.stat()
    if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
        raise ContentStoreError("pinned cache object root identity changed")
    return resolved


def visible_root_matches_descriptor(object_root: Path, descriptor: int) -> bool:
    visible = entry_metadata(object_root)
    if visible is None or S_ISLNK(visible.st_mode) or not S_ISDIR(visible.st_mode):
        return False
    opened = os.fstat(descriptor)
    return (visible.st_dev, visible.st_ino) == (opened.st_dev, opened.st_ino)


def destination_exists_at(descriptor: int, digest_hex: str) -> bool:
    try:
        metadata = os.stat(digest_hex, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ContentStoreError(f"cannot inspect cache object {digest_hex}") from error
    if S_ISLNK(metadata.st_mode):
        raise ContentStoreError(
            f"cache digest destination must not be a symlink: {digest_hex}"
        )
    if not S_ISDIR(metadata.st_mode):
        raise ContentStoreError(
            f"cache digest destination is not a directory: {digest_hex}"
        )
    return True


def create_private_directory(descriptor: int, digest_hex: str) -> tuple[str, int]:
    for _ in range(32):
        name = f".{digest_hex}.{secrets.token_hex(12)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=descriptor)
        except FileExistsError:
            continue
        except OSError as error:
            raise ContentStoreError("cannot create private cache directory") from error
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            temporary_descriptor = os.open(name, flags, dir_fd=descriptor)
        except OSError as error:
            raise ContentStoreError("cannot pin private cache directory") from error
        return name, temporary_descriptor
    raise ContentStoreError("cannot allocate a unique private cache directory")


def copy_file_at(source: Path, directory_descriptor: int, name: str) -> None:
    parts = PurePosixPath(name).parts
    current = os.dup(directory_descriptor)
    try:
        for component in parts[:-1]:
            try:
                os.mkdir(component, mode=0o700, dir_fd=current)
            except FileExistsError:
                pass
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            child = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = child
        source_flags = os.O_RDONLY | os.O_NOFOLLOW
        source_descriptor = os.open(source, source_flags)
        source_metadata = os.fstat(source_descriptor)
        if not S_ISREG(source_metadata.st_mode):
            os.close(source_descriptor)
            raise ContentStoreError(f"artifact source is not a regular file: {source}")
        target_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        try:
            target_descriptor = os.open(
                parts[-1],
                target_flags,
                0o600,
                dir_fd=current,
            )
        except OSError:
            os.close(source_descriptor)
            raise
        with (
            os.fdopen(source_descriptor, "rb") as source_stream,
            os.fdopen(target_descriptor, "wb") as target_stream,
        ):
            shutil.copyfileobj(source_stream, target_stream)
    except OSError as error:
        raise ContentStoreError(f"cannot copy artifact file {name!r}") from error
    finally:
        os.close(current)


def write_bytes_at(directory_descriptor: int, name: str, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_descriptor)
    except OSError as error:
        raise ContentStoreError(f"cannot write artifact file {name!r}") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def publish_directory_no_replace(
    rename_no_replace: Any,
    object_descriptor: int,
    source_name: str,
    destination_name: str,
) -> bool:
    """Atomically publish within a pinned directory and report a winner."""

    result = rename_no_replace(
        object_descriptor,
        os.fsencode(source_name),
        object_descriptor,
        os.fsencode(destination_name),
        _RENAME_NO_REPLACE,
    )
    if result == 0:
        return True
    error_number = get_errno()
    if error_number in {EEXIST, ENOTEMPTY}:
        return False
    error = OSError(error_number, os.strerror(error_number), destination_name)
    if error_number in _RENAME_UNSUPPORTED_ERRORS:
        raise ContentStoreError(
            "the cache filesystem does not support atomic no-clobber directory "
            "rename, so an artifact cannot be published there without risking a "
            "half-visible object; several parallel filesystems reject the "
            "operation outright"
        ) from error
    raise ContentStoreError(
        f"cannot publish cache object {destination_name}"
    ) from error
