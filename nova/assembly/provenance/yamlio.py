"""Deterministic, round-trippable YAML serialization.

Provenance documents must serialize to bytes that are stable across runs and
machines: the same object always produces the same bytes, so a digest of those
bytes is a meaningful content address. This module wraps PyYAML's safe loader
and dumper with settings that guarantee that property:

* mapping keys are sorted;
* the block (non-flow) style is used throughout;
* output is UTF-8 with unicode preserved literally;
* anchors and aliases are disabled, so shared sub-objects are expanded rather
  than referenced (a reference would make the bytes depend on object identity);
* floats are emitted via ``repr`` so that ``safe_load`` recovers the exact
  value.

Only :func:`yaml.safe_load` and the safe dumper are used; arbitrary Python
object construction is never enabled.
"""

import os

import yaml

# A generous line width defeats PyYAML's default scalar line-wrapping, which
# would otherwise make the byte output depend on string length in ways that
# complicate round-trip identity of long values.
_LINE_WIDTH = 1 << 20


class _CanonicalDumper(yaml.SafeDumper):
    """Safe dumper that never emits anchors or aliases."""

    def ignore_aliases(self, data):
        """Expand every node instead of aliasing shared references.

        Parameters
        ----------
        data : object
            Node being represented.

        Returns
        -------
        bool
            Always ``True`` so no anchors are generated.
        """
        return True


def canonical_yaml_bytes(obj) -> bytes:
    """Serialize an object to deterministic canonical YAML bytes.

    Parameters
    ----------
    obj : object
        A tree of ``dict``, ``list``, ``str``, ``int``, ``float``, ``bool`` and
        ``None`` values.

    Returns
    -------
    bytes
        UTF-8 encoded canonical YAML.

    Raises
    ------
    ValueError
        If the object contains a type PyYAML's safe dumper cannot represent.
    """
    try:
        text = yaml.dump(
            obj,
            Dumper=_CanonicalDumper,
            sort_keys=True,
            default_flow_style=False,
            allow_unicode=True,
            width=_LINE_WIDTH,
        )
    except yaml.representer.RepresenterError as error:
        raise ValueError(f"object is not YAML-serializable: {error}") from error
    return text.encode("utf-8")


def dumps(obj) -> str:
    """Return canonical YAML as a text string.

    Parameters
    ----------
    obj : object
        Serializable object tree.

    Returns
    -------
    str
        Canonical YAML text.
    """
    return canonical_yaml_bytes(obj).decode("utf-8")


def loads(text: str | bytes):
    """Parse a YAML document using the safe loader only.

    Parameters
    ----------
    text : str or bytes
        YAML source.

    Returns
    -------
    object
        The decoded object tree.
    """
    return yaml.safe_load(text)


def dump_yaml(obj, path: os.PathLike | str) -> None:
    """Write an object to a file as canonical YAML bytes.

    Parameters
    ----------
    obj : object
        Serializable object tree.
    path : os.PathLike or str
        Destination file.
    """
    with open(path, "wb") as handle:
        handle.write(canonical_yaml_bytes(obj))


def load_yaml(path: os.PathLike | str):
    """Read and safe-parse a YAML file.

    Parameters
    ----------
    path : os.PathLike or str
        Source file.

    Returns
    -------
    object
        The decoded object tree.
    """
    with open(path, "rb") as handle:
        return yaml.safe_load(handle)
