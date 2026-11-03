"""Durable, deterministic JSON receipt writing."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_atomic(
    path: Path | str,
    payload: Any,
    *,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    sort_keys: bool = True,
    newline: bool = False,
) -> None:
    """Replace a JSON receipt after its bytes and directory entry are durable."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, indent=indent, separators=separators, sort_keys=sort_keys
    ) + ("\n" if newline else "")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.{os.getpid()}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
