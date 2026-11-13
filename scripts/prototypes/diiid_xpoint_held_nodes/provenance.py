"""Keep executable measurement inputs inside the repository."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRACE_HELPER = ROOT / "scripts/prototypes/diiid_hollow_current/trace.py"


def load_trace(path=None):
    """Load the committed source reader and refuse alternate code inputs."""
    if path is not None and Path(path).resolve() != TRACE_HELPER.resolve():
        raise ValueError("source trace must be the repository source reader")
    if not TRACE_HELPER.is_file():
        raise FileNotFoundError(TRACE_HELPER)
    spec = importlib.util.spec_from_file_location("source_trace", TRACE_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    digest = hashlib.sha256(TRACE_HELPER.read_bytes()).hexdigest()
    print(f"SOURCE_TRACE_MODULE={module.__file__} SHA256={digest}", flush=True)
    return module
