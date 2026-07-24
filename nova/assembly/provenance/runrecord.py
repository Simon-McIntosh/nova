"""The immutable run record: what produced a fit, and from what.

A run record is written once, when a fit completes, and never edited. It binds
the fit's outputs to everything that determined them: the exact code revision
(git sha plus a dirty flag), the resolved dependency lock, the content digests
of every input dataset, the digest of the fit configuration, the recorded
outputs with their tolerance classes, the software environment, the operator,
and the wall-clock time the run occurred. Unlike a manifest, a run record
*does* record its timestamp -- when a fit ran is part of its provenance.

:func:`capture_environment` snapshots the live code revision and dependency
lock so a caller can stamp a record with the state of the working tree at run
time.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess

from nova.assembly.provenance import digest, yamlio

# nova/assembly/provenance/runrecord.py -> repo root is three parents up from
# the package directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]

_SHA256_PREFIX = "sha256:"
_HEX40_LENGTH = 40


def _is_hex40(value: str) -> bool:
    """Return whether a string is exactly 40 lower/upper hex characters.

    Parameters
    ----------
    value : str
        Candidate git sha.

    Returns
    -------
    bool
        ``True`` when the value is a 40-character hexadecimal string.
    """
    if not isinstance(value, str) or len(value) != _HEX40_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_digest(value, name: str) -> str:
    """Validate a ``"sha256:<hex>"`` digest string.

    Parameters
    ----------
    value : object
        Candidate digest.
    name : str
        Field name for error messages.

    Returns
    -------
    str
        The validated digest string.

    Raises
    ------
    ValueError
        If the value is not a ``sha256:`` prefixed 64-hex digest.
    """
    if (
        not isinstance(value, str)
        or not value.startswith(_SHA256_PREFIX)
        or len(value) != len(_SHA256_PREFIX) + 64
    ):
        raise ValueError(f"{name} must be a 'sha256:<64 hex>' digest, got {value!r}")
    try:
        int(value[len(_SHA256_PREFIX) :], 16)
    except ValueError as error:
        raise ValueError(f"{name} has a non-hex digest body") from error
    return value


@dataclass
class RunRecord:
    """Immutable provenance record for a single fit.

    Parameters
    ----------
    fit_id : str
        Stable identifier for the fit this record describes.
    code_git_sha : str
        40-character git commit sha of the code that produced the fit.
    code_dirty : bool
        Whether the working tree had uncommitted changes at run time.
    uv_lock_sha256 : str
        Digest of the resolved dependency lock file.
    input_digests : dict[str, str]
        Mapping of logical input name to a ``"sha256:<hex>"`` content digest.
    fit_config_sha256 : str
        Digest of the fit configuration that drove the run.
    outputs : list[dict]
        Recorded outputs, each ``{"name", "digest", "tolerance_class"}``.
    env : dict
        Software environment: ``{"packages": {name: version}, ...}`` including a
        ``"blas_single_thread"`` flag.
    operator : str
        Identifier of the person or agent that ran the fit.
    timestamp : str
        ISO-8601 UTC time the run occurred.
    """

    fit_id: str
    code_git_sha: str
    code_dirty: bool
    uv_lock_sha256: str
    input_digests: dict[str, str]
    fit_config_sha256: str
    outputs: list[dict]
    env: dict
    operator: str
    timestamp: str

    _REQUIRED = (
        "fit_id",
        "code_git_sha",
        "code_dirty",
        "uv_lock_sha256",
        "input_digests",
        "fit_config_sha256",
        "outputs",
        "env",
        "operator",
        "timestamp",
    )

    @classmethod
    def from_dict(cls, payload: dict) -> "RunRecord":
        """Validate and construct a run record from a mapping.

        Parameters
        ----------
        payload : dict
            Raw record mapping, typically loaded from YAML.

        Returns
        -------
        RunRecord
            The validated record.

        Raises
        ------
        ValueError
            If a required field is missing or a digest/sha is malformed.
        TypeError
            If a field has the wrong type.
        """
        if not isinstance(payload, dict):
            raise TypeError("run record must be a mapping")

        missing = [key for key in cls._REQUIRED if key not in payload]
        if missing:
            raise ValueError(f"missing required run-record fields: {missing}")

        if not _is_hex40(payload["code_git_sha"]):
            raise ValueError(
                f"code_git_sha must be a 40-character hex string, "
                f"got {payload['code_git_sha']!r}"
            )
        if not isinstance(payload["code_dirty"], bool):
            raise TypeError("code_dirty must be a bool")

        _require_digest(payload["uv_lock_sha256"], "uv_lock_sha256")
        _require_digest(payload["fit_config_sha256"], "fit_config_sha256")

        input_digests = payload["input_digests"]
        if not isinstance(input_digests, dict):
            raise TypeError("input_digests must be a mapping")
        for name, value in input_digests.items():
            _require_digest(value, f"input_digests[{name!r}]")

        outputs = payload["outputs"]
        if not isinstance(outputs, list):
            raise TypeError("outputs must be a list")
        for index, output in enumerate(outputs):
            if not isinstance(output, dict) or {
                "name",
                "digest",
                "tolerance_class",
            } - set(output):
                raise ValueError(
                    f"outputs[{index}] must have name, digest, tolerance_class"
                )
            _require_digest(output["digest"], f"outputs[{index}]['digest']")

        if not isinstance(payload["env"], dict):
            raise TypeError("env must be a mapping")
        if not isinstance(payload["operator"], str):
            raise TypeError("operator must be a string")
        if not isinstance(payload["timestamp"], str):
            raise TypeError("timestamp must be an ISO-8601 string")

        return cls(
            fit_id=payload["fit_id"],
            code_git_sha=payload["code_git_sha"],
            code_dirty=payload["code_dirty"],
            uv_lock_sha256=payload["uv_lock_sha256"],
            input_digests=dict(input_digests),
            fit_config_sha256=payload["fit_config_sha256"],
            outputs=[dict(output) for output in outputs],
            env=dict(payload["env"]),
            operator=payload["operator"],
            timestamp=payload["timestamp"],
        )

    def to_dict(self) -> dict:
        """Return the YAML-serializable record mapping.

        Returns
        -------
        dict
            Mapping of every recorded field.
        """
        return {
            "fit_id": self.fit_id,
            "code_git_sha": self.code_git_sha,
            "code_dirty": self.code_dirty,
            "uv_lock_sha256": self.uv_lock_sha256,
            "input_digests": self.input_digests,
            "fit_config_sha256": self.fit_config_sha256,
            "outputs": self.outputs,
            "env": self.env,
            "operator": self.operator,
            "timestamp": self.timestamp,
        }

    @property
    def record_digest(self) -> str:
        """Content digest of the whole normalized record.

        Returns
        -------
        str
            Digest string of the form ``"sha256:<hexdigest>"`` covering every
            field, timestamp included.
        """
        return digest.digest_bytes(yamlio.canonical_yaml_bytes(self.to_dict()))

    def write(self, path: os.PathLike | str) -> None:
        """Write the record to a file as canonical YAML.

        Parameters
        ----------
        path : os.PathLike or str
            Destination file.
        """
        yamlio.dump_yaml(self.to_dict(), path)

    @classmethod
    def load(cls, path: os.PathLike | str) -> "RunRecord":
        """Load and validate a record from a YAML file.

        Parameters
        ----------
        path : os.PathLike or str
            Source file.

        Returns
        -------
        RunRecord
            The validated record.
        """
        return cls.from_dict(yamlio.load_yaml(path))


def capture_environment(repo_root: os.PathLike | str | None = None) -> dict:
    """Snapshot the live code revision and dependency lock.

    Reads the current git commit sha and working-tree dirty flag via
    ``git`` subprocesses rooted at the repository, and digests the ``uv.lock``
    dependency lock.

    Parameters
    ----------
    repo_root : os.PathLike or str, optional
        Repository root. Defaults to the repository containing this package.

    Returns
    -------
    dict
        Mapping with ``code_git_sha`` (40-hex string), ``code_dirty`` (bool),
        and ``uv_lock_sha256`` (``"sha256:<hex>"`` digest of ``uv.lock``).
    """
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    code_dirty = bool(status.strip())

    lock_path = root / "uv.lock"
    uv_lock_sha256 = digest.digest_file(lock_path)

    return {
        "code_git_sha": sha,
        "code_dirty": code_dirty,
        "uv_lock_sha256": uv_lock_sha256,
    }
