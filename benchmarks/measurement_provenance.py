"""Attach benchmark measurements to a clean Git checkout."""

from dataclasses import dataclass
from pathlib import Path
import subprocess


class MeasurementProvenanceError(RuntimeError):
    """The measuring checkout cannot support an attributable result."""


@dataclass(frozen=True)
class CheckoutState:
    """The commit and porcelain state of a measuring checkout."""

    commit_sha: str
    has_porcelain_output: bool


def _git(checkout: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise MeasurementProvenanceError(
            f"{checkout} is not a readable Git worktree"
        ) from error
    return completed.stdout


def resolve_checkout(checkout: str | Path) -> CheckoutState:
    """Resolve ``HEAD`` and whether porcelain reports any checkout change."""

    path = Path(checkout).resolve()
    if _git(path, "rev-parse", "--is-inside-work-tree").strip() != "true":
        raise MeasurementProvenanceError(f"{path} is not a Git worktree")

    commit_sha = _git(path, "rev-parse", "--verify", "HEAD").strip()
    porcelain = _git(path, "status", "--porcelain")
    return CheckoutState(
        commit_sha=commit_sha,
        has_porcelain_output=bool(porcelain),
    )


def measurement_stamp(checkout: str | Path) -> str:
    """Return the measured commit SHA, refusing an unclean checkout."""

    path = Path(checkout).resolve()
    state = resolve_checkout(path)
    if state.has_porcelain_output:
        raise MeasurementProvenanceError(
            f"{path} has git status --porcelain output; measurement is unattributable"
        )
    return state.commit_sha
