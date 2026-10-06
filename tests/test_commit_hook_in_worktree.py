"""The commit hook resolves its checkout root inside a linked worktree.

The hook finds the checkout it guards from its own path, so the same shim
serves the primary checkout and every ``git worktree add`` tree: a clean commit
is admitted and a staged file naming a worktree path is refused, in both.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import nova
from nova.database.filepath import WORKTREE_ROOT


NOVA_PACKAGE = Path(nova.__file__).parent
REPO_ROOT = NOVA_PACKAGE.parent
VENV = Path(sys.prefix)
DEFAULT_HOOK = REPO_ROOT / "scripts" / "git-hooks" / "pre-commit"
# NOVA_COMMIT_HOOK points the scratch repo at an unpatched hook, so a
# negative-control run fails on the linked-worktree clean commit.
HOOK = Path(os.environ.get("NOVA_COMMIT_HOOK", DEFAULT_HOOK))

# Composed from the shared constant so this test file carries no worktree path.
OFFENDING_PATH = f"/home/user/.{WORKTREE_ROOT}/nova-abc/session/node/artifact.html"


def _git(args, cwd):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, text=True
    )


def _checked(args, cwd):
    result = _git(args, cwd)
    if result.returncode != 0:
        raise AssertionError(f"git {args} failed: {result.stderr}")
    return result


def _scratch_repo(tmp_path):
    repo = tmp_path / "main"
    repo.mkdir()
    _checked(["init", "-q", "-b", "main"], repo)
    _checked(["config", "user.email", "hook@test.invalid"], repo)
    _checked(["config", "user.name", "hook test"], repo)
    hooks = repo / "scripts" / "git-hooks"
    hooks.mkdir(parents=True)
    shutil.copy(HOOK, hooks / "pre-commit")
    (hooks / "pre-commit").chmod(0o755)
    (repo / "seed.txt").write_text("seed\n")
    _checked(["add", "scripts/git-hooks/pre-commit", "seed.txt"], repo)
    _checked(["commit", "-m", "init"], repo)
    (repo / "nova").symlink_to(NOVA_PACKAGE, target_is_directory=True)
    (repo / ".venv").symlink_to(VENV, target_is_directory=True)
    _checked(["config", "core.hooksPath", "scripts/git-hooks"], repo)
    return repo


def _linked_worktree(tmp_path, repo):
    worktree = tmp_path / "wt"
    _checked(["worktree", "add", "-q", str(worktree), "-b", "feature"], repo)
    (worktree / "nova").symlink_to(NOVA_PACKAGE, target_is_directory=True)
    (worktree / ".venv").symlink_to(VENV, target_is_directory=True)
    return worktree


def _commit(cwd, name, content):
    (cwd / name).write_text(content)
    _checked(["add", name], cwd)
    return _git(["commit", "-m", f"add {name}"], cwd)


def test_clean_commit_admitted_in_main_checkout(tmp_path):
    repo = _scratch_repo(tmp_path)
    result = _commit(repo, "clean_main.txt", "clean\n")
    assert result.returncode == 0, result.stderr


def test_clean_commit_admitted_in_linked_worktree(tmp_path):
    repo = _scratch_repo(tmp_path)
    worktree = _linked_worktree(tmp_path, repo)
    result = _commit(worktree, "clean_wt.txt", "clean\n")
    assert result.returncode == 0, result.stderr


def test_worktree_path_refused_in_main_checkout(tmp_path):
    repo = _scratch_repo(tmp_path)
    result = _commit(repo, "offender_main.txt", OFFENDING_PATH + "\n")
    assert result.returncode != 0
    assert "offender_main.txt" in result.stdout + result.stderr


def test_worktree_path_refused_in_linked_worktree(tmp_path):
    repo = _scratch_repo(tmp_path)
    worktree = _linked_worktree(tmp_path, repo)
    result = _commit(worktree, "offender_wt.txt", OFFENDING_PATH + "\n")
    assert result.returncode != 0
    assert "offender_wt.txt" in result.stdout + result.stderr
