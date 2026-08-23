from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import conftest


@dataclass
class Invocation:
    args: tuple[str, ...]


class Config:
    class ArgsSource:
        ARGS = "args"
        TESTPATHS = "testpaths"

    def __init__(
        self,
        *,
        args_source: str,
        invocation_args: tuple[str, ...] = (),
        markexpr: str = "not slow",
    ) -> None:
        self.args_source = args_source
        self.invocation_params = Invocation(invocation_args)
        self.option = SimpleNamespace(markexpr=markexpr)

    def getoption(self, name: str) -> str:
        assert name == "markexpr"
        return self.option.markexpr


def test_explicit_paths_remove_the_default_fast_lane_filter() -> None:
    config = Config(
        args_source=Config.ArgsSource.ARGS,
        invocation_args=("tests/test_equilibrium_forward_solve.py",),
    )

    conftest.pytest_configure(config)

    assert config.option.markexpr == ""


@pytest.mark.parametrize(
    ("args_source", "invocation_args", "markexpr"),
    [
        (Config.ArgsSource.TESTPATHS, (), "not slow"),
        (Config.ArgsSource.ARGS, ("tests/test_example.py", "-m", "slow"), "slow"),
        (Config.ArgsSource.ARGS, ("tests/test_example.py", "-m=slow"), "slow"),
    ],
)
def test_repo_default_and_explicit_marker_expressions_are_preserved(
    args_source: str,
    invocation_args: tuple[str, ...],
    markexpr: str,
) -> None:
    config = Config(
        args_source=args_source,
        invocation_args=invocation_args,
        markexpr=markexpr,
    )

    conftest.pytest_configure(config)

    assert config.option.markexpr == markexpr


def test_successful_empty_collection_exits_nonzero() -> None:
    session = SimpleNamespace(testscollected=0, exitstatus=pytest.ExitCode.OK)

    conftest.pytest_sessionfinish(session, pytest.ExitCode.OK)

    assert session.exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED


def test_nonempty_collection_keeps_its_exit_status() -> None:
    session = SimpleNamespace(testscollected=1, exitstatus=pytest.ExitCode.OK)

    conftest.pytest_sessionfinish(session, pytest.ExitCode.OK)

    assert session.exitstatus == pytest.ExitCode.OK
