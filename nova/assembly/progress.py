"""Rich progress monitoring for Monte Carlo trial simulations."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import time

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)


@dataclass
class TrialProgress:
    """Progress monitor for Monte Carlo trial simulations.

    Provides rich console progress tracking for trial build operations.
    Supports both step-based progress (discrete stages) and iterative
    progress (loops with known iteration counts).

    Parameters
    ----------
    description : str
        Overall description for the progress display
    console : Console | None
        Rich console instance. If None, creates a new one.

    Examples
    --------
    >>> with TrialProgress("Vault Monte Carlo") as progress:
    ...     with progress.step("Building signals"):
    ...         build_signal()
    ...     with progress.step("Computing gaps", total=20) as task:
    ...         for i in range(20):
    ...             compute_gap(i)
    ...             task.advance()
    """

    description: str = "Trial"
    console: Console | None = field(default=None, repr=False)
    _progress: Progress | None = field(default=None, init=False, repr=False)
    _main_task: int | None = field(default=None, init=False, repr=False)
    _start_time: float = field(default=0.0, init=False, repr=False)
    _steps: list[str] = field(default_factory=list, init=False, repr=False)
    _current_step: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        """Initialize console if not provided."""
        if self.console is None:
            self.console = Console()

    def __enter__(self):
        """Start progress tracking."""
        self._start_time = time()
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}", style="bold blue"),
            BarColumn(bar_width=20),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )
        self._progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete progress tracking and show summary."""
        if self._progress is not None:
            self._progress.__exit__(exc_type, exc_val, exc_tb)
        elapsed = time() - self._start_time
        if exc_type is None:
            self.console.print(
                f"[bold green]✓[/] {self.description} completed in {elapsed:.1f}s"
            )
        return False

    def configure(self, steps: list[str]) -> "TrialProgress":
        """Configure the progress monitor with a list of step names.

        Parameters
        ----------
        steps : list[str]
            Names of the steps to track

        Returns
        -------
        TrialProgress
            Self for method chaining
        """
        self._steps = steps
        self._current_step = 0
        if self._progress is not None:
            self._main_task = self._progress.add_task(
                f"[cyan]{self.description}", total=len(steps)
            )
        return self

    @contextmanager
    def step(self, name: str, total: int | None = None):
        """Context manager for a single build step.

        Parameters
        ----------
        name : str
            Description of the step
        total : int | None
            If provided, creates a sub-progress bar for iterations

        Yields
        ------
        StepTask | None
            Step task object with advance() method if total is provided
        """
        if self._progress is None:
            yield None
            return

        # Update main task description
        step_desc = f"[cyan]{self.description}[/] → {name}"
        if self._main_task is not None:
            self._progress.update(self._main_task, description=step_desc)

        if total is not None:
            # Create sub-task for iterations
            sub_task = self._progress.add_task(f"  {name}", total=total)
            yield StepTask(self._progress, sub_task)
            self._progress.update(sub_task, completed=total)
        else:
            yield None

        # Advance main task
        self._current_step += 1
        if self._main_task is not None:
            self._progress.update(self._main_task, advance=1)


@dataclass
class StepTask:
    """Handle for advancing progress within a step."""

    progress: Progress
    task_id: int

    def advance(self, amount: int = 1):
        """Advance the step progress.

        Parameters
        ----------
        amount : int
            Number of units to advance
        """
        self.progress.update(self.task_id, advance=amount)

    def update(self, completed: int | None = None, description: str | None = None):
        """Update the step task.

        Parameters
        ----------
        completed : int | None
            Set completed count directly
        description : str | None
            Update task description
        """
        kwargs = {}
        if completed is not None:
            kwargs["completed"] = completed
        if description is not None:
            kwargs["description"] = description
        if kwargs:
            self.progress.update(self.task_id, **kwargs)
